"""Merge MTEB result caches into a single comparison CSV.

Carried over unchanged apart from the import. WP-B seam: it keys rows on the old
`<backbone>_reduced_<dim>_batch_<n>_...` model-name convention, which the
run-id/config-hash scheme replaces.
"""

import os
import json
import pandas as pd
from argparse import ArgumentParser
from geopres_grid.config import STORAGE_PATH


INTRINSIC_METRICS = [
    "spearman_loss",
    "angular_loss",
    "positional_loss",
]

TASK_BENCHMARK_MAPPING = {
    "intrinsic": INTRINSIC_METRICS,
    "sts": [
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STSBenchmark",
        "SICK-R"
    ],
    "retrieval": [
        "QuoraRetrieval",
        "HotpotQA",
        "DBPedia",
        "NQ",
        "MSMARCO",
        
        # "QuoraRetrievalHardNegatives",
        # "HotpotQAHardNegatives",
        # "DBPediaHardNegatives",
        # "NQHardNegatives",
        # "MSMARCOHardNegatives",

        "ArguAna"    
    ],
    "classification": [
        "AmazonCounterfactualClassification",
        "AmazonPolarityClassification",  
        "AmazonReviewsClassification",
        "ImdbClassification",  
        "ToxicConversationsClassification",        
    ],
    "clustering": [
        "ArxivClusteringS2S", 
        "RedditClustering",  
        "StackExchangeClustering"  
    ]
}


def collect_results_to_df(results_dir: str) -> pd.DataFrame:
    """Collect all evaluation results and create a comparison CSV."""
    
    # Dictionary to store all results: {model_name: {task_name: score}}
    all_results = {}
    
    # Find all 'results' directories recursively
    for root, dirs, _ in os.walk(results_dir):
        if 'results' not in dirs:
            continue
        results_subdir = os.path.join(root, 'results')
        
        # Inside each results directory, look for model directories
        for model_name in os.listdir(results_subdir):
            model_path = os.path.join(results_subdir, model_name)
            if not os.path.isdir(model_path):
                continue
            
            # Initialize model entry if not exists
            if model_name not in all_results:
                all_results[model_name] = {}
            
            # Walk through model directory to find JSON files
            for model_root, _, files in os.walk(model_path):
                for result_file in files:
                    if not result_file.endswith(".json") or result_file == "model_meta.json":
                        continue
                    
                    task_name = os.path.splitext(result_file)[0]

                    with open(os.path.join(model_root, result_file), 'r') as f:
                        results = json.load(f)

                    if task_name == "intrinsic":
                        for metric in INTRINSIC_METRICS:
                            if metric not in all_results[model_name]:
                                all_results[model_name][metric] = results.get(metric)
                        continue
                    
                    # Only store if not already present (avoid duplicates)
                    if task_name in all_results[model_name]:
                        continue
                    test_scores = [subset['main_score'] for subset in results['scores']['test']]
                    all_results[model_name][task_name] = sum(test_scores) / len(test_scores)

    df = pd.DataFrame.from_dict(all_results, orient='index')
    df.index.name = 'Model'
    
    cols = []
    for task_category, benchmarks in TASK_BENCHMARK_MAPPING.items():
        # Add individual intrinsic metric columns
        if task_category == "intrinsic":
            for metric in INTRINSIC_METRICS:
                if metric in df.columns:
                    cols.append(metric)
            continue

        # Add individual benchmark columns
        for benchmark in benchmarks:
            if benchmark in df.columns:
                cols.append(benchmark)
        
        # Calculate and add average column for this category
        category_benchmarks = [b for b in benchmarks if b in df.columns]
        df[f"**AVG_{task_category.upper()}**"] = df[category_benchmarks].mean(axis=1)
    
    avg_cols = [c for c in df.columns if c.startswith("**AVG_")]
    
    # Calculate overall average across all tasks
    df["**AVG_MTEB**"] = df[avg_cols].mean(axis=1)
    
    # Reorder the dataframe columns
    ordered_cols = (
        ["**AVG_MTEB**"] + 
        [c for c in cols if c in INTRINSIC_METRICS] + 
        avg_cols +  # Include the category average columns
        [c for c in cols if c not in INTRINSIC_METRICS and not c.startswith("**AVG_")]
    )
    df = df[ordered_cols]
    df = df.sort_index()
     
    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--results_base_path", 
        type=str, 
        default=os.path.join(STORAGE_PATH, "evaluation_results"), 
        help="A directory that contains as subdirs result directories for multiple models"
    )

    args = parser.parse_args()
    df = collect_results_to_df(
        results_dir=args.results_base_path,
    )

    output_path = os.path.join(args.results_base_path, "comparison_results.csv")
    df = df.round(4)
    df.to_csv(output_path)
    
    print(f"Results saved to: {output_path}")
