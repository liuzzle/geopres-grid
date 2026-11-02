"""Train the GeoPres linear projection on cached embeddings.

Carried over from `train_model.py`. Changes:
  - imports repointed at the package;
  - `device="cuda"` and the `RuntimeError` on a missing GPU replaced by
    `resolve_device`, so a small run is possible on CPU;
  - the post-training MTEB block and its ten CLI flags are gone. WP-F owns the
    MTEB task set, and keeping a second copy of it here would guarantee the two
    drift apart;
  - training data is read from the embedding cache instead of upstream's
    `precalculated_embeddings/` (10M C4 passages, paraphrase validation set):
    query and document rows sampled from every task of a tier, unit-normed as
    `PostProcessingPipeline` feeds the reducer, with a held-out share for
    validation. Transductive, like our per-task PCA fit: the rows come from the
    corpora that are scored (logging.md §72);
  - the trained matrix is also written as `weights.npy`, the `(target, source)`
    array `reducers.GeoPres` applies, with its `identity.weights_id` and the
    training setup in `training.json`.

Intrinsic evaluation is not run from `main`: `intrinsic.py` still reads the old
layout (its WP-C seam), and the held-out loss is logged by the trainer anyway.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback, TrainingArguments

from geopres_grid.config import (
    EMBEDDING_CACHE_PATH,
    TRAINED_MODELS_PATH,
    resolve_device,
)
from geopres_grid.intrinsic import eval_intrinsic, find_checkpoint_lowest_val_loss
from geopres_grid.trainer import EmbeddingsDataset, GeoPresTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def train_projection(
    trainable_projection: nn.Module,
    custom_model_name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    train_batch_size: int,
    val_batch_size: int,
    backbone_model: str,
    spearman: bool,
    spearman_local_or_global: str,
    epochs: int = 10,
    optimizer_class: type[torch.optim.Optimizer] = torch.optim.AdamW,
    optimizer_params: dict[str, Any] | None = None,
    weight_decay: float = 0.0,
    output_path: str | None = None,
    positional_loss_factor: float = 1.0,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
    resume_from_checkpoint: str | None = None,
    weighted_loss: bool = False,
    eval_after_training: bool = True,
    eval_steps: int = 100,
    early_stopping_patience: int = 3,
    use_cpu: bool = False,
) -> Any:
    """Fit the projection, save the best weights, and score it intrinsically.

    Returns the trainer's state, for the best validation loss and the step count.
    """
    before = time.perf_counter()

    args = TrainingArguments(
        output_dir=output_path or "./output",
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=val_batch_size,
        weight_decay=weight_decay,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_drop_last=True,
        disable_tqdm=False,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        dataloader_pin_memory=not use_cpu,
        # The Trainer places the model itself (cuda, then mps); a forced CPU must say so.
        use_cpu=use_cpu,
    )

    if optimizer_params is None:
        optimizer_params = {"lr": 1e-2}
    optimizer = optimizer_class(trainable_projection.parameters(), **optimizer_params)

    trainer = GeoPresTrainer(
        model=trainable_projection,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        spearman=spearman,
        spearman_local_or_global=spearman_local_or_global,
        positional_loss_factor=positional_loss_factor,
        weighted_loss=weighted_loss,
        optimizers=(optimizer, None),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=0.001,
            )
        ],
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    logger.info(
        "Training completed in %.2f hours", (time.perf_counter() - before) / 3600
    )

    torch.save(
        trainable_projection.state_dict(), os.path.join(output_path, "best_model.pt")
    )

    if not eval_after_training:
        logger.info("Evaluation after training is disabled. Exiting.")
        return trainer.state

    if best_checkpoint_path := trainer.state.best_model_checkpoint:
        best_checkpoint_num = int(best_checkpoint_path.split("-")[-1])
    else:
        best_checkpoint_num = find_checkpoint_lowest_val_loss(output_path)[1]
    logger.info("Best checkpoint: %s", best_checkpoint_num)

    logger.info("Evaluating intrinsic metrics on test set")
    eval_intrinsic(
        projection=trainable_projection,
        backbone_model_path=backbone_model,
        checkpoint=best_checkpoint_num,
        cache_path=os.path.join(
            TRAINED_MODELS_PATH, backbone_model.replace("/", "__")
        ),
        model_name=custom_model_name,
        spearman_test_batch_size=val_batch_size,
    )
    return trainer.state


def training_rows(
    task_directories: list[Path],
    max_rows_per_task: int,
    seed: int,
) -> np.ndarray:
    """Unit-normed query and document rows, sampled per task and pooled.

    Each task contributes at most `max_rows_per_task`, sampled as calibration is
    (`evaluation.sample_rows` over both sides), so a large corpus does not drown
    the small ones. Unit-normed because `PostProcessingPipeline` hands the
    reducer unit vectors (`normalize_before`): the projection must be trained on
    the geometry it is applied to.
    """
    from geopres_grid.evaluation import cached_blocks, sample_rows
    from geopres_grid.reducers import _unit_norm

    pooled = []
    for directory in task_directories:
        blocks = cached_blocks(directory)
        rows = sample_rows(blocks["query"] + blocks["document"], max_rows_per_task, seed)
        logger.info("%s: %d rows", Path(directory).name, len(rows))
        pooled.append(_unit_norm(rows))
    return np.concatenate(pooled)


def holdout(values: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle once and split off `val_fraction` of the rows for validation."""
    order = np.random.default_rng(seed).permutation(len(values))
    n_val = max(1, int(round(len(values) * val_fraction)))
    return values[order[n_val:]], values[order[:n_val]]


def projection_weights(projection: nn.Linear) -> np.ndarray:
    """The `(target, source)` matrix `reducers.GeoPres` applies as `X @ W.T`."""
    return projection.weight.detach().cpu().numpy().astype(np.float32)


def main(argv: list[str] | None = None) -> None:
    from geopres_grid.backbones import PRIMARY_MAX_SEQ_LENGTH, get
    from geopres_grid.cache import CachedBackbone, resolve_encode_config
    from geopres_grid.evaluation import TIER_TASKS, task_names
    from geopres_grid.identity import weights_id

    parser = argparse.ArgumentParser(
        description="Train a GeoPres linear projection on cached embeddings"
    )
    parser.add_argument("--backbone", type=str, required=True, help="Registry key")
    parser.add_argument("--target_dim", type=int, required=True)
    parser.add_argument("--tier", choices=sorted(TIER_TASKS), default="tier1",
                        help="Tasks whose cached rows are pooled for training")
    parser.add_argument("--task", action="append",
                        help="Explicit task instead of the tier; repeat for several")
    parser.add_argument("--max_rows_per_task", type=int, default=10_000)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--data_seed", type=int, default=0)
    parser.add_argument("--cache_root", type=str, default=str(EMBEDDING_CACHE_PATH))
    parser.add_argument("--encode_hash", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=PRIMARY_MAX_SEQ_LENGTH)

    parser.add_argument("--epochs", type=int, default=200,
                        help="Upstream's 10 epochs were over 10M rows; a tier pool is "
                        "~50k, so the step count needs many more epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--positional_loss_factor", type=float, default=1.0,
                        help="Weight of the positional loss against the angular loss")
    parser.add_argument("--train_batch_size", type=int, default=4096)
    parser.add_argument("--val_batch_size", type=int, default=4096)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weighted_loss", action="store_true")
    parser.add_argument("--spearman", action="store_true",
                        help="Use the differentiable Spearman loss (needs the spearman extra)")
    parser.add_argument("--spearman_local_or_global", type=str, default="local",
                        choices=["local", "global"])

    parser.add_argument("--device", type=str, default=None,
                        help="Force a device. Defaults to cuda, then mps, then cpu.")
    parser.add_argument("--output_root", type=str, default=str(TRAINED_MODELS_PATH))
    parser.add_argument("--custom_suffix", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", action="store_true")

    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    backbone = get(args.backbone)
    encode_config = resolve_encode_config(
        args.cache_root, backbone,
        encode_hash=args.encode_hash, max_seq_length=args.max_seq_length,
    )
    cached = CachedBackbone(None, backbone, encode_config, Path(args.cache_root))
    tasks = tuple(args.task) if args.task else task_names(args.tier)
    source = "+".join(args.task) if args.task else args.tier

    suffix = f"_{args.custom_suffix}" if args.custom_suffix else ""
    loss_name = (
        "spearman" if args.spearman else f"poslossfactor_{args.positional_loss_factor}"
    )
    model_name = (
        f"{backbone.key}_reduced_{args.target_dim}_{source}"
        f"_batch_{args.train_batch_size}_{loss_name}{suffix}"
    )
    output_path = os.path.join(args.output_root, backbone.key, model_name)
    os.makedirs(output_path, exist_ok=True)

    logger.info("Backbone: %s (encode hash %s)", backbone.key, encode_config.hash)
    logger.info("Target dimension: %s", args.target_dim)
    logger.info("Device: %s", device)
    logger.info("Output path: %s", output_path)

    last_checkpoint = None
    if args.resume_from_checkpoint:
        checkpoints = [
            int(d.split("checkpoint-")[-1])
            for d in os.listdir(output_path)
            if d.startswith("checkpoint-")
        ]
        last_checkpoint = max(checkpoints) if checkpoints else None

    logger.info("Preparing datasets from %d cached tasks", len(tasks))
    before = time.perf_counter()
    rows = training_rows(
        [cached.task_directory(task) for task in tasks],
        args.max_rows_per_task, args.data_seed,
    )
    train_rows, val_rows = holdout(rows, args.val_fraction, args.data_seed)
    # Both dataloaders drop the last partial batch: a batch larger than the split
    # would leave no batch at all, and a zero validation loss would pick the best model.
    train_batch_size = min(args.train_batch_size, len(train_rows))
    val_batch_size = min(args.val_batch_size, len(val_rows))
    logger.info(
        "%d train rows, %d validation rows (batch %d / %d)",
        len(train_rows), len(val_rows), train_batch_size, val_batch_size,
    )

    torch.manual_seed(args.data_seed)
    trainable_projection = nn.Linear(
        backbone.native_dim, args.target_dim, bias=False, device=device
    )

    logger.info("Starting training")
    state = train_projection(
        trainable_projection=trainable_projection,
        custom_model_name=model_name,
        train_dataset=EmbeddingsDataset(torch.from_numpy(train_rows)),
        val_dataset=EmbeddingsDataset(torch.from_numpy(val_rows)),
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        backbone_model=backbone.model_id,
        spearman=args.spearman,
        spearman_local_or_global=args.spearman_local_or_global,
        epochs=args.epochs,
        optimizer_class=torch.optim.AdamW,
        weight_decay=args.weight_decay,
        optimizer_params={"lr": args.learning_rate},
        output_path=output_path,
        positional_loss_factor=args.positional_loss_factor,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        resume_from_checkpoint=(
            os.path.join(output_path, f"checkpoint-{last_checkpoint}")
            if last_checkpoint is not None
            else None
        ),
        weighted_loss=args.weighted_loss,
        eval_after_training=False,
        eval_steps=args.eval_steps,
        early_stopping_patience=args.early_stopping_patience,
        use_cpu=str(device) == "cpu",
    )
    seconds = time.perf_counter() - before

    weights = projection_weights(trainable_projection)
    np.save(os.path.join(output_path, "weights.npy"), weights)
    record = {
        "backbone": backbone.key,
        "encode_hash": encode_config.hash,
        "target_dim": args.target_dim,
        "weights_id": weights_id(weights),
        "tasks": list(tasks),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "best_val_loss": state.best_metric,
        "best_checkpoint": state.best_model_checkpoint,
        "steps": state.global_step,
        "seconds": round(seconds, 1),
        "device": str(device),
        "args": vars(args),
    }
    with open(os.path.join(output_path, "training.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    logger.info(
        "weights.npy %s, weights_id %s, best val loss %s after %d steps, %.1f min",
        weights.shape, record["weights_id"], state.best_metric, state.global_step, seconds / 60,
    )


if __name__ == "__main__":
    main()
