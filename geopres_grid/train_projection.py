"""Train the GeoPres linear projection on precalculated embeddings.

Carried over from `train_model.py`. Changes:
  - imports repointed at the package;
  - `device="cuda"` and the `RuntimeError` on a missing GPU replaced by
    `resolve_device`, so a small run is possible on CPU;
  - the post-training MTEB block and its ten CLI flags are gone. WP-F owns the
    MTEB task set, and keeping a second copy of it here would guarantee the two
    drift apart. Intrinsic evaluation, which needs no task list, is kept.

WP-C seam: both `torch.load` calls below read the old `precalculated_embeddings/`
layout. The embedding cache replaces them; nothing else in this file changes.
"""

import argparse
import logging
import os
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback, TrainingArguments

from geopres_grid.config import (
    STORAGE_PATH,
    TRAINED_MODELS_PATH,
    parse_dtype,
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
) -> None:
    """Fit the projection, save the best weights, and score it intrinsically."""
    before = time.perf_counter()

    args = TrainingArguments(
        output_dir=output_path or "./output",
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=val_batch_size,
        weight_decay=weight_decay,
        eval_strategy="steps",
        eval_steps=100,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_drop_last=True,
        disable_tqdm=False,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        dataloader_pin_memory=True,
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
                early_stopping_patience=3, early_stopping_threshold=0.001
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
        return

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


def _embeddings(dataset_name: str, backbone_model: str, split_file: str) -> Dataset:
    """Load one precalculated embedding tensor. WP-C replaces this with the cache."""
    path = os.path.join(
        STORAGE_PATH,
        "precalculated_embeddings",
        dataset_name,
        backbone_model.replace("/", "__"),
        split_file,
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Precalculated embeddings not found at {path}")
    return EmbeddingsDataset(torch.load(path, weights_only=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a GeoPres linear projection on precalculated embeddings"
    )
    parser.add_argument("--backbone_model", type=str, required=True)
    parser.add_argument("--source_dim", type=int, required=True)
    parser.add_argument("--target_dim", type=int, required=True)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--positional_loss_factor", type=float, default=1.0,
                        help="Weight of the positional loss against the angular loss")
    parser.add_argument("--train_batch_size", type=int, default=20000)
    parser.add_argument("--val_batch_size", type=int, default=20000)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weighted_loss", action="store_true")
    parser.add_argument("--spearman", action="store_true",
                        help="Use the differentiable Spearman loss (needs the spearman extra)")
    parser.add_argument("--spearman_local_or_global", type=str, default="local",
                        choices=["local", "global"])

    parser.add_argument("--skip_eval_after_training", action="store_true")
    parser.add_argument("--backbone_dtype", type=str, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="Force a device. Defaults to cuda, then mps, then cpu.")
    parser.add_argument("--custom_suffix", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", action="store_true")

    args = parser.parse_args()

    device = resolve_device(args.device)
    if args.backbone_dtype:
        parse_dtype(args.backbone_dtype)  # validated here, used by WP-C precompute

    suffix = f"_{args.custom_suffix}" if args.custom_suffix else ""
    if args.spearman:
        model_name = (
            f"{args.backbone_model}_reduced_{args.target_dim}"
            f"_batch_{args.train_batch_size}_spearman{suffix}"
        )
    else:
        model_name = (
            f"{args.backbone_model}_reduced_{args.target_dim}"
            f"_batch_{args.train_batch_size}"
            f"_poslossfactor_{args.positional_loss_factor}{suffix}"
        )

    output_path = os.path.join(
        TRAINED_MODELS_PATH,
        args.backbone_model.replace("/", "__"),
        model_name.replace("/", "__"),
    )
    os.makedirs(output_path, exist_ok=True)

    logger.info("Backbone model: %s", args.backbone_model)
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

    trainable_projection = nn.Linear(
        args.source_dim, args.target_dim, bias=False, device=device
    )

    logger.info("Preparing datasets")
    train_dataset = _embeddings("c4", args.backbone_model, "train_embeddings.pt")
    val_dataset = _embeddings(
        "sentence-paraphrases", args.backbone_model, "validation_embeddings.pt"
    )

    logger.info("Starting training")
    train_projection(
        trainable_projection=trainable_projection,
        custom_model_name=model_name,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        backbone_model=args.backbone_model,
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
        eval_after_training=not args.skip_eval_after_training,
    )


if __name__ == "__main__":
    main()
