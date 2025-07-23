"""VQA model training script"""

import argparse
import os
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from datasets.vqa_dataset import VQACollateFn, VQADataset
from models.butd import BUTD
from models.mcan import MCAN
from models.original_vqa import OriginalVQA
from models.san import SAN
from utils.vocab import Vocabulary


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_model(model_name: str, config: dict, vocab: Vocabulary):
    """Get model"""
    model_config = config["model"]

    if model_name == "original_vqa":
        return OriginalVQA(
            vocab_size=vocab.vocab_size,
            embed_dim=model_config["embed_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_answers=model_config["num_answers"],
            learning_rate=model_config["learning_rate"],
            weight_decay=model_config["weight_decay"],
            pretrained=model_config["pretrained"],
        )
    elif model_name == "san":
        return SAN(
            vocab_size=vocab.vocab_size,
            embed_dim=model_config["embed_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_answers=model_config["num_answers"],
            learning_rate=model_config["learning_rate"],
            weight_decay=model_config["weight_decay"],
            num_attention_layers=model_config["num_attention_layers"],
            pretrained=model_config["pretrained"],
        )
    elif model_name == "butd":
        return BUTD(
            vocab_size=vocab.vocab_size,
            embed_dim=model_config["embed_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_answers=model_config["num_answers"],
            learning_rate=model_config["learning_rate"],
            weight_decay=model_config["weight_decay"],
            feature_dim=model_config["feature_dim"],
            num_objects=model_config["num_objects"],
            pretrained=model_config["pretrained"],
        )
    elif model_name == "mcan":
        return MCAN(
            vocab_size=vocab.vocab_size,
            embed_dim=model_config["embed_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_answers=model_config["num_answers"],
            learning_rate=model_config["learning_rate"],
            weight_decay=model_config["weight_decay"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            pretrained=model_config["pretrained"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="VQA Model Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--model",
        type=str,
        choices=["original_vqa", "san", "butd", "mcan"],
        help="Model name (optional, will use config if not specified)",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/vqa_v2", help="Data directory"
    )
    parser.add_argument(
        "--vocab_path", type=str, default="data/vocab.pkl", help="Vocabulary path"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument(
        "--precision", type=int, default=32, help="Precision (16 or 32)"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases logging"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Load or create vocabulary
    if os.path.exists(args.vocab_path):
        vocab = Vocabulary.load(args.vocab_path)
        print(f"Loaded vocabulary from {args.vocab_path}")
    else:
        print("Building vocabulary from training data...")
        train_dataset = VQADataset(
            data_dir=args.data_dir,
            split="train",
            max_question_length=config["data"]["max_question_length"],
            image_size=config["data"]["image_size"],
        )
        vocab = train_dataset.vocab
        vocab.save(args.vocab_path)
        print(f"Saved vocabulary to {args.vocab_path}")

    # Create datasets
    train_dataset = VQADataset(
        data_dir=args.data_dir,
        split="train",
        vocab=vocab,
        max_question_length=config["data"]["max_question_length"],
        image_size=config["data"]["image_size"],
    )

    val_dataset = VQADataset(
        data_dir=args.data_dir,
        split="val",
        vocab=vocab,
        max_question_length=config["data"]["max_question_length"],
        image_size=config["data"]["image_size"],
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        collate_fn=VQACollateFn(vocab),
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=VQACollateFn(vocab),
        pin_memory=True,
    )

    # Get model name from config if not specified
    model_name = args.model if args.model else config["model"]["name"]

    # Create model
    model = get_model(model_name, config, vocab)

    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor=config["logging"]["monitor"],
            mode=config["logging"]["mode"],
            patience=config["training"]["early_stopping_patience"],
        ),
        ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_dir, model_name),
            filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.3f}}",
            monitor=config["logging"]["monitor"],
            mode=config["logging"]["mode"],
            save_top_k=config["logging"]["save_top_k"],
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Set up loggers
    loggers = [
        TensorBoardLogger(
            save_dir=args.log_dir,
            name=model_name,
            version=None,
        )
    ]

    if args.use_wandb:
        loggers.append(
            WandbLogger(
                project="vqa-implementation",
                name=f"{model_name}-training",
            )
        )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        val_check_interval=config["training"]["val_check_interval"],
        accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
        gradient_clip_val=config["training"]["gradient_clip_val"],
        callbacks=callbacks,
        logger=loggers,
        gpus=args.gpus,
        precision=args.precision,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        accelerator="auto",
        devices=args.gpus if args.gpus > 0 else None,
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)

    print(
        f"Training completed. Best model saved at: "
        f"{trainer.checkpoint_callback.best_model_path}"
    )


if __name__ == "__main__":
    main()
