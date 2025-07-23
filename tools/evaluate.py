"""VQA model evaluation script"""

import argparse
import json
from typing import Any, Dict

import torch
import yaml
from tqdm import tqdm

from datasets.vqa_dataset import VQACollateFn, VQADataset
from models.butd import BUTD
from models.mcan import MCAN
from models.original_vqa import OriginalVQA
from models.san import SAN
from utils.metrics import answer_type_accuracy, compute_metrics, question_type_accuracy
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


def evaluate_model(model, data_loader, device, vocab):
    """Evaluate model"""
    model.eval()
    model.to(device)

    predictions = []
    ground_truths = []
    question_types = []
    answer_types = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            images = batch["images"].to(device)
            questions = batch["questions"].to(device)
            question_lengths = batch["question_lengths"].to(device)

            # Prediction
            logits = model(images, questions, question_lengths)
            probs = torch.softmax(logits, dim=1)
            pred_indices = torch.argmax(probs, dim=1)

            # Convert predictions to strings (simplified implementation)
            for pred_idx in pred_indices:
                pred_answer = (
                    f"answer_{pred_idx.item()}"  # In practice, use answer dictionary
                )
                predictions.append(pred_answer)

            # Save ground truth data
            if "answers" in batch:
                ground_truths.extend(batch["answers"])
                question_types.extend(batch["question_types"])
                answer_types.extend(batch["answer_types"])

    return predictions, ground_truths, question_types, answer_types


def main():
    parser = argparse.ArgumentParser(description="VQA Model Evaluation")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--data_dir", type=str, default="data/vqa_v2", help="Data directory"
    )
    parser.add_argument(
        "--vocab_path", type=str, default="data/vocab.pkl", help="Vocabulary path"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Data split",
    )
    parser.add_argument(
        "--output_file", type=str, default="evaluation_results.json", help="Output file"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")

    args = parser.parse_args()

    # Set up device
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    print(f"Using device: {device}")

    # Load configuration
    config = load_config(args.config)

    # Load vocabulary
    vocab = Vocabulary.load(args.vocab_path)
    print(f"Loaded vocabulary from {args.vocab_path}")

    # Create dataset
    dataset = VQADataset(
        data_dir=args.data_dir,
        split=args.split,
        vocab=vocab,
        max_question_length=config["data"]["max_question_length"],
        image_size=config["data"]["image_size"],
    )

    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=VQACollateFn(vocab),
        pin_memory=True,
    )

    # Load model
    # Get model name from checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model_name = checkpoint["hyper_parameters"].get("model_name", "unknown")

    # Create model
    model = get_model(model_name, config, vocab)

    # Load checkpoint
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded model from {args.model_path}")

    # Run evaluation
    predictions, ground_truths, question_types, answer_types = evaluate_model(
        model, data_loader, device, vocab
    )

    # Calculate evaluation metrics
    if ground_truths:  # Not test data
        metrics = compute_metrics(predictions, ground_truths)

        # Accuracy by question type
        if question_types:
            question_type_acc = question_type_accuracy(
                predictions, ground_truths, question_types
            )
            metrics["question_type_accuracy"] = question_type_acc

        # Accuracy by answer type
        if answer_types:
            answer_type_acc = answer_type_accuracy(
                predictions, ground_truths, answer_types
            )
            metrics["answer_type_accuracy"] = answer_type_acc

        # Display results
        print("\n=== Evaluation Results ===")
        for metric_name, value in metrics.items():
            if isinstance(value, dict):
                print(f"\n{metric_name}:")
                for sub_metric, sub_value in value.items():
                    print(f"  {sub_metric}: {sub_value:.4f}")
            else:
                print(f"{metric_name}: {value:.4f}")

        # Save results
        results = {
            "model_path": args.model_path,
            "config": config,
            "split": args.split,
            "metrics": metrics,
            "predictions": predictions[:100],  # Save only first 100
            "ground_truths": ground_truths[:100],
        }

        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to {args.output_file}")
    else:
        # For test data, save only predictions
        results = {
            "model_path": args.model_path,
            "split": args.split,
            "predictions": predictions,
        }

        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()
