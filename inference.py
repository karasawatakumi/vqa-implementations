"""VQA model inference script"""

import argparse
import json
from typing import Any, Dict

import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image

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


def preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """Preprocess image"""
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Transform
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Add batch dimension
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor


def preprocess_question(
    question: str, vocab: Vocabulary, max_length: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess question"""
    # Encode question
    question_indices = vocab.encode(question)

    # Padding
    if len(question_indices) < max_length:
        question_indices += [vocab.word2idx["<PAD>"]] * (
            max_length - len(question_indices)
        )
    else:
        question_indices = question_indices[:max_length]

    # Convert to tensor
    question_tensor = torch.tensor(question_indices, dtype=torch.long).unsqueeze(0)
    question_length = torch.tensor(
        [min(len(vocab.encode(question)), max_length)], dtype=torch.long
    )

    return question_tensor, question_length


def predict_answer(
    model,
    image_tensor: torch.Tensor,
    question_tensor: torch.Tensor,
    question_length: torch.Tensor,
    device: torch.device,
) -> tuple[str, float]:
    """Predict answer"""
    model.eval()
    model.to(device)

    # Move data to device
    image_tensor = image_tensor.to(device)
    question_tensor = question_tensor.to(device)
    question_length = question_length.to(device)

    with torch.no_grad():
        # Prediction
        logits = model(image_tensor, question_tensor, question_length)
        probs = torch.softmax(logits, dim=1)
        pred_index = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_index].item()

    # Convert prediction to string (simplified implementation)
    pred_answer = f"answer_{pred_index}"  # In practice, use answer dictionary

    return pred_answer, confidence


def main():
    parser = argparse.ArgumentParser(description="VQA Model Inference")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--vocab_path", type=str, default="data/vocab.pkl", help="Vocabulary path"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to input image"
    )
    parser.add_argument("--question", type=str, required=True, help="Question to ask")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument(
        "--output_file", type=str, default=None, help="Output file for results"
    )

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

    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    model_name = checkpoint["hyper_parameters"].get("model_name", "unknown")

    model = get_model(model_name, config, vocab)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded model from {args.model_path}")

    # Preprocess image
    image_tensor = preprocess_image(args.image_path, config["data"]["image_size"])
    print(f"Loaded image from {args.image_path}")

    # Preprocess question
    question_tensor, question_length = preprocess_question(
        args.question, vocab, config["data"]["max_question_length"]
    )
    print(f"Question: {args.question}")

    # Prediction
    answer, confidence = predict_answer(
        model, image_tensor, question_tensor, question_length, device
    )

    # Display results
    print("\n=== Prediction Results ===")
    print(f"Question: {args.question}")
    print(f"Answer: {answer}")
    print(f"Confidence: {confidence:.4f}")

    # Save results
    if args.output_file:
        results = {
            "model_path": args.model_path,
            "image_path": args.image_path,
            "question": args.question,
            "answer": answer,
            "confidence": confidence,
            "model_name": model_name,
        }

        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
