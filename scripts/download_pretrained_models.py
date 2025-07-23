"""Script to download pretrained models"""

import argparse
import os

import torch
import torchvision.models as models


def download_pretrained_models(output_dir: str):
    """Download pretrained models"""
    os.makedirs(output_dir, exist_ok=True)

    # Models to download
    model_configs = [
        ("vgg16", models.vgg16, {"pretrained": True}),
        ("resnet152", models.resnet152, {"pretrained": True}),
    ]

    for model_name, model_fn, kwargs in model_configs:
        print(f"Downloading {model_name}...")

        try:
            # Download model
            model = model_fn(**kwargs)

            # Save model
            model_path = os.path.join(output_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)

            print(f"Saved {model_name} to {model_path}")

        except Exception as e:
            print(f"Error downloading {model_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download Pretrained Models")
    parser.add_argument(
        "--output_dir", type=str, default="pretrained_models", help="Output directory"
    )

    args = parser.parse_args()

    print("Downloading pretrained models...")
    download_pretrained_models(args.output_dir)
    print("Download completed!")


if __name__ == "__main__":
    main()
