"""VQA dataset preparation script"""

import argparse
import json
import os
import zipfile
from typing import Any, Dict, List

import nltk
from tqdm import tqdm

# Type definitions for VQA data
VQAAnnotation = Dict[str, Any]
VQAQuestion = Dict[str, Any]
VQAProcessedItem = Dict[str, Any]


def download_nltk_data():
    """Download NLTK data"""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt")


def extract_zip_files(data_dir: str):
    """Extract ZIP files"""
    zip_files = [
        "train2014.zip",
        "val2014.zip",
        "v2_Questions_Train_mscoco.zip",
        "v2_Questions_Val_mscoco.zip",
        "v2_Annotations_Train_mscoco.zip",
        "v2_Annotations_Val_mscoco.zip",
    ]

    for zip_file in zip_files:
        zip_path = os.path.join(data_dir, zip_file)
        if os.path.exists(zip_path):
            print(f"Extracting {zip_file}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"Extracted {zip_file}")
        else:
            print(f"Warning: {zip_file} not found. Please download it manually.")


def build_answer_vocab(annotations: List[Dict], min_freq: int = 8) -> Dict[str, int]:
    """Build answer vocabulary"""
    answer_counts: Dict[str, int] = {}

    print("Building answer vocabulary...")
    for ann in tqdm(annotations):
        for answer in ann["answers"]:
            answer_text = answer["answer"].lower().strip()
            answer_counts[answer_text] = answer_counts.get(answer_text, 0) + 1

    # Filter by minimum frequency
    filtered_answers = [
        answer for answer, count in answer_counts.items() if count >= min_freq
    ]

    # Create vocabulary
    answer_vocab = {}
    for i, answer in enumerate(filtered_answers):
        answer_vocab[answer] = i

    print(f"Built answer vocabulary with {len(answer_vocab)} answers")
    return answer_vocab


def process_annotations(data_dir: str, split: str) -> List[VQAAnnotation]:
    """Process annotation files"""
    annotation_file = os.path.join(
        data_dir, f"v2_Annotations_{split.capitalize()}_mscoco.json"
    )

    if not os.path.exists(annotation_file):
        print(f"Warning: {annotation_file} not found")
        return []

    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    print(f"Loaded {len(annotations['annotations'])} annotations for {split} split")
    return annotations["annotations"]


def process_questions(data_dir: str, split: str) -> List[VQAQuestion]:
    """Process question files"""
    question_file = os.path.join(
        data_dir, f"v2_Questions_{split.capitalize()}_mscoco.json"
    )

    if not os.path.exists(question_file):
        print(f"Warning: {question_file} not found")
        return []

    with open(question_file, "r") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions['questions'])} questions for {split} split")
    return questions["questions"]


def create_processed_dataset(
    data_dir: str, split: str, answer_vocab: Dict[str, int]
) -> List[VQAProcessedItem]:
    """Create processed dataset"""
    annotations = process_annotations(data_dir, split)
    questions = process_questions(data_dir, split)

    if not annotations or not questions:
        return []

    # Create index by question ID
    questions_dict = {q["question_id"]: q for q in questions}

    processed_data = []

    print(f"Processing {split} data...")
    for ann in tqdm(annotations):
        question_id = ann["question_id"]

        if question_id not in questions_dict:
            continue

        question = questions_dict[question_id]

        # Image path
        image_id = question["image_id"]
        image_path = f"{split}2014/COCO_{split}2014_{image_id:012d}.jpg"

        # Label answers
        answer_labels = []
        for answer in ann["answers"]:
            answer_text = answer["answer"].lower().strip()
            if answer_text in answer_vocab:
                answer_labels.append(answer_vocab[answer_text])

        # Select most frequent answer
        if answer_labels:
            from collections import Counter

            most_common_label = Counter(answer_labels).most_common(1)[0][0]
        else:
            most_common_label = 0  # Default label

        processed_item = {
            "question_id": question_id,
            "image_id": image_id,
            "image_path": image_path,
            "question": question["question"],
            "answer_label": most_common_label,
            "question_type": ann.get("question_type", ""),
            "answer_type": ann.get("answer_type", ""),
        }

        processed_data.append(processed_item)

    return processed_data


def save_processed_data(data: List[Dict], output_file: str):
    """Save processed data"""
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved processed data to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare VQA Dataset")
    parser.add_argument(
        "--data_dir", type=str, default="data/vqa_v2", help="Data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/processed", help="Output directory"
    )
    parser.add_argument(
        "--min_answer_freq", type=int, default=8, help="Minimum answer frequency"
    )
    parser.add_argument("--extract_zips", action="store_true", help="Extract ZIP files")

    args = parser.parse_args()

    # Download NLTK data
    download_nltk_data()

    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract ZIP files
    if args.extract_zips:
        extract_zip_files(args.data_dir)

    # Build answer vocabulary from training data
    train_annotations = process_annotations(args.data_dir, "train")
    if train_annotations:
        answer_vocab = build_answer_vocab(train_annotations, args.min_answer_freq)

        # Save answer vocabulary
        vocab_file = os.path.join(args.output_dir, "answer_vocab.json")
        with open(vocab_file, "w") as f:
            json.dump(answer_vocab, f, indent=2)
        print(f"Saved answer vocabulary to {vocab_file}")
    else:
        print("No training annotations found. Using empty vocabulary.")
        answer_vocab = {}

    # Process data for each split
    for split in ["train", "val"]:
        processed_data = create_processed_dataset(args.data_dir, split, answer_vocab)

        if processed_data:
            output_file = os.path.join(args.output_dir, f"{split}_processed.json")
            save_processed_data(processed_data, output_file)
        else:
            print(f"No processed data for {split} split")

    print("Dataset preparation completed!")


if __name__ == "__main__":
    main()
