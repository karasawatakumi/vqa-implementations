"""VQA evaluation metrics"""

from typing import Dict, List

import numpy as np


def accuracy_score(predictions: List[str], ground_truths: List[List[str]]) -> float:
    """
    Calculate accuracy against 10 annotators' answers

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
            (each element is a list of 10 answers)

    Returns:
        Accuracy score
    """
    correct = 0
    total = len(predictions)

    for pred, gt_list in zip(predictions, ground_truths):
        # Check if prediction is in ground truth list
        if pred in gt_list:
            correct += 1

    return correct / total if total > 0 else 0.0


def vqa_score(predictions: List[str], ground_truths: List[List[str]]) -> float:
    """
    Calculate VQA score (how many out of 10 annotators gave the same answer)

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
            (each element is a list of 10 answers)

    Returns:
        VQA score
    """
    scores = []

    for pred, gt_list in zip(predictions, ground_truths):
        # Get most frequent answer in ground truth list
        gt_counter: Dict[str, int] = {}
        for gt in gt_list:
            gt_counter[gt] = gt_counter.get(gt, 0) + 1

        max(gt_counter.items(), key=lambda x: x[1])[0]

        # If prediction is in ground truth list
        if pred in gt_list:
            # Get count of people who gave the same answer as prediction
            pred_count = gt_counter[pred]
            # How many out of 10 gave the same answer
            score = min(pred_count / 3.0, 1.0)
        else:
            score = 0.0

        scores.append(score)

    return float(np.mean(scores))


def compute_metrics(
    predictions: List[str], ground_truths: List[List[str]]
) -> Dict[str, float]:
    """
    Compute multiple evaluation metrics

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}

    # Accuracy
    metrics["accuracy"] = accuracy_score(predictions, ground_truths)

    # VQA score
    metrics["vqa_score"] = vqa_score(predictions, ground_truths)

    return metrics


def question_type_accuracy(
    predictions: List[str], ground_truths: List[List[str]], question_types: List[str]
) -> Dict[str, float]:
    """
    Calculate accuracy by question type

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        question_types: List of question types

    Returns:
        Dictionary of accuracy by question type
    """
    type_accuracies = {}

    # Group by question type
    type_groups: Dict[str, List[int]] = {}
    for i, q_type in enumerate(question_types):
        if q_type not in type_groups:
            type_groups[q_type] = []
        type_groups[q_type].append(i)

    # Calculate accuracy for each type
    for q_type, indices in type_groups.items():
        type_preds = [predictions[i] for i in indices]
        type_gts = [ground_truths[i] for i in indices]

        type_accuracies[q_type] = accuracy_score(type_preds, type_gts)

    return type_accuracies


def answer_type_accuracy(
    predictions: List[str], ground_truths: List[List[str]], answer_types: List[str]
) -> Dict[str, float]:
    """
    Calculate accuracy by answer type

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        answer_types: List of answer types

    Returns:
        Dictionary of accuracy by answer type
    """
    type_accuracies = {}

    # Group by answer type
    type_groups: Dict[str, List[int]] = {}
    for i, a_type in enumerate(answer_types):
        if a_type not in type_groups:
            type_groups[a_type] = []
        type_groups[a_type].append(i)

    # Calculate accuracy for each type
    for a_type, indices in type_groups.items():
        type_preds = [predictions[i] for i in indices]
        type_gts = [ground_truths[i] for i in indices]

        type_accuracies[a_type] = accuracy_score(type_preds, type_gts)

    return type_accuracies
