"""
Comprehensive evaluation module for Arabic NLP tasks.

Supports:
- Classification: accuracy, precision, recall, F1, confusion matrix, ROC/AUC
- NER: seqeval metrics (precision/recall/F1 per entity)
- Summarization: ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- Visualization: confusion matrix, ROC curve saved to files
"""

import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    plt = None

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Handles evaluation for all NLP tasks with visualization support."""

    def __init__(self, output_dir: str = "data/results"):
        """
        Args:
            output_dir: Directory to save evaluation reports and plots.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Classification Evaluation
    # ------------------------------------------------------------------

    def evaluate_classification(
        self,
        y_true: List[Union[str, int]],
        y_pred: List[Union[str, int]],
        label_names: Optional[List[str]] = None,
        plot_confusion_matrix: bool = True,
        plot_roc: bool = False,
        y_score: Optional[np.ndarray] = None,
        prefix: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate classification performance.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            label_names: Optional list of label names for display.
            plot_confusion_matrix: Save confusion matrix plot.
            plot_roc: Save ROC curve (requires y_score for binary).
            y_score: Predicted probabilities for ROC curve (binary only).
            prefix: Prefix for output filenames.

        Returns:
            Dict with metrics: accuracy, macro_f1, weighted_f1,
            classification_report, confusion_matrix.
        """
        labels = label_names or sorted(set(y_true) | set(y_pred))
        logger.info("Evaluating classification (%d samples, %d classes)", len(y_true), len(labels))

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # Detailed report
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
        metrics["classification_report"] = report

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        metrics["confusion_matrix"] = cm.tolist()

        if plot_confusion_matrix:
            self._plot_confusion_matrix(cm, labels, prefix)

        if plot_roc and y_score is not None and len(labels) == 2:
            self._plot_roc(y_true, y_score, labels, prefix)
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_score[:, 1] if y_score.ndim > 1 else y_score)
            except Exception:
                pass

        # Save report
        report_path = os.path.join(self.output_dir, f"{prefix}classification_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info("Classification report saved to %s", report_path)

        return metrics

    def _plot_confusion_matrix(
        self, cm: np.ndarray, labels: List[str], prefix: str
    ) -> Optional[str]:
        """Plot and save confusion matrix."""
        if not _MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available; skipping confusion matrix plot.")
            return None

        try:
            import seaborn as sns
        except ImportError:
            logger.warning("seaborn not available; using matplotlib for confusion matrix.")
            sns = None

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), max(5, len(labels) * 1.2)))
        if sns:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        else:
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{prefix}confusion_matrix.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Confusion matrix plot saved to %s", path)
        return path

    def _plot_roc(
        self, y_true: List[Union[str, int]], y_score: np.ndarray, labels: List[str], prefix: str
    ) -> Optional[str]:
        """Plot and save ROC curve for binary classification."""
        if not _MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available; skipping ROC plot.")
            return None
        if len(set(y_true)) != 2:
            logger.warning("ROC curve requires exactly 2 classes.")
            return None

        y_bin = np.array([1 if l == labels[1] else 0 for l in y_true])
        fpr, tpr, _ = roc_curve(y_bin, y_score[:, 1] if y_score.ndim > 1 else y_score)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve ({labels[0]} vs {labels[1]})")
        ax.legend(loc="lower right")
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{prefix}roc_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("ROC curve saved to %s", path)
        return path

    # ------------------------------------------------------------------
    # NER Evaluation
    # ------------------------------------------------------------------

    def evaluate_ner(
        self,
        true_tags: List[List[str]],
        pred_tags: List[List[str]],
        label_names: Optional[List[str]] = None,
        prefix: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate NER using seqeval.

        Args:
            true_tags: List of true tag sequences (list of lists).
            pred_tags: List of predicted tag sequences.
            label_names: Subset of tags to report (e.g. exclude 'O').
            prefix: Prefix for output filenames.

        Returns:
            Dict with overall and per-entity precision/recall/f1.
        """
        try:
            from seqeval.metrics import (
                classification_report as seqeval_report,
                accuracy_score as seqeval_accuracy,
                precision_score as seqeval_precision,
                recall_score as seqeval_recall,
                f1_score as seqeval_f1,
            )
        except ImportError:
            raise ImportError("seqeval is required for NER evaluation. pip install seqeval")

        logger.info("Evaluating NER (%d sequences)", len(true_tags))

        metrics = {
            "accuracy": seqeval_accuracy(true_tags, pred_tags),
            "precision": seqeval_precision(true_tags, pred_tags),
            "recall": seqeval_recall(true_tags, pred_tags),
            "f1": seqeval_f1(true_tags, pred_tags),
        }

        # Per-entity report (as dict)
        report_str = seqeval_report(true_tags, pred_tags, output_dict=False)
        report_dict = seqeval_report(true_tags, pred_tags, output_dict=True)
        metrics["per_entity"] = {
            k: v for k, v in report_dict.items() if isinstance(v, dict) and "precision" in v
        }
        metrics["report_string"] = report_str

        # Save
        report_path = os.path.join(self.output_dir, f"{prefix}ner_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info("NER report saved to %s", report_path)

        return metrics

    # ------------------------------------------------------------------
    # Summarization Evaluation
    # ------------------------------------------------------------------

    def evaluate_summarization(
        self,
        references: List[str],
        hypotheses: List[str],
        prefix: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate summarization using ROUGE scores.

        Args:
            references: List of reference summaries.
            hypotheses: List of generated summaries.
            prefix: Prefix for output filenames.

        Returns:
            Dict with ROUGE-1, ROUGE-2, ROUGE-L precision/recall/f1.
        """
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            raise ImportError("rouge_score is required for summarization evaluation. pip install rouge-score")

        logger.info("Evaluating summarization (%d pairs)", len(references))

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        agg_scores: Dict[str, List[float]] = {
            "rouge1_precision": [], "rouge1_recall": [], "rouge1_fmeasure": [],
            "rouge2_precision": [], "rouge2_recall": [], "rouge2_fmeasure": [],
            "rougeL_precision": [], "rougeL_recall": [], "rougeL_fmeasure": [],
        }

        for ref, hyp in zip(references, hypotheses):
            scores = scorer.score(ref, hyp)
            for metric in ["rouge1", "rouge2", "rougeL"]:
                agg_scores[f"{metric}_precision"].append(scores[metric].precision)
                agg_scores[f"{metric}_recall"].append(scores[metric].recall)
                agg_scores[f"{metric}_fmeasure"].append(scores[metric].fmeasure)

        metrics = {
            "num_samples": len(references),
        }
        for key, values in agg_scores.items():
            metrics[key] = round(float(np.mean(values)), 4)
            metrics[f"{key}_std"] = round(float(np.std(values)), 4)

        # Save
        report_path = os.path.join(self.output_dir, f"{prefix}summarization_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info("Summarization ROUGE report saved to %s", report_path)

        return metrics


# ------------------------------------------------------------------
# Standalone convenience functions
# ------------------------------------------------------------------


def evaluate_classification(
    y_true: List[Union[str, int]],
    y_pred: List[Union[str, int]],
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Quick classification evaluation with printed report.

    Args:
        y_true: Ground truth.
        y_pred: Predictions.
        label_names: Optional label names.

    Returns:
        Metrics dict.
    """
    evaluator = Evaluator()
    metrics = evaluator.evaluate_classification(y_true, y_pred, label_names, plot_confusion_matrix=False)
    report_str = classification_report(y_true, y_pred, target_names=label_names or None, zero_division=0)
    print(report_str)
    return metrics
