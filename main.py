#!/usr/bin/env python3
"""
Arabic NLP for Real Estate Documents — CLI Orchestrator.

Usage:
    python main.py preprocess --input <path> --output <path> [options]
    python main.py train --task <classification|ner|summarization> [options]
    python main.py inference --task <classification|ner|summarization> --model_path <dir> --input_path <path> --output_path <path>
    python main.py evaluate --task <classification|ner|summarization> --input_path <path> [--model_path <path>]
    python main.py version

Examples:
    python main.py preprocess --input data/raw/docs.txt --output data/processed/cleaned.csv
    python main.py train --task classification --model logistic_regression --data data/processed/cleaned.csv
    python main.py inference --task classification --model_path models/ --input_path data/processed/cleaned.csv --output_path data/results/predictions.csv
    python main.py evaluate --task classification --input_path data/processed/cleaned.csv --model_path models/
"""

import argparse
import logging
import sys

from src import __version__
from src.preprocess import preprocess_arabic_text
from src.train import train_text_classification, train_ner, train_summarization
from src.inference import perform_inference
from src.evaluate import Evaluator
from src.utils import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def _add_preprocess(subparsers) -> None:
    p = subparsers.add_parser("preprocess", help="Clean and normalize Arabic text data.")
    p.add_argument("--input", required=True, help="Path to raw input file (one doc per line).")
    p.add_argument("--output", required=True, help="Path to save preprocessed CSV.")
    p.add_argument("--remove-stops", action="store_true", default=True, help="Remove Arabic stopwords (default: True).")
    p.add_argument("--apply-stemming", action="store_true", default=False, help="Apply light stemming (default: False).")


def _add_train(subparsers) -> None:
    p = subparsers.add_parser("train", help="Train an NLP model.")
    p.add_argument(
        "--task", choices=["classification", "ner", "summarization"], required=True,
        help="NLP task to train for."
    )
    p.add_argument(
        "--model", default=None,
        help=(
            "Model type or HuggingFace checkpoint. For classification: "
            "'logistic_regression' or 'linear_svc'. For NER: e.g. "
            "'aubmindlab/bert-base-arabertv02'. For summarization: e.g. "
            "'google/mt5-small'."
        )
    )
    p.add_argument("--data", help="Path to training data CSV (for classification) or JSON/JSONL (for NER/summarization).")
    p.add_argument("--val-data", help="Optional validation data path.")
    p.add_argument("--output-dir", default="models", help="Directory to save trained model artifacts.")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs (for NER/summarization).")
    p.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    p.add_argument("--hyperparameter-tuning", action="store_true", help="Enable grid search for classification.")
    p.add_argument("--text-column", default="text", help="Column name for text (classification).")
    p.add_argument("--label-column", default="label", help="Column name for labels (classification).")


def _add_inference(subparsers) -> None:
    p = subparsers.add_parser("inference", help="Run inference with a trained model.")
    p.add_argument("--task", choices=["classification", "ner", "summarization"], required=True, help="Task type.")
    p.add_argument("--model-path", required=True, help="Directory containing trained model artifacts.")
    p.add_argument("--input-path", required=True, help="Path to input data (CSV or text file).")
    p.add_argument("--output-path", required=True, help="Path to save results.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for inference.")
    p.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")


def _add_evaluate(subparsers) -> None:
    p = subparsers.add_parser("evaluate", help="Evaluate a trained model.")
    p.add_argument("--task", choices=["classification", "ner", "summarization"], required=True, help="Task type.")
    p.add_argument("--input-path", required=True, help="Path to ground-truth data (CSV/JSON)")
    p.add_argument("--model-path", help="Path to model directory (not needed for some tasks).")
    p.add_argument("--prediction-path", help="Optional: path to pre-computed predictions file.")
    p.add_argument("--output-dir", default="data/results", help="Directory for evaluation reports and plots.")
    p.add_argument("--text-column", default="text", help="Column with input texts.")
    p.add_argument("--label-column", default="label", help="Column with ground-truth labels.")
    p.add_argument("--prediction-column", default="prediction", help="Column with predictions.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="arabic-nlp-realestate",
        description="Production-grade Arabic NLP toolkit for real estate document processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py preprocess --input data/raw/docs.txt --output data/processed/cleaned.csv\n"
            "  python main.py train --task classification --model logistic_regression --data data/processed/cleaned.csv\n"
            "  python main.py inference --task classification --model-path models/ --input-path data/processed/cleaned.csv --output-path data/results/predictions.csv\n"
            "  python main.py evaluate --task classification --input-path data/test.csv --model-path models/\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    _add_preprocess(subparsers)
    _add_train(subparsers)
    _add_inference(subparsers)
    _add_evaluate(subparsers)

    args = parser.parse_args()

    try:
        _dispatch(args)
    except Exception as exc:
        logger.exception("Command '%s' failed: %s", args.command, exc)
        sys.exit(1)


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "preprocess":
        preprocess_arabic_text(
            args.input,
            args.output,
            remove_stops=args.remove_stops,
            apply_stemming=args.apply_stemming,
        )

    elif args.command == "train":
        if args.task == "classification":
            train_text_classification(
                model_type=args.model or "logistic_regression",
                data_path=args.data,
                text_column=args.text_column,
                label_column=args.label_column,
                output_dir=args.output_dir,
                hyperparameter_tuning=args.hyperparameter_tuning,
            )
        elif args.task == "ner":
            train_ner(
                model_type="bert",
                model_name=args.model or "aubmindlab/bert-base-arabertv02",
                train_data_path=args.data,
                val_data_path=args.val_data,
                output_dir=args.output_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
            )
        elif args.task == "summarization":
            train_summarization(
                model_type="mt5",
                model_name=args.model or "google/mt5-small",
                train_data_path=args.data,
                val_data_path=args.val_data,
                output_dir=args.output_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
            )

    elif args.command == "inference":
        perform_inference(
            task=args.task,
            model_path=args.model_path,
            input_path=args.input_path,
            output_path=args.output_path,
            device=args.device,
            batch_size=args.batch_size,
        )

    elif args.command == "evaluate":
        evaluator = Evaluator(output_dir=args.output_dir)

        if args.task == "classification":
            df = DataLoader.load_csv(args.input_path)
            text_col = args.text_column if args.text_column in df.columns else df.columns[0]
            label_col = args.label_column if args.label_column in df.columns else df.columns[1]

            if args.prediction_path:
                pred_df = DataLoader.load_csv(args.prediction_path)
                pred_col = args.prediction_column if args.prediction_column in pred_df.columns else pred_df.columns[1]
                y_true = df[label_col].tolist()
                y_pred = pred_df[pred_col].tolist()
            elif args.model_path:
                from src.inference import InferenceEngine
                engine = InferenceEngine()
                engine.load_classification_model(args.model_path)
                y_pred = engine.predict_classification(df[text_col].tolist())
                y_true = df[label_col].tolist()
            else:
                raise ValueError("Either --model-path or --prediction-path required for classification evaluation.")

            evaluator.evaluate_classification(y_true, y_pred)

        elif args.task == "ner":
            if not args.prediction_path:
                raise ValueError("--prediction-path required for NER evaluation.")
            with open(args.prediction_path, encoding="utf-8") as f:
                pred_data = json.load(f)
            true_data = DataLoader.load_json(args.input_path)
            evaluator.evaluate_ner(
                [item["tags"] for item in true_data],
                [item["tags"] for item in pred_data],
            )

        elif args.task == "summarization":
            df = DataLoader.load_csv(args.input_path)
            ref_col = args.label_column if args.label_column in df.columns else "summary"
            if args.prediction_path:
                pred_df = DataLoader.load_csv(args.prediction_path)
                hyp_col = args.prediction_column if args.prediction_column in pred_df.columns else "summary"
                evaluator.evaluate_summarization(df[ref_col].tolist(), pred_df[hyp_col].tolist())
            else:
                logger.error("Summarization evaluation requires --prediction-path")
        else:
            raise ValueError(f"Unknown evaluation task: {args.task}")


if __name__ == "__main__":
    main()
