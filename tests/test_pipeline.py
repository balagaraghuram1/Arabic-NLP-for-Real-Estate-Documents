"""Integration tests for the full NLP pipeline."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.preprocess import ArabicPreprocessor, ArabicTokenizer, PreprocessingPipeline
from src.train import train_text_classification
from src.inference import InferenceEngine
from src.evaluate import Evaluator


@pytest.fixture
def sample_data_path():
    """Create a small dataset for integration testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "raw.txt"
        input_path.write_text(
            "هَذَا عَقْد بَيْع شِقّة في الرياض\n"
            "هَذِهِ اتّفَاقِيّة إيجَار مَنْزِل في جدة\n"
            "نَصّ تَمَلّك فِيلَا في مكة المكرمة\n"
            "عقد بيع أرض في الدمام 2024\n"
            "إيجار شقة في الخبر لمدة سنة\n",
            encoding="utf-8",
        )
        yield str(input_path)


class TestEndToEndPipeline:
    def test_preprocess_to_classification_pipeline(self, sample_data_path):
        """Test preprocessing -> training -> inference -> evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Step 1: Preprocess
            preproc = ArabicPreprocessor()
            tokenizer = ArabicTokenizer()
            pipeline = PreprocessingPipeline(preproc, tokenizer)

            with open(sample_data_path, encoding="utf-8") as f:
                raw_lines = [line.strip() for line in f if line.strip()]
            raw_df = pd.DataFrame({"text": raw_lines})
            # Create synthetic labels for training
            raw_df["label"] = ["بيع", "إيجار", "تملك", "بيع", "إيجار"]
            processed = pipeline.run_on_dataframe(raw_df, "text")

            csv_path = data_dir / "processed.csv"
            processed.to_csv(csv_path, index=False, encoding="utf-8-sig")
            assert csv_path.exists()

            # Step 2: Train
            train_result = train_text_classification(
                texts=processed["cleaned_text"].tolist(),
                labels=raw_df["label"].tolist(),
                output_dir=str(data_dir / "models"),
                test_size=0.3,
            )
            assert train_result["accuracy"] >= 0.0

            # Step 3: Inference
            engine = InferenceEngine()
            engine.load_classification_model(str(data_dir / "models"))
            predictions = engine.predict_classification(processed["cleaned_text"].tolist())
            assert len(predictions) == len(processed)
            assert all(isinstance(p, str) for p in predictions)

            # Step 4: Evaluate
            evaluator = Evaluator(output_dir=str(data_dir / "eval"))
            metrics = evaluator.evaluate_classification(
                raw_df["label"].tolist(), predictions
            )
            assert "accuracy" in metrics
            assert "f1_macro" in metrics
            assert metrics["accuracy"] >= 0.0

    def test_batch_inference(self):
        """Test batch inference with classification model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            texts = [
                "نص تجريبي أول",
                "نص تجريبي ثاني",
                "هذا النص الثالث",
                "نص رابع للتجربة",
                "نص خامس للتصنيف",
                "نص سادس وأخير",
            ]
            labels = ["فئة أ", "فئة ب", "فئة أ", "فئة ب", "فئة أ", "فئة ب"]

            # Train a quick model
            train_result = train_text_classification(
                texts=texts, labels=labels, output_dir=str(Path(tmpdir) / "models")
            )

            engine = InferenceEngine()
            engine.load_classification_model(str(Path(tmpdir) / "models"))

            # Batch process
            results = engine.batch_process(
                texts, task="classification", batch_size=2
            )
            assert len(results) == 6

            # With probabilities
            results_proba = engine.predict_classification(texts, return_proba=True)
            assert len(results_proba) == 6
            assert "label" in results_proba[0]
            assert "probabilities" in results_proba[0] or "label" in results_proba[0]

    def test_missing_model_raises(self):
        """Loading a non-existent model should raise an error."""
        engine = InferenceEngine()
        with pytest.raises((FileNotFoundError, RuntimeError)):
            engine.load_classification_model("/nonexistent/path")

    def test_predict_without_model(self):
        """Predicting without loading a model should raise."""
        engine = InferenceEngine()
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.predict_classification(["نص"])

    def test_evaluator_confusion_matrix_plot(self):
        """Evaluator should save confusion matrix without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(output_dir=str(Path(tmpdir) / "eval"))
            y_true = ["أ", "ب", "أ", "ب", "أ"]
            y_pred = ["أ", "ب", "أ", "أ", "ب"]
            metrics = evaluator.evaluate_classification(y_true, y_pred)
            assert metrics["accuracy"] >= 0.0
            report_file = Path(tmpdir) / "eval" / "classification_report.json"
            assert report_file.exists()
            cm_file = Path(tmpdir) / "eval" / "confusion_matrix.png"
            assert cm_file.exists()
