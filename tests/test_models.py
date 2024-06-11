"""Unit tests for model training functions."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.train import train_text_classification


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def sample_classification_data():
    return pd.DataFrame({
        "text": [
            "عقد بيع شقة في الرياض",
            "إيجار منزل في جدة",
            "تملك فيلا في مكة",
            "بيع أرض في الدمام",
            "إيجار شقة في الخبر",
            "تملك مزرعة في الطائف",
        ],
        "label": ["بيع", "إيجار", "تملك", "بيع", "إيجار", "تملك"],
    })


# ------------------------------------------------------------------
# Classification Training Tests
# ------------------------------------------------------------------

class TestTrainTextClassification:
    def test_logistic_regression(self, sample_classification_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_text_classification(
                model_type="logistic_regression",
                texts=sample_classification_data["text"].tolist(),
                labels=sample_classification_data["label"].tolist(),
                output_dir=str(Path(tmpdir) / "models"),
                test_size=0.5,
            )
            assert "model" in result
            assert "vectorizer" in result
            assert "accuracy" in result
            assert isinstance(result["accuracy"], float)
            assert result["accuracy"] >= 0.0

    def test_linear_svc(self, sample_classification_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_text_classification(
                model_type="linear_svc",
                texts=sample_classification_data["text"].tolist(),
                labels=sample_classification_data["label"].tolist(),
                output_dir=str(Path(tmpdir) / "models"),
                test_size=0.5,
            )
            assert "model" in result
            assert result["accuracy"] >= 0.0

    def test_data_path_loading(self, sample_classification_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "train.csv"
            sample_classification_data.to_csv(data_path, index=False, encoding="utf-8")

            result = train_text_classification(
                model_type="logistic_regression",
                data_path=str(data_path),
                output_dir=str(Path(tmpdir) / "models"),
                test_size=0.5,
            )
            assert "model" in result

    def test_artifacts_saved(self, sample_classification_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_text_classification(
                texts=sample_classification_data["text"].tolist(),
                labels=sample_classification_data["label"].tolist(),
                output_dir=str(tmpdir),
                test_size=0.5,
            )
            artifacts = list(Path(tmpdir).iterdir())
            assert any(p.suffix == ".pkl" for p in artifacts)
            assert any("metadata" in p.name for p in artifacts)
            assert any("tfidf" in p.name for p in artifacts)

    def test_empty_data_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            train_text_classification(texts=[], labels=[])

    def test_single_class_raises(self):
        texts = ["نص واحد", "نص ثاني", "نص ثالث"]
        labels = ["نفس", "نفس", "نفس"]
        with pytest.raises((ValueError, RuntimeError)):
            train_text_classification(
                texts=texts, labels=labels,
            )

    def test_unknown_model_type(self, sample_classification_data):
        with pytest.raises(ValueError, match="unknown model_type|Unknown model_type"):
            train_text_classification(
                model_type="unknown_model",
                texts=sample_classification_data["text"].tolist(),
                labels=sample_classification_data["label"].tolist(),
                test_size=0.5,
            )


# ------------------------------------------------------------------
# NER & Summarization training tests (lightweight)
# ------------------------------------------------------------------

class TestTrainNER:
    def test_imports_available(self):
        """NER training requires transformers; verify it raises if not installed."""
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            import datasets  # noqa: F401
        except ImportError:
            with pytest.raises((ImportError, ModuleNotFoundError)):
                from src.train import train_ner
                train_ner(texts=[], tags=[])
        else:
            from src.train import train_ner
            with pytest.raises(ValueError, match="must be provided"):
                train_ner()


class TestTrainSummarization:
    def test_imports_available(self):
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            import datasets  # noqa: F401
        except ImportError:
            with pytest.raises((ImportError, ModuleNotFoundError)):
                from src.train import train_summarization
                train_summarization(texts=[], summaries=[])
        else:
            from src.train import train_summarization
            with pytest.raises((ValueError, AssertionError)):
                train_summarization()
