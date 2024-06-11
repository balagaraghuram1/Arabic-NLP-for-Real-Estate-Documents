"""
Utility functions for data loading, saving, serialization, and text statistics.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Load data from various file formats."""

    @staticmethod
    def load_text(filepath: str, encoding: str = "utf-8") -> List[str]:
        """Load lines from a plain text file."""
        with open(filepath, encoding=encoding) as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def load_csv(
        filepath: str,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load data from CSV file."""
        df = pd.read_csv(filepath, encoding="utf-8", **kwargs)
        logger.info("Loaded CSV from %s: %d rows, columns=%s", filepath, len(df), list(df.columns))
        return df

    @staticmethod
    def load_json(filepath: str) -> Union[Dict[str, Any], List[Any]]:
        """Load data from JSON file."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded JSON from %s", filepath)
        return data

    @staticmethod
    def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
        """Load data from JSON Lines file (one JSON object per line)."""
        records = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        logger.info("Loaded JSONL from %s: %d records", filepath, len(records))
        return records

    @staticmethod
    def load_dataframe(
        filepath: str,
        file_type: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Auto-detect file type and load into DataFrame.

        Args:
            filepath: Path to file.
            file_type: One of 'csv', 'json', 'jsonl', 'txt'. If None, inferred from extension.

        Returns:
            DataFrame with data.
        """
        ext = (file_type or Path(filepath).suffix.lower()).lstrip(".")
        if ext == "csv":
            return DataLoader.load_csv(filepath, **kwargs)
        elif ext == "json":
            data = DataLoader.load_json(filepath)
            if isinstance(data, list):
                return pd.DataFrame(data)
            return pd.DataFrame([data])
        elif ext == "jsonl":
            return pd.DataFrame(DataLoader.load_jsonl(filepath))
        elif ext == "txt":
            lines = DataLoader.load_text(filepath)
            return pd.DataFrame({"text": lines})
        else:
            raise ValueError(f"Unsupported file type: {ext}")


class ResultSaver:
    """Save results to various output formats."""

    @staticmethod
    def save_csv(data: pd.DataFrame, filepath: str, **kwargs: Any) -> None:
        """Save DataFrame to CSV."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data.to_csv(filepath, index=False, encoding="utf-8-sig", **kwargs)
        logger.info("Saved CSV to %s (%d rows)", filepath, len(data))

    @staticmethod
    def save_json(data: Any, filepath: str, **kwargs: Any) -> None:
        """Save data to JSON file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, **kwargs)
        logger.info("Saved JSON to %s", filepath)

    @staticmethod
    def save_text(lines: List[str], filepath: str, encoding: str = "utf-8") -> None:
        """Save lines to a text file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding=encoding) as f:
            f.writelines(line + "\n" for line in lines)
        logger.info("Saved text to %s (%d lines)", filepath, len(lines))


class ModelSerializer:
    """Serialize and deserialize models and vectorizers."""

    @staticmethod
    def save(model: Any, filepath: str) -> None:
        """Save model/vectorizer using joblib."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        joblib.dump(model, filepath)
        logger.info("Model saved to %s", filepath)

    @staticmethod
    def load(filepath: str) -> Any:
        """Load model/vectorizer using joblib."""
        model = joblib.load(filepath)
        logger.info("Model loaded from %s", filepath)
        return model


class ArabicTextStats:
    """Compute statistics for Arabic text."""

    @staticmethod
    def word_count(text: str) -> int:
        """Count number of words in text."""
        return len(text.split())

    @staticmethod
    def char_count(text: str) -> int:
        """Count number of characters."""
        return len(text)

    @staticmethod
    def avg_word_length(text: str) -> float:
        """Average word length in characters."""
        words = text.split()
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)

    @staticmethod
    def arabic_char_ratio(text: str) -> float:
        """Ratio of Arabic characters to total characters."""
        if not text:
            return 0.0
        arabic_count = sum(1 for c in text if "\u0600" <= c <= "\u06ff" or "\u0750" <= c <= "\u077f")
        return arabic_count / len(text)

    @staticmethod
    def unique_word_count(text: str) -> int:
        """Count of unique words."""
        return len(set(text.split()))

    @staticmethod
    def lexical_diversity(text: str) -> float:
        """Type-token ratio (unique words / total words)."""
        words = text.split()
        total = len(words)
        if total == 0:
            return 0.0
        return len(set(words)) / total

    @staticmethod
    def summary(text: str) -> Dict[str, Any]:
        """Return a summary dictionary of all statistics."""
        return {
            "word_count": ArabicTextStats.word_count(text),
            "char_count": ArabicTextStats.char_count(text),
            "avg_word_length": round(ArabicTextStats.avg_word_length(text), 2),
            "arabic_char_ratio": round(ArabicTextStats.arabic_char_ratio(text), 4),
            "unique_words": ArabicTextStats.unique_word_count(text),
            "lexical_diversity": round(ArabicTextStats.lexical_diversity(text), 4),
        }


def load_text_data(filepath: str) -> List[str]:
    """Legacy wrapper — load lines from a text file."""
    return DataLoader.load_text(filepath)


def save_results(filepath: str, results: List[str]) -> None:
    """Legacy wrapper — save lines to a text file."""
    ResultSaver.save_text(results, filepath)
