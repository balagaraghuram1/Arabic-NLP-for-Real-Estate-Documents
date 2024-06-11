"""
Arabic NLP for Real Estate Documents.

A production-grade NLP toolkit for processing Arabic real estate documents,
supporting text classification, named entity recognition, and summarization.
"""

__version__ = "1.0.0"
__author__ = "balaga raghuram"

from src.preprocess import ArabicPreprocessor, PreprocessingPipeline
from src.train import (
    train_text_classification,
    train_ner,
    train_summarization,
)
from src.inference import InferenceEngine
from src.evaluate import Evaluator
from src.utils import DataLoader, ResultSaver, ArabicTextStats

__all__ = [
    "ArabicPreprocessor",
    "PreprocessingPipeline",
    "train_text_classification",
    "train_ner",
    "train_summarization",
    "InferenceEngine",
    "Evaluator",
    "DataLoader",
    "ResultSaver",
    "ArabicTextStats",
]
