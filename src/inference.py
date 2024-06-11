"""
Inference engine for Arabic NLP tasks.

Supports classification, NER, and summarization inference
with batch processing and multiple output formats.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Production inference engine for Arabic NLP models."""

    def __init__(self, device: str = "cpu"):
        """
        Args:
            device: 'cpu' or 'cuda'.
        """
        self.device = device
        self._classification_model = None
        self._vectorizer = None
        self._ner_model = None
        self._ner_tokenizer = None
        self._summarization_model = None
        self._summarization_tokenizer = None
        self._labels = None
        self._label_to_id = None

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def load_classification_model(self, model_dir: str) -> None:
        """Load a trained classification model and TF-IDF vectorizer."""
        model_path = None
        for fname in os.listdir(model_dir):
            if fname.startswith("text_classification") and fname.endswith(".pkl"):
                model_path = os.path.join(model_dir, fname)
                break
        if model_path is None:
            raise FileNotFoundError(f"No classification model (*.pkl) found in {model_dir}")

        vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        meta_path = os.path.join(model_dir, "classification_metadata.json")

        self._classification_model = joblib.load(model_path)
        self._vectorizer = joblib.load(vec_path) if os.path.exists(vec_path) else None

        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            self._labels = meta.get("labels")
            self._label_to_id = meta.get("label_to_id")

        logger.info("Classification model loaded from %s", model_path)

    def predict_classification(
        self,
        texts: Union[str, List[str]],
        return_proba: bool = False,
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """
        Predict class labels for input texts.

        Args:
            texts: Single string or list of strings.
            return_proba: If True, return dicts with label + probabilities.

        Returns:
            List of predicted labels, or list of dicts with 'label' and 'probabilities'.
        """
        if isinstance(texts, str):
            texts = [texts]
        if self._classification_model is None:
            raise RuntimeError("Classification model not loaded. Call load_classification_model() first.")

        X = self._vectorizer.transform(texts) if self._vectorizer else texts
        preds = self._classification_model.predict(X)

        # Map numeric predictions back to label strings using label_to_id
        if self._label_to_id:
            id_to_label = {v: k for k, v in self._label_to_id.items()}
            pred_labels = [id_to_label.get(p, str(p)) for p in preds]
        elif self._labels:
            pred_labels = [self._labels[p] if isinstance(p, (int, np.integer)) else p for p in preds]
        else:
            pred_labels = [str(p) for p in preds]

        if not return_proba:
            logger.info("Classification complete: %d samples", len(texts))
            return pred_labels

        results = []
        for i, text in enumerate(texts):
            entry = {"text": text, "label": pred_labels[i]}
            if hasattr(self._classification_model, "predict_proba"):
                proba = self._classification_model.predict_proba(X[i:i+1])[0]
                entry["probabilities"] = {
                    (self._labels[j] if self._labels and j < len(self._labels) else str(j)): float(p)
                    for j, p in enumerate(proba)
                }
            results.append(entry)
        logger.info("Classification complete with probabilities: %d samples", len(texts))
        return results

    # ------------------------------------------------------------------
    # NER
    # ------------------------------------------------------------------

    def load_ner_model(self, model_dir: str) -> None:
        """Load a trained NER model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
        except ImportError:
            raise ImportError("transformers required for NER inference. pip install transformers torch")

        model_path = os.path.join(model_dir, "ner_model")
        labels_path = os.path.join(model_dir, "ner_labels.json")

        self._ner_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._ner_model = AutoModelForTokenClassification.from_pretrained(model_path)
        self._ner_model.to(self.device)
        self._ner_model.eval()

        if os.path.exists(labels_path):
            with open(labels_path, encoding="utf-8") as f:
                ner_labels = json.load(f)
            self._ner_labels = ner_labels.get("labels", [])
        else:
            self._ner_labels = list(self._ner_model.config.id2label.values())

        logger.info("NER model loaded from %s (%d labels)", model_path, len(self._ner_labels))

    def predict_ner(
        self,
        texts: Union[str, List[str]],
        aggregation_strategy: str = "simple",
    ) -> List[Dict[str, Any]]:
        """
        Run NER on input texts.

        Args:
            texts: Single string or list of strings.
            aggregation_strategy: How to group sub-token predictions ('simple', 'none', 'first').

        Returns:
            List of dicts with 'text', 'entities' (list of entity dicts).
        """
        if isinstance(texts, str):
            texts = [texts]
        if self._ner_model is None:
            raise RuntimeError("NER model not loaded. Call load_ner_model() first.")

        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("transformers required for NER inference.")

        ner_pipeline = pipeline(
            "ner",
            model=self._ner_model,
            tokenizer=self._ner_tokenizer,
            device=0 if self.device == "cuda" else -1,
            aggregation_strategy=aggregation_strategy,
        )

        results = []
        for text in tqdm(texts, desc="NER inference"):
            entities = ner_pipeline(text)
            results.append({"text": text, "entities": entities})

        logger.info("NER inference complete: %d samples", len(texts))
        return results

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def load_summarization_model(self, model_dir: str) -> None:
        """Load a trained summarization model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError:
            raise ImportError("transformers required for summarization inference.")

        model_path = os.path.join(model_dir, "summarization_model")
        self._summarization_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._summarization_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self._summarization_model.to(self.device)
        self._summarization_model.eval()

        logger.info("Summarization model loaded from %s", model_path)

    def predict_summarization(
        self,
        texts: Union[str, List[str]],
        max_input_length: int = 512,
        max_summary_length: int = 128,
        num_beams: int = 4,
        **generate_kwargs: Any,
    ) -> List[Dict[str, str]]:
        """
        Generate summaries for input texts.

        Args:
            texts: Single string or list of strings.
            max_input_length: Truncate source text to this length.
            max_summary_length: Max generated summary length.
            num_beams: Beam search width.
            **generate_kwargs: Additional kwargs for model.generate().

        Returns:
            List of dicts with 'text' and 'summary'.
        """
        if isinstance(texts, str):
            texts = [texts]
        if self._summarization_model is None:
            raise RuntimeError("Summarization model not loaded. Call load_summarization_model() first.")

        results = []
        for text in tqdm(texts, desc="Summarization inference"):
            inputs = self._summarization_tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_input_length,
                return_tensors="pt",
            ).to(self.device)

            output_ids = self._summarization_model.generate(
                **inputs,
                max_length=max_summary_length,
                num_beams=num_beams,
                early_stopping=True,
                **generate_kwargs,
            )
            summary = self._summarization_tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
            results.append({"text": text, "summary": summary})

        logger.info("Summarization inference complete: %d samples", len(texts))
        return results

    # ------------------------------------------------------------------
    # Batch / Save
    # ------------------------------------------------------------------

    def batch_process(
        self,
        texts: List[str],
        task: str = "classification",
        batch_size: int = 32,
        output_path: Optional[str] = None,
    ) -> List[Any]:
        """
        Process texts in batches.

        Args:
            texts: Input texts.
            task: 'classification', 'ner', or 'summarization'.
            batch_size: Batch size.
            output_path: If provided, save results to file.

        Returns:
            List of results.
        """
        all_results = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Batch {task}"):
            batch = texts[i : i + batch_size]
            if task == "classification":
                batch_results = self.predict_classification(batch)
            elif task == "ner":
                batch_results = self.predict_ner(batch)
            elif task == "summarization":
                batch_results = self.predict_summarization(batch)
            else:
                raise ValueError(f"Unknown task: {task}")
            all_results.extend(batch_results)

        if output_path:
            self._save_results(all_results, output_path, task)

        return all_results

    def _save_results(
        self, results: List[Any], output_path: str, task: str
    ) -> None:
        """Save results to file based on task type."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if task == "classification":
            df = pd.DataFrame(results) if isinstance(results[0], dict) else pd.DataFrame({"prediction": results})
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
        elif task == "ner":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        elif task == "summarization":
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False, encoding="utf-8-sig")

        logger.info("Results saved to %s", output_path)


def perform_inference(
    task: str,
    model_path: str,
    input_path: str,
    output_path: str,
    device: str = "cpu",
    batch_size: int = 32,
) -> None:
    """
    Convenience function: load model, run inference, save results.

    Args:
        task: 'classification', 'ner', or 'summarization'.
        model_path: Path to directory containing model artifacts.
        input_path: Path to input CSV/text file.
        output_path: Path to save results.
        device: 'cpu' or 'cuda'.
        batch_size: Inference batch size.
    """
    engine = InferenceEngine(device=device)

    if task == "classification":
        engine.load_classification_model(model_path)
    elif task == "ner":
        engine.load_ner_model(model_path)
    elif task == "summarization":
        engine.load_summarization_model(model_path)
    else:
        raise ValueError(f"Unknown task: {task}")

    # Load input data
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path, encoding="utf-8")
        text_col = next((c for c in df.columns if c.lower() in ("text", "cleaned_text", "content")), df.columns[0])
        texts = df[text_col].astype(str).tolist()
    else:
        with open(input_path, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    if not texts:
        logger.warning("No input texts found in %s", input_path)
        return

    engine.batch_process(texts, task=task, batch_size=batch_size, output_path=output_path)
