"""
Training module for Arabic NLP models.

Supports:
- Text classification (LogisticRegression, LinearSVC) with TF-IDF features
- Named Entity Recognition (BERT-based via transformers)
- Summarization (T5/mT5/AraT5 via transformers)
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text Classification
# ---------------------------------------------------------------------------


def train_text_classification(
    model_type: str = "logistic_regression",
    texts: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    data_path: Optional[str] = None,
    text_column: str = "text",
    label_column: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
    tfidf_max_features: int = 5000,
    tfidf_ngram_range: Tuple[int, int] = (1, 2),
    output_dir: str = "models",
    hyperparameter_tuning: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Train a text classification model with TF-IDF features.

    Args:
        model_type: One of 'logistic_regression', 'linear_svc'.
        texts: List of training texts. If None, loaded from data_path.
        labels: List of training labels. If None, loaded from data_path.
        data_path: Path to CSV with text and label columns.
        text_column: Column name for texts.
        label_column: Column name for labels.
        test_size: Fraction for validation split.
        random_state: Seed for reproducibility.
        tfidf_max_features: Max features for TF-IDF vectorizer.
        tfidf_ngram_range: N-gram range for TF-IDF.
        output_dir: Directory to save model artifacts.
        hyperparameter_tuning: Run GridSearchCV if True.
        **kwargs: Additional params passed to the classifier.

    Returns:
        Dict with keys: 'model', 'vectorizer', 'label_encoder',
                        'accuracy', 'classification_report'.
    """
    logger.info("Starting text classification training (model=%s)", model_type)

    if texts is None or labels is None:
        if data_path is None:
            raise ValueError("Either texts/labels or data_path must be provided.")
        df = pd.read_csv(data_path, encoding="utf-8")
        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].astype(str).tolist()

    unique_labels = sorted(set(labels))
    label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
    y = np.array([label_to_id[lbl] for lbl in labels])
    logger.info("Labels: %s", unique_labels)

    # Split (try stratified; fall back to non-stratified for small datasets)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            texts, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        logger.warning("Stratified split failed (too few samples per class). Falling back to random split.")
        X_train, X_val, y_train, y_val = train_test_split(
            texts, y, test_size=test_size, random_state=random_state
        )
    if len(set(y_val)) < 1:
        raise ValueError("Validation set must contain at least one sample.")

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=tfidf_max_features,
        ngram_range=tfidf_ngram_range,
        analyzer="word",
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # Model selection
    if model_type == "logistic_regression":
        base_model = LogisticRegression(
            max_iter=1000, random_state=random_state, **kwargs
        )
        param_grid = {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs", "liblinear"]}
    elif model_type == "linear_svc":
        base_model = LinearSVC(max_iter=10000, random_state=random_state, **kwargs)
        param_grid = {"C": [0.1, 1.0, 10.0], "loss": ["hinge", "squared_hinge"]}
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'logistic_regression' or 'linear_svc'.")

    if hyperparameter_tuning:
        logger.info("Running GridSearchCV for %s", model_type)
        model = GridSearchCV(
            base_model, param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=0
        )
        model.fit(X_train_vec, y_train)
        logger.info("Best params: %s, Best CV score: %.4f", model.best_params_, model.best_score_)
    else:
        model = base_model
        model.fit(X_train_vec, y_train)

    # Evaluate
    from sklearn.metrics import classification_report, accuracy_score

    y_pred = model.predict(X_val_vec)
    acc = accuracy_score(y_val, y_pred)
    present_labels = sorted(set(y_val) | set(y_pred))
    present_names = [unique_labels[i] for i in present_labels if i < len(unique_labels)]
    report = classification_report(y_val, y_pred, labels=present_labels, target_names=present_names, output_dict=True, zero_division=0)

    logger.info("Validation accuracy: %.4f", acc)

    # Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"text_classification_{model_type}.pkl")
    vec_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    metadata_path = os.path.join(output_dir, "classification_metadata.json")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": model_type,
                "labels": unique_labels,
                "label_to_id": label_to_id,
                "accuracy": acc,
                "val_samples": len(y_val),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Model saved to %s", model_path)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "label_encoder": label_to_id,
        "labels": unique_labels,
        "accuracy": acc,
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Named Entity Recognition
# ---------------------------------------------------------------------------


def train_ner(
    model_type: str = "bert",
    model_name: str = "aubmindlab/bert-base-arabertv02",
    train_data_path: Optional[str] = None,
    val_data_path: Optional[str] = None,
    texts: Optional[List[str]] = None,
    tags: Optional[List[List[str]]] = None,
    output_dir: str = "models",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    device: str = "cpu",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Train a token classification model for Arabic NER.

    Uses HuggingFace transformers (BertForTokenClassification / AraBERT).

    Args:
        model_type: Model architecture ('bert', 'arabert').
        model_name: HuggingFace model checkpoint name.
        train_data_path: Path to JSON/JSONL with 'tokens' and 'tags' fields.
        val_data_path: Optional validation data path.
        texts: List of token lists (alternative to data_path).
        tags: List of tag lists (alternative to data_path).
        output_dir: Directory to save model and tokenizer.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Peak learning rate.
        max_length: Max token sequence length.
        device: 'cpu' or 'cuda'.
        **kwargs: Additional Trainer arguments.

    Returns:
        Dict with keys: 'model', 'tokenizer', 'label_list', 'trainer'.
    """
    logger.info("Starting NER training (model=%s, checkpoint=%s)", model_type, model_name)

    # Lazy import to avoid hard dependency
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForTokenClassification,
            TrainingArguments,
            Trainer,
            DataCollatorForTokenClassification,
        )
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(
            "transformers, torch, and datasets required for NER training. "
            "Install with: pip install transformers torch datasets seqeval"
        ) from exc

    # Load data
    if train_data_path:
        with open(train_data_path, encoding="utf-8") as f:
            raw = json.load(f) if train_data_path.endswith(".json") else [json.loads(line) for line in f]
        texts = [item["tokens"] for item in raw]
        tags = [item["tags"] for item in raw]

    if not texts or not tags:
        raise ValueError("Training texts/tags must be provided via data_path or arguments.")

    # Build label set
    unique_tags = sorted(set(t for tag_seq in tags for t in tag_seq))
    tag_to_id = {t: i for i, t in enumerate(unique_tags)}
    id_to_tag = {i: t for t, i in tag_to_id.items()}
    logger.info("NER tags (%d): %s", len(unique_tags), unique_tags)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label_all_tokens = kwargs.pop("label_all_tokens", True)

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            is_split_into_words=True,
        )
        labels = []
        for i, tag_seq in enumerate(examples["tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx or label_all_tokens:
                    label_ids.append(tag_to_id.get(tag_seq[word_idx], -100))
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized["labels"] = labels
        return tokenized

    dataset = Dataset.from_dict({"tokens": texts, "tags": tags})
    dataset = dataset.map(tokenize_and_align, batched=True, remove_columns=["tokens", "tags"])

    if val_data_path is not None:
        with open(val_data_path, encoding="utf-8") as f:
            val_raw = json.load(f) if val_data_path.endswith(".json") else [json.loads(line) for line in f]
        val_dataset = Dataset.from_dict(
            {"tokens": [item["tokens"] for item in val_raw], "tags": [item["tags"] for item in val_raw]}
        )
        val_dataset = val_dataset.map(tokenize_and_align, batched=True, remove_columns=["tokens", "tags"])
    else:
        split = dataset.train_test_split(test_size=0.1, seed=42)
        dataset = split["train"]
        val_dataset = split["test"]

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(unique_tags), id2label=id_to_tag, label2id=tag_to_id
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "ner_checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "ner_logs"),
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        **kwargs,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save final model
    model.save_pretrained(os.path.join(output_dir, "ner_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "ner_model"))
    with open(os.path.join(output_dir, "ner_labels.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": unique_tags, "tag_to_id": tag_to_id}, f, ensure_ascii=False, indent=2)

    logger.info("NER model saved to %s/ner_model", output_dir)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "label_list": unique_tags,
        "trainer": trainer,
    }


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


def train_summarization(
    model_type: str = "t5",
    model_name: str = "google/mt5-small",
    train_data_path: Optional[str] = None,
    val_data_path: Optional[str] = None,
    texts: Optional[List[str]] = None,
    summaries: Optional[List[str]] = None,
    output_dir: str = "models",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 3e-5,
    max_input_length: int = 512,
    max_target_length: int = 128,
    device: str = "cpu",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Train a seq2seq summarization model (T5/mT5/AraT5).

    Args:
        model_type: Architecture ('t5', 'mt5').
        model_name: HuggingFace model checkpoint (e.g. 'google/mt5-small',
                    'UBC-NLP/AraT5-base').
        train_data_path: Path to CSV/JSON with 'text' and 'summary' fields.
        val_data_path: Optional validation data path.
        texts: List of input texts (alternative to data_path).
        summaries: List of target summaries.
        output_dir: Directory to save model and tokenizer.
        num_epochs: Training epochs.
        batch_size: Training batch size.
        learning_rate: Peak learning rate.
        max_input_length: Max source sequence length.
        max_target_length: Max target sequence length.
        device: 'cpu' or 'cuda'.
        **kwargs: Additional TrainingArguments.

    Returns:
        Dict with 'model', 'tokenizer', 'trainer'.
    """
    logger.info("Starting summarization training (model=%s, checkpoint=%s)", model_type, model_name)

    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
            Seq2SeqTrainingArguments,
            Seq2SeqTrainer,
            DataCollatorForSeq2Seq,
        )
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(
            "transformers, torch, datasets required for summarization training."
        ) from exc

    # Load data
    if train_data_path:
        if train_data_path.endswith(".csv"):
            df = pd.read_csv(train_data_path, encoding="utf-8")
            texts = df["text"].astype(str).tolist()
            summaries = df["summary"].astype(str).tolist()
        else:
            with open(train_data_path, encoding="utf-8") as f:
                raw = json.load(f) if train_data_path.endswith(".json") else [json.loads(line) for line in f]
            texts = [item.get("text", item.get("source", "")) for item in raw]
            summaries = [item.get("summary", item.get("target", "")) for item in raw]

    if not texts or not summaries:
        raise ValueError("Training texts/summaries must be provided.")
    if len(texts) != len(summaries):
        raise ValueError("texts and summaries must have the same length.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def preprocess_fn(examples):
        inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_input_length,
        )
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                examples["summary"],
                truncation=True,
                padding="max_length",
                max_length=max_target_length,
            )
        inputs["labels"] = targets["input_ids"]
        return inputs

    dataset = Dataset.from_dict({"text": texts, "summary": summaries})
    dataset = dataset.map(preprocess_fn, batched=True, remove_columns=["text", "summary"])

    if val_data_path is not None:
        if val_data_path.endswith(".csv"):
            df_val = pd.read_csv(val_data_path, encoding="utf-8")
            val_texts = df_val["text"].astype(str).tolist()
            val_summaries = df_val["summary"].astype(str).tolist()
        else:
            with open(val_data_path, encoding="utf-8") as f:
                val_raw = json.load(f) if val_data_path.endswith(".json") else [json.loads(line) for line in f]
            val_texts = [item.get("text", item.get("source", "")) for item in val_raw]
            val_summaries = [item.get("summary", item.get("target", "")) for item in val_raw]
        val_dataset = Dataset.from_dict({"text": val_texts, "summary": val_summaries})
        val_dataset = val_dataset.map(preprocess_fn, batched=True, remove_columns=["text", "summary"])
    else:
        split = dataset.train_test_split(test_size=0.1, seed=42)
        dataset = split["train"]
        val_dataset = split["test"]

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(output_dir, "summarization_checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "summarization_logs"),
        logging_steps=10,
        save_total_limit=2,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        **kwargs,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(os.path.join(output_dir, "summarization_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "summarization_model"))
    logger.info("Summarization model saved to %s/summarization_model", output_dir)

    return {"model": model, "tokenizer": tokenizer, "trainer": trainer}
