"""
Arabic text preprocessing module for real estate documents.

Provides configurable pipelines for cleaning, normalizing, tokenizing,
and stemming Arabic text data.
"""

import logging
import re
import string
from typing import List, Optional, Dict, Any, Callable

import nltk
import pandas as pd

logger = logging.getLogger(__name__)

# Arabic character ranges
ARIC_CHAR_RANGE = r"\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff"
HARAKAT = r"\u064b-\u065f"  # Fatha, Damma, Kasra, etc.
TATWEEL = r"\u0640"
ALEF_VARIANTS = {
    "\u0622": "\u0627",  # ALEF WITH MADDA ABOVE -> ALEF
    "\u0623": "\u0627",  # ALEF WITH HAMZA ABOVE -> ALEF
    "\u0625": "\u0627",  # ALEF WITH HAMZA BELOW -> ALEF
    "\u0627": "\u0627",  # ALEF
}
TEH_MARBUTA = "\u0629"
HA = "\u0647"


class ArabicPreprocessor:
    """Handles Arabic text cleaning and normalization."""

    def __init__(self, config: Optional[Dict[str, bool]] = None):
        """
        Initialize preprocessor with optional configuration.

        Args:
            config: Dictionary of step -> bool flags. Defaults to all True.
                    Keys: remove_diacritics, remove_tatweel, normalize_alef,
                          normalize_teh_marbuta, remove_punctuation,
                          remove_english, remove_numbers, remove_extra_whitespace
        """
        self.config = {
            "remove_diacritics": True,
            "remove_tatweel": True,
            "normalize_alef": True,
            "normalize_teh_marbuta": True,
            "remove_punctuation": True,
            "remove_english": False,
            "remove_numbers": False,
            "remove_extra_whitespace": True,
        }
        if config:
            self.config.update(config)
        self._arabic_punct = (
            "،؛؟!""''""«»-ـ.():[]{}<>/\\@#$%^&*~`+|…"
        )
        self._punct_table = str.maketrans("", "", self._arabic_punct + string.punctuation)

    def remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics (Tashkeel)."""
        return re.sub(r"[\u064b-\u065f]", "", text)

    def remove_tatweel(self, text: str) -> str:
        """Remove Tatweel/Kashida characters."""
        return re.sub(r"\u0640+", "", text)

    def normalize_alef(self, text: str) -> str:
        """Normalize all Alef variants to bare Alef."""
        return "".join(ALEF_VARIANTS.get(c, c) for c in text)

    def normalize_teh_marbuta(self, text: str) -> str:
        """Normalize Teh Marbuta to Ha."""
        return text.replace(TEH_MARBUTA, HA)

    def remove_punctuation(self, text: str) -> str:
        """Remove Arabic and English punctuation."""
        return text.translate(self._punct_table)

    def remove_english(self, text: str) -> str:
        """Remove English/Latin characters."""
        return re.sub(r"[a-zA-Z]", "", text)

    def remove_numbers(self, text: str) -> str:
        """Remove digits (Arabic and Western)."""
        return re.sub(r"[0-9\u0660-\u0669\u06f0-\u06f9]", "", text)

    def remove_extra_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace and strip."""
        return re.sub(r"\s+", " ", text).strip()

    def clean(self, text: str) -> str:
        """
        Run all configured cleaning steps on a single text.

        Args:
            text: Raw Arabic text string.

        Returns:
            Cleaned text string.
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        if self.config["remove_diacritics"]:
            text = self.remove_diacritics(text)
        if self.config["remove_tatweel"]:
            text = self.remove_tatweel(text)
        if self.config["normalize_alef"]:
            text = self.normalize_alef(text)
        if self.config["normalize_teh_marbuta"]:
            text = self.normalize_teh_marbuta(text)
        if self.config["remove_punctuation"]:
            text = self.remove_punctuation(text)
        if self.config["remove_english"]:
            text = self.remove_english(text)
        if self.config["remove_numbers"]:
            text = self.remove_numbers(text)
        if self.config["remove_extra_whitespace"]:
            text = self.remove_extra_whitespace(text)
        return text

    def clean_dataframe(
        self, df: pd.DataFrame, text_column: str, new_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply cleaning to all rows in a DataFrame column.

        Args:
            df: Input DataFrame.
            text_column: Name of column containing text.
            new_column: Name for cleaned column. If None, overwrites text_column.

        Returns:
            DataFrame with cleaned text column added/updated.
        """
        out_col = new_column or text_column
        df = df.copy()
        df[out_col] = df[text_column].astype(str).apply(self.clean)
        logger.info("Cleaned %d rows in column '%s' -> '%s'", len(df), text_column, out_col)
        return df


class ArabicTokenizer:
    """Tokenizes Arabic text using NLTK with Arabic-specific tweaks."""

    def __init__(self, stopwords_path: Optional[str] = None):
        """
        Initialize tokenizer.

        Args:
            stopwords_path: Optional path to custom Arabic stopwords file.
        """
        self.stopwords: set = set()
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        try:
            arabic_stopwords = nltk.corpus.stopwords.words("arabic")
            self.stopwords = set(arabic_stopwords)
        except IOError:
            logger.warning("NLTK Arabic stopwords not available; using built-in fallback.")
            self.stopwords = self._default_arabic_stopwords()

        if stopwords_path:
            try:
                with open(stopwords_path, encoding="utf-8") as f:
                    custom_stops = {line.strip() for line in f if line.strip()}
                self.stopwords.update(custom_stops)
                logger.info("Loaded %d custom stopwords from %s", len(custom_stops), stopwords_path)
            except (FileNotFoundError, IOError) as exc:
                logger.error("Could not load stopwords from %s: %s", stopwords_path, exc)

    @staticmethod
    def _default_arabic_stopwords() -> set:
        """Return a built-in set of common Arabic stopwords."""
        return {
            "في", "من", "إلى", "على", "عن", "مع", "كان", "هذا", "هذه",
            "ذلك", "تلك", "هو", "هي", "هم", "هن", "أن", "إن", "ما",
            "لا", "هل", "أي", "كل", "بعض", "لم", "لن", "قد", "لقد",
            "إنه", "إنها", "حتى", "بعد", "قبل", "عند", "تحت", "فوق",
            "بين", "خلال", "حول", "حيث", "هناك", "هنا", "ثم", "أو",
            "و", "ف", "ب", "ل", "ك", "ا", "فقد", "لقد", "بل", "لكن",
            "ليس", "ليست", "كن", "تكون", "يكون", "كانت", "كانوا",
        }

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Arabic text into words.

        Args:
            text: Cleaned Arabic text.

        Returns:
            List of tokens.
        """
        if not text or not text.strip():
            return []
        tokens = nltk.word_tokenize(text)
        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Filter out stopwords from token list.

        Args:
            tokens: List of token strings.

        Returns:
            Filtered token list without stopwords.
        """
        return [t for t in tokens if t not in self.stopwords and len(t) > 1]

    def process(self, text: str, remove_stops: bool = True) -> List[str]:
        """
        Full tokenization pipeline: tokenize + optional stopword removal.

        Args:
            text: Input Arabic text.
            remove_stops: Whether to remove stopwords.

        Returns:
            List of processed tokens.
        """
        tokens = self.tokenize(text)
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        return tokens


class ArabicStemmer:
    """Light Arabic stemmer using pattern-based prefix/suffix stripping.

    Strips known prefixes and suffixes iteratively to handle stacked affixes
    (e.g. ``بالكتاب`` -> ``كتاب`` after removing ``ب`` then ``ال``).
    """

    # Grammatical prefixes sorted by length (longest first for greedy match)
    PREFIXES = ["ال", "س", "ف", "و", "ل", "ب", "ت", "ي", "ن", "أ", "إ"]
    SUFFIXES = ["ها", "هم", "هن", "كما", "كم", "كن", "نا", "ني", "وا", "ون", "ين", "ات", "ان", "ية"]

    def stem(self, word: str) -> str:
        """
        Light stemming: remove common prefixes and suffixes iteratively.

        Args:
            word: Arabic word.

        Returns:
            Stemmed word.
        """
        original = word
        # Iteratively strip stacked prefixes
        max_passes = 3
        for _ in range(max_passes):
            stripped = False
            for prefix in sorted(self.PREFIXES, key=len, reverse=True):
                if word.startswith(prefix) and len(word) > len(prefix) + 1:
                    word = word[len(prefix):]
                    stripped = True
                    break
            if not stripped:
                break
        # Iteratively strip stacked suffixes
        for _ in range(max_passes):
            stripped = False
            for suffix in sorted(self.SUFFIXES, key=len, reverse=True):
                if word.endswith(suffix) and len(word) > len(suffix) + 1:
                    word = word[:-len(suffix)]
                    stripped = True
                    break
            if not stripped:
                break
        return word if word else original

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stemming to a list of tokens."""
        return [self.stem(t) for t in tokens]


class PreprocessingPipeline:
    """
    Composable pipeline for Arabic text preprocessing.

    Steps (in order): clean -> tokenize -> stopword_removal -> stem
    """

    def __init__(
        self,
        preprocessor: Optional[ArabicPreprocessor] = None,
        tokenizer: Optional[ArabicTokenizer] = None,
        stemmer: Optional[ArabicStemmer] = None,
    ):
        self.preprocessor = preprocessor or ArabicPreprocessor()
        self.tokenizer = tokenizer or ArabicTokenizer()
        self.stemmer = stemmer or ArabicStemmer()

    def run(
        self,
        text: str,
        remove_stops: bool = True,
        apply_stemming: bool = False,
    ) -> Dict[str, Any]:
        """
        Run full preprocessing pipeline on a single text.

        Args:
            text: Raw Arabic text.
            remove_stops: Whether to remove stopwords after tokenization.
            apply_stemming: Whether to apply light stemming.

        Returns:
            Dictionary with keys: 'original', 'cleaned', 'tokens', 'stemmed_tokens'
        """
        cleaned = self.preprocessor.clean(text)
        tokens = self.tokenizer.process(cleaned, remove_stops=remove_stops)
        stemmed = self.stemmer.stem_tokens(tokens) if apply_stemming else tokens

        return {
            "original": text,
            "cleaned": cleaned,
            "tokens": tokens,
            "stemmed_tokens": stemmed,
        }

    def run_on_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        remove_stops: bool = True,
        apply_stemming: bool = False,
    ) -> pd.DataFrame:
        """
        Run pipeline on a DataFrame column, expanding results into new columns.

        Args:
            df: Input DataFrame.
            text_column: Name of text column.
            remove_stops: Passed to tokenizer.
            apply_stemming: Whether to stem tokens.

        Returns:
            DataFrame with added columns: cleaned_text, tokens, stemmed_tokens
        """
        results = df[text_column].astype(str).apply(
            lambda t: self.run(t, remove_stops=remove_stops, apply_stemming=apply_stemming)
        )
        df_out = df.copy()
        df_out["cleaned_text"] = results.apply(lambda r: r["cleaned"])
        df_out["tokens"] = results.apply(lambda r: r["tokens"])
        if apply_stemming:
            df_out["stemmed_tokens"] = results.apply(lambda r: r["stemmed_tokens"])
        logger.info("Preprocessing pipeline completed on %d rows", len(df_out))
        return df_out


def preprocess_arabic_text(
    input_path: str,
    output_path: str,
    config: Optional[Dict[str, bool]] = None,
    remove_stops: bool = True,
    apply_stemming: bool = False,
) -> None:
    """
    Convenience function: load text file, preprocess, save to CSV.

    Args:
        input_path: Path to raw text file (one document per line).
        output_path: Path to save preprocessed CSV.
        config: Optional preprocessor config dict.
        remove_stops: Remove Arabic stopwords.
        apply_stemming: Apply light stemming.
    """
    logger.info("Loading text from %s", input_path)
    with open(input_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    df = pd.DataFrame({"Text": lines})

    pipeline = PreprocessingPipeline(
        preprocessor=ArabicPreprocessor(config) if config else ArabicPreprocessor()
    )
    df = pipeline.run_on_dataframe(df, "Text", remove_stops=remove_stops, apply_stemming=apply_stemming)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Preprocessed data saved to %s (%d rows)", output_path, len(df))
