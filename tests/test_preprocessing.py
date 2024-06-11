"""Unit tests for Arabic text preprocessing."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.preprocess import (
    ArabicPreprocessor,
    ArabicTokenizer,
    ArabicStemmer,
    PreprocessingPipeline,
    preprocess_arabic_text,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def sample_text() -> str:
    return (
        "هَذَا نَصّ عَرَبِي للتّجربة فيه تشكيل وتطويل وكثير من علامات الترقيم! "
        "ويحتوي أيضاً على أرقام 123 وأحرف إنجليزية English."
    )


@pytest.fixture
def preprocessor() -> ArabicPreprocessor:
    return ArabicPreprocessor()


@pytest.fixture
def tokenizer() -> ArabicTokenizer:
    return ArabicTokenizer()


@pytest.fixture
def stemmer() -> ArabicStemmer():
    return ArabicStemmer()


# ------------------------------------------------------------------
# ArabicPreprocessor tests
# ------------------------------------------------------------------

class TestArabicPreprocessor:
    def test_remove_diacritics(self, preprocessor):
        assert preprocessor.remove_diacritics("هَذَا") == "هذا"

    def test_remove_tatweel(self, preprocessor):
        assert preprocessor.remove_tatweel("نصـــ طويل") == "نص طويل"

    def test_normalize_alef(self, preprocessor):
        assert preprocessor.normalize_alef("آية إيمان أحمد") == "اية ايمان احمد"

    def test_normalize_teh_marbuta(self, preprocessor):
        assert preprocessor.normalize_teh_marbuta("مدرسة جامعة") == "مدرسه جامعه"

    def test_remove_punctuation(self, preprocessor):
        result = preprocessor.remove_punctuation("مرحباً! كيف حالك؟")
        assert "!" not in result
        assert "؟" not in result

    def test_remove_punctuation_preserves_arabic(self, preprocessor):
        result = preprocessor.remove_punctuation("مرحباً، كيف حالك")
        assert "مرحباً" in result
        assert "كيف" in result
        assert "حالك" in result

    def test_remove_english(self, preprocessor):
        assert preprocessor.remove_english("نص مع English حروف") == "نص مع  حروف"

    def test_remove_numbers(self, preprocessor):
        assert preprocessor.remove_numbers("123 و ٤٥٦") == " و "

    def test_remove_extra_whitespace(self, preprocessor):
        assert preprocessor.remove_extra_whitespace("  كثير    مسافات   ") == "كثير مسافات"

    def test_clean_full_pipeline(self, preprocessor, sample_text):
        cleaned = preprocessor.clean(sample_text)
        assert "هَذَا" not in cleaned  # diacritics removed
        assert "English" in cleaned or "انجليزي" not in cleaned  # depends on config
        assert len(cleaned) > 0
        assert "  " not in cleaned  # no double spaces

    def test_clean_empty_string(self, preprocessor):
        assert preprocessor.clean("") == ""
        assert preprocessor.clean(None) == ""  # type: ignore

    def test_clean_dataframe(self, preprocessor):
        df = pd.DataFrame({"text": ["هَذَا نَص", "نَصّ آخر"]})
        result = preprocessor.clean_dataframe(df, "text", "cleaned")
        assert "cleaned" in result.columns
        assert result.loc[0, "cleaned"] == "هذا نص"

    @pytest.mark.parametrize("config,expected_note", [
        ({"remove_diacritics": False}, "with-diacritics"),
        ({"remove_punctuation": False}, "with-punctuation"),
    ])
    def test_config_overrides(self, config, expected_note):
        proc = ArabicPreprocessor(config)
        text = "هَذَا، نَصّ!"
        result = proc.clean(text)
        if expected_note == "with-diacritics":
            assert "َ" in result
        elif expected_note == "with-punctuation":
            assert "،" in result or "!" in result


# ------------------------------------------------------------------
# ArabicTokenizer tests
# ------------------------------------------------------------------

class TestArabicTokenizer:
    def test_tokenize(self, tokenizer):
        tokens = tokenizer.tokenize("هذا نص عربي للتجربة")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_remove_stopwords(self, tokenizer):
        tokens = ["في", "المنزل", "هذا", "كتاب"]
        filtered = tokenizer.remove_stopwords(tokens)
        assert "في" not in filtered
        assert "هذا" not in filtered
        assert "المنزل" in filtered
        assert "كتاب" in filtered

    def test_process_no_stops(self, tokenizer, sample_text):
        cleaned = ArabicPreprocessor().clean(sample_text)
        tokens = tokenizer.process(cleaned, remove_stops=False)
        assert len(tokens) > 0

    def test_process_remove_stops(self, tokenizer, sample_text):
        cleaned = ArabicPreprocessor().clean(sample_text)
        tokens = tokenizer.process(cleaned, remove_stops=True)
        assert len(tokens) > 0

    def test_empty_text(self, tokenizer):
        assert tokenizer.process("") == []
        assert tokenizer.process("   ") == []

    def test_default_stopwords_present(self, tokenizer):
        assert len(tokenizer.stopwords) > 0
        assert "في" in tokenizer.stopwords


# ------------------------------------------------------------------
# ArabicStemmer tests
# ------------------------------------------------------------------

class TestArabicStemmer:
    def test_stem_prefix(self, stemmer):
        assert stemmer.stem("المدرسة") == "مدرسة"
        assert stemmer.stem("بالكتاب") == "كتاب"

    def test_stem_suffix(self, stemmer):
        assert stemmer.stem("كتابهم") == "كتاب"

    def test_stem_short_word_unchanged(self, stemmer):
        assert stemmer.stem("من") == "من"

    def test_stem_tokens(self, stemmer):
        tokens = ["المدرسة", "الكتاب", "طلابهم"]
        stemmed = stemmer.stem_tokens(tokens)
        assert len(stemmed) == 3
        assert stemmed[0] == "مدرسة"
        assert stemmed[1] == "كتاب"


# ------------------------------------------------------------------
# PreprocessingPipeline tests
# ------------------------------------------------------------------

class TestPreprocessingPipeline:
    def test_run(self, sample_text):
        pipeline = PreprocessingPipeline()
        result = pipeline.run(sample_text)
        assert "original" in result
        assert "cleaned" in result
        assert "tokens" in result
        assert result["original"] == sample_text

    def test_run_with_stemming(self, sample_text):
        pipeline = PreprocessingPipeline()
        result = pipeline.run(sample_text, apply_stemming=True)
        assert "stemmed_tokens" in result
        assert len(result["stemmed_tokens"]) > 0

    def test_run_on_dataframe(self, sample_text):
        pipeline = PreprocessingPipeline()
        df = pd.DataFrame({"text": [sample_text, "نص عربي آخر"]})
        result = pipeline.run_on_dataframe(df, "text")
        assert "cleaned_text" in result.columns
        assert "tokens" in result.columns
        assert len(result) == 2


# ------------------------------------------------------------------
# Convenience function integration test
# ------------------------------------------------------------------

class TestPreprocessArabicText:
    def test_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.txt"
            output_path = Path(tmpdir) / "output.csv"

            input_path.write_text("هَذَا نَصّ عَرَبِي للتّجربة\nنَصّ آخر للتّجربة", encoding="utf-8")
            preprocess_arabic_text(str(input_path), str(output_path))

            assert output_path.exists()
            df = pd.read_csv(str(output_path))
            assert len(df) == 2
            assert "cleaned_text" in df.columns
            assert "tokens" in df.columns
