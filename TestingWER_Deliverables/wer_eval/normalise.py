# ─────────────────────────────────────────────────────────────────────────────
# wer_eval/normalise.py  —  Text normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

import re
import unicodedata


# Characters that are always stripped before scoring
_PUNCT = re.compile(r"[.,;:!?\"\'«»„""\(\)\[\]\{\}\-–—/\\@#$%^&*+=|<>~`]")

# Collapse runs of whitespace
_SPACES = re.compile(r"\s+")


def basic_normalise(text: str) -> str:
    """
    Lightweight normalisation suitable for Icelandic ASR evaluation:
      - Unicode NFKC
      - Lowercase
      - Strip punctuation
      - Collapse whitespace
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    return text


def normalise(text: str, pre_normalised: str | None = None) -> str:
    """
    Return the best available normalised form.
    If the dataset ships a pre-normalised column, use it as-is (after NFKC).
    Otherwise fall back to basic_normalise().
    """
    if pre_normalised is not None and pre_normalised.strip():
        return unicodedata.normalize("NFKC", pre_normalised).strip()
    return basic_normalise(text)
