"""Transparent Mandarin transcript normalization.

Traditional-to-simplified conversion is optional at import time so that the
core protocol and tests remain usable before the ASR environment is installed.
Formal normalized CER requires an OpenCC-compatible converter.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable

TextConverter = Callable[[str], str]


def load_opencc_t2s_converter() -> TextConverter:
    """Load the formal traditional-to-simplified converter.

    The dependency is intentionally imported lazily. This keeps the existing
    DSP environment lightweight and produces an actionable error before formal
    scoring when the robust-ASR environment has not been installed.
    """

    try:
        from opencc import OpenCC
    except ImportError as exc:  # pragma: no cover - dependency is optional now
        raise RuntimeError(
            "formal Mandarin normalization requires "
            "opencc-python-reimplemented"
        ) from exc
    converter = OpenCC("t2s")
    return converter.convert


class ChineseTextNormalizer:
    """Normalize Chinese ASR text with an explicit, symmetric policy."""

    def __init__(
        self,
        *,
        traditional_to_simplified: bool = True,
        converter: TextConverter | None = None,
    ) -> None:
        if converter is not None and not traditional_to_simplified:
            raise ValueError(
                "converter cannot be supplied when conversion is disabled"
            )
        self.traditional_to_simplified = traditional_to_simplified
        self._converter = converter

    def _get_converter(self) -> TextConverter:
        if self._converter is None:
            self._converter = load_opencc_t2s_converter()
        return self._converter

    def normalize(self, text: str) -> str:
        """Return NFKC, simplified, case-folded letters and numbers only."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized = unicodedata.normalize("NFKC", text)
        if self.traditional_to_simplified:
            normalized = self._get_converter()(normalized)
        normalized = normalized.casefold()
        # Unicode L* keeps Chinese and alphabetic scripts; N* keeps all numeric
        # forms. Whitespace, punctuation, symbols, marks and controls are
        # deliberately removed.
        return "".join(
            character
            for character in normalized
            if unicodedata.category(character)[0] in {"L", "N"}
        )

    def normalize_pair(self, reference: str, hypothesis: str) -> tuple[str, str]:
        """Normalize both sides through the same object and policy."""

        return self.normalize(reference), self.normalize(hypothesis)

