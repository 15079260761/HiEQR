"""MultiTQ answer normalization."""

import re
import unicodedata


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).replace("_", " ").casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", "", value)).strip()

