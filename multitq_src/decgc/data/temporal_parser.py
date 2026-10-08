"""Strict temporal parsing; failures remain unknown rather than becoming zero."""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, datetime

_DAY = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
_MONTH = re.compile(r"^\d{4}-\d{1,2}$")
_YEAR = re.compile(r"^\d{4}$")
_DOTTED_DAY = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$")
_NATURAL_DAY = re.compile(r"^\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}$", re.IGNORECASE)
_NATURAL_DAY_MDY = re.compile(
    r"^[A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}$",
    re.IGNORECASE,
)
_NATURAL_MONTH = re.compile(r"^[A-Za-z]+,?\s+\d{4}$", re.IGNORECASE)


def parse_time(raw: str) -> dict[str, str | None]:
    text = str(raw).strip()
    clean_text = re.sub(
        r"(?<=\d)(?:st|nd|rd|th)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    try:
        if _DAY.fullmatch(text):
            year, month, day = map(int, text.split("-"))
            parsed = date(year, month, day)
            normalized = parsed.isoformat()
            return {"normalized": normalized, "start": normalized, "end": normalized, "granularity": "day"}
        if _DOTTED_DAY.fullmatch(text):
            day, month, year = map(int, text.split("."))
            parsed = date(year, month, day)
            normalized = parsed.isoformat()
            return {"normalized": normalized, "start": normalized, "end": normalized, "granularity": "day"}
        if _NATURAL_DAY.fullmatch(text):
            try:
                parsed_day = datetime.strptime(clean_text, "%d %B %Y")
            except ValueError:
                parsed_day = datetime.strptime(clean_text, "%d %b %Y")
            normalized = parsed_day.date().isoformat()
            return {"normalized": normalized, "start": normalized, "end": normalized, "granularity": "day"}
        if _NATURAL_DAY_MDY.fullmatch(text):
            without_comma = clean_text.replace(",", "")
            try:
                parsed_day = datetime.strptime(without_comma, "%B %d %Y")
            except ValueError:
                parsed_day = datetime.strptime(without_comma, "%b %d %Y")
            normalized = parsed_day.date().isoformat()
            return {"normalized": normalized, "start": normalized, "end": normalized, "granularity": "day"}
        if _NATURAL_MONTH.fullmatch(text):
            month_text = text.replace(",", "")
            try:
                parsed = datetime.strptime(month_text, "%B %Y")
            except ValueError:
                parsed = datetime.strptime(month_text, "%b %Y")
            normalized = f"{parsed.year:04d}-{parsed.month:02d}"
            end_day = monthrange(parsed.year, parsed.month)[1]
            return {
                "normalized": normalized,
                "start": f"{normalized}-01",
                "end": f"{normalized}-{end_day:02d}",
                "granularity": "month",
            }
        if _MONTH.fullmatch(text):
            year, month = map(int, text.split("-"))
            if not 1 <= month <= 12:
                raise ValueError
            end_day = monthrange(year, month)[1]
            return {
                "normalized": f"{year:04d}-{month:02d}",
                "start": f"{year:04d}-{month:02d}-01",
                "end": f"{year:04d}-{month:02d}-{end_day:02d}",
                "granularity": "month",
            }
        if _YEAR.fullmatch(text):
            year = int(text)
            return {"normalized": text, "start": f"{year:04d}-01-01", "end": f"{year:04d}-12-31", "granularity": "year"}
    except ValueError:
        return {"normalized": None, "start": None, "end": None, "granularity": "unknown"}
    return {"normalized": None, "start": None, "end": None, "granularity": "unknown"}


def temporal_compatible(timestamp: str, constraints: list[str], operator: str | None) -> bool:
    if not constraints:
        return True
    parsed = parse_time(timestamp)
    references = [parse_time(item) for item in constraints]
    valid = [item for item in references if item["normalized"] is not None]
    if parsed["normalized"] is None or not valid:
        return True
    parsed_start = parsed["start"] or parsed["normalized"]
    parsed_end = parsed["end"] or parsed_start
    for reference in valid:
        reference_start = reference["start"] or reference["normalized"]
        reference_end = reference["end"] or reference_start
        if operator in {"before", "before_after", "before_last"}:
            compatible = parsed_end < reference_start
        elif operator in {"after", "after_first"}:
            compatible = parsed_start > reference_end
        else:
            compatible = not (
                parsed_end < reference_start or parsed_start > reference_end
            )
        if compatible:
            return True
    return False
