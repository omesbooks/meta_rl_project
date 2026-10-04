"""Strict numeric input rules shared by dashboard actions and previews."""
import math
import re


def number(raw, label, minimum=None, maximum=None, integer=False):
    text = str(raw).strip()
    if not text:
        raise ValueError(f"{label}: enter a value")
    if "," in text and not re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?", text):
        raise ValueError(f"{label}: invalid thousands separators")
    try:
        value = float(text.replace(",", ""))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}: enter a valid number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label}: NaN and infinity are not allowed")
    if integer and not value.is_integer():
        raise ValueError(f"{label}: enter a whole number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label}: must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label}: must be <= {maximum}")
    return int(value) if integer else value


def fraction(raw, label="Percentage", allow_zero=True, allow_one=True):
    text = str(raw).strip()
    explicit_percent = text.endswith("%")
    value = number(text[:-1] if explicit_percent else text, label)
    if explicit_percent or value > 1:
        value /= 100
    if value < 0 or value > 1 or (not allow_zero and value == 0) or (not allow_one and value == 1):
        raise ValueError(f"{label}: outside the allowed range "
                         f"({'including' if allow_zero else 'excluding'} 0%, "
                         f"{'including' if allow_one else 'excluding'} 100%)")
    return value


def percent_units(raw, label, minimum=None):
    return number(str(raw).strip().removesuffix("%"), label, minimum=minimum) / 100
