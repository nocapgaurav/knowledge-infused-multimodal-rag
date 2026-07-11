"""Parsing printed figure/table numbers, including roman numerals.

IEEE-style papers caption tables (and occasionally figures) with roman
numerals ("TABLE I", "TABLE II") while questions and in-text mentions use
arabic ("Table 1"). Both the caption-number lookup (`knowledge_builder`)
and in-text mention detection (`paragraph_strategy`) must agree on how a
printed number is read, so the shared parsing lives here. Observed live:
with arabic-only parsing, an IEEE paper's "TABLE I" produced no table
number at all -- no `Table 1` identity and no MENTIONS relationships.
"""

import re

NUMBER_TOKEN_PATTERN = r"(\d+|[IVXLCDM]+\b)"
"""The printed-number alternatives: arabic digits or an upper-case roman
numeral. Lower-case roman is deliberately excluded -- 'i'/'v' appear as
ordinary words far too often for `re.IGNORECASE` matching to be safe."""

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

_ROMAN_PATTERN = re.compile(r"^[IVXLCDM]+$")


def parse_printed_number(token: str) -> int | None:
    """Parse a printed figure/table number token into an integer.

    Args:
        token: The captured number token -- arabic digits ("12") or an
            upper-case roman numeral ("IV").

    Returns:
        The integer value, or `None` if the token is neither valid arabic
        nor a well-formed roman numeral.
    """
    if token.isdigit():
        return int(token)
    if not _ROMAN_PATTERN.match(token):
        return None
    total = 0
    previous = 0
    for char in reversed(token):
        value = _ROMAN_VALUES[char]
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total if total > 0 else None
