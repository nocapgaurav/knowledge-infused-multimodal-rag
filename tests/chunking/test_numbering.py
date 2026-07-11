"""Tests for printed-number parsing (arabic and roman)."""

from backend.chunking.strategies.numbering import parse_printed_number


def test_parses_arabic_numbers() -> None:
    assert parse_printed_number("1") == 1
    assert parse_printed_number("12") == 12


def test_parses_roman_numerals() -> None:
    assert parse_printed_number("I") == 1
    assert parse_printed_number("IV") == 4
    assert parse_printed_number("IX") == 9
    assert parse_printed_number("XII") == 12


def test_rejects_non_numbers() -> None:
    assert parse_printed_number("A") is None
    assert parse_printed_number("") is None
