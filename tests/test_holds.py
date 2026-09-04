"""Tests for summarize_holds helper function."""
import pytest
import flint.engine as eng


class TestSummarizeHolds:
    """Tests for summarize_holds()."""

    def test_collapses_same_reason_with_different_parenthesized_suffixes(self):
        """Two symbols sharing a reason with different parenthesised suffixes collapse to one row with n=2."""
        suggestions = {
            "AAA": {"reasons": ["no track record (3/8 calls)"], "muted": False},
            "BBB": {"reasons": ["no track record (5/8 calls)"], "muted": False},
        }
        result = eng.summarize_holds(suggestions)
        assert len(result) == 1
        assert result[0]["reason"] == "no track record"
        assert result[0]["n"] == 2

    def test_rows_sorted_by_n_descending_empty_reasons_adds_nothing(self):
        """Rows sorted by n descending, empty reasons list adds nothing."""
        suggestions = {
            "AAA": {"reasons": ["low band", "low band", "low band"], "muted": False},
            "BBB": {"reasons": ["muted"], "muted": False},
            "CCC": {"reasons": [], "muted": False},
            "DDD": {"reasons": ["low band"], "muted": False},
        }
        result = eng.summarize_holds(suggestions)
        assert len(result) == 2
        # low band has 4 occurrences, muted has 1
        assert result[0]["reason"] == "low band"
        assert result[0]["n"] == 4
        assert result[1]["reason"] == "muted"
        assert result[1]["n"] == 1

    def test_muted_symbol_counts_under_muted_label(self):
        """Muted symbol (dict with muted: True and reasons ["muted"]) counts under muted."""
        suggestions = {
            "AAA": {"reasons": ["muted"], "muted": True},
            "BBB": {"reasons": ["muted"], "muted": True},
            "CCC": {"reasons": ["low band"], "muted": False},
        }
        result = eng.summarize_holds(suggestions)
        assert len(result) == 2
        # Two muted symbols count under "muted", one non-muted symbol has "low band"
        muted_row = next(r for r in result if r["reason"] == "muted")
        assert muted_row["n"] == 2
        lowband_row = next(r for r in result if r["reason"] == "low band")
        assert lowband_row["n"] == 1
