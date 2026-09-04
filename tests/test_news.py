"""Tests for news.py pure helpers: score_headline, _clean_text, _http_candidates, parse_sources, Headline.to_json."""

import pytest

from flint.news import (
    _GENERIC,
    _POS,
    _NEG,
    _ASSET,
    DEFAULT_SOURCES,
    Headline,
    _clean_text,
    _http_candidates,
    parse_sources,
    score_headline,
)


class TestParseSources:
    """Tests for parse_sources function."""

    def test_parse_sources_valid_format(self):
        """Parse sources with name|url format."""
        result = parse_sources("Reuters|https://r.example/a, Bloomberg | https://b.example/b ,broken,, |https://x")
        # Note: The code skips empty items; "broken," results in name="" and url=""
        # which is skipped because url is falsy. The last ", |https://x" gives ("", "https://x")
        assert result == [
            ("Reuters", "https://r.example/a"),
            ("Bloomberg", "https://b.example/b"),
            ("", "https://x"),
        ]

    def test_parse_sources_empty_string(self):
        """Empty string returns empty list."""
        assert parse_sources("") == []

    def test_parse_sources_single_source(self):
        """Single source parses correctly."""
        result = parse_sources("Test|https://test.example")
        assert result == [("Test", "https://test.example")]

    def test_parse_sources_skips_empty_items(self):
        """Empty items between commas are skipped."""
        result = parse_sources("A|http://a,,B|http://b")
        assert result == [("A", "http://a"), ("B", "http://b")]

    def test_parse_sources_skips_without_url(self):
        """Items without | are skipped (no url part)."""
        result = parse_sources("A|http://a,B,C|http://c")
        assert result == [("A", "http://a"), ("C", "http://c")]


class TestCleanText:
    """Tests for _clean_text function."""

    def test_clean_text_basic_html(self):
        """Strip HTML tags and unescape entities."""
        assert _clean_text("<b>Fed &amp; markets</b>   rally<br/>hard") == "Fed & markets rally hard"

    def test_clean_text_preserves_text(self):
        """Text content is preserved."""
        assert _clean_text("<p>Hello World</p>") == "Hello World"

    def test_clean_text_multiple_spaces(self):
        """Multiple whitespace is collapsed to single space."""
        assert _clean_text("a   b    c") == "a b c"

    def test_clean_text_leading_trailing(self):
        """Leading and trailing whitespace is stripped."""
        assert _clean_text("  text  ") == "text"

    def test_clean_text_mixed_whitespace(self):
        """Mixed whitespace types are collapsed."""
        assert _clean_text("line1<br>line2") == "line1 line2"
        assert _clean_text("a\t\r\nb") == "a b"


class TestHttpCandidates:
    """Tests for _http_candidates function."""

    def test_candidates_from_anchors_and_headings(self):
        """Anchors with titles and headings produce candidates."""
        # Note: The <h3> heading is inside <body> not <a>, so it's captured by the <h[1-4]> regex
        html = """
        <html>
        <body>
            <a href="/story/1"><h2>This is a headline with thirty five chars</h2></a>
            <a href="/story/2"><h2>Hi</h2></a>
            <a href="/story/3"><h2>Long headline with exactly one hundred forty nine chars</h2></a>
            <a href="/story/4"><h2>This is a headline with thirty five chars</h2></a>
            <h3>Heading with thirty chars title</h3>
        </body>
        </html>
        """
        candidates = _http_candidates(html, "https://news.example/section/")
        # First anchor: 41 chars -> valid, URL = /story/1
        # Second anchor: "Hi" (2 chars) -> too short, skipped
        # Third anchor: 149 chars -> within 25-220, valid, URL = /story/3
        # Fourth anchor: duplicate of first -> skipped
        # h3 heading: "Heading with thirty chars title" (31 chars) -> valid, not in seen, URL = ""
        # Expected: 3 candidates
        assert len(candidates) == 3
        assert candidates[0] == ("This is a headline with thirty five chars", "https://news.example/story/1")
        assert candidates[1] == ("Long headline with exactly one hundred forty nine chars", "https://news.example/story/3")
        assert candidates[2] == ("Heading with thirty chars title", "")

    def test_candidates_caps_at_400(self):
        """Page with 450 distinct valid anchors yields exactly 400 candidates."""
        # Build HTML with 450 anchors
        anchors = []
        for i in range(450):
            title = f"Headline number {i:03d} with valid length"
            anchors.append(f'<a href="/story/{i}">{title}</a>')
        html = "<html><body>" + "".join(anchors) + "</body></html>"

        candidates = _http_candidates(html, "https://example.com/")

        assert len(candidates) == 400

    def test_candidates_skips_short_titles(self):
        """Titles under 25 characters are skipped."""
        # "Short title" is 11 chars, too short
        # "This is a longer title with enough words" is 39 chars, valid
        html = '<a href="/a">Short title</a><a href="/b">This is a longer title with enough words</a>'
        candidates = _http_candidates(html, "https://example.com/")
        assert len(candidates) == 1
        assert candidates[0][0] == "This is a longer title with enough words"

    def test_candidates_skips_titles_over_220_chars(self):
        """Titles over 220 characters are skipped."""
        # 221 chars exceeds the 220 limit
        long_title = "A" * 221
        short_title = "A" * 25  # exactly at minimum, valid
        html = f'<a href="/a">{long_title}</a><a href="/b">{short_title}</a>'
        candidates = _http_candidates(html, "https://example.com/")
        assert len(candidates) == 1
        assert candidates[0][0] == short_title

    def test_candidates_skips_titles_without_three_letters(self):
        """Titles without 3 consecutive letters are skipped."""
        # "123 456 789" has no 3 consecutive letters
        # "ABC is letters here today" has "ABC", "let", "ters" with 3+ consecutive letters and 31 chars
        html = '<a href="/a">123 456 789</a><a href="/b">ABC is letters here today</a>'
        candidates = _http_candidates(html, "https://example.com/")
        assert len(candidates) == 1
        assert candidates[0][0] == "ABC is letters here today"

    def test_candidates_deduplicates_titles(self):
        """Duplicate titles (after cleaning) are only included once."""
        # Both titles need to be >= 25 chars
        # "Same Title Here Today X" (23 chars) - still too short
        # "Same Title Here Today XY" (24 chars) - still too short
        # "Same Title Here Today XYZ" (25 chars) - valid
        html = '<a href="/a">Same Title Here Today XYZ</a><a href="/b">Same Title Here Today XYZ</a><a href="/c">A Different Title Here Today</a>'
        candidates = _http_candidates(html, "https://example.com/")
        assert len(candidates) == 2
        # First occurrence of "Same Title Here Today XYZ" is included, "A Different Title Here Today" is included
        titles = [c[0] for c in candidates]
        assert "Same Title Here Today XYZ" in titles
        assert "A Different Title Here Today" in titles


class TestScoreHeadline:
    """Tests for score_headline function."""

    def test_score_headline_positive_sentiment(self):
        """Two positive words, no negative = sentiment 1.0."""
        # "surge" and "rally" are both in POSITIVE
        result = score_headline("Markets surge and rally hard")
        assets, generic, sent = result
        assert sent == 1.0
        assert assets == []
        assert generic is False

    def test_score_headline_negative_sentiment(self):
        """Two negative words, no positive = sentiment -1.0."""
        # "plunge" and "crash" are both in NEGATIVE
        result = score_headline("Markets plunge and crash")
        assets, generic, sent = result
        assert sent == -1.0

    def test_score_headline_mixed_sentiment(self):
        """One positive, one negative = sentiment 0.0."""
        result = score_headline("Markets surge but could crash")
        assets, generic, sent = result
        assert sent == 0.0

    def test_score_headline_no_sentiment_words(self):
        """No sentiment words = sentiment 0.0."""
        result = score_headline("Markets are open today")
        assets, generic, sent = result
        assert sent == 0.0

    def test_score_headline_assets_btc(self):
        """BTC asset is detected."""
        result = score_headline("Bitcoin surges past $60K")
        assets, generic, sent = result
        assert "BTC" in assets
        assert sent == 1.0

    def test_score_headline_assets_eth(self):
        """ETH asset is detected."""
        result = score_headline("Ethereum rallies on news")
        assets, generic, sent = result
        assert "ETH" in assets

    def test_score_headline_assets_multiple(self):
        """Multiple assets can be detected."""
        result = score_headline("Bitcoin and Ethereum surge together")
        assets, generic, sent = result
        assert "BTC" in assets
        assert "ETH" in assets

    def test_score_headline_generic_true(self):
        """Generic crypto terms set generic=True."""
        result = score_headline("Crypto markets rally today")
        assets, generic, sent = result
        assert generic is True

    def test_score_headline_generic_false(self):
        """No generic terms = generic=False."""
        result = score_headline("Bitcoin surges")
        assets, generic, sent = result
        assert generic is False

    def test_score_headline_asset_and_generic(self):
        """Headline can have both assets and generic terms."""
        result = score_headline("Bitcoin is a digital asset that surges")
        assets, generic, sent = result
        assert "BTC" in assets
        assert generic is True


class TestHeadlineToJson:
    """Tests for Headline.to_json method."""

    def test_headline_to_json_round_trip(self):
        """to_json produces correct dict representation."""
        h = Headline(
            title="Test Title",
            source="TestSource",
            url="https://example.com/test",
            assets=["AAPL"],
            generic=False,
            sentiment=0.5,
            new=True,
            ts=1.0,
        )
        result = h.to_json()
        assert result == {
            "title": "Test Title",
            "source": "TestSource",
            "url": "https://example.com/test",
            "assets": ["AAPL"],
            "generic": False,
            "sentiment": 0.5,
            "new": True,
            "ts": 1.0,
        }

    def test_headline_to_json_empty_assets(self):
        """to_json handles empty assets list."""
        h = Headline(
            title="Title",
            source="Source",
            url="",
            assets=[],
            generic=False,
            sentiment=0.0,
            new=False,
            ts=0.0,
        )
        result = h.to_json()
        assert result["assets"] == []
