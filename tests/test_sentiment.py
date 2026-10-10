"""Headline tone: the graded word lists, the FinBERT scorer's cache and its fallback (no model
is loaded here), and the relative attention and own-headline tone of the news skim."""
import asyncio
from types import SimpleNamespace

from flint import sentiment
from flint.news import FinnhubNews, Headline, attention_of, name_tone, rescore, score_headline
from flint.sentiment import FinBertScorer, LexiconScorer, get_scorer, is_graded, lexicon_tone


def test_lexicon_tone_is_graded_and_hedged():
    assert lexicon_tone("Markets are open today") == 0.0
    assert lexicon_tone("Bitcoin surges past $60K") == 0.5                    # one word: halfway
    assert abs(lexicon_tone("Markets surge and rally hard") - 2 / 3) < 1e-3   # two agreeing words: closer to certain
    assert abs(lexicon_tone("Stocks plunge, crash and tumble") + 0.75) < 1e-3
    assert lexicon_tone("Markets surge but crash later") == 0.0               # mixed words cancel
    assert abs(lexicon_tone("Could markets surge?") - 1 / 3) < 1e-3           # a hedge takes a third off
    assert score_headline("Bitcoin surges past $60K")[2] == 0.5
    tones = [lexicon_tone(t) for t in ["surge", "surge rally", "plunge", "plunge crash", "may surge", "flat day", "surge rally jump"]]
    assert is_graded(tones) and not is_graded([1.0, -1.0, 0.0, 1.0, 0.0])


class _StubPipe:
    """Stands in for the model: positive if the title says 'beats', negative if 'misses'."""
    def __init__(self):
        self.calls = []

    def __call__(self, chunk):
        self.calls.append(list(chunk))
        import torch
        rows = []
        for t in chunk:
            rows.append([0.9, 0.05, 0.05] if "beats" in t else [0.05, 0.9, 0.05] if "misses" in t else [0.1, 0.1, 0.8])
        return SimpleNamespace(logits=torch.log(torch.tensor(rows)))


def _stubbed(scorer: FinBertScorer) -> _StubPipe:
    import torch
    pipe = _StubPipe()
    tok = lambda chunk, **kw: {"chunk": chunk}  # noqa: E731
    mdl = lambda chunk: pipe(chunk)  # noqa: E731
    scorer._pipe = (tok, mdl, {0: "positive", 1: "negative", 2: "neutral"}, torch)
    return pipe


def test_finbert_scorer_caches_per_title_and_scores_only_new_ones():
    fb = FinBertScorer(batch=2)
    pipe = _stubbed(fb)
    v = fb.score(["Acme beats", "Acme misses", "Acme reports", "Acme beats"])
    assert v[0] > 0.8 and v[1] < -0.8 and abs(v[2]) < 0.1 and v[3] == v[0]
    assert sum(len(c) for c in pipe.calls) == 3                               # the repeat was not scored twice
    fb.score(["Acme beats", "Acme reports"])
    assert sum(len(c) for c in pipe.calls) == 3                               # all cached: no inference
    fb.score(["Zed misses"])
    assert sum(len(c) for c in pipe.calls) == 4


def test_finbert_scorer_falls_back_to_the_lexicon_when_the_model_is_unavailable():
    fb = FinBertScorer(model="nonexistent/model-that-does-not-exist")
    fb._failed = True                                                         # as after a failed load
    assert fb.score(["Markets surge and rally hard"])[0] - 2 / 3 < 1e-3 and fb.score(["x"])[0] == 0.0
    fresh = FinBertScorer(model="nonexistent/model-that-does-not-exist")
    fresh._load = lambda: (setattr(fresh, "_failed", True), setattr(fresh, "name", "lexicon (finbert unavailable)"), False)[-1]
    assert fresh.score(["Bitcoin surges"]) == [0.5] and fresh.name.startswith("lexicon")


def test_get_scorer_reads_the_config():
    assert isinstance(get_scorer(SimpleNamespace(news_scorer="lexicon")), LexiconScorer)
    assert isinstance(get_scorer(SimpleNamespace()), LexiconScorer)             # no setting: the word lists, no download
    fb = get_scorer(SimpleNamespace(news_scorer="finbert", finbert_model="x/y", torch_threads=3))
    assert isinstance(fb, FinBertScorer) and fb.model_name == "x/y" and fb.threads == 3 and fb._pipe is None


def test_attention_is_relative_to_the_pass_leader_and_tone_follows_own_headlines():
    assert attention_of(10, 10) == 1.0 and attention_of(0, 10) == 0.0 and attention_of(5, 0) == 0.0
    assert abs(attention_of(1, 10) - 0.316) < 1e-3 and abs(attention_of(5, 10) - 0.707) < 1e-3
    h = lambda s: Headline("t", "s", "", ["A"], False, s, True, 0.0)  # noqa: E731
    assert name_tone([], 0.2) == 0.2                                           # no coverage: the market tone
    assert abs(name_tone([h(-0.8), h(-0.6)], 0.2) - (-1.4 + 0.2) / 3) < 1e-9    # own headlines dominate, one headline of prior
    assert abs(name_tone([h(0.5)] * 20, -0.9) - (10.0 - 0.9) / 21) < 1e-9


def test_rescore_runs_the_scorer_off_the_loop_and_the_hub_wires_it():
    hs = [Headline("Acme beats", "s", "", ["ACME"], False, 0.0, True, 0.0), Headline("Acme misses", "s", "", ["ACME"], False, 0.0, True, 0.0)]
    fb = FinBertScorer(); _stubbed(fb)
    asyncio.run(rescore(fb, hs))
    assert hs[0].sentiment > 0.8 and hs[1].sentiment < -0.8
    asyncio.run(rescore(None, hs))                                             # no scorer: untouched
    assert hs[0].sentiment > 0.8
    fn = FinnhubNews(["AAPL"], lambda: "", scorer=fb)
    assert fn.scorer is fb
