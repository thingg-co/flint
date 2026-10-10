# Flint roadmap

The future-work list from the whitepaper (Section 11), in the order it is being worked. Each
item is a hypothesis for the walk-forward backtest: it ships behind a switch where it can, lands
in one commit with tests, and is judged by a backtest before and after. Items that change the
feature set reset the model once, so they are batched into a single reset on a weekend.

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Market-relative target | shipped 0.5.14, awaiting the backtest verdict | `FLINT_TARGET=relative`: label = the residual against the modeled index names (SPY, QQQ, IWM, SMH, XLF), loadings refit each session, one hedge per index in the book; `backtest_target=both` compares the targets in one weekend run. Flip the live default once the relative run wins |
| 2 | Pricing-model feature batch (Table 4) | queued | implied-vol horizon band, term slope / event flag, VRP, dealer gamma; Yang-Zhang and Garman-Klass in place; HAR; beta, momentum, reversal; Amihud, quote OFI, Roll; ex-div flag; macro ETFs as names. One reset, with the calendar and valuation features below |
| 2b | Calendar and daily-lag features | queued | same-half-hour lag over 40 days, first-half-hour index return, overnight vs intraday split. Joins the reset of item 2 |
| 2c | Valuation features | queued | sector-relative EV/EBITDA, P/B, FCF yield, dividend yield; a DCF gap; quality composite. Joins the reset of item 2 |
| 2d | Other forecasters as opinions | queued | GARCH/HAR size, implied distribution, OU residual z and half-life, Kalman trend, ridge and GBM baselines. Joins the reset of item 2 |
| 3 | Policy pass (no reset) | queued | theta in the option cost edge, the gamma breakeven for straddles, square-root impact, the Amihud gate; exits with target, stop, time and daily loss limits |
| 4 | Ensemble of two 4XL networks | queued | aggregate members' distributions; fits the box next to the replay |
| 5 | Meta-labeling | queued | a small online classifier on whether acting on each forecast paid after costs; size by its probability |
| 6 | Calibration to decision loss | queued | size by P(move clears the spread) from the full quantile set |
| 7 | Policy sweep over stored forecasts | storage shipped 0.5.14; the sweep is next | every backtest run writes `state/backtest_forecasts_<target>.npz` (raw quantiles, direction probabilities, the label scored against, masks, loadings, timestamps); the sweep over the gates, tuned on the first half and confirmed on the second, proposes on the dashboard and never applies |
| 8 | SEC filings beyond 13F | queued | Form 4, 13D/G, 8-K item codes as features; free-text filings summarized by a local 7–8B model into a fixed schema; filings as a fourth universe source |

Baseline: the first full walk-forward backtest on the 362-name universe runs the weekend of
2026-10-10 (absolute target, 4XL, 30 h budget); its report is the "before" for everything above.
