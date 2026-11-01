# Precedent

Market-structure recognition and historical-analog search for XAU/USD (gold).

Precedent reads price data, labels its structure the way a discretionary trader
reads a chart (swings, break of structure, change of character, supply and
demand, liquidity sweeps, trend vs range), and then answers one question for any
moment: **"when has price looked structurally like this before, and what
happened next?"** It retrieves the most similar past windows and summarises their
outcomes, with a strict no-look-ahead rule so the research is honest.

It is a research and decision-support tool for a human reviewer. It does not
predict price, generate signals, or promise profit.

Author: A. Amini

## What it does

- **Data pipeline** (`data_pipeline.py`): loads OHLCV, checks integrity (duplicate
  timestamps, missing candles, impossible bars) and resamples across timeframes
  (M1 to D1). Integrity issues are reported, not silently patched.
- **Structure recognition** (`structure.py`): fractal swings, BOS and CHOCH
  labelling with a running trend read, supply/demand zones, and liquidity-sweep
  detection. Every feature at bar *i* uses only bars up to *i*.
- **Analog search** (`features.py`, `similarity.py`): encodes a window of price
  action into a price- and volatility-invariant vector, then finds the k nearest
  historical windows. Each query may only retrieve past windows whose outcome
  already completed, so there is no look-ahead. It reports the analogs' mean
  forward return, agreement (win rate) and dispersion (how reliable the
  precedent is).
- **Walk-forward evaluation** (`backtest.py`): measures, out-of-sample, whether
  the analog outcomes carry any signal about the forward return, including on the
  subset where analogs agreed most. It reports what is there, honestly, including
  when the edge is weak.

Run it:

```bash
pip install -r requirements.txt
python scripts/generate_sample.py      # writes data/sample_xauusd_h1.csv (synthetic)
python demo.py                         # full pipeline + report
python -m pytest tests/ -q             # 6 tests, incl. causality / no-look-ahead
```

Point it at your own data with `python demo.py --csv your_xauusd_h1.csv` (columns:
`time, open, high, low, close, volume`).

## The data here is synthetic

The bundled sample is **generated**, not real market data, so the repo is
self-contained and carries no data-licensing issues. On a near-random synthetic
series the analog edge is near zero by construction, and the demo says so. That
is the correct baseline: it shows the machinery is honest, not that gold is
predictable. Real signal, where it exists, comes from the reserved parts below
run on real data.

## What is reserved

This repository is a capability demonstration under an open-core model. The parts
that carry the actual retrieval edge are deliberately not published here; they
are delivered under a paid engagement. They are present as clear interfaces
(`learning.py`) so the architecture is complete and testable, and the reserved
methods raise `ReservedComponent` rather than shipping a fake implementation.

Reserved:

1. **Learned similarity metric** — the public demo ranks analogs by cosine
   distance on a baseline feature set. The fitted metric that pulls together
   windows with similar *outcomes* (not just similar shapes) is reserved.
2. **Production feature / embedding set** — the baseline encoder uses normalised
   returns, path and range. The tuned feature set (microstructure terms,
   multi-timeframe structural alignment, volume-shape features) that materially
   improves retrieval is reserved.
3. **Feedback-driven learning loop** — learning from human-labelled examples,
   the labelling schema, model-versioning and the out-of-sample promotion gate
   (`promote_if_better`) are reserved.
4. **Calibrated detection thresholds and labelled dataset** — the demo ships
   sensible defaults; the calibrated configuration and the labelled structure set
   used to validate precision/recall are reserved.

What ships here runs, is tested, and is enough to evaluate the engineering. The
reserved parts are what turn it into a system tuned to a specific methodology.

## Layout

```
src/precedent/
  data_pipeline.py   load, validate, resample (MTF)
  structure.py       swings, BOS, CHOCH, zones, liquidity sweeps, trend
  features.py        causal window embedding (baseline)
  similarity.py      no-look-ahead analog search + outcome stats
  learning.py        reserved interfaces (learned metric, quality model, versioning)
  backtest.py        walk-forward, out-of-sample evaluation
scripts/generate_sample.py
demo.py
tests/test_precedent.py
```

## Scope and honesty

Precedent recognises and retrieves; it does not forecast. Whether similar
structures lead to similar outcomes on real markets is a weak, non-stationary
relationship, and this tool is built to measure that honestly rather than to
assume it. Any use for live decisions is the user's own, at the user's own risk.

## License

See `LICENSE.txt`. Private portfolio sample; no reuse without written consent.
