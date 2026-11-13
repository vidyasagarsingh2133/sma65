# TrendCraft Indian Equity Scanner

A Streamlit research app that scans Nifty 50, Nifty 200, Nifty Midcap 150, and Nifty Smallcap 250 using daily or weekly candles. The scanner includes a confirmed-pivot breakout with configurable N-candle confirmation/entry window, plus an optional forced bullish reversal entry after red candles close below the lower Bollinger Band and a green candle closes back inside the lower half of the Bollinger envelope.

The backtest supports 1, 2, 3, 5, or 10 years, Excel export, and selectable exits: a 5%, 10%, or 20% profit target, or a profitable-only SMA-channel exit. Each trade also reports when the 5%, 10%, and 20% levels were reached.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

The app downloads constituents from Wikipedia and historical prices from Yahoo Finance. Both are third-party services. A CSV with a `Symbol` column can be uploaded to scan a custom NSE symbol list.

## Important assumptions

- The forced Bollinger reversal is implemented as a bullish candle closing at/above the lower band and at/below the Bollinger middle line after the configured number of consecutive red candles closed below the lower band. It enters at that candle's close, even if the SMA channel is not green; this is a higher-risk reversal setup.
- Profit target exits are triggered when a candle's high touches the selected target. A gap above the target is filled at the open. Target-hit dates for all three levels are shown in the trade table.
- The SMA-channel exit only occurs when the channel is red and the close is strictly above entry. If not, the position remains open; no stop loss is applied.
- Fees, taxes, slippage, liquidity, and portfolio-level capital limits are not modeled. This is educational software, not investment advice.
