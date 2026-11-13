# TrendCraft Indian Equity Scanner

A Streamlit research app that scans Nifty 50, Nifty 200, Nifty Midcap 150, and Nifty Smallcap 250 on daily or weekly bars. The scanner uses a confirmed-pivot breakout, an SMA channel and ADX/rising-average filter, an optional N-consecutive-candle signal confirmation, and an N-candle entry window. Backtesting supports 1, 2, 3, 5, or 10 years, a trade table, and Excel export.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

The app downloads constituents from Wikipedia and historical prices from Yahoo Finance. Both are third-party sources and may rate-limit, change format, or be unavailable. Constituents are cached for one day; price history for 15 minutes. A CSV with a `Symbol` column can be uploaded to scan a custom list; symbols should be NSE tickers (for example `RELIANCE`) and `.NS` is added automatically.

## Strategy and limitations

- Configure index, daily/weekly timeframe, SMA length, ADX threshold, consecutive signal-confirmation candles, and the number of candles allowed for entry.
- Entry occurs when a candle after the signal trades above the signal candle high; a candidate is shown when the latest completed close is above that level.
- A position exits only when a candle closes below both SMA channel lines **and** that closing price is strictly above the entry price. If the channel turns red at a loss or breakeven, the position remains open; no stop-loss is used. It can therefore remain open indefinitely.
- The backtest lets users choose 1, 2, 3, 5, or 10 years. It uses a fixed amount per trade, permits overlapping positions, fills gap-ups at the open, and marks open trades to market at the final close without treating that mark as an exit.
- Commissions, taxes, slippage, liquidity, corporate actions beyond adjusted-price data, and portfolio-level capital limits are not modeled. This is educational software, not investment advice. Verify the strategy and data independently before trading.
