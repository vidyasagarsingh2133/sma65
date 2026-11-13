# TrendCraft Indian Equity Scanner

A Streamlit research app that scans Nifty 50, Nifty 200, Nifty Midcap 150, and Nifty Smallcap 250 on daily or weekly bars. It adapts the provided Pine Script concept into a confirmed-pivot breakout strategy with an SMA channel, an ADX/rising-average filter for sideways markets, a signal-candle-high entry trigger, and an exit when price closes below both SMA lines. It includes a historical trade table and Excel export.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

The app downloads constituents from Wikipedia and historical prices from Yahoo Finance. Both are third-party sources and may rate-limit, change format, or be unavailable. Constituents are cached for one day; price history for 15 minutes. A CSV with a `Symbol` column can be uploaded to scan a custom list; symbols should be NSE tickers (for example `RELIANCE`) and `.NS` is added automatically.

## Notes and limitations

- Configure index, timeframe, SMA length, ADX threshold, signal-age limit, and fixed capital per trade in the sidebar/backtest panel.
- Scanner candidates are based on the latest completed bar, not a live tick. A candidate is shown when its close is above a recent bullish signal candle's high.
- The backtest enters only after a later candle crosses the signal high, fills gap-ups at the open, exits on the first close below both channel lines, and marks open trades to market at the end. Positions can overlap and each trade uses fixed capital.
- Commissions, taxes, slippage, liquidity, corporate actions beyond adjusted-price data, and portfolio-level capital limits are not modeled. No stop loss is included by design.
- This is educational software, not investment advice. Verify the strategy and data independently before trading.
