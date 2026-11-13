import io
import re
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="TrendCraft India Scanner", page_icon="📈", layout="wide")

UNIVERSES = {
    "Nifty 50": "https://en.wikipedia.org/wiki/NIFTY_50",
    "Nifty 200": "https://en.wikipedia.org/wiki/NIFTY_200",
    "Nifty Midcap 150": "https://en.wikipedia.org/wiki/NIFTY_Midcap_150",
    "Nifty Smallcap 250": "https://en.wikipedia.org/wiki/NIFTY_Smallcap_250",
}

st.title("📈 TrendCraft — Indian Equity Scanner")
st.caption("Daily / weekly trend-following scanner and historical backtester. For research and education—not investment advice.")


def clean_symbol(value):
    value = str(value).strip().upper()
    value = re.sub(r"^(NSE:|NSE\s*:\s*)", "", value)
    value = value.removesuffix(".NS")
    if not re.fullmatch(r"[A-Z0-9&_-]+", value):
        return None
    return value.replace(".", "-") + ".NS"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_universe(name):
    url = UNIVERSES[name]
    response = requests.get(url, headers={"User-Agent": "TrendCraftScanner/1.0 (research app)"}, timeout=20)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    candidates = []
    for table in tables:
        table.columns = [str(c).strip() for c in table.columns]
        symbol_col = next((c for c in table.columns if c.lower() in {"symbol", "ticker", "security symbol"}), None)
        if symbol_col:
            candidates.extend(table[symbol_col].dropna().astype(str).tolist())
    symbols = sorted({s for raw in candidates if (s := clean_symbol(raw))})
    if not symbols:
        raise ValueError("Could not find a constituent symbol column on the source page.")
    return symbols


def get_field(frame, field):
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    return frame[field]


@st.cache_data(ttl=900, show_spinner=False)
def load_history(symbol, period, interval):
    frame = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    return frame[list(required)].sort_index().dropna(subset=["High", "Low", "Close"])


def calculate_signals(frame, sma_length=65, pivot_bars=2, adx_length=14, adx_floor=18.0, slope_bars=5):
    """Approximate confirmed-pivot BOS plus SMA-channel and anti-sideways filters."""
    data = frame.copy()
    data["sma_low"] = data["Low"].rolling(sma_length, min_periods=sma_length).mean()
    data["sma_high"] = data["High"].rolling(sma_length, min_periods=sma_length).mean()
    # Wilder-style directional movement and ADX, smoothed with an EMA.
    up = data["High"].diff()
    down = -data["Low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    previous_close = data["Close"].shift(1)
    tr = pd.concat([data["High"] - data["Low"], (data["High"] - previous_close).abs(), (data["Low"] - previous_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / adx_length, min_periods=adx_length, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / adx_length, min_periods=adx_length, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / adx_length, min_periods=adx_length, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    data["adx"] = dx.ewm(alpha=1 / adx_length, min_periods=adx_length, adjust=False).mean()

    # A pivot is only made available after its right-side confirmation bars have closed.
    pivot = data["High"].shift(pivot_bars)
    is_pivot = pd.Series(True, index=data.index)
    for offset in range(1, pivot_bars + 1):
        is_pivot &= (pivot > data["High"].shift(pivot_bars + offset)) & (pivot >= data["High"].shift(pivot_bars - offset))
    data["pivot_high"] = pivot.where(is_pivot).ffill()
    data["bos"] = (data["Close"] > data["pivot_high"]) & (data["Close"].shift(1) <= data["pivot_high"].shift(1))
    data["above_channel"] = (data["Close"] > data["sma_high"]) & (data["Close"] > data["sma_low"])
    data["up_slope"] = data["sma_low"] > data["sma_low"].shift(slope_bars)
    data["trend_ok"] = data["above_channel"] & data["up_slope"] & (data["adx"] >= adx_floor)
    data["signal"] = data["bos"] & data["trend_ok"]
    data["red"] = (data["Close"] < data["sma_low"]) & (data["Close"] < data["sma_high"])
    return data


def scan_symbol(symbol, frame, lookback, adx_floor):
    if len(frame) < 100:
        return None
    data = calculate_signals(frame, adx_floor=adx_floor)
    recent_signals = data.index[data["signal"]].tolist()
    if not recent_signals:
        return None
    last = data.iloc[-1]
    for signal_date in reversed(recent_signals):
        if data.index.get_loc(data.index[-1]) - data.index.get_loc(signal_date) > lookback:
            break
        signal_bar = data.loc[signal_date]
        # Entry condition: latest completed price has cleared the signal candle high.
        if last["Close"] > signal_bar["High"] and bool(last["trend_ok"]) and not bool(last["red"]):
            return {
                "Symbol": symbol.removesuffix(".NS"), "Signal date": pd.Timestamp(signal_date).date(),
                "Signal high": round(float(signal_bar["High"]), 2), "Latest close": round(float(last["Close"]), 2),
                "ADX": round(float(last["adx"]), 1), "SMA low": round(float(last["sma_low"]), 2),
                "SMA high": round(float(last["sma_high"]), 2), "Status": "BUY — above signal high",
            }
    return None


def run_backtest(symbol, frame, capital_per_trade, adx_floor):
    data = calculate_signals(frame, adx_floor=adx_floor)
    trades = []
    active = None
    for i in range(len(data)):
        row = data.iloc[i]
        stamp = data.index[i]
        if active is not None and bool(row["red"]):
            exit_price = float(row["Close"])
            units = capital_per_trade / active["entry"]
            trades.append({"Symbol": symbol.removesuffix(".NS"), "Signal date": active["signal_date"], "Entry date": active["entry_date"], "Entry price": active["entry"], "Exit date": pd.Timestamp(stamp).date(), "Exit price": exit_price, "Candles held": i - active["entry_index"], "Investment (₹)": capital_per_trade, "P&L (₹)": (exit_price - active["entry"]) * units, "Return (%)": (exit_price / active["entry"] - 1) * 100, "Status": "CLOSED"})
            active = None
        if active is None and bool(row["signal"]):
            # Fill only on a later candle after its high trades through the signal candle high.
            signal_high = float(row["High"])
            active = {"signal_date": pd.Timestamp(stamp).date(), "signal_high": signal_high, "entry": None, "entry_date": None, "entry_index": None}
        elif active is not None and active["entry"] is None and i > 0:
            if float(row["High"]) >= active["signal_high"]:
                fill = max(float(row["Open"]), active["signal_high"])
                active.update(entry=fill, entry_date=pd.Timestamp(stamp).date(), entry_index=i)
        # Mark open positions at the last available close for transparent backtest reporting.
        if i == len(data) - 1 and active is not None and active["entry"] is not None:
            close = float(row["Close"])
            units = capital_per_trade / active["entry"]
            trades.append({"Symbol": symbol.removesuffix(".NS"), "Signal date": active["signal_date"], "Entry date": active["entry_date"], "Entry price": active["entry"], "Exit date": pd.Timestamp(stamp).date(), "Exit price": close, "Candles held": i - active["entry_index"], "Investment (₹)": capital_per_trade, "P&L (₹)": (close - active["entry"]) * units, "Return (%)": (close / active["entry"] - 1) * 100, "Status": "OPEN — marked to market"})
    return trades


with st.sidebar:
    st.header("⚙️ Scanner settings")
    universe_name = st.selectbox("📚 Index universe", list(UNIVERSES))
    timeframe = st.radio("⏱️ Timeframe", ["1 Day", "1 Week"], horizontal=True)
    sma_length = st.number_input("SMA channel length", min_value=10, max_value=200, value=65)
    adx_floor = st.slider("Minimum ADX (sideways-market filter)", 10, 40, 18)
    lookback = st.slider("Signal valid for candles", 1, 20, 5)
    st.caption("The SMA/channel and ADX/slope filters are applied to avoid weak or sideways trends.")
    upload = st.file_uploader("Optional: upload a CSV to scan a custom list (column: Symbol)", type=["csv"])

period = "10y" if timeframe == "1 Week" else "5y"
interval = "1wk" if timeframe == "1 Week" else "1d"
scan_tab, backtest_tab, method_tab = st.tabs(["🔎 Daily scan", "🧪 Backtest & Excel", "ℹ️ Strategy"])

with scan_tab:
    if st.button("🔄 Scan now", type="primary"):
        try:
            if upload is not None:
                uploaded = pd.read_csv(upload)
                symbol_col = next((c for c in uploaded.columns if str(c).strip().lower() in {"symbol", "ticker", "stock"}), None)
                if symbol_col is None:
                    st.error("CSV needs a Symbol (or Ticker/Stock) column.")
                    st.stop()
                symbols = sorted({s for raw in uploaded[symbol_col].dropna() if (s := clean_symbol(raw))})
            else:
                with st.spinner(f"Loading current constituents for {universe_name}…"):
                    symbols = fetch_universe(universe_name)
            st.info(f"Scanning {len(symbols)} symbols · {timeframe} · {date.today().isoformat()}")
            results, failures = [], 0
            progress = st.progress(0)
            for i, symbol in enumerate(symbols):
                try:
                    history = load_history(symbol, period, interval)
                    candidate = scan_symbol(symbol, history, lookback, adx_floor) if not history.empty else None
                    if candidate:
                        results.append(candidate)
                except Exception:
                    failures += 1
                progress.progress((i + 1) / len(symbols))
            st.session_state["scan_results"] = pd.DataFrame(results)
            st.session_state["scan_failures"] = failures
            st.session_state["scan_universe"] = universe_name
        except Exception as exc:
            st.error(f"Unable to load/scan this universe: {exc}")
    if "scan_results" in st.session_state:
        result = st.session_state["scan_results"]
        st.caption(f"Universe: {st.session_state.get('scan_universe', universe_name)} · Scan completed {date.today().isoformat()} · {st.session_state.get('scan_failures', 0)} symbols could not be fetched.")
        if result.empty:
            st.warning("No stocks currently meet all conditions. Try again after the next completed candle.")
        else:
            st.success(f"{len(result)} candidate(s) met the rules.")
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download scan as CSV", result.to_csv(index=False).encode("utf-8"), "trendcraft_scan.csv", "text/csv")
    else:
        st.write("Choose an index and timeframe, then select **Scan now**. Universe constituents are refreshed daily.")

with backtest_tab:
    st.subheader("Historical simulation")
    st.write("Runs the same confirmed-pivot, rising-SMA, ADX and channel-exit rules over the selected index. Each trade uses the chosen fixed rupee amount; positions may overlap.")
    capital_per_trade = st.number_input("Investment per trade (₹)", min_value=1000, max_value=10000000, value=100000, step=10000)
    if st.button("🧪 Run backtest", key="backtest"):
        try:
            symbols = fetch_universe(universe_name)
            trade_rows = []
            progress = st.progress(0)
            for i, symbol in enumerate(symbols):
                try:
                    history = load_history(symbol, period, interval)
                    if not history.empty:
                        trade_rows.extend(run_backtest(symbol, history, capital_per_trade, adx_floor))
                except Exception:
                    pass
                progress.progress((i + 1) / len(symbols))
            trades = pd.DataFrame(trade_rows)
            st.session_state["backtest_trades"] = trades
        except Exception as exc:
            st.error(f"Backtest could not run: {exc}")
    if "backtest_trades" in st.session_state:
        trades = st.session_state["backtest_trades"]
        if trades.empty:
            st.warning("No completed or open trades were found for this period.")
        else:
            investment = float(trades["Investment (₹)"].sum())
            pnl = float(trades["P&L (₹)"].sum())
            a, b, c, d = st.columns(4)
            a.metric("Trades", len(trades))
            b.metric("Total invested", f"₹{investment:,.0f}")
            c.metric("Net P&L", f"₹{pnl:,.0f}")
            d.metric("P&L / invested", f"{(pnl / investment * 100) if investment else 0:.2f}%")
            st.dataframe(trades.sort_values(["Entry date", "Symbol"], ascending=[False, True]), use_container_width=True, hide_index=True)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                trades.to_excel(writer, sheet_name="Trades", index=False)
                pd.DataFrame([{"Trades": len(trades), "Total investment (₹)": investment, "Net P&L (₹)": pnl, "Return on deployed capital (%)": (pnl / investment * 100) if investment else 0}]).to_excel(writer, sheet_name="Summary", index=False)
            st.download_button("⬇️ Download backtest (Excel)", output.getvalue(), "trendcraft_backtest.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with method_tab:
    st.subheader("Rules used by this app")
    st.markdown("""
- **Timeframes:** daily or weekly candles; weekly is built from Yahoo Finance weekly data.
- **Trend channel:** 65-period SMA of highs and lows. A bullish setup requires the close above both lines and the lower SMA rising.
- **Structure:** a confirmed swing high (two bars on each side) is used as a simplified bullish Break of Structure (BoS). A signal candle closes above that pivot.
- **Entry:** after the signal candle, price must trade above its high. The scanner shows a candidate only when the latest completed close is already above that level.
- **Avoiding sideways action:** ADX must meet the selected threshold (default 18), and the lower SMA must be rising. ADX can lag and does not guarantee a trend.
- **Exit:** exit at the close of the first candle below both SMA lines (the channel turns red). There is deliberately no stop-loss rule, as requested.
- **Backtest assumptions:** one fixed rupee amount per trade, overlapping positions allowed, signal candle excluded from entry, gap-up entries filled at the open. Open trades are marked to market at the final close. Fees, taxes, slippage, dividends and liquidity constraints are not modeled.

This is an approximate Python implementation of the supplied Pine Script idea, not a financial recommendation or a guarantee of results. Data and constituent pages are third-party services and can be delayed or unavailable. Validate signals and market data before making decisions.
""")
