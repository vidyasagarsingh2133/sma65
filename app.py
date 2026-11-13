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
    if not re.fullmatch(r"[A-Z0-9&_.-]+", value):
        return None
    return value.replace(".", "-") + ".NS"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_universe(name):
    response = requests.get(UNIVERSES[name], headers={"User-Agent": "TrendCraftScanner/1.0 (research app)"}, timeout=20)
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


def calculate_signals(frame, sma_length=65, pivot_bars=2, adx_length=14, adx_floor=18.0, slope_bars=5, signal_confirm_bars=1):
    """Approximate confirmed-pivot BOS plus SMA-channel, trend and N-bar confirmation filters."""
    data = frame.copy()
    data["sma_low"] = data["Low"].rolling(sma_length, min_periods=sma_length).mean()
    data["sma_high"] = data["High"].rolling(sma_length, min_periods=sma_length).mean()

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

    # The pivot is not available until its right-hand confirmation bars have closed.
    pivot = data["High"].shift(pivot_bars)
    is_pivot = pd.Series(True, index=data.index)
    for offset in range(1, pivot_bars + 1):
        is_pivot &= (pivot > data["High"].shift(pivot_bars + offset)) & (pivot >= data["High"].shift(pivot_bars - offset))
    data["pivot_high"] = pivot.where(is_pivot).ffill()
    data["above_channel"] = (data["Close"] > data["sma_high"]) & (data["Close"] > data["sma_low"])
    data["up_slope"] = data["sma_low"] > data["sma_low"].shift(slope_bars)
    data["trend_ok"] = data["above_channel"] & data["up_slope"] & (data["adx"] >= adx_floor)

    above_pivot = data["Close"] > data["pivot_high"]
    confirmed_break = above_pivot.rolling(signal_confirm_bars, min_periods=signal_confirm_bars).sum().eq(signal_confirm_bars)
    confirmed_trend = data["trend_ok"].rolling(signal_confirm_bars, min_periods=signal_confirm_bars).sum().eq(signal_confirm_bars)
    setup = confirmed_break & confirmed_trend
    data["signal"] = setup & ~setup.shift(1, fill_value=False)
    data["red"] = (data["Close"] < data["sma_low"]) & (data["Close"] < data["sma_high"])
    return data


def scan_symbol(symbol, frame, entry_window, adx_floor, signal_confirm_bars):
    if len(frame) < 100:
        return None
    data = calculate_signals(frame, adx_floor=adx_floor, signal_confirm_bars=signal_confirm_bars)
    signal_positions = np.flatnonzero(data["signal"].fillna(False).to_numpy())
    if not len(signal_positions):
        return None
    last = data.iloc[-1]
    for position in signal_positions[::-1]:
        age = len(data) - 1 - position
        if age > entry_window:
            break
        signal_bar = data.iloc[position]
        if float(last["Close"]) > float(signal_bar["High"]) and bool(last["trend_ok"]) and not bool(last["red"]):
            return {
                "Symbol": symbol.removesuffix(".NS"), "Signal date": pd.Timestamp(data.index[position]).date(),
                "Signal high": round(float(signal_bar["High"]), 2), "Latest close": round(float(last["Close"]), 2),
                "Candles since signal": age, "ADX": round(float(last["adx"]), 1),
                "SMA low": round(float(last["sma_low"]), 2), "SMA high": round(float(last["sma_high"]), 2),
                "Status": "BUY — above signal high",
            }
    return None


def run_backtest(symbol, frame, capital_per_trade, adx_floor, signal_confirm_bars, entry_window):
    data = calculate_signals(frame, adx_floor=adx_floor, signal_confirm_bars=signal_confirm_bars)
    trades = []
    active = None
    for i in range(len(data)):
        row = data.iloc[i]
        stamp = data.index[i]
        # Exit only when the channel is red AND the closing exit price exceeds entry.
        if active is not None and active["entry"] is not None and bool(row["red"]) and float(row["Close"]) > active["entry"]:
            exit_price = float(row["Close"])
            units = capital_per_trade / active["entry"]
            trades.append({"Symbol": symbol.removesuffix(".NS"), "Signal date": active["signal_date"], "Entry date": active["entry_date"], "Entry price": active["entry"], "Exit date": pd.Timestamp(stamp).date(), "Exit price": exit_price, "Candles held": i - active["entry_index"], "Investment (₹)": capital_per_trade, "P&L (₹)": (exit_price - active["entry"]) * units, "Return (%)": (exit_price / active["entry"] - 1) * 100, "Status": "CLOSED — profitable exit"})
            active = None

        # An unfilled setup expires after N candles; it cannot remain pending forever.
        if active is not None and active["entry"] is None and i - active["signal_index"] > entry_window:
            active = None

        if active is None and bool(row["signal"]):
            active = {"signal_date": pd.Timestamp(stamp).date(), "signal_index": i, "signal_high": float(row["High"]), "entry": None, "entry_date": None, "entry_index": None}
        elif active is not None and active["entry"] is None and i > active["signal_index"]:
            if float(row["High"]) > active["signal_high"]:
                fill = max(float(row["Open"]), active["signal_high"])
                active.update(entry=fill, entry_date=pd.Timestamp(stamp).date(), entry_index=i)

        # Open positions remain open if a red-channel exit is at/below entry; mark to market only.
        if i == len(data) - 1 and active is not None and active["entry"] is not None:
            close = float(row["Close"])
            units = capital_per_trade / active["entry"]
            trades.append({"Symbol": symbol.removesuffix(".NS"), "Signal date": active["signal_date"], "Entry date": active["entry_date"], "Entry price": active["entry"], "Exit date": pd.Timestamp(stamp).date(), "Exit price": close, "Candles held": i - active["entry_index"], "Investment (₹)": capital_per_trade, "P&L (₹)": (close - active["entry"]) * units, "Return (%)": (close / active["entry"] - 1) * 100, "Status": "OPEN — marked to market, no profitable exit yet"})
    return trades


with st.sidebar:
    st.header("⚙️ Scanner settings")
    universe_name = st.selectbox("📚 Index universe", list(UNIVERSES))
    timeframe = st.radio("⏱️ Timeframe", ["1 Day", "1 Week"], horizontal=True)
    sma_length = st.number_input("SMA channel length", min_value=10, max_value=200, value=65)
    adx_floor = st.slider("Minimum ADX (sideways-market filter)", 10, 40, 18)
    signal_confirm_bars = st.slider("N candles to confirm signal", 1, 10, 1, help="Require this many consecutive closes above the confirmed pivot while the trend filter is valid.")
    entry_window = st.slider("N candles allowed for entry", 1, 30, 5, help="The entry must cross above the signal candle high within this many candles after the signal.")
    st.caption("The SMA/channel and ADX/slope filters help avoid weak or sideways trends.")
    upload = st.file_uploader("Optional: upload a CSV to scan a custom list (column: Symbol)", type=["csv"])

scan_period = "10y" if timeframe == "1 Week" else "5y"
interval = "1wk" if timeframe == "1 Week" else "1d"
scan_tab, backtest_tab, method_tab = st.tabs(["🔎 Stock scan", "🧪 Backtest & Excel", "ℹ️ Strategy"])

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
                    history = load_history(symbol, scan_period, interval)
                    candidate = scan_symbol(symbol, history, entry_window, adx_floor, signal_confirm_bars) if not history.empty else None
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
    st.write("Select how many years of data to backtest. The strategy enters after a later candle crosses the signal high and only closes a position at a red-channel candle whose close is above entry.")
    backtest_years = st.selectbox("Backtest history", [1, 2, 3, 5, 10], index=2, format_func=lambda years: f"{years} year" if years == 1 else f"{years} years")
    capital_per_trade = st.number_input("Investment per trade (₹)", min_value=1000, max_value=10000000, value=100000, step=10000)
    if st.button("🧪 Run backtest", key="backtest"):
        try:
            symbols = fetch_universe(universe_name)
            trade_rows = []
            progress = st.progress(0)
            period = f"{backtest_years}y"
            for i, symbol in enumerate(symbols):
                try:
                    history = load_history(symbol, period, interval)
                    if not history.empty:
                        trade_rows.extend(run_backtest(symbol, history, capital_per_trade, adx_floor, signal_confirm_bars, entry_window))
                except Exception:
                    pass
                progress.progress((i + 1) / len(symbols))
            st.session_state["backtest_trades"] = pd.DataFrame(trade_rows)
            st.session_state["backtest_years"] = backtest_years
        except Exception as exc:
            st.error(f"Backtest could not run: {exc}")
    if "backtest_trades" in st.session_state:
        trades = st.session_state["backtest_trades"]
        st.caption(f"Period: {st.session_state.get('backtest_years', backtest_years)} years · {timeframe} candles · {universe_name}")
        if trades.empty:
            st.warning("No trades were found for this period.")
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
                pd.DataFrame([{"Backtest years": st.session_state.get("backtest_years", backtest_years), "Trades": len(trades), "Total investment (₹)": investment, "Net P&L (₹)": pnl, "Return on deployed capital (%)": (pnl / investment * 100) if investment else 0}]).to_excel(writer, sheet_name="Summary", index=False)
            st.download_button("⬇️ Download backtest (Excel)", output.getvalue(), "trendcraft_backtest.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with method_tab:
    st.subheader("Rules used by this app")
    st.markdown("""
- **Timeframes:** daily or weekly candles.
- **Signal:** confirmed swing-high breakout with an optional N consecutive-candle confirmation; trend filter requires price above the SMA channel, a rising lower SMA, and the selected ADX threshold.
- **Entry:** after the signal, price must trade above the signal candle high within the selected N-candle entry window. The scanner lists a candidate only if the latest completed close is already above that high.
- **Exit:** the channel must turn red (close below both SMA lines) AND the exit close must be strictly higher than entry. If price is red at a loss or breakeven, the backtest keeps the position open; there is no stop loss.
- **Backtest period:** choose 1, 2, 3, 5, or 10 years; daily and weekly data are supported.
- **Backtest assumptions:** one fixed rupee amount per trade; overlapping positions are allowed. Gap-up entries fill at the open. Positions without a profitable exit are reported as open and marked to market at the final close; this mark is not an exit.

This is an approximate implementation of the supplied Pine Script idea, not financial advice or a guarantee of results. Data and constituent pages are third-party services. Fees, taxes, slippage, dividends and portfolio-level capital limits are not modeled. A strict profitable-exit-only rule can leave losing positions open indefinitely.
""")
