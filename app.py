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
    value = re.sub(r"^NSE\s*:\s*", "", value)
    value = value.removesuffix(".NS")
    if not re.fullmatch(r"[A-Z0-9&_.-]+", value):
        return None
    return value.replace(".", "-") + ".NS"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_universe(name):
    response = requests.get(UNIVERSES[name], headers={"User-Agent": "TrendCraftScanner/1.0 (research app)"}, timeout=20)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    symbols = []
    for table in tables:
        table.columns = [str(column).strip() for column in table.columns]
        column = next((c for c in table.columns if c.lower() in {"symbol", "ticker", "security symbol"}), None)
        if column:
            symbols.extend(table[column].dropna().astype(str).tolist())
    result = sorted({symbol for value in symbols if (symbol := clean_symbol(value))})
    if not result:
        raise ValueError("Could not find constituent symbols on the source page.")
    return result


@st.cache_data(ttl=900, show_spinner=False)
def load_history(symbol, period, interval):
    frame = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(frame.columns):
        return pd.DataFrame()
    return frame[list(needed)].sort_index().dropna(subset=["Open", "High", "Low", "Close"])


def calculate_signals(frame, sma_length=65, pivot_bars=2, adx_length=14, adx_floor=18.0, slope_bars=5,
                      signal_confirm_bars=1, bb_length=20, bb_std=2.0, reversal_red_bars=2):
    """Calculate confirmed bullish breakouts and a lower-Bollinger-band reversal setup."""
    data = frame.copy()
    data["sma_low"] = data["Low"].rolling(sma_length, min_periods=sma_length).mean()
    data["sma_high"] = data["High"].rolling(sma_length, min_periods=sma_length).mean()

    up, down = data["High"].diff(), -data["Low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    previous_close = data["Close"].shift(1)
    tr = pd.concat([data["High"] - data["Low"], (data["High"] - previous_close).abs(), (data["Low"] - previous_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / adx_length, min_periods=adx_length, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / adx_length, min_periods=adx_length, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / adx_length, min_periods=adx_length, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    data["adx"] = dx.ewm(alpha=1 / adx_length, min_periods=adx_length, adjust=False).mean()

    # A pivot becomes usable only after its right-hand confirmation bars have closed.
    pivot = data["High"].shift(pivot_bars)
    is_pivot = pd.Series(True, index=data.index)
    for offset in range(1, pivot_bars + 1):
        is_pivot &= (pivot > data["High"].shift(pivot_bars + offset)) & (pivot >= data["High"].shift(pivot_bars - offset))
    data["pivot_high"] = pivot.where(is_pivot).ffill()
    data["above_channel"] = (data["Close"] > data["sma_low"]) & (data["Close"] > data["sma_high"])
    data["up_slope"] = data["sma_low"] > data["sma_low"].shift(slope_bars)
    data["trend_ok"] = data["above_channel"] & data["up_slope"] & (data["adx"] >= adx_floor)

    above_pivot = data["Close"] > data["pivot_high"]
    confirmed_break = above_pivot.rolling(signal_confirm_bars, min_periods=signal_confirm_bars).sum().eq(signal_confirm_bars)
    confirmed_trend = data["trend_ok"].rolling(signal_confirm_bars, min_periods=signal_confirm_bars).sum().eq(signal_confirm_bars)
    setup = confirmed_break & confirmed_trend
    data["signal"] = setup & ~setup.shift(1, fill_value=False)

    # Bollinger reversal interpretation: after N red closes below the lower band,
    # a bullish candle closes back inside the lower half of the Bollinger envelope.
    data["bb_mid"] = data["Close"].rolling(bb_length, min_periods=bb_length).mean()
    bb_sigma = data["Close"].rolling(bb_length, min_periods=bb_length).std(ddof=0)
    data["bb_lower"] = data["bb_mid"] - bb_std * bb_sigma
    is_red_outside = (data["Close"] < data["Open"]) & (data["Close"] < data["bb_lower"])
    prior_red_run = is_red_outside.shift(1).rolling(reversal_red_bars, min_periods=reversal_red_bars).sum().eq(reversal_red_bars)
    inside_lower_half = (data["Close"] >= data["bb_lower"]) & (data["Close"] <= data["bb_mid"])
    data["reversal_signal"] = (data["Close"] > data["Open"]) & inside_lower_half & prior_red_run
    data["red"] = (data["Close"] < data["sma_low"]) & (data["Close"] < data["sma_high"])
    return data


def scan_symbol(symbol, frame, entry_window, adx_floor, signal_confirm_bars, reversal_red_bars, sma_length):
    if len(frame) < max(100, sma_length + 30):
        return None
    data = calculate_signals(frame, sma_length=sma_length, adx_floor=adx_floor,
                             signal_confirm_bars=signal_confirm_bars, reversal_red_bars=reversal_red_bars)
    last = data.iloc[-1]
    if bool(last["reversal_signal"]):
        return {"Symbol": symbol.removesuffix(".NS"), "Signal date": pd.Timestamp(data.index[-1]).date(),
                "Signal high": round(float(last["High"]), 2), "Latest close": round(float(last["Close"]), 2),
                "Candles since signal": 0, "Signal type": "Forced Bollinger reversal", "ADX": round(float(last["adx"]), 1),
                "Status": "BUY — reversal candle closed inside lower Bollinger band"}

    positions = np.flatnonzero(data["signal"].fillna(False).to_numpy())
    for position in positions[::-1]:
        age = len(data) - 1 - position
        if age > entry_window:
            break
        signal_bar = data.iloc[position]
        if float(last["Close"]) > float(signal_bar["High"]) and bool(last["trend_ok"]) and not bool(last["red"]):
            return {"Symbol": symbol.removesuffix(".NS"), "Signal date": pd.Timestamp(data.index[position]).date(),
                    "Signal high": round(float(signal_bar["High"]), 2), "Latest close": round(float(last["Close"]), 2),
                    "Candles since signal": age, "Signal type": "BOS breakout", "ADX": round(float(last["adx"]), 1),
                    "Status": "BUY — above signal high"}
    return None


def run_backtest(symbol, frame, capital_per_trade, adx_floor, signal_confirm_bars, entry_window,
                 exit_mode, reversal_red_bars, sma_length):
    data = calculate_signals(frame, sma_length=sma_length, adx_floor=adx_floor,
                             signal_confirm_bars=signal_confirm_bars, reversal_red_bars=reversal_red_bars)
    trades, active = [], None

    def record_trade(position, exit_price, exit_stamp, status):
        entry = position["entry"]
        units = capital_per_trade / entry
        row = {"Symbol": symbol.removesuffix(".NS"), "Signal type": position["signal_type"],
               "Signal date": position["signal_date"], "Entry date": position["entry_date"],
               "Entry price": entry, "Exit date": pd.Timestamp(exit_stamp).date(), "Exit price": exit_price,
               "Candles held": position["entry_index"] if exit_stamp is None else None,
               "Investment (₹)": capital_per_trade, "P&L (₹)": (exit_price - entry) * units,
               "Return (%)": (exit_price / entry - 1) * 100, "Status": status}
        # Caller supplies holding count separately to avoid date arithmetic ambiguities.
        return row

    for i, (stamp, row) in enumerate(data.iterrows()):
        if active is not None and active["entry"] is not None and i > active["entry_index"]:
            entry = active["entry"]
            exit_price, exit_reason = None, None
            if exit_mode in {"5% target", "10% target", "20% target"}:
                target_pct = int(exit_mode.split("%")[0])
                target = entry * (1 + target_pct / 100)
                if float(row["High"]) >= target:
                    exit_price = max(float(row["Open"]), target)
                    exit_reason = f"CLOSED — {target_pct}% target"
            elif bool(row["red"]) and float(row["Close"]) > entry:
                exit_price, exit_reason = float(row["Close"]), "CLOSED — profitable channel exit"
            if exit_price is not None:
                units = capital_per_trade / entry
                trades.append({"Symbol": symbol.removesuffix(".NS"), "Signal type": active["signal_type"],
                               "Signal date": active["signal_date"], "Entry date": active["entry_date"],
                               "Entry price": entry, "Exit date": pd.Timestamp(stamp).date(), "Exit price": exit_price,
                               "Candles held": i - active["entry_index"], "Investment (₹)": capital_per_trade,
                               "P&L (₹)": (exit_price - entry) * units, "Return (%)": (exit_price / entry - 1) * 100,
                               "5% target hit": active["target_dates"][5] or "No",
                               "10% target hit": active["target_dates"][10] or "No",
                               "20% target hit": active["target_dates"][20] or "No", "Status": exit_reason})
                active = None

        if active is not None and active["entry"] is None and i - active["signal_index"] > entry_window:
            active = None

        if active is None:
            if bool(row["reversal_signal"]):
                active = {"signal_date": pd.Timestamp(stamp).date(), "signal_index": i,
                          "signal_type": "Forced Bollinger reversal", "entry": float(row["Close"]),
                          "entry_date": pd.Timestamp(stamp).date(), "entry_index": i,
                          "target_dates": {5: None, 10: None, 20: None}}
            elif bool(row["signal"]):
                active = {"signal_date": pd.Timestamp(stamp).date(), "signal_index": i,
                          "signal_type": "BOS breakout", "signal_high": float(row["High"]),
                          "entry": None, "entry_date": None, "entry_index": None,
                          "target_dates": {5: None, 10: None, 20: None}}
        elif active["entry"] is None and i > active["signal_index"] and float(row["High"]) > active["signal_high"]:
            fill = max(float(row["Open"]), active["signal_high"])
            active.update(entry=fill, entry_date=pd.Timestamp(stamp).date(), entry_index=i)

        if active is not None and active["entry"] is not None:
            for pct in (5, 10, 20):
                target = active["entry"] * (1 + pct / 100)
                if active["target_dates"][pct] is None and float(row["High"]) >= target:
                    active["target_dates"][pct] = pd.Timestamp(stamp).date()

        if i == len(data) - 1 and active is not None and active["entry"] is not None:
            close = float(row["Close"])
            units = capital_per_trade / active["entry"]
            trades.append({"Symbol": symbol.removesuffix(".NS"), "Signal type": active["signal_type"],
                           "Signal date": active["signal_date"], "Entry date": active["entry_date"],
                           "Entry price": active["entry"], "Exit date": "—", "Exit price": np.nan,
                           "Candles held": i - active["entry_index"], "Investment (₹)": capital_per_trade,
                           "P&L (₹)": (close - active["entry"]) * units,
                           "Return (%)": (close / active["entry"] - 1) * 100,
                           "5% target hit": active["target_dates"][5] or "No",
                           "10% target hit": active["target_dates"][10] or "No",
                           "20% target hit": active["target_dates"][20] or "No",
                           "Status": "OPEN — marked to market; no exit condition yet"})
    return trades


with st.sidebar:
    st.header("⚙️ Scanner settings")
    universe_name = st.selectbox("📚 Index universe", list(UNIVERSES))
    timeframe = st.radio("⏱️ Timeframe", ["1 Day", "1 Week"], horizontal=True)
    sma_length = st.number_input("SMA channel length", min_value=10, max_value=200, value=65)
    adx_floor = st.slider("Minimum ADX (sideways-market filter)", 10, 40, 18)
    signal_confirm_bars = st.slider("N candles to confirm BOS signal", 1, 10, 1)
    entry_window = st.slider("N candles allowed for breakout entry", 1, 30, 5)
    reversal_red_bars = st.slider("Red candles below lower Bollinger band for reversal entry", 1, 5, 2)
    st.caption("Forced reversal entry is separate from the green SMA-channel trend setup.")
    upload = st.file_uploader("Optional: upload a CSV to scan a custom list (column: Symbol)", type=["csv"])

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
                    history = load_history(symbol, "10y", interval)
                    candidate = scan_symbol(symbol, history, entry_window, adx_floor,
                                            signal_confirm_bars, reversal_red_bars, sma_length) if not history.empty else None
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
            st.warning("No stocks currently meet the scan conditions.")
        else:
            st.success(f"{len(result)} candidate(s) met the rules.")
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download scan as CSV", result.to_csv(index=False).encode("utf-8"), "trendcraft_scan.csv", "text/csv")
    else:
        st.write("Choose an index and timeframe, then select **Scan now**. Constituents are refreshed daily.")

with backtest_tab:
    st.subheader("Historical simulation")
    st.write("Target exits close a position at 5%, 10%, or 20% profit. Channel exit closes only on a red SMA channel when the close is above entry. Target-hit columns are also reported for each trade.")
    backtest_years = st.selectbox("Backtest history", [1, 2, 3, 5, 10], index=2, format_func=lambda y: f"{y} year" if y == 1 else f"{y} years")
    exit_mode = st.selectbox("Exit rule", ["Profitable SMA channel", "5% target", "10% target", "20% target"])
    capital_per_trade = st.number_input("Investment per trade (₹)", min_value=1000, max_value=10000000, value=100000, step=10000)
    if st.button("🧪 Run backtest", key="backtest"):
        try:
            symbols = fetch_universe(universe_name)
            trade_rows = []
            progress = st.progress(0)
            for i, symbol in enumerate(symbols):
                try:
                    history = load_history(symbol, f"{backtest_years}y", interval)
                    if not history.empty:
                        trade_rows.extend(run_backtest(symbol, history, capital_per_trade, adx_floor,
                                                       signal_confirm_bars, entry_window, exit_mode,
                                                       reversal_red_bars, sma_length))
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
                pd.DataFrame([{"Backtest years": st.session_state.get("backtest_years", backtest_years),
                               "Exit rule": exit_mode, "Trades": len(trades), "Total investment (₹)": investment,
                               "Net P&L (₹)": pnl, "Return on deployed capital (%)": (pnl / investment * 100) if investment else 0}]).to_excel(writer, sheet_name="Summary", index=False)
            st.download_button("⬇️ Download backtest (Excel)", output.getvalue(), "trendcraft_backtest.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with method_tab:
    st.subheader("Rules used by this app")
    st.markdown("""
- **BOS entry:** confirmed pivot breakout and trend filters; price must cross above the signal candle high within the selected N-candle window.
- **Forced Bollinger reversal entry:** regardless of SMA channel color, a bullish candle must close from the lower band through the middle Bollinger line after the selected number of consecutive red candles closed below the lower Bollinger Band. This reversal enters at that candle's close; it is a separate, higher-risk rule.
- **Profit target exits:** select a 5%, 10%, or 20% target, or use the profitable SMA-channel exit. Target exits trigger when the candle high reaches the target; a gap above target is filled at the open. Backtest rows also include dates when 5%, 10%, and 20% were reached.
- **Channel exit:** a red SMA channel is not enough on its own; its closing price must be above the entry price. Otherwise the position remains open. There is no stop loss.
- **Backtest:** choose 1, 2, 3, 5, or 10 years, daily/weekly bars, and fixed investment per trade. Open positions are marked to market but not treated as exits.

This is an approximate educational strategy, not financial advice or a guarantee of results. Data sources may be delayed or unavailable. Fees, taxes, slippage, liquidity, dividends, and portfolio-level capital limits are not modeled. Strict profitable-only channel exits can leave losing trades open indefinitely.
""")
