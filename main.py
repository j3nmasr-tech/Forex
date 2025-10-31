#!/usr/bin/env python3
# Forex+Metals+Indices Top40 signal bot using Yahoo Finance (yfinance)
# Single-message alerts to Telegram. Designed for high-quality signals (>=75% conf)
# Requires: yfinance, pandas, numpy, requests, pytz

import time, requests, csv, os
from datetime import datetime, timezone
import pandas as pd, numpy as np
import yfinance as yf
import pytz

# ========== CONFIG ==========
BOT_TOKEN = "8336065665:AAGRSvFfTmxbTlXRikA99Wg4qmxwTTzrD1I"
CHAT_ID   = "7087925615"

CAPITAL = 100.0                # account USD (adjust)
BASE_RISK = 0.02               # base risk (2%)
LEVERAGE = 30                  # informational (used in margin estimate)
CHECK_INTERVAL = 300           # seconds between cycles
TIMEFRAMES = ["15m","30m","1h","4h"]
CONF_MIN_TFS = 3               # require >=3 TF confirmations (75%)
MIN_QUOTE_VOLUME = 0.0         # not used for Yahoo FX tickers (kept for compatibility)

VOLATILITY_THRESHOLD_PCT = 1.8 # percent move in short window to pause
VOLATILITY_PAUSE = 1800        # seconds pause after volatility spike
TRADE_SESSION_UTC = (8, 21)    # active trading hours UTC (8:00 - 21:00)

LOG_CSV = "/tmp/top40_yf_signals.csv"

# ========== SYMBOLS (Top 40 mix of FX majors/crosses + metals + indices) ==========
# Yahoo tickers: FX pairs use e.g. 'EURUSD=X', metals 'XAUUSD=X', indices '^GSPC' etc.
SYMBOLS = [
    # Majors & crosses (20)
    "EURUSD=X","GBPUSD=X","USDJPY=X","USDCHF=X","USDCAD=X","AUDUSD=X","NZDUSD=X",
    "EURGBP=X","EURJPY=X","GBPJPY=X","EURCHF=X","AUDJPY=X","CADJPY=X","GBPCHF=X",
    "AUDCAD=X","AUDNZD=X","EURNZD=X","GBPAUD=X","EURCAD=X","NZDJPY=X",
    # Other crosses & exotics (6)
    "TRYUSD=X","ZARUSD=X","MXNUSD=X","SGDUSD=X","HKDUSD=X","CNH=X",
    # Metals & commodities (6)
    "XAUUSD=X","XAGUSD=X","GC=F","SI=F","CL=F","NG=F",
    # Indices / futures (8)
    "^GSPC","^IXIC","^DJI","NQ=F","ES=F","YM=F","^FTSE","^GDAXI"
]

# Ensure exactly 40 items (we provided 40). Trim/pad if needed.
SYMBOLS = SYMBOLS[:40]

# ========== HELPERS ==========
def send_message(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured")
        return False
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        return True
    except Exception as e:
        print("Telegram send error:", e)
        return False

def init_csv():
    try:
        if not os.path.exists(LOG_CSV):
            with open(LOG_CSV, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["utc","symbol","side","entry","tp1","sl","conf_pct","units","risk_pct","status","breakdown"])
    except Exception as e:
        print("init csv err", e)

def log_signal(row):
    try:
        with open(LOG_CSV, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(row)
    except Exception as e:
        print("log signal err", e)

# ========== YFINANCE DATA FETCH ==========
# map our TF names to yfinance intervals
YF_INTERVAL = {"15m":"15m","30m":"30m","1h":"60m","4h":"240m"}

def get_klines_yf(symbol, tf, limit=300):
    """Download historical candlesticks via yfinance for a given symbol/timeframe."""
    interval = YF_INTERVAL.get(tf, "60m")
    period_map = {
        "15m":"7d", "30m":"7d", "1h":"30d", "4h":"90d"
    }
    period = period_map.get(tf, "30d")
    try:
        df = yf.download(tickers=symbol, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        # ensure columns: Datetime, Open, High, Low, Close, Volume
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume","Datetime":"t"})
        # ensure numeric
        df = df[["open","high","low","close","volume"]].astype(float)
        return df
    except Exception as e:
        # network or symbol not available at requested interval
        # print("yf dl err", symbol, tf, e)
        return None

def get_last_price_yf(sym):
    try:
        ticker = yf.Ticker(sym)
        price = ticker.history(period="1d", interval="1m")["Close"].dropna()
        if price.empty:
            # fallback to fast info
            info = ticker.fast_info
            return float(info.get("lastPrice") or info.get("last_price") or 0.0)
        return float(price.iloc[-1])
    except Exception:
        return None

# ========== INDICATORS ==========
def detect_crt(df):
    if df is None or len(df) < 8: return (False, False)
    o,c,h,l,v = df["open"].iloc[-1], df["close"].iloc[-1], df["high"].iloc[-1], df["low"].iloc[-1], df["volume"].iloc[-1]
    avg_body = df.apply(lambda x: abs(x["open"]-x["close"]), axis=1).rolling(8).mean().iloc[-1]
    avg_vol = df["volume"].rolling(8).mean().iloc[-1]
    if pd.isna(avg_body) or pd.isna(avg_vol): return (False, False)
    body = abs(c-o); wick_up = h - max(o,c); wick_down = min(o,c) - l
    bull = (body < avg_body*0.7) and (wick_down > avg_body*0.6) and (v < avg_vol*1.2) and (c > o)
    bear = (body < avg_body*0.7) and (wick_up > avg_body*0.6) and (v < avg_vol*1.2) and (c < o)
    return bull, bear

def detect_turtle(df, look=20):
    if df is None or len(df) < look+2: return (False, False)
    ph = df["high"].iloc[-look-1:-1].max(); pl = df["low"].iloc[-look-1:-1].min()
    last = df.iloc[-1]
    bull = (last["low"] < pl) and (last["close"] > pl*1.002)
    bear = (last["high"] > ph) and (last["close"] < ph*0.998)
    return bull, bear

def smc_bias(df):
    if df is None or len(df) < 60: return "neutral"
    e20 = df["close"].ewm(span=20).mean().iloc[-1]; e50 = df["close"].ewm(span=50).mean().iloc[-1]
    return "bull" if e20 > e50 else "bear"

def volume_ok(df):
    if df is None or "volume" not in df.columns: return True
    ma = df["volume"].rolling(20).mean().iloc[-1]
    if pd.isna(ma): return True
    return df["volume"].iloc[-1] > ma * 1.2

# ========== ATR & POSITION SIZE ==========
def get_atr_from_df(df, period=14):
    if df is None or len(df) < period+1: return None
    h = df["high"].values; l = df["low"].values; c = df["close"].values
    trs = []
    for i in range(1, len(df)):
        trs.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    return float(np.mean(trs)) if trs else None

def trade_params_from_df(symbol, entry, side, df_h1):
    atr = get_atr_from_df(df_h1, period=14)
    if atr is None: return None
    atr = max(min(atr, abs(entry)*0.2), abs(entry)*0.0001)
    if side == "BUY":
        sl = round(entry - atr*1.7, 8)
        tp1 = round(entry + atr*1.8, 8)
        tp2 = round(entry + atr*2.8, 8)
        tp3 = round(entry + atr*3.8, 8)
    else:
        sl = round(entry + atr*1.7, 8)
        tp1 = round(entry - atr*1.8, 8)
        tp2 = round(entry - atr*2.8, 8)
        tp3 = round(entry - atr*3.8, 8)
    return sl, tp1, tp2, tp3, atr

def pos_size_units(entry, sl, conf_pct):
    # adaptive risk by confidence
    if conf_pct >= 100: risk_pct = 0.04
    elif conf_pct >= 90: risk_pct = 0.03
    elif conf_pct >= 75: risk_pct = 0.02
    else: risk_pct = BASE_RISK
    risk_usd = CAPITAL * risk_pct
    dist = abs(entry - sl)
    if dist <= 0: return 0,0,0,risk_pct
    units = risk_usd / dist
    exposure = units * entry
    margin_required = exposure / LEVERAGE
    return round(units,8), round(margin_required,6), round(exposure,6), risk_pct

# ========== SESSION & VOL PAUSE ==========
def in_trade_session():
    utcnow = datetime.utcnow().hour
    start, end = TRADE_SESSION_UTC
    # simple inclusive test (handles start < end)
    return (utcnow >= start) and (utcnow < end)

def short_volatility_spike_check():
    # check EURUSD 5m small snapshot for sudden move
    try:
        df = get_klines_yf("EURUSD=X", "15m", limit=10)
        if df is None or len(df) < 3: return False
        # use quick last 3 candle pct move
        c0 = df["close"].iloc[-3]; c2 = df["close"].iloc[-1]
        pct = abs((c2 - c0) / c0) * 100
        return pct >= VOLATILITY_THRESHOLD_PCT
    except Exception:
        return False

# ========== ANALYSIS & SIGNAL CREATION ==========
last_trade_time = {}
open_trades = []
signals_sent = signals_hit = signals_fail = 0
last_summary = last_heartbeat = time.time()
vol_pause_until = 0

def analyze_symbol(symbol):
    global signals_sent, vol_pause_until
    # only during session
    if not in_trade_session():
        return False
    if time.time() < vol_pause_until:
        return False
    # per-TF confirmations
    confs = 0
    chosen_dir = None
    chosen_entry = None
    chosen_tf = None
    confirming_tfs = []
    breakdown = []
    # we'll need H1 df for ATR later
    df_h1 = None
    for tf in TIMEFRAMES:
        df = get_klines_yf(symbol, tf)
        if df is None or len(df) < 60:
            breakdown.append(f"{tf}: NO")
            continue
        if tf == "1h":
            df_h1 = df.copy()
        crt_b, crt_s = detect_crt(df)
        ts_b, ts_s = detect_turtle(df)
        bias = smc_bias(df)
        volok = volume_ok(df)
        bull_score = ( (1 if crt_b else 0)*0.20 + (1 if ts_b else 0)*0.25 + (1 if volok else 0)*0.15 + (1 if bias=="bull" else 0)*0.40 )*100
        bear_score = ( (1 if crt_s else 0)*0.20 + (1 if ts_s else 0)*0.25 + (1 if volok else 0)*0.15 + (1 if bias=="bear" else 0)*0.40 )*100
        breakdown.append(f"{tf} b{int(bull_score)}/r{int(bear_score)} bias:{bias}")
        if bull_score >= 50:
            confs += 1; chosen_dir="BUY"; chosen_entry = float(df["close"].iloc[-1]); chosen_tf = tf; confirming_tfs.append(tf)
        elif bear_score >= 50:
            confs += 1; chosen_dir="SELL"; chosen_entry = float(df["close"].iloc[-1]); chosen_tf = tf; confirming_tfs.append(tf)

    if confs >= CONF_MIN_TFS and chosen_dir and chosen_entry:
        conf_pct = int(confs*25)
        # simple trend filter: require 4h and daily bias to match (daily using 1d data)
        df_4h = get_klines_yf(symbol, "4h")
        df_d = get_klines_yf(symbol, "1h")  # yfinance daily intervals sometimes behave differently; use 1h as fallback
        if df_4h is None:
            return False
        bias4 = smc_bias(df_4h)
        biasd = smc_bias(df_d) if df_d is not None else bias4
        if bias4 != biasd:
            # skip if 4h and daily/1h don't align (strict trend check)
            return False

        # cooldown for symbol
        if time.time() - last_trade_time.get(symbol,0) < 1800:
            return False
        last_trade_time[symbol] = time.time()

        # compute SL/TP using H1 for ATR
        if df_h1 is None:
            df_h1 = get_klines_yf(symbol, "1h")
        params = trade_params_from_df(symbol, chosen_entry, chosen_dir, df_h1)
        if not params:
            return False
        sl,tp1,tp2,tp3, atr = params
        units, margin, exposure, risk_pct = pos_size_units(chosen_entry, sl, conf_pct)
        # build compact breakdown message
        confirmed_str = ", ".join(confirming_tfs)
        per_tf_text = " | ".join(breakdown)
        header = (
            f"✅ {chosen_dir} {symbol} ({conf_pct}% CONF)\n"
            f"🕒 Entry TF: {chosen_tf} | Confirmed on: {confirmed_str}\n"
            f"💵 Entry: {chosen_entry:.6f}\n"
            f"🎯 TP1:{tp1:.6f} TP2:{tp2:.6f} TP3:{tp3:.6f}\n"
            f"🛑 SL:{sl:.6f}\n"
            f"💰 Units:{units} | Margin≈${margin} | Exposure≈${exposure}\n"
            f"⚠ Risk used: {risk_pct*100:.2f}% | ATR:{atr:.6f}"
        )
        full = f"{header}\n\n📊 Per-TF: {per_tf_text}"
        send_message(full)
        signals_sent += 1
        ts = datetime.utcnow().isoformat()
        log_signal([ts, symbol, chosen_dir, chosen_entry, tp1, sl, conf_pct, units, f"{risk_pct*100:.2f}%", "open", per_tf_text])
        open_trades.append({"s":symbol,"side":chosen_dir,"entry":chosen_entry,"tp1":tp1,"sl":sl,"st":"open","units":units})
        return True
    return False

# ========== TRADE CHECK ==========
def check_trades():
    global signals_hit, signals_fail
    for t in list(open_trades):
        p = get_last_price_yf(t["s"])
        if p is None: continue
        if t["side"]=="BUY":
            if p >= t["tp1"]:
                t["st"]="hit"; signals_hit += 1
                send_message(f"🎯 {t['s']} TP1 Hit {p:.6f}")
                log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t["entry"], t["tp1"], t["sl"], "NA", t.get("units"), "NA", "hit", ""])
                open_trades.remove(t)
            elif p <= t["sl"]:
                t["st"]="fail"; signals_fail += 1
                send_message(f"❌ {t['s']} SL Hit {p:.6f}")
                log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t["entry"], t["tp1"], t["sl"], "NA", t.get("units"), "NA", "fail", ""])
                open_trades.remove(t)
        else:
            if p <= t["tp1"]:
                t["st"]="hit"; signals_hit += 1
                send_message(f"🎯 {t['s']} TP1 Hit {p:.6f}")
                log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t["entry"], t["tp1"], t["sl"], "NA", t.get("units"), "NA", "hit", ""])
                open_trades.remove(t)
            elif p >= t["sl"]:
                t["st"]="fail"; signals_fail += 1
                send_message(f"❌ {t['s']} SL Hit {p:.6f}")
                log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t["entry"], t["tp1"], t["sl"], "NA", t.get("units"), "NA", "fail", ""])
                open_trades.remove(t)

# ========== STARTUP & MAIN LOOP ==========
init_csv()
send_message("✅ Top40 YahooFX bot started (75%+ TF filter, session filters).")

while True:
    try:
        # volatility check using EURUSD quick snapshot
        if short_volatility_spike_check():
            vol_pause_until = time.time() + VOLATILITY_PAUSE
            send_message(f"⚠️ Volatility spike detected — pausing signals for {VOLATILITY_PAUSE//60} minutes.")
        # iterate symbols
        for i, sym in enumerate(SYMBOLS, start=1):
            analyze_symbol(sym)
            if i % 10 == 0:
                print(f"Analyzed {i}/{len(SYMBOLS)} symbols...")
            time.sleep(0.25)
        check_trades()
        # heartbeat + daily summary
        now = time.time()
        if now - last_heartbeat > 43200:
            send_message(f"💓 Heartbeat OK {datetime.utcnow().strftime('%H:%M UTC')}")
            last_heartbeat = now
        if now - last_summary > 86400:
            total = signals_sent
            hits = signals_hit
            fails = signals_fail
            acc = (hits/total*100) if total>0 else 0.0
            send_message(f"📊 Daily Summary\nSignals:{total}\nHits:{hits}\nFails:{fails}\nAccuracy:{acc:.1f}%")
            last_summary = now
        print("Cycle", datetime.utcnow().strftime("%H:%M:%S"), "UTC")
        time.sleep(CHECK_INTERVAL)
    except Exception as e:
        print("Main loop error:", e)
        time.sleep(10)
