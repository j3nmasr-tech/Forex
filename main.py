#!/usr/bin/env python3
# SIRTS Swing — Pure Swing (1H, 4H, 1D) | Bybit v5 (adapted from v10)
# Requirements: requests, pandas, numpy
# Environment variables: BOT_TOKEN, CHAT_ID
# Notes: uses btc_trend_agree() and entry_allowed() exactly as you provided (adapted to symbol formats)

import os
import re
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import csv

# ====== SANITIZE ======
def sanitize_symbol(symbol: str) -> str:
    if not symbol or not isinstance(symbol, str):
        return ""
    s = re.sub(r"[^A-Z0-9_.-]", "", symbol.upper())
    return s[:20]

# ====== CONFIG (Swing-specific) ======
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

CAPITAL = 200.0
LEVERAGE = 10

# cooldowns (swing slower)
COOLDOWN_TIME_DEFAULT = 6 * 60 * 60   # 6 hours between signals per symbol
COOLDOWN_TIME_SUCCESS = 24 * 60 * 60  # 24 hours after win
COOLDOWN_TIME_FAIL    = 12 * 60 * 60  # 12 hours after loss

VOLATILITY_THRESHOLD_PCT = 3.0
VOLATILITY_PAUSE = 60 * 60  # pause 1 hour on volatility spike
CHECK_INTERVAL = 10 * 60    # 10 minutes per cycle for swing

API_CALL_DELAY = 0.1

TIMEFRAMES = ["1h", "4h", "1d"]

WEIGHT_BIAS   = 0.25
WEIGHT_TURTLE = 0.35
WEIGHT_CRT    = 0.25
WEIGHT_VOLUME = 0.15

# stricter for swing
MIN_TF_SCORE  = 60
CONF_MIN_TFS  = 2
CONFIDENCE_MIN = 65.0

MIN_QUOTE_VOLUME = 2_000_000.0   # top pairs only
TOP_SYMBOLS = 40

# Bybit endpoints (v5)
BYBIT_BASE    = "https://api.bybit.com/v5/market"
BYBIT_KLINES  = f"{BYBIT_BASE}/kline"
BYBIT_TICKERS = f"{BYBIT_BASE}/tickers"
FNG_API       = "https://api.alternative.me/fng/?limit=1"

LOG_CSV = "./sirts_swing_signals.csv"

# Limits
MAX_OPEN_TRADES = 40
MAX_EXPOSURE_PCT = 0.20
MIN_MARGIN_USD = 0.25
MIN_SL_DISTANCE_PCT = 0.0015

RECENT_SIGNAL_SIGNATURE_EXPIRE = 3600 * 6  # 6 hours
recent_signals = {}
open_trades = []
last_trade_time = {}

# Stats
signals_sent_total = 0
signals_hit_total = 0
signals_fail_total = 0
signals_breakeven = 0
total_checked_signals = 0
skipped_signals = 0

# ====== HELPERS ======
def send_message(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured:", text)
        return False
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      json={"chat_id": CHAT_ID, "text": text}, timeout=10)
        return True
    except Exception as e:
        print("Telegram send error:", e)
        return False

def safe_get_json(url, params=None, timeout=8, retries=1):
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠ API request error ({e}) for {url} params={params} attempt={attempt+1}/{retries+1}")
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            return None
        except Exception as e:
            print(f"⚠ Unexpected error fetching {url}: {e}")
            return None

# ====== BYBIT ACCESSORS ======
_interval_map = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "1d": "D", "1w": "W"
}

def interval_to_bybit(interval):
    return _interval_map.get(interval, interval)

def get_top_symbols(n=TOP_SYMBOLS):
    j = safe_get_json(BYBIT_TICKERS, params={"category":"linear"}, timeout=8, retries=1)
    if not j or "result" not in j or "list" not in j["result"]:
        return ["BTCUSDT","ETHUSDT"]
    usdt_pairs = []
    for d in j["result"]["list"]:
        sym = sanitize_symbol(d.get("symbol","").upper())
        if not sym.endswith("USDT"):
            continue
        try:
            last = float(d.get("lastPrice") or d.get("last_price") or 0)
            vol = float(d.get("volume24h", 0) or d.get("volume", 0) or 0)
            quote_vol = vol * (last or 1.0)
            usdt_pairs.append((sym, quote_vol))
        except:
            continue
    usdt_pairs.sort(key=lambda x: x[1], reverse=True)
    syms = [s[0] for s in usdt_pairs[:n]]
    return syms if syms else ["BTCUSDT","ETHUSDT"]

def get_24h_quote_volume(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return 0.0
    j = safe_get_json(BYBIT_TICKERS, params={"category":"linear","symbol":symbol}, timeout=6, retries=1)
    if not j or "result" not in j or "list" not in j["result"]:
        return 0.0
    for d in j["result"]["list"]:
        if d.get("symbol","").upper() == symbol:
            try:
                vol  = float(d.get("volume24h", 0) or d.get("volume", 0) or 0)
                last = float(d.get("lastPrice", 0) or d.get("last_price", 0) or 0)
                return vol * (last or 1.0)
            except:
                return 0.0
    return 0.0

def get_klines(symbol, interval="1h", limit=200):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    iv = interval_to_bybit(interval)
    params = {"category":"linear","symbol":symbol,"interval":iv,"limit":limit}
    j = safe_get_json(BYBIT_KLINES, params=params, timeout=8, retries=1)
    if not j or "result" not in j or "list" not in j["result"]:
        return None
    data = j["result"]["list"]
    if not data:
        return None
    try:
        if isinstance(data[0], dict):
            df = pd.DataFrame(data)
            if set(["open","high","low","close","volume"]).issubset(df.columns):
                df = df[["open","high","low","close","volume"]].astype(float)
            elif set(["o","h","l","c","v"]).issubset(df.columns):
                df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
                df = df[["open","high","low","close","volume"]].astype(float)
            else:
                numeric_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
                if len(numeric_cols) >= 5:
                    df = df[numeric_cols[:5]].astype(float)
                    df.columns = ["open","high","low","close","volume"]
                else:
                    return None
        elif isinstance(data[0], list):
            df = pd.DataFrame(data)
            if df.shape[1] >= 6:
                df = df.iloc[:,1:6]
                df.columns = ["open","high","low","close","volume"]
                df = df.astype(float)
            else:
                return None
        else:
            return None
        return df
    except Exception as e:
        print(f"⚠ get_klines parse error for {symbol} {interval}: {e}")
        return None

def get_price(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    j = safe_get_json(BYBIT_TICKERS, params={"category":"linear","symbol":symbol}, timeout=6, retries=1)
    if not j or "result" not in j or "list" not in j["result"]:
        return None
    for d in j["result"]["list"]:
        if d.get("symbol","").upper() == symbol:
            try:
                return float(d.get("lastPrice") or d.get("last_price") or d.get("last") or 0)
            except:
                return None
    return None

# ====== INDICATORS (from v10) ======
def detect_crt(df):
    if len(df) < 12:
        return False, False
    last = df.iloc[-1]
    o = float(last["open"]); h = float(last["high"]); l = float(last["low"])
    c = float(last["close"]); v = float(last["volume"])
    body_series = (df["close"] - df["open"]).abs()
    avg_body = body_series.rolling(10, min_periods=6).mean().iloc[-1]
    avg_vol  = df["volume"].rolling(10, min_periods=6).mean().iloc[-1]
    if np.isnan(avg_body) or np.isnan(avg_vol):
        return False, False
    body = abs(c - o)
    wick_up   = h - max(o, c)
    wick_down = min(o, c) - l
    bull = (
        (body < avg_body * 0.7) and
        (wick_down > avg_body * 0.6) and
        (v < avg_vol * 1.2) and
        (c > o)
    )
    bear = (
        (body < avg_body * 0.7) and
        (wick_up > avg_body * 0.6) and
        (v < avg_vol * 1.2) and
        (c < o)
    )
    return bull, bear

def detect_turtle(df, look=20):
    if len(df) < look+2:
        return False, False
    ph = df["high"].iloc[-look-1:-1].max()
    pl = df["low"].iloc[-look-1:-1].min()
    last = df.iloc[-1]
    bull = (last["low"] < pl) and (last["close"] > pl * 1.005)
    bear = (last["high"] > ph) and (last["close"] < ph * 0.995)
    return bull, bear

def smc_bias(df):
    e20 = df["close"].ewm(span=20).mean().iloc[-1]
    e50 = df["close"].ewm(span=50).mean().iloc[-1]
    if (e20 - e50) / e50 > 0.002:
        return "bull"
    else:
        return "bear"

def volume_ok(df):
    ma = df["volume"].rolling(20, min_periods=8).mean().iloc[-1]
    if np.isnan(ma):
        return True
    current = df["volume"].iloc[-1]
    e20 = df["close"].ewm(span=20).mean().iloc[-1]
    e50 = df["close"].ewm(span=50).mean().iloc[-1]
    mult = 1.2 if e20 > e50 else 1.1
    return current > ma * mult

# ====== ATR & SIZING ======
def get_atr(symbol, period=14):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    # use 4H-based ATR for swing stability (we'll fetch 1h but could be adapted)
    df = get_klines(symbol, "1h", period+1)
    if df is None or len(df) < period+1:
        return None
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    trs = []
    for i in range(1, len(df)):
        trs.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    if not trs:
        return None
    return max(float(np.mean(trs)), 1e-8)

def trade_params(symbol, entry, side, atr_multiplier_sl=2.2, tp_mults=(3.0,5.0,8.0), conf_multiplier=1.0):
    atr = get_atr(symbol)
    if atr is None:
        return None
    atr = max(min(atr, entry * 0.05), entry * 0.0001)
    adj_sl_multiplier = atr_multiplier_sl * (1.0 + (0.5 - conf_multiplier) * 0.5)
    if side == "BUY":
        sl  = round(entry - atr * adj_sl_multiplier, 8)
        tp1 = round(entry + atr * tp_mults[0] * conf_multiplier, 8)
        tp2 = round(entry + atr * tp_mults[1] * conf_multiplier, 8)
        tp3 = round(entry + atr * tp_mults[2] * conf_multiplier, 8)
    else:
        sl  = round(entry + atr * adj_sl_multiplier, 8)
        tp1 = round(entry - atr * tp_mults[0] * conf_multiplier, 8)
        tp2 = round(entry - atr * tp_mults[1] * conf_multiplier, 8)
        tp3 = round(entry - atr * tp_mults[2] * conf_multiplier, 8)
    return sl, tp1, tp2, tp3

def pos_size_units(entry, sl, confidence_pct):
    conf = max(0.0, min(100.0, confidence_pct))
    MIN_RISK = 0.01
    MAX_RISK = 0.04
    risk_percent = MIN_RISK + (MAX_RISK - MIN_RISK) * (conf / 100.0)
    risk_percent = max(MIN_RISK, min(MAX_RISK, risk_percent))
    risk_usd     = CAPITAL * risk_percent
    sl_dist      = abs(entry - sl)
    min_sl = max(entry * MIN_SL_DISTANCE_PCT, 1e-8)
    if sl_dist < min_sl:
        return 0.0, 0.0, 0.0, risk_percent
    units = risk_usd / sl_dist
    exposure = units * entry
    max_exposure = CAPITAL * MAX_EXPOSURE_PCT
    if exposure > max_exposure and exposure > 0:
        units = max_exposure / entry
        exposure = units * entry
    margin_req = exposure / LEVERAGE
    if margin_req < MIN_MARGIN_USD:
        return 0.0, 0.0, 0.0, risk_percent
    return round(units,8), round(margin_req,6), round(exposure,6), risk_percent

# ====== Sentiment (optional) ======
def get_fear_greed_value():
    j = safe_get_json(FNG_API, {}, timeout=3, retries=1)
    try:
        return int(j["data"][0]["value"])
    except:
        return 50

def sentiment_label():
    v = get_fear_greed_value()
    if v < 25:
        return "fear"
    if v > 75:
        return "greed"
    return "neutral"

# ===== BTC TREND FOR SWING =====
def btc_trend_agree():
    # Use 4H + 1D for swing
    df4 = get_klines("BTCUSDT", "4h", 300)
    dfD = get_klines("BTCUSDT", "1d", 300)
    if df4 is None or dfD is None:
        return None, None, None

    # Compute SMC bias
    b4 = smc_bias(df4)
    bD = smc_bias(dfD)

    # Macro trend using daily 200 SMA
    sma200 = dfD["close"].rolling(200).mean().iloc[-1] if len(dfD) >= 200 else None
    btc_price = float(dfD["close"].iloc[-1])
    trend_by_sma = "bull" if (sma200 and btc_price > sma200) else ("bear" if sma200 and btc_price < sma200 else None)

    # Confirm strict agreement
    strict_agree = (b4 == bD and b4 is not None)
    final_dir = b4 if strict_agree else None

    return strict_agree, final_dir, trend_by_sma


# ===== ENTRY FILTERS FOR SWING =====
def entry_allowed(symbol, df):
    # 1. Skip huge candle (ATR spike)
    atr = get_atr(symbol)
    last_candle = df.iloc[-1]
    if atr is not None and abs(last_candle['close'] - last_candle['open']) > 1.8 * atr:
        return False

    # 2. Skip tight consolidation (tiny range)
    recent_high = df['high'].iloc[-5:].max()
    recent_low  = df['low'].iloc[-5:].min()
    if (recent_high - recent_low)/recent_low < 0.003:
        return False

    # 3. Skip if BTC 4H and 1D disagree
    btc_agree, btc_dir, btc_macro = btc_trend_agree()
    if btc_agree is None or not btc_agree:
        return False

    # 4. Optional: enforce that the symbol trade matches BTC direction
    if btc_dir is not None:
        current_dir = get_direction_from_ma(df, span=50)  # use higher MA for swing
        if current_dir != btc_dir:
            return False

    return True

# ====== LOGGING ======
def init_csv():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV,"w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_utc","symbol","side","entry","tp1","tp2","tp3","sl",
                "tf","units","margin_usd","exposure_usd","risk_pct","confidence_pct","status","breakdown"
            ])

def log_signal(row):
    try:
        with open(LOG_CSV,"a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print("log_signal error:", e)

def log_trade_close(trade):
    try:
        with open(LOG_CSV,"a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(), trade.get("s"), trade.get("side"), trade.get("entry"),
                trade.get("tp1"), trade.get("tp2"), trade.get("tp3"), trade.get("sl"),
                trade.get("entry_tf"), trade.get("units"), trade.get("margin"), trade.get("exposure"),
                (trade.get("risk_pct")*100) if trade.get("risk_pct") else None, trade.get("confidence_pct"),
                trade.get("st"), trade.get("close_breakdown", "")
            ])
    except Exception as e:
        print("log_trade_close error:", e)

# ====== ANALYSIS & SIGNAL GENERATION ======
def current_total_exposure():
    return sum([t.get("exposure", 0) for t in open_trades if t.get("st") == "open"])

def analyze_symbol(symbol):
    global total_checked_signals, skipped_signals, signals_sent_total, recent_signals, last_trade_time
    total_checked_signals += 1
    now = time.time()

    if not symbol or not isinstance(symbol, str):
        skipped_signals += 1
        return False

    symbol = sanitize_symbol(symbol)

    vol24 = get_24h_quote_volume(symbol)
    if vol24 < MIN_QUOTE_VOLUME:
        skipped_signals += 1
        return False

    if last_trade_time.get(symbol, 0) > now:
        skipped_signals += 1
        return False

    tf_confirmations = 0
    chosen_dir      = None
    chosen_entry    = None
    chosen_tf       = None
    confirming_tfs  = []
    breakdown_per_tf = {}
    per_tf_scores = []

    # BTC trend pre-check (use same function that entry_allowed uses)
    btc_agree, btc_dir, btc_sma_trend = btc_trend_agree()
    # if btc_agree is None -> API problems; we will skip symbol to be safe
    if btc_agree is None:
        skipped_signals += 1
        return False

    for tf in TIMEFRAMES:
        df = get_klines(symbol, tf)
        if df is None or len(df) < 60:
            breakdown_per_tf[tf] = None
            continue

        # apply entry_allowed early (uses BTC check + ATR/consolidation rules)
        if not entry_allowed(symbol, df):
            breakdown_per_tf[tf] = {"entry_allowed": False}
            continue

        crt_b, crt_s = detect_crt(df)
        ts_b, ts_s = detect_turtle(df)
        bias        = smc_bias(df)
        vol_ok      = volume_ok(df)

        bull_score = (WEIGHT_CRT*(1 if crt_b else 0) + WEIGHT_TURTLE*(1 if ts_b else 0) +
                      WEIGHT_VOLUME*(1 if vol_ok else 0) + WEIGHT_BIAS*(1 if bias=="bull" else 0))*100
        bear_score = (WEIGHT_CRT*(1 if crt_s else 0) + WEIGHT_TURTLE*(1 if ts_s else 0) +
                      WEIGHT_VOLUME*(1 if vol_ok else 0) + WEIGHT_BIAS*(1 if bias=="bear" else 0))*100

        breakdown_per_tf[tf] = {
            "bull_score": int(bull_score),
            "bear_score": int(bear_score),
            "bias": bias,
            "vol_ok": vol_ok,
            "crt_b": bool(crt_b),
            "crt_s": bool(crt_s),
            "ts_b": bool(ts_b),
            "ts_s": bool(ts_s)
        }

        per_tf_scores.append(max(bull_score, bear_score))

        if bull_score >= MIN_TF_SCORE:
            tf_confirmations += 1
            chosen_dir    = "BUY"
            chosen_entry  = float(df["close"].iloc[-1])
            chosen_tf     = tf
            confirming_tfs.append(tf)
        elif bear_score >= MIN_TF_SCORE:
            tf_confirmations += 1
            chosen_dir   = "SELL"
            chosen_entry = float(df["close"].iloc[-1])
            chosen_tf    = tf
            confirming_tfs.append(tf)

    # require confirmations for swing
    if not (tf_confirmations >= CONF_MIN_TFS and chosen_dir and chosen_entry is not None):
        skipped_signals += 1
        return False

    confidence_pct = float(np.mean(per_tf_scores)) if per_tf_scores else 0.0
    confidence_pct = max(0.0, min(100.0, confidence_pct))

    if confidence_pct < CONFIDENCE_MIN:
        skipped_signals += 1
        return False

    # enforce BTC direction exactly as in scalp: block trades against btc_dir when btc_agree True
    # btc_dir is "bull" or "bear" or None
    if btc_agree and btc_dir is not None:
        if (btc_dir == "bull" and chosen_dir == "SELL") or (btc_dir == "bear" and chosen_dir == "BUY"):
            # skip trade if against BTC direction
            skipped_signals += 1
            return False

    # exposure / open trades check
    if len([t for t in open_trades if t.get("st") == "open"]) >= MAX_OPEN_TRADES:
        skipped_signals += 1
        return False

    # dedupe signature
    sig = (symbol, chosen_dir, round(chosen_entry, 6))
    if recent_signals.get(sig, 0) + RECENT_SIGNAL_SIGNATURE_EXPIRE > time.time():
        skipped_signals += 1
        return False
    recent_signals[sig] = time.time()

    sentiment = sentiment_label()
    entry = get_price(symbol)
    if entry is None:
        skipped_signals += 1
        return False

    conf_multiplier = max(0.6, min(1.2, confidence_pct / 100.0 + 0.6))
    tp_sl = trade_params(symbol, entry, chosen_dir, conf_multiplier=conf_multiplier)
    if not tp_sl:
        skipped_signals += 1
        return False
    sl, tp1, tp2, tp3 = tp_sl

    units, margin, exposure, risk_used = pos_size_units(entry, sl, confidence_pct)
    if units <= 0 or margin <= 0 or exposure <= 0:
        skipped_signals += 1
        return False

    if exposure > CAPITAL * MAX_EXPOSURE_PCT:
        skipped_signals += 1
        return False

    header = (f"✅ {chosen_dir} {symbol}\n"
              f"💵 Entry: {entry}\n"
              f"🎯 TP1:{tp1} TP2:{tp2} TP3:{tp3}\n"
              f"🛑 SL: {sl}\n"
              f"💰 Units:{units} | Margin≈${margin} | Exposure≈${exposure}\n"
              f"⚠ Risk used: {risk_used*100:.2f}% | Confidence: {confidence_pct:.1f}% | Sentiment:{sentiment}")

    send_message(header)

    trade_obj = {
        "s": symbol,
        "side": chosen_dir,
        "entry": entry,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl": sl,
        "st": "open",
        "units": units,
        "margin": margin,
        "exposure": exposure,
        "risk_pct": risk_used,
        "confidence_pct": confidence_pct,
        "tp1_taken": False,
        "tp2_taken": False,
        "tp3_taken": False,
        "placed_at": time.time(),
        "entry_tf": chosen_tf,
    }
    open_trades.append(trade_obj)

    log_signal([
        datetime.utcnow().isoformat(), symbol, chosen_dir, entry,
        tp1, tp2, tp3, sl, chosen_tf, units, margin, exposure,
        risk_used*100, confidence_pct, "open", str(breakdown_per_tf)
    ])
    signals_sent_total_local = 1
    return True

# ====== TRADE MONITOR ======
def check_trades():
    global signals_hit_total, signals_fail_total, signals_breakeven
    for t in list(open_trades):
        if t.get("st") != "open":
            continue
        p = get_price(t["s"])
        if p is None:
            continue
        side = t["side"]
        if side == "BUY":
            if not t["tp1_taken"] and p >= t["tp1"]:
                t["tp1_taken"] = True
                t["sl"] = t["entry"]
                send_message(f"🎯 {t['s']} TP1 Hit {p} — SL moved to breakeven.")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp1_taken"] and not t["tp2_taken"] and p >= t["tp2"]:
                t["tp2_taken"] = True
                send_message(f"🎯 {t['s']} TP2 Hit {p}")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp2_taken"] and not t["tp3_taken"] and p >= t["tp3"]:
                t["tp3_taken"] = True
                t["st"] = "closed"
                send_message(f"🏁 {t['s']} TP3 Hit {p} — Trade closed.")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                log_trade_close(t)
                continue
            if p <= t["sl"]:
                if abs(t["sl"] - t["entry"]) < 1e-8:
                    t["st"] = "breakeven"
                    signals_breakeven += 1
                    send_message(f"⚖️ {t['s']} Breakeven SL Hit {p}")
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    log_trade_close(t)
                else:
                    t["st"] = "fail"
                    signals_fail_total += 1
                    send_message(f"❌ {t['s']} SL Hit {p}")
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_FAIL
                    log_trade_close(t)
        else:  # SELL
            if not t["tp1_taken"] and p <= t["tp1"]:
                t["tp1_taken"] = True
                t["sl"] = t["entry"]
                send_message(f"🎯 {t['s']} TP1 Hit {p} — SL moved to breakeven.")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp1_taken"] and not t["tp2_taken"] and p <= t["tp2"]:
                t["tp2_taken"] = True
                send_message(f"🎯 {t['s']} TP2 Hit {p}")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp2_taken"] and not t["tp3_taken"] and p <= t["tp3"]:
                t["tp3_taken"] = True
                t["st"] = "closed"
                send_message(f"🏁 {t['s']} TP3 Hit {p} — Trade closed.")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                log_trade_close(t)
                continue
            if p >= t["sl"]:
                if abs(t["sl"] - t["entry"]) < 1e-8:
                    t["st"] = "breakeven"
                    signals_breakeven += 1
                    send_message(f"⚖️ {t['s']} Breakeven SL Hit {p}")
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    log_trade_close(t)
                else:
                    t["st"] = "fail"
                    signals_fail_total += 1
                    send_message(f"❌ {t['s']} SL Hit {p}")
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_FAIL
                    log_trade_close(t)
    # cleanup closed trades
    for t in list(open_trades):
        if t.get("st") in ("closed", "fail", "breakeven"):
            try:
                open_trades.remove(t)
            except Exception:
                pass

# ====== HEARTBEAT & SUMMARY ======
def heartbeat():
    send_message(f"💓 SIRTS Swing Heartbeat {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("Heartbeat sent.")

def summary():
    total = signals_sent_total
    hits  = signals_hit_total
    fails = signals_fail_total
    breakev = signals_breakeven
    acc   = (hits / total * 100) if total > 0 else 0.0
    send_message(f"📊 Swing Summary\nSignals Sent: {total}\nSignals Checked: {total_checked_signals}\nSignals Skipped: {skipped_signals}\n✅ Hits: {hits}\n⚖️ Breakeven: {breakev}\n❌ Fails: {fails}\n🎯 Accuracy: {acc:.1f}%")
    print(f"Swing Summary. Accuracy: {acc:.1f}%")

# ====== STARTUP ======
init_csv()
send_message("✅ SIRTS Swing deployed — Pure Swing (1H/4H/1D) with scalp BTC-direction logic adapted.")
print("SIRTS Swing started.")

try:
    SYMBOLS = get_top_symbols(TOP_SYMBOLS)
    print(f"Monitoring {len(SYMBOLS)} symbols (Top {TOP_SYMBOLS}).")
except Exception as e:
    SYMBOLS = ["BTCUSDT","ETHUSDT"]
    print("Warning retrieving top symbols, defaulting to BTCUSDT & ETHUSDT.")

# ====== MAIN LOOP ======
last_heartbeat = time.time()
last_summary = time.time()

while True:
    try:
        # check BTC volatility spike
        def btc_volatility_spike():
            df = get_klines("BTCUSDT", "1h", 3)
            if df is None or len(df) < 3:
                return False
            pct = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100.0
            return abs(pct) >= VOLATILITY_THRESHOLD_PCT

        if btc_volatility_spike():
            print("BTC volatility spike detected — pausing briefly")
            time.sleep(VOLATILITY_PAUSE)

        for i, sym in enumerate(SYMBOLS, start=1):
            print(f"[{i}/{len(SYMBOLS)}] Scanning {sym} …")
            try:
                analyze_symbol(sym)
            except Exception as e:
                print(f"Error scanning {sym}: {e}")
            time.sleep(API_CALL_DELAY)

        check_trades()

        now = time.time()
        if now - last_heartbeat > 86400:
            heartbeat()
            last_heartbeat = now
        if now - last_summary > 86400:
            summary()
            last_summary = now

        print("Cycle completed at", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
        time.sleep(CHECK_INTERVAL)
    except Exception as e:
        print("Main loop error:", e)
        time.sleep(30)