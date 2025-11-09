#!/usr/bin/env python3
# SIRTS v11 Swing — Top 80 | Bybit USDT Perpetual (v5 API)
# Fully runnable single-file version
# Requires: requests, pandas, numpy
# ENV: BOT_TOKEN, CHAT_ID, DEBUG_LEVEL (TRACE/INFO/OFF)

import os, re, time, requests, pandas as pd, numpy as np, csv
from datetime import datetime, timezone

# ===== DEBUG / LOGGING =====
DEBUG_LEVEL = os.environ.get("DEBUG_LEVEL", "TRACE").upper()
def dbg(msg, lvl="INFO"):
    if DEBUG_LEVEL == "OFF": return
    if DEBUG_LEVEL == "INFO" and lvl == "TRACE": return
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{lvl}] {ts} — {msg}")

# ===== SYMBOL SANITIZATION =====
def sanitize_symbol(symbol: str) -> str:
    if not symbol or not isinstance(symbol, str): return ""
    s = re.sub(r"[^A-Z0-9_.-]", "", symbol.upper())
    return s[:20]

# ===== CONFIG =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

CAPITAL = 80.0
LEVERAGE = 30
BASE_RISK = 0.02
MAX_EXPOSURE_PCT = 0.2

COOLDOWN_TIME_SUCCESS = 15*60
COOLDOWN_TIME_FAIL    = 45*60
VOLATILITY_THRESHOLD_PCT = 2.5
VOLATILITY_PAUSE = 1800
CHECK_INTERVAL = 1800
API_CALL_DELAY = 0.05

TIMEFRAMES = ["1h","4h","1d"]
CONFIDENCE_MIN = 55.0
TOP_SYMBOLS = 80
RECENT_SIGNAL_SIGNATURE_EXPIRE = 300

BYBIT_BASE = "https://api.bybit.com"
BYBIT_KLINE = f"{BYBIT_BASE}/v5/market/kline"
BYBIT_TICKERS = f"{BYBIT_BASE}/v5/market/tickers"
FNG_API = "https://api.alternative.me/fng/?limit=1"

LOG_CSV = "./sirts_v11_swing_signals_bybit.csv"

# ===== STATE =====
open_trades = []
recent_signals = {}
last_trade_time = {}
last_directional_trade = {}
signals_sent_total = 0
signals_hit_total = 0
signals_fail_total = 0
signals_breakeven = 0
total_checked_signals = 0
skipped_signals = 0
volatility_pause_until = 0
last_heartbeat = time.time()
last_summary = time.time()
last_trade_result = {}

# ===== HELPERS =====
def send_message(text):
    if not BOT_TOKEN or not CHAT_ID:
        dbg("Telegram not configured; msg not sent", "INFO")
        dbg(text, "TRACE")
        return False
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        return True
    except Exception as e:
        dbg(f"Telegram send error: {e}", "INFO")
        return False

def safe_get_json(url, params=None, timeout=8, retries=1):
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params or {}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            dbg(f"API request error ({e}) for {url} params={params} attempt={attempt+1}/{retries+1}", "TRACE")
            if attempt < retries: time.sleep(0.6*(attempt+1)); continue
            return None
        except Exception as e:
            dbg(f"Unexpected error fetching {url}: {e}", "INFO")
            return None

def tf_to_bybit_interval(tf: str) -> str:
    return {"1m":"1","3m":"3","5m":"5","15m":"15","30m":"30",
            "1h":"60","2h":"120","4h":"240","6h":"360","12h":"720",
            "1d":"D","1w":"W"}.get(tf.lower(),"60")

# ===== PART 1: Bybit MARKET DATA =====
def get_top_symbols(n=TOP_SYMBOLS):
    j = safe_get_json(BYBIT_TICKERS, {"category":"linear"})
    if not j or "result" not in j or "list" not in j["result"]: return ["BTCUSDT","ETHUSDT"]
    results = j["result"]["list"]
    try:
        df = pd.DataFrame(results)
        df["turnover24h"] = df.get("turnover24h",0).astype(float, errors="ignore").fillna(0)
        df = df.sort_values("turnover24h", ascending=False)
        syms = [sanitize_symbol(s) for s in df["symbol"].tolist() if s.endswith("USDT")]
        return syms[:n] if syms else ["BTCUSDT","ETHUSDT"]
    except Exception:
        return [sanitize_symbol(r["symbol"]) for r in results if r.get("symbol","").endswith("USDT")][:n]

def get_price(symbol):
    symbol = sanitize_symbol(symbol)
    j = safe_get_json(BYBIT_TICKERS, {"category":"linear"})
    if not j or "result" not in j: return None
    items = j["result"].get("list", [])
    for item in items:
        if sanitize_symbol(item.get("symbol","")) == symbol:
            try: return float(item.get("lastPrice", item.get("last_price",0)))
            except: continue
    return None

def get_klines(symbol, interval="1h", limit=200):
    symbol = sanitize_symbol(symbol)
    params = {"category":"linear","symbol":symbol,"interval":tf_to_bybit_interval(interval),"limit":limit}
    j = safe_get_json(BYBIT_KLINE, params=params)
    rows=[]
    if isinstance(j, dict) and "result" in j and "list" in j["result"]:
        for r in j["result"]["list"]:
            rows.append([r.get("open_time") or r.get("start_at"), r.get("open"), r.get("high"), r.get("low"), r.get("close"), r.get("volume")])
    if not rows: return None
    df = pd.DataFrame(rows, columns=["t","o","h","l","c","v"])
    try:
        df = df.astype(float)
        df.columns = ["t","open","high","low","close","volume"]
        return df
    except: return None

# ===== PART 2: INDICATORS =====
def atr(df, period=14):
    if df is None or len(df)<period: return None
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period,min_periods=1).mean().iloc[-1]

def ema(series, period):
    if series is None or len(series)<period: return None
    return series.ewm(span=period, adjust=False).mean().iloc[-1]

def ema200(df):
    if df is None or "close" not in df.columns or len(df)<200: return None
    return ema(df["close"],200)

def calc_position_size(price, risk=BASE_RISK, capital=CAPITAL, leverage=LEVERAGE, atr_val=None, stop_loss=None):
    if not price or price<=0: return 0.0
    if stop_loss and atr_val:
        distance = abs(price-stop_loss)
        if distance<=0: return 0.0
        return min(capital*risk/distance*leverage, capital*leverage)
    return capital*risk*leverage/price

def fear_and_greed_index():
    try:
        j = safe_get_json(FNG_API, timeout=6, retries=1)
        return int(j["data"][0].get("value",50)) if j and "data" in j and len(j["data"])>0 else None
    except: return None

BTC_SYMBOL="BTCUSDT"
def btc_volatility_check():
    global volatility_pause_until
    df = get_klines(BTC_SYMBOL,"1h",limit=48)
    if df is None: return False
    max_p, min_p = df["high"].max(), df["low"].min()
    if min_p<=0: return False
    change_pct = 100*(max_p-min_p)/min_p
    if change_pct>=VOLATILITY_THRESHOLD_PCT:
        volatility_pause_until = time.time()+VOLATILITY_PAUSE
        dbg(f"BTC volatility too high: {change_pct:.2f}%, pausing new signals {VOLATILITY_PAUSE/60:.1f}min","INFO")
        return False
    return True

def btc_trend_check():
    df = get_klines(BTC_SYMBOL,"4h",limit=50)
    if df is None or len(df)<50: return None
    ema_fast = ema(df["close"],20)
    ema_slow = ema(df["close"],50)
    if ema_fast>ema_slow: return "UP"
    elif ema_fast<ema_slow: return "DOWN"
    return "NEUTRAL"

def ema200_filter(symbol):
    df_daily = get_klines(symbol,"1d",limit=250)
    if df_daily is None or len(df_daily)<200: return None
    price_now = df_daily["close"].iloc[-1]
    ema_val = ema(df_daily["close"],200)
    return "BULL" if price_now>ema_val else "BEAR" if price_now<ema_val else "NEUTRAL"

def compute_signal_strength(symbol):
    scores={}
    for tf in TIMEFRAMES:
        df = get_klines(symbol,tf,limit=50)
        if df is None: continue
        ema_val = ema(df["close"],20)
        last_close = df["close"].iloc[-1]
        scores[tf]=100 if last_close>ema_val else 0
    return np.mean(list(scores.values())) if scores else 0.0

def trade_params(symbol, entry, side):
    atr_val = atr(get_klines(symbol,"1h",limit=50)) or 0
    if side=="BUY":
        return entry-atr_val, entry+atr_val*1, entry+atr_val*2, entry+atr_val*3
    else:
        return entry+atr_val, entry-atr_val*1, entry-atr_val*2, entry-atr_val*3

def pos_size_units(entry, sl, confidence):
    atr_val = abs(entry-sl)
    if atr_val<=0: return 0,0,0,0
    risk_pct = BASE_RISK*(confidence/100.0)
    units = CAPITAL*risk_pct/atr_val*LEVERAGE
    margin = units*entry/LEVERAGE
    exposure = units*entry
    return units, margin, exposure, risk_pct

def log_signal(symbol, tf, side, price, stop_loss, take_profit, confidence, atr_val):
    row={"timestamp":datetime.now(timezone.utc).isoformat(),"symbol":symbol,"tf":tf,"side":side,"price":price,
         "stop_loss":stop_loss,"take_profit":take_profit,"confidence":confidence,"atr":atr_val}
    try:
        file_exists = os.path.isfile(LOG_CSV)
        with open(LOG_CSV,'a',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(row)
        dbg(f"Signal logged: {symbol} {side} {tf} conf={confidence:.1f}","TRACE")
    except Exception as e: dbg(f"Error logging signal: {e}","INFO")

def log_trade_close(trade_obj):
    try:
        row = {"timestamp":datetime.now(timezone.utc).isoformat(),"symbol":trade_obj["s"],"side":trade_obj["side"],
               "entry":trade_obj["entry"],"tp1":trade_obj["tp1"],"tp2":trade_obj["tp2"],"tp3":trade_obj["tp3"],
               "sl":trade_obj["sl"],"units":trade_obj["units"],"margin":trade_obj["margin"],"exposure":trade_obj["exposure"],
               "risk_pct":trade_obj["risk_pct"],"status":trade_obj["st"]}
        file_exists=os.path.isfile(LOG_CSV)
        with open(LOG_CSV,'a',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(row)
        dbg(f"Trade closed: {trade_obj['s']} status={trade_obj['st']}","TRACE")
    except Exception as e: dbg(f"Error logging trade close: {e}","INFO")

def init_csv():
    if not os.path.isfile(LOG_CSV):
        try:
            with open(LOG_CSV,'w',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(["timestamp","symbol","tf","side","price","stop_loss","take_profit","confidence","atr"])
            dbg(f"CSV log created: {LOG_CSV}","INFO")
        except Exception as e: dbg(f"Failed to create CSV log: {e}","INFO")

def analyze_symbol(symbol):
    global total_checked_signals, skipped_signals, signals_sent_total, recent_signals, last_directional_trade
    total_checked_signals+=1
    now=time.time()
    if now<volatility_pause_until: skipped_signals+=1; return False

    df1h=get_klines(symbol,"1h",200)
    df4h=get_klines(symbol,"4h",200)
    df1d=get_klines(symbol,"1d",200)
    if df1h is None or df4h is None or df1d is None: skipped_signals+=1; return False

    trend=ema200_filter(symbol)
    if trend is None: skipped_signals+=1; return False

    conf=compute_signal_strength(symbol)
    if conf<CONFIDENCE_MIN: skipped_signals+=1; return False

    side="BUY" if trend=="BULL" else "SELL"
    entry=get_price(symbol)
    if entry is None: skipped_signals+=1; return False

    atr_val=atr(df1h)
    if atr_val is None: skipped_signals+=1; return False

    sl,tp1,tp2,tp3=trade_params(symbol,entry,side)
    units,margin,exposure,risk_used=pos_size_units(entry,sl,conf)

    if units<=0 or exposure>CAPITAL*MAX_EXPOSURE_PCT: skipped_signals+=1; return False

    sig_key=(symbol,side)
    if recent_signals.get(sig_key,0)+RECENT_SIGNAL_SIGNATURE_EXPIRE>now: skipped_signals+=1; return False
    recent_signals[sig_key]=now
    last_directional_trade[sig_key]=now

    msg=f"SWING {side} {symbol}\nEntry:{entry}\nTP1:{tp1} TP2:{tp2} TP3:{tp3}\nSL:{sl}\nUnits:{units} | Margin:${margin} | Exposure:${exposure}\nConfidence:{conf:.1f}%"
    send_message(msg)

    trade_obj={"s":symbol,"side":side,"entry":entry,"tp1":tp1,"tp2":tp2,"tp3":tp3,"sl":sl,
               "st":"open","units":units,"margin":margin,"exposure":exposure,"risk_pct":risk_used,
               "confidence_pct":conf,"tp1_taken":False,"tp2_taken":False,"tp3_taken":False,
               "placed_at":now,"entry_tf":"1h"}
    open_trades.append(trade_obj)
    signals_sent_total+=1
    log_signal(symbol,"1h",side,entry,sl,(tp1,tp2,tp3),conf,atr_val)
    dbg(f"✅ Signal sent for {symbol} {side} at {entry}","INFO")
    return True

def check_trades():
    global open_trades, signals_hit_total, signals_fail_total, last_trade_result, last_trade_time
    now=time.time()
    for t in list(open_trades):
        if t["st"]!="open": continue
        p=get_price(t["s"])
        if p is None: continue
        side=t["side"]
        # BUY
        if side=="BUY":
            if not t["tp1_taken"] and p>=t["tp1"]: t.update({"tp1_taken":True,"sl":t["entry"]}); send_message(f"TP1 Hit {t['s']} {p} — SL breakeven"); signals_hit_total+=1; last_trade_result[t["s"]]="win"; last_trade_time[t["s"]]=now+COOLDOWN_TIME_SUCCESS; continue
            if not t["tp2_taken"] and t["tp1_taken"] and p>=t["tp2"]: t["tp2_taken"]=True; send_message(f"TP2 Hit {t['s']} {p}"); signals_hit_total+=1; last_trade_result[t["s"]]="win"; last_trade_time[t["s"]]=now+COOLDOWN_TIME_SUCCESS; continue
            if not t["tp3_taken"] and t["tp2_taken"] and p>=t["tp3"]: t.update({"tp3_taken":True,"st":"closed"}); send_message(f"TP3 Hit {t['s']} {p} — Trade closed"); signals_hit_total+=1; last_trade_result[t["s"]]="win"; last_trade_time[t["s"]]=now+COOLDOWN_TIME_SUCCESS; log_trade_close(t); continue
            if p<=t["sl"]: t.update({"st":"fail"}); signals_fail_total+=1; last_trade_result[t["s"]]="loss"; last_trade_time[t["s"]]=now+COOLDOWN_TIME_FAIL; send_message(f"SL Hit {t['s']} {p}"); log_trade_close(t)
        # SELL
        else:
                        if not t["tp1_taken"] and p <= t["tp1"]:
                t.update({"tp1_taken": True, "sl": t["entry"]})
                send_message(f"TP1 Hit {t['s']} {p} — SL breakeven")
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = now + COOLDOWN_TIME_SUCCESS
                continue
            if not t["tp2_taken"] and t["tp1_taken"] and p <= t["tp2"]:
                t["tp2_taken"] = True
                send_message(f"TP2 Hit {t['s']} {p}")
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = now + COOLDOWN_TIME_SUCCESS
                continue
            if not t["tp3_taken"] and t["tp2_taken"] and p <= t["tp3"]:
                t.update({"tp3_taken": True, "st": "closed"})
                send_message(f"TP3 Hit {t['s']} {p} — Trade closed")
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = now + COOLDOWN_TIME_SUCCESS
                log_trade_close(t)
                continue
            if p >= t["sl"]:
                t.update({"st": "fail"})
                signals_fail_total += 1
                last_trade_result[t["s"]] = "loss"
                last_trade_time[t["s"]] = now + COOLDOWN_TIME_FAIL
                send_message(f"SL Hit {t['s']} {p}")
                log_trade_close(t)

    # Remove closed or failed trades from open_trades
    open_trades[:] = [t for t in open_trades if t["st"] == "open"]

# ===== HEARTBEAT & SUMMARY =====
def heartbeat():
    global last_heartbeat
    send_message(f"💓 Heartbeat OK — {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
    dbg("Heartbeat sent.", "INFO")
    last_heartbeat = time.time()

def summary():
    global last_summary
    total = signals_sent_total
    hits = signals_hit_total
    fails = signals_fail_total
    breakev = signals_breakeven
    acc = (hits / total * 100) if total > 0 else 0
    send_message(f"📊 Daily Summary\nSignals Sent:{total}\nSkipped:{skipped_signals}\n✅ Hits:{hits}\n⚖ Breakeven:{breakev}\n❌ Fails:{fails}\nAccuracy:{acc:.1f}%")
    dbg(f"Daily Summary. Accuracy: {acc:.1f}%", "INFO")
    last_summary = time.time()

# ===== MAIN LOOP =====
init_csv()
send_message("✅ SIRTS v11 Swing deployed — EMA200 Swing Filters active.")

try:
    SYMBOLS = get_top_symbols(TOP_SYMBOLS)
    dbg(f"Monitoring {len(SYMBOLS)} symbols (Top {TOP_SYMBOLS}).", "INFO")
except:
    SYMBOLS = ["BTCUSDT","ETHUSDT"]
    dbg("Default symbols BTC/ETH loaded.", "INFO")

while True:
    try:
        btc_volatility_check()
        for i, sym in enumerate(SYMBOLS, start=1):
            dbg(f"[{i}/{len(SYMBOLS)}] Scanning {sym} …", "TRACE")
            analyze_symbol(sym)
            time.sleep(API_CALL_DELAY)
        check_trades()
        now = time.time()
        if now - last_heartbeat > 43200:  # 12 hours
            heartbeat()
        if now - last_summary > 86400:  # 24 hours
            summary()
        dbg("Swing cycle completed", "INFO")
        time.sleep(CHECK_INTERVAL)
    except Exception as e:
        dbg(f"Main loop error: {e}", "INFO")
        time.sleep(5)