import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf # Cuma buat Kurs IDR
from datetime import datetime
import pytz

# --- KONFIGURASI ENGINE ---
SYMBOL_BINANCE = "PAXGUSDT" # Pair Likuiditas Terbesar
TICKER_IDR = "IDR=X"
MODAL_GAJI = 5000000 
SPREAD_AJAIB = 1.015 

# --- HELPER ---
def fmt_idr(val): 
    if pd.isna(val) or val == 0: return "Rp 0"
    return f"Rp {val:,.0f}".replace(",", ".")
def fmt_usd(val): return f"${val:,.2f}"

# --- ENGINE 1: DATA BINANCE (REAL-TIME 100%) ---
def get_binance_klines(symbol, interval, limit=200):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params)
        data = r.json()
        df = pd.DataFrame(data, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Vol', 'x', 'y', 'z', 'a', 'b', 'c'])
        df['Close'] = df['Close'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['Volume'] = df['Vol'].astype(float)
        # Convert Time
        df['Time'] = pd.to_datetime(df['Time'], unit='ms')
        return df
    except Exception as e:
        print(f"❌ Gagal Fetch Binance: {e}")
        return pd.DataFrame()

# --- ENGINE 2: INDIKATOR PRO (EMA 200 ADDED) ---
def add_indicators(df):
    df = df.copy()
    
    # 1. EMA 200 (Tren Jangka Panjang)
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # 2. Bollinger Bands (20, 2)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BBU'] = df['SMA20'] + (df['STD20'] * 2)
    df['BBL'] = df['SMA20'] - (df['STD20'] * 2)
    
    # 3. MACD
    k = df['Close'].ewm(span=12, adjust=False).mean()
    d = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = k - d
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 4. Stochastic RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    min_rsi = df['RSI'].rolling(window=14).min()
    max_rsi = df['RSI'].rolling(window=14).max()
    stoch = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)
    df['STOCHRSIk'] = stoch.rolling(window=3).mean() * 100
    
    return df

# --- ENGINE 3: FIBONACCI & POC ---
def get_analysis_levels(df):
    if df.empty: return {}, 0
    
    # Fibo
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    fibo = {
        "TOP (1.0)": high,
        "GOLDEN (0.618)": high - (diff * 0.618),
        "MID (0.5)": high - (diff * 0.5),
        "BOTTOM (0.0)": low
    }
    
    # POC (Point of Control / Volume Terbanyak)
    price_bins = pd.cut(df['Close'], bins=50)
    vpvr = df.groupby(price_bins, observed=True)['Volume'].sum()
    poc = vpvr.idxmax().mid
    
    return fibo, poc

# --- ENGINE 4: KURS FALLBACK SYSTEM ---
def get_kurs():
    print("⏳ Mengambil Kurs IDR...")
    try:
        # Coba Yahoo Finance
        t = yf.Ticker(TICKER_IDR)
        hist = t.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except: pass
    
    # Hardcode Fallback (Update berkala kalau perlu)
    print("⚠️ Yahoo Kurs Error. Pakai Fallback 16.500")
    return 16500.0

# --- CORE LOGIC: DYNAMIC DCA & REPORTING ---
def generate_strategy(df_d, df_4h, kurs):
    last = df_d.iloc[-1]
    fibo, poc = get_analysis_levels(df_d)
    
    # 1. VARIABLE KUNCI
    price = last['Close']
    ema200 = last['EMA200']
    
    # Logic Checks
    is_uptrend = price > ema200
    is_overbought = last['STOCHRSIk'] > 80
    is_oversold = last['STOCHRSIk'] < 20
    is_breakout = price > last['BBU']
    
    # 2. MATRIX 5 STATUS GENERATOR
    # (1) EMA 200 Status
    st_ema = "🟢 UPTREND" if is_uptrend else "🔴 DOWNTREND"
    val_ema = f"Price ${price:.0f} > EMA ${ema200:.0f}" if is_uptrend else f"Price ${price:.0f} < EMA ${ema200:.0f}"
    
    # (2) Stoch Status
    if is_overbought: st_stoch = "🔴 OVERBOUGHT (>80)"
    elif is_oversold: st_stoch = "🟢 OVERSOLD (<20)"
    else: st_stoch = "⚪ NEUTRAL"
    
    # (3) Bollinger Status
    if price > last['BBU']: st_bb = "🔴 BREAKOUT UPPER"
    elif price < last['BBL']: st_bb = "🟢 BREAKOUT LOWER"
    else: st_bb = "⚪ INSIDE BANDS"
    
    # (4) MACD Status
    if last['MACD'] > last['MACD_Signal']: st_macd = "🟢 BULLISH"
    else: st_macd = "🔴 BEARISH"
    
    # (5) Fibonacci Check
    # Cari support terdekat di bawah harga
    supports = [v for k,v in fibo.items() if v < price]
    nearest_support = max(supports) if supports else 0
    st_fibo = "⚪ ABOVE SUPPORT" if nearest_support > 0 else "⚠️ BOTTOM DISCOVERY"
    
    # 3. DECISION ENGINE (THE +1)
    mode = ""
    split_market = 0 
    split_limit = 0  
    msg = ""

    # Skenario 1: Strong Buy (Diskon)
    if is_oversold or price < last['BBL']:
        mode = "💎 STRONG BUY (DISCOUNT)"
        split_market, split_limit = 70, 30
        msg = "Indikator Jenuh Jual / Harga di Bawah BB."

    # Skenario 2: Normal Uptrend
    elif is_uptrend and not is_overbought:
        mode = "✅ NORMAL DCA (UPTREND)"
        split_market, split_limit = 50, 50
        msg = "Tren Naik Sehat (Above EMA 200). Lanjut SOP."

    # Skenario 3: Super Cycle (Ride Wave)
    elif is_uptrend and (is_overbought or is_breakout):
        mode = "🚀 RIDE THE WAVE (CAUTIOUS)"
        split_market, split_limit = 20, 80
        msg = "Pasar Panas (Overbought). Masuk dikit (20%), sisa antre bawah."

    # Skenario 4: Bearish Rejection
    elif not is_uptrend and is_overbought:
        mode = "🛑 WAIT / DEFENSIVE"
        split_market, split_limit = 0, 100
        msg = "Tren Turun & Mentok Atap. Jangan FOMO."
        
    else:
        mode = "⚪ NEUTRAL DCA"
        split_market, split_limit = 40, 60
        msg = "Pasar Ragu-ragu."

    # Hitung Duit
    rp_market = MODAL_GAJI * (split_market / 100)
    rp_limit = MODAL_GAJI * (split_limit / 100)
    
    # Target Limit (Smart Support)
    # Prioritas: EMA200 -> POC -> Fibo Golden -> Fibo Mid
    candidates = [ema200, poc, fibo['GOLDEN (0.618)'], fibo['MID (0.5)']]
    valid_supports = [x for x in candidates if x < price]
    target_limit_usd = max(valid_supports) if valid_supports else price * 0.95
    est_limit_idr = target_limit_usd * kurs * SPREAD_AJAIB

    # 4. FORMAT REPORT TEXT
    now = datetime.now(pytz.timezone('Asia/Jakarta'))
    
    report = f"""🦅 GOLD MASTER ULTIMATE (V3.0)
📅 {now.strftime('%d %b %Y | %H:%M WIB')}
🌍 DATA: BINANCE REAL-TIME
=======================================

💰 MARKET STATUS
PAXG/USD : {fmt_usd(price)}
KURS IDR : {fmt_idr(kurs)}
EMA 200  : {fmt_usd(ema200)} (Trend Filter)

📊 MATRIX 5 INDIKATOR (+1)
1. EMA 200     [{st_ema}]
   👉 {val_ema}

2. Stoch RSI   [{st_stoch}]
   👉 Value: {last['STOCHRSIk']:.1f}

3. Bollinger   [{st_bb}]
   👉 Upper: {fmt_usd(last['BBU'])} | Lower: {fmt_usd(last['BBL'])}

4. MACD        [{st_macd}]
   👉 Hist: {last['MACD'] - last['MACD_Signal']:.2f}

5. Fibonacci   [{st_fibo}]
   👉 Nearest Supp: {fmt_usd(nearest_support)}

=======================================
🧠 (+1) AI STRATEGY : [ {mode} ]
💡 ALASAN: {msg}
=======================================

📋 INSTRUKSI EKSEKUSI (MODAL 5 JUTA):

1️⃣ MARKET ORDER ({split_market}%)
   👉 Nominal: {fmt_idr(rp_market)}
   👉 Eksekusi: SEKARANG.

2️⃣ LIMIT ORDER ({split_limit}%)
   👉 Nominal: {fmt_idr(rp_limit)}
   👉 Pasang di: {fmt_usd(target_limit_usd)}
   👉 Est. IDR : {fmt_idr(est_limit_idr)}

=======================================
🎯 LEVEL PENTING (BINANCE DATA)
"""
    # Sorting Harga dari Tertinggi ke Terendah
    sorted_fibo = dict(sorted(fibo.items(), key=lambda item: item[1], reverse=True))
    for k, v in sorted_fibo.items():
        report += f"{k:<15}: {fmt_usd(v)}\n"
        
    return report

def send_telegram(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {"chat_id": chat_id, "text": message}
    try:
        requests.get(url, params=params)
        print("✅ Sent to Telegram.")
    except: pass

# --- MAIN ---
if __name__ == "__main__":
    print("🚀 Starting Gold Master Ultimate...")
    
    # 1. Get Secrets
    TOKEN = os.environ.get("TELEGRAM_TOKEN")
    CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
    
    # 2. Get Data
    kurs = get_kurs()
    df_d = get_binance_klines(SYMBOL_BINANCE, "1d", 300) # 300 hari buat EMA 200
    df_4h = get_binance_klines(SYMBOL_BINANCE, "4h", 100)
    
    if not df_d.empty:
        # 3. Process
        df_d = add_indicators(df_d)
        df_4h = add_indicators(df_4h)
        
        # 4. Generate Strategy
        report = generate_strategy(df_d, df_4h, kurs)
        
        print("\n" + report + "\n")
        
        if TOKEN and CHAT_ID:
            send_telegram(TOKEN, CHAT_ID, report)
    else:
        print("❌ Gagal connect Binance API.")
