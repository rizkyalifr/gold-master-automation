import os
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
import pytz

# --- KONFIGURASI ENGINE ---
TICKER_PAXG = "PAXG-USD" 
TICKER_IDR = "IDR=X"
MODAL_GAJI = 5000000 
SPREAD_AJAIB = 1.015 
PAXG_MULTIPLIER = 0.99048968

# --- HELPER FORMATTING ---
def fmt_idr(val): 
    if pd.isna(val) or val == 0: return "Rp 0"
    return f"Rp {val:,.0f}".replace(",", ".")

def fmt_usd(val): return f"${val:,.2f}"

# --- FUNGSI INDIKATOR (V6) ---
def process_data_smart(df):
    df = df.copy()
    # 1. EMA 200
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # 2. Bollinger Bands & Width
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BBU'] = df['SMA20'] + (df['STD20'] * 2)
    df['BBL'] = df['SMA20'] - (df['STD20'] * 2)
    df['BBM'] = df['SMA20']
    df['BB_Width'] = (df['BBU'] - df['BBL']) / df['BBM']
    
    # 3. MACD
    k = df['Close'].ewm(span=12, adjust=False).mean()
    d = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = k - d
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 4. Stoch RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    min_rsi = df['RSI'].rolling(window=14).min()
    max_rsi = df['RSI'].rolling(window=14).max()
    stoch = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)
    df['STOCHRSIk'] = stoch.rolling(window=3).mean() * 100
    df['STOCHRSId'] = df['STOCHRSIk'].rolling(window=3).mean()
    
    return df

# --- HITUNG POC ---
def get_poc(df):
    price_bins = pd.cut(df['Close'], bins=50)
    vpvr = df.groupby(price_bins, observed=True)['Volume'].sum()
    return vpvr.idxmax().mid

# --- FIBONACCI LEVELS ---
def calculate_fibonacci_levels(df):
    if df.empty: return {}
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    return {
        "MOONBAG (1.618)": high + (diff * 0.618),
        "RESISTANCE (High)": high,
        "0.236": high - (diff * 0.236),
        "0.382": high - (diff * 0.382),
        "MID (0.5)": high - (diff * 0.5),
        "GOLDEN (0.618)": high - (diff * 0.618),
        "0.786": high - (diff * 0.786),
        "FLOOR (Low)": low
    }

# --- SCORING ENGINE (V6: DETAIL + FLAGS) ---
def calculate_quant_score(row, prev_row, poc, fibo_golden):
    price = row['Close']
    scores = {}
    details = {}
    bullish_flags = 0

    # 1. EMA 200
    ema = row['EMA200']
    dist_pct = ((price - ema) / ema) * 100
    if dist_pct >= 12: score_ema = 100
    elif 8 <= dist_pct < 12: score_ema = 85
    elif 4 <= dist_pct < 8: score_ema = 70
    elif 0 <= dist_pct < 4: score_ema = 55
    else: score_ema = 30 
    scores['EMA'] = score_ema
    trend_txt = "BULLISH" if price > ema else "BEARISH"
    details['EMA'] = f"{trend_txt} | Price ${price:.0f} vs EMA ${ema:.0f} ({dist_pct:+.1f}%)"
    if score_ema >= 55: bullish_flags += 1

    # 2. VPVR POC
    if price > poc: score_vpvr = 100 
    elif abs(price - poc)/poc < 0.03: score_vpvr = 60 
    else: score_vpvr = 30 
    scores['VPVR'] = score_vpvr
    pos_txt = "Above Wall" if price > poc else "Below Wall"
    details['VPVR'] = f"{pos_txt} | POC: ${poc:.0f}"
    if score_vpvr >= 60: bullish_flags += 1

    # 3. MACD
    hist = row['MACD_Hist']
    prev_hist = prev_row['MACD_Hist']
    if hist > 0 and hist > prev_hist: score_macd = 100
    elif hist > 0: score_macd = 70
    elif hist < 0 and hist > prev_hist: score_macd = 50
    else: score_macd = 30
    scores['MACD'] = score_macd
    details['MACD'] = f"Hist: {hist:+.2f} | Line: {row['MACD']:.1f} | Sig: {row['MACD_Signal']:.1f}"
    if score_macd >= 70: bullish_flags += 1

    # 4. Stoch RSI
    k = row['STOCHRSIk']
    d = row['STOCHRSId']
    if k < 20 and k > d: score_stoch = 90
    elif k < 20: score_stoch = 65
    elif k > 80: score_stoch = 35
    else: score_stoch = 50
    scores['STOCH'] = score_stoch
    details['STOCH'] = f"K: {k:.1f} | D: {d:.1f}"
    if score_stoch >= 65: bullish_flags += 1

    # 5. Bollinger
    if price <= row['BBL']: score_bb = 85
    elif price < row['BBM']: score_bb = 65
    elif price < row['BBU']: score_bb = 35
    else: score_bb = 20
    scores['BB'] = score_bb
    details['BB'] = f"Upper: ${row['BBU']:.0f} | Lower: ${row['BBL']:.0f}"
    if score_bb >= 65: bullish_flags += 1

    # 6. Fibonacci
    dist_fibo_pct = abs((price - fibo_golden) / fibo_golden) * 100
    if dist_fibo_pct <= 1.5: score_fibo = 90
    elif price > fibo_golden: score_fibo = 55
    elif price < fibo_golden * 0.95: score_fibo = 40
    else: score_fibo = 75
    scores['FIBO'] = score_fibo
    details['FIBO'] = f"Dist Golden: {dist_fibo_pct:.1f}%"
    if score_fibo >= 55: bullish_flags += 1

    final_score = (
        (scores['EMA'] * 0.30) + (scores['VPVR'] * 0.20) + (scores['MACD'] * 0.15) +
        (scores['STOCH'] * 0.10) + (scores['BB'] * 0.10) + (scores['FIBO'] * 0.15)
    )
    
    return final_score, scores, details, bullish_flags

# --- DATA ENGINE ---
def get_data_engine():
    try:
        # Fetch Kurs (Single)
        idr_df = yf.download(TICKER_IDR, period="1d", progress=False)
        if not idr_df.empty:
            kurs = idr_df['Close'].iloc[-1]
            if isinstance(kurs, pd.Series): kurs = kurs.item()
            kurs = float(kurs) if kurs > 10000 else 16500.0
        else: kurs = 16500.0
    except: kurs = 16500.0

    print(f"⏳ Mengambil Data Market... Kurs: {fmt_idr(kurs)}")

    try:
        # 1. Fetch 2 TAHUN (Untuk EMA 200)
        df_full = yf.download(TICKER_PAXG, period="2y", interval="1d", progress=False)
        # Fetch 1 Bulan Hourly
        df_hourly = yf.download(TICKER_PAXG, period="1mo", interval="1h", progress=False)
        
        if df_full.empty or df_hourly.empty:
            print("❌ Gagal Download Data PAXG")
            return pd.DataFrame(), pd.DataFrame(), kurs

        paxg_d = df_full
        if isinstance(paxg_d.columns, pd.MultiIndex): paxg_d.columns = paxg_d.columns.get_level_values(0)
        
        paxg_h = df_hourly
        if isinstance(paxg_h.columns, pd.MultiIndex): paxg_h.columns = paxg_h.columns.get_level_values(0)

        # Kalibrasi
        for df in [paxg_d, paxg_h]:
            df['Close'] *= PAXG_MULTIPLIER
            df['High'] *= PAXG_MULTIPLIER
            df['Low'] *= PAXG_MULTIPLIER
            df['Open'] *= PAXG_MULTIPLIER

        # 2. HITUNG INDIKATOR
        paxg_d = process_data_smart(paxg_d)
        paxg_6mo = paxg_d.tail(180).copy()
        
        # 3. HOURLY
        paxg_4h = paxg_h.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        paxg_4h = process_data_smart(paxg_4h)
        
        return paxg_6mo, paxg_4h, kurs

    except Exception as e:
        print(f"❌ Error Data: {e}")
        return pd.DataFrame(), pd.DataFrame(), kurs

# --- SEND TELEGRAM ---
def send_telegram(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {"chat_id": chat_id, "text": message}
    try:
        r = requests.get(url, params=params)
        if r.status_code == 200: print("✅ Sent to Telegram!")
        else: print(f"❌ Failed: {r.text}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

# --- REPORT GENERATOR (V6 LOGIC + DETAIL) ---
def generate_sop_report(df_6mo, df_4h, kurs):
    last_d = df_6mo.iloc[-1]
    prev_d = df_6mo.iloc[-2]
    price = last_d['Close']
    
    fibo = calculate_fibonacci_levels(df_6mo) 
    poc = get_poc(df_6mo)
    
    # 1. QUANT SCORE
    final_score, scores, details, bullish_count = calculate_quant_score(last_d, prev_d, poc, fibo['GOLDEN (0.618)'])
    
    # 2. MARKET REGIME & VOLATILITY
    ema_dist = ((price - last_d['EMA200']) / last_d['EMA200']) * 100
    if ema_dist > 5: regime = "🐎 TRENDING BULLISH"
    elif ema_dist < -5: regime = "🐻 TRENDING BEARISH"
    else: regime = "🦀 RANGING / SIDEWAYS"
    
    bb_width = last_d.get('BB_Width', 0.1)
    is_high_vol = bb_width > 0.15
    vol_status = "⚡ HIGH VOLATILITY" if is_high_vol else "🌊 NORMAL/LOW VOL"
    
    # 3. SELL LOGIC (PARTIAL LADDER V6)
    action_type = "BUY"
    sell_reason = ""
    sell_pct = 0
    momentum_decay = last_d['MACD_Hist'] < prev_d['MACD_Hist'] and last_d['MACD_Hist'] > 0
    
    if final_score < 40: 
        action_type = "SELL"
        sell_reason = "Score < 40 (Defensive Exit)"
        sell_pct = 50 
    elif price >= fibo['MOONBAG (1.618)']:
        action_type = "SELL"
        sell_reason = "Moonbag Target (1.618)"
        sell_pct = 50
    elif price >= fibo['RESISTANCE (High)'] and momentum_decay:
        action_type = "SELL"
        sell_reason = "Resistance + Momentum Decay"
        sell_pct = 30
    elif price >= fibo['0.236'] and last_d['STOCHRSIk'] > 80:
        action_type = "SELL"
        sell_reason = "Overbought at Resistance"
        sell_pct = 20

    # 4. BUY LOGIC & SIZING (V6)
    dana_market = 0
    dana_limit = 0
    decision_title = ""
    prob_desc = ""
    agreement = f"{bullish_count}/6 Bullish"
    
    if action_type == "BUY":
        if final_score >= 80:
            alloc_market, alloc_limit = 0.7, 0.3
            decision_title = "🚀 AGGRESSIVE BUY"
            prob_desc = "High Probability Setup"
        elif 65 <= final_score < 80:
            alloc_market, alloc_limit = 0.5, 0.5
            decision_title = "✅ STANDARD ACCUMULATION"
            prob_desc = "Moderate/Healthy Trend"
        elif 50 <= final_score < 65:
            alloc_market, alloc_limit = 0.2, 0.8
            decision_title = "⚠️ SNIPER ENTRY (WAIT DIP)"
            prob_desc = "Price Extended / Wait Pullback"
        else: 
            alloc_market, alloc_limit = 0.0, 1.0
            decision_title = "🛡️ DEFENSIVE / WAIT"
            prob_desc = "Weak Structure"

        if is_high_vol and final_score >= 50:
            alloc_market *= 0.7 
            alloc_limit = 1.0 - alloc_market 
            prob_desc += " (Vol Adjusted)"

        dana_market = MODAL_GAJI * alloc_market
        dana_limit = MODAL_GAJI * alloc_limit

    # 5. SMART LIMIT (V6)
    candidates = [
        {'price': poc, 'label': 'POC'},
        {'price': last_d['EMA200'], 'label': 'EMA 200'},
        {'price': fibo['0.382'], 'label': 'Fibo 0.382'},
        {'price': fibo['MID (0.5)'], 'label': 'Fibo 0.5'},
        {'price': fibo['GOLDEN (0.618)'], 'label': 'Golden'}
    ]
    valid_supports = [c for c in candidates if c['price'] < price]
    
    if valid_supports:
        valid_supports.sort(key=lambda x: x['price'], reverse=True)
        target_limit_usd = valid_supports[0]['price']
        target_label = valid_supports[0]['label']
    else:
        target_limit_usd = fibo['MID (0.5)']
        target_label = "Deep Support"
        
    est_limit_idr = target_limit_usd * kurs * SPREAD_AJAIB

    # 6. TEXT REPORT
    now = datetime.now(pytz.timezone('Asia/Jakarta'))
    
    if action_type == "SELL":
        main_action_txt = f"""
🚨 **SELL SIGNAL TRIGGERED**
👉 **Action:** JUAL {sell_pct}% Posisi
👉 **Alasan:** {sell_reason}
👉 **Next:** Simpan Cash, tunggu Score membaik.
        """
    else:
        main_action_txt = f"""
1️⃣ **MARKET ORDER**
   👉 Nominal: {fmt_idr(dana_market)}
   👉 Eksekusi: SEKARANG.

2️⃣ **LIMIT ORDER**
   👉 Nominal: {fmt_idr(dana_limit)}
   👉 Target: {fmt_usd(target_limit_usd)} ({target_label})
   👉 Est. IDR: {fmt_idr(est_limit_idr)}
        """

    report = f"""🦅 GOLD MASTER V6 (HEADLESS QUANT)
📅 {now.strftime('%d %b %Y | %H:%M WIB')}
=======================================

💰 **MARKET DATA**
PAXG : {fmt_usd(price)}
EMA  : {fmt_usd(last_d['EMA200'])}
VOL  : {vol_status}

📊 **MATRIX 6 INDIKATOR**
1. EMA 200    [{scores['EMA']}] {details['EMA']}
2. VPVR POC   [{scores['VPVR']}] {details['VPVR']}
3. MACD       [{scores['MACD']}] {details['MACD']}
4. Stoch RSI  [{scores['STOCH']}] {details['STOCH']}
5. Bollinger  [{scores['BB']}] {details['BB']}
6. Fibonacci  [{scores['FIBO']}] {details['FIBO']}

🧮 SCORE: {final_score:.1f}/100 | {agreement}
🌍 REGIME: {regime}
=======================================
🧠 DECISION : [ {decision_title} ]
🎲 LOGIC    : {prob_desc}
=======================================

📋 **EXECUTION PLAN (MODAL 5 JUTA)**
{main_action_txt}

🎯 **KEY LEVELS**
"""
    sorted_fibo = dict(sorted(fibo.items(), key=lambda item: item[1], reverse=True))
    for k, v in sorted_fibo.items():
        report += f"{k:<15}: {fmt_usd(v)}\n"
        
    return report

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("🤖 Robot Start (Headless Quant)...")
    
    TOKEN = os.environ.get("TELEGRAM_TOKEN")
    CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
    
    df_6mo, df_4h, kurs = get_data_engine()
    
    if df_6mo.empty:
        print("❌ Data PAXG Kosong.")
    else:
        final_report = generate_sop_report(df_6mo, df_4h, kurs)
        
        print("\n" + "="*50)
        print(final_report)
        print("="*50 + "\n")

        if TOKEN and CHAT_ID:
            print("🚀 Sending to Telegram...")
            send_telegram(TOKEN, CHAT_ID, final_report)
        else:
            print("⚠️ Token Missing.")
