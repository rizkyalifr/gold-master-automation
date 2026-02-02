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

# --- FUNGSI INDIKATOR (ALL IN ONE) ---
def process_data_smart(df):
    df = df.copy()
    
    # 1. EMA 200
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # 2. Bollinger Bands
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BBU'] = df['SMA20'] + (df['STD20'] * 2)
    df['BBL'] = df['SMA20'] - (df['STD20'] * 2)
    df['BBM'] = df['SMA20']
    # Volatility Width
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

# --- HITUNG POC (VPVR) ---
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

# --- SCORING ENGINE (COMPREHENSIVE) ---
def calculate_quant_score(row, prev_row, poc, fibo_golden):
    price = row['Close']
    scores = {}
    details = {}
    bullish_flags = 0

    # 1. EMA 200 (Weight 30%)
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

    # 2. VPVR POC (Weight 20%)
    if price > poc: score_vpvr = 100 
    elif abs(price - poc)/poc < 0.03: score_vpvr = 60 
    else: score_vpvr = 30 
    
    scores['VPVR'] = score_vpvr
    pos_txt = "Above Wall" if price > poc else "Below Wall"
    details['VPVR'] = f"{pos_txt} | POC: ${poc:.0f}"
    if score_vpvr >= 60: bullish_flags += 1

    # 3. MACD (Weight 15%)
    hist = row['MACD_Hist']
    prev_hist = prev_row['MACD_Hist']
    if hist > 0 and hist > prev_hist: score_macd = 100
    elif hist > 0: score_macd = 70
    elif hist < 0 and hist > prev_hist: score_macd = 50
    else: score_macd = 30
    
    scores['MACD'] = score_macd
    line = row['MACD']
    sig = row['MACD_Signal']
    details['MACD'] = f"Hist: {hist:+.2f} | Line: {line:.1f} | Sig: {sig:.1f}"
    if score_macd >= 70: bullish_flags += 1

    # 4. Stoch RSI (Weight 10%)
    k = row['STOCHRSIk']
    d = row['STOCHRSId']
    if k < 20 and k > d: score_stoch = 90
    elif k < 20: score_stoch = 65
    elif k > 80: score_stoch = 35
    else: score_stoch = 50
    
    scores['STOCH'] = score_stoch
    details['STOCH'] = f"Value: {k:.1f} (D: {d:.1f})"
    if score_stoch >= 65: bullish_flags += 1

    # 5. Bollinger Bands (Weight 10%)
    if price <= row['BBL']: score_bb = 85
    elif price < row['BBM']: score_bb = 65
    elif price < row['BBU']: score_bb = 35
    else: score_bb = 20
    
    scores['BB'] = score_bb
    details['BB'] = f"Upper: ${row['BBU']:.0f} | Lower: ${row['BBL']:.0f}"
    if score_bb >= 65: bullish_flags += 1

    # 6. Fibonacci (Weight 15%)
    dist_fibo_pct = abs((price - fibo_golden) / fibo_golden) * 100
    if dist_fibo_pct <= 1.5: score_fibo = 90
    elif price > fibo_golden: score_fibo = 55
    elif price < fibo_golden * 0.95: score_fibo = 40
    else: score_fibo = 75
    
    scores['FIBO'] = score_fibo
    details['FIBO'] = f"Dist to Golden: {dist_fibo_pct:.1f}% | Target: ${fibo_golden:.0f}"
    if score_fibo >= 55: bullish_flags += 1

    # COMPOSITE SCORE
    final_score = (
        (scores['EMA'] * 0.30) + (scores['VPVR'] * 0.20) + (scores['MACD'] * 0.15) +
        (scores['STOCH'] * 0.10) + (scores['BB'] * 0.10) + (scores['FIBO'] * 0.15)
    )
    
    return final_score, scores, details, bullish_flags

# --- FETCH KURS ---
def get_kurs_idr():
    try:
        idr_df = yf.download(TICKER_IDR, period="1d", progress=False)
        if not idr_df.empty:
            kurs = idr_df['Close'].iloc[-1]
            if isinstance(kurs, pd.Series): kurs = kurs.item()
            return float(kurs) if kurs > 10000 else 16500.0
        return 16500.0
    except: return 16500.0

# --- DATA ENGINE (SMART SLICING) ---
def get_data_engine():
    kurs = get_kurs_idr()
    print(f"⏳ Mengambil Data Market (2 Tahun untuk EMA -> Slice 6 Bulan)... Kurs: {fmt_idr(kurs)}")

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

        # 2. HITUNG INDIKATOR DI DATA FULL (2 TAHUN)
        paxg_d = process_data_smart(paxg_d)
        
        # 3. POTONG DATA JADI 6 BULAN (SLICING) -> BUAT FIBO & VPVR
        paxg_6mo = paxg_d.tail(180).copy()
        
        # Resample Hourly ke 4H
        paxg_4h = paxg_h.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        paxg_4h = process_data_smart(paxg_4h)
        
        return paxg_6mo, paxg_4h, kurs

    except Exception as e:
        print(f"❌ Error Data Processing: {e}")
        return pd.DataFrame(), pd.DataFrame(), kurs

# --- FUNGSI KIRIM TELEGRAM ---
def send_telegram(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {"chat_id": chat_id, "text": message}
    try:
        r = requests.get(url, params=params)
        if r.status_code == 200: print("✅ Laporan Terkirim ke Telegram!")
        else: print(f"❌ Gagal Kirim: {r.text}")
    except Exception as e:
        print(f"❌ Error Koneksi Telegram: {e}")

# --- REPORT GENERATOR (V5.0: BUY & SELL LOGIC) ---
def generate_sop_report(df_6mo, df_4h, kurs):
    last_d = df_6mo.iloc[-1]
    prev_d = df_6mo.iloc[-2]
    
    fibo = calculate_fibonacci_levels(df_6mo) 
    poc = get_poc(df_6mo)
    
    # 1. HITUNG SKOR KUANTITATIF
    final_score, scores, details, bullish_count = calculate_quant_score(last_d, prev_d, poc, fibo['GOLDEN (0.618)'])
    
    # 2. DETEKSI MODE (Labeling)
    if final_score >= 65:
        market_mode = "✅ INVESTOR MODE (ACCUMULATION)"
    elif 50 <= final_score < 65:
        market_mode = "⚠️ SWINGER MODE (WAIT FOR DIP)"
    else:
        market_mode = "🛡️ DEFENSIVE MODE (CASH IS KING)"

    # 3. DETEKSI SINYAL JUAL (TAKE PROFIT) - NEW FEATURE!
    # Syarat Jual: Harga kena Resistance Tinggi ATAU Extreme Overbought
    take_profit_signal = False
    tp_reason = ""
    
    price = last_d['Close']
    if price >= fibo['MOONBAG (1.618)']:
        take_profit_signal = True
        tp_reason = "Hit Moonbag Target (1.618)"
    elif last_d['STOCHRSIk'] > 90 and price > last_d['BBU']:
        take_profit_signal = True
        tp_reason = "Extreme Overbought + Breakout BB"

    # 4. DECISION ENGINE (BUY vs SELL)
    dana_market, dana_limit = 0, 0
    target_limit_usd, est_limit_idr = 0, 0
    
    if take_profit_signal:
        decision = "💰 TAKE PROFIT / SELL"
        prob = "Exit Signal Triggered"
        action_txt = f"""
1. JUAL SEBAGIAN ASET (20-50%).
   👉 Alasan: {tp_reason}
   👉 Amankan profit dalam bentuk Cash/USDT.

2. JANGAN BELI DULU (Tunggu Koreksi).
        """
    elif final_score >= 80:
        decision = "🚀 STRONG BUY"
        prob = "High (>80%) - Aggressive Entry"
        dana_market, dana_limit = MODAL_GAJI * 0.7, MODAL_GAJI * 0.3
    elif 65 <= final_score < 80:
        decision = "✅ GRADUAL BUY"
        prob = "Moderate (65-79%) - Standard Entry"
        dana_market, dana_limit = MODAL_GAJI * 0.5, MODAL_GAJI * 0.5
    elif 50 <= final_score < 65:
        decision = "⚠️ WAIT PULLBACK"
        prob = "Low (Wait Dip) - Sniper Entry"
        dana_market, dana_limit = MODAL_GAJI * 0.2, MODAL_GAJI * 0.8
    else:
        decision = "🛑 AVOID ENTRY"
        prob = "Negative - Save Cash"
        dana_market, dana_limit = 0, MODAL_GAJI

    # 5. SUSUN TEKS INSTRUKSI (Jika Bukan Sinyal Jual)
    if not take_profit_signal:
        # Target Limit Logic
        candidates = [poc, last_d['EMA200'], fibo['0.382'], fibo['MID (0.5)'], fibo['GOLDEN (0.618)']]
        valid_supports = [x for x in candidates if x < last_d['Close']]
        target_limit_usd = max(valid_supports) if valid_supports else fibo['MID (0.5)']
        est_limit_idr = target_limit_usd * kurs * SPREAD_AJAIB
        
        action_txt = f"""
1. MARKET ORDER
   👉 Nominal: {fmt_idr(dana_market)}
   👉 Eksekusi: SEKARANG.

2. LIMIT ORDER
   👉 Nominal: {fmt_idr(dana_limit)}
   👉 Pasang di: {fmt_usd(target_limit_usd)}
   👉 Est. IDR : {fmt_idr(est_limit_idr)}
        """

    # 6. INFO TAMBAHAN
    ema_dist = ((last_d['Close'] - last_d['EMA200']) / last_d['EMA200']) * 100
    regime = "TRENDING" if abs(ema_dist) > 5 else "RANGING"
    agreement = f"{bullish_count}/6 Bullish"

    now = datetime.now(pytz.timezone('Asia/Jakarta'))
    
    report = f"""🦅 GOLD MASTER QUANTITATIVE (V5.0 COMPLETE)
📅 Waktu: {now.strftime('%d %b %Y | %H:%M WIB')}
=======================================

💰 UPDATE HARGA
PAXG/USD : {fmt_usd(last_d['Close'])}
KURS IDR : {fmt_idr(kurs)}

📊 MATRIX 6 INDIKATOR
1. EMA 200    [{scores['EMA']}] {details['EMA']}
2. VPVR POC   [{scores['VPVR']}] {details['VPVR']}
3. MACD       [{scores['MACD']}] {details['MACD']}
4. Stoch RSI  [{scores['STOCH']}] {details['STOCH']}
5. Bollinger  [{scores['BB']}] {details['BB']}
6. Fibonacci  [{scores['FIBO']}] {details['FIBO']}

🧮 SCORE: {final_score:.1f}/100 | {agreement}
🌍 MODE : {market_mode}
=======================================
🧠 KEPUTUSAN : [ {decision} ]
🎲 STATUS    : {prob}
=======================================

📋 INSTRUKSI EKSEKUSI (MODAL 5 JUTA):
{action_txt}

🎯 MAPPING AREA FIBONACCI
"""
    sorted_fibo = dict(sorted(fibo.items(), key=lambda item: item[1], reverse=True))
    for k, v in sorted_fibo.items():
        report += f"{k:<15}: {fmt_usd(v)}\n"
        
    return report

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("🤖 Robot Start (Quantitative Mode)...")
    
    TOKEN = os.environ.get("TELEGRAM_TOKEN")
    CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
    
    df_6mo, df_4h, kurs = get_data_engine()
    
    if df_6mo.empty:
        print("❌ Data PAXG Kosong. Sinyal Internet?")
    else:
        final_report = generate_sop_report(df_6mo, df_4h, kurs)
        
        print("\n" + "="*50)
        print(final_report)
        print("="*50 + "\n")

        if TOKEN and CHAT_ID:
            print("🚀 Mengirim ke Telegram...")
            send_telegram(TOKEN, CHAT_ID, final_report)
        else:
            print("⚠️ Token Telegram Kosong.")

