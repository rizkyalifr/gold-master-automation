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
    
    # 1. HITUNG EMA 200 (Pakai Data Full 2 Tahun)
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # 2. HITUNG Indikator Lain
    # Bollinger
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BBU'] = df['SMA20'] + (df['STD20'] * 2)
    df['BBL'] = df['SMA20'] - (df['STD20'] * 2)
    
    # MACD
    k = df['Close'].ewm(span=12, adjust=False).mean()
    d = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = k - d
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Stochastic RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    min_rsi = df['RSI'].rolling(window=14).min()
    max_rsi = df['RSI'].rolling(window=14).max()
    stoch_rsi = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)
    df['STOCHRSIk'] = stoch_rsi.rolling(window=3).mean() * 100
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
    # Fibo dihitung DARI DATA 6 BULAN SAJA (Biar akurat ke range sekarang)
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    levels = {
        "MOONBAG (1.618)": high + (diff * 0.618),
        "RESISTANCE (High)": high,
        "0.236 (Pullback)": high - (diff * 0.236),
        "0.382 (Shallow)": high - (diff * 0.382),
        "MID (0.5)": high - (diff * 0.5),
        "GOLDEN (0.618)": high - (diff * 0.618),
        "0.786 (Deep)": high - (diff * 0.786),
        "FLOOR (Low)": low
    }
    return levels

# --- FETCH KURS (FIXED) ---
def get_kurs_idr():
    try:
        idr_df = yf.download(TICKER_IDR, period="1d", progress=False)
        if not idr_df.empty:
            kurs = idr_df['Close'].iloc[-1]
            if isinstance(kurs, pd.Series): kurs = kurs.item()
            if pd.isna(kurs) or kurs < 10000: return 16500.0
            return float(kurs)
        else: return 16500.0
    except: return 16500.0

# --- DATA ENGINE (SMART SLICING) ---
def get_data_engine():
    # 1. Ambil Kurs
    kurs = get_kurs_idr()
    print(f"⏳ Mengambil Data Market (2 Tahun untuk EMA -> Slice 6 Bulan)... Kurs: {fmt_idr(kurs)}")

    try:
        # 2. Fetch 2 TAHUN (Untuk EMA 200)
        df_full = yf.download(TICKER_PAXG, period="2y", interval="1d", progress=False)
        # Fetch 1 Bulan Hourly (Untuk Swinger Check)
        df_hourly = yf.download(TICKER_PAXG, period="1mo", interval="1h", progress=False)
        
        if df_full.empty or df_hourly.empty:
            print("❌ Gagal Download Data PAXG")
            return pd.DataFrame(), pd.DataFrame(), kurs

        paxg_d = df_full
        paxg_h = df_hourly

        # Fix MultiIndex
        if isinstance(paxg_d.columns, pd.MultiIndex): paxg_d.columns = paxg_d.columns.get_level_values(0)
        if isinstance(paxg_h.columns, pd.MultiIndex): paxg_h.columns = paxg_h.columns.get_level_values(0)

        # Kalibrasi
        for df in [paxg_d, paxg_h]:
            df['Close'] *= PAXG_MULTIPLIER
            df['High'] *= PAXG_MULTIPLIER
            df['Low'] *= PAXG_MULTIPLIER
            df['Open'] *= PAXG_MULTIPLIER

        # 3. HITUNG INDIKATOR DI DATA FULL (2 TAHUN)
        paxg_d = process_data_smart(paxg_d)
        
        # 4. POTONG DATA JADI 6 BULAN (SLICING) -> BUAT FIBO & VPVR
        paxg_6mo = paxg_d.tail(180).copy()
        
        # Resample Hourly ke 4H
        paxg_4h = paxg_h.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
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

# --- REPORT GENERATOR ---
def generate_sop_report(df_6mo, df_4h, kurs):
    # Gunakan data 6 bulan untuk analisa Fibo & POC
    last_d = df_6mo.iloc[-1]
    last_4h = df_4h.iloc[-1]
    
    fibo = calculate_fibonacci_levels(df_6mo) 
    poc = get_poc(df_6mo) 
    
    # 1. MATRIX 6 INDIKATOR
    ema200 = last_d['EMA200']
    price = last_d['Close']
    if price > ema200: ema_stat = "🟢 UPTREND"
    else: ema_stat = "🔴 DOWNTREND"

    stoch_val = last_d['STOCHRSIk']
    if stoch_val > 80: st_stat = "🔴 OVERBOUGHT"
    elif stoch_val < 20: st_stat = "🟢 OVERSOLD"
    else: st_stat = "⚪ NEUTRAL"
    
    if last_d['MACD'] > last_d['MACD_Signal']: mac_stat = "🟢 BULLISH"
    else: mac_stat = "🔴 BEARISH"
    
    if last_d['Close'] > poc: vp_stat = "🟢 STRONG (Above POC)"
    else: vp_stat = "🔴 WEAK (Below POC)"
    
    if last_d['Close'] >= last_d['BBU']: bb_stat = "🔴 BREAKOUT UPPER"
    elif last_d['Close'] <= last_d['BBL']: bb_stat = "🟢 BREAKOUT LOWER"
    else: bb_stat = "⚪ INSIDE BANDS"
    
    dist_gold = last_d['Close'] - fibo['GOLDEN (0.618)']
    if abs(dist_gold) < 20: fib_stat = "⚠️ TESTING GOLDEN"
    elif dist_gold > 0: fib_stat = "⚪ ABOVE SUPPORT"
    else: fib_stat = "🟢 DISCOUNT AREA"

    # 2. DECISION LOGIC (SOP TANGGAL 25)
    is_swinger = (price > last_d['BBU']) or (stoch_val > 80)
    
    if is_swinger:
        decision = "🚨 SKENARIO 1: SWINGER MODE"
        validation = "Pasar Gejolak / Pucuk. TAHAN CASH."
        action_txt = f"""
1. JANGAN MASUK DULU.
2. Pantau Stoch RSI 4H (Saat ini: {last_4h['STOCHRSIk']:.1f}).
3. Tunggu Stoch 4H < 20 baru MARKET ORDER.
        """
    else:
        decision = "✅ SKENARIO 2: INVESTOR MODE"
        validation = "Pasar Stabil / Diskon. MASUK."
        
        dana_market = MODAL_GAJI * 0.5
        dana_limit = MODAL_GAJI * 0.5
        
        # --- LOGIC 100/100: SMART AGGRESSIVE LIMIT ---
        # Masukkan SEMUA level support potensial termasuk Fibo Dnagkal (0.382 & 0.5)
        candidates = [
            poc, 
            ema200, 
            fibo['0.382 (Shallow)'], # Biar gak ketinggalan kalau trend kuat
            fibo['MID (0.5)'], 
            fibo['GOLDEN (0.618)']
        ]
        
        # Ambil support TERTINGGI yang masih di bawah harga sekarang
        valid_supports = [x for x in candidates if x < price]
        
        if valid_supports: limit_target = max(valid_supports)
        else: limit_target = fibo['MID (0.5)'] # Fallback

        est_market = dana_market / (price * kurs * SPREAD_AJAIB)
        est_limit_idr = limit_target * kurs * SPREAD_AJAIB
        
        action_txt = f"""
1. MARKET ORDER (50%): Rp {dana_market:,.0f}
   (Estimasi dapat: {est_market:.4f} PAXG)

2. LIMIT ORDER (50%): Rp {dana_limit:,.0f}
   @ Harga ${limit_target:.2f} (Est. IDR: {fmt_idr(est_limit_idr)})
   *(Target: Support Terdekat 0.382/0.5/POC/EMA)*
        """

    now = datetime.now(pytz.timezone('Asia/Jakarta'))
    
    report = f"""🦅 GOLD MASTER GUIDE (EMA 200 + SMART AGGRESSIVE)
📅 Waktu: {now.strftime('%d %b %Y | %H:%M WIB')}
============================================================

💰 UPDATE HARGA (PAXG)
💵 KURS USD/IDR : {fmt_idr(kurs)}
------------------------------------------------------------
💎 PAXG/USD      : {fmt_usd(price)}
💎 EMA 200       : {fmt_usd(ema200)}
💎 PAXG/IDR      : {fmt_idr(price * kurs)}
   *(Est. Ajaib    : {fmt_idr(price * kurs * SPREAD_AJAIB)})*
------------------------------------------------------------

📊 MATRIX 6 INDIKATOR
1. EMA 200     [{ema_stat}]
   👉 Price ${price:.0f} vs EMA ${ema200:.0f}

2. Stoch RSI   [{st_stat}]
   👉 Value: {stoch_val:.2f}

3. MACD        [{mac_stat}]

4. VPVR POC    [{vp_stat}]
   👉 POC Price: ${poc:.2f} (Area 6 Bulan)

5. Bollinger   [{bb_stat}]

6. Fibonacci   [{fib_stat}]
   👉 Golden Pkt: ${fibo['GOLDEN (0.618)']:.2f}

============================================================
🧠 KEPUTUSAN SOP : [ {decision} ]
🔐 KONDISI PASAR : {validation}
============================================================

📝 INSTRUKSI EKSEKUSI (MODAL 5 JUTA):
{action_txt}

🎯 MAPPING AREA (DATA 6 BULAN)
"""
    # Sorting Harga dari Tertinggi ke Terendah
    sorted_fibo = dict(sorted(fibo.items(), key=lambda item: item[1], reverse=True))

    for name, val in sorted_fibo.items():
        paxg_idr = val * kurs * SPREAD_AJAIB
        report += f"{name:<20} : {fmt_usd(val)} | {fmt_idr(paxg_idr)}\n"

    return report

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("🤖 Robot Start (Headless Mode)...")
    
    TOKEN = os.environ.get("TELEGRAM_TOKEN")
    CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
    
    # 1. Fetch & Process
    df_6mo, df_4h, kurs = get_data_engine()
    
    if df_6mo.empty:
        print("❌ Data PAXG Kosong. Sinyal Internet?")
    else:
        # 2. Generate
        final_report = generate_sop_report(df_6mo, df_4h, kurs)
        
        # 3. Print
        print("\n" + "="*50)
        print(final_report)
        print("="*50 + "\n")

        # 4. Kirim
        if TOKEN and CHAT_ID:
            print("🚀 Mengirim ke Telegram...")
            send_telegram(TOKEN, CHAT_ID, final_report)
        else:
            print("⚠️ Token Telegram Kosong.")
