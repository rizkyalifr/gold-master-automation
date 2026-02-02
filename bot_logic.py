import os
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
import pytz

# --- KONFIGURASI ENGINE ---
TICKERS = ["PAXG-USD", "IDR=X"]
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

# --- DATA ENGINE (SMART SLICING) ---
def get_data_engine():
    print("⏳ Mengambil Data Market (2 Tahun untuk EMA -> Slice 6 Bulan)...")
    
    try:
        # 1. Fetch 2 TAHUN (Untuk EMA 200)
        df_full = yf.download(TICKERS, period="2y", interval="1d", group_by='ticker', progress=False)
        # Fetch 1 Bulan Hourly (Untuk Swinger Check)
        df_hourly = yf.download(TICKERS, period="1mo", interval="1h", group_by='ticker', progress=False)
        
        if df_full.empty or df_hourly.empty:
            print("❌ Gagal Download Data PAXG")
            return pd.DataFrame(), pd.DataFrame(), 16500.0

        if isinstance(df_full.columns, pd.MultiIndex):
            paxg_d = df_full['PAXG-USD'].dropna()
            paxg_h = df_hourly['PAXG-USD'].dropna()
            kurs = df_full['IDR=X']['Close'].iloc[-1]
        else:
            print("❌ Struktur Data Salah")
            return pd.DataFrame(), pd.DataFrame(), 16500.0

        # Kalibrasi Harga User
        for df in [paxg_d, paxg_h]:
            df['Close'] *= PAXG_MULTIPLIER
            df['High'] *= PAXG_MULTIPLIER
            df['Low'] *= PAXG_MULTIPLIER
            df['Open'] *= PAXG_MULTIPLIER

        # 2. HITUNG INDIKATOR DI DATA FULL (2 TAHUN) -> BIAR EMA VALID
        paxg_d = process_data_smart(paxg_d)
        
        # 3. POTONG DATA JADI 6 BULAN (SLICING) -> BUAT FIBO & VPVR
        # Kita ambil 180 candle terakhir (estimasi 6 bulan hari kalender / trading days)
        paxg_6mo = paxg_d.tail(180).copy()
        
        # Resample Hourly ke 4H
        paxg_4h = paxg_h.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        paxg_4h = process_data_smart(paxg_4h)
        
        return paxg_6mo, paxg_4h, kurs

    except Exception as e:
        print(f"❌ Error Data Processing: {e}")
        return pd.DataFrame(), pd.DataFrame(), 16500.0

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

# --- REPORT GENERATOR (SYNCED WITH APP.PY) ---
def generate_sop_report(df_6mo, df_4h, kurs):
    # Gunakan data 6 bulan untuk analisa Fibo & POC
    last_d = df_6mo.iloc[-1]
    last_4h = df_4h.iloc[-1]
    
    fibo = calculate_fibonacci_levels(df_6mo) 
    poc = get_poc(df_6mo) 
    
    # 1. MATRIX 6 INDIKATOR (FULL QUANTITATIVE)
    
    # EMA 200
    ema200 = last_d['EMA200']
    price = last_d['Close']
    if price > ema200: ema_stat = "🟢 UPTREND"
    else: ema_stat = "🔴 DOWNTREND"
    ema_dist_pct = ((price - ema200) / ema200) * 100
    ema_txt = f"Price ${price:.0f} vs EMA ${ema200:.0f} (Diff: {ema_dist_pct:+.1f}%)"

    # Stoch RSI
    stoch_k = last_d['STOCHRSIk']
    stoch_d = last_d['STOCHRSId']
    if stoch_k > 80: st_stat = "🔴 OVERBOUGHT"
    elif stoch_k < 20: st_stat = "🟢 OVERSOLD"
    else: st_stat = "⚪ NEUTRAL"
    stoch_txt = f"K: {stoch_k:.1f} | D: {stoch_d:.1f}"
    
    # MACD
    macd_line = last_d['MACD']
    macd_sig = last_d['MACD_Signal']
    hist = macd_line - macd_sig
    if hist > 0: mac_stat = "🟢 BULLISH"
    else: mac_stat = "🔴 BEARISH"
    mac_txt = f"Hist: {hist:+.2f} | Line: {macd_line:.2f}"
    
    # VPVR
    if price > poc: vp_stat = "🟢 STRONG (Above POC)"
    else: vp_stat = "🔴 WEAK (Below POC)"
    vp_txt = f"Price ${price:.0f} vs POC ${poc:.0f}"
    
    # Bollinger
    bbu = last_d['BBU']
    bbl = last_d['BBL']
    if price >= bbu: bb_stat = "🔴 BREAKOUT UPPER"
    elif price <= bbl: bb_stat = "🟢 BREAKOUT LOWER"
    else: bb_stat = "⚪ INSIDE BANDS"
    bb_txt = f"Upper: ${bbu:.0f} | Lower: ${bbl:.0f}"
    
    # Fibo
    dist_gold = price - fibo['GOLDEN (0.618)']
    if abs(dist_gold) < 20: fib_stat = "⚠️ TESTING GOLDEN"
    elif dist_gold > 0: fib_stat = "⚪ ABOVE SUPPORT"
    else: fib_stat = "🟢 DISCOUNT AREA"
    fib_txt = f"Dist to Golden: ${dist_gold:+.1f} (Target: ${fibo['GOLDEN (0.618)']:.0f})"

    # 2. DECISION LOGIC (SOP TANGGAL 25)
    is_swinger = (price > bbu) or (stoch_k > 80)
    
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
        candidates = [
            poc, 
            ema200, 
            fibo['0.382 (Shallow)'],
            fibo['MID (0.5)'], 
            fibo['GOLDEN (0.618)']
        ]
        
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
    
    report = f"""🦅 GOLD MASTER GUIDE (EMA 200 + QUANTITATIVE)
📅 Waktu: {now.strftime('%d %b %Y | %H:%M WIB')}
============================================================

💰 UPDATE HARGA (PAXG)
💵 KURS USD/IDR : {fmt_idr(kurs)}
------------------------------------------------------------
💎 PAXG/USD      : {fmt_usd(price)}
💎 EMA 200       : {fmt_usd(ema200)}
💎 PAXG/IDR      : {fmt_idr(price * kurs)}
------------------------------------------------------------

📊 MATRIX 6 INDIKATOR (QUANTITATIVE)
1. EMA 200     [{ema_stat}]
   👉 {ema_txt}

2. Stoch RSI   [{st_stat}]
   👉 {stoch_txt}

3. MACD        [{mac_stat}]
   👉 {mac_txt}

4. VPVR POC    [{vp_stat}]
   👉 {vp_txt}

5. Bollinger   [{bb_stat}]
   👉 {bb_txt}

6. Fibonacci   [{fib_stat}]
   👉 {fib_txt}

============================================================
🧠 KEPUTUSAN SOP : [ {decision} ]
🔐 KONDISI PASAR : {validation}
============================================================

📝 INSTRUKSI EKSEKUSI (MODAL 5 JUTA):
{action_txt}

🎯 MAPPING AREA (DATA 6 BULAN)
"""
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
