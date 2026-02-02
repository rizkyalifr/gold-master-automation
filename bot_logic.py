import os
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
import pytz

# --- KONFIGURASI ENGINE ---
TICKER_PAXG = "PAXG-USD" # Download terpisah
TICKER_IDR = "IDR=X"     # Download terpisah
MODAL_GAJI = 5000000     # 5 Juta Rupiah
SPREAD_AJAIB = 1.015 
PAXG_MULTIPLIER = 0.99048968

# --- HELPER FORMATTING (ANTI ERROR) ---
def fmt_idr(val): 
    # Cek jika value NaN atau invalid
    if pd.isna(val) or val == float('inf'):
        return "Rp 0"
    return f"Rp {val:,.0f}".replace(",", ".")

def fmt_usd(val): return f"${val:,.2f}"

# --- FUNGSI INDIKATOR (PANDAS ONLY) ---
def add_indicators(df):
    df = df.copy()
    
    # 1. MACD (12, 26, 9)
    k = df['Close'].ewm(span=12, adjust=False).mean()
    d = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = k - d
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 2. Bollinger Bands (20, 2)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BBU'] = df['SMA20'] + (df['STD20'] * 2) # Upper
    df['BBL'] = df['SMA20'] - (df['STD20'] * 2) # Lower
    df['BBM'] = df['SMA20'] # Middle
    
    # 3. Stochastic RSI (14, 14, 3, 3)
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

# --- HITUNG POC (VPVR SIMPLIFIED) ---
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

# --- KHUSUS FETCH KURS (FIXED) ---
def get_kurs_idr():
    try:
        # Download Kurs Saja (Hanya hari ini)
        print("⏳ Mengambil Data Kurs IDR...")
        idr_df = yf.download(TICKER_IDR, period="1d", progress=False)
        
        if not idr_df.empty:
            # Ambil data terakhir
            kurs = idr_df['Close'].iloc[-1]
            
            # Jika formatnya Series/item, ambil valuenya saja
            if isinstance(kurs, pd.Series):
                kurs = kurs.item()
            
            # Cek Validitas
            if pd.isna(kurs) or kurs < 10000:
                print("⚠️ Data Kurs Aneh/NaN. Pakai Fallback 16.800")
                return 16800.0
                
            return float(kurs)
        else:
            print("⚠️ Data Kurs Kosong. Pakai Fallback 16.800")
            return 16800.0
            
    except Exception as e:
        print(f"⚠️ Error Ambil Kurs: {e}. Pakai Fallback 16.800")
        return 16800.0 # Hardcode Aman

# --- DATA ENGINE (DUAL TIMEFRAME) ---
def get_data_engine():
    # 1. Ambil Kurs Dulu (Terpisah)
    kurs = get_kurs_idr()
    print(f"✅ Kurs Terpakai: {fmt_idr(kurs)}")

    print("⏳ Mengambil Data PAXG (6 Bulan Daily & 1 Bulan Hourly)...")
    
    try:
        # 2. Fetch PAXG Sendirian (Lebih Stabil)
        df_daily = yf.download(TICKER_PAXG, period="6mo", interval="1d", progress=False)
        df_hourly = yf.download(TICKER_PAXG, period="1mo", interval="1h", progress=False)
        
        if df_daily.empty or df_hourly.empty:
            print("❌ Gagal Download Data PAXG")
            return pd.DataFrame(), pd.DataFrame(), kurs

        paxg_d = df_daily
        paxg_h = df_hourly

        # Fix MultiIndex Column issue (Yfinance update baru sering bikin ini)
        if isinstance(paxg_d.columns, pd.MultiIndex):
            paxg_d.columns = paxg_d.columns.get_level_values(0)
        if isinstance(paxg_h.columns, pd.MultiIndex):
            paxg_h.columns = paxg_h.columns.get_level_values(0)

        # Kalibrasi Harga User
        for df in [paxg_d, paxg_h]:
            df['Close'] *= PAXG_MULTIPLIER
            df['High'] *= PAXG_MULTIPLIER
            df['Low'] *= PAXG_MULTIPLIER
            df['Open'] *= PAXG_MULTIPLIER

        # Indikator
        paxg_d = add_indicators(paxg_d)
        
        # Resample Hourly ke 4H
        paxg_4h = paxg_h.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        paxg_4h = add_indicators(paxg_4h)
        
        return paxg_d, paxg_4h, kurs

    except Exception as e:
        print(f"❌ Error Data Processing: {e}")
        return pd.DataFrame(), pd.DataFrame(), kurs

# --- FUNGSI KIRIM TELEGRAM ---
def send_telegram(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {"chat_id": chat_id, "text": message}
    try:
        r = requests.get(url, params=params)
        if r.status_code == 200:
            print("✅ Laporan Terkirim ke Telegram!")
        else:
            print(f"❌ Gagal Kirim: {r.text}")
    except Exception as e:
        print(f"❌ Error Koneksi Telegram: {e}")

# --- REPORT GENERATOR (THE BRAIN) ---
def generate_sop_report(df_d, df_4h, kurs):
    last_d = df_d.iloc[-1]
    last_4h = df_4h.iloc[-1]
    
    # 1. TENTUKAN SKENARIO (DAILY)
    is_swinger = (last_d['Close'] > last_d['BBU']) or (last_d['STOCHRSIk'] > 80)
    
    fibo = calculate_fibonacci_levels(df_d)
    poc = get_poc(df_d)
    
    # --- MATRIX 5-5 STATUS & ANGKA ---
    stoch_val = last_d['STOCHRSIk']
    if stoch_val > 80: st_stat = "🔴 OVERBOUGHT"
    elif stoch_val < 20: st_stat = "🟢 OVERSOLD"
    else: st_stat = "⚪ NEUTRAL"
    
    macd_val = last_d['MACD']
    sig_val = last_d['MACD_Signal']
    if macd_val > sig_val: mac_stat = "🟢 BULLISH"
    else: mac_stat = "🔴 BEARISH"
    
    if last_d['Close'] > poc: vp_stat = "🟢 STRONG (Above POC)"
    else: vp_stat = "🔴 WEAK (Below POC)"
    
    bb_upper = last_d['BBU']
    bb_lower = last_d['BBL']
    if last_d['Close'] >= bb_upper: bb_stat = "🔴 AT UPPER BAND"
    elif last_d['Close'] <= bb_lower: bb_stat = "🟢 AT LOWER BAND"
    else: bb_stat = "⚪ INSIDE BANDS"
    
    dist_gold = last_d['Close'] - fibo['GOLDEN (0.618)']
    if abs(dist_gold) < 20: fib_stat = "⚠️ TESTING GOLDEN"
    elif dist_gold > 0: fib_stat = "⚪ ABOVE SUPPORT"
    else: fib_stat = "🟢 DISCOUNT AREA"

    # --- KEPUTUSAN SOP ---
    if is_swinger:
        decision = "🚨 SKENARIO 1: SWINGER MODE"
        validation = "Pasar Gejolak / Pucuk. TAHAN CASH."
        action_txt = f"""
1. JANGAN MASUK DULU.
2. Pantau Stoch RSI 4H (Saat ini: {last_4h['STOCHRSIk']:.1f}).
3. Tunggu Stoch 4H < 20 baru MARKET ORDER.
4. Jual Sebagian Aset Lama jika kena ${fibo['MOONBAG (1.618)']:.2f}
        """
    else:
        decision = "✅ SKENARIO 2: INVESTOR MODE"
        validation = "Pasar Stabil / Diskon. MASUK."
        
        # Hitung Split
        dana_market = MODAL_GAJI * 0.5
        dana_limit = MODAL_GAJI * 0.5
        
        # Target Limit
        limit_target = max(poc, fibo['GOLDEN (0.618)'])
        if limit_target >= last_d['Close']: limit_target = fibo['MID (0.5)']
        
        # Kalkulasi Estimasi (Anti NaN)
        harga_paxg_idr = last_d['Close'] * kurs * SPREAD_AJAIB
        est_market = dana_market / harga_paxg_idr if harga_paxg_idr > 0 else 0
        
        est_limit_idr = limit_target * kurs * SPREAD_AJAIB
        
        action_txt = f"""
1. MARKET ORDER (50%): Rp {dana_market:,.0f}
   (Estimasi dapat: {est_market:.4f} PAXG)

2. LIMIT ORDER (50%): Rp {dana_limit:,.0f}
   @ Harga ${limit_target:.2f} (Est. IDR: {fmt_idr(est_limit_idr)})
   
3. HOLD SELAMANYA (Akumulasi).
        """

    now = datetime.now(pytz.timezone('Asia/Jakarta'))
    
    report = f"""🦅 GOLD INVESTMENT GUIDE
📅 Waktu: {now.strftime('%d %b %Y | %H:%M WIB')}
============================================================

💰 UPDATE HARGA (PAXG)
💵 KURS USD/IDR : {fmt_idr(kurs)}
------------------------------------------------------------
💎 PAXG/USD      : {fmt_usd(last_d['Close'])}
💎 PAXG/IDR      : {fmt_idr(last_d['Close'] * kurs)}
   *(Est. Ajaib    : {fmt_idr(last_d['Close'] * kurs * SPREAD_AJAIB)})*
------------------------------------------------------------

📊 HASIL ANALISIS (MATRIX 5-5)
1. Stoch RSI   [{st_stat}]
   👉 Value: {stoch_val:.2f} (D: {last_d['STOCHRSId']:.2f})

2. MACD        [{mac_stat}]
   👉 Histogram: {macd_val - sig_val:.4f} (Line: {macd_val:.2f})

3. VPVR POC    [{vp_stat}]
   👉 POC Price: ${poc:.2f}

4. Bollinger   [{bb_stat}]
   👉 Upper: ${bb_upper:.2f} | Lower: ${bb_lower:.2f}

5. Fibonacci   [{fib_stat}]
   👉 Golden Pkt: ${fibo['GOLDEN (0.618)']:.2f}

============================================================
🧠 KEPUTUSAN SOP : [ {decision} ]
🔐 KONDISI PASAR : {validation}
============================================================

📝 INSTRUKSI EKSEKUSI (MODAL 5 JUTA):
{action_txt}

🎯 MAPPING AREA (SORTED BY PRICE)
"""
    sorted_fibo = dict(sorted(fibo.items(), key=lambda item: item[1], reverse=True))

    for name, val in sorted_fibo.items():
        paxg_idr = val * kurs * SPREAD_AJAIB
        report += f"{name:<20} : {fmt_usd(val)} | {fmt_idr(paxg_idr)}\n"

    return report

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("🤖 Robot Start (SOP Tanggal 25 Mode)...")
    
    # 1. Secrets
    try:
        TOKEN = os.environ.get("TELEGRAM_TOKEN")
        CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
    except KeyError:
        pass

    # 2. Jalanin Analisis
    df_d, df_4h, kurs = get_data_engine()
    
    if df_d.empty:
        print("❌ Data PAXG Kosong. Sinyal Internet?")
    else:
        # 3. Generate Report
        final_report = generate_sop_report(df_d, df_4h, kurs)
        
        # 4. Print Log
        print("\n" + "="*50)
        print(final_report)
        print("="*50 + "\n")

        # 5. Kirim Telegram
        if TOKEN and CHAT_ID:
            print("🚀 Mengirim ke Telegram...")
            send_telegram(TOKEN, CHAT_ID, final_report)
        else:
            print("⚠️ Token Telegram Kosong / Tidak di-set di Environment Variable.")
