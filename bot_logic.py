import os
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
import pytz

# --- KONFIGURASI ENGINE ---
# ✅ HAPUS GC=F. Cuma ambil PAXG dan KURS.
TICKERS = ["PAXG-USD", "IDR=X"]
INTERVAL = "1h"
PERIOD = "1mo"
SPREAD_AJAIB = 1.015 
CALIBRATION_FACTOR = 0.99048968 # Faktor kalibrasi manual dari user

# --- HELPER FORMATTING ---
def fmt_idr(val): return f"Rp {val:,.0f}".replace(",", ".")
def fmt_usd(val): return f"${val:,.2f}"

# --- FUNGSI INDIKATOR MANUAL (PENGGANTI PANDAS_TA) ---
def add_manual_indicators(df):
    df = df.copy()
    
    # 1. MACD (12, 26, 9)
    k = df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
    d = df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
    df['MACD'] = k - d
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()
    
    # 2. Bollinger Bands (20, 2)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BBU'] = df['SMA20'] + (df['STD20'] * 2) # Upper
    df['BBL'] = df['SMA20'] - (df['STD20'] * 2) # Lower
    
    # 3. Stochastic RSI (14, 14, 3, 3)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Hitung Stoch
    min_rsi = df['RSI'].rolling(window=14).min()
    max_rsi = df['RSI'].rolling(window=14).max()
    stoch_rsi = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)
    df['STOCHRSIk'] = stoch_rsi.rolling(window=3).mean() * 100
    df['STOCHRSId'] = df['STOCHRSIk'].rolling(window=3).mean()
    
    return df

# --- FUNGSI GET DATA (HEADLESS VERSION) ---
def get_data_engine():
    # Download cuma 2 Ticker (PAXG & IDR)
    df = yf.download(TICKERS, period=PERIOD, interval=INTERVAL, group_by='ticker', progress=False, threads=False)
    
    try:
        # 1. Ambil Data PAXG & Terapkan Kalibrasi
        if isinstance(df.columns, pd.MultiIndex):
            # Menggunakan Multiplier 0.99048968 sesuai request
            paxg = df['PAXG-USD'].dropna() * CALIBRATION_FACTOR
        else:
            # Fallback untuk single index
            paxg = df * CALIBRATION_FACTOR
        
        # 2. 🔥 JURUS CERMIN: XAU DIANGGAP SAMA DENGAN PAXG (YANG SUDAH DIKALIBRASI) 🔥
        xau = paxg.copy() 
        
        # 3. Ambil Kurs IDR
        if isinstance(df.columns, pd.MultiIndex):
            kurs_raw = df['IDR=X']['Close'].dropna()
        else:
            kurs_raw = pd.Series([16800]) # Default fallback

        kurs = kurs_raw.iloc[-1] if not kurs_raw.empty else 16800

    except Exception as e:
        print(f"❌ Error Data Fetching: {e}")
        return pd.DataFrame(), pd.DataFrame(), 16800
        
    if kurs < 10000: kurs = 16800
    return xau, paxg, kurs

def calculate_fibonacci_levels(df):
    if df.empty: return {}
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    levels = {
        "MOONBAG (1.618)": high + (diff * 0.618),
        "RESISTANCE (High)": high,
        "GOLDEN POCKET (0.618)": high - (diff * 0.618),
        "FLOOR (Low)": low
    }
    return levels

def send_telegram(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {"chat_id": chat_id, "text": message}
    try:
        requests.get(url, params=params)
        print("✅ Pesan Terkirim ke Telegram!")
    except Exception as e:
        print(f"❌ Gagal Kirim: {e}")

# --- LOGIC ANALYSIS & REPORT ---
def generate_bot_report(xau, paxg, kurs):
    if xau.empty: return "Data Kosong", "WAIT"

    # Analisa teknikal menggunakan data PAXG (via variabel xau)
    xau = add_manual_indicators(xau)
    
    # VPVR Logic
    price_bins = pd.cut(xau['Close'], bins=50)
    vpvr = xau.groupby(price_bins, observed=True)['Volume'].sum()
    poc = vpvr.idxmax().mid
    
    xau_fib = calculate_fibonacci_levels(xau)
    paxg_fib = calculate_fibonacci_levels(paxg)

    last_xau = xau.iloc[-1]
    last_paxg = paxg.iloc[-1]
    
    # Indikator
    stoch_k = last_xau['STOCHRSIk']
    stoch_d = last_xau['STOCHRSId']
    
    if stoch_k < 20 and stoch_k > stoch_d: res_stoch = ("🟢 BULLISH", "Golden Cross")
    elif stoch_k > 80 and stoch_k < stoch_d: res_stoch = ("🔴 BEARISH", "Death Cross")
    elif stoch_k < 20: res_stoch = ("⚪ WAIT", "Oversold")
    else: res_stoch = ("⚪ NEUTRAL", f"{stoch_k:.1f}")
    
    # MACD
    if last_xau['MACD'] > last_xau['MACD_Signal']: res_macd = ("🟢 BULLISH", "Trend Naik")
    else: res_macd = ("🔴 BEARISH", "Trend Turun")
    
    # VPVR
    if last_xau['Close'] > poc: res_vpvr = ("🟢 STRONG", "Above POC")
    else: res_vpvr = ("🔴 WEAK", "Below POC")
    
    # Bollinger
    if last_xau['Close'] <= last_xau['BBL']: res_bb = ("🟢 BUY ZONE", "Lower Band")
    elif last_xau['Close'] >= last_xau['BBU']: res_bb = ("🔴 SELL ZONE", "Upper Band")
    else: res_bb = ("⚪ INSIDE", "Normal")
    
    dist_to_gold = last_xau['Close'] - xau_fib["GOLDEN POCKET (0.618)"]
    if abs(dist_to_gold) < 15: res_fib = ("⚠️ ALERT", "Testing Golden Pocket")
    elif dist_to_gold > 0: res_fib = ("🔴 ABOVE", "Above Support")
    else: res_fib = ("🟢 BELOW", "Discount Area")

    current_paxg_usd = last_paxg['Close']
    target_buy_usd = paxg_fib["GOLDEN POCKET (0.618)"]
    target_sell_usd = paxg_fib["RESISTANCE (High)"]
    
    decision = "WAIT / HOLD"
    validation = "Market sideways."
    
    # LOGIKA PENGAMBILAN KEPUTUSAN
    # Note: Variabel current_paxg_usd sudah kena kalibrasi di get_data_engine
    if (res_stoch[0] == "🟢 BULLISH") and (current_paxg_usd <= target_buy_usd + 10):
        decision = "🔵 BUY / LONG"
        validation = "✅ VALIDATED: Rebound Golden Pocket + Stoch Cross Up."
    elif (res_bb[0] == "🟢 BUY ZONE") and (res_stoch[0] == "🟢 BULLISH"):
        decision = "🔵 BUY / SCALP"
        validation = "✅ VALIDATED: Pantulan Lower BB + Momentum."
    elif (res_stoch[0] == "🔴 BEARISH") and (current_paxg_usd >= target_sell_usd - 10):
        decision = "🟠 SELL / TAKE PROFIT"
        validation = "✅ VALIDATED: Rejection Resistance + Stoch Cross Down."
    elif current_paxg_usd < (target_buy_usd - 20):
        decision = "🛑 CUT LOSS / STOP BUY"
        validation = "⚠️ INVALID: Jebol Support Kuat."

    now = datetime.now(pytz.timezone('Asia/Jakarta'))
    report = f"""🦅 GOLD MASTER AUTOMATION
📅 Waktu: {now.strftime('%d %b %Y | %H:%M WIB')}
============================================================

💰 UPDATE HARGA (SOURCE: PAXG + CALIBRATION)
💵 KURS USD/IDR : {fmt_idr(kurs)}
------------------------------------------------------------
💎 PAXG/USD      : {fmt_usd(current_paxg_usd)}
💎 PAXG/IDR      : {fmt_idr(current_paxg_usd * kurs)}
   *(Estimasi Ajaib +1.5%: {fmt_idr(current_paxg_usd * kurs * SPREAD_AJAIB)})*
------------------------------------------------------------
(Note: Analisa Teknikal 100% menggunakan grafik PAXG yang dikalibrasi)

📊 HASIL ANALISIS (5 METODE)
1. Stoch RSI   [{res_stoch[0]}] : {res_stoch[1]}
2. MACD        [{res_macd[0]}] : {res_macd[1]}
3. VPVR POC    [{res_vpvr[0]}] : {res_vpvr[1]} (Area ${poc:.0f})
4. Bollinger   [{res_bb[0]}] : {res_bb[1]}
5. Fibonacci   [{res_fib[0]}] : {res_fib[1]}

============================================================
🧠 ENSEMBLE DECISION : [ {decision} ]
🔐 VALIDATED BY      : {validation}
============================================================

🎯 MAPPING AREA TERDEKAT (PAXG)
"""
    levels_sorted = ["MOONBAG (1.618)", "RESISTANCE (High)", "GOLDEN POCKET (0.618)", "FLOOR (Low)"]
    for name in levels_sorted:
        paxg_val = paxg_fib[name]
        paxg_idr = paxg_val * kurs * SPREAD_AJAIB
        report += f"\n📍 LEVEL: {name}"
        report += f"\n   • USD : {fmt_usd(paxg_val)}"
        report += f"\n   • IDR : {fmt_idr(paxg_idr)} (Est. Ajaib)"
        if "MOONBAG" in name: report += "\n   👉 [TARGET] TP 2 / Jual Semua."
        elif "RESISTANCE" in name: report += "\n   👉 [UJI NYALI] Tembus=Moonbag. Gagal=Turun."
        elif "GOLDEN POCKET" in name: report += "\n   👉 [BUY ZONE] Mantul=Buy. Jebol=Cut Loss."
        elif "FLOOR" in name: report += "\n   👉 [BAHAYA] Pertahanan Terakhir."
        report += "\n"

    return report, decision

# --- EKSEKUTOR UTAMA (MAIN) ---
if __name__ == "__main__":
    print("🤖 Robot Start (PAXG-Only Mode)...")
    
    # 1. Ambil Secrets
    try:
        TOKEN = os.environ["TELEGRAM_TOKEN"]
        CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
    except KeyError:
        print("❌ Error: Secrets TELEGRAM tidak ditemukan!")
        exit()

    # 2. Jalanin Analisis
    try:
        xau, paxg, kurs = get_data_engine()
        
        if xau.empty:
            print("❌ Data Kosong, skip cycle ini.")
        else:
            final_report, decision = generate_bot_report(xau, paxg, kurs)
            print(f"🧐 Decision saat ini: {decision}")

            # 3. Filter Kirim
            # Kirim kalau ada sinyal BUY/SELL/CUT LOSS
            if "BUY" in decision or "SELL" in decision or "CUT LOSS" in decision:
                print("🚀 Sinyal Penting! Mengirim ke Telegram...")
                send_telegram(TOKEN, CHAT_ID, final_report)
            else:
                # Kalau mau tetap kirim saat WAIT/HOLD buat debug, uncomment baris ini:
                send_telegram(TOKEN, CHAT_ID, final_report)
                # print("💤 Market Sideways (WAIT/HOLD). Tidak kirim laporan.")
            
    except Exception as e:
        print(f"❌ Terjadi Error di Logic: {e}")
