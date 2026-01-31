import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
from datetime import datetime
import pytz

# --- KONFIGURASI ---
TICKERS = ["GC=F", "PAXG-USD", "IDR=X"]
INTERVAL = "1h"
PERIOD = "1mo"
SPREAD_AJAIB = 1.015 

# --- HELPER ---
def fmt_idr(val): return f"Rp {val:,.0f}".replace(",", ".")
def fmt_usd(val): return f"${val:,.2f}"

# --- FUNGSI DATA ---
def get_data_engine():
    # Progress=False biar log bersih
    df = yf.download(TICKERS, period=PERIOD, interval=INTERVAL, group_by='ticker', progress=False, threads=False)
    try:
        xau = df['GC=F'].dropna()
        paxg = df['PAXG-USD'].dropna()
        kurs_raw = df['IDR=X']['Close'].dropna()
        kurs = kurs_raw.iloc[-1] if not kurs_raw.empty else 16800
    except:
        xau = df.xs('GC=F', axis=1, level=0).dropna()
        paxg = df.xs('PAXG-USD', axis=1, level=0).dropna()
        kurs = 16800
    if kurs < 10000: kurs = 16800
    return xau, paxg, kurs

def calculate_fibonacci_levels(df):
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

# --- CORE LOGIC (GENERATOR LAPORAN) ---
def generate_report_string(xau, paxg, kurs):
    # Hitung Indikator
    xau = xau.copy()
    xau.ta.stochrsi(append=True)
    xau.ta.macd(append=True)
    xau.ta.bbands(append=True)
    
    price_bins = pd.cut(xau['Close'], bins=50)
    vpvr = xau.groupby(price_bins, observed=True)['Volume'].sum()
    poc = vpvr.idxmax().mid
    
    xau_fib = calculate_fibonacci_levels(xau)
    paxg_fib = calculate_fibonacci_levels(paxg)

    last_xau = xau.iloc[-1]
    last_paxg = paxg.iloc[-1]
    
    # Analysis Logic (SAMA PERSIS DENGAN APP.PY)
    k_col = [c for c in xau.columns if "STOCHRSIk" in c][0]
    d_col = [c for c in xau.columns if "STOCHRSId" in c][0]
    stoch_k = last_xau[k_col]
    stoch_d = last_xau[d_col]
    
    if stoch_k < 20 and stoch_k > stoch_d: res_stoch = ("🟢 BULLISH", "Golden Cross")
    elif stoch_k > 80 and stoch_k < stoch_d: res_stoch = ("🔴 BEARISH", "Death Cross")
    elif stoch_k < 20: res_stoch = ("⚪ WAIT", "Oversold")
    else: res_stoch = ("⚪ NEUTRAL", f"{stoch_k:.1f}")
    
    macd_col = [c for c in xau.columns if "MACD_" in c and "s_" not in c][0]
    sig_col = [c for c in xau.columns if "MACDs_" in c][0]
    if last_xau[macd_col] > last_xau[sig_col]: res_macd = ("🟢 BULLISH", "Trend Naik")
    else: res_macd = ("🔴 BEARISH", "Trend Turun")
    
    if last_xau['Close'] > poc: res_vpvr = ("🟢 STRONG", "Above POC")
    else: res_vpvr = ("🔴 WEAK", "Below POC")
    
    bbu_col = [c for c in xau.columns if "BBU_" in c][0]
    bbl_col = [c for c in xau.columns if "BBL_" in c][0]
    if last_xau['Close'] <= last_xau[bbl_col]: res_bb = ("🟢 BUY ZONE", "Lower Band")
    elif last_xau['Close'] >= last_xau[bbu_col]: res_bb = ("🔴 SELL ZONE", "Upper Band")
    else: res_bb = ("⚪ INSIDE", "Normal")
    
    dist_to_gold = last_xau['Close'] - xau_fib["GOLDEN POCKET (0.618)"]
    if abs(dist_to_gold) < 15: res_fib = ("⚠️ ALERT", "Testing Golden Pocket")
    elif dist_to_gold > 0: res_fib = ("🔴 ABOVE", "Above Support")
    else: res_fib = ("🟢 BELOW", "Discount Area")

    # Ensemble Decision
    current_paxg_usd = last_paxg['Close']
    target_buy_usd = paxg_fib["GOLDEN POCKET (0.618)"]
    target_sell_usd = paxg_fib["RESISTANCE (High)"]
    
    decision = "WAIT / HOLD"
    validation = "Market sideways."
    
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

    # Construct Report (FORMAT SAMA PERSIS)
    now = datetime.now(pytz.timezone('Asia/Jakarta'))
    report = f"""🦅 GOLD MASTER AUTOMATION
📅 Waktu: {now.strftime('%d %b %Y | %H:%M WIB')}
============================================================

💰 UPDATE HARGA & RANGE
💵 KURS USD/IDR : {fmt_idr(kurs)}
------------------------------------------------------------
🏆 XAU/USD      : {fmt_usd(last_xau['Close'])}
🏆 XAU/IDR Gram : {fmt_idr((last_xau['Close'] * kurs) / 31.1035)}
------------------------------------------------------------
💎 PAXG/USD     : {fmt_usd(current_paxg_usd)}
💎 PAXG/IDR     : {fmt_idr(current_paxg_usd * kurs)} - {fmt_idr(current_paxg_usd * kurs * SPREAD_AJAIB)}
   *(Range: Harga Wajar s.d. Estimasi App Ajaib +1.5%)*

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

🎯 MAPPING AREA TERDEKAT & SKENARIO
"""
    levels_sorted = ["MOONBAG (1.618)", "RESISTANCE (High)", "GOLDEN POCKET (0.618)", "FLOOR (Low)"]
    for name in levels_sorted:
        xau_val = xau_fib[name]
        paxg_val = paxg_fib[name]
        paxg_idr = paxg_val * kurs * SPREAD_AJAIB
        report += f"\n📍 LEVEL: {name}"
        report += f"\n   • XAU : {fmt_usd(xau_val)}"
        report += f"\n   • PAXG: {fmt_usd(paxg_val)} | {fmt_idr(paxg_idr)} (Est. Ajaib)"
        if "MOONBAG" in name: report += "\n   👉 [TARGET] TP 2 / Jual Semua."
        elif "RESISTANCE" in name: report += "\n   👉 [UJI NYALI] Tembus=Moonbag. Gagal=Turun."
        elif "GOLDEN POCKET" in name: report += "\n   👉 [BUY ZONE] Mantul=Buy. Jebol=Cut Loss."
        elif "FLOOR" in name: report += "\n   👉 [BAHAYA] Pertahanan Terakhir."
        report += "\n"

    return report, decision

def send_telegram(token, chat_id, msg):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {"chat_id": chat_id, "text": msg} # Plain text (Sesuai request)
    requests.get(url, params=params)

# --- EKSEKUTOR UTAMA ---
if __name__ == "__main__":
    print("🤖 Robot Start...")
    
    # Ambil Secrets dari GitHub Environment
    try:
        TOKEN = os.environ["TELEGRAM_TOKEN"]
        CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
    except KeyError:
        print("❌ Error: Token/ID belum diset di Secrets GitHub!")
        exit()

    # Jalankan Analisis
    xau, paxg, kurs = get_data_engine()
    final_report, decision = generate_report_string(xau, paxg, kurs)
    
    print(f"🧐 Decision: {decision}")

    # --- LOGIKA PENGIRIMAN ---
    # Opsi 1: Kirim HANYA jika ada sinyal BUY/SELL (Biar gak spam WAIT)
    if "WAIT" not in decision and "HOLD" not in decision:
        print("🚀 Sinyal Valid! Mengirim ke Telegram...")
        send_telegram(TOKEN, CHAT_ID, final_report)
    else:
        # Opsi 2: Kalau mau kirim report per 4 jam walau WAIT, uncomment baris bawah ini:
        # send_telegram(TOKEN, CHAT_ID, final_report) 
        print("💤 Market Sideways (WAIT). Tidak kirim pesan.")
    
    print("✅ Robot Selesai.")