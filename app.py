import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
import pytz

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Gold Master Automation", page_icon="🦅", layout="wide")

# --- CSS FIX ---
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 24px; }
    .report-text { 
        font-family: 'Courier New', monospace; 
        white-space: pre-wrap; 
        background-color: #f0f2f6; 
        padding: 15px; 
        border-radius: 10px; 
        color: #000000; 
        border: 1px solid #ccc;
    }
</style>
""", unsafe_allow_html=True)

# --- KONFIGURASI ENGINE ---
TICKERS = ["GC=F", "PAXG-USD", "IDR=X"]
INTERVAL = "1h"
PERIOD = "1mo"
SPREAD_AJAIB = 1.015 

# --- HELPER FORMATTING ---
def fmt_idr(val): return f"Rp {val:,.0f}".replace(",", ".")
def fmt_usd(val): return f"${val:,.2f}"

# --- FUNGSI INDIKATOR MANUAL (PENGGANTI PANDAS_TA) ---
def add_manual_indicators(df):
    df = df.copy()
    
    # 1. MACD (12, 26, 9)
    # EMA Short & Long
    k = df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
    d = df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
    df['MACD'] = k - d
    # Signal Line
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()
    
    # 2. Bollinger Bands (20, 2)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BBU'] = df['SMA20'] + (df['STD20'] * 2) # Upper
    df['BBL'] = df['SMA20'] - (df['STD20'] * 2) # Lower
    
    # 3. Stochastic RSI (14, 14, 3, 3)
    # Hitung RSI Biasa Dulu
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Hitung Stoch dari RSI
    min_rsi = df['RSI'].rolling(window=14).min()
    max_rsi = df['RSI'].rolling(window=14).max()
    # Rumus Stoch K & D
    stoch_rsi = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)
    df['STOCHRSIk'] = stoch_rsi.rolling(window=3).mean() * 100 # Kali 100 biar skala 0-100
    df['STOCHRSId'] = df['STOCHRSIk'].rolling(window=3).mean()
    
    return df

# --- FUNGSI GET DATA ---
@st.cache_data(ttl=300)
def get_data_engine():
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

def send_telegram_alert(token, chat_id, message):
    if not token or not chat_id: return False, "Token/ID Kosong"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {"chat_id": chat_id, "text": message}
    try:
        r = requests.get(url, params=params)
        return (True, "Sukses") if r.status_code == 200 else (False, r.text)
    except Exception as e:
        return False, str(e)

# --- LOGIC ANALYSIS & REPORT ---
def generate_analysis_report(xau, paxg, kurs):
    # >>> INI BEDANYA: Panggil fungsi manual, bukan pandas_ta <<<
    xau = add_manual_indicators(xau)
    
    # VPVR Logic (Tetap pake pandas native)
    price_bins = pd.cut(xau['Close'], bins=50)
    vpvr = xau.groupby(price_bins, observed=True)['Volume'].sum()
    poc = vpvr.idxmax().mid
    
    xau_fib = calculate_fibonacci_levels(xau)
    paxg_fib = calculate_fibonacci_levels(paxg)

    last_xau = xau.iloc[-1]
    last_paxg = paxg.iloc[-1]
    
    # Ambil Data (Sekarang nama kolomnya sudah pasti, gak perlu search aneh-aneh)
    stoch_k = last_xau['STOCHRSIk']
    stoch_d = last_xau['STOCHRSId']
    
    if stoch_k < 20 and stoch_k > stoch_d: res_stoch = ("🟢 BULLISH", "Golden Cross")
    elif stoch_k > 80 and stoch_k < stoch_d: res_stoch = ("🔴 BEARISH", "Death Cross")
    elif stoch_k < 20: res_stoch = ("⚪ WAIT", "Oversold")
    else: res_stoch = ("⚪ NEUTRAL", f"{stoch_k:.1f}")
    
    # MACD Logic
    if last_xau['MACD'] > last_xau['MACD_Signal']: res_macd = ("🟢 BULLISH", "Trend Naik")
    else: res_macd = ("🔴 BEARISH", "Trend Turun")
    
    # POC Logic
    if last_xau['Close'] > poc: res_vpvr = ("🟢 STRONG", "Above POC")
    else: res_vpvr = ("🔴 WEAK", "Below POC")
    
    # BB Logic
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

    return report, xau, xau_fib, last_xau, last_paxg

# --- HALAMAN UTAMA ---
st.title("🦅 Gold Master Automation")

with st.spinner("Sedang Menganalisis (Manual Calculation)..."):
    xau_data, paxg_data, kurs_val = get_data_engine()
    final_report, xau_processed, xau_fib_levels, last_xau_row, last_paxg_row = generate_analysis_report(xau_data, paxg_data, kurs_val)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Konfigurasi")

if "TELEGRAM_TOKEN" in st.secrets:
    bot_token = st.secrets["TELEGRAM_TOKEN"]
    chat_id = st.secrets["TELEGRAM_CHAT_ID"]
    st.sidebar.success("✅ Login via Secrets")
else:
    bot_token = st.sidebar.text_input("Bot Token", type="password")
    chat_id = st.sidebar.text_input("Chat ID")

st.sidebar.markdown("---")
st.sidebar.header("💰 Live Price")
st.sidebar.metric("Kurs USD/IDR", fmt_idr(kurs_val))

est_ajaib = last_paxg_row['Close'] * kurs_val * SPREAD_AJAIB
st.sidebar.metric("PAXG/IDR (Ajaib)", fmt_idr(est_ajaib))

st.sidebar.markdown("---")
st.sidebar.metric("XAU/USD (Spot)", fmt_usd(last_xau_row['Close']))
xau_gram = (last_xau_row['Close'] * kurs_val) / 31.1035
st.sidebar.metric("XAU/IDR (Gram)", fmt_idr(xau_gram))

# --- CHART ---
st.subheader("📊 Chart XAU/USD + Fibonacci Levels")
fig = go.Figure(data=[go.Candlestick(x=xau_processed.index,
                open=xau_processed['Open'], high=xau_processed['High'],
                low=xau_processed['Low'], close=xau_processed['Close'],
                name='XAU/USD')])
colors = {"MOONBAG": "lime", "RESISTANCE": "red", "GOLDEN POCKET": "gold", "FLOOR": "white"}
for label, val in xau_fib_levels.items():
    c = "gray"
    for k, v in colors.items():
        if k in label: c = v
    fig.add_hline(y=val, line_dash="dash", line_color=c, 
                  annotation_text=f"{label} : ${val:.2f}", 
                  annotation_position="top right")
fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# --- REPORT ---
st.subheader("📋 Laporan Analisis Lengkap")
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("📩 Kirim ke Telegram"):
        success, msg = send_telegram_alert(bot_token, chat_id, final_report)
        if success: st.success("Terkirim!")
        else: st.error(f"Gagal: {msg}")

st.text_area("Output Logika:", value=final_report, height=600, label_visibility="collapsed")


