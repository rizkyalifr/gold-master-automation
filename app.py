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
# ✅ HAPUS GC=F. Cuma ambil PAXG dan KURS.
TICKERS = ["PAXG-USD", "IDR=X"]
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

# --- FUNGSI GET DATA (REVISI: PAXG ONLY) ---
@st.cache_data(ttl=300)
def get_data_engine():
    # Download cuma 2 Ticker (PAXG & IDR)
    df = yf.download(TICKERS, period=PERIOD, interval=INTERVAL, group_by='ticker', progress=False, threads=False)
    
    try:
        # 1. Ambil Data PAXG
        # Handle beda format yfinance (kadang return MultiIndex, kadang tidak)
        if isinstance(df.columns, pd.MultiIndex):
            paxg = df['PAXG-USD'].dropna()*  0.99048968
        else:
            # Fallback kalau yfinance ngaco strukturnya
            # Kita cari kolom yang ada bau-bau PAXG
            paxg = df # Asumsi df cuma isi paxg kalau single ticker (jarang terjadi krn ada IDR)
        
        # 2. 🔥 JURUS CERMIN: XAU DIANGGAP SAMA DENGAN PAXG 🔥
        # Ini bikin analisa teknikal lu 100% sinkron sama barang yang lu beli.
        xau = paxg.copy() 
        
        # 3. Ambil Kurs IDR
        if isinstance(df.columns, pd.MultiIndex):
            kurs_raw = df['IDR=X']['Close'].dropna()
        else:
            # Fallback (biasanya jarang masuk sini kalau tickers > 1)
             kurs_raw = pd.Series([16800])

        kurs = kurs_raw.iloc[-1] if not kurs_raw.empty else 16800

    except Exception as e:
        st.error(f"Error Data Fetching: {e}")
        # Data Darurat biar gak crash
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
    if xau.empty: return "Data Kosong/Error", xau, {}, xau, paxg

    # Analisa tetap pakai variabel 'xau', tapi isinya sekarang adalah data PAXG
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
    # Karena XAU = PAXG, logikanya jadi lebih simpel dan akurat
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

💰 UPDATE HARGA (SOURCE: PAXG REAL-TIME)
💵 KURS USD/IDR : {fmt_idr(kurs)}
------------------------------------------------------------
💎 PAXG/USD      : {fmt_usd(current_paxg_usd)}
💎 PAXG/IDR      : {fmt_idr(current_paxg_usd * kurs)}
   *(Estimasi Ajaib +1.5%: {fmt_idr(current_paxg_usd * kurs * SPREAD_AJAIB)})*
------------------------------------------------------------
(Note: Analisa Teknikal 100% menggunakan grafik PAXG)

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

    return report, xau, xau_fib, last_xau, last_paxg

# --- HALAMAN UTAMA ---
st.title("🦅 Gold Master Automation")

with st.spinner("Sedang Menganalisis PAXG Market..."):
    xau_data, paxg_data, kurs_val = get_data_engine()
    
    if xau_data.empty:
        st.error("Gagal mengambil data PAXG. Coba refresh atau cek koneksi yfinance.")
    else:
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
        st.sidebar.header("💰 Live Price (PAXG)")
        st.sidebar.metric("Kurs USD/IDR", fmt_idr(kurs_val))

        est_ajaib = last_paxg_row['Close'] * kurs_val * SPREAD_AJAIB
        st.sidebar.metric("PAXG/IDR (Ajaib)", fmt_idr(est_ajaib))

        st.sidebar.markdown("---")
        st.sidebar.metric("PAXG/USD (Spot)", fmt_usd(last_paxg_row['Close']))

        # --- CHART ---
        st.subheader("📊 Chart PAXG/USD + Fibonacci Levels")
        fig = go.Figure(data=[go.Candlestick(x=xau_processed.index,
                        open=xau_processed['Open'], high=xau_processed['High'],
                        low=xau_processed['Low'], close=xau_processed['Close'],
                        name='PAXG/USD')])
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

