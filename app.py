import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
import pytz

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Gold Master SOP 25", page_icon="🦅", layout="wide")

# --- CSS FIX ---
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 24px; }
    .report-text { 
        font-family: 'Consolas', 'Courier New', monospace; 
        white-space: pre-wrap; 
        background-color: #0e1117; 
        padding: 15px; 
        border-radius: 10px; 
        color: #00ff00; 
        border: 1px solid #333;
        font-size: 14px;
    }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- KONFIGURASI ENGINE ---
TICKERS = ["PAXG-USD", "IDR=X"]
MODAL_GAJI = 5000000  # 5 Juta Rupiah
SPREAD_AJAIB = 1.015 
PAXG_MULTIPLIER = 0.99048968

# --- HELPER FORMATTING ---
def fmt_idr(val): return f"Rp {val:,.0f}".replace(",", ".")
def fmt_usd(val): return f"${val:,.2f}"

# --- FUNGSI INDIKATOR MANUAL ---
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
        "0.786": low + (diff * 0.786),
        "GOLDEN POCKET (0.618)": high - (diff * 0.618), # Retracement dari High
        "MID (0.5)": high - (diff * 0.5),
        "0.382": high - (diff * 0.382),
        "FLOOR (Low)": low
    }
    return levels

# --- DATA ENGINE (SOP: 1D & 4H) ---
@st.cache_data(ttl=300)
def get_data_engine():
    # Fetch 6 Bulan Daily (Big Picture)
    df_daily = yf.download(TICKERS, period="6mo", interval="1d", group_by='ticker', progress=False)
    # Fetch 1 Bulan Hourly (Untuk konversi ke 4H)
    df_hourly = yf.download(TICKERS, period="1mo", interval="1h", group_by='ticker', progress=False)
    
    try:
        if isinstance(df_daily.columns, pd.MultiIndex):
            paxg_d = df_daily['PAXG-USD'].dropna()
            paxg_h = df_hourly['PAXG-USD'].dropna()
            kurs = df_daily['IDR=X']['Close'].iloc[-1]
        else:
            return pd.DataFrame(), pd.DataFrame(), 16800

        # Kalibrasi Harga User
        for df in [paxg_d, paxg_h]:
            df['Close'] *= PAXG_MULTIPLIER
            df['High'] *= PAXG_MULTIPLIER
            df['Low'] *= PAXG_MULTIPLIER
            df['Open'] *= PAXG_MULTIPLIER

        # Indikator Daily
        paxg_d = add_indicators(paxg_d)
        
        # Resample Hourly ke 4H & Indikator
        paxg_4h = paxg_h.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        paxg_4h = add_indicators(paxg_4h)

    except Exception as e:
        st.error(f"Error Data: {e}")
        return pd.DataFrame(), pd.DataFrame(), 16800
        
    return paxg_d, paxg_4h, kurs

def send_telegram_alert(token, chat_id, message):
    if not token or not chat_id: return False, "Token/ID Kosong"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {"chat_id": chat_id, "text": message}
    try:
        r = requests.get(url, params=params)
        return (True, "Sukses") if r.status_code == 200 else (False, r.text)
    except Exception as e:
        return False, str(e)

# --- REPORT GENERATOR (THE BRAIN) ---
def generate_sop_report(df_d, df_4h, kurs):
    last_d = df_d.iloc[-1]
    last_4h = df_4h.iloc[-1]
    
    # 1. TENTUKAN SKENARIO (DAILY)
    is_swinger = (last_d['Close'] > last_d['BBU']) or (last_d['STOCHRSIk'] > 80)
    
    fibo = calculate_fibonacci_levels(df_d)
    poc = get_poc(df_d)
    
    # --- MATRIX 5-5 STATUS & ANGKA ---
    # 1. Stoch RSI
    stoch_val = last_d['STOCHRSIk']
    if stoch_val > 80: st_stat = "🔴 OVERBOUGHT"
    elif stoch_val < 20: st_stat = "🟢 OVERSOLD"
    else: st_stat = "⚪ NEUTRAL"
    
    # 2. MACD
    macd_val = last_d['MACD']
    sig_val = last_d['MACD_Signal']
    if macd_val > sig_val: mac_stat = "🟢 BULLISH"
    else: mac_stat = "🔴 BEARISH"
    
    # 3. VPVR
    if last_d['Close'] > poc: vp_stat = "🟢 STRONG (Above POC)"
    else: vp_stat = "🔴 WEAK (Below POC)"
    
    # 4. Bollinger
    bb_upper = last_d['BBU']
    bb_lower = last_d['BBL']
    if last_d['Close'] >= bb_upper: bb_stat = "🔴 AT UPPER BAND"
    elif last_d['Close'] <= bb_lower: bb_stat = "🟢 AT LOWER BAND"
    else: bb_stat = "⚪ INSIDE BANDS"
    
    # 5. Fibo
    dist_gold = last_d['Close'] - fibo['GOLDEN POCKET (0.618)']
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
        
        # Target Limit: Max(POC, Fibo 0.618) tapi di bawah harga skrg
        limit_target = max(poc, fibo['GOLDEN POCKET (0.618)'])
        if limit_target >= last_d['Close']: limit_target = fibo['MID (0.5)']
        
        est_market = dana_market / (last_d['Close'] * kurs * SPREAD_AJAIB)
        est_limit_idr = limit_target * kurs * SPREAD_AJAIB
        
        action_txt = f"""
1. MARKET ORDER (50%): Rp {dana_market:,.0f}
   (Estimasi dapat: {est_market:.4f} PAXG)

2. LIMIT ORDER (50%): Rp {dana_limit:,.0f}
   @ Harga ${limit_target:.2f} (Est. IDR: {fmt_idr(est_limit_idr)})
   
3. HOLD SELAMANYA (Akumulasi).
        """

    now = datetime.now(pytz.timezone('Asia/Jakarta'))
    
    report = f"""🦅 GOLD MASTER SOP REPORT
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
   👉 Golden Pkt: ${fibo['GOLDEN POCKET (0.618)']:.2f}

============================================================
🧠 KEPUTUSAN SOP : [ {decision} ]
🔐 KONDISI PASAR : {validation}
============================================================

📝 INSTRUKSI EKSEKUSI (MODAL 5 JUTA):
{action_txt}

🎯 MAPPING AREA (DAILY 6 MO)
"""
    for name, val in fibo.items():
        paxg_idr = val * kurs * SPREAD_AJAIB
        report += f"{name:<20} : {fmt_usd(val)} | {fmt_idr(paxg_idr)}\n"

    return report, df_d, fibo, is_swinger

# --- MAIN APP ---
st.title("🦅 Gold Master SOP (Tanggal 25)")

with st.spinner("Menganalisa Data SOP..."):
    df_d, df_4h, kurs_val = get_data_engine()
    
    if df_d.empty:
        st.error("Gagal Data.")
    else:
        final_report, xau_processed, fib_levels, mode_swinger = generate_sop_report(df_d, df_4h, kurs_val)

        # --- SIDEBAR ---
        st.sidebar.header("⚙️ Konfigurasi")
        if "TELEGRAM_TOKEN" in st.secrets:
            bot_token = st.secrets["TELEGRAM_TOKEN"]
            chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        else:
            bot_token = st.sidebar.text_input("Bot Token", type="password")
            chat_id = st.sidebar.text_input("Chat ID")

        st.sidebar.markdown("---")
        # Visual Indikator Mode
        if mode_swinger:
            st.sidebar.error("🚨 MODE SWINGER (WAIT)")
        else:
            st.sidebar.success("✅ MODE INVESTOR (BUY)")
            
        st.sidebar.metric("PAXG/USD", fmt_usd(xau_processed.iloc[-1]['Close']))

        # --- CHART ---
        st.subheader("📊 Chart Daily (6 Bulan) + Fibo & BB")
        fig = go.Figure(data=[go.Candlestick(x=xau_processed.index,
                                open=xau_processed['Open'], high=xau_processed['High'],
                                low=xau_processed['Low'], close=xau_processed['Close'],
                                name='PAXG/USD')])
        
        # BB Lines
        fig.add_trace(go.Scatter(x=xau_processed.index, y=xau_processed['BBU'], line=dict(color='red', width=1, dash='dot'), name='Upper BB'))
        fig.add_trace(go.Scatter(x=xau_processed.index, y=xau_processed['BBL'], line=dict(color='green', width=1, dash='dot'), name='Lower BB'))
        
        # Fibo Lines
        colors = {"MOONBAG": "lime", "RESISTANCE": "red", "GOLDEN": "gold", "FLOOR": "white", "MID": "gray"}
        for label, val in fib_levels.items():
            c = "gray"
            for k, v in colors.items():
                if k in label: c = v
            fig.add_hline(y=val, line_dash="dash", line_color=c, annotation_text=f"{label}")
            
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- REPORT ---
        st.subheader("📋 Output Logika SOP")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("📩 Kirim ke Telegram"):
                success, msg = send_telegram_alert(bot_token, chat_id, final_report)
                if success: st.success("Terkirim!")
                else: st.error(f"Gagal: {msg}")

        st.text_area("Report:", value=final_report, height=700, label_visibility="collapsed")
