import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import pytz

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Gold Master Investment", page_icon="🦅", layout="wide")

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
MODAL_GAJI = 5000000 
SPREAD_AJAIB = 1.015 
PAXG_MULTIPLIER = 0.99048968

# --- HELPER FORMATTING ---
def fmt_idr(val): return f"Rp {val:,.0f}".replace(",", ".")
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
@st.cache_data(ttl=300)
def get_data_engine():
    # 1. Fetch 2 TAHUN (Untuk EMA 200)
    df_full = yf.download(TICKERS, period="2y", interval="1d", group_by='ticker', progress=False)
    # Fetch 1 Bulan Hourly (Untuk Swinger Check)
    df_hourly = yf.download(TICKERS, period="1mo", interval="1h", group_by='ticker', progress=False)
    
    try:
        if isinstance(df_full.columns, pd.MultiIndex):
            paxg_d = df_full['PAXG-USD'].dropna()
            paxg_h = df_hourly['PAXG-USD'].dropna()
            kurs = df_full['IDR=X']['Close'].iloc[-1]
        else:
            return pd.DataFrame(), pd.DataFrame(), 16800

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

    except Exception as e:
        st.error(f"Error Data: {e}")
        return pd.DataFrame(), pd.DataFrame(), 16800
        
    # Return paxg_6mo (Data potong) bukan paxg_d (Data full)
    return paxg_6mo, paxg_4h, kurs

def send_telegram_alert(token, chat_id, message):
    if not token or not chat_id: return False, "Token/ID Kosong"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {"chat_id": chat_id, "text": message}
    try:
        r = requests.get(url, params=params)
        return (True, "Sukses") if r.status_code == 200 else (False, r.text)
    except Exception as e:
        return False, str(e)

# --- REPORT GENERATOR ---
def generate_sop_report(df_6mo, df_4h, kurs):
    # Gunakan data 6 bulan untuk analisa Fibo & POC
    last_d = df_6mo.iloc[-1]
    last_4h = df_4h.iloc[-1]
    
    fibo = calculate_fibonacci_levels(df_6mo) # Fibo akurat range 6 bulan
    poc = get_poc(df_6mo) # POC akurat range 6 bulan
    
    # 1. MATRIX 5+1 STATUS
    
    # EMA 200 (Nilainya sudah terbawa dari perhitungan 2 tahun)
    ema200 = last_d['EMA200']
    price = last_d['Close']
    if price > ema200: ema_stat = "🟢 UPTREND"
    else: ema_stat = "🔴 DOWNTREND"

    # Stoch RSI
    stoch_val = last_d['STOCHRSIk']
    if stoch_val > 80: st_stat = "🔴 OVERBOUGHT"
    elif stoch_val < 20: st_stat = "🟢 OVERSOLD"
    else: st_stat = "⚪ NEUTRAL"
    
    # MACD
    if last_d['MACD'] > last_d['MACD_Signal']: mac_stat = "🟢 BULLISH"
    else: mac_stat = "🔴 BEARISH"
    
    # VPVR
    if last_d['Close'] > poc: vp_stat = "🟢 STRONG (Above POC)"
    else: vp_stat = "🔴 WEAK (Below POC)"
    
    # Bollinger
    if last_d['Close'] >= last_d['BBU']: bb_stat = "🔴 BREAKOUT UPPER"
    elif last_d['Close'] <= last_d['BBL']: bb_stat = "🟢 BREAKOUT LOWER"
    else: bb_stat = "⚪ INSIDE BANDS"
    
    # Fibo
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
        
        # Target Limit: Prioritaskan Support Terkuat di Bawah Harga
        candidates = [poc, fibo['GOLDEN (0.618)'], ema200]
        valid_supports = [x for x in candidates if x < price]
        
        if valid_supports:
            limit_target = max(valid_supports)
        else:
            limit_target = fibo['MID (0.5)'] # Fallback

        est_market = dana_market / (price * kurs * SPREAD_AJAIB)
        est_limit_idr = limit_target * kurs * SPREAD_AJAIB
        
        action_txt = f"""
1. MARKET ORDER (50%): Rp {dana_market:,.0f}
   (Estimasi dapat: {est_market:.4f} PAXG)

2. LIMIT ORDER (50%): Rp {dana_limit:,.0f}
   @ Harga ${limit_target:.2f} (Est. IDR: {fmt_idr(est_limit_idr)})
   *(Target: EMA200/POC/Fibo)*
        """

    now = datetime.now(pytz.timezone('Asia/Jakarta'))
    
    report = f"""GOLD INVESTMENT REPORT (SMART SLICE)
📅 Waktu: {now.strftime('%d %b %Y | %H:%M WIB')}
============================================================

💰 UPDATE HARGA (PAXG)
💵 KURS USD/IDR : {fmt_idr(kurs)}
------------------------------------------------------------
💎 PAXG/USD      : {fmt_usd(price)}
💎 EMA 200       : {fmt_usd(ema200)}
💎 PAXG/IDR      : {fmt_idr(price * kurs)}
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
    sorted_fibo = dict(sorted(fibo.items(), key=lambda item: item[1], reverse=True))
    for name, val in sorted_fibo.items():
        paxg_idr = val * kurs * SPREAD_AJAIB
        report += f"{name:<20} : {fmt_usd(val)} | {fmt_idr(paxg_idr)}\n"

    return report, df_6mo, fibo, is_swinger

# --- MAIN APP ---
st.title("Gold Master Guide (EMA 200 + 6 Mo Focus)")

with st.spinner("Processing Data (2 Years Fetch -> 6 Months Slice)..."):
    df_6mo, df_4h, kurs_val = get_data_engine()
    
    if df_6mo.empty:
        st.error("Gagal Data.")
    else:
        final_report, xau_processed, fib_levels, mode_swinger = generate_sop_report(df_6mo, df_4h, kurs_val)

        # --- SIDEBAR ---
        st.sidebar.header("⚙️ Konfigurasi")
        if "TELEGRAM_TOKEN" in st.secrets:
            bot_token = st.secrets["TELEGRAM_TOKEN"]
            chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        else:
            bot_token = st.sidebar.text_input("Bot Token", type="password")
            chat_id = st.sidebar.text_input("Chat ID")

        st.sidebar.markdown("---")
        if mode_swinger:
            st.sidebar.error("🚨 MODE SWINGER")
        else:
            st.sidebar.success("✅ MODE INVESTOR")
            
        st.sidebar.metric("PAXG/USD", fmt_usd(xau_processed.iloc[-1]['Close']))
        st.sidebar.metric("EMA 200", fmt_usd(xau_processed.iloc[-1]['EMA200']))

        # --- CHART (Hanya Menampilkan 6 Bulan Terakhir) ---
        st.subheader("📊 Chart Daily (Fokus 6 Bulan Terakhir)")
        
        fig = go.Figure(data=[go.Candlestick(x=xau_processed.index,
                                open=xau_processed['Open'], high=xau_processed['High'],
                                low=xau_processed['Low'], close=xau_processed['Close'],
                                name='PAXG/USD')])
        
        # EMA 200 (Garis Biru)
        fig.add_trace(go.Scatter(x=xau_processed.index, y=xau_processed['EMA200'], line=dict(color='blue', width=2), name='EMA 200'))
        
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
