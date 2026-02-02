import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import pytz

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Gold Master Quantitative", page_icon="🦅", layout="wide")

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

# --- FUNGSI INDIKATOR (DATA PROCESSING) ---
def process_data_smart(df):
    df = df.copy()
    # 1. EMA 200
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    # 2. Bollinger
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BBU'] = df['SMA20'] + (df['STD20'] * 2)
    df['BBL'] = df['SMA20'] - (df['STD20'] * 2)
    df['BBM'] = df['SMA20']
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

# --- SCORING ENGINE (THE QUANTITATIVE BRAIN) ---
def calculate_quant_score(row, prev_row, poc, fibo_golden):
    price = row['Close']
    scores = {}
    details = {}

    # 1. EMA 200 (Weight 30%)
    # Logic: Distance price vs EMA
    ema = row['EMA200']
    dist_pct = ((price - ema) / ema) * 100
    
    if dist_pct >= 12: score_ema = 100
    elif 8 <= dist_pct < 12: score_ema = 85
    elif 4 <= dist_pct < 8: score_ema = 70
    elif 0 <= dist_pct < 4: score_ema = 55
    else: score_ema = 30 # Below EMA
    
    scores['EMA'] = score_ema
    details['EMA'] = f"Dist: {dist_pct:+.2f}%"

    # 2. VPVR POC (Weight 20%)
    # Logic: Position vs POC
    dist_poc_pct = abs((price - poc) / poc) * 100
    if price > poc: score_vpvr = 100 # Strong Support Below
    elif dist_poc_pct <= 3: score_vpvr = 60 # Near POC
    else: score_vpvr = 30 # Below POC (Resistance Overhead)
    
    scores['VPVR'] = score_vpvr
    details['VPVR'] = "Above POC" if price > poc else "Below/Near POC"

    # 3. MACD (Weight 15%)
    # Logic: Momentum
    hist = row['MACD_Hist']
    prev_hist = prev_row['MACD_Hist']
    
    if hist > 0 and hist > prev_hist: score_macd = 100 # Bullish Expanding
    elif hist > 0 and hist < prev_hist: score_macd = 70 # Bullish Weakening
    elif hist < 0 and hist > prev_hist: score_macd = 50 # Bearish Weakening (Turning up)
    else: score_macd = 30 # Bearish Expanding
    
    scores['MACD'] = score_macd
    details['MACD'] = f"Hist: {hist:.2f}"

    # 4. Stoch RSI (Weight 10%)
    # Logic: Entry Timing
    k = row['STOCHRSIk']
    d = row['STOCHRSId']
    
    if k < 20 and k > d: score_stoch = 90 # Oversold + Golden Cross
    elif k < 20: score_stoch = 65 # Oversold only
    elif k > 80: score_stoch = 35 # Overbought
    else: score_stoch = 50 # Neutral
    
    scores['STOCH'] = score_stoch
    details['STOCH'] = f"K: {k:.1f}"

    # 5. Bollinger Bands (Weight 10%)
    # Logic: Volatility & Mean Reversion
    if price <= row['BBL']: score_bb = 85 # Touch Lower
    elif price < row['BBM']: score_bb = 65 # Lower Half
    elif price < row['BBU']: score_bb = 35 # Upper Half
    else: score_bb = 20 # Touch Upper
    
    scores['BB'] = score_bb
    details['BB'] = "Lower Zone" if price < row['BBM'] else "Upper Zone"

    # 6. Fibonacci (Weight 15%)
    # Logic: Near 0.618 Support
    dist_fibo_pct = abs((price - fibo_golden) / fibo_golden) * 100
    
    if dist_fibo_pct <= 1.5: score_fibo = 90 # Near Golden
    elif price > fibo_golden: score_fibo = 55 # Above Support
    elif price < fibo_golden * 0.95: score_fibo = 40 # Below Support (Jebol)
    else: score_fibo = 75 # Between levels
    
    scores['FIBO'] = score_fibo
    details['FIBO'] = f"Dist Golden: {dist_fibo_pct:.2f}%"

    # --- COMPOSITE SCORE ---
    final_score = (
        (scores['EMA'] * 0.30) +
        (scores['VPVR'] * 0.20) +
        (scores['MACD'] * 0.15) +
        (scores['STOCH'] * 0.10) +
        (scores['BB'] * 0.10) +
        (scores['FIBO'] * 0.15)
    )
    
    return final_score, scores, details

# --- DATA ENGINE ---
@st.cache_data(ttl=300)
def get_data_engine():
    df_full = yf.download(TICKERS, period="2y", interval="1d", group_by='ticker', progress=False)
    df_hourly = yf.download(TICKERS, period="1mo", interval="1h", group_by='ticker', progress=False)
    
    try:
        if isinstance(df_full.columns, pd.MultiIndex):
            paxg_d = df_full['PAXG-USD'].dropna()
            paxg_h = df_hourly['PAXG-USD'].dropna()
            kurs = df_full['IDR=X']['Close'].iloc[-1]
        else:
            return pd.DataFrame(), pd.DataFrame(), 16800

        for df in [paxg_d, paxg_h]:
            df['Close'] *= PAXG_MULTIPLIER
            df['High'] *= PAXG_MULTIPLIER
            df['Low'] *= PAXG_MULTIPLIER
            df['Open'] *= PAXG_MULTIPLIER

        # Process Indicators on Full Data
        paxg_d = process_data_smart(paxg_d)
        
        # Slicing for Analysis
        paxg_6mo = paxg_d.tail(180).copy()
        
        # Hourly Data
        paxg_4h = paxg_h.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        paxg_4h = process_data_smart(paxg_4h)

    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), 16800
        
    return paxg_6mo, paxg_4h, kurs

def send_telegram_alert(token, chat_id, message):
    if not token or not chat_id: return False, "Token/ID Kosong"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {"chat_id": chat_id, "text": message}
    try:
        requests.get(url, params=params)
        return True, "Sukses"
    except Exception as e:
        return False, str(e)

# --- REPORT GENERATOR (QUANTITATIVE EDITION) ---
def generate_sop_report(df_6mo, df_4h, kurs):
    last_d = df_6mo.iloc[-1]
    prev_d = df_6mo.iloc[-2]
    
    fibo = calculate_fibonacci_levels(df_6mo) 
    poc = get_poc(df_6mo)
    
    # 1. HITUNG SKOR KUANTITATIF
    final_score, scores, details = calculate_quant_score(last_d, prev_d, poc, fibo['GOLDEN (0.618)'])
    
    # 2. MARKET REGIME LOGIC
    # Distance price vs EMA > 10% -> TREND REGIME
    ema_dist = ((last_d['Close'] - last_d['EMA200']) / last_d['EMA200']) * 100
    regime = "🐎 TREND REGIME" if ema_dist > 5 else "🦀 RANGE REGIME" # Adjusted 5% for Gold (10% is too rare for Gold)
    
    # 3. PROBABILITY DECISION MODEL
    if final_score >= 80:
        decision = "🚀 STRONG BUY"
        action = "Aggressive Entry (70% Market / 30% Limit)"
        prob = "High Probability (>80%)"
    elif 65 <= final_score < 80:
        decision = "✅ GRADUAL BUY"
        action = "Standard SOP (50% Market / 50% Limit)"
        prob = "Moderate Probability (65-79%)"
    elif 50 <= final_score < 65:
        decision = "⚠️ WAIT PULLBACK"
        action = "Defensive (20% Market / 80% Limit)"
        prob = "Low Probability (Wait for Dip)"
    else:
        decision = "🛑 AVOID ENTRY"
        action = "Full Cash / Wait"
        prob = "Negative/Risk High"

    # 4. AGREEMENT RATIO
    bullish_count = sum(1 for s in scores.values() if s >= 60)
    agreement = f"{bullish_count}/6 Indikator Bullish"

    # Hitung Nominal
    dana_market = MODAL_GAJI * 0.5 # Default split, adjusted by decision logic below if needed
    dana_limit = MODAL_GAJI * 0.5
    
    # Adjust Split based on Decision
    if "STRONG" in decision: dana_market, dana_limit = MODAL_GAJI * 0.7, MODAL_GAJI * 0.3
    elif "WAIT" in decision: dana_market, dana_limit = MODAL_GAJI * 0.2, MODAL_GAJI * 0.8
    elif "AVOID" in decision: dana_market, dana_limit = 0, MODAL_GAJI
    
    # Target Limit Smart Aggressive
    candidates = [poc, last_d['EMA200'], fibo['0.382'], fibo['MID (0.5)'], fibo['GOLDEN (0.618)']]
    valid_supports = [x for x in candidates if x < last_d['Close']]
    target_limit_usd = max(valid_supports) if valid_supports else fibo['MID (0.5)']
    est_limit_idr = target_limit_usd * kurs * SPREAD_AJAIB

    now = datetime.now(pytz.timezone('Asia/Jakarta'))
    
    report = f"""🦅 GOLD MASTER QUANTITATIVE (V4.0)
📅 Waktu: {now.strftime('%d %b %Y | %H:%M WIB')}
=======================================

💰 UPDATE HARGA
PAXG/USD : {fmt_usd(last_d['Close'])}
EMA 200  : {fmt_usd(last_d['EMA200'])} ({ema_dist:+.2f}%)
KURS IDR : {fmt_idr(kurs)}

📊 MATRIX 6 INDIKATOR (SCORED)
1. EMA 200    [{scores['EMA']}] : {details['EMA']}
2. VPVR POC   [{scores['VPVR']}] : {details['VPVR']}
3. MACD       [{scores['MACD']}] : {details['MACD']}
4. Stoch RSI  [{scores['STOCH']}] : {details['STOCH']}
5. Bollinger  [{scores['BB']}] : {details['BB']}
6. Fibonacci  [{scores['FIBO']}] : {details['FIBO']}

🧮 COMPOSITE SCORE: {final_score:.1f} / 100
🌍 MARKET REGIME : {regime}
🤝 AGREEMENT     : {agreement}

=======================================
🧠 KEPUTUSAN : [ {decision} ]
🎲 PROBABILITAS: {prob}
=======================================

📋 INSTRUKSI EKSEKUSI (MODAL 5 JUTA):
1️⃣ MARKET ORDER
   👉 Nominal: {fmt_idr(dana_market)}
   👉 Eksekusi: SEKARANG.

2️⃣ LIMIT ORDER
   👉 Nominal: {fmt_idr(dana_limit)}
   👉 Pasang di: {fmt_usd(target_limit_usd)}
   👉 Est. IDR : {fmt_idr(est_limit_idr)}

🎯 MAPPING AREA FIBONACCI
"""
    sorted_fibo = dict(sorted(fibo.items(), key=lambda item: item[1], reverse=True))
    for k, v in sorted_fibo.items():
        report += f"{k:<15}: {fmt_usd(v)}\n"
        
    return report, df_6mo, fibo, final_score

# --- MAIN APP ---
st.title("Gold Master Quantitative (Score Based)")

with st.spinner("Processing Quantitative Data..."):
    df_6mo, df_4h, kurs_val = get_data_engine()
    
    if df_6mo.empty:
        st.error("Gagal Data.")
    else:
        final_report, xau_processed, fib_levels, score_val = generate_sop_report(df_6mo, df_4h, kurs_val)

        # --- SIDEBAR ---
        st.sidebar.header("⚙️ Konfigurasi")
        if "TELEGRAM_TOKEN" in st.secrets:
            bot_token = st.secrets["TELEGRAM_TOKEN"]
            chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        else:
            bot_token = st.sidebar.text_input("Bot Token", type="password")
            chat_id = st.sidebar.text_input("Chat ID")

        st.sidebar.markdown("---")
        
        # Color based on score
        score_color = "off"
        if score_val >= 80: score_color = "normal" # Greenish default
        elif score_val < 50: score_color = "inverse" # Reddish/Alert
        
        st.sidebar.metric("COMPOSITE SCORE", f"{score_val:.1f}/100", delta="Quality", delta_color=score_color)
        st.sidebar.metric("PAXG/USD", fmt_usd(xau_processed.iloc[-1]['Close']))

        # --- CHART ---
        st.subheader("📊 Chart Daily (Quantitative View)")
        fig = go.Figure(data=[go.Candlestick(x=xau_processed.index,
                                open=xau_processed['Open'], high=xau_processed['High'],
                                low=xau_processed['Low'], close=xau_processed['Close'],
                                name='PAXG/USD')])
        fig.add_trace(go.Scatter(x=xau_processed.index, y=xau_processed['EMA200'], line=dict(color='blue', width=2), name='EMA 200'))
        fig.add_trace(go.Scatter(x=xau_processed.index, y=xau_processed['BBU'], line=dict(color='red', width=1, dash='dot'), name='Upper BB'))
        fig.add_trace(go.Scatter(x=xau_processed.index, y=xau_processed['BBL'], line=dict(color='green', width=1, dash='dot'), name='Lower BB'))
        
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- REPORT ---
        st.subheader("📋 Output Logika SOP (Quantitative)")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("📩 Kirim ke Telegram"):
                success, msg = send_telegram_alert(bot_token, chat_id, final_report)
                if success: st.success("Terkirim!")
                else: st.error(f"Gagal: {msg}")

        st.text_area("Report:", value=final_report, height=700, label_visibility="collapsed")
