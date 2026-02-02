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

# --- FUNGSI INDIKATOR ---
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

# --- SCORING ENGINE (DETAIL RESTORED) ---
def calculate_quant_score(row, prev_row, poc, fibo_golden):
    price = row['Close']
    scores = {}
    details = {}

    # 1. EMA 200 (Weight 30%)
    ema = row['EMA200']
    dist_pct = ((price - ema) / ema) * 100
    if dist_pct >= 12: score_ema = 100
    elif 8 <= dist_pct < 12: score_ema = 85
    elif 4 <= dist_pct < 8: score_ema = 70
    elif 0 <= dist_pct < 4: score_ema = 55
    else: score_ema = 30 
    
    scores['EMA'] = score_ema
    # DETAIL DIKEMBALIKAN:
    trend_txt = "BULLISH" if price > ema else "BEARISH"
    details['EMA'] = f"{trend_txt} | Price ${price:.0f} vs EMA ${ema:.0f} ({dist_pct:+.1f}%)"

    # 2. VPVR POC (Weight 20%)
    if price > poc: score_vpvr = 100 
    elif abs(price - poc)/poc < 0.03: score_vpvr = 60 
    else: score_vpvr = 30 
    
    scores['VPVR'] = score_vpvr
    # DETAIL DIKEMBALIKAN:
    pos_txt = "Above Wall" if price > poc else "Below Wall"
    details['VPVR'] = f"{pos_txt} | POC: ${poc:.0f}"

    # 3. MACD (Weight 15%)
    hist = row['MACD_Hist']
    prev_hist = prev_row['MACD_Hist']
    if hist > 0 and hist > prev_hist: score_macd = 100
    elif hist > 0: score_macd = 70
    elif hist < 0 and hist > prev_hist: score_macd = 50
    else: score_macd = 30
    
    scores['MACD'] = score_macd
    # DETAIL DIKEMBALIKAN:
    line = row['MACD']
    sig = row['MACD_Signal']
    details['MACD'] = f"Hist: {hist:+.2f} | Line: {line:.1f} | Sig: {sig:.1f}"

    # 4. Stoch RSI (Weight 10%)
    k = row['STOCHRSIk']
    d = row['STOCHRSId']
    if k < 20 and k > d: score_stoch = 90
    elif k < 20: score_stoch = 65
    elif k > 80: score_stoch = 35
    else: score_stoch = 50
    
    scores['STOCH'] = score_stoch
    # DETAIL DIKEMBALIKAN:
    details['STOCH'] = f"Value: {k:.1f} (D: {d:.1f})"

    # 5. Bollinger Bands (Weight 10%)
    if price <= row['BBL']: score_bb = 85
    elif price < row['BBM']: score_bb = 65
    elif price < row['BBU']: score_bb = 35
    else: score_bb = 20
    
    scores['BB'] = score_bb
    # DETAIL DIKEMBALIKAN:
    details['BB'] = f"Upper: ${row['BBU']:.0f} | Lower: ${row['BBL']:.0f}"

    # 6. Fibonacci (Weight 15%)
    dist_fibo_pct = abs((price - fibo_golden) / fibo_golden) * 100
    if dist_fibo_pct <= 1.5: score_fibo = 90
    elif price > fibo_golden: score_fibo = 55
    elif price < fibo_golden * 0.95: score_fibo = 40
    else: score_fibo = 75
    
    scores['FIBO'] = score_fibo
    # DETAIL DIKEMBALIKAN:
    details['FIBO'] = f"Dist to Golden: {dist_fibo_pct:.1f}% | Target: ${fibo_golden:.0f}"

    # COMPOSITE SCORE
    final_score = (
        (scores['EMA'] * 0.30) + (scores['VPVR'] * 0.20) + (scores['MACD'] * 0.15) +
        (scores['STOCH'] * 0.10) + (scores['BB'] * 0.10) + (scores['FIBO'] * 0.15)
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

        paxg_d = process_data_smart(paxg_d)
        paxg_6mo = paxg_d.tail(180).copy()
        
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

# --- REPORT GENERATOR ---
def generate_sop_report(df_6mo, df_4h, kurs):
    last_d = df_6mo.iloc[-1]
    prev_d = df_6mo.iloc[-2]
    
    fibo = calculate_fibonacci_levels(df_6mo) 
    poc = get_poc(df_6mo)
    
    # 1. HITUNG SKOR KUANTITATIF
    final_score, scores, details = calculate_quant_score(last_d, prev_d, poc, fibo['GOLDEN (0.618)'])
    
    # 2. PROBABILITY MODEL
    if final_score >= 80:
        decision = "🚀 STRONG BUY"
        prob = "High Probability (>80%)"
        dana_market, dana_limit = MODAL_GAJI * 0.7, MODAL_GAJI * 0.3
    elif 65 <= final_score < 80:
        decision = "✅ GRADUAL BUY"
        prob = "Moderate Probability (65-79%)"
        dana_market, dana_limit = MODAL_GAJI * 0.5, MODAL_GAJI * 0.5
    elif 50 <= final_score < 65:
        decision = "⚠️ WAIT PULLBACK"
        prob = "Low Probability (Wait for Dip)"
        dana_market, dana_limit = MODAL_GAJI * 0.2, MODAL_GAJI * 0.8
    else:
        decision = "🛑 AVOID ENTRY"
        prob = "Negative/Risk High"
        dana_market, dana_limit = 0, MODAL_GAJI

    # 3. LIMIT TARGETING
    candidates = [poc, last_d['EMA200'], fibo['0.382'], fibo['MID (0.5)'], fibo['GOLDEN (0.618)']]
    valid_supports = [x for x in candidates if x < last_d['Close']]
    target_limit_usd = max(valid_supports) if valid_supports else fibo['MID (0.5)']
    est_limit_idr = target_limit_usd * kurs * SPREAD_AJAIB

    now = datetime.now(pytz.timezone('Asia/Jakarta'))
    
    report = f"""🦅 GOLD MASTER QUANTITATIVE (V4.1 FIXED)
📅 Waktu: {now.strftime('%d %b %Y | %H:%M WIB')}
=======================================

💰 UPDATE HARGA
PAXG/USD : {fmt_usd(last_d['Close'])}
KURS IDR : {fmt_idr(kurs)}

📊 MATRIX 6 INDIKATOR (DETAILED)
1. EMA 200    [{scores['EMA']}]
   👉 {details['EMA']}

2. VPVR POC   [{scores['VPVR']}]
   👉 {details['VPVR']}

3. MACD       [{scores['MACD']}]
   👉 {details['MACD']}

4. Stoch RSI  [{scores['STOCH']}]
   👉 {details['STOCH']}

5. Bollinger  [{scores['BB']}]
   👉 {details['BB']}

6. Fibonacci  [{scores['FIBO']}]
   👉 {details['FIBO']}

🧮 COMPOSITE SCORE: {final_score:.1f} / 100
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
st.title("Gold Master Quantitative (Detailed View)")

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
        score_color = "normal" if score_val >= 65 else "inverse"
        st.sidebar.metric("COMPOSITE SCORE", f"{score_val:.1f}/100", delta="Quality", delta_color=score_color)
        st.sidebar.metric("PAXG/USD", fmt_usd(xau_processed.iloc[-1]['Close']))

        # --- CHART (DENGAN FIBO & EMA KEMBALI) ---
        st.subheader("📊 Chart Daily + Indicators")
        fig = go.Figure(data=[go.Candlestick(x=xau_processed.index,
                                open=xau_processed['Open'], high=xau_processed['High'],
                                low=xau_processed['Low'], close=xau_processed['Close'],
                                name='PAXG/USD')])
        
        # EMA 200
        fig.add_trace(go.Scatter(x=xau_processed.index, y=xau_processed['EMA200'], line=dict(color='blue', width=2), name='EMA 200'))
        
        # Bollinger
        fig.add_trace(go.Scatter(x=xau_processed.index, y=xau_processed['BBU'], line=dict(color='red', width=1, dash='dot'), name='Upper BB'))
        fig.add_trace(go.Scatter(x=xau_processed.index, y=xau_processed['BBL'], line=dict(color='green', width=1, dash='dot'), name='Lower BB'))
        
        # ✅ FIBONACCI LINES (RESTORED)
        colors_fib = {"MOONBAG": "lime", "RESISTANCE": "red", "GOLDEN": "gold", "FLOOR": "white", "MID": "gray"}
        for label, val in fib_levels.items():
            c = "gray"
            for k, v in colors_fib.items():
                if k in label: c = v
            fig.add_hline(y=val, line_dash="dash", line_color=c, annotation_text=f"{label}")
            
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
