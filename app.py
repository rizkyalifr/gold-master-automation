import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import pytz

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Gold Master Institutional V6", page_icon="🦅", layout="wide")

# --- CSS ---
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
    
    # 2. Bollinger Bands (Volatility Proxy)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BBU'] = df['SMA20'] + (df['STD20'] * 2)
    df['BBL'] = df['SMA20'] - (df['STD20'] * 2)
    df['BBM'] = df['SMA20']
    df['BB_Width'] = (df['BBU'] - df['BBL']) / df['BBM']
    
    # 3. MACD (Momentum)
    k = df['Close'].ewm(span=12, adjust=False).mean()
    d = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = k - d
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 4. Stoch RSI (Timing)
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

# --- HITUNG POC ---
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

# --- SCORING ENGINE ---
def calculate_quant_score(row, prev_row, poc, fibo_golden):
    price = row['Close']
    scores = {}
    details = {}
    bullish_flags = 0

    # 1. EMA 200 (Weight 30%)
    ema = row['EMA200']
    dist_pct = ((price - ema) / ema) * 100
    if dist_pct >= 12: score_ema = 100
    elif 8 <= dist_pct < 12: score_ema = 85
    elif 4 <= dist_pct < 8: score_ema = 70
    elif 0 <= dist_pct < 4: score_ema = 55
    else: score_ema = 30 
    scores['EMA'] = score_ema
    details['EMA'] = f"Price vs EMA ({dist_pct:+.1f}%)"
    if score_ema >= 55: bullish_flags += 1

    # 2. VPVR POC (Weight 20%)
    if price > poc: score_vpvr = 100 
    elif abs(price - poc)/poc < 0.03: score_vpvr = 60 
    else: score_vpvr = 30 
    scores['VPVR'] = score_vpvr
    pos_txt = "Above" if price > poc else "Below"
    details['VPVR'] = f"{pos_txt} Volume Wall"
    if score_vpvr >= 60: bullish_flags += 1

    # 3. MACD (Weight 15%)
    hist = row['MACD_Hist']
    prev_hist = prev_row['MACD_Hist']
    if hist > 0 and hist > prev_hist: score_macd = 100
    elif hist > 0: score_macd = 70
    elif hist < 0 and hist > prev_hist: score_macd = 50
    else: score_macd = 30
    scores['MACD'] = score_macd
    details['MACD'] = f"Hist: {hist:+.2f}"
    if score_macd >= 70: bullish_flags += 1

    # 4. Stoch RSI (Weight 10%)
    k = row['STOCHRSIk']
    d = row['STOCHRSId']
    if k < 20 and k > d: score_stoch = 90
    elif k < 20: score_stoch = 65
    elif k > 80: score_stoch = 35
    else: score_stoch = 50
    scores['STOCH'] = score_stoch
    details['STOCH'] = f"K: {k:.1f}"
    if score_stoch >= 65: bullish_flags += 1

    # 5. Bollinger Bands (Weight 10%)
    if price <= row['BBL']: score_bb = 85
    elif price < row['BBM']: score_bb = 65
    elif price < row['BBU']: score_bb = 35
    else: score_bb = 20
    scores['BB'] = score_bb
    details['BB'] = "Lower Half" if price < row['BBM'] else "Upper Half"
    if score_bb >= 65: bullish_flags += 1

    # 6. Fibonacci (Weight 15%)
    dist_fibo_pct = abs((price - fibo_golden) / fibo_golden) * 100
    if dist_fibo_pct <= 1.5: score_fibo = 90
    elif price > fibo_golden: score_fibo = 55
    elif price < fibo_golden * 0.95: score_fibo = 40
    else: score_fibo = 75
    scores['FIBO'] = score_fibo
    details['FIBO'] = f"Dist Golden: {dist_fibo_pct:.1f}%"
    if score_fibo >= 55: bullish_flags += 1

    # COMPOSITE SCORE
    final_score = (
        (scores['EMA'] * 0.30) + (scores['VPVR'] * 0.20) + (scores['MACD'] * 0.15) +
        (scores['STOCH'] * 0.10) + (scores['BB'] * 0.10) + (scores['FIBO'] * 0.15)
    )
    
    return final_score, scores, details, bullish_flags

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

# --- REPORT GENERATOR (V6: INSTITUTIONAL LOGIC) ---
def generate_sop_report(df_6mo, df_4h, kurs):
    last_d = df_6mo.iloc[-1]
    prev_d = df_6mo.iloc[-2]
    price = last_d['Close']
    
    fibo = calculate_fibonacci_levels(df_6mo) 
    poc = get_poc(df_6mo)
    
    # 1. QUANTITATIVE SCORE
    final_score, scores, details, bullish_count = calculate_quant_score(last_d, prev_d, poc, fibo['GOLDEN (0.618)'])
    
    # 2. MARKET REGIME & VOLATILITY (ISSUE #3 FIX)
    ema_dist = ((price - last_d['EMA200']) / last_d['EMA200']) * 100
    bb_width = last_d.get('BB_Width', 0.1) # Default 0.1 if calc fail
    
    # Volatility Classification
    is_high_vol = bb_width > 0.15
    vol_status = "⚡ HIGH VOLATILITY" if is_high_vol else "🌊 NORMAL/LOW VOL"
    
    # 3. MOMENTUM DECAY CHECK (ISSUE #5 FIX)
    momentum_decay = last_d['MACD_Hist'] < prev_d['MACD_Hist'] and last_d['MACD_Hist'] > 0
    
    # 4. SELL / EXIT LOGIC (LADDER SYSTEM - ISSUE #1 & #7 FIX)
    # Priority: Risk Management > Profit Taking
    action_type = "BUY"
    sell_reason = ""
    sell_pct = 0
    
    # Tier 1: Risk Exit (Score Hancur / Trend Broken)
    if final_score < 40: 
        action_type = "SELL"
        sell_reason = "Score < 40 (Defensive Exit)"
        sell_pct = 50 # Jual separuh buat amanin cash
        
    # Tier 2: Technical Exit (Partial TP)
    elif price >= fibo['MOONBAG (1.618)']:
        action_type = "SELL"
        sell_reason = "Moonbag Target (1.618)"
        sell_pct = 50
    elif price >= fibo['RESISTANCE (High)'] and momentum_decay:
        action_type = "SELL"
        sell_reason = "Resistance + Momentum Decay"
        sell_pct = 30
    elif price >= fibo['0.236'] and last_d['STOCHRSIk'] > 80:
        action_type = "SELL"
        sell_reason = "Overbought at Resistance 0.236"
        sell_pct = 20

    # 5. BUY LOGIC & SIZING (VOLATILITY ADJUSTED - ISSUE #3 FIX)
    dana_market = 0
    dana_limit = 0
    decision_title = ""
    prob_desc = ""
    
    if action_type == "BUY":
        # Base Allocation based on Score (ISSUE #2 & #6 FIX)
        if final_score >= 80:
            alloc_market, alloc_limit = 0.7, 0.3
            decision_title = "🚀 AGGRESSIVE BUY"
            prob_desc = "High Probability Setup"
        elif 65 <= final_score < 80:
            alloc_market, alloc_limit = 0.5, 0.5
            decision_title = "✅ STANDARD ACCUMULATION"
            prob_desc = "Moderate/Healthy Trend"
        elif 50 <= final_score < 65:
            alloc_market, alloc_limit = 0.2, 0.8
            decision_title = "⚠️ SNIPER ENTRY (WAIT DIP)"
            prob_desc = "Price Extended / Wait Pullback"
        else: # Score 40-50 (Grey Area)
            alloc_market, alloc_limit = 0.0, 1.0
            decision_title = "🛡️ DEFENSIVE / WAIT"
            prob_desc = "Weak Structure"

        # Volatility Adjustment (Institutional Rule)
        if is_high_vol and final_score >= 50:
            # Kalau volatil tinggi, kurangi market order, perbesar limit (hindari slippage/whipsaw)
            alloc_market *= 0.7 # Reduce market exposure
            alloc_limit = 1.0 - alloc_market # Shift to limit
            prob_desc += " (Vol Adjusted)"

        dana_market = MODAL_GAJI * alloc_market
        dana_limit = MODAL_GAJI * alloc_limit

    # 6. SMART LIMIT TARGETING (ISSUE #2 FIX)
    # Pilih support terdekat yg masuk akal, bukan cuma max()
    candidates = [
        {'price': poc, 'label': 'POC'},
        {'price': last_d['EMA200'], 'label': 'EMA 200'},
        {'price': fibo['0.382'], 'label': 'Fibo 0.382'},
        {'price': fibo['MID (0.5)'], 'label': 'Fibo 0.5'},
        {'price': fibo['GOLDEN (0.618)'], 'label': 'Golden'}
    ]
    
    # Filter yang di bawah harga sekarang
    valid_supports = [c for c in candidates if c['price'] < price]
    
    if valid_supports:
        # Sort by distance to current price (descending price)
        valid_supports.sort(key=lambda x: x['price'], reverse=True)
        # Ambil yang terdekat (index 0)
        target_limit_usd = valid_supports[0]['price']
        target_label = valid_supports[0]['label']
    else:
        target_limit_usd = fibo['MID (0.5)']
        target_label = "Deep Support"
        
    est_limit_idr = target_limit_usd * kurs * SPREAD_AJAIB

    # 7. CONSTRUCTION REPORT TEXT
    now = datetime.now(pytz.timezone('Asia/Jakarta'))
    
    if action_type == "SELL":
        main_action_txt = f"""
🚨 **SELL SIGNAL TRIGGERED**
👉 **Action:** JUAL {sell_pct}% Posisi
👉 **Alasan:** {sell_reason}
👉 **Next:** Simpan Cash/USDT, tunggu Score membaik atau harga diskon.
        """
    else:
        main_action_txt = f"""
1️⃣ **MARKET ORDER**
   👉 Nominal: {fmt_idr(dana_market)}
   👉 Eksekusi: SEKARANG.

2️⃣ **LIMIT ORDER**
   👉 Nominal: {fmt_idr(dana_limit)}
   👉 Target: {fmt_usd(target_limit_usd)} ({target_label})
   👉 Est. IDR: {fmt_idr(est_limit_idr)}
        """

    report = f"""🦅 GOLD MASTER V6 (INSTITUTIONAL QUANT)
📅 {now.strftime('%d %b %Y | %H:%M WIB')}
=======================================

💰 **MARKET DATA**
PAXG : {fmt_usd(price)}
EMA  : {fmt_usd(last_d['EMA200'])}
VOL  : {vol_status}

📊 **SCORING ENGINE**
Total Score: **{final_score:.1f} / 100**
• EMA Trend  : {scores['EMA']}
• Vol Profile: {scores['VPVR']}
• Momentum   : {scores['MACD']}
• Timing     : {scores['STOCH']}
• Mean Rev   : {scores['BB']}
• Valuation  : {scores['FIBO']}

🧠 **DECISION MODULE**
Status: **{decision_title}**
Logic : {prob_desc}
=======================================

📋 **EXECUTION PLAN (MODAL 5 JUTA)**
{main_action_txt}

🎯 **KEY LEVELS (FIBO + STRUCTURAL)**
"""
    sorted_fibo = dict(sorted(fibo.items(), key=lambda item: item[1], reverse=True))
    for k, v in sorted_fibo.items():
        report += f"{k:<15}: {fmt_usd(v)}\n"
        
    return report, df_6mo, fibo, final_score

# --- MAIN APP ---
st.title("Gold Master V6 (Institutional)")

with st.spinner("Running Quant Engine..."):
    df_6mo, df_4h, kurs_val = get_data_engine()
    
    if df_6mo.empty:
        st.error("Data Error.")
    else:
        final_report, xau_processed, fib_levels, score_val = generate_sop_report(df_6mo, df_4h, kurs_val)

        # Sidebar
        st.sidebar.header("⚙️ Settings")
        if "TELEGRAM_TOKEN" in st.secrets:
            bot_token = st.secrets["TELEGRAM_TOKEN"]
            chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        else:
            bot_token = st.sidebar.text_input("Bot Token", type="password")
            chat_id = st.sidebar.text_input("Chat ID")

        st.sidebar.markdown("---")
        score_color = "normal" if score_val >= 65 else "inverse"
        st.sidebar.metric("QUANT SCORE", f"{score_val:.1f}", delta="Strength", delta_color=score_color)
        st.sidebar.metric("PRICE", fmt_usd(xau_processed.iloc[-1]['Close']))

        # Chart
        st.subheader("Institutional Chart View")
        fig = go.Figure(data=[go.Candlestick(x=xau_processed.index,
                                open=xau_processed['Open'], high=xau_processed['High'],
                                low=xau_processed['Low'], close=xau_processed['Close'],
                                name='PAXG/USD')])
        
        fig.add_trace(go.Scatter(x=xau_processed.index, y=xau_processed['EMA200'], line=dict(color='blue', width=2), name='EMA 200'))
        fig.add_trace(go.Scatter(x=xau_processed.index, y=xau_processed['BBU'], line=dict(color='red', width=1, dash='dot'), name='Upper BB'))
        fig.add_trace(go.Scatter(x=xau_processed.index, y=xau_processed['BBL'], line=dict(color='green', width=1, dash='dot'), name='Lower BB'))
        
        # Fibo
        colors_fib = {"MOONBAG": "lime", "RESISTANCE": "red", "GOLDEN": "gold", "FLOOR": "white", "MID": "gray"}
        for label, val in fib_levels.items():
            c = "gray"
            for k, v in colors_fib.items():
                if k in label: c = v
            fig.add_hline(y=val, line_dash="dash", line_color=c, annotation_text=f"{label}")
            
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Output
        st.subheader("📋 Institutional Execution Report")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("📩 Broadcast Telegram"):
                success, msg = send_telegram_alert(bot_token, chat_id, final_report)
                if success: st.success("Sent!")
                else: st.error(f"Failed: {msg}")

        st.text_area("Log Output:", value=final_report, height=700, label_visibility="collapsed")
