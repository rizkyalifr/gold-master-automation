import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
import pytz
from scipy.optimize import minimize
from scipy.signal import argrelextrema
import warnings

warnings.filterwarnings("ignore")

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Gold Master V6 AI Enhanced", page_icon="🦅", layout="wide")

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

# ==========================================
# 1. AI OPTIMIZATION ENGINE (THE BRAIN)
# ==========================================

# --- LOSS FUNCTIONS ---
def loss_ema(params, df, pivot_prices, pivot_indices):
    period = int(params[0])
    if period < 50 or period > 300: return 1e9 # Bounds wajar untuk long term trend
    ema = df['Close'].ewm(span=period, adjust=False).mean()
    ema_vals = ema.iloc[pivot_indices].values
    diffs = pivot_prices - ema_vals
    penalty = diffs < -(pivot_prices * 0.01) # Penalty jika jebol support
    error = np.abs(diffs)
    error[penalty] *= 10 
    return np.mean(error)

def loss_bb(params, df, high_idx, low_idx):
    window, std_dev = int(params[0]), params[1]
    if window < 10 or window > 50 or std_dev < 1.5 or std_dev > 3.0: return 1e9
    sma = df['Close'].rolling(window).mean()
    std = df['Close'].rolling(window).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    # Minimalkan jarak band ke ekor candle pivot
    # Handling NaN dengan fillna method backfill
    upper = upper.fillna(method='bfill')
    lower = lower.fillna(method='bfill')
    err_h = np.abs(df['High'].iloc[high_idx] - upper.iloc[high_idx]).mean()
    err_l = np.abs(df['Low'].iloc[low_idx] - lower.iloc[low_idx]).mean()
    return err_h + err_l

def loss_macd(params, df):
    fast, slow, signal = int(params[0]), int(params[1]), int(params[2])
    if fast < 5 or slow < 15 or signal < 5: return 1e9
    if fast >= slow: return 1e9
    
    # Simple strategy: Crossover profit maximization
    k = df['Close'].ewm(span=fast, adjust=False).mean()
    d = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = k - d
    sig = macd.ewm(span=signal, adjust=False).mean()
    
    cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    cross_down = (macd < sig) & (macd.shift(1) >= sig.shift(1))
    
    entries = df.loc[cross_up, 'Close']
    exits = df.loc[cross_down, 'Close']
    
    if len(entries) == 0 or len(exits) == 0: return 1e9
    
    # Simple profit sum logic
    n = min(len(entries), len(exits))
    profits = (exits.values[:n] - entries.values[:n]) / entries.values[:n]
    return -np.sum(profits) # Negative profit agar minimize cari maximum

def loss_fibo(params, future_lows, orig_low, orig_high):
    guess_low, guess_high = params
    # Bounds: Max geser 3%
    if abs(guess_low - orig_low) > (orig_low * 0.03): return 1e9
    if abs(guess_high - orig_high) > (orig_high * 0.03): return 1e9
    if guess_low >= guess_high: return 1e9

    diff = guess_high - guess_low
    targets = [guess_high - (diff * 0.618), guess_high - (diff * 0.5)]
    errors = []
    for val in future_lows:
        dists = [abs(val - t) for t in targets]
        errors.append(min(dists))
    return np.mean(errors) if errors else 1e9

def loss_stoch_rsi(params, df):
    length, k_smooth, d_smooth = int(params[0]), int(params[1]), int(params[2])
    if length < 10 or length > 30 or k_smooth < 2 or d_smooth < 2: return 1e9
    
    # Calc Stoch RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    min_rsi = rsi.rolling(window=length).min()
    max_rsi = rsi.rolling(window=length).max()
    stoch = (rsi - min_rsi) / (max_rsi - min_rsi)
    k_line = stoch.rolling(window=k_smooth).mean() * 100
    d_line = k_line.rolling(window=d_smooth).mean()
    
    # Strategy: Buy when K crosses D upward below 20, Sell when K crosses D downward above 80
    buy_sig = (k_line > d_line) & (k_line.shift(1) <= d_line.shift(1)) & (k_line < 25)
    sell_sig = (k_line < d_line) & (k_line.shift(1) >= d_line.shift(1)) & (k_line > 75)
    
    entries = df.loc[buy_sig, 'Close']
    exits = df.loc[sell_sig, 'Close']
    
    if len(entries) == 0 or len(exits) == 0: return 1e9
    
    n = min(len(entries), len(exits))
    profits = (exits.values[:n] - entries.values[:n]) / entries.values[:n]
    return -np.sum(profits)

# --- AI RUNNER ---
@st.cache_data(ttl=3600) # Cache 1 jam agar tidak berat
def run_ai_optimizer(df):
    # Prepare Pivot Data
    high_idx = argrelextrema(df['High'].values, np.greater, order=5)[0]
    low_idx = argrelextrema(df['Low'].values, np.less, order=5)[0]
    
    # 1. OPTIMIZE EMA
    res_ema = minimize(loss_ema, x0=[200], args=(df, df['Low'].iloc[low_idx], low_idx), method='Nelder-Mead', tol=1.0)
    opt_ema = int(res_ema.x[0])
    
    # 2. OPTIMIZE BB
    res_bb = minimize(loss_bb, x0=[20, 2.0], args=(df, high_idx, low_idx), method='Nelder-Mead', tol=0.1)
    opt_bb_win, opt_bb_std = int(res_bb.x[0]), res_bb.x[1]
    
    # 3. OPTIMIZE MACD
    res_macd = minimize(loss_macd, x0=[12, 26, 9], args=(df,), method='Nelder-Mead', tol=0.1)
    opt_macd = (int(res_macd.x[0]), int(res_macd.x[1]), int(res_macd.x[2]))
    
    # 4. OPTIMIZE STOCH RSI
    res_stoch = minimize(loss_stoch_rsi, x0=[14, 3, 3], args=(df,), method='Nelder-Mead', tol=0.1)
    opt_stoch = (int(res_stoch.x[0]), int(res_stoch.x[1]), int(res_stoch.x[2]))
    
    # 5. OPTIMIZE FIBO
    mid = len(df) // 2
    orig_l, orig_h = df['Low'].min(), df['High'].max()
    future_lows = df['Low'].iloc[mid:].values # Pakai data separuh akhir untuk validasi
    res_fibo = minimize(loss_fibo, x0=[orig_l, orig_h], args=(future_lows, orig_l, orig_h), method='Nelder-Mead', tol=0.1)
    opt_fibo = (res_fibo.x[0], res_fibo.x[1])
    
    return {
        "EMA": opt_ema,
        "BB": (opt_bb_win, opt_bb_std),
        "MACD": opt_macd,
        "STOCH": opt_stoch,
        "FIBO_ANCHORS": opt_fibo
    }

# ==========================================
# 2. DATA PROCESSING (DYNAMIC)
# ==========================================

def process_data_ai(df, params):
    df = df.copy()
    
    # 1. AI EMA
    df['EMA200'] = df['Close'].ewm(span=params['EMA'], adjust=False).mean()
    
    # 2. AI Bollinger
    win, std = params['BB']
    df['SMA20'] = df['Close'].rolling(window=win).mean()
    df['STD20'] = df['Close'].rolling(window=win).std()
    df['BBU'] = df['SMA20'] + (df['STD20'] * std)
    df['BBL'] = df['SMA20'] - (df['STD20'] * std)
    df['BBM'] = df['SMA20']
    df['BB_Width'] = (df['BBU'] - df['BBL']) / df['BBM']
    
    # 3. AI MACD
    fast, slow, sig = params['MACD']
    k = df['Close'].ewm(span=fast, adjust=False).mean()
    d = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = k - d
    df['MACD_Signal'] = df['MACD'].ewm(span=sig, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 4. AI Stoch RSI
    length, k_smooth, d_smooth = params['STOCH']
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    min_rsi = df['RSI'].rolling(window=length).min()
    max_rsi = df['RSI'].rolling(window=length).max()
    stoch = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)
    df['STOCHRSIk'] = stoch.rolling(window=k_smooth).mean() * 100
    df['STOCHRSId'] = df['STOCHRSIk'].rolling(window=d_smooth).mean()
    
    return df

# --- HITUNG POC ---
def get_poc(df):
    price_bins = pd.cut(df['Close'], bins=50)
    vpvr = df.groupby(price_bins, observed=True)['Volume'].sum()
    return vpvr.idxmax().mid

# --- AI FIBONACCI LEVELS ---
def calculate_fibonacci_levels(df, anchors):
    low, high = anchors # Ini hasil optimasi AI
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

# ==========================================
# 3. SCORING ENGINE (LOGIC SAMA)
# ==========================================

def calculate_quant_score(row, prev_row, poc, fibo_golden, opt_params):
    price = row['Close']
    scores = {}
    details = {}
    bullish_flags = 0

    # 1. EMA (AI)
    ema = row['EMA200']
    dist_pct = ((price - ema) / ema) * 100
    if dist_pct >= 12: score_ema = 100
    elif 8 <= dist_pct < 12: score_ema = 85
    elif 4 <= dist_pct < 8: score_ema = 70
    elif 0 <= dist_pct < 4: score_ema = 55
    else: score_ema = 30 
    scores['EMA'] = score_ema
    trend_txt = "BULLISH" if price > ema else "BEARISH"
    details['EMA'] = f"AI-EMA({opt_params['EMA']}) | Price ${price:.0f} vs EMA ${ema:.0f} ({dist_pct:+.1f}%)"
    if score_ema >= 55: bullish_flags += 1

    # 2. VPVR POC
    if price > poc: score_vpvr = 100 
    elif abs(price - poc)/poc < 0.03: score_vpvr = 60 
    else: score_vpvr = 30 
    scores['VPVR'] = score_vpvr
    pos_txt = "Above Wall" if price > poc else "Below Wall"
    details['VPVR'] = f"{pos_txt} | POC: ${poc:.0f}"
    if score_vpvr >= 60: bullish_flags += 1

    # 3. MACD (AI)
    hist = row['MACD_Hist']
    prev_hist = prev_row['MACD_Hist']
    if hist > 0 and hist > prev_hist: score_macd = 100
    elif hist > 0: score_macd = 70
    elif hist < 0 and hist > prev_hist: score_macd = 50
    else: score_macd = 30
    scores['MACD'] = score_macd
    p = opt_params['MACD']
    details['MACD'] = f"AI-MACD({p[0]},{p[1]},{p[2]}) | Hist: {hist:+.2f}"
    if score_macd >= 70: bullish_flags += 1

    # 4. Stoch RSI (AI)
    k = row['STOCHRSIk']
    d = row['STOCHRSId']
    if k < 20 and k > d: score_stoch = 90
    elif k < 20: score_stoch = 65
    elif k > 80: score_stoch = 35
    else: score_stoch = 50
    scores['STOCH'] = score_stoch
    p = opt_params['STOCH']
    details['STOCH'] = f"AI-Stoch({p[0]},{p[1]},{p[2]}) | K: {k:.1f} | D: {d:.1f}"
    if score_stoch >= 65: bullish_flags += 1

    # 5. Bollinger (AI)
    if price <= row['BBL']: score_bb = 85
    elif price < row['BBM']: score_bb = 65
    elif price < row['BBU']: score_bb = 35
    else: score_bb = 20
    scores['BB'] = score_bb
    p = opt_params['BB']
    details['BB'] = f"AI-BB({p[0]},{p[1]:.1f}) | Up: ${row['BBU']:.0f} | Low: ${row['BBL']:.0f}"
    if score_bb >= 65: bullish_flags += 1

    # 6. Fibonacci (AI)
    dist_fibo_pct = abs((price - fibo_golden) / fibo_golden) * 100
    if dist_fibo_pct <= 1.5: score_fibo = 90
    elif price > fibo_golden: score_fibo = 55
    elif price < fibo_golden * 0.95: score_fibo = 40
    else: score_fibo = 75
    scores['FIBO'] = score_fibo
    details['FIBO'] = f"AI-Fibo | Dist to Golden: {dist_fibo_pct:.1f}% | Target: ${fibo_golden:.0f}"
    if score_fibo >= 55: bullish_flags += 1

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
            return pd.DataFrame(), pd.DataFrame(), 16800, {}

        for df in [paxg_d, paxg_h]:
            df['Close'] *= PAXG_MULTIPLIER
            df['High'] *= PAXG_MULTIPLIER
            df['Low'] *= PAXG_MULTIPLIER
            df['Open'] *= PAXG_MULTIPLIER

        # --- JALANKAN AI DISINI ---
        # Kita optimasi berdasarkan data daily (6 bulan terakhir cukup)
        opt_data = paxg_d.tail(200).copy()
        ai_params = run_ai_optimizer(opt_data)

        # Apply Params ke Data
        paxg_d = process_data_ai(paxg_d, ai_params)
        paxg_6mo = paxg_d.tail(180).copy()
        
        paxg_4h = paxg_h.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        paxg_4h = process_data_ai(paxg_4h, ai_params)

    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), 16800, {}
        
    return paxg_6mo, paxg_4h, kurs, ai_params

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
def generate_sop_report(df_6mo, df_4h, kurs, ai_params):
    last_d = df_6mo.iloc[-1]
    prev_d = df_6mo.iloc[-2]
    price = last_d['Close']
    
    # Pakai AI Fibo Anchors
    fibo = calculate_fibonacci_levels(df_6mo, ai_params['FIBO_ANCHORS']) 
    poc = get_poc(df_6mo)
    
    # 1. QUANT SCORE
    final_score, scores, details, bullish_count = calculate_quant_score(last_d, prev_d, poc, fibo['GOLDEN (0.618)'], ai_params)
    
    # 2. MARKET REGIME
    ema_dist = ((price - last_d['EMA200']) / last_d['EMA200']) * 100
    if ema_dist > 5: regime = "🐎 TRENDING BULLISH"
    elif ema_dist < -5: regime = "🐻 TRENDING BEARISH"
    else: regime = "🦀 RANGING / SIDEWAYS"
    
    bb_width = last_d.get('BB_Width', 0.1)
    is_high_vol = bb_width > 0.15
    vol_status = "⚡ HIGH VOLATILITY" if is_high_vol else "🌊 NORMAL/LOW VOL"
    
    # 3. SELL LOGIC
    action_type = "BUY"
    sell_reason = ""
    sell_pct = 0
    momentum_decay = last_d['MACD_Hist'] < prev_d['MACD_Hist'] and last_d['MACD_Hist'] > 0
    
    if final_score < 40: 
        action_type = "SELL"
        sell_reason = "Score < 40 (Defensive Exit)"
        sell_pct = 50 
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
        sell_reason = "Overbought at Resistance"
        sell_pct = 20

    # 4. BUY LOGIC & SIZING
    dana_market = 0
    dana_limit = 0
    decision_title = ""
    prob_desc = ""
    
    if action_type == "BUY":
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
        else: 
            alloc_market, alloc_limit = 0.0, 1.0
            decision_title = "🛡️ DEFENSIVE / WAIT"
            prob_desc = "Weak Structure"

        if is_high_vol and final_score >= 50:
            alloc_market *= 0.7 
            alloc_limit = 1.0 - alloc_market 
            prob_desc += " (Vol Adjusted)"

        dana_market = MODAL_GAJI * alloc_market
        dana_limit = MODAL_GAJI * alloc_limit

    # 5. SMART LIMIT
    candidates = [
        {'price': poc, 'label': 'POC'},
        {'price': last_d['EMA200'], 'label': f"EMA {ai_params['EMA']}"},
        {'price': fibo['0.382'], 'label': 'Fibo 0.382'},
        {'price': fibo['MID (0.5)'], 'label': 'Fibo 0.5'},
        {'price': fibo['GOLDEN (0.618)'], 'label': 'Golden'}
    ]
    valid_supports = [c for c in candidates if c['price'] < price]
    
    if valid_supports:
        valid_supports.sort(key=lambda x: x['price'], reverse=True)
        target_limit_usd = valid_supports[0]['price']
        target_label = valid_supports[0]['label']
    else:
        target_limit_usd = fibo['MID (0.5)']
        target_label = "Deep Support"
        
    est_limit_idr = target_limit_usd * kurs * SPREAD_AJAIB

    # 6. TEXT REPORT
    now = datetime.now(pytz.timezone('Asia/Jakarta'))
    agreement = f"{bullish_count}/6 Bullish"
    
    if action_type == "SELL":
        main_action_txt = f"""
🚨 **SELL SIGNAL TRIGGERED**
👉 **Action:** JUAL {sell_pct}% Posisi
👉 **Alasan:** {sell_reason}
👉 **Next:** Simpan Cash, tunggu Score membaik.
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

    report = f"""🦅 GOLD MASTER V6 (AI ENHANCED)
📅 {now.strftime('%d %b %Y | %H:%M WIB')}
=======================================

💰 **MARKET DATA**
PAXG : {fmt_usd(price)}
EMA  : {fmt_usd(last_d['EMA200'])} (Per: {ai_params['EMA']})
VOL  : {vol_status}

📊 **MATRIX 6 INDIKATOR (AI OPTIMIZED)**
1. EMA       [{scores['EMA']}] {details['EMA']}
2. VPVR POC  [{scores['VPVR']}] {details['VPVR']}
3. MACD      [{scores['MACD']}] {details['MACD']}
4. Stoch RSI [{scores['STOCH']}] {details['STOCH']}
5. Bollinger [{scores['BB']}] {details['BB']}
6. Fibonacci [{scores['FIBO']}] {details['FIBO']}

🧮 SCORE: {final_score:.1f}/100 | {agreement}
🌍 REGIME: {regime}
=======================================
🧠 DECISION : [ {decision_title} ]
🎲 LOGIC    : {prob_desc}
=======================================

📋 **EXECUTION PLAN (MODAL 5 JUTA)**
{main_action_txt}

🎯 **KEY LEVELS**
"""
    sorted_fibo = dict(sorted(fibo.items(), key=lambda item: item[1], reverse=True))
    for k, v in sorted_fibo.items():
        report += f"{k:<15}: {fmt_usd(v)}\n"
        
    return report, df_6mo, fibo, final_score

# --- MAIN APP ---
st.title("Gold Master V6 (AI Institutional)")

with st.spinner("Initializing AI Engines & Optimizing Parameters..."):
    df_6mo, df_4h, kurs_val, ai_params = get_data_engine()
    
    if df_6mo.empty:
        st.error("Data Error.")
    else:
        final_report, xau_processed, fib_levels, score_val = generate_sop_report(df_6mo, df_4h, kurs_val, ai_params)

        # Sidebar
        st.sidebar.header("⚙️ AI Parameters")
        st.sidebar.info(f"EMA Period: {ai_params['EMA']}")
        st.sidebar.info(f"BB: Win {ai_params['BB'][0]} | Std {ai_params['BB'][1]:.2f}")
        st.sidebar.info(f"MACD: {ai_params['MACD']}")
        st.sidebar.info(f"Stoch: {ai_params['STOCH']}")
        
        st.sidebar.header("⚙️ Telegram")
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
        st.subheader("Institutional Chart View (AI Optimized)")
        fig = go.Figure(data=[go.Candlestick(x=xau_processed.index,
                                open=xau_processed['Open'], high=xau_processed['High'],
                                low=xau_processed['Low'], close=xau_processed['Close'],
                                name='PAXG/USD')])
        
        # EMA
        fig.add_trace(go.Scatter(x=xau_processed.index, y=xau_processed['EMA200'], line=dict(color='blue', width=2), name=f'EMA {ai_params["EMA"]}'))
        # BB
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
