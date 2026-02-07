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
import time

warnings.filterwarnings("ignore")

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Crypto Master V7 Multi-Asset", page_icon="🦅", layout="wide")

# --- CSS PRO ---
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 24px; font-weight: bold; }
    .report-box { 
        font-family: 'Consolas', 'Courier New', monospace; 
        white-space: pre-wrap; 
        background-color: #0e1117; 
        padding: 20px; 
        border-radius: 8px; 
        color: #00ff00; 
        border: 1px solid #333;
        font-size: 14px;
        line-height: 1.5;
    }
    .stButton>button { 
        width: 100%; 
        border-radius: 5px; 
        font-weight: bold;
        height: 50px;
    }
</style>
""", unsafe_allow_html=True)

# --- KONFIGURASI ENGINE DEFAULT ---
MODAL_GAJI = 5000000 
SPREAD_AJAIB = 1.015 
# Multiplier hanya untuk PAXG karena ada selisih harga on-chain vs spot biasa
PAXG_MULTIPLIER = 0.99432278994 

# --- HELPER FORMATTING ---
def fmt_idr(val): return f"Rp {val:,.0f}".replace(",", ".")
def fmt_usd(val): return f"${val:,.2f}"

# ==========================================
# 1. AI OPTIMIZATION ENGINE (MULTI-TIMEFRAME)
# ==========================================

# --- LOSS FUNCTIONS (STRICT BOUNDS) ---
def loss_ema(params, df, pivot_prices, pivot_indices):
    period = int(params[0])
    if period < 50 or period > 300: return 1e12 
    
    ema = df['Close'].ewm(span=period, adjust=False).mean()
    ema_vals = ema.iloc[pivot_indices].values
    
    if len(ema_vals) != len(pivot_prices): return 1e12
    
    diffs = pivot_prices - ema_vals
    penalty = diffs < -(pivot_prices * 0.01)
    error = np.abs(diffs)
    error[penalty] *= 10 
    return np.mean(error)

def loss_bb(params, df, high_idx, low_idx):
    window, std_dev = int(params[0]), params[1]
    if window < 10 or window > 50 or std_dev < 1.5 or std_dev > 3.0: return 1e12
    
    sma = df['Close'].rolling(window).mean()
    std = df['Close'].rolling(window).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    upper = upper.fillna(method='bfill')
    lower = lower.fillna(method='bfill')
    
    valid_h = [i for i in high_idx if i < len(df)]
    valid_l = [i for i in low_idx if i < len(df)]
    
    if not valid_h or not valid_l: return 1e12

    err_h = np.abs(df['High'].iloc[valid_h] - upper.iloc[valid_h]).mean()
    err_l = np.abs(df['Low'].iloc[valid_l] - lower.iloc[valid_l]).mean()
    return err_h + err_l

def loss_macd(params, df):
    fast, slow, signal = int(params[0]), int(params[1]), int(params[2])
    if fast < 5 or slow < 15 or signal < 5: return 1e12
    if fast >= slow: return 1e12
    
    k = df['Close'].ewm(span=fast, adjust=False).mean()
    d = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = k - d
    sig = macd.ewm(span=signal, adjust=False).mean()
    
    cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    cross_down = (macd < sig) & (macd.shift(1) >= sig.shift(1))
    
    entries = df.loc[cross_up, 'Close']
    exits = df.loc[cross_down, 'Close']
    
    if len(entries) == 0 or len(exits) == 0: return 0
    
    n = min(len(entries), len(exits))
    profits = (exits.values[:n] - entries.values[:n]) / entries.values[:n]
    return -np.sum(profits) 

def loss_fibo(params, future_lows, orig_low, orig_high):
    guess_low, guess_high = params
    if abs(guess_low - orig_low) > (orig_low * 0.05): return 1e12
    if abs(guess_high - orig_high) > (orig_high * 0.05): return 1e12
    if guess_low >= guess_high: return 1e12

    diff = guess_high - guess_low
    targets = [guess_high - (diff * 0.618), guess_high - (diff * 0.5)]
    errors = []
    
    for t in targets:
        dist = np.min(np.abs(future_lows - t))
        errors.append(dist)
        
    return np.mean(errors) if errors else 1e12

def loss_stoch_rsi(params, df):
    length, k_smooth, d_smooth = int(params[0]), int(params[1]), int(params[2])
    if length < 10 or length > 30 or k_smooth < 2 or d_smooth < 2: return 1e12
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    
    loss = loss.replace(0, 0.0001)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    min_rsi = rsi.rolling(window=length).min()
    max_rsi = rsi.rolling(window=length).max()
    
    denominator = (max_rsi - min_rsi).replace(0, 0.0001)
    stoch = (rsi - min_rsi) / denominator
    
    k_line = stoch.rolling(window=k_smooth).mean() * 100
    d_line = k_line.rolling(window=d_smooth).mean()
    
    buy_sig = (k_line > d_line) & (k_line.shift(1) <= d_line.shift(1)) & (k_line < 25)
    sell_sig = (k_line < d_line) & (k_line.shift(1) >= d_line.shift(1)) & (k_line > 75)
    
    entries = df.loc[buy_sig, 'Close']
    exits = df.loc[sell_sig, 'Close']
    
    if len(entries) == 0 or len(exits) == 0: return 0
    
    n = min(len(entries), len(exits))
    profits = (exits.values[:n] - entries.values[:n]) / entries.values[:n]
    return -np.sum(profits)

# --- HELPER: FIBO SCORE CALCULATOR ---
def get_fibo_quality_score(df, high, low):
    if high <= low: return -1
    diff = high - low
    levels = [high - (diff * 0.618), high - (diff * 0.5), high - (diff * 0.382)]
    
    closes = df['Close'].values
    score = 0
    tolerance = diff * 0.015
    
    for lvl in levels:
        hits = np.sum(np.abs(closes - lvl) < tolerance)
        score += hits
        
    density_score = score / len(df) if len(df) > 0 else 0
    return density_score

# --- AI RUNNER (RANGE + ANCHOR OPTIMIZER) ---
@st.cache_data(ttl=3600, show_spinner=False)
def run_ai_optimizer(df_daily, df_momentum):
    # Gunakan data momentum (1H/4H/1D) yang dipilih user untuk indikator cepat
    df_mom = df_momentum
    
    # 1. SETUP TRAINING DATA (Momentum)
    split_mom = int(len(df_mom) * 0.8)
    train_mom = df_mom.iloc[:split_mom].copy()
    
    # Pivot for BB/EMA logic
    high_idx = argrelextrema(train_mom['High'].values, np.greater, order=5)[0]
    low_idx = argrelextrema(train_mom['Low'].values, np.less, order=5)[0]
    
    results = {}
    
    # 2. OPTIMIZE MOMENTUM (EMA, BB, MACD, STOCH)
    try:
        res_ema = minimize(loss_ema, x0=[200], args=(train_mom, train_mom['Low'].iloc[low_idx], low_idx), method='Nelder-Mead', tol=1.0)
        results['EMA'] = int(res_ema.x[0])
    except: results['EMA'] = 200

    try:
        res_bb = minimize(loss_bb, x0=[20, 2.0], args=(train_mom, high_idx, low_idx), method='Nelder-Mead', tol=0.1)
        results['BB'] = (int(res_bb.x[0]), res_bb.x[1])
    except: results['BB'] = (20, 2.0)

    try:
        res_macd = minimize(loss_macd, x0=[12, 26, 9], args=(train_mom,), method='Nelder-Mead', tol=0.1)
        results['MACD'] = (int(res_macd.x[0]), int(res_macd.x[1]), int(res_macd.x[2]))
    except: results['MACD'] = (12, 26, 9)

    try:
        res_stoch = minimize(loss_stoch_rsi, x0=[14, 3, 3], args=(train_mom,), method='Nelder-Mead', tol=0.1)
        results['STOCH'] = (int(res_stoch.x[0]), int(res_stoch.x[1]), int(res_stoch.x[2]))
    except: results['STOCH'] = (14, 3, 3)

    # 3. OPTIMIZE FIBO (ALWAYS ON DAILY STRUCTURE)
    candidate_windows = [60, 90, 180, 250, 365]
    best_window_data = df_daily.tail(250)
    best_range_score = -1
    
    try:
        for w in candidate_windows:
            if len(df_daily) < w: continue
            temp_df = df_daily.tail(w)
            temp_l = temp_df['Low'].min()
            temp_h = temp_df['High'].max()
            score = get_fibo_quality_score(temp_df, temp_h, temp_l)
            if score > best_range_score:
                best_range_score = score
                best_window_data = temp_df

        mid = len(best_window_data) // 2
        orig_l = best_window_data['Low'].min()
        orig_h = best_window_data['High'].max()
        future_lows = best_window_data['Low'].iloc[mid:].values
        
        res_fibo = minimize(loss_fibo, x0=[orig_l, orig_h], args=(future_lows, orig_l, orig_h), method='Nelder-Mead', tol=0.1)
        results['FIBO_ANCHORS'] = (res_fibo.x[0], res_fibo.x[1])
        
    except:
        results['FIBO_ANCHORS'] = (df_daily['Low'].min(), df_daily['High'].max())
    
    return results

# ==========================================
# 2. DATA PROCESSING
# ==========================================

def process_data_indicators(df, params):
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
    
    bbu = df['BBU'].fillna(method='bfill')
    bbl = df['BBL'].fillna(method='bfill')
    bbm = df['BBM'].fillna(method='bfill')
    bbm = bbm.replace(0, 0.0001)
    df['BB_Width'] = (bbu - bbl) / bbm
    
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
    
    loss = loss.replace(0, 0.0001)
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    min_rsi = df['RSI'].rolling(window=length).min()
    max_rsi = df['RSI'].rolling(window=length).max()
    
    denominator = (max_rsi - min_rsi).replace(0, 0.0001)
    stoch = (df['RSI'] - min_rsi) / denominator
    df['STOCHRSIk'] = stoch.rolling(window=k_smooth).mean() * 100
    df['STOCHRSId'] = df['STOCHRSIk'].rolling(window=d_smooth).mean()
    
    return df

def get_poc(df):
    if df.empty: return 0
    try:
        price_bins = pd.cut(df['Close'], bins=50)
        vpvr = df.groupby(price_bins, observed=True)['Volume'].sum()
        return vpvr.idxmax().mid
    except:
        return df['Close'].median()

def calculate_fibonacci_levels(df, anchors):
    low, high = anchors
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
# 3. SCORING ENGINE (DYNAMIC TF)
# ==========================================

def calculate_mtf_quant_score(row_mom, prev_row_mom, price_curr, poc_daily, fibo_golden, opt_params):
    scores = {}
    details = {}
    bullish_flags = 0

    # 1. EMA (Using Momentum TF)
    ema_val = row_mom['EMA200']
    dist_pct = ((price_curr - ema_val) / ema_val) * 100
    if dist_pct >= 1.5: score_ema = 100 
    elif 0.5 <= dist_pct < 1.5: score_ema = 85
    elif -0.5 <= dist_pct < 0.5: score_ema = 55
    elif -1.5 <= dist_pct < -0.5: score_ema = 40
    else: score_ema = 30 
    scores['EMA'] = score_ema
    details['EMA'] = f"Price ${price_curr:.0f} vs EMA ${ema_val:.0f} ({dist_pct:+.1f}%)"
    if score_ema >= 55: bullish_flags += 1

    # 2. VPVR POC (Using DAILY Structure - Always)
    if price_curr > poc_daily: score_vpvr = 100 
    elif abs(price_curr - poc_daily)/poc_daily < 0.03: score_vpvr = 60 
    else: score_vpvr = 30 
    scores['VPVR'] = score_vpvr
    pos_txt = "Above" if price_curr > poc_daily else "Below"
    details['VPVR'] = f"{pos_txt} POC Daily (${poc_daily:.0f})"
    if score_vpvr >= 60: bullish_flags += 1

    # 3. MACD (Using Momentum TF)
    hist = row_mom['MACD_Hist']
    macd_val = row_mom['MACD']
    sig_val = row_mom['MACD_Signal']
    prev_hist = prev_row_mom['MACD_Hist']
    
    if hist > 0 and hist > prev_hist: score_macd = 100
    elif hist > 0: score_macd = 70
    elif hist < 0 and hist > prev_hist: score_macd = 50
    else: score_macd = 30
    scores['MACD'] = score_macd
    details['MACD'] = f"Hist: {hist:+.1f} | M: {macd_val:.1f} | S: {sig_val:.1f}"
    if score_macd >= 70: bullish_flags += 1

    # 4. Stoch RSI (Using Momentum TF)
    k = row_mom['STOCHRSIk']
    d = row_mom['STOCHRSId']
    if k < 20 and k > d: score_stoch = 90
    elif k < 20: score_stoch = 65
    elif k > 80: score_stoch = 35
    else: score_stoch = 50
    scores['STOCH'] = score_stoch
    status_stoch = "OB" if k > 80 else "OS" if k < 20 else "N"
    details['STOCH'] = f"K: {k:.1f} | D: {d:.1f} ({status_stoch})"
    if score_stoch >= 65: bullish_flags += 1

    # 5. Bollinger (Using Momentum TF)
    if price_curr <= row_mom['BBL']: score_bb = 85
    elif price_curr < row_mom['BBM']: score_bb = 65
    elif price_curr < row_mom['BBU']: score_bb = 35
    else: score_bb = 20
    scores['BB'] = score_bb
    details['BB'] = f"Band: ${row_mom['BBL']:.0f} - ${row_mom['BBU']:.0f}"
    if score_bb >= 65: bullish_flags += 1

    # 6. Fibonacci (Using DAILY Structure - Always)
    dist_fibo_pct = abs((price_curr - fibo_golden) / fibo_golden) * 100
    if dist_fibo_pct <= 1.5: score_fibo = 90
    elif price_curr > fibo_golden: score_fibo = 55
    elif price_curr < fibo_golden * 0.95: score_fibo = 40
    else: score_fibo = 75
    scores['FIBO'] = score_fibo
    details['FIBO'] = f"Target Golden: ${fibo_golden:.0f} (Dist {dist_fibo_pct:.1f}%)"
    if score_fibo >= 55: bullish_flags += 1

    final_score = (
        (scores['EMA'] * 0.30) + (scores['VPVR'] * 0.20) + (scores['MACD'] * 0.15) +
        (scores['STOCH'] * 0.10) + (scores['BB'] * 0.10) + (scores['FIBO'] * 0.15)
    )
    
    return final_score, scores, details, bullish_flags

# --- DATA ENGINE (DYNAMIC ASSET & TF) ---
@st.cache_data(ttl=300, show_spinner=False)
def get_data_engine(ticker_symbol, momentum_tf_choice):
    # Mapping pilihan user ke parameter yfinance
    # Logic: 1H perlu raw hourly, 4H perlu raw hourly resampled, 1D perlu raw daily
    
    targets = [ticker_symbol, "IDR=X"]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Download Daily (Selalu butuh ini untuk Struktur/Fibo)
            df_daily = yf.download(targets, period="2y", interval="1d", group_by='ticker', progress=False)
            
            # Download Hourly (Base untuk 1H dan 4H)
            df_hourly = yf.download(targets, period="1y", interval="1h", group_by='ticker', progress=False)
            
            if not df_daily.empty:
                break
        except Exception as e:
            if attempt == max_retries - 1:
                return pd.DataFrame(), pd.DataFrame(), 16800
            time.sleep(1)

    try:
        # Handling MultiIndex
        asset_daily = pd.DataFrame()
        asset_hourly = pd.DataFrame()
        
        # Ekstrak data aset
        if isinstance(df_daily.columns, pd.MultiIndex):
            if ticker_symbol in df_daily.columns.levels[0]:
                asset_daily = df_daily[ticker_symbol].dropna()
                asset_hourly = df_hourly[ticker_symbol].dropna() if not df_hourly.empty else pd.DataFrame()
                kurs = df_daily['IDR=X']['Close'].iloc[-1]
            else:
                return pd.DataFrame(), pd.DataFrame(), 16800
        else:
            return pd.DataFrame(), pd.DataFrame(), 16800

        if isinstance(kurs, pd.Series): kurs = kurs.iloc[0]

        # Multiplier Correction (Hanya untuk PAXG)
        is_paxg = "PAXG" in ticker_symbol
        multiplier = PAXG_MULTIPLIER if is_paxg else 1.0
        
        cols = ['Close', 'High', 'Low', 'Open']
        for c in cols:
            if c in asset_daily.columns: asset_daily[c] = asset_daily[c] * multiplier
            if not asset_hourly.empty and c in asset_hourly.columns: asset_hourly[c] = asset_hourly[c] * multiplier

        # PREPARE MOMENTUM DATA BASED ON USER CHOICE
        df_momentum = pd.DataFrame()
        
        if momentum_tf_choice == "1H":
            df_momentum = asset_hourly.tail(1000) # Raw hourly
        elif momentum_tf_choice == "4H":
            # Resample Hourly to 4H
            df_momentum = asset_hourly.resample('4h').agg({
                'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'
            }).dropna().tail(500)
        elif momentum_tf_choice == "1D":
            # Use Daily as momentum too
            df_momentum = asset_daily.tail(365)
            
        return asset_daily, df_momentum, float(kurs)

    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), 16800

def send_telegram_alert(token, chat_id, message):
    if not token or not chat_id: return False, "Token/ID Kosong"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {"chat_id": chat_id, "text": message}
    try:
        r = requests.get(url, params=params)
        if r.status_code == 200: return True, "Sukses"
        else: return False, r.text
    except Exception as e:
        return False, str(e)

# --- REPORT GENERATOR (DYNAMIC TF) ---
def generate_mtf_report(df_daily, df_mom, kurs, ai_params, asset_name, tf_label):
    # Data Momentum (Last Candle)
    last_mom = df_mom.iloc[-1]
    prev_mom = df_mom.iloc[-2]
    
    # Data Price
    price = last_mom['Close']
    
    # Calculate Levels (Daily Structure - Always Fixed)
    fibo = calculate_fibonacci_levels(df_daily, ai_params['FIBO_ANCHORS']) 
    poc = get_poc(df_daily.tail(180)) 
    
    # 1. QUANT SCORE
    final_score, scores, details, bullish_count = calculate_mtf_quant_score(
        last_mom, prev_mom, price, poc, fibo['GOLDEN (0.618)'], ai_params
    )
    
    # 2. MARKET REGIME (Based on Selected TF)
    ema_dist = ((price - last_mom['EMA200']) / last_mom['EMA200']) * 100
    if ema_dist > 1.0: regime = f"🐎 TRENDING BULLISH ({tf_label})"
    elif ema_dist < -1.0: regime = f"🐻 TRENDING BEARISH ({tf_label})"
    else: regime = f"🦀 RANGING / SIDEWAYS ({tf_label})"
    
    bb_width = last_mom.get('BB_Width', 0.1)
    is_high_vol = bb_width > 0.15
    vol_status = "⚡ HIGH VOLATILITY" if is_high_vol else "🌊 NORMAL/LOW VOL"
    
    # Direction Flag (MACD Accel)
    macd_accel = last_mom['MACD_Hist'] - prev_mom['MACD_Hist']
    if macd_accel > 0: dir_flag = "↗️ UP"
    elif macd_accel < 0: dir_flag = "↘️ DOWN"
    else: dir_flag = "➡️ FLAT"
    
    # 3. SELL LOGIC
    action_type = "BUY"
    sell_reason = ""
    sell_pct = 0
    momentum_decay = last_mom['MACD_Hist'] < prev_mom['MACD_Hist'] and last_mom['MACD_Hist'] > 0
    
    if final_score < 40: 
        action_type = "SELL"
        sell_reason = "Score < 40 (Defensive Exit)"
        sell_pct = 50 
    elif price >= fibo['MOONBAG (1.618)']:
        action_type = "SELL"
        sell_reason = "Moonbag Target (Daily 1.618)"
        sell_pct = 50
    elif price >= fibo['RESISTANCE (High)'] and momentum_decay:
        action_type = "SELL"
        sell_reason = f"Daily Res + {tf_label} Mom Decay"
        sell_pct = 30
    elif price >= fibo['0.236'] and last_mom['STOCHRSIk'] > 80:
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
            prob_desc = "High Prob (Structure + Mom)"
        elif 65 <= final_score < 80:
            alloc_market, alloc_limit = 0.5, 0.5
            decision_title = "✅ STANDARD ACCUMULATION"
            prob_desc = "Healthy Trend (MTF Confirmed)"
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
        
    else:
        decision_title = "🚨 SELL / TAKE PROFIT"
        prob_desc = sell_reason

    # 5. SMART LIMIT (Uses DAILY Support for better safety)
    ema_daily_val = df_daily['Close'].ewm(span=ai_params['EMA'], adjust=False).mean().iloc[-1]
    
    candidates = [
        {'price': poc, 'label': 'POC Daily'},
        {'price': ema_daily_val, 'label': f"EMA Daily"},
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

    report = f"""🦅 {asset_name} MASTER V7 (MTF)
📅 {now.strftime('%d %b %Y | %H:%M WIB')}
=======================================

💰 **MARKET DATA ({tf_label}/Daily Hybrid)**
PRICE: {fmt_usd(price)}
EMA  : {fmt_usd(last_mom['EMA200'])} ({tf_label} Trend)
VOL  : {vol_status}

📊 **MATRIX 6 INDIKATOR ({tf_label})**
1. EMA        [{scores['EMA']}] {details['EMA']}
2. VPVR POC   [{scores['VPVR']}] {details['VPVR']}
3. MACD       [{scores['MACD']}] {details['MACD']}
4. Stoch RSI  [{scores['STOCH']}] {details['STOCH']}
5. Bollinger  [{scores['BB']}] {details['BB']}
6. Fibonacci  [{scores['FIBO']}] {details['FIBO']}

🧮 SCORE: {final_score:.1f}/100 | {agreement}
🌍 REGIME: {regime} | DIR: {dir_flag}
=======================================
🧠 DECISION : [ {decision_title} ]
🎲 LOGIC    : {prob_desc}
=======================================

📋 **EXECUTION PLAN (MODAL 5 JUTA)**
{main_action_txt}

🎯 **KEY LEVELS (DAILY STRUCTURE)**
"""
    sorted_fibo = dict(sorted(fibo.items(), key=lambda item: item[1], reverse=True))
    for k, v in sorted_fibo.items():
        report += f"{k:<15}: {fmt_usd(v)}\n"
        
    return report, df_daily, fibo, final_score

# --- MAIN APP ---
st.title("🦅 Crypto Master V7 (Multi-Asset Institutional)")

# --- USER SELECTION (SIDEBAR) ---
st.sidebar.header("⚙️ Konfigurasi Aset & TF")
asset_choice = st.sidebar.selectbox("Pilih Aset:", ["PAXG (Gold)", "BTC (Bitcoin)"])
tf_choice = st.sidebar.selectbox("Timeframe Momentum:", ["1H", "4H", "1D"])

# Map selection to Ticker & value
ticker_map = {"PAXG (Gold)": "PAXG-USD", "BTC (Bitcoin)": "BTC-USD"}
selected_ticker = ticker_map[asset_choice]

# --- STATUS BAR (PRO UI) ---
with st.status(f"🦅 Menganalisa {asset_choice} pada TF {tf_choice}...", expanded=True) as status:
    st.write(f"📡 Fetching Data ({selected_ticker})...")
    df_d_raw, df_mom_raw, kurs_val = get_data_engine(selected_ticker, tf_choice)
    
    if df_d_raw.empty or df_mom_raw.empty:
        status.update(label="❌ Data Error", state="error")
        st.error(f"Gagal mengambil data {selected_ticker}. Cek koneksi / Yahoo API.")
        st.stop()
    
    st.write(f"🧠 Training AI Model pada data {tf_choice}...")
    # PASS BOTH DATAFRAMES TO OPTIMIZER
    ai_params = run_ai_optimizer(df_d_raw, df_mom_raw)
    
    st.write("⚙️ Applying Indicators...")
    # Process separately
    df_d = process_data_indicators(df_d_raw, ai_params) # For Fibo/Structure
    df_mom = process_data_indicators(df_mom_raw, ai_params) # For Momentum
    
    status.update(label="✅ AI Ready!", state="complete", expanded=False)

# --- REPORT GENERATION ---
# Use MTF Generator
final_report, chart_daily, fib_levels, score_val = generate_mtf_report(
    df_d, df_mom, kurs_val, ai_params, asset_choice.split()[0], tf_choice
)

# # --- SIDEBAR INFO ---
# st.sidebar.markdown("---")
# st.sidebar.info(f"AI {tf_choice} EMA: {ai_params['EMA']}")
# st.sidebar.info(f"AI {tf_choice} BB: {ai_params['BB']}")
# st.sidebar.info(f"Daily Anchors: {fmt_usd(ai_params['FIBO_ANCHORS'][0])} - {fmt_usd(ai_params['FIBO_ANCHORS'][1])}")

score_color = "normal" if score_val >= 65 else "inverse"
st.sidebar.metric(f"SCORE ({tf_choice})", f"{score_val:.1f}", delta="Strength", delta_color=score_color)
st.sidebar.metric("PRICE (Realtime)", fmt_usd(df_mom.iloc[-1]['Close']))

# --- TELEGRAM CONFIG ---
st.sidebar.header("⚙️ Telegram")
if "TELEGRAM_TOKEN" in st.secrets:
    bot_token = st.secrets["TELEGRAM_TOKEN"]
    chat_id = st.secrets["TELEGRAM_CHAT_ID"]
else:
    bot_token = st.sidebar.text_input("Bot Token", type="password")
    chat_id = st.sidebar.text_input("Chat ID")

# --- CHART (Visualizing Structure) ---
st.subheader(f"Structural View (Daily Context + Key Levels)")
fig = go.Figure(data=[go.Candlestick(x=chart_daily.index,
                                open=chart_daily['Open'], high=chart_daily['High'],
                                low=chart_daily['Low'], close=chart_daily['Close'],
                                name=asset_choice)])

# EMA (Daily Reference for Chart)
ema_d_val = chart_daily['Close'].ewm(span=ai_params['EMA'], adjust=False).mean()
fig.add_trace(go.Scatter(x=chart_daily.index, y=ema_d_val, line=dict(color='blue', width=2), name=f'EMA {ai_params["EMA"]} (Daily)'))

# Fibo
colors_fib = {"MOONBAG": "lime", "RESISTANCE": "red", "GOLDEN": "gold", "FLOOR": "white", "MID": "gray"}
for label, val in fib_levels.items():
    c = "gray"
    for k, v in colors_fib.items():
        if k in label: c = v
    fig.add_hline(y=val, line_dash="dash", line_color=c, annotation_text=f"{label}")
    
fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

# --- REPORT OUTPUT ---
st.subheader("📋 Institutional Execution Report")
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("📩 Broadcast Telegram"):
        with st.spinner("Sending..."):
            success, msg = send_telegram_alert(bot_token, chat_id, final_report)
            if success: st.success("Sent!")
            else: st.error(f"Failed: {msg}")

# Render Text Report
st.code(final_report, language="yaml")

