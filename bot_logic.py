import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz
from scipy.optimize import minimize
from scipy.signal import argrelextrema
import warnings
import time

# Matikan warning agar log bersih
warnings.filterwarnings("ignore")

# --- KONFIGURASI ENGINE ---
TICKER_PAXG = "PAXG-USD"
TICKER_IDR = "IDR=X"
MODAL_GAJI = 5000000 
SPREAD_AJAIB = 1.015 
PAXG_MULTIPLIER = 0.99048968

# --- HELPER FORMATTING ---
def fmt_idr(val): return f"Rp {val:,.0f}".replace(",", ".")
def fmt_usd(val): return f"${val:,.2f}"

# ==========================================
# 1. AI OPTIMIZATION ENGINE
# ==========================================

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

def run_ai_optimizer(df):
    print("   🧠 AI Brain: Optimizing Indicators...")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    high_idx = argrelextrema(train_df['High'].values, np.greater, order=5)[0]
    low_idx = argrelextrema(train_df['Low'].values, np.less, order=5)[0]
    results = {}
    try:
        res_ema = minimize(loss_ema, x0=[200], args=(train_df, train_df['Low'].iloc[low_idx], low_idx), method='Nelder-Mead', tol=1.0)
        results['EMA'] = int(res_ema.x[0])
    except: results['EMA'] = 200
    try:
        res_bb = minimize(loss_bb, x0=[20, 2.0], args=(train_df, high_idx, low_idx), method='Nelder-Mead', tol=0.1)
        results['BB'] = (int(res_bb.x[0]), res_bb.x[1])
    except: results['BB'] = (20, 2.0)
    try:
        res_macd = minimize(loss_macd, x0=[12, 26, 9], args=(train_df,), method='Nelder-Mead', tol=0.1)
        results['MACD'] = (int(res_macd.x[0]), int(res_macd.x[1]), int(res_macd.x[2]))
    except: results['MACD'] = (12, 26, 9)
    try:
        res_stoch = minimize(loss_stoch_rsi, x0=[14, 3, 3], args=(train_df,), method='Nelder-Mead', tol=0.1)
        results['STOCH'] = (int(res_stoch.x[0]), int(res_stoch.x[1]), int(res_stoch.x[2]))
    except: results['STOCH'] = (14, 3, 3)
    try:
        mid = len(df) // 2
        orig_l, orig_h = df['Low'].min(), df['High'].max()
        future_lows = df['Low'].iloc[mid:].values 
        res_fibo = minimize(loss_fibo, x0=[orig_l, orig_h], args=(future_lows, orig_l, orig_h), method='Nelder-Mead', tol=0.1)
        results['FIBO_ANCHORS'] = (res_fibo.x[0], res_fibo.x[1])
    except: 
        results['FIBO_ANCHORS'] = (df['Low'].min(), df['High'].max())
    return results

# ==========================================
# 2. DATA PROCESSING
# ==========================================

def process_data_ai(df, params):
    df = df.copy()
    df['EMA200'] = df['Close'].ewm(span=params['EMA'], adjust=False).mean()
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
    fast, slow, sig = params['MACD']
    k = df['Close'].ewm(span=fast, adjust=False).mean()
    d = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = k - d
    df['MACD_Signal'] = df['MACD'].ewm(span=sig, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
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
    
    # Volatility (ATR Simple) for Human Report
    df['TR'] = np.maximum((df['High'] - df['Low']), 
                          np.maximum(abs(df['High'] - df['Close'].shift()), 
                                     abs(df['Low'] - df['Close'].shift())))
    df['ATR'] = df['TR'].rolling(14).mean()
    
    return df

def get_poc(df):
    if df.empty: return 0
    try:
        price_bins = pd.cut(df['Close'], bins=50)
        vpvr = df.groupby(price_bins, observed=True)['Volume'].sum()
        return vpvr.idxmax().mid
    except: return df['Close'].median()

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
# 3. SCORING & REPORT
# ==========================================

def calculate_quant_score(row, prev_row, poc, fibo_golden, opt_params):
    price = row['Close']
    scores = {}
    details = {}
    bullish_flags = 0

    # 1. EMA
    ema = row['EMA200']
    dist_pct = ((price - ema) / ema) * 100
    if dist_pct >= 12: score_ema = 100
    elif 8 <= dist_pct < 12: score_ema = 85
    elif 4 <= dist_pct < 8: score_ema = 70
    elif 0 <= dist_pct < 4: score_ema = 55
    else: score_ema = 30 
    scores['EMA'] = score_ema
    details['EMA'] = f"AI-EMA({opt_params['EMA']}) | Price ${price:.0f} vs EMA ${ema:.0f} ({dist_pct:+.1f}%)"
    if score_ema >= 55: bullish_flags += 1

    # 2. VPVR POC
    if price > poc: score_vpvr = 100 
    elif abs(price - poc)/poc < 0.03: score_vpvr = 60 
    else: score_vpvr = 30 
    scores['VPVR'] = score_vpvr
    details['VPVR'] = f"{'Above' if price > poc else 'Below'} Wall | POC: ${poc:.0f}"
    if score_vpvr >= 60: bullish_flags += 1

    # 3. MACD
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

    # 4. Stoch RSI
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

    # 5. Bollinger
    if price <= row['BBL']: score_bb = 85
    elif price < row['BBM']: score_bb = 65
    elif price < row['BBU']: score_bb = 35
    else: score_bb = 20
    scores['BB'] = score_bb
    p = opt_params['BB']
    details['BB'] = f"AI-BB({p[0]},{p[1]:.1f}) | Up: ${row['BBU']:.0f} | Low: ${row['BBL']:.0f}"
    if score_bb >= 65: bullish_flags += 1

    # 6. Fibonacci
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

# --- DATA ENGINE (VERSI ANTI-GAGAL) ---
def get_data_engine():
    print("⏳ Connecting to Market Data...")
    
    # 1. DOWNLOAD PAXG DAILY (2 Tahun)
    try:
        paxg_d = yf.download("PAXG-USD", period="2y", interval="1d", progress=False)
        if paxg_d.empty:
            print("   ❌ Gagal download PAXG Daily.")
            return pd.DataFrame(), pd.DataFrame(), 16800, {}
    except Exception as e:
        print(f"   ❌ Error PAXG Daily: {e}")
        return pd.DataFrame(), pd.DataFrame(), 16800, {}

    # 2. DOWNLOAD PAXG HOURLY (1 Bulan) - Coba retry kalau gagal
    try:
        paxg_h = yf.download("PAXG-USD", period="1mo", interval="1h", progress=False)
        if paxg_h.empty:
            print("   ⚠️ Gagal download PAXG Hourly (Skip hourly logic).")
            # Fallback: Pakai data daily kalau hourly gagal
            paxg_h = paxg_d.tail(30).copy() 
    except:
        paxg_h = paxg_d.tail(30).copy()

    # 3. DOWNLOAD KURS IDR (Terpisah)
    try:
        idr_df = yf.download("IDR=X", period="1d", progress=False)
        if not idr_df.empty:
            kurs = idr_df['Close'].iloc[-1]
            if isinstance(kurs, pd.Series): kurs = kurs.iloc[0]
            kurs = float(kurs)
        else:
            kurs = 16800.0 # Fallback manual jika Yahoo error
    except:
        kurs = 16800.0

    print(f"   ✅ Data Loaded. Price: {fmt_usd(paxg_d['Close'].iloc[-1])} | IDR: {fmt_idr(kurs)}")

    # 4. DATA PROCESSING
    try:
        # Fix MultiIndex Columns (Masalah umum yfinance terbaru)
        if isinstance(paxg_d.columns, pd.MultiIndex):
            paxg_d.columns = paxg_d.columns.get_level_values(0)
        if isinstance(paxg_h.columns, pd.MultiIndex):
            paxg_h.columns = paxg_h.columns.get_level_values(0)

        # Kalibrasi Harga User (Multiplier)
        cols = ['Close', 'High', 'Low', 'Open']
        for col in cols:
            if col in paxg_d.columns: paxg_d[col] = paxg_d[col] * PAXG_MULTIPLIER
            if col in paxg_h.columns: paxg_h[col] = paxg_h[col] * PAXG_MULTIPLIER

        # AI Optimization
        opt_data = paxg_d.tail(200).copy()
        ai_params = run_ai_optimizer(opt_data)

        # Process Indicators
        paxg_d = process_data_ai(paxg_d, ai_params)
        paxg_6mo = paxg_d.tail(180).copy()
        
        # Hourly Processing
        if not paxg_h.empty:
            paxg_4h = paxg_h.resample('4h').agg({
                'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'
            }).dropna()
            paxg_4h = process_data_ai(paxg_4h, ai_params)
        else:
            paxg_4h = paxg_d.tail(30) # Fallback

        return paxg_6mo, paxg_4h, kurs, ai_params

    except Exception as e:
        print(f"   ❌ Processing Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), 16800, {}

def send_telegram_alert(token, chat_id, message):
    if not token or not chat_id: return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {"chat_id": chat_id, "text": message}
    try:
        requests.get(url, params=params)
        return True
    except: return False

# --- NEW: HUMAN READABLE REPORT (PESAN KE-2) ---
def generate_human_report(df_6mo, kurs, score, ai_params, poc, fibo):
    last_d = df_6mo.iloc[-1]
    price = last_d['Close']
    open_price = last_d['Open']
    
    if score >= 80: 
        confidence = "Sangat Tinggi (Strong Bullish) 🔥"
        desc_score = f"{score:.1f}%"
    elif score >= 65: 
        confidence = "Cukup Kuat (Moderate Bullish) ✅"
        desc_score = f"{score:.1f}%"
    elif score >= 50: 
        confidence = "Sedang/Ragu (Wait & See) ⚠️"
        desc_score = f"{score:.1f}%"
    else: 
        confidence = "Lemah/Bearish (Hati-hati) 🛑"
        desc_score = f"{score:.1f}%"

    trend_icon = "📈" if price >= open_price else "📉"
    trend_txt = "mengalami kenaikan" if price >= open_price else "mengalami penurunan"

    # Map Levels for Human Context
    levels = [
        (last_d['BBU'], "Bollinger Atas"),
        (poc, "Area Volume Padat"),
        (fibo['RESISTANCE (High)'], "Puncak Tertinggi"),
        (fibo['GOLDEN (0.618)'], "Fibo Golden"),
        (fibo['0.382'], "Fibo 0.382"),
        (fibo['MID (0.5)'], "Fibo 0.5"),
        (last_d['EMA200'], "Garis Tren")
    ]

    # Skenario Kenaikan
    resistances = sorted([x for x in levels if x[0] > price], key=lambda x: x[0])
    if resistances:
        next_res = resistances[0]
        upside_txt = f"sedang mencoba menembus **{next_res[1]}** di {fmt_usd(next_res[0])}. "
        if len(resistances) > 1:
            upside_txt += f"Jika tembus, lanjut ke {resistances[1][1]} ({fmt_usd(resistances[1][0])})"
            if len(resistances) > 2:
                upside_txt += f" dan {resistances[2][1]}."
            else: upside_txt += "."
    else:
        upside_txt = "sudah menembus semua atap (Breakout)! 🚀"

    # Skenario Penurunan
    supports = sorted([x for x in levels if x[0] < price], key=lambda x: x[0], reverse=True)
    if supports:
        next_sup = supports[0]
        downside_txt = f"penahan terdekat adalah **{next_sup[1]}** di {fmt_usd(next_sup[0])}. "
        if len(supports) > 1:
            downside_txt += f"Jika jebol, kemungkinan turun ke {supports[1][1]} ({fmt_usd(supports[1][0])})."
    else:
        downside_txt = "waspada, tidak ada penahan dekat di bawah."

    atr = last_d.get('ATR', price * 0.01)
    range_high = price + atr
    range_low = price - atr
    est_idr = price * kurs * SPREAD_AJAIB
    now = datetime.now(pytz.timezone('Asia/Jakarta'))
    
    msg = f"""
🤖 **UPDATE PASAR (BAHASA MANUSIA)**
🕒 {now.strftime('%H:%M WIB')} | 💵 {fmt_usd(price)} | 🇮🇩 {fmt_idr(est_idr)}
---------------------------------------
Saat ini emas sedang **{trend_txt}** {trend_icon} dengan tingkat keyakinan **{confidence}** ({desc_score}).

🎯 **Skenario Kenaikan:**
Harga saat ini sekitar {fmt_usd(price)} {upside_txt}

🛡️ **Skenario Penurunan:**
Sebaliknya, jika terjadi koreksi, {downside_txt}

📊 **Range Potensial (Hari Ini - Besok):**
• Atas: {fmt_usd(range_high)}
• Bawah: {fmt_usd(range_low)}

💡 *Saran: Gunakan data ini sebagai referensi psikologis agar tidak FOMO atau Panic Selling.*
    """
    return msg

# --- TECHNICAL REPORT GENERATOR (PESAN 1: STRICT FORMAT) ---
def generate_sop_report(df_6mo, df_4h, kurs, ai_params):
    last_d = df_6mo.iloc[-1]
    prev_d = df_6mo.iloc[-2]
    price = last_d['Close']
    fibo = calculate_fibonacci_levels(df_6mo, ai_params['FIBO_ANCHORS']) 
    poc = get_poc(df_6mo)
    
    final_score, scores, details, bullish_count = calculate_quant_score(last_d, prev_d, poc, fibo['GOLDEN (0.618)'], ai_params)
    
    ema_dist = ((price - last_d['EMA200']) / last_d['EMA200']) * 100
    regime = "TRENDING BULLISH" if ema_dist > 5 else "TRENDING BEARISH" if ema_dist < -5 else "RANGING"
    bb_width = last_d.get('BB_Width', 0.1)
    is_high_vol = bb_width > 0.15
    vol_status = "HIGH VOLATILITY" if is_high_vol else "NORMAL VOLATILITY"
    
    action_type = "BUY"
    sell_reason = ""
    sell_pct = 0
    momentum_decay = last_d['MACD_Hist'] < prev_d['MACD_Hist'] and last_d['MACD_Hist'] > 0
    
    if final_score < 40: 
        action_type = "SELL"
        sell_reason = "Score < 40 (Defensive)"
        sell_pct = 50 
    elif price >= fibo['MOONBAG (1.618)']:
        action_type = "SELL"
        sell_reason = "Moonbag Target"
        sell_pct = 50
    elif price >= fibo['RESISTANCE (High)'] and momentum_decay:
        action_type = "SELL"
        sell_reason = "Resist + Decay"
        sell_pct = 30
    elif price >= fibo['0.236'] and last_d['STOCHRSIk'] > 80:
        action_type = "SELL"
        sell_reason = "Overbought"
        sell_pct = 20

    dana_market, dana_limit = 0, 0
    decision_title, prob_desc = "", ""
    agreement = f"{bullish_count}/6 Bullish"
    
    if action_type == "BUY":
        if final_score >= 80:
            alloc_market, alloc_limit = 0.7, 0.3
            decision_title = "AGGRESSIVE BUY"
            prob_desc = "High Prob Setup"
        elif 65 <= final_score < 80:
            alloc_market, alloc_limit = 0.5, 0.5
            decision_title = "STANDARD ACCUMULATION"
            prob_desc = "Moderate/Healthy Trend"
        elif 50 <= final_score < 65:
            alloc_market, alloc_limit = 0.2, 0.8
            decision_title = "SNIPER ENTRY"
            prob_desc = "Price Extended/Wait"
        else: 
            alloc_market, alloc_limit = 0.0, 1.0
            decision_title = "DEFENSIVE"
            prob_desc = "Weak Structure"

        if is_high_vol and final_score >= 50:
            alloc_market *= 0.7 
            alloc_limit = 1.0 - alloc_market 
            prob_desc += " (Vol Adjusted)"

        dana_market = MODAL_GAJI * alloc_market
        dana_limit = MODAL_GAJI * alloc_limit

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
    now = datetime.now(pytz.timezone('Asia/Jakarta'))
    
    if action_type == "SELL":
        main_action_txt = f"""**SELL SIGNAL**
    Action: JUAL {sell_pct}%
    Alasan: {sell_reason}"""
    else:
        main_action_txt = f"""**MARKET ORDER**
    Nominal: {fmt_idr(dana_market)}
    Eksekusi: SEKARANG.

**LIMIT ORDER**
    Nominal: {fmt_idr(dana_limit)}
    Target: {fmt_usd(target_limit_usd)} ({target_label})
    Est. IDR: {fmt_idr(est_limit_idr)}"""

    # CONSTRUCTION OF THE REPORT STRING (STRICT FORMAT)
    report = f""" GOLD MASTER V6 (AI ENHANCED) {now.strftime('%d %b %Y | %H:%M WIB')}
======================================= **MARKET DATA**
PAXG : {fmt_usd(price)}
EMA  : {fmt_usd(last_d['EMA200'])} (Per: {ai_params['EMA']})
VOL  : {vol_status} **MATRIX 6 INDIKATOR (AI OPTIMIZED)**
1. EMA       [{scores['EMA']}] {details['EMA']}
2. VPVR POC  [{scores['VPVR']}] {details['VPVR']}
3. MACD      [{scores['MACD']}] {details['MACD']}
4. Stoch RSI [{scores['STOCH']}] {details['STOCH']}
5. Bollinger [{scores['BB']}] {details['BB']}
6. Fibonacci [{scores['FIBO']}] {details['FIBO']} SCORE: {final_score:.1f}/100 | {agreement} REGIME: {regime}
======================================= DECISION : [ {decision_title} ] LOGIC    : {prob_desc}
======================================= **EXECUTION PLAN (MODAL 5 JUTA)** {main_action_txt}
         **KEY LEVELS**
"""
    # Append Key Levels
    sorted_fibo = dict(sorted(fibo.items(), key=lambda item: item[1], reverse=True))
    for k, v in sorted_fibo.items():
        report += f"{k:<15}: {fmt_usd(v)}\n"
        
    return report, final_score, fibo, poc

# --- MAIN EXECUTION ---
def run_bot():
    print("="*50)
    print("🦅 Gold Master V6 (AI Headless) Started")
    print("="*50)
    
    raw_d, raw_h, kurs, ai_params = get_data_engine()
    
    if raw_d.empty:
        print("❌ Critical Error: Data Empty.")
        return

    # Generate Technical Report
    tech_report, score, fibo, poc = generate_sop_report(raw_d.tail(180), raw_d.tail(180), kurs, ai_params)
    
    # Generate Human Report (NEW)
    human_report = generate_human_report(raw_d.tail(180), kurs, score, ai_params, poc, fibo)
    
    print("\n" + tech_report)
    print("\n" + human_report)
    
    TOKEN = os.environ.get("TELEGRAM_TOKEN") 
    CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
    
    if TOKEN and CHAT_ID:
        print("🚀 Sending 2 Messages to Telegram...")
        send_telegram_alert(TOKEN, CHAT_ID, tech_report) # Pesan 1 (Teknis)
        time.sleep(1) # Jeda dikit biar urutan bener
        send_telegram_alert(TOKEN, CHAT_ID, human_report) # Pesan 2 (Manusia)
    else:
        print("⚠️ Skip Telegram.")

if __name__ == "__main__":
    run_bot()

