import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import pytz

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="SOP TANGGAL 25 (PAXG)", page_icon="🦅", layout="wide")

# --- CSS KHUSUS ---
st.markdown("""
<style>
    .big-font { font-size: 20px !important; font-weight: bold; }
    .scenario-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; }
    .swinger { background-color: #3d0000; border: 2px solid #ff4b4b; color: white; }
    .investor { background-color: #002b00; border: 2px solid #00ff00; color: white; }
    .instruction { font-family: 'Consolas', monospace; white-space: pre-wrap; background-color: #1e1e1e; padding: 15px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- KONFIGURASI ---
TICKERS = ["PAXG-USD", "IDR=X"]
MODAL_GAJI = 5000000  # 5 Juta Rupiah
SPREAD_AJAIB = 1.015 
PAXG_MULTIPLIER = 0.99048968

# --- HELPER ---
def fmt_idr(val): return f"Rp {val:,.0f}".replace(",", ".")
def fmt_usd(val): return f"${val:,.2f}"

# --- FUNGSI INDIKATOR ---
def add_indicators(df):
    df = df.copy()
    # Bollinger Bands (20, 2)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BBU'] = df['SMA20'] + (df['STD20'] * 2)
    df['BBL'] = df['SMA20'] - (df['STD20'] * 2)
    df['BBM'] = df['SMA20'] # Middle Band

    # Stochastic RSI (14, 14, 3, 3)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    min_rsi = df['RSI'].rolling(window=14).min()
    max_rsi = df['RSI'].rolling(window=14).max()
    stoch_rsi = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)
    df['STOCHRSIk'] = stoch_rsi.rolling(window=3).mean() * 100
    
    return df

# --- FIBONACCI (LENGKAP) ---
def calculate_fibonacci(df):
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    return {
        "1.618 (TP MAX)": high + (diff * 0.618),
        "1.0 (TOP)": high,
        "0.786": low + (diff * 0.786),
        "0.618 (GOLDEN)": low + (diff * 0.618),
        "0.5 (MID)": low + (diff * 0.5),
        "0.382": low + (diff * 0.382),
        "0.236": low + (diff * 0.236),
        "0.0 (BOTTOM)": low
    }

# --- VPVR SIMPLIFIED ---
def get_poc(df):
    price_bins = pd.cut(df['Close'], bins=50)
    vpvr = df.groupby(price_bins, observed=True)['Volume'].sum()
    return vpvr.idxmax().mid

# --- FETCH DATA ENGINE ---
@st.cache_data(ttl=300)
def get_data_sop():
    # 1. AMBIL DATA DAILY (6 BULAN) -> UNTUK TREN BESAR
    df_daily = yf.download(TICKERS, period="6mo", interval="1d", group_by='ticker', progress=False)
    
    # 2. AMBIL DATA HOURLY (1 BULAN) -> UNTUK EKSEKUSI 4H
    df_hourly = yf.download(TICKERS, period="1mo", interval="1h", group_by='ticker', progress=False)

    # Handling MultiIndex
    if isinstance(df_daily.columns, pd.MultiIndex):
        paxg_d = df_daily['PAXG-USD'].dropna()
        paxg_h = df_hourly['PAXG-USD'].dropna()
        kurs = df_daily['IDR=X']['Close'].iloc[-1]
    else:
        return None, None, 16500

    # Kalibrasi User
    for df in [paxg_d, paxg_h]:
        df['Close'] = df['Close'] * PAXG_MULTIPLIER
        df['High'] = df['High'] * PAXG_MULTIPLIER
        df['Low'] = df['Low'] * PAXG_MULTIPLIER
        df['Open'] = df['Open'] * PAXG_MULTIPLIER

    # Indikator Daily
    paxg_d = add_indicators(paxg_d)

    # Resample Hourly ke 4H untuk Skenario Swinger
    paxg_4h = paxg_h.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    paxg_4h = add_indicators(paxg_4h)

    return paxg_d, paxg_4h, kurs

def send_telegram(token, chat_id, msg):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.get(url, params={"chat_id": chat_id, "text": msg})
        return True
    except: return False

# --- UI UTAMA ---
st.title("🦅 SOP GAJIAN TANGGAL 25")
st.markdown("### Algoritma Eksekusi Modal 5 Juta")

with st.spinner("Menjalankan SOP..."):
    df_d, df_4h, kurs = get_data_sop()

    if df_d is None:
        st.error("Gagal ambil data.")
    else:
        last_d = df_d.iloc[-1]
        last_4h = df_4h.iloc[-1]
        
        # 1. TENTUKAN SKENARIO (PAKAI DATA DAILY)
        # Skenario 1 (Swinger/Huru-hara): Harga > Upper BB ATAU StochRSI > 80
        is_swinger = (last_d['Close'] > last_d['BBU']) or (last_d['STOCHRSIk'] > 80)
        
        poc_d = get_poc(df_d)
        fibo_d = calculate_fibonacci(df_d)

        # --- LOGIC OUTPUT ---
        
        # === SKENARIO 1: PASAR GEJOLAK (SWINGER) ===
        if is_swinger:
            st.markdown("""
            <div class="scenario-box swinger">
                <h2>🚨 SKENARIO 1: PASAR GEJOLAK (SWINGER MODE)</h2>
                <p><b>Kondisi:</b> Harga tembus Upper BB atau Stoch RSI Daily > 80. Pasar lagi "Huru-hara" atau Pucuk.</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🛑 NASIB UANG 5 JUTA")
                st.warning("JANGAN MASUK DULU! TAHAN CASH.")
                
                # Cek Sinyal Masuk di 4H
                stoch_4h = last_4h['STOCHRSIk']
                
                st.markdown(f"""
                **Analisa Data 4 Jam (Last 1 Month):**
                - Stoch RSI (4H) Saat Ini: **{stoch_4h:.2f}**
                
                **INSTRUKSI:**
                1. Tahan IDR/USDT.
                2. Tunggu Stoch RSI (4H) turun ke **bawah 20**.
                3. Jika nanti < 20: **MARKET ORDER** semua 5 Juta.
                """)
                
                if stoch_4h < 20:
                    st.success("✅ SAATNYA MASUK! Stoch RSI 4H sudah < 20. Gas Market Order!")
                else:
                    st.error(f"❌ BELUM SAATNYA. Tunggu indikator dingin (<20). Sekarang {stoch_4h:.2f}")

            with col2:
                st.subheader("💰 ASET LAMA (TP)")
                st.markdown(f"""
                **Target Jual Sebagian (10-20%):**
                - Upper BB (4H): **{fmt_usd(last_4h['BBU'])}**
                - Fibo 1.618 (Daily): **{fmt_usd(fibo_d['1.618 (TP MAX)'])}**
                
                *Jika harga menyentuh salah satu angka di atas, TP sebagian.*
                """)
            
            # TAMPILKAN CHART 4H (KARENA MODE SWINGER)
            st.subheader("📉 Chart Eksekusi: 4H (1 Month)")
            active_df = df_4h
            active_fibo = calculate_fibonacci(df_4h) # Fibo pendek untuk 4H

        # === SKENARIO 2: PASAR STABIL (INVESTOR) ===
        else:
            st.markdown("""
            <div class="scenario-box investor">
                <h2>✅ SKENARIO 2: PASAR STABIL (INVESTOR MODE)</h2>
                <p><b>Kondisi:</b> Harga Normal/Diskon (Middle/Lower BB) dan Stoch RSI Daily < 80.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Hitung Split Order
            paxg_idr_real = last_d['Close'] * kurs * SPREAD_AJAIB
            
            # 50% Market
            dana_market = MODAL_GAJI * 0.5
            est_dapat_market = dana_market / paxg_idr_real
            
            # 50% Limit (Cari Support Terdekat: POC atau Fibo 0.618)
            target_limit_usd = max(poc_d, fibo_d['0.618 (GOLDEN)']) # Ambil support paling atas biar kena
            # Tapi harus di bawah harga sekarang
            if target_limit_usd >= last_d['Close']:
                target_limit_usd = fibo_d['0.5 (MID)'] # Kalau support deket banget, turunin ke 0.5

            target_limit_idr = target_limit_usd * kurs * SPREAD_AJAIB
            dana_limit = MODAL_GAJI * 0.5
            est_dapat_limit = dana_limit / target_limit_idr

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🛒 EKSEKUSI: SPLIT ORDER")
                st.markdown(f"""
                **1. MARKET ORDER (50%) - Rp {dana_market:,.0f}**
                - Beli Sekarng di Harga: **{fmt_idr(paxg_idr_real)}**
                - Estimasi Dapat: **{est_dapat_market:.4f} PAXG**
                
                **2. LIMIT ORDER (50%) - Rp {dana_limit:,.0f}**
                - Pasang Antrean di: **{fmt_usd(target_limit_usd)}**
                - Estimasi Rupiah (Ajaib): **{fmt_idr(target_limit_idr)}**
                - Estimasi Dapat: **{est_dapat_limit:.4f} PAXG**
                
                *Alasan Limit: Support VPVR/Fibo 0.618 Terdekat.*
                """)

            with col2:
                st.subheader("HOLDING STRATEGY")
                st.markdown("""
                - **JANGAN TP.**
                - Fokus akumulasi jumlah gram.
                - Jika Limit Order tidak tereksekusi sampai tanggal 25 bulan depan, hapus order dan gabungkan dananya.
                """)

            # TAMPILKAN CHART 1D (KARENA MODE INVESTOR)
            st.subheader("📈 Chart Eksekusi: Daily (6 Months)")
            active_df = df_d
            active_fibo = fibo_d

        # --- GAMBAR CHART SESUAI SKENARIO ---
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(x=active_df.index, open=active_df['Open'], high=active_df['High'],
                                     low=active_df['Low'], close=active_df['Close'], name='PAXG'))
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=active_df.index, y=active_df['BBU'], line=dict(color='red', width=1, dash='dot'), name='Upper BB'))
        fig.add_trace(go.Scatter(x=active_df.index, y=active_df['BBL'], line=dict(color='green', width=1, dash='dot'), name='Lower BB'))
        fig.add_trace(go.Scatter(x=active_df.index, y=active_df['BBM'], line=dict(color='yellow', width=1), name='Middle BB'))

        # Fibonacci (LENGKAP)
        colors_fib = {"1.618": "magenta", "1.0": "red", "0.618": "gold", "0.5": "white", "0.0": "green"}
        for name, val in active_fibo.items():
            c = "gray"
            for k, v in colors_fib.items():
                if k in name: c = v
            fig.add_hline(y=val, line_dash="dash", line_color=c, annotation_text=f"{name}: ${val:.2f}")

        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- LAPORAN TEXT ---
        st.markdown("### 📋 Laporan SOP Telegram")
        
        report_text = f"""🦅 SOP TANGGAL 25 REPORT
📅 {datetime.now().strftime('%d %b %Y')}
============================
STATUS: {"🚨 SKENARIO 1 (SWINGER)" if is_swinger else "✅ SKENARIO 2 (INVESTOR)"}

📊 DATA PASAR
Harga PAXG: {fmt_usd(last_d['Close'])}
Stoch Daily: {last_d['STOCHRSIk']:.2f}

📝 INSTRUKSI:
"""
        if is_swinger:
            report_text += f"""1. TAHAN CASH 5 Juta.
2. Pantau Chart 4 Jam.
3. Stoch RSI 4H saat ini: {stoch_4h:.2f}
4. Tunggu Stoch 4H < 20 baru MARKET ORDER.
5. Cek Aset Lama: Jual sebagian jika kena ${active_fibo['1.618 (TP MAX)']:.2f}"""
        else:
            report_text += f"""1. MARKET ORDER 50% (Rp 2.5jt) SEKARANG.
2. LIMIT ORDER 50% (Rp 2.5jt) di {fmt_usd(target_limit_usd)}.
   (Estimasi Rupiah Limit: {fmt_idr(target_limit_idr)})
3. JANGAN TP. HOLD."""

        st.text_area("Copy Text ini:", value=report_text, height=300)

        # Telegram
        if "TELEGRAM_TOKEN" not in st.secrets:
            token = st.text_input("Telegram Token", type="password")
            chatid = st.text_input("Chat ID")
        else:
            token = st.secrets["TELEGRAM_TOKEN"]
            chatid = st.secrets["TELEGRAM_CHAT_ID"]

        if st.button("📩 Kirim Laporan"):
            if send_telegram(token, chatid, report_text):
                st.success("Terkirim!")
            else:
                st.error("Gagal.")
