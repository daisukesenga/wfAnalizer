import streamlit as st
import gpxpy
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from geopy.distance import geodesic

# ページの設定
st.set_page_config(page_title="Wing Foil GPX Analyzer", layout="wide")
st.title("🏄‍♂️ ウィングフォイル GPXアナライザー")
st.write("GPXファイルをアップロードして、フォイリング率、ジャイブ成功率、沈（ワイプアウト）ポイントを分析します。")

# --- サイドバーの設定 ---
st.sidebar.header("⚙️ 解析パラメータ設定")
foil_threshold = st.sidebar.slider("フォイリング開始速度 (km/h)", 10.0, 18.0, 13.5, 0.5)
jibe_speed_threshold = st.sidebar.slider("ジャイブ成功とみなす最低速度 (km/h)", 8.0, 15.0, 11.0, 0.5)
speed_max_limit = st.sidebar.number_input("GPSノイズカットの上限速度 (km/h)", 50, 100, 60)

# --- 方位角（Heading）を計算する関数 ---
def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lon = np.radians(lon2 - lon1)
    
    y = np.sin(delta_lon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon)
    
    bearing = np.degrees(np.arctan2(y, x))
    return (bearing + 360) % 360

# --- ファイルアップローダー ---
uploaded_file = st.file_uploader("GPXファイルをドラッグ＆ドロップ、またはブラウズ", type=["gpx"])

if uploaded_file is not None:
    # 1. GPXデータのパース
    with st.spinner("GPXファイルを解析中..."):
        gpx = gpxpy.parse(uploaded_file)
        raw_data = []
        
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    raw_data.append({
                        "time": point.time,
                        "lat": point.latitude,
                        "lon": point.longitude
                    })
                    
        if len(raw_data) < 10:
            st.error("データポイントが少なすぎます。正しいGPXファイルか確認してください。")
            st.stop()
            
        df = pd.DataFrame(raw_data)
        df['time'] = pd.to_datetime(df['time'])
        
        if df['time'].dt.tz is not None:
            df['time'] = df['time'].dt.tz_convert('Asia/Tokyo').dt.tz_localize(None)
        
        # 2. 速度・時間差・方位角の計算
        df['time_diff'] = df['time'].diff().dt.total_seconds()
        
        speeds = [0.0]
        bearings = [0.0]
        
        for i in range(1, len(df)):
            p1 = (df.loc[i-1, 'lat'], df.loc[i-1, 'lon'])
            p2 = (df.loc[i, 'lat'], df.loc[i, 'lon'])
            t_diff = df.loc[i, 'time_diff']
            
            if t_diff > 0:
                dist = geodesic(p1, p2).meters
                speed_kmh = (dist / t_diff) * 3.6
                speeds.append(speed_kmh)
                
                brng = calculate_bearing(p1[0], p1[1], p2[0], p2[1])
                bearings.append(brng)
            else:
                speeds.append(0.0)
                bearings.append(bearings[-1] if bearings else 0.0)
                
        df['speed'] = speeds
        df['bearing'] = bearings
        
        # 3. データの平滑化（ノイズ除去）
        df = df[df['speed'] < speed_max_limit].reset_index(drop=True)
        df['speed_smooth'] = df['speed'].rolling(window=3, min_periods=1).mean()
        
        # 4. 特徴量の計算（方位角の変化量など）
        df['bearing_diff'] = df['bearing'].diff().abs()
        df['bearing_diff'] = df['bearing_diff'].map(lambda x: 360 - x if x > 180 else x)
        df['turn_cum'] = df['bearing_diff'].rolling(window=5, min_periods=1).sum()

    # --- 高度な分析ロジック ---
    
    # ① フォイリング率
    foiling_df = df[df['speed_smooth'] >= foil_threshold]
    foiling_ratio = (len(fo
