import streamlit as st
import gpxpy
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
from haversine import haversine, Unit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

st.set_page_config(page_title="Wing Foil GPX Analyzer", layout="wide")

st.title("🪁 Wing Foil GPX Analyzer")
st.write("Wing foilセッションを詳細に分析します")

# ファイルアップロード
uploaded_file = st.file_uploader("GPXファイルをアップロード", type="gpx")

def detect_jibes(df, speed_threshold=2.0, min_speed_before=3.0, rel_drop=0.4):
    """ジャイブ失敗を検出（速度が急激に低下した箇所）

    改善点:
    - 速度を短い窓で平滑化してノイズを低減
    - 絶対的な速度低下（speed_threshold）に加え、
      事前速度がある程度以上であること（min_speed_before）と
      相対的な減少率（rel_drop）を満たす場合のみ検出する
    """
    jibes_failed = []

    if len(df) > 1:
        # 短い窓で平滑化してノイズを抑える
        df['speed_smooth'] = df['speed'].rolling(window=3, center=True, min_periods=1).mean()
        df['speed_diff'] = df['speed_smooth'].diff()

        for idx in range(1, len(df)):
            speed_before = df['speed_smooth'].iloc[idx - 1]
            speed_after = df['speed_smooth'].iloc[idx]

            if pd.isna(speed_before) or pd.isna(speed_after):
                continue

            abs_drop = speed_before - speed_after
            rel_drop_actual = (abs_drop / speed_before) if speed_before > 0 else 0

            # raw の前後速度も取得して追加チェックに使う
            raw_before = df['speed'].iloc[idx - 1]
            raw_after = df['speed'].iloc[idx]

            # 絶対閾値、事前速度、相対減少率を満たし、かつ raw でも減速している場合のみ検出
            if (
                abs_drop > speed_threshold
                and speed_before >= min_speed_before
                and rel_drop_actual >= rel_drop
                and raw_before > raw_after
            ):
                speed_loss_raw = raw_before - raw_after
                if speed_loss_raw <= 0:
                    # 念のため raw の損失が正でない場合はスキップ
                    continue

                jibes_failed.append({
                    'index': idx,
                    'latitude': df['latitude'].iloc[idx],
                    'longitude': df['longitude'].iloc[idx],
                    'speed_before': raw_before,
                    'speed_after': raw_after,
                    'speed_loss': speed_loss_raw,
                    'time': df['time'].iloc[idx]
                })

    return jibes_failed

def detect_crashes(df, elevation_threshold=0.5):
    """沈を検出（標高が急激に低下した箇所）"""
    crashes = []
    
    if len(df) > 1 and df['elevation'].notna().any():
        df['elevation_diff'] = df['elevation'].diff()
        
        for idx in range(1, len(df)):
            # 標高が大きく低下した場合（沈と判定）
            if pd.notna(df['elevation_diff'].iloc[idx]) and df['elevation_diff'].iloc[idx] < -elevation_threshold:
                crashes.append({
                    'index': idx,
                    'latitude': df['latitude'].iloc[idx],
                    'longitude': df['longitude'].iloc[idx],
                    'elevation_before': df['elevation'].iloc[idx - 1],
                    'elevation_after': df['elevation'].iloc[idx],
                    'elevation_loss': abs(df['elevation'].iloc[idx - 1] - df['elevation'].iloc[idx]),
                    'time': df['time'].iloc[idx]
                })
    
    return crashes

def get_speed_color(speed, vmin, vmax):
    """速度に基づいて色を取得"""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap('RdYlBu_r')
    return mcolors.to_hex(cmap(norm(speed)))

if uploaded_file is not None:
    gpx_file = gpxpy.parse(uploaded_file)
    
    # トラック情報を抽出
    track_data = []
    for track in gpx_file.tracks:
        for segment in track.segments:
            for point in segment.points:
                track_data.append({
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'elevation': point.elevation,
                    'time': point.time
                })
    
    df = pd.DataFrame(track_data)
    
    # 速度を計算（隣接ポイント間の距離と時間差から）
    speeds = []
    for i in range(len(df)):
        if i == 0:
            speeds.append(0)
        else:
            coords_1 = (df['latitude'].iloc[i-1], df['longitude'].iloc[i-1])
            coords_2 = (df['latitude'].iloc[i], df['longitude'].iloc[i])
            distance_km = haversine(coords_1, coords_2, unit=Unit.KILOMETERS)
            
            time_diff = (df['time'].iloc[i] - df['time'].iloc[i-1]).total_seconds() / 3600
            
            if time_diff > 0:
                speed = distance_km / time_diff
            else:
                speed = 0
            
            speeds.append(speed)
    
    df['speed'] = speeds
    
    # 異常値を修正（移動速度が100km/h以上の場合は前値を使用）
    df.loc[df['speed'] > 100, 'speed'] = df['speed'].shift(1)
    
    # ジャイブ失敗と沈を検出
    jibes_failed = detect_jibes(df, speed_threshold=1.5)
    crashes = detect_crashes(df, elevation_threshold=0.3)
    
    # ジャイブ成功率を計算
    total_jibes = len(jibes_failed)
    jibe_success_rate = max(0, 100 - (total_jibes * 20)) if total_jibes > 0 else 100
    jibe_success_rate = min(100, jibe_success_rate)
    
    # 速度統計
    avg_speed = df['speed'].mean()
    max_speed = df['speed'].max()
    min_speed = df['speed'].min()
    
    # 左右に分割したレイアウト
    col_map, col_stats = st.columns([3, 1])
    
    with col_map:
        st.subheader("🗺️ トラック地図 (速度色分け)")
        
        # 地図作成
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles="OpenStreetMap"
        )
        
        # 速度に基づいて軌跡を描画
        vmin, vmax = df['speed'].min(), df['speed'].max()
        
        for i in range(len(df) - 1):
            color = get_speed_color(df['speed'].iloc[i], vmin, vmax)
            
            folium.PolyLine(
                locations=[
                    (df['latitude'].iloc[i], df['longitude'].iloc[i]),
                    (df['latitude'].iloc[i+1], df['longitude'].iloc[i+1])
                ],
                color=color,
                weight=3,
                opacity=0.8,
                popup=f"Speed: {df['speed'].iloc[i]:.1f} km/h"
            ).add_to(m)
        
        # スタート地点（緑）
        folium.CircleMarker(
            location=(df['latitude'].iloc[0], df['longitude'].iloc[0]),
            radius=8,
            popup="🟢 Start",
            color='green',
            fill=True,
            fillColor='green',
            fillOpacity=0.9,
            weight=2
        ).add_to(m)
        
        # ゴール地点（青）
        folium.CircleMarker(
            location=(df['latitude'].iloc[-1], df['longitude'].iloc[-1]),
            radius=8,
            popup="🔵 Finish",
            color='blue',
            fill=True,
            fillColor='blue',
            fillOpacity=0.9,
            weight=2
        ).add_to(m)
        
        # ジャイブ失敗マーク（オレンジの✘）
        for jibe in jibes_failed:
            folium.Marker(
                location=(jibe['latitude'], jibe['longitude']),
                popup=f"❌ Jibe Failed<br>Speed Loss: {jibe['speed_loss']:.1f} km/h<br>Before: {jibe['speed_before']:.1f} km/h → After: {jibe['speed_after']:.1f} km/h",
                icon=folium.Icon(color='orange', icon='times', prefix='fa', icon_color='white')
            ).add_to(m)
        
        # 沈マーク（赤の✘）
        for crash in crashes:
            folium.Marker(
                location=(crash['latitude'], crash['longitude']),
                popup=f"💧 Crash<br>Elevation Loss: {crash['elevation_loss']:.1f}m<br>Before: {crash['elevation_before']:.1f}m → After: {crash['elevation_after']:.1f}m",
                icon=folium.Icon(color='red', icon='times', prefix='fa', icon_color='white')
            ).add_to(m)
        
        # カラーバーを追加
        colormap = folium.LinearColormap(
            colors=['#0000ff', '#00ffff', '#ffff00', '#ff0000'],
            vmin=vmin,
            vmax=vmax,
            caption='Speed (km/h)'
        )
        colormap.add_to(m)
        
        st_folium(m, width=1000, height=600)
    
    with col_stats:
        st.subheader("📊 統計情報")
        st.metric("✅ Jibe成功率", f"{jibe_success_rate:.0f}%")
        st.metric("📈 平均速度", f"{avg_speed:.1f} km/h")
        st.metric("⚡ 最高速度", f"{max_speed:.1f} km/h")
        st.metric("🚫 ジャイブ失敗", len(jibes_failed))
        st.metric("💧 沈", len(crashes))
        st.metric("⏱️ セッション時間", f"{len(df)} points")
    
    # 詳細分析タブ
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["📋 詳細データ", "❌ ジャイブ失敗", "💧 沈", "📈 速度グラフ"])
    
    with tab1:
        st.subheader("全トラックデータ")
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.subheader("ジャイブ失敗の詳細")
        if jibes_failed:
            jibes_df = pd.DataFrame(jibes_failed)
            st.dataframe(jibes_df[['time', 'speed_before', 'speed_after', 'speed_loss']], use_container_width=True)
            st.info(f"合計 {len(jibes_failed)} 回のジャイブ失敗を検出しました")
        else:
            st.success("🎉 ジャイブ失敗なし！完璧なセッションです！")
    
    with tab3:
        st.subheader("沈の詳細")
        if crashes:
            crashes_df = pd.DataFrame(crashes)
            st.dataframe(crashes_df[['time', 'elevation_before', 'elevation_after', 'elevation_loss']], use_container_width=True)
            st.info(f"合計 {len(crashes)} 回の沈を検出しました")
        else:
            st.success("🎉 沈なし！素晴らしいセッションです！")
    
    with tab4:
        st.subheader("速度の時系列グラフ")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(range(len(df)), df['speed'], linewidth=2, color='#1f77b4')
        ax.fill_between(range(len(df)), df['speed'], alpha=0.3, color='#1f77b4')
        ax.set_xlabel("Time Point")
        ax.set_ylabel("Speed (km/h)")
        ax.set_title("Speed Profile")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
else:
    st.info("👆 GPXファイルをアップロードして開始してください")
