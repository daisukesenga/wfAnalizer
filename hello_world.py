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

def bearing_between(lat1, lon1, lat2, lon2):
    """2点間の方位角を度で返す（0-360）"""
    # ラジアン変換
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)

    x = np.sin(dlambda) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    bearing = np.degrees(np.arctan2(x, y))
    bearing = (bearing + 360) % 360
    return bearing


def angle_diff_signed(a, b):
    """a->b の符号付き角差を -180..180 の範囲で返す"""
    diff = (b - a + 180) % 360 - 180
    return diff


def detect_jibes_by_turn(df, angle_threshold=60.0, duration_threshold=20.0, min_angle=10.0, min_avg_speed=5.0):
    """20秒以内の区間でジャイブを判定する。

    - ジャイブ: 進入角度と脱出角度の差が `min_angle` 以上かつ `angle_threshold` 以下（度）で、
      開始から終了までの時間が `duration_threshold` 秒以内の場合。
    - 速度条件: 区間内の平均速度が `min_avg_speed` km/h 未満なら無視。

    戻り値: (jibes_all, jibes_failed)
    各要素は dict: start_index,end_index,start_time,end_time,duration_s,angle_deg,direction
    """
    jibes = []
    jibes_failed = []

    n = len(df)
    if n < 2:
        return jibes, jibes_failed

    # セグメント方位（points i -> i+1）
    bearings = []
    for i in range(n - 1):
        bearings.append(bearing_between(df['latitude'].iloc[i], df['longitude'].iloc[i], df['latitude'].iloc[i + 1], df['longitude'].iloc[i + 1]))

    i = 0
    while i < n - 1:
        start_idx = i
        start_time = pd.to_datetime(df['time'].iloc[start_idx])
        start_bearing = bearings[start_idx]
        found = False

        # 20秒以内の最大区間を探す
        end_idx = start_idx + 1
        last_valid_end = None
        while end_idx < n:
            end_time = pd.to_datetime(df['time'].iloc[end_idx])
            duration_s = (end_time - start_time).total_seconds()
            if duration_s > duration_threshold:
                break
            last_valid_end = end_idx
            end_idx += 1

        if last_valid_end is None:
            i += 1
            continue

        end_bearing = bearings[last_valid_end - 1]
        angle_deg = abs(angle_diff_signed(start_bearing, end_bearing))
        if angle_deg < min_angle or angle_deg > angle_threshold:
            i += 1
            continue

        window_speeds = df['speed'].iloc[start_idx:last_valid_end + 1] if 'speed' in df.columns else pd.Series([0.0])
        avg_speed = window_speeds.mean() if len(window_speeds) > 0 else 0.0
        if avg_speed < min_avg_speed:
            i += 1
            continue

        end_time = pd.to_datetime(df['time'].iloc[last_valid_end])
        duration_s = (end_time - start_time).total_seconds()
        direction = 'starboard' if angle_diff_signed(start_bearing, end_bearing) > 0 else 'port'
        event = {
            'start_index': int(start_idx),
            'end_index': int(last_valid_end),
            'start_time': start_time,
            'end_time': end_time,
            'duration_s': duration_s,
            'angle_deg': float(angle_deg),
            'direction': direction,
        }
        jibes.append(event)
        if duration_s >= duration_threshold:
            jibes_failed.append(event)
        i = last_valid_end + 1

    return jibes, jibes_failed

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


def detect_speed_drop_crashes(df, speed_threshold=5.0, sustained_seconds=10):
    """速度による沈を検出.

    - 条件: 直前の連続した区間で速度が `speed_threshold` 以上が
      `sustained_seconds` 以上続いた後、現在点で速度が `speed_threshold` 未満になった場合。

    戻り値: list of crashes with keys: index, latitude, longitude, time, prior_start_time, prior_end_time, prior_duration_s
    """
    crashes = []
    if 'speed' not in df.columns or len(df) < 2:
        return crashes

    speeds = df['speed'].fillna(0).values

    for i in range(1, len(df)):
        # 現在点が閾値未満で、直前は閾値以上であった場合にのみ検査
        if speeds[i] < speed_threshold and speeds[i-1] >= speed_threshold:
            # 直前の連続区間の開始を探す
            j = i-1
            while j >= 0 and speeds[j] >= speed_threshold:
                j -= 1
            start_idx = j + 1
            end_idx = i - 1
            # 時間差を計算
            start_time = pd.to_datetime(df['time'].iloc[start_idx])
            end_time = pd.to_datetime(df['time'].iloc[end_idx])
            prior_duration_s = (end_time - start_time).total_seconds()
            if prior_duration_s >= sustained_seconds:
                crashes.append({
                    'index': int(i),
                    'latitude': df['latitude'].iloc[i],
                    'longitude': df['longitude'].iloc[i],
                    'time': df['time'].iloc[i],
                    'prior_start_time': start_time,
                    'prior_end_time': end_time,
                    'prior_duration_s': prior_duration_s,
                    'crash_type': 'speed_drop',
                })

    return crashes


def get_speed_color(speed, vmin, vmax):
    """速度に基づいて色を取得"""
    speed_clamped = min(speed, vmax)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap('RdYlBu_r')
    return mcolors.to_hex(cmap(norm(speed_clamped)))

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
    
    # ジャイブ（角度ベース）と沈を検出
    jibes, jibes_failed = detect_jibes_by_turn(df, angle_threshold=60.0, duration_threshold=20.0)
    crashes = detect_crashes(df, elevation_threshold=0.3)
    # 速度低下による沈も検出（5 km/h閾値、継続10秒）
    speed_crashes = detect_speed_drop_crashes(df, speed_threshold=5.0, sustained_seconds=10)
    if speed_crashes:
        crashes.extend(speed_crashes)

    # ジャイブ成功率を計算（失敗率ベース）
    total_jibes = len(jibes)
    failed_jibes = len(jibes_failed)
    if total_jibes > 0:
        jibe_success_rate = max(0.0, min(100.0, (1.0 - (failed_jibes / total_jibes)) * 100.0))
    else:
        jibe_success_rate = 100.0
    
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
        vmin, vmax = df['speed'].min(), 20.0
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
        
        # ジャイブ（角度ベース）のマーク（成功:緑, 失敗:オレンジ）
        failed_end_idxs = set([e['end_index'] for e in jibes_failed])
        for j in jibes:
            idx = j['end_index']
            is_failed = idx in failed_end_idxs
            color = 'orange' if is_failed else 'green'
            folium.CircleMarker(
                location=(df['latitude'].iloc[idx], df['longitude'].iloc[idx]),
                radius=6,
                popup=f"{'❌' if is_failed else '✅'} Jibe<br>Angle: {j['angle_deg']:.0f}°<br>Duration: {j['duration_s']:.1f}s<br>Direction: {j['direction']}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.9,
                weight=1
            ).add_to(m)

        # 沈マーク（赤の✘）
        for crash in crashes:
            if crash.get('crash_type') == 'speed_drop':
                popup_text = (
                    f"💧 Crash (speed drop)<br>Prior run: {crash['prior_duration_s']:.0f}s @ ≥5 km/h<br>"
                    f"Dropped below 5 km/h at {crash['time']}"
                )
            else:
                popup_text = (
                    f"💧 Crash<br>Elevation Loss: {crash['elevation_loss']:.1f}m<br>"
                    f"Before: {crash['elevation_before']:.1f}m → After: {crash['elevation_after']:.1f}m"
                )

            folium.Marker(
                location=(crash['latitude'], crash['longitude']),
                popup=popup_text,
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
    tab2, tab3, tab4 = st.tabs(["❌ ジャイブ失敗", "💧 沈", "📈 速度グラフ"])
    
    with tab2:
        st.subheader("🌀 ジャイブの詳細")
        if total_jibes > 0:
            jibes_df = pd.DataFrame(jibes)
            jibes_df['start_time'] = jibes_df['start_time']
            jibes_df['end_time'] = jibes_df['end_time']
            st.dataframe(jibes_df[['start_time', 'end_time', 'duration_s', 'angle_deg', 'direction']], use_container_width=True)
            st.info(f"合計 {total_jibes} 回のジャイブを検出しました（失敗: {failed_jibes}）")
        else:
            st.success("🎉 ジャイブなし！完璧なセッションです！")
    
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
