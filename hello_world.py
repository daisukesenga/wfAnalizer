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
st.write("GPXファイルをアップロードして、左右別のジャイブ成功率、区間距離、フォイル速度の統計を分析します。")

# --- サイドバーの設定 ---
st.sidebar.header("⚙️ 解析パラメータ設定")
foil_start_threshold = st.sidebar.slider("フォイリング開始速度 (km/h)", 0.0, 20.0, 13.5, 0.5)
foil_end_threshold = st.sidebar.slider("フォイリング終了速度 (km/h)", 0.0, 20.0, 10.0, 0.5)

if foil_end_threshold > foil_start_threshold:
    st.sidebar.error("⚠️ 終了速度は、開始速度以下の値に設定してください。")

jibe_speed_threshold = st.sidebar.slider("ジャイブ成功とみなす最低速度 (km/h)", 0.0, 20.0, 11.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 ジャイブ（ターン）判定設定")
jibe_turn_angle_threshold = st.sidebar.slider("直線とみなす最大角度（度）", 10, 120, 45, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("🧼 スムージング（ノイズ除去）設定")
smoothing_window = st.sidebar.slider("軌跡の平滑化強度 (データ点数)", 1, 15, 7, 1)

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
        
        # ファイル内の時刻をそのままローカル時間として扱う
        df['time'] = df['time'].dt.tz_localize(None)
        df['time_jst_str'] = df['time'].dt.strftime('%H:%M:%S')
        
        # 位置データのスムージング
        if smoothing_window > 1:
            df['lat'] = df['lat'].rolling(window=smoothing_window, min_periods=1, center=True).mean()
            df['lon'] = df['lon'].rolling(window=smoothing_window, min_periods=1, center=True).mean()

        # 2. 速度・時間差・方位角・移動距離の計算
        df['time_diff'] = df['time'].diff().dt.total_seconds()
        
        speeds = [0.0]
        bearings = [0.0]
        distances = [0.0] # 💡 各点間の移動距離（メートル）
        
        for i in range(1, len(df)):
            p1 = (df.loc[i-1, 'lat'], df.loc[i-1, 'lon'])
            p2 = (df.loc[i, 'lat'], df.loc[i, 'lon'])
            t_diff = df.loc[i, 'time_diff']
            
            dist = geodesic(p1, p2).meters
            distances.append(dist)
            
            if t_diff > 0:
                speed_kmh = (dist / t_diff) * 3.6
                speeds.append(speed_kmh)
                brng = calculate_bearing(p1[0], p1[1], p2[0], p2[1])
                bearings.append(brng)
            else:
                speeds.append(0.0)
                bearings.append(bearings[-1] if bearings else 0.0)
                
        df['speed'] = speeds
        df['bearing'] = bearings
        df['distance_m'] = distances
        
        # 3. データの平滑化
        df = df[df['speed'] < speed_max_limit].reset_index(drop=True)
        df['speed_smooth'] = df['speed'].rolling(window=smoothing_window, min_periods=1, center=True).mean()
        
        # 4. 特徴量の計算
        raw_bearing_diff = df['bearing'].diff()
        raw_bearing_diff = raw_bearing_diff.map(lambda x: x - 360 if x > 180 else (x + 360 if x < -180 else x))
        df['bearing_diff_signed'] = raw_bearing_diff
        df['bearing_diff'] = df['bearing_diff_signed'].abs()
        
        df['turn_cum'] = df['bearing_diff'].rolling(window=5, min_periods=1).sum()
        df['turn_dir_cum'] = df['bearing_diff_signed'].rolling(window=5, min_periods=1).sum()

    # --- 🏄‍♂️ フォイリング状態追跡ロジック ---
    is_foiling_list = []
    currently_foiling = False
    
    for s in df['speed_smooth']:
        if not currently_foiling:
            if s >= foil_start_threshold:
                currently_foiling = True
        else:
            if s < foil_end_threshold:
                currently_foiling = False
        is_foiling_list.append(currently_foiling)
        
    df['is_foiling'] = is_foiling_list
    
    # ② 基本判定のトグル
    df['is_straight'] = df['turn_cum'] <= jibe_turn_angle_threshold
    jibe_zone = df[df['turn_cum'] > jibe_turn_angle_threshold]
    
    right_jibe_success = 0
    right_jibe_fail = 0
    left_jibe_success = 0
    left_jibe_fail = 0
    
    df['segment_type'] = 'normal'
    
    if not jibe_zone.empty:
        jibe_zone = jibe_zone.copy()
        jibe_zone['group'] = (jibe_zone['time'].diff().dt.total_seconds() > 10).cumsum()
        
        for g_id, group in jibe_zone.groupby('group'):
            entry_idx = group.index.min()
            
            if df.loc[entry_idx, 'is_foiling']:
                min_speed_in_turn = group['speed_smooth'].min()
                is_right_turn = group['turn_dir_cum'].sum() >= 0
                
                if min_speed_in_turn >= jibe_speed_threshold:
                    df.loc[group.index, 'segment_type'] = 'success'
                    if is_right_turn:
                        right_jibe_success += 1
                    else:
                        left_jibe_success += 1
                else:
                    df.loc[group.index, 'segment_type'] = 'fail'
                    if is_right_turn:
                        right_jibe_fail += 1
                    else:
                        left_jibe_fail += 1
                    
    df.loc[(df['segment_type'] == 'normal') & (df['is_straight']) & (df['is_foiling']), 'segment_type'] = 'straight_internal'

    # 【ステップ3】「直線からの沈」検出
    wipeout_count = 0
    wipeout_times = []
    i = 0
    n = len(df)
    
    while i < n - 1:
        if df.loc[i, 'segment_type'] == 'straight_internal':
            j = i
            while j < n and df.loc[j, 'segment_type'] == 'straight_internal':
                j += 1
                
            if j < n:
                if df.loc[j, 'segment_type'] in ['success', 'fail']:
                    i = j
                    continue
                
                if not df.loc[j, 'is_foiling']:
                    fall_idx = j
                    
                    k = fall_idx
                    while k < n and not df.loc[k, 'is_foiling'] and df.loc[k, 'segment_type'] == 'normal':
                        k += 1
                        
                    duration = (df.loc[min(k, n-1), 'time'] - df.loc[fall_idx, 'time']).total_seconds()
                    
                    if duration >= 10:
                        wipeout_count += 1
                        wipeout_times.append(df.loc[fall_idx, 'time'])
                        end_red_idx = min(fall_idx + 10, k, n - 1)
                        df.loc[fall_idx:end_red_idx, 'segment_type'] = 'wipeout_line'
                    i = k
                    continue
            i = j
        else:
            i += 1
            
    df.loc[df['segment_type'] == 'straight_internal', 'segment_type'] = 'normal'
                
    total_right_jibes = right_jibe_success + right_jibe_fail
    right_jibe_ratio = (right_jibe_success / total_right_jibes) * 100 if total_right_jibes > 0 else 0
    
    total_left_jibes = left_jibe_success + left_jibe_fail
    left_jibe_ratio = (left_jibe_success / total_left_jibes) * 100 if total_left_jibes > 0 else 0

    # 💡 --- 新設：走行距離およびフォイル速度の統計計算 ---
    # 通常走行（normal および wipeout_line）の距離
    normal_distance_km = df[df['segment_type'].isin(['normal', 'wipeout_line'])]['distance_m'].sum() / 1000.0
    # ジャイブ区間（success および fail）の距離
    jibe_distance_km = df[df['segment_type'].isin(['success', 'fail'])]['distance_m'].sum() / 1000.0
    
    # フォイリング中（is_foiling == True）の速度データ抽出
    foiling_speeds = df[df['is_foiling'] == True]['speed_smooth']
    if not foiling_speeds.empty:
        foil_mean_speed = foiling_speeds.mean()
        foil_median_speed = foiling_speeds.median()
    else:
        foil_mean_speed = 0.0
        foil_median_speed = 0.0

    # --- UI表示エリア （上段：基本メリック） ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🚀 最高速度", f"{df['speed_smooth'].max():.1f} km/h")
    col2.metric("➡️ 右ターン(時計回り) 成功率", f"{right_jibe_ratio:.1f} %", f"成功:{right_jibe_success} / 全体:{total_right_jibes}")
    col3.metric("⬅️ 左ターン(反時計) 成功率", f"{left_jibe_ratio:.1f} %", f"成功:{left_jibe_success} / 全体:{total_left_jibes}")
    col4.metric("⚠️ 直線からの沈回数", f"{wipeout_count} 回")
    
    # 💡 --- UI表示エリア （中段：ご要望の各種統計を配置） ---
    st.markdown("### 📊 走行距離＆フォイリング速度統計")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    stat_col1.metric("🛣️ 通常走行 合計距離", f"{normal_distance_km:.2f} km")
    stat_col2.metric("🔄 ジャイブ区間 合計距離", f"{jibe_distance_km:.2f} km", "成功＋失敗の合計")
    stat_col3.metric("📈 フォイル中 平均速度", f"{foil_mean_speed:.1f} km/h")
    stat_col4.metric("🎯 フォイル中 中央値速度", f"{foil_median_speed:.1f} km/h")
    
    st.markdown("---")
    
    left_col, right_col = st.columns([6, 4])
    
    with left_col:
        st.subheader("🗺️ セッションマップ（成功＝緑 / 失敗＝赤 / 直線沈＝黄線）")
        
        fig_map = go.Figure()
        
        # --- 1. ベースの通常走行軌跡（グレー） ---
        fig_map.add_trace(go.Scattermapbox(
            lat=df['lat'], lon=df['lon'],
            mode='lines',
            line=dict(width=2, color='#A0AEC0'), 
            name='通常走行（直線含む）',
            hoverinfo='text',
            text=df.apply(lambda row: f"時刻: {row['time_jst_str']}<br>速度: {row['speed_smooth']:.1f} km/h", axis=1)
        ))
        
        # --- 2. 各種イベント区間の重ね描き ---
        df['type_block'] = (df['segment_type'] != df['segment_type'].shift()).cumsum()
        
        success_legend = False
        fail_legend = False
        wipeout_legend = False
        
        for b_id, block in df.groupby('type_block'):
            seg_type = block['segment_type'].iloc[0]
            if seg_type == 'normal':
                continue
                
            start_idx = max(0, block.index.min() - 1)
            end_idx = min(len(df) - 1, block.index.max() + 1)
            sub_seg = df.loc[start_idx:end_idx]
            
            if seg_type == 'success':
                color = '#2ECC71'
                name = 'ジャイブ成功区間'
                show_leg = not success_legend
                success_legend = True
                width = 5
            elif seg_type == 'fail':
                color = '#E74C3C'
                name = 'ジャイブ失敗区間'
                show_leg = not fail_legend
                fail_legend = True
                width = 5
            elif seg_type == 'wipeout_line':
                color = '#F1C40F'
                name = '直線からの沈（落水減速区間）'
                show_leg = not wipeout_legend
                wipeout_legend = True
                width = 5.5
                
            fig_map.add_trace(go.Scattermapbox(
                lat=sub_seg['lat'], lon=sub_seg['lon'],
                mode='lines',
                line=dict(width=width, color=color),
                name=name,
                showlegend=show_leg,
                hoverinfo='text',
                text=sub_seg.apply(lambda row: f"時刻: {row['time_jst_str']}<br>速度: {row['speed_smooth']:.1f} km/h", axis=1)
            ))
            
        fig_map.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=df['lat'].mean(), lon=df['lon'].mean()),
                zoom=14
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
            height=550,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.9)")
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
    with right_col:
        st.subheader("📈 タイムライン分析")
        
        fig_speed = px.line(
            df, x='time', y='speed_smooth',
            title="速度推移 (km/h)",
            labels={'speed_smooth': '速度 (km/h)', 'time': '時刻'}
        )
        fig_speed.add_hline(y=foil_start_threshold, line_dash="dash", line_color="red", annotation_text="開始閾値")
        fig_speed.add_hline(y=foil_end_threshold, line_dash="dot", line_color="orange", annotation_text="終了閾値")
        
        for w_time in wipeout_times:
            fig_speed.add_vline(x=w_time, line_color="black", line_dash="dash")
            
        fig_speed.update_layout(height=260, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_speed, use_container_width=True)
        
        fig_bearing = px.line(
            df, x='time', y='bearing',
            title="進行方向（方位/0-360度）",
            labels={'bearing': '方位角 (度)', 'time': '時刻'}
        )
        fig_bearing.update_layout(height=260, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_bearing, use_container_width=True)

    with st.expander("📂 解析データテーブルの表示"):
        st.dataframe(df[['time_jst_str', 'speed_smooth', 'bearing', 'turn_cum', 'is_foiling', 'segment_type']].rename(columns={'time_jst_str': '時刻'}).head(100))

else:
    st.info("👆 上記のエリアにスマートウォッチやGPSロガーから出力したGPXファイルをアップロードしてください。")
