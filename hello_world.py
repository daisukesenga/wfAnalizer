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
        # タイムゾーンを日本時間に変換（必要に応じて）
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
                
                # 方位角計算
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
        # 短時間（例: 5秒間）での方位角の変化量を算出
        df['bearing_diff'] = df['bearing'].diff().abs()
        df['bearing_diff'] = df['bearing_diff'].map(lambda x: 360 - x if x > 180 else x) # 回り込み補正
        df['turn_cum'] = df['bearing_diff'].rolling(window=5, min_periods=1).sum()

    # --- 高度な分析ロジック ---
    
    # ① フォイリング率
    foiling_df = df[df['speed_smooth'] >= foil_threshold]
    foiling_ratio = (len(foiling_df) / len(df)) * 100 if len(df) > 0 else 0
    
    # ② 直線での沈（ワイプアウト）
    # 条件：方位角の変化が小さく（直線）、3秒間で時速12km以上急減速し、最終的に5km/h以下になった箇所
    df['speed_drop'] = df['speed_smooth'].diff(periods=3) * -1
    wipeouts = df[
        (df['speed_drop'] > 12) & 
        (df['speed_smooth'] < 5) & 
        (df['turn_cum'] < 45)  # ターン中ではない
    ]
    
    # ③ ジャイブ（ターン）の成功・失敗判定
    # 条件：5秒間で方位角が70度以上変わった場所を「ターン」と認識
    turns = df[df['turn_cum'] > 70]
    
    jibe_success_points = []
    jibe_fail_points = []
    
    # 連続したターンポイントをグループ化して1回のジャイブイベントとする
    if not turns.empty:
        turns = turns.copy()
        turns['group'] = (turns['time'].diff().dt.total_seconds() > 10).cumsum()
        
        for g_id, group in turns.groupby('group'):
            # ターンイベント中の最低速度を調査
            min_speed_in_turn = group['speed_smooth'].min()
            center_idx = group.index[len(group) // 2]
            point_info = df.loc[center_idx]
            
            if min_speed_in_turn >= jibe_speed_threshold:
                jibe_success_points.append(point_info)
            else:
                jibe_fail_points.append(point_info)
                
    js_df = pd.DataFrame(jibe_success_points)
    jf_df = pd.DataFrame(jibe_fail_points)
    
    total_jibes = len(js_df) + len(jf_df)
    jibe_success_ratio = (len(js_df) / total_jibes) * 100 if total_jibes > 0 else 0

    # --- UI表示エリア ---
    
    # ダッシュボード統計（4列）
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🚀 最高速度", f"{df['speed_smooth'].max():.1f} km/h")
    col2.metric("📊 推定フォイリング率", f"{foiling_ratio:.1f} %")
    col3.metric("🔄 ジャイブ成功率", f"{jibe_success_ratio:.1f} %", f"成功:{len(js_df)} / 全体:{total_jibes}")
    col4.metric("⚠️ 直線での沈回数", f"{len(wipeouts)} 回")
    
    st.markdown("---")
    
    # メインレイアウト（地図とグラフを左右に並べる）
    left_col, right_col = st.columns([6, 4])
    
    with left_col:
        st.subheader("🗺️ セッションマップ")
        
        # Plotly Mapboxによる地図描画
        fig_map = go.Figure()
        
        # 走行軌跡
        fig_map.add_trace(go.Scattermapbox(
            lat=df['lat'], lon=df['lon'],
            mode='lines',
            line=dict(width=3, color='#1f77b4'),
            name='走行軌跡',
            hoverinfo='text',
            text=df['speed_smooth'].map(lambda x: f"速度: {x:.1f} km/h")
        ))
        
        # 直線での沈ポイント
        if not wipeouts.empty:
            fig_map.add_trace(go.Scattermapbox(
                lat=wipeouts['lat'], lon=wipeouts['lon'],
                mode='markers',
                marker=dict(size=12, color='red', symbol='cross'),
                name='直線での沈 (Wipeout)'
            ))
            
        # ジャイブ成功ポイント
        if not js_df.empty:
            fig_map.add_trace(go.Scattermapbox(
                lat=js_df['lat'], lon=js_df['lon'],
                mode='markers',
                marker=dict(size=10, color='green', symbol='circle'),
                name='ジャイブ成功'
            ))
            
        # ジャイブ失敗ポイント
        if not jf_df.empty:
            fig_map.add_trace(go.Scattermapbox(
                lat=jf_df['lat'], lon=jf_df['lon'],
                mode='markers',
                marker=dict(size=10, color='orange', symbol='circle'),
                name='ジャイブ失速/失敗'
            ))
            
        # マップのレイアウト設定
        fig_map.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=df['lat'].mean(), lon=df['lon'].mean()),
                zoom=14
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
            height=550,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)")
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
    with right_col:
        st.subheader("📈 タイムライン分析")
        
        # 速度の推移グラフ
        fig_speed = px.line(
            df, x='time', y='speed_smooth',
            title="速度推移 (km/h)",
            labels={'speed_smooth': '速度 (km/h)', 'time': '時刻'}
        )
        # フォイリング閾値の基準線を引く
        fig_speed.add_hline(y=foil_threshold, line_dash="dash", line_color="red", annotation_text="フォイリング閾値")
        fig_speed.update_layout(height=260, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_speed, use_container_width=True)
        
        # 進行方向（方位角）の推移グラフ
        fig_bearing = px.line(
            df, x='time', y='bearing',
            title="進行方向（方位/0-360度）",
            labels={'bearing': '方位角 (度)', 'time': '時刻'}
        )
        fig_bearing.update_layout(height=260, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_bearing, use_container_width=True)

    # ログデータのチラ見せ（デバッグ用）
    with st.expander("📂 解析データテーブルの表示"):
        st.dataframe(df[['time', 'speed_smooth', 'bearing', 'turn_cum']].head(100))

else:
    st.info("👆 上記のエリアにスマートウォッチやGPSロガーから出力したGPXファイルをアップロードしてください。")
