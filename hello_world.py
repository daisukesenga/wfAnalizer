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
foil_threshold = st.sidebar.slider("フォイリング開始速度 (km/h)", 0.0, 20.0, 13.5, 0.5)
jibe_speed_threshold = st.sidebar.slider("ジャイブ成功とみなす最低速度 (km/h)", 0.0, 20.0, 11.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 ジャイブ（ターン）判定設定")
jibe_turn_angle_threshold = st.sidebar.slider("直線とみなす最大角度（度）", 10, 120, 45, 5,
                                              help="この角度以下しか曲がっていない区間を『直線（黄色）』とみなします。これを超えるとジャイブ区間になります。")

st.sidebar.markdown("---")
st.sidebar.subheader("🧼 スムージング（ノイズ除去）設定")
smoothing_window = st.sidebar.slider("軌跡の平滑化強度 (データ点数)", 1, 15, 7, 1, 
                                    help="値を大きくするとガタガタが減ります。")

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
        
        # 【位置データのスムージング】
        if smoothing_window > 1:
            df['lat'] = df['lat'].rolling(window=smoothing_window, min_periods=1, center=True).mean()
            df['lon'] = df['lon'].rolling(window=smoothing_window, min_periods=1, center=True).mean()

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
        
        # 3. データの平滑化
        df = df[df['speed'] < speed_max_limit].reset_index(drop=True)
        df['speed_smooth'] = df['speed'].rolling(window=smoothing_window, min_periods=1, center=True).mean()
        
        # 4. 特徴量の計算（方位角の変化量）
        df['bearing_diff'] = df['bearing'].diff().abs()
        df['bearing_diff'] = df['bearing_diff'].map(lambda x: 360 - x if x > 180 else x)
        df['turn_cum'] = df['bearing_diff'].rolling(window=5, min_periods=1).sum()

    # --- 分析ロジック ---
    
    # ① フォイリング率
    foiling_df = df[df['speed_smooth'] >= foil_threshold]
    foiling_ratio = (len(foiling_df) / len(df)) * 100 if len(df) > 0 else 0
    
    # ② 直線での沈（ワイプアウト）
    df['speed_drop'] = df['speed_smooth'].diff(periods=3) * -1
    wipeouts = df[
        (df['speed_drop'] > 12) & 
        (df['speed_smooth'] < 5) & 
        (df['turn_cum'] < jibe_turn_angle_threshold)
    ]
    
    # ③ ジャイブ判定と直線判定の分離
    # 💡 ターン変化量が閾値以下の区間を「直線区間」とするフラグを作成
    df['is_straight'] = df['turn_cum'] <= jibe_turn_angle_threshold
    
    jibe_zone = df[df['turn_cum'] > jibe_turn_angle_threshold]
    
    jibe_success_count = 0
    jibe_fail_count = 0
    valid_turn_indices = [] 
    
    # 各データ点が成功ジャイブか失敗ジャイブかを識別するラベル初期化
    df['segment_type'] = 'normal'
    
    if not jibe_zone.empty:
        jibe_zone = jibe_zone.copy()
        jibe_zone['group'] = (jibe_zone['time'].diff().dt.total_seconds() > 10).cumsum()
        
        for g_id, group in jibe_zone.groupby('group'):
            entry_idx = group.index.min()
            entry_speed = df.loc[entry_idx, 'speed_smooth']
            
            # フォイリング中からのターンのみ有効
            if entry_speed >= foil_threshold:
                valid_turn_indices.extend(group.index.tolist())
                min_speed_in_turn = group['speed_smooth'].min()
                
                # 該当するインデックスの区間タイプを書き換え
                if min_speed_in_turn >= jibe_speed_threshold:
                    jibe_success_count += 1
                    df.loc[group.index, 'segment_type'] = 'success'
                else:
                    jibe_fail_count += 1
                    df.loc[group.index, 'segment_type'] = 'fail'
                    
    # 直線区間のラベルを上書き（ジャイブとして上書きされていない、かつフォイリング速度以上の場所）
    df.loc[(df['is_straight']) & (df['speed_smooth'] >= foil_threshold), 'segment_type'] = 'straight'
                
    total_jibes = jibe_success_count + jibe_fail_count
    jibe_success_ratio = (jibe_success_count / total_jibes) * 100 if total_jibes > 0 else 0

    # --- UI表示エリア ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🚀 最高速度", f"{df['speed_smooth'].max():.1f} km/h")
    col2.metric("📊 推定フォイリング率", f"{foiling_ratio:.1f} %")
    col3.metric("🔄 ジャイブ成功率", f"{jibe_success_ratio:.1f} %", f"成功:{jibe_success_count} / 全体:{total_jibes}")
    col4.metric("⚠️ 直線での沈回数", f"{len(wipeouts)} 回")
    
    st.markdown("---")
    
    left_col, right_col = st.columns([6, 4])
    
    with left_col:
        st.subheader("🗺️ セッションマップ（直線＝黄 / 成功＝緑 / 失敗＝赤）")
        
        fig_map = go.Figure()
        
        # --- 1. ベースの通常・低速軌跡（グレー） ---
        fig_map.add_trace(go.Scattermapbox(
            lat=df['lat'], lon=df['lon'],
            mode='lines',
            line=dict(width=1.5, color='#CBD5E0'), 
            name='通常・低速走行',
            hoverinfo='skip'
        ))
        
        # --- 2. 状態が変わるポイントごとにグループ化して色分け描画 ---
        # 連続する同じセグメントタイプをまとめて描画するロジック
        df['type_block'] = (df['segment_type'] != df['segment_type'].shift()).cumsum()
        
        straight_legend = False
        success_legend = False
        fail_legend = False
        
        for b_id, block in df.groupby('type_block'):
            seg_type = block['segment_type'].iloc[0]
            if seg_type == 'normal':
                continue
                
            # 前後の繋がりを滑らかにするマージン
            start_idx = max(0, block.index.min() - 1)
            end_idx = min(len(df) - 1, block.index.max() + 1)
            sub_seg = df.loc[start_idx:end_idx]
            
            if seg_type == 'straight':
                color = '#F1C40F'  # 💡 直線区間：鮮やかな黄色
                name = 'フォイリング直線区間'
                show_leg = not straight_legend
                straight_legend = True
                width = 3
            elif seg_type == 'success':
                color = '#2ECC71'  # ジャイブ成功：緑
                name = 'ジャイブ成功区間'
                show_leg = not success_legend
                success_legend = True
                width = 5
            elif seg_type == 'fail':
                color = '#E74C3C'  # ジャイブ失敗：赤
                name = 'ジャイブ失敗区間'
                show_leg = not fail_legend
                fail_legend = True
                width = 5
                
            fig_map.add_trace(go.Scattermapbox(
                lat=sub_seg['lat'], lon=sub_seg['lon'],
                mode='lines',
                line=dict(width=width, color=color),
                name=name,
                showlegend=show_leg,
                hoverinfo='text',
                text=sub_seg['speed_smooth'].map(lambda x: f"速度: {x:.1f} km/h")
            ))
        
        # --- 3. 直線での沈（ワイプアウト）ポイント ---
        if not wipeouts.empty:
            fig_map.add_trace(go.Scattermapbox(
                lat=wipeouts['lat'], lon=wipeouts['lon'],
                mode='markers',
                marker=dict(size=12, color='black', symbol='cross'),
                name='直線での沈 (Wipeout)'
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
        fig_speed.add_hline(y=foil_threshold, line_dash="dash", line_color="red", annotation_text="フォイリング閾値")
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
        st.dataframe(df[['time', 'speed_smooth', 'bearing', 'turn_cum']].head(100))

else:
    st.info("👆 上記のエリアにスマートウォッチやGPSロガーから出力したGPXファイルをアップロードしてください。")
