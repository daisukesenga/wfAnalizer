st.subheader("🗺️ セッションマップ（ジャイブ成否のライン色分け）")
        
        fig_map = go.Figure()
        
        # --- 1. 通常の走行軌跡（ベースの線） ---
        fig_map.add_trace(go.Scattermapbox(
            lat=df['lat'], lon=df['lon'],
            mode='lines',
            line=dict(width=2, color='#A0AEC0'), # 目立ちすぎないグレー
            name='通常走行',
            hoverinfo='text',
            text=df['speed_smooth'].map(lambda x: f"速度: {x:.1f} km/h")
        ))
        
        # --- 2. ジャイブ成功・失敗の軌跡をラインで強調 ---
        # 連続したターンポイント（group）ごとに、その区間のラインを描画する
        if not turns.empty:
            for g_id, group in turns.groupby('group'):
                # ターンの前後のマージンを取るため、インデックスを少し広げる（前後3データ点分）
                start_idx = max(0, group.index.min() - 3)
                end_idx = min(len(df) - 1, group.index.max() + 3)
                turn_segment = df.loc[start_idx:end_idx]
                
                # このターンが成功か失敗か判定
                min_speed_in_turn = group['speed_smooth'].min()
                
                if min_speed_in_turn >= jibe_speed_threshold:
                    # ジャイブ成功：緑のライン
                    line_color = '#2ECC71' # 鮮やかな緑
                    line_name = 'ジャイブ成功区間'
                    show_legend = True if 'success_legend_done' not in locals() else False
                    success_legend_done = True
                else:
                    # ジャイブ失敗：赤のライン
                    line_color = '#E74C3C' # 鮮やかな赤
                    line_name = 'ジャイブ失敗区間'
                    show_legend = True if 'fail_legend_done' not in locals() else False
                    fail_legend_done = True
                
                # マップに区間線を追加
                fig_map.add_trace(go.Scattermapbox(
                    lat=turn_segment['lat'], lon=turn_segment['lon'],
                    mode='lines',
                    line=dict(width=5, color=line_color), # ターンは太い線にする
                    name=line_name,
                    showlegend=show_legend, # 凡例が大量に出るのを防ぐ
                    hoverinfo='text',
                    text=turn_segment['speed_smooth'].map(lambda x: f"ターン中 速度: {x:.1f} km/h")
                ))
        
        # --- 3. 直線での沈（ワイプアウト）ポイント ---
        # （※ご要望は「緑と黄のプロットは不要」でしたので、直線沈の赤クロスだけ残しています。不要ならここも削除可能です）
        if not wipeouts.empty:
            fig_map.add_trace(go.Scattermapbox(
                lat=wipeouts['lat'], lon=wipeouts['lon'],
                mode='markers',
                marker=dict(size=12, color='black', symbol='cross'), # 目立つように黒の×に
                name='直線での沈 (Wipeout)'
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
