import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ANN 多層原理視覺化", layout="wide")

st.title("🌊 多層神經網路：物理上的『連鎖壓力』")

# --- 側邊欄：參數調整 ---
st.sidebar.header("🛠️ 調整水流閥門 (權重)")
w1 = st.sidebar.slider("第一層閥門 W1 (輸入能量放大率)", 0.0, 5.0, 1.5)
w2 = st.sidebar.slider("第二層閥門 W2 (中間能量轉化率)", 0.0, 5.0, 0.8)
target = st.sidebar.slider("🎯 目標水量 (正確答案)", 0.0, 1.0, 0.9)

# --- 計算邏輯 (物理公式) ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 前向傳播 (Forward Pass)
input_val = 1.0 # 假設輸入一個單位的努力
hidden_out = sigmoid(input_val * w1)
final_out = sigmoid(hidden_out * w2)

# --- 視覺化呈現 (改用 Plotly 避免中文亂碼) ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💧 能量流動圖")
    fig = go.Figure()
    # 畫節點與連線
    fig.add_trace(go.Scatter(x=[0, 1, 2], y=[0, 0, 0], mode='markers+text',
                             marker=dict(size=[40, 60, 80], color=['#AEEEEE', '#5CACEE', '#1874CD']),
                             text=["輸入", "隱藏層", "輸出"], textposition="top center"))
    # 畫水管 (權重)
    fig.add_annotation(x=0.5, y=0.05, text=f"W1={w1}", showarrow=False)
    fig.add_annotation(x=1.5, y=0.05, text=f"W2={w2}", showarrow=False)
    
    fig.update_layout(height=300, showlegend=False, yaxis=dict(visible=False), xaxis=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("⚖️ 誤差比對")
    error = final_out - target
    loss = 0.5 * (error**2)
    st.metric("最終水量", f"{final_out:.2f}")
    st.metric("誤差 (Loss)", f"{loss:.4f}", delta=f"{error:.2f}", delta_color="inverse")

# --- 核心邏輯：反向傳播 (物理含意) ---
st.divider()
st.header("🧬 為什麼後面的錯，要前面的人負責？")

# 計算梯度 (連鎖律)
# dLoss/dw2 = (y-t) * y(1-y) * h
grad_w2 = error * (final_out * (1 - final_out)) * hidden_out
# dLoss/dw1 = (y-t) * y(1-y) * w2 * h(1-h) * x
grad_w1 = error * (final_out * (1 - final_out)) * w2 * (hidden_out * (1 - hidden_out)) * input_val

c1, c2 = st.columns(2)
with c1:
    st.write("### 📍 後層 (W2) 的責任")
    st.write(f"當前壓力回傳值：**{grad_w2:.4f}**")
    st.markdown("> **物理含意**：這就像是在水管末端。如果水太多，末端閥門直接關小一點最有效。")

with c2:
    st.write("### 📍 前層 (W1) 的責任")
    st.write(f"當前壓力回傳值：**{grad_w1:.4f}**")
    st.markdown(f"> **物理含意**：這節水管要動多少，取決於 W2 傳回來的『指令』。如果 W2 關得很死，前面動再多也沒用。這就是**連鎖律**的體現！")

st.info(f"💡 **AI 學習指令**：系統現在會命令 W1 {'調小' if grad_w1 > 0 else '調大'}，且命令 W2 {'調小' if grad_w2 > 0 else '調大'}。")
