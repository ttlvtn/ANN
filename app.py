import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 頁面配置
st.set_page_config(page_title="ANN 運算邏輯實驗室", layout="wide")

st.title("🧠 多層 AI 怎麼『比對』誤差？")
st.subheader("用「水管壓力」理解反向傳播 (Backpropagation)")

# --- 物理含意說明 ---
st.markdown("""
> **給高中生的物理小筆記：**
> * **前向傳播**：水從水源流向出口（預測結果）。
> * **權重 (Weight)**：水管的粗細。
> * **誤差 (Loss)**：出口水量與目標的差距。
> * **反向傳播**：水量不對時，產生的「反向壓力」，用來告訴每一段水管要調粗還是調細。
""")

# --- 側邊欄控制 ---
st.sidebar.header("🛠️ 調整水管 (權重)")
w1 = st.sidebar.slider("第一段水管粗細 (W1)", 0.0, 2.0, 0.8)
w2 = st.sidebar.slider("第二段水管粗細 (W2)", 0.0, 2.0, 1.2)
target = st.sidebar.number_input("🎯 目標出水量 (正確答案)", 0.0, 1.0, 0.9)

# --- 核心運算 ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 運算過程
x_input = 1.0
h_hidden = sigmoid(x_input * w1)
y_output = sigmoid(h_hidden * w2)
loss = 0.5 * (y_output - target)**2

# --- 視覺化：使用 Plotly 確保無亂碼 ---
fig = go.Figure()

# 繪製節點 (輸入 -> 隱藏 -> 輸出)
node_x = [0, 1, 2]
node_y = [0, 0, 0]
fig.add_trace(go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    marker=dict(size=50, color=['#636EFA', '#EF553B', '#00CC96']),
    text=["輸入 (努力)", "隱藏層 (轉換)", "輸出 (成果)"],
    textposition="bottom center"
))

# 繪製連線並根據權重改變粗細
fig.add_annotation(x=0.5, y=0.1, text=f"W1 壓力傳導: {w1}", showarrow=False)
fig.add_annotation(x=1.5, y=0.1, text=f"W2 壓力傳導: {w2}", showarrow=False)

fig.update_layout(
    title="神經網路水流示意圖",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False, range=[-0.5, 0.5]),
    height=300,
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# --- 比對邏輯視覺化 ---
col1, col2 = st.columns(2)

with col1:
    st.metric("目前最終產出", f"{y_output:.4f}")
    st.write("### 📊 誤差比對區")
    # 建立一個動態條形圖顯示差距
    chart_data = pd.DataFrame({
        "類別": ["目標", "目前預測"],
        "數值": [target, y_output]
    })
    st.bar_chart(chart_data, x="類別", y="數值", color="#FF4B4B")

with col2:
    # 計算連鎖律梯度
    error = y_output - target
    grad_w2 = error * (y_output * (1 - y_output)) * h_hidden
    grad_w1 = error * (y_output * (1 - y_output)) * w2 * (h_hidden * (1 - h_hidden)) * x_input
    
    st.write("### 🧬 反向壓力 (梯度) 計算")
    st.write(f"W2 的修正壓力：`{grad_w2:.4f}`")
    st.write(f"W1 的修正壓力：`{grad_w1:.4f}`")
    
    if abs(error) < 0.01:
        st.success("✅ 完美比對！水管粗細已調整至最佳狀態。")
    else:
        st.warning(f"⚠️ 誤差仍有 {abs(error):.4f}，系統正試圖調整閥門...")

st.divider()
st.markdown("""
### 🏗️ 多層比對的關鍵：連鎖律 (Chain Rule)
在多層網路中，我們無法直接比對第一層的權重。我們是：
1. **第一步**：比對「輸出」與「目標」，算出最後一關 (W2) 的責任。
2. **第二步**：把這個責任乘上 W2 的大小，**回傳**給前一關 (W1)。
3. **物理意義**：這就像如果最後流不出水，我們會先檢查最後一個水龍頭，再檢查前面的總開關。
""")
