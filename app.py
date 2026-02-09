import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 設定網頁標題
st.set_page_config(page_title="ANN 運算邏輯視覺化", layout="wide")
st.title("🧠 AI 的大腦是怎麼轉的？—— ANN 運算與物理類比")

st.markdown("""
### 💡 物理類比：水流系統
想像神經網路是一個**水管系統**：
* **輸入 (Input)**：水源的壓力（例如：讀書的時間）。
* **權重 (Weight)**：水管轉接頭的**鬆緊度**。轉得越鬆，水流越大；轉得越緊，水流越小。
* **偏置 (Bias)**：加壓幫浦，給水流一個基礎的推力。
* **輸出 (Output)**：最後水桶裡裝了多少水（預測的分數）。
""")

# --- 側邊欄控制 ---
st.sidebar.header("🛠️ 手動調整神經元")
w = st.sidebar.slider("調整權重 (Weight)", -2.0, 2.0, 0.5)
b = st.sidebar.slider("調整偏置 (Bias)", -5.0, 5.0, 0.0)
target = st.sidebar.number_input("目標分數 (Target Score)", 0.0, 100.0, 80.0)

# --- 前向傳播計算 ---
x = np.linspace(0, 10, 100)  # 假設讀書時間 0~10 小時
z = w * x + b
prediction = 1 / (1 + np.exp(-z)) * 100  # 使用 Sigmoid 將結果轉為 0~100 分

# --- 視覺化圖表 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 前向傳播：預測結果")
    fig, ax = plt.subplots()
    ax.plot(x, prediction, label="預測曲線", color="#1f77b4", linewidth=3)
    ax.axhline(y=target, color='r', linestyle='--', label="目標分數")
    ax.set_xlabel("讀書時間 (Input)")
    ax.set_ylabel("預測分數 (Output)")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("📉 損失函數：物理上的「位能」")
    # 這裡類比物理上的位能：距離目標越遠，位能（誤差）越高
    w_range = np.linspace(-2, 2, 100)
    # 簡化版損失函數 (L2 Loss)
    loss = (target - (1 / (1 + np.exp(-(w_range * 5 + b))) * 100))**2
    
    fig2, ax2 = plt.subplots()
    ax2.plot(w_range, loss, color="#ff7f0e")
    # 標出當前權重的位置
    current_loss = (target - (1 / (1 + np.exp(-(w * 5 + b))) * 100))**2
    ax2.scatter([w], [current_loss], color='red', s=100, label="目前權重位置")
    ax2.set_xlabel("權重 (水管鬆緊度)")
    ax2.set_ylabel("誤差位能 (Loss)")
    ax2.legend()
    st.pyplot(fig2)

st.info(f"**目前狀態：** 當權重為 {w} 時，誤差位能為 {current_loss:.2f}。")

# --- 背後邏輯解釋 ---
st.divider()
st.header("🔍 背後的運算邏輯")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
    #### 1. 前向傳播 (Forward Pass)
    這就像水從水源流向水桶。
    $$ \text{Output} = \sigma(\text{Input} \cdot W + b) $$
    我們把讀書時間乘以**權重**，加上**偏置**，再透過一個轉換函數，得到最後的分數。
    """)
    
with col_b:
    st.markdown("""
    #### 2. 反向傳播 (Backpropagation)
    當我們發現預測分數跟目標不一樣時，我們會**倒著走回去**。
    在物理上，這就像是**球往低處滾**：
    * 計算**梯度 (Gradient)**：看哪邊比較陡。
    * **更新權重**：往坡度低的方向轉動水管轉接頭。
    """)

st.success("試著滑動左側的權重，看看右圖紅點（你的權重）是如何在『誤差山谷』中移動的！")
