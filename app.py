import streamlit as st
import numpy as np

st.title("🧠 ANN 運作模擬器（高中生入門版）")

# --- Step 1: 輸入 ---
x = st.slider("輸入數據 (例如: 你的讀書時間)", 0.0, 10.0, 5.0)

# --- Step 2: 權重 ---
w = st.sidebar.slider("調整權重 (Weight)", -2.0, 2.0, 0.8)
b = st.sidebar.slider("調整偏差 (Bias)", -5.0, 5.0, 0.0)

# 計算線性結果
z = w * x + b
st.write(f"加權計算後的數值 (z) = {z:.2f}")

# --- Step 3: 激活函數 ---
activation_type = st.radio("選擇激活函數", ["ReLU", "Sigmoid"])
if activation_type == "ReLU":
    output = max(0, z)
else:
    output = 1 / (1 + np.exp(-z))

st.metric("AI 的最終輸出 (y_pred)", f"{output:.2f}")

# --- Step 4: Loss 計算 ---
target = 5.0  # 假設標準答案是 5
loss = (output - target) ** 2

st.subheader("📊 準確度分析 (Loss)")
if loss < 1:
    st.success(f"目前 Loss: {loss:.4f} (預測得很準！)")
else:
    st.error(f"目前 Loss: {loss:.4f} (失誤很大，請回去調整權重！)")

# 繪製 Loss 變化圖
st.write("目標是讓 Loss 歸零，這就是 AI 訓練的目的！")
