import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 定義函數
def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z): return np.maximum(0, z)
def gelu(z): return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))

st.title("🧪 激活函數實驗室：誰是最好的開關？")

# 側邊欄控制
z_val = st.slider("調整神經元的輸入值 (z)", -5.0, 5.0, 0.0)
func_name = st.selectbox("切換函數觀看邏輯", ["Sigmoid", "ReLU", "GeLU"])

# 計算數值
z_range = np.linspace(-5, 5, 200)
if func_name == "Sigmoid":
    y_range = sigmoid(z_range)
    current_y = sigmoid(z_val)
    desc = "就像一個平滑的 S 型閥門，常用於最後一層判斷『是或不是』。"
elif func_name == "ReLU":
    y_range = relu(z_range)
    current_y = relu(z_val)
    desc = "最受歡迎的開關！負數直接歸零，正數直接通過，讓學習變快。"
else:
    y_range = gelu(z_range)
    current_y = gelu(z_val)
    desc = "ChatGPT 的秘密武器！它是機率性的 ReLU，在 0 附近處理得更細緻。"

# 繪圖
fig, ax = plt.subplots()
ax.plot(z_range, y_range, label=func_name, color='blue', lw=2)
ax.scatter([z_val], [current_y], color='red', s=100, zorder=5) # 標示當前點
ax.axhline(0, color='black', lw=1)
ax.axvline(0, color='black', lw=1)
ax.set_title(f"{func_name} 曲線與當前輸出")
ax.grid(alpha=0.3)
st.pyplot(fig)

st.info(f"**物理意義：** {desc}")
st.metric(f"{func_name} 輸出強度", f"{current_y:.4f}")
