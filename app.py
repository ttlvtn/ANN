import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 定義函數
def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z): return np.maximum(0, z)
def gelu(z): return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))

st.title("🧪 Activation VS Loss: Why GeLU Wins?")

act_name = st.selectbox("Choose Activation", ["Sigmoid", "ReLU", "GeLU"])
w = st.slider("Current Weight (w)", -5.0, 5.0, 4.0) # 預設給一個較大的值

# 模擬 Loss 曲線
w_range = np.linspace(-5, 5, 100)
def get_loss(weight):
    # 假設 Input=2, Target=3
    z = 2 * weight
    if act_name == "Sigmoid": y = sigmoid(z) * 5
    elif act_name == "ReLU": y = relu(z)
    else: y = gelu(z)
    return (y - 3)**2

losses = [get_loss(wv) for wv in w_range]
current_loss = get_loss(w)

fig, ax = plt.subplots()
ax.plot(w_range, losses, color='black', label="Loss Path")
ax.scatter([w], [current_loss], color='red', s=100, label="Your AI State")
ax.set_title(f"Loss Landscape using {act_name}")
st.pyplot(fig)

if act_name == "Sigmoid" and abs(w) > 3:
    st.error("⚠️ **Gradient Vanishing!** Notice how flat the curve is. The ball won't roll down!")
elif act_name == "GeLU":
    st.success("✅ **Optimal Curve.** The valley is smooth and easy to navigate.")
