import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Activation Functions & Their Logic ---
def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z): return np.maximum(0, z)
def gelu(z): return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))

st.set_page_config(layout="wide")
st.title("🔬 ANN Signal & Loss Lab (Interactive)")

# --- 2. Sidebar: Manual Input Control ---
st.sidebar.header("📥 Data Input Center")
# 讓學生可以手動輸入任何數字
user_input = st.sidebar.number_input("Enter Input Signal (x)", value=2.0, step=0.1)
target_goal = st.sidebar.number_input("Enter Target Goal (y)", value=5.0, step=0.1)

st.sidebar.divider()
st.sidebar.subheader("Adjustable Parameters")
w = st.sidebar.slider("Weight (w)", -5.0, 5.0, 1.5)
b = st.sidebar.slider("Bias (b)", -5.0, 5.0, 0.0)

# --- 3. Processing Logic ---
z = (user_input * w) + b

# 計算三種函數的輸出
out_sigmoid = sigmoid(z) * 10  # 縮放以便與目標比較
out_relu = relu(z)
out_gelu = gelu(z)

# 計算 Loss (MSE)
loss_s = (out_sigmoid - target_goal)**2
loss_r = (out_relu - target_goal)**2
loss_g = (out_gelu - target_goal)**2

# --- 4. Visualization Layout ---
# 第一列：數值儀表板
c1, c2, c3 = st.columns(3)
c1.metric("Internal Sum (z)", f"{z:.2f}")
c2.metric("Target Goal", f"{target_goal:.2f}")
c3.info(f"Activation Choice Affects the Output Shape")

# 第二列：三大激活函數的對比呈現
st.divider()
st.subheader("⚡ Activation Comparison: How they handle your input")

col_s, col_r, col_g = st.columns(3)

with col_s:
    st.write("### Sigmoid (The S-Curve)")
    st.write(f"Output: **{out_sigmoid:.4f}**")
    st.error(f"Loss: {loss_s:.4f}")
    # 畫小圖
    fig_s, ax_s = plt.subplots(figsize=(4,3))
    zr = np.linspace(-10, 10, 100)
    ax_s.plot(zr, sigmoid(zr)*10, color='blue')
    ax_s.scatter([z], [out_sigmoid], color='red')
    ax_s.set_title("Soft & Saturated")
    st.pyplot(fig_s)

with col_r:
    st.write("### ReLU (The Hard Filter)")
    st.write(f"Output: **{out_relu:.4f}**")
    st.error(f"Loss: {loss_r:.4f}")
    fig_r, ax_r = plt.subplots(figsize=(4,3))
    ax_r.plot(zr, relu(zr), color='orange')
    ax_r.scatter([z], [out_relu], color='red')
    ax_r.set_title("Direct & Fast")
    st.pyplot(fig_r)

with col_g:
    st.write("### GeLU (The Smooth Gate)")
    st.write(f"Output: **{out_gelu:.4f}**")
    st.error(f"Loss: {loss_g:.4f}")
    fig_g, ax_g = plt.subplots(figsize=(4,3))
    ax_g.plot(zr, gelu(zr), color='green')
    ax_g.scatter([z], [out_gelu], color='red')
    ax_g.set_title("Smart & Balanced")
    st.pyplot(fig_g)

# --- 5. Summary Analysis ---
st.divider()
st.subheader("📝 Lab Report")
best_act = "Sigmoid" if loss_s < loss_r and loss_s < loss_g else ("ReLU" if loss_r < loss_g else "GeLU")
st.success(f"With current input and weight, **{best_act}** provides the lowest Loss!")

st.markdown("""
### 如何引導學生觀察？
1. **輸入負數**：讓學生輸入 `Input = -5`。他們會發現 **ReLU** 變成了 0 (Dead)，而 **GeLU** 在 0 附近還有一點起伏，**Sigmoid** 則是趨近於 0 但沒死掉。
2. **調整權重至極端**：讓 `w` 很大。觀察 **Sigmoid** 的紅點是否卡在最頂端（這就是梯度消失，它再也動不了了）。
3. **對齊目標**：嘗試調整 `w` 和 `b`，看看哪一個函數能讓你最容易達到 **Target Goal**。
""")
