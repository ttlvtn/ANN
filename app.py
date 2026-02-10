import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Activation Functions ---
def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z): return np.maximum(0, z)
def gelu(z): return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))

st.set_page_config(layout="wide")
st.title("🔬 ANN Signal & Loss Lab: The Mapping of W, X, and Loss")

# --- 2. Sidebar: Manual Input Control ---
st.sidebar.header("📥 Data Input Center")
user_input = st.sidebar.number_input("Enter Input Signal (x)", value=2.0, step=0.1)
target_goal = st.sidebar.number_input("Enter Target Goal (y)", value=5.0, step=0.1)

st.sidebar.divider()
st.sidebar.subheader("Adjustable Parameters")
w_current = st.sidebar.slider("Current Weight (w)", -10.0, 10.0, 1.5)
bias = st.sidebar.slider("Bias (b)", -5.0, 5.0, 0.0)
act_choice = st.sidebar.selectbox("Select Activation for Loss Curve", ["ReLU", "Sigmoid", "GeLU"])

# --- 3. Processing Logic ---
z_current = (user_input * w_current) + bias

def get_output(z_val, mode):
    if mode == "Sigmoid": return sigmoid(z_val) * 10
    if mode == "ReLU": return relu(z_val)
    if mode == "GeLU": return gelu(z_val)
    return 0

out_current = get_output(z_current, act_choice)
loss_current = (out_current - target_goal)**2

# --- 4. Loss Surface Calculation (Loss vs Weight) ---
# 我們想看：當 x 固定，改變 w 時，Loss 怎麼變？
w_range = np.linspace(-10, 10, 200)
z_range = (user_input * w_range) + bias
loss_range = [(get_output(zv, act_choice) - target_goal)**2 for zv in z_range]

# --- 5. Visualization ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"📈 Loss Curve: $Loss$ vs $w$ (given $x={user_input}$)")
    fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
    ax_loss.plot(w_range, loss_range, color='black', lw=2, label=f'Loss Landscape ({act_choice})')
    ax_loss.scatter([w_current], [loss_current], color='red', s=100, zorder=5, label='Current AI State')
    
    # 標示最低點（AI 的目標）
    min_loss_w = w_range[np.argmin(loss_range)]
    ax_loss.axvline(min_loss_w, color='green', linestyle='--', alpha=0.5, label='Target Minimum')
    
    ax_loss.set_xlabel("Weight (w)")
    ax_loss.set_ylabel("Loss (Error)")
    ax_loss.legend()
    st.pyplot(fig_loss)
    
    st.info(f"💡 **AI 的任務**：調整 **w** 讓紅點滾到綠色虛線（最低點）。")

with col2:
    st.subheader("📊 Mathematical Breakdown")
    st.write(f"**Current Formula:**")
    st.latex(rf"Loss = (\text{{{act_choice}}}({w_current} \cdot {user_input} + {bias}) - {target_goal})^2")
    
    st.metric("Final Loss", f"{loss_current:.4f}")
    
    # 解釋 x 的影響
    st.write("---")
    st.write("**$x$ (Input) 的角色：**")
    if abs(user_input) > 1:
        st.write(f"現在 $x={user_input}$，這會**放大** Weight 的影響力，讓 Loss 曲線變得更陡峭。")
    else:
        st.write(f"現在 $x={user_input}$ 較小，這會**縮小** Weight 的影響力，讓 Loss 曲線變得平緩。")

# --- 6. 互動總結 ---
st.divider()
st.subheader("📝 觀察重點：$x$ 與 $w$ 的連鎖反應")
st.markdown(f"""
1. **$x$ 決定了「敏感度」**：嘗試把 `Input (x)` 調到 0。你會發現 Loss 變成一條水平線！這代表不管你怎麼調 `w`，AI 都學不到東西。
2. **激活函數決定了「地形」**：
    * 切換到 **Sigmoid**：你會發現兩端變得很平（梯度消失），紅點很難滾動。
    * 切換到 **ReLU**：地形會變得像一個「V」字型的山谷。
3. **目標點位移**：當你改變 `Target Goal`，你會發現整個山谷（Loss Curve）會在水平方向上位移，AI 必須重新尋找新的最優 `w`。
""")
