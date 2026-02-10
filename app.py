import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Activation & Derivatives (Gradients) ---
def sigmoid(z): return 1 / (1 + np.exp(-z))
def d_sigmoid(z): s = sigmoid(z); return s * (1 - s)

def relu(z): return np.maximum(0, z)
def d_relu(z): return 1.0 if z > 0 else 0.0

def gelu(z): return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))
def d_gelu(z): return 0.5 * (1 + np.tanh(0.797 * (z + 0.044 * z**3))) # Approximate

st.set_page_config(layout="wide")
st.title("🚀 Why Activation Matters: The Gradient Perspective")

# --- 2. Sidebar Control ---
st.sidebar.header("🕹️ Parameters")
input_x = st.sidebar.slider("Input Signal (x)", -10.0, 10.0, 4.0)
target_y = st.sidebar.slider("Target (y)", 0.0, 10.0, 5.0)
act_choice = st.sidebar.radio("Activation Function", ["Sigmoid", "ReLU", "GeLU"])

# Current weight for interactive manual iteration
w = st.sidebar.slider("Current Weight (w)", -5.0, 5.0, 0.5)

# --- 3. Forward & Backward Logic ---
z = input_x * w
if act_choice == "Sigmoid": 
    y_hat = sigmoid(z) * 10 # Scale up for demo
    grad_act = d_sigmoid(z)
elif act_choice == "ReLU": 
    y_hat = relu(z)
    grad_act = d_relu(z)
else: 
    y_hat = gelu(z)
    grad_act = d_gelu(z)

loss = (y_hat - target_y)**2
# Total Gradient: dLoss/dw = dLoss/dy * dy/dz * dz/dw
grad_loss_y = 2 * (y_hat - target_y)
total_grad = grad_loss_y * grad_act * input_x

# --- 4. Visualization ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌋 Loss Landscape & Slope")
    w_range = np.linspace(-5, 5, 100)
    # Re-calculate loss curve based on current settings
    def get_l(w_val):
        z_v = input_x * w_val
        if act_choice == "Sigmoid": y = sigmoid(z_v) * 10
        elif act_choice == "ReLU": y = relu(z_v)
        else: y = gelu(z_v)
        return (y - target_y)**2
    
    losses = [get_l(wv) for wv in w_range]
    
    fig, ax = plt.subplots()
    ax.plot(w_range, losses, 'k-', alpha=0.6, label="Loss Path")
    ax.scatter([w], [loss], color='red', s=100, zorder=5, label="Current Weight")
    
    # Draw tangent line (Gradient)
    slope_x = np.array([w - 0.5, w + 0.5])
    slope_y = loss + total_grad * (slope_x - w)
    ax.plot(slope_x, slope_y, 'r--', label="Gradient (Slope)")
    
    ax.set_xlabel("Weight (w)")
    ax.set_ylabel("Loss")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("⚡ Training Vitality")
    st.metric("Current Loss", f"{loss:.4f}")
    
    # 這裡是最重要的視覺化：梯度的大小決定了迭代的速度
    grad_magnitude = abs(total_grad)
    st.write("**Gradient Magnitude (Learning Speed):**")
    st.progress(min(grad_magnitude / 20.0, 1.0))
    
    if grad_magnitude < 0.01:
        st.error("⚠️ **Signal Dead (Gradient Vanishing)!** The AI cannot learn from here.")
    elif grad_magnitude < 0.5:
        st.warning("🐢 **Learning Slow...** The curve is too flat.")
    else:
        st.success("🏎️ **Fast Learning!** The gradient is strong.")

    st.write(f"The gradient is **{total_grad:.4f}**. In the next iteration, we update weight by: $w = w - \\eta \\times {total_grad:.2f}$")
