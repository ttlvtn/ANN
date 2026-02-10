import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define Activation Functions
def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z): return np.maximum(0, z)
def gelu(z): return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))

st.set_page_config(layout="wide")
st.title("🧠 Neural Network Interactive Lab")

# --- Sidebar: Control Room ---
st.sidebar.header("🕹️ Control Panel")
input_x = st.sidebar.slider("Input Signal (x)", 0.0, 10.0, 5.0)
weight_w = st.sidebar.slider("Weight (w)", -2.0, 2.0, 0.5)
bias_b = st.sidebar.slider("Bias (b)", -5.0, 5.0, 0.0)
target_y = st.sidebar.number_input("Target Value (Ground Truth)", value=5.0)
act_func = st.sidebar.selectbox("Activation Function", ["ReLU", "Sigmoid", "GeLU"])

# --- Core Calculation ---
z = (input_x * weight_w) + bias_b

if act_func == "ReLU":
    output_y = relu(z)
elif act_func == "Sigmoid":
    output_y = sigmoid(z) * 10 # Scaled for visibility
else:
    output_y = gelu(z)

loss = (output_y - target_y) ** 2

# --- Layout: Visualization ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🕸️ ANN Architecture")
    # Using a simple plot to visualize weights
    fig_net, ax_net = plt.subplots(figsize=(5, 4))
    ax_net.add_patch(plt.Circle((0.2, 0.5), 0.1, color='skyblue', label='Input'))
    ax_net.add_patch(plt.Circle((0.8, 0.5), 0.1, color='orange', label='Output'))
    
    # Weight Line: thickness depends on weight magnitude
    line_width = abs(weight_w) * 5 + 1
    color = 'green' if weight_w > 0 else 'red'
    ax_net.annotate("", xy=(0.7, 0.5), xytext=(0.3, 0.5),
                arrowprops=dict(arrowstyle="->", lw=line_width, color=color))
    
    ax_net.text(0.1, 0.65, f"Input: {input_x}", fontsize=12)
    ax_net.text(0.45, 0.55, f"W: {weight_w}", fontsize=12, color=color)
    ax_net.text(0.7, 0.65, f"Output: {output_y:.2f}", fontsize=12)
    
    ax_net.set_xlim(0, 1)
    ax_net.set_ylim(0, 1)
    ax_net.axis('off')
    st.pyplot(fig_net)
    st.write(f"**Calculation:** $z = {input_x} \\times {weight_w} + {bias_b} = {z:.2f}$")

with col2:
    st.subheader("📉 Performance Metrics")
    st.metric(label="Current Prediction (y_hat)", value=f"{output_y:.2f}", delta=f"{output_y - target_y:.2f}")
    st.metric(label="Loss (Mean Squared Error)", value=f"{loss:.4f}", delta_color="inverse")
    
    # Progress bar for Loss
    st.write("Loss Bar (Goal: 0)")
    st.progress(min(float(loss/25), 1.0))

    if loss < 0.1:
        st.balloons()
        st.success("Perfectly Trained!")

# --- Activation Curve ---
st.divider()
st.subheader(f"Function Visualization: {act_func}")
z_range = np.linspace(-10, 10, 100)
if act_func == "ReLU": y_range = relu(z_range)
elif act_func == "Sigmoid": y_range = sigmoid(z_range) * 10
else: y_range = gelu(z_range)

fig_act, ax_act = plt.subplots(figsize=(10, 3))
ax_act.plot(z_range, y_range, color='gray', alpha=0.5)
ax_act.scatter([z], [output_y], color='red', s=100, label='Current State')
ax_act.set_xlabel("Internal Signal (z)")
ax_act.set_ylabel("Activated Output")
ax_act.legend()
st.pyplot(fig_act)
