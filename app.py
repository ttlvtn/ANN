import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Functions ---
def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z): return np.maximum(0, z)
def gelu(z): return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))

# --- 2. Page Config ---
st.set_page_config(layout="wide", page_title="ANN Loss Explorer")
st.title("📉 Neural Network Optimization: The Loss Journey")
st.markdown("How does **Activation** affect the **Loss**? Adjust weights to find the 'Minimum Error' valley.")

# --- 3. Sidebar: Simulation Setup ---
st.sidebar.header("🛠️ Simulation Setup")
input_x = st.sidebar.slider("Input (x)", -5.0, 5.0, 2.0)
target_y = st.sidebar.slider("Target Goal (y)", -5.0, 5.0, 3.0)
act_choice = st.sidebar.selectbox("Activation Function", ["ReLU", "Sigmoid", "GeLU"])

st.sidebar.divider()
st.sidebar.subheader("Adjust Weights (W)")
# We focus on one key weight to visualize the loss curve
current_w = st.sidebar.slider("Current Weight (w)", -5.0, 5.0, 0.5)
bias = st.sidebar.slider("Bias (b)", -2.0, 2.0, 0.0)

# --- 4. Logic & Loss Calculation ---
def predict(x, w, b, func):
    z = x * w + b
    if func == "ReLU": return relu(z)
    if func == "Sigmoid": return sigmoid(z) * 5 # Scaled for MSE comparison
    if func == "GeLU": return gelu(z)
    return 0

# Calculate current state
y_hat = predict(input_x, current_w, bias, act_choice)
current_loss = (y_hat - target_y) ** 2

# Calculate Loss Curve over a range of Weights
w_range = np.linspace(-5, 5, 100)
loss_range = []
for w_val in w_range:
    pred = predict(input_x, w_val, bias, act_choice)
    loss_range.append((pred - target_y) ** 2)

# --- 5. Visualization Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🕸️ 2x2 Network Signal Flow")
    # Simplify visual for 2x2 architecture showing one active path
    fig_net, ax_net = plt.subplots(figsize=(6, 5))
    nodes = {'IN': (0.1, 0.5), 'H1': (0.5, 0.7), 'H2': (0.5, 0.3), 'OUT': (0.9, 0.5)}
    
    # Draw connections
    for n in ['H1', 'H2']:
        lw = abs(current_w) * 2 + 1
        ax_net.annotate("", xy=nodes[n], xytext=nodes['IN'], arrowprops=dict(arrowstyle="->", lw=lw, color="blue" if current_w > 0 else "red"))
        ax_net.annotate("", xy=nodes['OUT'], xytext=nodes[n], arrowprops=dict(arrowstyle="->", lw=2, color="gray", alpha=0.5))
    
    for k, v in nodes.items():
        ax_net.add_artist(plt.Circle(v, 0.06, color='black', zorder=5))
        ax_net.text(v[0], v[1]-0.15, k, ha='center', fontweight='bold')
    
    ax_net.set_xlim(0, 1); ax_net.set_ylim(0, 1); ax_net.axis('off')
    st.pyplot(fig_net)
    
    st.metric("Final Prediction (y_hat)", f"{y_hat:.2f}")

with col2:
    st.subheader("🌋 Loss Landscape (Error Valley)")
    fig_loss, ax_loss = plt.subplots(figsize=(6, 5))
    ax_loss.plot(w_range, loss_range, color='black', lw=2, label="Loss Curve")
    ax_loss.scatter([current_w], [current_loss], color='red', s=150, zorder=5, label="Your AI State")
    
    ax_loss.set_xlabel("Weight (w)")
    ax_loss.set_ylabel("Loss (Error)")
    ax_loss.set_title(f"Loss Profile using {act_choice}")
    ax_loss.grid(alpha=0.2)
    ax_loss.legend()
    st.pyplot(fig_loss)
    
    st.metric("Current Loss", f"{current_loss:.4f}", delta=f"{current_loss:.4f}", delta_color="inverse")

# --- 6. Explaining the Difference ---
st.divider()
st.subheader("💡 Why the Activation Function changes the 'Map'?")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("**Sigmoid: The Steep Cliff**")
    st.write("Notice how the loss curve becomes very flat at the edges. This is why learning is slow—the AI gets stuck on the 'Plateau'.")


[Image of Sigmoid function and its derivative]


with info_col2:
    st.markdown("**ReLU: The Sharp Valley**")
    st.write("The loss curve is often a 'V' shape. It's easy to slide down, but if the weight goes negative, the signal might 'Die' (Loss becomes flat).")


with info_col3:
    st.markdown("**GeLU: The Smooth Path**")
    st.write("The most advanced choice. It combines the speed of ReLU with a smoother valley, helping the AI find the bottom without jumping out.")
