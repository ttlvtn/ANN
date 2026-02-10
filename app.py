import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Core Mathematical Functions ---
def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z): return np.maximum(0, z)
def gelu(z): return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))

# Function to calculate gradient (derivative) to show why Sigmoid "dies"
def get_gradient(z, func_name):
    if func_name == "Sigmoid":
        s = sigmoid(z)
        return s * (1 - s)
    elif func_name == "ReLU":
        return 1.0 if z > 0 else 0.0
    elif func_name == "GeLU":
        # Simplified GeLU gradient for visualization
        return 0.5 * (1 + np.tanh(0.797 * (z + 0.044 * z**3)))
    return 0

# --- 2. Page Setup ---
st.set_page_config(layout="wide", page_title="ANN Logic Lab")
st.title("🔬 ANN Deep Dive: 2x2 Multi-Layer Logic")
st.markdown("This lab visualizes how signals flow through a **2-Layer / 2-Neuron** network and why **Activation Functions** matter.")

# --- 3. Sidebar: Weight & Bias Control ---
st.sidebar.header("🕹️ Parameter Control")
input_x = st.sidebar.slider("Input Signal (x)", -10.0, 10.0, 4.0)
act_choice = st.sidebar.radio("Switch Activation Function", ["Sigmoid", "ReLU", "GeLU"])

st.sidebar.divider()
st.sidebar.subheader("Layer 1 Weights (W1)")
w11 = st.sidebar.slider("w1_1 (to Neuron 1)", -2.0, 2.0, 1.2)
w12 = st.sidebar.slider("w1_2 (to Neuron 2)", -2.0, 2.0, -0.8)

st.sidebar.subheader("Layer 2 Weights (W2)")
w21 = st.sidebar.slider("w2_1 (from N1)", -2.0, 2.0, 1.0)
w22 = st.sidebar.slider("w2_2 (from N2)", -2.0, 2.0, 0.5)

# --- 4. Computation Flow ---
# Layer 1
z1_1, z1_2 = input_x * w11, input_x * w12
if act_choice == "Sigmoid": a1_1, a1_2 = sigmoid(z1_1), sigmoid(z1_2)
elif act_choice == "ReLU": a1_1, a1_2 = relu(z1_1), relu(z1_2)
else: a1_1, a1_2 = gelu(z1_1), gelu(z1_2)

# Layer 2
z2_1, z2_2 = a1_1 * w21, a1_2 * w22
if act_choice == "Sigmoid": a2_1, a2_2 = sigmoid(z2_1), sigmoid(z2_2)
elif act_choice == "ReLU": a2_1, a2_2 = relu(z2_1), relu(z2_2)
else: a2_1, a2_2 = gelu(z2_1), gelu(z2_2)

# Calculate Gradients (Learning Signal)
grad1 = get_gradient(z1_1, act_choice)

# --- 5. Visualization ---
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🕸️ Network Topology")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Node Coordinates
    pos = {'IN': (0.1, 0.5), 'N1_1': (0.4, 0.75), 'N1_2': (0.4, 0.25), 'N2_1': (0.8, 0.75), 'N2_2': (0.8, 0.25)}
    
    # Draw Weights (Lines)
    def draw_edge(p1, p2, w, label):
        color = 'blue' if w > 0 else 'red'
        ax.annotate("", xy=pos[p2], xytext=pos[p1], arrowprops=dict(arrowstyle="->", lw=abs(w)*4+1, color=color, alpha=0.6))
        ax.text((pos[p1][0]+pos[p2][0])/2, (pos[p1][1]+pos[p2][1])/2 + 0.05, label, fontsize=9, fontweight='bold')

    draw_edge('IN', 'N1_1', w11, f"w={w11}")
    draw_edge('IN', 'N1_2', w12, f"w={w12}")
    draw_edge('N1_1', 'N2_1', w21, f"w={w21}")
    draw_edge('N1_2', 'N2_2', w22, f"w={w22}")

    # Draw Neurons (Circles)
    for name, p in pos.items():
        val = a1_1 if name == 'N1_1' else (a1_2 if name == 'N1_2' else (a2_1 if name == 'N2_1' else (a2_2 if name == 'N2_2' else input_x)))
        color = 'yellow' if val > 0.5 else 'gray'
        circle = plt.Circle(p, 0.06, color=color, ec='black', zorder=5)
        ax.add_artist(circle)
        ax.text(p[0], p[1]-0.15, f"{name}\n(val={val:.2f})", ha='center', fontsize=9)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    st.pyplot(fig)

with col2:
    st.subheader("📉 Signal Analysis")
    
    # Metric Display
    st.write(f"**Current Activation: {act_choice}**")
    st.metric("Signal Strength at N1_1", f"{a1_1:.4f}")
    
    # Gradient Visualization (The "Why" part)
    st.write("---")
    st.write("**Learning Potential (Gradient)**")
    st.progress(min(float(grad1), 1.0))
    st.caption(f"Gradient Value: {grad1:.4f}")
    
    if act_choice == "Sigmoid" and (z1_1 > 4 or z1_1 < -4):
        st.warning("⚠️ **Gradient Vanishing!** The signal is too flat. AI stops learning.")
    elif act_choice == "ReLU" and z1_1 <= 0:
        st.error("💀 **Dead ReLU!** The neuron is inactive. Signal is blocked.")
    else:
        st.success("✅ **Healthy Signal!** Information is flowing perfectly.")

# --- 6. Activation Curve Comparison ---
st.divider()
st.subheader("📈 Activation Curve Spotlight")
z_axis = np.linspace(-10, 10, 200)
if act_choice == "Sigmoid": y_axis = sigmoid(z_axis)
elif act_choice == "ReLU": y_axis = relu(z_axis)
else: y_axis = gelu(z_axis)

fig_curve, ax_curve = plt.subplots(figsize=(12, 3))
ax_curve.plot(z_axis, y_axis, color='navy', lw=2)
ax_curve.scatter([z1_1], [a1_1], color='red', s=100, label="N1_1 Position")
ax_curve.axvline(0, color='black', lw=1, alpha=0.3)
ax_curve.set_title(f"Visualizing {act_choice} Response")
ax_curve.legend()
st.pyplot(fig_curve)
