import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Activation Functions ---
def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z): return np.maximum(0, z)
def gelu(z): return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))

st.set_page_config(layout="wide")
st.title("🌐 Multi-layer Neural Network Explorer")

# --- Sidebar Controls ---
st.sidebar.header("🛠️ Network Hyperparameters")
n_layers = st.sidebar.slider("Number of Hidden Layers", 1, 3, 2)
act_choice = st.sidebar.selectbox("Select Activation Function", ["ReLU", "Sigmoid", "GeLU"])
input_val = st.sidebar.slider("Input Signal Intensity", -10.0, 10.0, 2.0)

# Simulate Weights and Biases for each layer
weights = [st.sidebar.slider(f"Layer {i+1} Weight", -2.0, 2.0, 0.5) for i in range(n_layers)]
biases = [st.sidebar.slider(f"Layer {i+1} Bias", -2.0, 2.0, 0.0) for i in range(n_layers)]

# --- Main Simulation ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Signal Transformation Flow")
    
    current_signal = input_val
    signals = [input_val]
    
    for i in range(n_layers):
        # Linear transform
        z = current_signal * weights[i] + biases[i]
        # Activation
        if act_choice == "ReLU": current_signal = relu(z)
        elif act_choice == "Sigmoid": current_signal = sigmoid(z) * 10 # Scaled for visibility
        else: current_signal = gelu(z)
        signals.append(current_signal)
    
    # Plotting the signal change across layers
    fig, ax = plt.subplots()
    layer_names = ["Input"] + [f"Hidden {i+1}" for i in range(n_layers)]
    ax.plot(layer_names, signals, marker='o', linestyle='-', color='blue', lw=2)
    ax.set_ylabel("Signal Strength")
    ax.set_title(f"How Signal Changes (Activation: {act_choice})")
    for i, txt in enumerate(signals):
        ax.annotate(f"{txt:.2f}", (layer_names[i], signals[i]), textcoords="offset points", xytext=(0,10), ha='center')
    st.pyplot(fig)

with col2:
    st.subheader("📈 Activation Effect: The 'Shape' of Logic")
    z_range = np.linspace(-10, 10, 200)
    
    if act_choice == "ReLU": y_plot = relu(z_range)
    elif act_choice == "Sigmoid": y_plot = sigmoid(z_range) * 10
    else: y_plot = gelu(z_range)
    
    fig2, ax2 = plt.subplots()
    ax2.plot(z_range, y_plot, label=act_choice, color='green', lw=3)
    # Mark the signal state of the LAST layer
    last_z = signals[-2] * weights[-1] + biases[-1]
    ax2.scatter([last_z], [signals[-1]], color='red', s=100, zorder=5, label='Last Layer State')
    ax2.axhline(0, color='black', alpha=0.3)
    ax2.axvline(0, color='black', alpha=0.3)
    ax2.set_xlabel("Internal Sum (z)")
    ax2.set_ylabel("Activated Output")
    ax2.legend()
    st.pyplot(fig2)

# --- Final Loss Calculation ---
st.divider()
target = st.number_input("Desired Target Value", value=8.0)
final_loss = (signals[-1] - target) ** 2
st.metric("Final Model Loss", f"{final_loss:.4f}")
