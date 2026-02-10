import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Activation Functions Definition ---
def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z): return np.maximum(0, z)
def gelu(z): return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))

# --- 2. Page Configuration ---
st.set_page_config(layout="wide", page_title="ANN Multi-Layer Lab")
st.title("🌐 Multi-Layer Neural Network: Visual Logic Lab")
st.markdown("""
This lab demonstrates how an **Artificial Neural Network (ANN)** processes signals through multiple layers and neurons. 
Adjust the weights and observe how different **Activation Functions** shape the 'brain's' response.
""")

# --- 3. Sidebar: Architecture & Hyperparameters ---
st.sidebar.header("🏗️ Network Configuration")
st.sidebar.subheader("Layer Settings")
n_hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 3, 2)
neurons_per_layer = st.sidebar.slider("Neurons per Layer (Width)", 2, 8, 4)

st.sidebar.divider()
st.sidebar.subheader("Signal Input")
input_vector = st.sidebar.multiselect(
    "Input Signals (Select up to 4)", 
    ["Feature A", "Feature B", "Feature C", "Feature D"],
    default=["Feature A", "Feature B"]
)
# Convert selected features to a numeric vector
raw_input = np.zeros(neurons_per_layer)
for i in range(len(input_vector)):
    raw_input[i] = st.sidebar.slider(f"Intensity of {input_vector[i]}", -5.0, 5.0, 1.0)

st.sidebar.divider()
st.sidebar.subheader("Decision Logic")
act_choice = st.sidebar.selectbox("Activation Function", ["ReLU", "Sigmoid", "GeLU"])

# --- 4. Core Computation (Forward Propagation) ---
# We store outputs of each layer for visualization
layer_outputs = [raw_input]
current_input = raw_input

# Generate stable random weights for demonstration based on seed
np.random.seed(42)

for i in range(n_hidden_layers):
    # Matrix Weight (Shape: Neurons x Neurons)
    W = np.random.randn(neurons_per_layer, neurons_per_layer) * 0.5
    b = np.random.randn(neurons_per_layer) * 0.1
    
    # Linear Transformation: Z = WX + B
    z = np.dot(W, current_input) + b
    
    # Non-linear Activation
    if act_choice == "ReLU":
        current_input = relu(z)
    elif act_choice == "Sigmoid":
        current_input = sigmoid(z) * 5 # Scale for visualization
    else:
        current_input = gelu(z)
    
    layer_outputs.append(current_input)

# Final Output (Average of last layer)
final_prediction = np.mean(layer_outputs[-1])

# --- 5. Visualization Layout ---
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🕸️ Network Structure & Signal Flow")
    fig_net, ax_net = plt.subplots(figsize=(10, 6))
    
    # Draw neurons and connections
    layer_sizes = [neurons_per_layer] * (n_hidden_layers + 1)
    v_spacing = 1.0 / max(layer_sizes)
    h_spacing = 1.0 / len(layer_sizes)
    
    for l, layer_size in enumerate(layer_sizes):
        for i in range(layer_size):
            # Neuron color based on activation value
            val = layer_outputs[l][i]
            color = plt.cm.viridis(val / 5.0) if val > 0 else 'black'
            circle = plt.Circle((l * h_spacing + 0.1, i * v_spacing + 0.1), 0.03, color=color, ec='white', zorder=4)
            ax_net.add_artist(circle)
            
            # Draw connections to next layer
            if l < len(layer_sizes) - 1:
                for j in range(layer_sizes[l+1]):
                    ax_net.plot([l * h_spacing + 0.1, (l+1) * h_spacing + 0.1], 
                                [i * v_spacing + 0.1, j * v_spacing + 0.1], 
                                color='gray', alpha=0.2, lw=1)
    
    ax_net.set_axis_off()
    st.pyplot(fig_net)
    st.caption("Bright colors = Firing Neurons | Black = Inactive (Dead) Neurons")

with col2:
    st.subheader("🔥 Layer Activity (Heatmap)")
    # Prepare data for Heatmap
    heatmap_data = np.array(layer_outputs).T
    fig_hm, ax_hm = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax_hm)
    ax_hm.set_xticklabels(["Input"] + [f"Hidden {i+1}" for i in range(n_hidden_layers)])
    ax_hm.set_ylabel("Neuron Index")
    st.pyplot(fig_hm)
    st.info(f"**Insight:** Notice how **{act_choice}** turns some neurons to 0.00 (Sparsity).")

# --- 6. Loss & Performance ---
st.divider()
c1, c2, c3 = st.columns(3)
target_val = c1.number_input("Target Value (Ground Truth)", value=3.0)
loss = (final_prediction - target_val)**2

c2.metric("Final Prediction (ŷ)", f"{final_prediction:.4f}")
c3.metric("Loss (Error Rate)", f"{loss:.4f}", delta=f"{final_prediction - target_val:.2f}", delta_color="inverse")

if loss < 0.5:
    st.balloons()
    st.success("Target Reached! The network has 'learned' the pattern.")
