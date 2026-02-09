import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 解決中文顯示問題 (針對 Windows/Mac 常用字體)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="多層 ANN 視覺化", layout="wide")

st.title("🌊 為什麼神經網路可以『深』？—— 多層水管的比對邏輯")

# --- 物理含意導讀 ---
with st.expander("📖 點開看：給高中生的物理類比"):
    st.write("""
    1. **前向傳播 = 水流方向**：水從第 1 層流到第 N 層，每一層的閥門（權重）都會影響最後流出的水量。
    2. **損失函數 = 重力位能**：最後的水量與目標不符時，這份「誤差」就像位能，會產生一股推力。
    3. **反向傳播 = 壓力回溯**：當最後一關出問題，壓力會「由後往前」推回每一層，告訴前面的閥門：『嘿！你剛才開太大了！』
    4. **連鎖律 = 槓桿原理**：後面動一點，前面可能要動很多。連鎖律就是計算這個『聯動比例』。
    """)

# --- 側邊欄：多層控制 ---
st.sidebar.header("🛠️ 調整多層閥門 (Weights)")
st.sidebar.subheader("第一層 (基礎理解力)")
w1 = st.sidebar.slider("閥門 W1 (對細節的吸收)", 0.0, 2.0, 0.8)
st.sidebar.subheader("第二層 (邏輯整合力)")
w2 = st.sidebar.slider("閥門 W2 (將細節轉為觀念)", 0.0, 2.0, 1.2)

target_y = st.sidebar.number_input("🎯 目標學習成果 (0~1.0)", 0.0, 1.0, 0.9)

# --- 模擬多層運算 ---
# 假設輸入 x = 1.0 (一個單位的努力)
x = 1.0
hidden_layer = np.tanh(x * w1)  # 第一層輸出
final_output = np.tanh(hidden_layer * w2)  # 第二層輸出 (最終結果)

# --- 視覺化圖表 ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💧 多層水管示意圖")
    # 畫出簡單的神經網路架構圖
    fig, ax = plt.subplots(figsize=(8, 4))
    nodes = [0, 1, 2] # Input, Hidden, Output
    y_pos = [0, 0, 0]
    
    # 節點
    ax.scatter(nodes, y_pos, s=1000, c=['#87CEEB', '#4682B4', '#1E90FF'], zorder=3)
    # 連線與權重標註
    ax.annotate('', xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=w1*5))
    ax.annotate('', xy=(2, 0), xytext=(1, 0), arrowprops=dict(arrowstyle="->", lw=w2*5))
    
    ax.text(0, 0.1, "輸入 (努力)", ha='center')
    ax.text(1, 0.1, f"中間層\n訊號:{hidden_layer:.2f}", ha='center')
    ax.text(2, 0.1, f"輸出 (成果)\n{final_output:.2f}", ha='center')
    ax.text(0.5, -0.1, f"權重 W1: {w1}", ha='center', color='red')
    ax.text(1.5, -0.1, f"權重 W2: {w2}", ha='center', color='red')
    
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    st.pyplot(fig)

with col2:
    st.subheader("🎯 比對結果")
    loss = 0.5 * (final_output - target_y)**2
    st.metric("目前產出", f"{final_output:.2f}")
    st.metric("誤差位能 (Loss)", f"{loss:.4f}", delta=f"{final_output - target_y:.2f}", delta_color="inverse")

# --- 反向傳播的視覺化解釋 ---
st.divider()
st.subheader("🧬 誤差是怎麼『比對』回去的？")

# 計算梯度 (簡化版：使用 tanh 的導數 1-tanh^2)
grad_w2 = (final_output - target_y) * (1 - final_output**2) * hidden_layer
grad_w1 = (final_output - target_y) * (1 - final_output**2) * w2 * (1 - hidden_layer**2) * x

c1, c2 = st.columns(2)
with c1:
    st.info(f"**後層比對 (W2 的責任)：**\n\n直接看輸出差多少。梯度 = {grad_w2:.4f}")
    st.write("物理意義：這節水管離出口最近，修正最直接。")

with c2:
    st.info(f"**前層比對 (W1 的責任)：**\n\n透過 W2 傳回來的壓力。梯度 = {grad_w1:.4f}")
    st.write("物理意義：這節水管要修正，得考慮後面 W2 是開還是關。")

st.warning(f"💡 **下一步動作：** 為了降低誤差，W1 應該 {'調大' if grad_w1 < 0 else '調小'}，W2 應該 {'調大' if grad_w2 < 0 else '調小'}。")
