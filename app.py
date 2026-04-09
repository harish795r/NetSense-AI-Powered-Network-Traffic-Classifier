import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="NetSense | Traffic Classifier",
    page_icon="🌐",
    layout="wide",
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0b0f1a;
    color: #e0e6f0;
}

.stApp {
    background: linear-gradient(135deg, #0b0f1a 0%, #0f172a 60%, #0b1120 100%);
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: -1px;
    line-height: 1.1;
}

.hero-sub {
    font-size: 1rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    font-weight: 300;
}

.metric-card {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    backdrop-filter: blur(8px);
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #38bdf8;
}

.metric-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

.badge-0 { background:#1e3a5f; color:#60a5fa; border-radius:20px; padding:3px 12px; font-size:0.8rem; }
.badge-1 { background:#1a3d2b; color:#34d399; border-radius:20px; padding:3px 12px; font-size:0.8rem; }
.badge-2 { background:#3d1a1a; color:#f87171; border-radius:20px; padding:3px 12px; font-size:0.8rem; }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #38bdf8;
    border-bottom: 1px solid rgba(56,189,248,0.2);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.stFileUploader {
    border: 2px dashed rgba(56,189,248,0.3) !important;
    border-radius: 12px !important;
    background: rgba(15,23,42,0.6) !important;
}

div[data-testid="stFileUploadDropzone"] {
    background: rgba(15,23,42,0.5) !important;
    border-radius: 10px;
}

.stButton > button {
    background: linear-gradient(90deg, #0ea5e9, #38bdf8);
    color: #0b0f1a;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
    transition: all 0.2s;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(56,189,248,0.4);
}

.info-box {
    background: rgba(14,165,233,0.08);
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    color: #94a3b8;
}

div[data-testid="stDataFrame"] {
    background: rgba(15,23,42,0.8);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LSTM MODEL DEFINITION (must match train2.py)
# ─────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=3):
        super(LSTMClassifier, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_dim // 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
LABEL_NAMES = {0: "Low Traffic", 1: "Medium Traffic", 2: "High Traffic"}
LABEL_COLORS = {0: "#60a5fa", 1: "#34d399", 2: "#f87171"}

@st.cache_resource
def load_model(model_path):
    model = LSTMClassifier(input_dim=5)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def preprocess(df):
    df = df.copy()
    df['Protocol'] = df['Protocol'].fillna(0)
    df['Length'] = df['Length'].fillna(0)
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])

    df['packet_count'] = 1
    df['avg_size'] = df['Length']
    df['size_variation'] = df['Length'].diff().fillna(0)
    df['packet_rate'] = df['Length'].rolling(2).sum().fillna(0)
    df['rate_change'] = df['packet_rate'].diff().fillna(0)
    return df


def make_sequences(df, timesteps=10):
    features = ['packet_count', 'avg_size', 'size_variation', 'packet_rate', 'rate_change']
    data = df[features].values
    X = []
    for i in range(len(data) - timesteps):
        X.append(data[i:i + timesteps])
    return np.array(X), features


def predict(model, X_seq):
    scaler = MinMaxScaler()
    nsamples, ntimesteps, nfeatures = X_seq.shape
    X_scaled = scaler.fit_transform(X_seq.reshape(-1, nfeatures)).reshape(nsamples, ntimesteps, nfeatures)
    tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(tensor)
        preds = torch.argmax(outputs, dim=1).numpy()
        probs = torch.softmax(outputs, dim=1).numpy()
    return preds, probs


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
col_logo, col_title = st.columns([1, 6])
with col_title:
    st.markdown('<div class="hero-title">🌐 NetSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">LSTM-powered Network Traffic Classifier · TCP/UDP Analysis</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    model_path = st.text_input("Model Path (.pt)", value="tcp_udp_lstm_pytorch.pt")
    timesteps = st.slider("Sequence Timesteps", 5, 20, 10)
    st.markdown("---")
    st.markdown("### 📘 Label Guide")
    st.markdown("🔵 **0** — Low Traffic")
    st.markdown("🟢 **1** — Medium Traffic")
    st.markdown("🔴 **2** — High Traffic")
    st.markdown("---")
    st.markdown("### ℹ️ Features Used")
    for f in ['packet_count', 'avg_size', 'size_variation', 'packet_rate', 'rate_change']:
        st.markdown(f"• `{f}`")


# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2 = st.tabs(["📂  Upload & Predict", "📊  Dashboard"])

# ══════════════════════════════════════════
# TAB 1 — UPLOAD & PREDICT
# ══════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Upload Packet Capture CSV</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">📌 CSV must have columns: <code>Timestamp</code>, <code>Length</code>, <code>Protocol</code> — as exported from Wireshark.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop your CSV here", type=["csv"])

    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded **{len(df_raw):,}** rows · Columns: {list(df_raw.columns)}")

        with st.expander("🔍 Preview Raw Data"):
            st.dataframe(df_raw.head(20), use_container_width=True)

        if st.button("🚀 Run Classification"):
            try:
                model = load_model(model_path)
            except Exception as e:
                st.error(f"❌ Could not load model from `{model_path}`. Make sure the .pt file is in the same folder.\n\n`{e}`")
                st.stop()

            with st.spinner("Processing sequences and running LSTM..."):
                df_proc = preprocess(df_raw)
                X_seq, features = make_sequences(df_proc, timesteps)

                if len(X_seq) == 0:
                    st.warning("⚠️ Not enough rows to form sequences. Need at least timesteps+1 rows.")
                    st.stop()

                preds, probs = predict(model, X_seq)
                time.sleep(0.5)

            # Store in session state for dashboard tab
            st.session_state['preds'] = preds
            st.session_state['probs'] = probs
            st.session_state['df_proc'] = df_proc
            st.session_state['df_raw'] = df_raw

            # ── Traffic Congestion Visual ──
            avg_congestion = np.mean(preds)
            if avg_congestion < 0.6:
                status = "✅ LOW TRAFFIC (CLEAR)"
                color = "#34d399" # Green
                anim_dur = "0.5s" # Fast moving
                window_kb = "64"
                window_pct = "100%"
            elif avg_congestion < 1.6:
                status = "⚠️ MEDIUM TRAFFIC"
                color = "#fbbf24" # Yellow
                anim_dur = "2.0s" # Medium moving
                window_kb = "32"
                window_pct = "50%"
            else:
                status = "🛑 HIGH CONGESTION!"
                color = "#f87171" # Red
                anim_dur = "6.0s" # Slow moving / clogged
                window_kb = "8"
                window_pct = "15%"

            unique_id = color.replace('#','')
            html_string = f"""
<style>
@keyframes pulse-glow-{unique_id} {{
0% {{ box-shadow: 0 0 5px {color}, 0 0 10px {color}; }}
50% {{ box-shadow: 0 0 20px {color}, 0 0 30px {color}; }}
100% {{ box-shadow: 0 0 5px {color}, 0 0 10px {color}; }}
}}
@keyframes flow-data-{unique_id} {{
0% {{ background-position: 200px 0; }}
100% {{ background-position: 0 0; }}
}}
.traffic-pipe-{unique_id} {{
height: 35px;
background: #0b0f1a;
border-radius: 18px;
border: 2px solid rgba(255,255,255,0.05);
position: relative;
overflow: hidden;
margin: 0;
width: {window_pct};
transition: width 1s ease-in-out;
box-shadow: inset 0 0 15px rgba(0,0,0,0.9), 0 0 15px {color};
}}
.data-stream-{unique_id} {{
width: 200%;
height: 100%;
background-image: repeating-linear-gradient(
-45deg,
{color} 0,
{color} 15px,
transparent 15px,
transparent 30px
);
animation: flow-data-{unique_id} {anim_dur} linear infinite;
opacity: 0.85;
}}
.status-badge-{unique_id} {{
animation: pulse-glow-{unique_id} 2s infinite;
color: #111827;
background: {color};
padding: 6px 18px;
border-radius: 20px;
font-family: 'Space Mono', monospace;
font-weight: 700;
font-size: 1.1rem;
display: inline-block;
}}
.window-size-label {{
font-family: 'Space Mono', monospace;
font-size: 1.05rem;
color: #e0e6f0;
margin-top: 20px;
}}
</style>
<div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(56,189,248,0.2); border-radius: 12px; padding: 2rem; text-align: center; margin-top: 2rem;">
<div style="font-family: 'Space Mono', monospace; font-size: 1.2rem; color: #38bdf8; margin-bottom: 15px; letter-spacing: 1px; text-transform: uppercase;">Real-Time Congestion Control</div>
<div class="status-badge-{unique_id}">{status}</div>
<br>
<div class="window-size-label">
Dynamic TCP Window Size (cwnd): <strong style="color: {color}; font-size: 1.2rem;">{window_kb} KB</strong>
</div>
<div style="width: 90%; background: rgba(0,0,0,0.4); height: 50px; border-radius: 25px; margin: 15px auto 20px; border: 1px dashed rgba(255,255,255,0.2); display: flex; align-items: center; justify-content: flex-start; padding: 5px;">
<div class="traffic-pipe-{unique_id}">
<div class="data-stream-{unique_id}"></div>
</div>
</div>
<div style="font-size: 0.9rem; color: #94a3b8; font-family: 'Inter', sans-serif;">When the AI detects high traffic, the simulated TCP connection throttles the window size to prevent packet collision and loss.</div>
</div>
"""
            st.markdown(html_string, unsafe_allow_html=True)


            # ── Metrics Row ──
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            counts = {0: int(np.sum(preds == 0)), 1: int(np.sum(preds == 1)), 2: int(np.sum(preds == 2))}

            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{len(preds):,}</div>
                    <div class="metric-label">Total Sequences</div></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:#60a5fa">{counts[0]:,}</div>
                    <div class="metric-label">Low Traffic</div></div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:#34d399">{counts[1]:,}</div>
                    <div class="metric-label">Medium Traffic</div></div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:#f87171">{counts[2]:,}</div>
                    <div class="metric-label">High Traffic</div></div>""", unsafe_allow_html=True)

            # ── Results Table ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

            result_df = pd.DataFrame({
                'Sequence #': range(1, len(preds) + 1),
                'Predicted Class': preds,
                'Label': [LABEL_NAMES[p] for p in preds],
                'P(Low)': np.round(probs[:, 0], 3),
                'P(Medium)': np.round(probs[:, 1], 3),
                'P(High)': np.round(probs[:, 2], 3),
            })
            st.dataframe(result_df, use_container_width=True, height=350)

            # Download
            csv_out = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results CSV", csv_out, "predictions.csv", "text/csv")

            st.info("🔀 Switch to the **Dashboard** tab to explore charts!")


# ══════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════
with tab2:
    if 'preds' not in st.session_state:
        st.markdown('<div class="info-box">⬅️ Upload a CSV and run classification first to see the dashboard.</div>', unsafe_allow_html=True)
        st.stop()

    preds = st.session_state['preds']
    probs = st.session_state['probs']
    df_proc = st.session_state['df_proc']
    df_raw = st.session_state['df_raw']

    st.markdown('<div class="section-header">TCP Congestion Window (AIMD Simulation)</div>', unsafe_allow_html=True)

    # ── TCP AIMD Simulation Graph ──
    sim_length = min(len(preds), 40) # 40 looks cleaner than 60
    sim_preds = preds[:sim_length]
    
    cwnd_history = []
    events = [] # stores (x, y, label, type)
    cwnd = 1.0
    ssthresh = 32.0 # Standard start
    
    can_annotate = True
    
    for t, p in enumerate(sim_preds):
        cwnd_history.append(cwnd)
        
        if cwnd >= 6.0:
            can_annotate = True
            
        if p == 0:  # CLEAR: Grow window
            if cwnd < ssthresh:
                # Slow Start -> cwnd = cwnd * 2
                cwnd = min(cwnd * 2, ssthresh) if cwnd * 2 <= ssthresh else cwnd + 1 
            else:
                # Congestion Avoidance -> cwnd = cwnd + 1
                cwnd += 1  
        elif p == 1:  # MEDIUM TRAFFIC: 3 Duplicate ACKs
            ssthresh = max(2.0, cwnd // 2)
            if can_annotate and cwnd >= 4.0:
                events.append((t, cwnd_history[-1], f"3 Ack SS thresh = {int(ssthresh)}", '3ack'))
                can_annotate = False
            else:
                events.append((t, cwnd_history[-1], "", 'quiet'))
            cwnd = ssthresh
        elif p == 2:  # HIGH CONGESTION: Timeout
            ssthresh = max(2.0, cwnd // 2)
            if can_annotate and cwnd >= 4.0:
                events.append((t, cwnd_history[-1], f"Time Out SSthresh = {int(ssthresh)}", 'timeout'))
                can_annotate = False
            else:
                events.append((t, cwnd_history[-1], "", 'quiet'))
            cwnd = 1.0
            
    fig_tcp, ax_tcp = plt.subplots(figsize=(10, 4))
    fig_tcp.patch.set_facecolor('#0f172a')
    ax_tcp.set_facecolor('#111827')
    
    x_vals = np.arange(len(cwnd_history))
    ax_tcp.plot(x_vals, cwnd_history, color='#0ea5e9', linewidth=2.5, marker='o', markersize=5, markerfacecolor='#ef4444')
    
    for (x, y, label, ev_type) in events:
        ax_tcp.plot(x, y, marker='o', color='#ef4444', markersize=8)
        
        if label != "":
            y_offset = -6 if ev_type == '3ack' else 6
            x_offset = 1 if ev_type == 'timeout' else 0.5
            ax_tcp.annotate(label, (x, y), xytext=(x+x_offset, y+y_offset), color='#e0e6f0', fontsize=10, fontweight='bold')
            
            ax_tcp.vlines(x, ymin=0, ymax=y, colors='#94a3b8', linestyles='dotted', alpha=0.9, linewidth=1.5)
            ax_tcp.hlines(y, xmin=0, xmax=x, colors='#94a3b8', linestyles='dotted', alpha=0.9, linewidth=1.5)
        
    ax_tcp.set_xlabel("Transmission Round", color='#94a3b8', fontsize=11, fontweight='bold')
    ax_tcp.set_ylabel("Congestion Window Size", color='#94a3b8', fontsize=11, fontweight='bold')
    
    y_max_bound = max(cwnd_history)
    y_max_bound = y_max_bound + 15 if y_max_bound > 0 else 40
    ax_tcp.set_ylim(0, y_max_bound)
    ax_tcp.set_xlim(0, len(cwnd_history))
    
    ax_tcp.set_xticks(np.arange(0, len(cwnd_history)+1, 2))
    ax_tcp.tick_params(colors='#e0e6f0', labelsize=10)
    ax_tcp.spines['bottom'].set_color('#1e293b')
    ax_tcp.spines['left'].set_color('#1e293b')
    ax_tcp.spines['top'].set_visible(False)
    ax_tcp.spines['right'].set_visible(False)
    ax_tcp.grid(color='#1e293b', linestyle='-', linewidth=0.3, alpha=0.5)
    
    st.pyplot(fig_tcp, use_container_width=True)
    plt.close()

    st.markdown('<br><div class="section-header">Traffic Overview</div>', unsafe_allow_html=True)
    
    # ── Row 1: Pie + Bar ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Class Distribution**")
        labels = [LABEL_NAMES[i] for i in range(3)]
        counts = [int(np.sum(preds == i)) for i in range(3)]
        colors = ["#60a5fa", "#34d399", "#f87171"]

        fig1, ax1 = plt.subplots(figsize=(4.5, 4))
        fig1.patch.set_facecolor('#0f172a')
        ax1.set_facecolor('#0f172a')
        wedges, texts, autotexts = ax1.pie(
            counts, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=140,
            textprops={'color': '#e0e6f0', 'fontsize': 10},
            wedgeprops={'edgecolor': '#0b0f1a', 'linewidth': 2}
        )
        for at in autotexts:
            at.set_color('#0b0f1a')
            at.set_fontweight('bold')
        ax1.set_title("Sequence Class Split", color='#94a3b8', fontsize=11)
        st.pyplot(fig1, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("**Prediction Timeline**")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor('#0f172a')
        ax2.set_facecolor('#111827')
        x = np.arange(len(preds))
        for cls, color in zip([0, 1, 2], colors):
            mask = preds == cls
            ax2.scatter(x[mask], preds[mask], c=color, s=6, alpha=0.7, label=LABEL_NAMES[cls])
        ax2.set_xlabel("Sequence Index", color='#94a3b8')
        ax2.set_ylabel("Class", color='#94a3b8')
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Low', 'Medium', 'High'], color='#e0e6f0')
        ax2.tick_params(colors='#64748b')
        ax2.spines['bottom'].set_color('#1e293b')
        ax2.spines['left'].set_color('#1e293b')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.legend(fontsize=8, facecolor='#1e293b', edgecolor='#334155', labelcolor='#e0e6f0')
        ax2.set_title("Predictions Over Time", color='#94a3b8', fontsize=11)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ── Row 2: Confidence + Packet Rate ──
    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Average Confidence per Class**")
        avg_conf = [probs[preds == i, i].mean() if np.sum(preds == i) > 0 else 0 for i in range(3)]
        fig3, ax3 = plt.subplots(figsize=(4.5, 3.5))
        fig3.patch.set_facecolor('#0f172a')
        ax3.set_facecolor('#111827')
        bars = ax3.bar(LABEL_NAMES.values(), avg_conf, color=colors, edgecolor='#0b0f1a', linewidth=1.5)
        for bar, val in zip(bars, avg_conf):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.2f}', ha='center', color='#e0e6f0', fontsize=9, fontweight='bold')
        ax3.set_ylim(0, 1.1)
        ax3.set_ylabel("Avg Confidence", color='#94a3b8')
        ax3.tick_params(colors='#64748b')
        ax3.spines['bottom'].set_color('#1e293b')
        ax3.spines['left'].set_color('#1e293b')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.set_title("Model Confidence by Class", color='#94a3b8', fontsize=11)
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    with col4:
        st.markdown("**Packet Rate Over Time**")
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        fig4.patch.set_facecolor('#0f172a')
        ax4.set_facecolor('#111827')
        rate = df_proc['packet_rate'].values[:500]
        ax4.plot(rate, color='#38bdf8', linewidth=1, alpha=0.9)
        ax4.fill_between(range(len(rate)), rate, alpha=0.15, color='#38bdf8')
        ax4.set_xlabel("Packet Index", color='#94a3b8')
        ax4.set_ylabel("Packet Rate", color='#94a3b8')
        ax4.tick_params(colors='#64748b')
        ax4.spines['bottom'].set_color('#1e293b')
        ax4.spines['left'].set_color('#1e293b')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.set_title("Packet Rate (first 500)", color='#94a3b8', fontsize=11)
        st.pyplot(fig4, use_container_width=True)
        plt.close()

    # ── Probability Heatmap ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Probability Heatmap (first 60 sequences)</div>', unsafe_allow_html=True)

    fig5, ax5 = plt.subplots(figsize=(12, 2.5))
    fig5.patch.set_facecolor('#0f172a')
    ax5.set_facecolor('#0f172a')
    prob_sample = probs[:60].T
    sns.heatmap(prob_sample, ax=ax5, cmap="YlOrRd", cbar=True,
                xticklabels=10, yticklabels=["Low", "Medium", "High"],
                linewidths=0.3, linecolor='#0b0f1a')
    ax5.set_xlabel("Sequence Index", color='#94a3b8')
    ax5.tick_params(colors='#e0e6f0')
    ax5.set_title("Class Probabilities Across Sequences", color='#94a3b8')
    st.pyplot(fig5, use_container_width=True)
    plt.close()

    # ── Packet Length Distribution ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Packet Length Distribution</div>', unsafe_allow_html=True)

    fig6, ax6 = plt.subplots(figsize=(10, 3))
    fig6.patch.set_facecolor('#0f172a')
    ax6.set_facecolor('#111827')
    ax6.hist(df_raw['Length'].dropna(), bins=60, color='#818cf8', edgecolor='#0b0f1a', alpha=0.85)
    ax6.set_xlabel("Packet Length (bytes)", color='#94a3b8')
    ax6.set_ylabel("Count", color='#94a3b8')
    ax6.tick_params(colors='#64748b')
    ax6.spines['bottom'].set_color('#1e293b')
    ax6.spines['left'].set_color('#1e293b')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    st.pyplot(fig6, use_container_width=True)
    plt.close()