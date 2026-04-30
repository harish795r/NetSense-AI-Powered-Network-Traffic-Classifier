import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time
import threading
import collections
import scapy

# Catch scapy import errors if missing gracefully for UI warning
try:
    from scapy.all import sniff, IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="NetSense | Traffic Classifier",
    page_icon="🌐",
    layout="wide",
)

# ─────────────────────────────────────────
# THEME STATE & SIDEBAR TOGGLE
# ─────────────────────────────────────────
if 'theme' not in st.session_state:
    st.session_state.theme = "Dark"

with st.sidebar:
    st.markdown("### 🌓 UI Settings")
    is_light = st.toggle("Enable Light Mode", value=(st.session_state.theme == "Light"))
    st.session_state.theme = "Light" if is_light else "Dark"

# Theme Tokens
if st.session_state.theme == "Dark":
    BG_COLOR = "#0b0f1a"
    APP_BG = "linear-gradient(135deg, #0b0f1a 0%, #0f172a 60%, #0b1120 100%)"
    TEXT_COLOR = "#e0e6f0"
    SUBTEXT_COLOR = "#94a3b8"
    CARD_BG = "rgba(30, 41, 59, 0.8)"
    CARD_BORDER = "rgba(56, 189, 248, 0.2)"
    MODAL_BG = "rgba(15, 23, 42, 0.7)"
    PIPE_BG = "#0b0f1a"
    INPUT_BG = "rgba(15,23,42,0.6)"
    PHASE_BOX_BG = "rgba(30,41,59,0.6)"
    TITLE_COLOR = "#38bdf8"
    BORDER_COLOR = "rgba(56, 189, 248, 0.2)"
    SIDEBAR_BG = "#0f172a"
    PLOT_BG = "#0b0f1a"
    PLOT_CARD_BG = "#111827"
    PLOT_BORDER_COLOR = "#1e293b"
else:
    BG_COLOR = "#f8fafc"
    APP_BG = "linear-gradient(135deg, #f8fafc 0%, #f1f5f9 60%, #e2e8f0 100%)"
    TEXT_COLOR = "#0f172a"
    SUBTEXT_COLOR = "#475569"
    CARD_BG = "rgba(255, 255, 255, 0.9)"
    CARD_BORDER = "rgba(56, 189, 248, 0.3)"
    MODAL_BG = "rgba(248, 250, 252, 0.98)"
    PIPE_BG = "#f1f5f9"
    INPUT_BG = "rgba(255,255,255,0.8)"
    PHASE_BOX_BG = "rgba(241,245,249,0.9)"
    TITLE_COLOR = "#0284c7"
    BORDER_COLOR = "rgba(2, 132, 199, 0.2)"
    SIDEBAR_BG = "#f1f5f9"
    PLOT_BG = "#f8fafc"
    PLOT_CARD_BG = "#f1f5f9"
    PLOT_BORDER_COLOR = "#e2e8f0"

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, .stApp {{
    font-family: 'Inter', sans-serif;
    background-color: {BG_COLOR} !important;
    color: {TEXT_COLOR} !important;
}}

/* Sidebar Theme Override */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {{
    background-color: {SIDEBAR_BG} !important;
}}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, 
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] span {{
    color: {TEXT_COLOR} !important;
}}

/* Sidebar Inputs (Text, Slider) */
[data-testid="stSidebar"] input {{
    background-color: {INPUT_BG} !important;
    color: {TEXT_COLOR} !important;
    border: 1px solid {BORDER_COLOR} !important;
}}

/* Main Area Widget Labels (Radio/Checkbox/Select) */
[data-testid="stWidgetLabel"] p, 
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label div,
div[data-testid="stRadio"] label span {{
    color: {TEXT_COLOR} !important;
    font-weight: 500 !important;
}}

/* Fix for unreadable Radio labels seen in screenshot */
div[data-testid="stRadio"] [role="radiogroup"] label span {{
    color: {TEXT_COLOR} !important;
}}

/* Streamlit Tabs Styling */
button[data-baseweb="tab"] {{
    color: {SUBTEXT_COLOR} !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {TITLE_COLOR} !important;
    border-bottom-color: {TITLE_COLOR} !important;
}}

/* Hide Streamlit Header and Footer */
header {{ display: none !important; }}
footer {{ display: none !important; }}
[data-testid="stHeader"] {{ display: none !important; }}
.stAppDeployButton {{ display: none !important; }}

.stApp {{
    background: {APP_BG};
}}

h1, h2, h3 {{
    font-family: 'Space Mono', monospace !important;
    color: {TEXT_COLOR} !important;
}}

.hero-title {{
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: {TITLE_COLOR};
    letter-spacing: -1px;
    line-height: 1.1;
}}

.hero-sub {{
    font-size: 1rem;
    color: {SUBTEXT_COLOR};
    margin-top: 0.4rem;
    font-weight: 300;
}}

.metric-card {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}}

.metric-value {{
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: {TITLE_COLOR};
}}

.metric-label {{
    font-size: 0.78rem;
    color: {SUBTEXT_COLOR};
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}}

.badge-0 {{ background:#1e3a5f; color:#60a5fa; border-radius:20px; padding:3px 12px; font-size:0.8rem; }}
.badge-1 {{ background:#1a3d2b; color:#34d399; border-radius:20px; padding:3px 12px; font-size:0.8rem; }}
.badge-2 {{ background:#3d1a1a; color:#f87171; border-radius:20px; padding:3px 12px; font-size:0.8rem; }}

.section-header {{
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: {TITLE_COLOR} !important;
    border-bottom: 1px solid {BORDER_COLOR};
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}}

.stFileUploader {{
    border: 2px dashed {BORDER_COLOR} !important;
    border-radius: 12px !important;
    background: {INPUT_BG} !important;
}}

div[data-testid="stFileUploadDropzone"] {{
    background: {INPUT_BG} !important;
    border-radius: 10px;
}}

.stButton > button {{
    background: linear-gradient(90deg, #0ea5e9, #38bdf8);
    color: #0b0f1a;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
    white-space: nowrap;
    transition: all 0.2s;
}}

.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(56,189,248,0.4);
}}

.info-box {{
    background: {PHASE_BOX_BG};
    border-left: 3px solid {TITLE_COLOR};
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    color: {TEXT_COLOR} !important;
}}

div[data-testid="stDataFrame"] {{
    background: {CARD_BG};
    border-radius: 10px;
}}

/* ── MODAL STYLES ── */
.learn-section {{
    background: {MODAL_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}}
.learn-section h3 {{
    color: {TITLE_COLOR} !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.05rem !important;
    margin-top: 0 !important;
    margin-bottom: 0.7rem !important;
    letter-spacing: 0.5px;
}}
.learn-section h4 {{
    color: #7dd3fc !important;
    font-size: 0.92rem !important;
    margin-top: 1rem !important;
    margin-bottom: 0.3rem !important;
}}
.learn-section p, .learn-section li {{
    color: {SUBTEXT_COLOR};
    font-size: 0.88rem;
    line-height: 1.7;
}}
.learn-section strong {{ color: {TEXT_COLOR}; }}
.learn-section code {{
    background: rgba(56,189,248,0.1);
    color: #38bdf8;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.82rem;
}}
.concept-pill {{
    display: inline-block;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    margin: 2px 3px;
}}
.phase-box {{
    background: {PHASE_BOX_BG};
    border-left: 3px solid {TITLE_COLOR};
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0;
}}
.phase-box.green  {{ border-left-color: #34d399; }}
.phase-box.yellow {{ border-left-color: #fbbf24; }}
.phase-box.red    {{ border-left-color: #f87171; }}
.phase-box.purple {{ border-left-color: #a78bfa; }}
.phase-label {{
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    margin-bottom: 3px;
}}
.phase-box.green  .phase-label {{ color: #34d399; }}
.phase-box.yellow .phase-label {{ color: #fbbf24; }}
.phase-box.red    .phase-label {{ color: #f87171; }}
.phase-box.purple .phase-label {{ color: #a78bfa; }}
.ascii-art {{
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: {SUBTEXT_COLOR};
    background: rgba(0,0,0,0.1);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    line-height: 1.5;
    overflow-x: auto;
    white-space: pre;
}}
.step-num {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px; height: 22px;
    background: {TITLE_COLOR};
    color: {BG_COLOR};
    border-radius: 50%;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    flex-shrink: 0;
    margin-right: 8px;
}}
.step-row {{
    display: flex;
    align-items: flex-start;
    margin: 0.5rem 0;
    color: {SUBTEXT_COLOR};
    font-size: 0.87rem;
    line-height: 1.6;
}}
.learn-divider {{ border-color: {BORDER_COLOR}; margin: 0.2rem 0 1rem; }}

/* ── HELP MODAL STYLES ── */
.help-step-card {{
    display: flex;
    gap: 1rem;
    background: {MODAL_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
    align-items: flex-start;
}}
.help-step-num {{
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #0ea5e9, #38bdf8);
    color: #0b0f1a;
    border-radius: 50%;
    font-size: 0.9rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    flex-shrink: 0;
    margin-top: 2px;
}}
.help-step-body {{ flex: 1; }}
.help-step-title {{
    font-family: 'Space Mono', monospace;
    font-size: 0.88rem;
    font-weight: 700;
    color: {TEXT_COLOR};
    margin-bottom: 0.35rem;
}}
.help-step-desc {{
    font-size: 0.84rem;
    color: {SUBTEXT_COLOR};
    line-height: 1.65;
}}
.help-step-desc strong {{ color: {TEXT_COLOR}; }}
.help-step-desc code {{
    background: rgba(56,189,248,0.1);
    color: #38bdf8;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-family: 'Space Mono', monospace;
}}
.help-tip {{
    background: rgba(14,165,233,0.07);
    border: 1px solid {BORDER_COLOR};
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.82rem;
    color: {SUBTEXT_COLOR};
    line-height: 1.6;
}}
.help-tip strong {{ color: #38bdf8; }}
.help-warn {{
    background: rgba(251,191,36,0.07);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.82rem;
    color: {SUBTEXT_COLOR};
    line-height: 1.6;
}}
.help-warn strong {{ color: #fbbf24; }}
.help-section-title {{
    font-family: 'Space Mono', monospace;
    font-size: 0.95rem;
    font-weight: 700;
    color: {TITLE_COLOR};
    border-bottom: 1px solid {BORDER_COLOR};
    padding-bottom: 0.45rem;
    margin: 1.2rem 0 0.8rem;
    letter-spacing: 0.4px;
}}
.help-badge {{
    display: inline-block;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.76rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    margin-right: 6px;
    vertical-align: middle;
}}
.help-badge.blue   {{ background:#1e3a5f; color:#60a5fa; }}
.help-badge.green  {{ background:#1a3d2b; color:#34d399; }}
.help-badge.red    {{ background:#3d1a1a; color:#f87171; }}
.help-badge.yellow {{ background:#3d2f0a; color:#fbbf24; }}
.help-faq {{
    background: {PHASE_BOX_BG};
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
}}
.help-faq-q {{
    font-size: 0.85rem;
    font-weight: 700;
    color: #7dd3fc;
    margin-bottom: 0.3rem;
}}
.help-faq-a {{
    font-size: 0.82rem;
    color: {SUBTEXT_COLOR};
    line-height: 1.6;
}}
.help-faq-a strong {{ color: {TEXT_COLOR}; }}
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


@st.cache_resource
def load_scaler():
    import joblib
    import os
    if os.path.exists("scaler.pkl"):
        return joblib.load("scaler.pkl")
    return MinMaxScaler()

def predict(model, X_seq):
    scaler = load_scaler()
    nsamples, ntimesteps, nfeatures = X_seq.shape
    
    try:
        X_scaled = scaler.transform(X_seq.reshape(-1, nfeatures)).reshape(nsamples, ntimesteps, nfeatures)
    except:
        X_scaled = scaler.fit_transform(X_seq.reshape(-1, nfeatures)).reshape(nsamples, ntimesteps, nfeatures)
        
    tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(tensor)
        preds = torch.argmax(outputs, dim=1).numpy()
        probs = torch.softmax(outputs, dim=1).numpy()
    return preds, probs

from scapy.all import rdpcap

def parse_pcap(file):
    import tempfile
    import os
    
    if hasattr(file, 'getbuffer'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name
        try:
            packets = rdpcap(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        packets = rdpcap(file)

    data = []

    for pkt in packets:
        try:
            if pkt.haslayer("IP"):
                length = len(pkt)

                if pkt.haslayer("TCP"):
                    proto = "TCP"
                elif pkt.haslayer("UDP"):
                    proto = "UDP"
                else:
                    proto = "OTHER"

                data.append({
                    "Timestamp": pkt.time,
                    "Length": length,
                    "Protocol": proto
                })
        except:
            continue

    return pd.DataFrame(data)

import tempfile
import os
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns

def generate_pdf_report(preds, probs, df_raw, window_kb, current_status, low, med, high, total_pkts):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(15, 23, 42) # #0f172a
    pdf.rect(0, 0, 210, 297, "F")
    pdf.set_text_color(224, 230, 240) # #e0e6f0
    
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, "NETSENSE TRAFFIC REPORT", ln=True, align="C")
    pdf.set_font("Helvetica", size=12)
    pdf.ln(5)
    
    # Centered Header text
    pdf.set_font("Helvetica", style="B", size=13)
    pdf.set_text_color(56, 189, 248) # #38bdf8
    pdf.cell(0, 8, "REAL-TIME CONGESTION CONTROL", ln=True, align="C")
    
    # Status Pill
    y_pill = pdf.get_y() + 2
    pill_w = 56
    if preds[-1] == 0:
        pill_color = (34, 197, 94) # Green
    elif preds[-1] == 1:
        pill_color = (250, 204, 21) # Yellow
    else:
        pill_color = (239, 68, 68) # Red
        
    pdf.set_fill_color(*pill_color)
    pdf.set_xy((210 - pill_w) / 2, y_pill)
    pdf.rect((210 - pill_w) / 2, y_pill, pill_w, 8, "F")
    pdf.set_text_color(255, 255, 255) # White text on pill
    pdf.set_font("Helvetica", style="B", size=11)
    pdf.cell(pill_w, 8, str(current_status), align="C")
    pdf.ln(12)
    
    # TCP Window Dynamic Text
    pdf.set_text_color(224, 230, 240)
    pdf.set_font("Helvetica", size=11)
    # manual center
    w1 = pdf.get_string_width("Dynamic TCP Window Size (cwnd): ")
    w2 = pdf.get_string_width(f"{window_kb} KB")
    pdf.set_x((210 - (w1 + w2)) / 2)
    pdf.cell(w1, 6, "Dynamic TCP Window Size (cwnd): ")
    pdf.set_text_color(*pill_color)
    pdf.set_font("Helvetica", style="B", size=11)
    pdf.cell(w2, 6, f"{window_kb} KB", ln=True)
    pdf.ln(8)
    
    # 4 Metric Cards
    card_w = 42
    card_h = 22
    spacing = 4
    start_x = (210 - (4 * card_w + 3 * spacing)) / 2
    y_cards = pdf.get_y()
    
    cards_data = [
        ("LIVE SEQUENCES", str(len(preds)), (56, 189, 248)),
        ("LOW TRAFFIC", str(low), (96, 165, 250)),
        ("MEDIUM TRAFFIC", str(med), (34, 197, 94)),
        ("HIGH TRAFFIC", str(high), (239, 68, 68))
    ]
    
    for i, (title, val, color) in enumerate(cards_data):
        cx = start_x + (card_w + spacing) * i
        # Draw Card Background
        pdf.set_fill_color(17, 24, 39)
        pdf.set_draw_color(30, 41, 59)
        pdf.rect(cx, y_cards, card_w, card_h, "FD")
        
        # Draw Value
        pdf.set_font("Helvetica", style="B", size=18)
        pdf.set_text_color(*color)
        pdf.set_xy(cx, y_cards + 4)
        pdf.cell(card_w, 8, val, align="C")
        
        # Draw Title
        pdf.set_font("Helvetica", size=7)
        pdf.set_text_color(148, 163, 184)
        pdf.set_xy(cx, y_cards + 14)
        pdf.cell(card_w, 5, title, align="C")
        
    pdf.set_y(y_cards + card_h + 8)
    
    tmp_files = []
    import numpy as np
    
    # ── TCP Simulation ──
    sim_length = min(len(preds), 40)
    sim_preds = preds[-sim_length:]
    cwnd_history = []
    events = []
    cwnd, ssthresh = 1.0, 32.0
    can_annotate = True
    for t, p in enumerate(sim_preds):
        cwnd_history.append(cwnd)
        if cwnd >= 6.0: can_annotate = True
        if p == 0:
            cwnd = min(cwnd * 2, ssthresh) if cwnd * 2 <= ssthresh else cwnd + 1
        elif p == 1:
            ssthresh = max(2.0, cwnd // 2)
            if can_annotate and cwnd >= 4.0:
                events.append((t, cwnd_history[-1], f"3 Ack SSthresh = {int(ssthresh)}", '3ack'))
                can_annotate = False
            cwnd = ssthresh
        elif p == 2:
            ssthresh = max(2.0, cwnd // 2)
            if can_annotate and cwnd >= 4.0:
                events.append((t, cwnd_history[-1], f"Time Out SSthresh = {int(ssthresh)}", 'timeout'))
                can_annotate = False
            cwnd = 1.0
            
    fig_tcp, ax_tcp = plt.subplots(figsize=(8, 3.2))
    fig_tcp.patch.set_facecolor('#0f172a')
    ax_tcp.set_facecolor('#111827')
    ax_tcp.plot(np.arange(len(cwnd_history)), cwnd_history, color='#0ea5e9', linewidth=2.5, marker='o', markersize=5, markerfacecolor='#ef4444')
    for (x, y, label, ev_type) in events:
        ax_tcp.plot(x, y, marker='o', color='#ef4444', markersize=8)
        if label != "":
            y_offset = -6 if ev_type == '3ack' else 6
            x_offset = 1 if ev_type == 'timeout' else 0.5
            ax_tcp.annotate(label, (x, y), xytext=(x+x_offset, y+y_offset), color='#e0e6f0', fontsize=10, fontweight='bold')
            ax_tcp.vlines(x, ymin=0, ymax=y, colors='#94a3b8', linestyles='dotted', alpha=0.9, linewidth=1.5)
            ax_tcp.hlines(y, xmin=0, xmax=x, colors='#94a3b8', linestyles='dotted', alpha=0.9, linewidth=1.5)
    ax_tcp.set_title("TCP Congestion Window (AIMD Simulation)", color='#94a3b8', fontsize=12)
    ax_tcp.set_xlabel("Transmission Round (Latest 40)", color='#94a3b8', fontsize=10)
    ax_tcp.set_ylabel("Cwnd Size", color='#94a3b8', fontsize=10)
    ax_tcp.tick_params(colors='#e0e6f0')
    ax_tcp.spines['bottom'].set_color('#1e293b')
    ax_tcp.spines['left'].set_color('#1e293b')
    ax_tcp.spines['top'].set_visible(False)
    ax_tcp.spines['right'].set_visible(False)
    ax_tcp.grid(color='#1e293b', linestyle='-', linewidth=0.3, alpha=0.5)
    
    y_max_bound = max(cwnd_history) if len(cwnd_history) > 0 else 10
    ax_tcp.set_ylim(0, y_max_bound + 15)
    ax_tcp.set_xlim(0, len(cwnd_history))

    pf1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    fig_tcp.tight_layout()
    fig_tcp.savefig(pf1, facecolor=fig_tcp.get_facecolor(), edgecolor='none')
    plt.close(fig_tcp)
    tmp_files.append(pf1)
    
    # ── Pie Chart ──
    recent_preds = preds[-200:]
    counts = [int(np.sum(np.array(recent_preds) == i)) for i in range(3)]
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    fig1.patch.set_facecolor('#0f172a')
    ax1.set_facecolor('#0f172a')
    if sum(counts) > 0:
        wedges, texts, autotexts = ax1.pie(counts, labels=["Low", "Medium", "High"], autopct='%1.1f%%', colors=["#60a5fa", "#34d399", "#f87171"], textprops={'color': '#e0e6f0', 'fontsize': 10}, wedgeprops={'edgecolor': '#0b0f1a', 'linewidth': 2})
        for at in autotexts: at.set_color('#0b0f1a'); at.set_fontweight('bold')
    ax1.set_title("Sequence Class Split", color='#94a3b8')
    pf2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    fig1.savefig(pf2, facecolor=fig1.get_facecolor(), edgecolor='none')
    plt.close(fig1)
    tmp_files.append(pf2)

    # ── Timeline Forecast ──
    trace_preds = preds[-80:]
    x = np.arange(len(trace_preds))
    actual = np.roll(trace_preds, 1)
    actual[0] = trace_preds[0]
    y_pred, y_actual = np.array(trace_preds, dtype=float), np.array(actual, dtype=float)
    
    last_trend = y_pred[-5:]
    future = []
    for _ in range(40):
        next_val = np.clip(round(np.mean(last_trend)), 0, 2)
        future.append(next_val)
        last_trend = np.append(last_trend[1:], next_val)
    y_pred_extended = np.concatenate([y_pred, np.array(future)])
    x_extended = np.arange(len(y_pred_extended))

    def smooth(y, window=7): return np.convolve(y, np.ones(window)/window, mode='same')
    y_pred_smooth, y_actual_smooth = smooth(y_pred_extended) + 0.12, smooth(y_actual) - 0.12
    
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    fig2.patch.set_facecolor('#0f172a')
    ax2.set_facecolor('#111827')
# FIX: match dimensions properly
x_actual = np.arange(len(y_actual_smooth))
x_pred = np.arange(len(y_pred_smooth))

    ax2.plot(
        x_actual,
        y_actual_smooth,
        color="#22c55e",
        linewidth=2.8,
        label="Actual Traffic"
    )
    
    ax2.plot(
        x_pred,
        y_pred_smooth,
        color="#ef4444",
        linewidth=2.8,
        label="Predicted Traffic"
    )
    ax2.axvspan(len(x)-1, len(x_extended), color='#ef4444', alpha=0.08)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Low', 'Medium', 'High'], color='#e0e6f0')
    ax2.set_xlim(0, len(x_extended))
    ax2.set_xlabel("Recent Sequence Index", color='#94a3b8')
    ax2.set_ylabel("Traffic Level", color='#94a3b8')
    ax2.tick_params(colors='#64748b')
    ax2.spines['bottom'].set_color('#1e293b')
    ax2.spines['left'].set_color('#1e293b')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(facecolor='#111827', edgecolor='none', labelcolor='white')
    ax2.set_title("Prediction vs Actual Timeline", color='#94a3b8')
    
    pf3 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    fig2.tight_layout()
    fig2.savefig(pf3, facecolor=fig2.get_facecolor(), edgecolor='none')
    plt.close(fig2)
    tmp_files.append(pf3)
    
    # ── Explicit Exact Y Positioning to prevent FPDF gaps ──
    y_tcp = y_cards + card_h + 12
    pdf.image(pf1, x=10, y=y_tcp, w=190, h=70) # TCP Window
    
    y_pie = y_tcp + 70 + 8
    pdf.image(pf2, x=6, y=y_pie, w=74, h=74) # Pie Chart
    pdf.image(pf3, x=82, y=y_pie + 2, w=122, h=70) # Timeline Chart
    
    pdf.set_y(y_pie + 76)
    
    # ── Heatmap ──
    pdf.add_page()
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 297, "F")
    if probs is not None and len(probs) > 0:
        fig5, ax5 = plt.subplots(figsize=(8, 2.5))
        fig5.patch.set_facecolor('#0f172a')
        ax5.set_facecolor('#0f172a')
        prob_sample = np.array(probs[-60:]).T
        sns.heatmap(prob_sample, ax=ax5, cmap="YlOrRd", yticklabels=["Low", "Med", "High"], linewidths=0.3, linecolor='#0b0f1a')
        ax5.set_title("Probability Heatmap (Latest 60)", color='#94a3b8')
        ax5.set_xlabel("Recent Sequence Index", color='#94a3b8')
        ax5.tick_params(colors='#e0e6f0')
        pf5 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        fig5.tight_layout()
        fig5.savefig(pf5, facecolor=fig5.get_facecolor(), edgecolor='none')
        plt.close(fig5)
        tmp_files.append(pf5)
        pdf.image(pf5, x=10, w=190)
        pdf.ln(5)
    
    # ── Packet dist ──
    if df_raw is not None and 'Length' in df_raw.columns:
        fig6, ax6 = plt.subplots(figsize=(8, 3))
        fig6.patch.set_facecolor('#0f172a')
        ax6.set_facecolor('#111827')
        ax6.hist(df_raw['Length'].dropna(), bins=60, color='#818cf8', edgecolor='#0b0f1a', alpha=0.85)
        ax6.set_title("Packet Length Distribution", color='#94a3b8')
        ax6.set_xlabel("Packet Length (bytes)", color='#94a3b8')
        ax6.set_ylabel("Count", color='#94a3b8')
        ax6.tick_params(colors='#64748b')
        ax6.spines['bottom'].set_color('#1e293b')
        ax6.spines['left'].set_color('#1e293b')
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        pf6 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        fig6.tight_layout()
        fig6.savefig(pf6, facecolor=fig6.get_facecolor(), edgecolor='none')
        plt.close(fig6)
        tmp_files.append(pf6)
        pdf.image(pf6, x=10, w=190)
        
    for pf in tmp_files:
        if os.path.exists(pf):
            try: os.remove(pf)
            except: pass
            
    return bytes(pdf.output())

# ─────────────────────────────────────────
# DIALOGS
# ─────────────────────────────────────────
import base64
import os

def get_img_src(filepath):
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
             data = base64.b64encode(f.read()).decode("utf-8")
        ext = filepath.split('.')[-1]
        return f"data:image/{ext};base64,{data}"
    return "https://via.placeholder.com/150?text=No+Image"

@st.dialog("👥 Developed by", width="large")
def modal_developed_by():
    # Detect current theme's colors for modal
    m_text = "#e0e6f0" if st.session_state.theme == "Dark" else "#0f172a"
    m_sub = "#94a3b8" if st.session_state.theme == "Dark" else "#475569"
    m_card = "rgba(255,255,255,0.05)" if st.session_state.theme == "Dark" else "rgba(0,0,0,0.05)"
    
    st.html(f"""
    <div style="text-align:center;">
      <h2 style="margin-bottom:10px; color:{m_text}; font-family:'Space Mono', monospace;">Developed by</h2>
      <div style="display:flex;gap:30px;flex-wrap:wrap;justify-content:center;">
        <div style="background:{m_card}; padding:15px; border-radius:15px; width:180px; transition:all 0.3s ease;">
          <img src="{get_img_src('images/thisiskanika.jpeg')}" alt="Kanika Rathore" style="width:150px;height:150px;border-radius:50%;object-fit:cover;border:2px solid #38bdf8;">
          <p style="margin-top:8px;font-size:14px;color:{m_text}">
            <strong>Kanika Rathore</strong><br>
            <span style="color:{m_sub};">24BYB1080</span>
          </p>
        </div>
        <div style="background:{m_card}; padding:15px; border-radius:15px; width:180px; transition:all 0.3s ease;">
          <img src="{get_img_src('images/thisisharish.jpeg')}" alt="R Harish" style="width:150px;height:150px;border-radius:50%;object-fit:cover;border:2px solid #38bdf8;">
          <p style="margin-top:8px;font-size:14px;color:{m_text}">
            <strong>R Harish</strong><br>
            <span style="color:{m_sub};">24BYB1159</span>
          </p>
        </div>
        <div style="background:{m_card}; padding:15px; border-radius:15px; width:180px; transition:all 0.3s ease;">
          <img src="{get_img_src('images/thisisakshaya.jpeg')}" alt="Akshaya H" style="width:150px;height:150px;border-radius:50%;object-fit:cover;border:2px solid #38bdf8;">
          <p style="margin-top:8px;font-size:14px;color:{m_text}">
            <strong>Akshaya H</strong><br>
            <span style="color:{m_sub};">24BYB1124</span>
          </p>
        </div>
      </div>
      <hr style="margin:25px 0;border-color:rgba(255,255,255,0.1)">
      <h2 style="margin-bottom:10px; color:{m_text}; font-family:'Space Mono', monospace;">Guided by</h2>
      <div style="display:flex;justify-content:center;">
        <div style="background:{m_card}; padding:15px; border-radius:15px; width:200px; transition:all 0.3s ease;">
          <img src="{get_img_src('images/thisisswami.jpg')}" alt="Swaminathan A" style="width:160px;height:160px;border-radius:50%;object-fit:cover;border:2px solid #10b981;">
          <p style="margin-top:8px;font-size:14px;color:{m_text}">
            <strong>Dr. Swaminathan A</strong><br>
            <span style="color:{m_sub};">Faculty, Computer Networks</span>
          </p>
        </div>
      </div>
    </div>
    """)

@st.dialog("📚 Learn — Network Traffic & TCP Congestion Control", width="large")
def modal_learn():
    m_text = TEXT_COLOR
    m_sub = SUBTEXT_COLOR
    st.html(f"""
    <div style="color:{m_text}; max-width:100%;">

    <!-- ── INTRO ── -->
    <div class="learn-section">
        <h3>👋 Welcome — What is NetSense?</h3>
        <p>
            <strong>NetSense</strong> is an AI-powered network monitoring tool that watches the data flowing through
            your computer in real time, decides whether the network is congested or clear, and then
            <em>simulates</em> how TCP — the protocol that carries most of the internet — would respond to that congestion.
        </p>
        <p>
            Don't worry if those words mean nothing yet. This page explains everything from scratch —
            starting with <em>"what is a network packet?"</em> all the way to <em>"how does an AI learn to detect congestion?"</em>
        </p>
        <div style="margin-top:0.8rem;">
            <span class="concept-pill">Computer Networks</span>
            <span class="concept-pill">TCP Protocol</span>
            <span class="concept-pill">Congestion Control</span>
            <span class="concept-pill">RNN / LSTM</span>
            <span class="concept-pill">Real-Time AI</span>
        </div>
    </div>

    <!-- ── CHAPTER 1: NETWORKS ── -->
    <div class="learn-section">
        <h3>📡 Chapter 1 — How Data Travels Over a Network</h3>

        <h4>What is a Packet?</h4>
        <p>
            When you send a message or load a webpage, your computer does NOT send all the data in one giant chunk.
            It breaks the data into small pieces called <strong>packets</strong>. Each packet is like a labelled
            envelope — it contains a tiny chunk of your data plus information like <em>where it came from</em>,
            <em>where it's going</em>, and <em>how big it is</em>.
        </p>
        <div class="ascii-art">┌─────────────────────────────────────────┐
│  PACKET                                 │
│  ┌──────────┬──────────┬─────────────┐  │
│  │  Source  │  Dest.   │   Payload   │  │
│  │  IP Addr │  IP Addr │  (your data)│  │
│  └──────────┴──────────┴─────────────┘  │
│  Length: 512 bytes   Protocol: TCP      │
└─────────────────────────────────────────┘</div>

        <h4>What is a Protocol?</h4>
        <p>
            A <strong>protocol</strong> is just a set of rules that both sides agree to follow. The two most
            important protocols for most internet traffic are:
        </p>
        <ul>
            <li><strong>TCP (Transmission Control Protocol)</strong> — Reliable delivery. It checks that every
                packet arrives, and resends any that get lost. Used by web browsing, email, file downloads.</li>
            <li><strong>UDP (User Datagram Protocol)</strong> — Fast but no guarantee. Used where speed matters
                more than perfection — like video calls or online gaming.</li>
        </ul>
        <p>NetSense monitors both, but focuses on TCP because TCP is where <em>congestion control</em> happens.</p>

        <h4>What is Network Congestion?</h4>
        <p>
            Imagine a highway. During rush hour, too many cars enter and traffic jams form — cars slow down or stop.
            A network works the same way. When too many packets are sent at once, <strong>routers</strong>
            (the traffic directors of the internet) get overwhelmed, <strong>buffers overflow</strong>, and
            packets get <strong>dropped</strong> (lost). This is called <strong>congestion</strong>.
        </p>
        <div class="ascii-art">Normal:   [P]→[P]→[P]→→→[Router]→→→[Destination]  ✅

Congested: [P][P][P][P][P][P][P]→→[Router🔥]→ packet DROPPED ❌
                                    buffer full!</div>
    </div>

    <!-- ── CHAPTER 2: TCP CONGESTION CONTROL ── -->
    <div class="learn-section">
        <h3>🔧 Chapter 2 — TCP Congestion Control (The Core Theory)</h3>
        <p>
            TCP has a built-in system to detect congestion and slow itself down <em>before</em> things get
            completely jammed. The key variable is called the <strong>Congestion Window (cwnd)</strong>.
        </p>

        <h4>What is the Congestion Window (cwnd)?</h4>
        <p>
            <code>cwnd</code> controls <em>how many packets TCP is allowed to have "in flight" at once</em>
            (sent but not yet acknowledged). A large <code>cwnd</code> = sending data fast.
            A small <code>cwnd</code> = sending slowly, carefully.
        </p>
        <div class="ascii-art">cwnd = 1  →  [P] in flight         (very slow, safe)
cwnd = 8  →  [P][P][P][P][P][P][P][P] in flight  (fast)
cwnd = 32 →  [P]×32 in flight        (very fast — risky if network is full)</div>

        <h4>The AIMD Rule — Additive Increase, Multiplicative Decrease</h4>
        <p>
            TCP follows the <strong>AIMD</strong> strategy to grow its window fairly and shrink it when problems arise:
        </p>
        <div class="phase-box green">
            <div class="phase-label">ADDITIVE INCREASE ↗</div>
            <div style="font-size:0.85rem;color:{m_sub};">When things are going well (ACKs arriving), increase cwnd by +1 each round. Grow slowly and steadily.</div>
        </div>
        <div class="phase-box red">
            <div class="phase-label">MULTIPLICATIVE DECREASE ↘↘</div>
            <div style="font-size:0.85rem;color:{m_sub};">When congestion is detected, cut cwnd aggressively — either halve it (3 duplicate ACKs) or reset to 1 (timeout). React fast to give the network relief.</div>
        </div>

        <hr class="learn-divider">

        <h4>📈 Phase 1 — Slow Start</h4>
        <p>
            When a TCP connection <em>first begins</em>, it knows nothing about how much capacity the network has.
            It starts cautiously with <code>cwnd = 1</code> and <strong>doubles</strong> the window every round
            until it hits a threshold (called <code>ssthresh</code>). Despite the name, this phase grows
            <em>exponentially fast</em>.
        </p>
        <div class="ascii-art">Round 1: cwnd = 1
Round 2: cwnd = 2   (doubled)
Round 3: cwnd = 4   (doubled)
Round 4: cwnd = 8   (doubled)
...until cwnd reaches ssthresh → switch to Congestion Avoidance</div>

        <h4>📉 Phase 2 — Congestion Avoidance</h4>
        <p>
            Once <code>cwnd ≥ ssthresh</code>, TCP slows its growth to <strong>+1 per round</strong>
            (linear growth, not exponential). This is the steady cruising phase — probing for more bandwidth
            carefully.
        </p>

        <h4>⚠️ Phase 3 — Fast Retransmit (3 Duplicate ACKs)</h4>
        <p>
            When the receiver gets packets <em>out of order</em>, it keeps sending the same ACK (acknowledgement)
            for the last packet it received correctly. If the sender receives <strong>3 duplicate ACKs</strong>,
            it knows a specific packet was lost — but the network is still working. So it reacts moderately:
        </p>
        <div class="phase-box yellow">
            <div class="phase-label">3 DUPLICATE ACKs DETECTED</div>
            <div style="font-size:0.85rem;color:#94a3b8;">
                ssthresh = cwnd / 2 &nbsp;|&nbsp; cwnd = ssthresh &nbsp;(halved, NOT reset to 1)<br>
                → Resume from Congestion Avoidance. Less drastic than a full timeout.
            </div>
        </div>

        <h4>🛑 Phase 4 — Timeout (Full Congestion)</h4>
        <p>
            If the sender waits too long and <em>no ACK arrives at all</em>, it means packets were completely
            lost — the network is severely congested. TCP reacts harshly:
        </p>
        <div class="phase-box red">
            <div class="phase-label">TIMEOUT DETECTED</div>
            <div style="font-size:0.85rem;color:#94a3b8;">
                ssthresh = cwnd / 2 &nbsp;|&nbsp; cwnd = 1 &nbsp;(full reset!)<br>
                → Restart from Slow Start. Severe punishment because the network needs maximum relief.
            </div>
        </div>

        <h4>How NetSense Maps This</h4>
        <p>The LSTM classifies every 10-packet window into one of three states, and the AIMD simulation responds:</p>
        <div class="phase-box green">
            <div class="phase-label">🔵 LOW TRAFFIC (Class 0)</div>
            <div style="font-size:0.85rem;color:{m_sub};">Network clear → run Slow Start or Congestion Avoidance → cwnd grows.</div>
        </div>
        <div class="phase-box yellow">
            <div class="phase-label">🟡 MEDIUM TRAFFIC (Class 1)</div>
            <div style="font-size:0.85rem;color:{m_sub};">Mild congestion detected → simulate 3 Duplicate ACKs → ssthresh = cwnd/2, cwnd = ssthresh.</div>
        </div>
        <div class="phase-box red">
            <div class="phase-label">🔴 HIGH CONGESTION (Class 2)</div>
            <div style="font-size:0.85rem;color:{m_sub};">Severe congestion detected → simulate Timeout → ssthresh = cwnd/2, cwnd = 1 (full reset!).</div>
        </div>
    </div>

    <!-- ── CHAPTER 3: RNN & LSTM ── -->
    <div class="learn-section">
        <h3>🧠 Chapter 3 — Recurrent Neural Networks & LSTM</h3>

        <h4>Why Normal Neural Networks Are Not Enough</h4>
        <p>
            A regular neural network (like a feedforward network) looks at <em>one snapshot</em> and makes a
            decision. But network traffic is a <strong>sequence</strong> — what happened in the last 10 packets
            <em>matters</em> for predicting what's happening now. You can't understand congestion from one
            packet alone, just like you can't understand a sentence from one word alone.
        </p>

        <h4>What is an RNN (Recurrent Neural Network)?</h4>
        <p>
            An <strong>RNN</strong> has a <em>memory</em>. At each time step, it takes the current input
            AND a hidden state carried over from the previous step. This lets it understand sequences.
        </p>
        <div class="ascii-art">Time step:  t=1        t=2        t=3        t=4
Input:   [packet1]  [packet2]  [packet3]  [packet4]
              ↓          ↓          ↓          ↓
RNN:    [hidden1]→ [hidden2]→ [hidden3]→ [hidden4] → OUTPUT (Low/Med/High)
         memory      memory      memory      memory</div>
        <p>
            The arrows show the hidden state being passed forward — the RNN <em>remembers</em> earlier packets
            when making its decision at each step.
        </p>

        <h4>The Problem with Plain RNNs — Vanishing Gradients</h4>
        <p>
            Plain RNNs struggle to remember things from <em>many steps ago</em>. During training, the signal
            used to teach the network (the <em>gradient</em>) keeps getting multiplied by small numbers as it
            travels back through time, eventually becoming so tiny it effectively disappears.
            This is called the <strong>vanishing gradient problem</strong>.
        </p>

        <h4>What is LSTM (Long Short-Term Memory)?</h4>
        <p>
            <strong>LSTM</strong> solves the vanishing gradient problem with a clever <em>gating mechanism</em>.
            Each LSTM cell has three gates that control what to remember and what to forget:
        </p>
        <div class="phase-box green">
            <div class="phase-label">🟢 INPUT GATE</div>
            <div style="font-size:0.85rem;color:{m_sub};">Decides what new information from the current packet to write into memory.</div>
        </div>
        <div class="phase-box purple">
            <div class="phase-label">🟣 FORGET GATE</div>
            <div style="font-size:0.85rem;color:{m_sub};">Decides what old information to erase from memory. (e.g., "the burst from 8 packets ago is no longer relevant")</div>
        </div>
        <div class="phase-box yellow">
            <div class="phase-label">🟡 OUTPUT GATE</div>
            <div style="font-size:0.85rem;color:{m_sub};">Decides what part of memory to use for the current prediction.</div>
        </div>
        <div class="ascii-art">       ┌────────────────────────────────────┐
       │  LSTM Cell                         │
       │                                    │
  h(t-1) →┤ Forget Gate → erase old memory  │
  x(t)  →┤ Input  Gate → write new memory  │→ h(t) → next cell
          │ Output Gate → read for output   │
          │             ↓                   │
          └─────────── ŷ(t) ───────────────┘
                 (Low / Medium / High)</div>

        <h4>NetSense's LSTM Architecture</h4>
        <p>NetSense uses a <strong>2-layer stacked LSTM</strong> built with PyTorch:</p>
        <div class="phase-box" style="border-left-color:#38bdf8;">
            <div class="phase-label" style="color:#38bdf8;">LAYER 1 — LSTM (hidden size: 64)</div>
            <div style="font-size:0.85rem;color:{m_sub};">Takes the 10-packet sequence (5 features each) and learns low-level temporal patterns — like "packet size is growing rapidly".</div>
        </div>
        <div class="phase-box" style="border-left-color:#818cf8;">
            <div class="phase-label" style="color:#818cf8;">DROPOUT (30%) — Regularisation</div>
            <div style="font-size:0.85rem;color:{m_sub};">Randomly disables 30% of neurons during training so the network can't memorise training data. Forces it to generalise.</div>
        </div>
        <div class="phase-box" style="border-left-color:#38bdf8;">
            <div class="phase-label" style="color:#38bdf8;">LAYER 2 — LSTM (hidden size: 32)</div>
            <div style="font-size:0.85rem;color:{m_sub};">Takes the output of Layer 1 and learns higher-level patterns — like "this is a sustained burst consistent with high congestion".</div>
        </div>
        <div class="phase-box" style="border-left-color:#34d399;">
            <div class="phase-label" style="color:#34d399;">FULLY CONNECTED LAYERS → Softmax</div>
            <div style="font-size:0.85rem;color:{m_sub};">32 → 3 neurons. Outputs a probability for each class: P(Low), P(Medium), P(High). The highest one wins.</div>
        </div>
    </div>

    <!-- ── CHAPTER 4: FEATURES ── -->
    <div class="learn-section">
        <h3>⚙️ Chapter 4 — What the AI Actually Looks At</h3>
        <p>
            For each packet, NetSense extracts <strong>5 numerical features</strong>. These 5 numbers,
            collected across 10 consecutive packets, form one input sequence to the LSTM.
        </p>
        <table style="width:100%; border-collapse:collapse; font-size:0.85rem; margin-top:0.5rem;">
            <thead>
                <tr style="border-bottom:1px solid rgba(56,189,248,0.2);">
                    <th style="color:#38bdf8; text-align:left; padding:6px 8px; font-family:'Space Mono',monospace; font-size:0.78rem;">Feature</th>
                    <th style="color:#38bdf8; text-align:left; padding:6px 8px; font-family:'Space Mono',monospace; font-size:0.78rem;">What it Measures</th>
                    <th style="color:#38bdf8; text-align:left; padding:6px 8px; font-family:'Space Mono',monospace; font-size:0.78rem;">Why it Matters</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="color:#7dd3fc; padding:6px 8px; font-family:'Space Mono',monospace;"><code>packet_count</code></td>
                    <td style="color:{m_sub}; padding:6px 8px;">Number of packets seen</td>
                    <td style="color:{m_sub}; padding:6px 8px;">High counts suggest burst activity</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="color:#7dd3fc; padding:6px 8px; font-family:'Space Mono',monospace;"><code>avg_size</code></td>
                    <td style="color:{m_sub}; padding:6px 8px;">Average packet length (bytes)</td>
                    <td style="color:{m_sub}; padding:6px 8px;">Large packets = bulk transfers; small = control traffic</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="color:#7dd3fc; padding:6px 8px; font-family:'Space Mono',monospace;"><code>size_variation</code></td>
                    <td style="color:{m_sub}; padding:6px 8px;">How much packet size changes between consecutive packets</td>
                    <td style="color:{m_sub}; padding:6px 8px;">High variation = mixed traffic types (e.g., streaming + ACKs)</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="color:#7dd3fc; padding:6px 8px; font-family:'Space Mono',monospace;"><code>packet_rate</code></td>
                    <td style="color:{m_sub}; padding:6px 8px;">Rolling sum of bytes (intensity over last 2 packets)</td>
                    <td style="color:{m_sub}; padding:6px 8px;">High rate = heavy load on the network link</td>
                </tr>
                <tr>
                    <td style="color:#7dd3fc; padding:6px 8px; font-family:'Space Mono',monospace;"><code>rate_change</code></td>
                    <td style="color:{m_sub}; padding:6px 8px;">How much the packet rate changes step to step</td>
                    <td style="color:{m_sub}; padding:6px 8px;">Sudden spikes in rate_change signal the start of congestion</td>
                </tr>
            </tbody>
        </table>
        <p style="margin-top:0.8rem;">
            Before being fed to the LSTM, all 5 features are <strong>normalised to the [0, 1] range</strong>
            using <code>MinMaxScaler</code>. This prevents one feature (like packet size in bytes) from
            dominating over another (like packet count = 1) just because of scale differences.
        </p>
    </div>

    <!-- ── CHAPTER 5: END-TO-END FLOW ── -->
    <div class="learn-section">
        <h3>🔄 Chapter 5 — How NetSense Works End-to-End</h3>
        <div class="step-row"><span class="step-num">1</span><div><strong style="color:{m_text};">Packet Capture</strong> — Scapy listens on your network interface on a background thread, intercepting every TCP/UDP packet in real time (or you upload a <code>.pcap</code> file recorded earlier).</div></div>
        <div class="step-row"><span class="step-num">2</span><div><strong style="color:{m_text};">Feature Extraction</strong> — Each raw packet is parsed: timestamp, length, protocol. Then the 5 derived features are computed per packet.</div></div>
        <div class="step-row"><span class="step-num">3</span><div><strong style="color:{m_text};">Sequence Creation</strong> — Packets are grouped into overlapping windows of 10. Window 1 = packets 1–10, Window 2 = packets 2–11, etc. Each window = one LSTM input.</div></div>
        <div class="step-row"><span class="step-num">4</span><div><strong style="color:{m_text};">Normalisation</strong> — Features in each window are scaled to [0,1] so the LSTM receives consistent input magnitudes.</div></div>
        <div class="step-row"><span class="step-num">5</span><div><strong style="color:{m_text};">LSTM Inference</strong> — The trained PyTorch model processes the window through 2 LSTM layers + fully connected layers and outputs three probabilities: P(Low), P(Medium), P(High).</div></div>
        <div class="step-row"><span class="step-num">6</span><div><strong style="color:{m_text};">Classification</strong> — The class with the highest probability wins: <span style="color:#60a5fa;">Low</span> / <span style="color:#34d399;">Medium</span> / <span style="color:#f87171;">High</span>.</div></div>
        <div class="step-row"><span class="step-num">7</span><div><strong style="color:{m_text};">AIMD Simulation</strong> — Each prediction drives the TCP congestion window simulation: Low → grow cwnd, Medium → halve it (3 Dup ACKs), High → reset to 1 (Timeout). The graph in the Dashboard tab shows this live.</div></div>
        <div class="step-row"><span class="step-num">8</span><div><strong style="color:{m_text};">Visualisation</strong> — The Live Monitor tab shows the animated traffic pipe and real-time metrics. The Dashboard tab shows the AIMD graph, probability heatmap, class distribution pie, and packet length histogram.</div></div>
        <div class="ascii-art">Network Interface / PCAP File
        ↓
  [Scapy Packet Capture]
        ↓
  [Feature Extraction: 5 features/packet]
        ↓
  [Sliding Window: groups of 10 packets]
        ↓
  [MinMaxScaler Normalisation]
        ↓
  [LSTM Layer 1 → Dropout → LSTM Layer 2]
        ↓
  [FC Layer → Softmax → Class: 0/1/2]
        ↓
  [AIMD Engine: update cwnd]    [Dashboard Charts]
        ↓                              ↓
  Live Traffic Pipe Display    AIMD Graph + Heatmaps</div>
    </div>

    <!-- ── CHAPTER 6: WHY AI ── -->
    <div class="learn-section">
        <h3>💡 Chapter 6 — Why Use AI Instead of Simple Rules?</h3>
        <p>
            A naive approach to congestion detection might be: <em>"if more than 1000 packets per second → congested."</em>
            But this breaks on a fast fibre link (1000 pps is normal) and misses subtle patterns on slow links.
        </p>
        <p>
            The LSTM learns <strong>from data</strong> — it discovers which combinations of packet size, rate,
            and variation actually predict congestion on your specific network, without you having to hand-tune thresholds.
            It can detect <em>multi-feature patterns</em> across time that simple rules completely miss —
            like "small packets, rapidly varying size, low rate = probably ACK storm, not congestion."
        </p>
        <p>
            This is the core advantage of machine learning for network analysis: <strong>adaptability to context</strong>.
        </p>
    </div>

    <div style="text-align:center; padding:1rem 0 0.5rem; color:#64748b; font-family:'Space Mono',monospace; font-size:0.75rem;">
        NetSense · Built with PyTorch, Streamlit & Scapy · Computer Networks Project
    </div>
    </div>
    """)

@st.dialog("❓ Help — How to Use NetSense", width="large")
def modal_help():
    m_text = TEXT_COLOR
    m_sub = SUBTEXT_COLOR
    st.html(f"""
    <div style="color:{m_text}; max-width:100%; font-family:'Inter',sans-serif;">

    <!-- INTRO -->
    <div style="background:rgba(56,189,248,0.06); border:1px solid rgba(56,189,248,0.2); border-radius:12px; padding:1rem 1.3rem; margin-bottom:1.2rem;">
        <div style="font-family:'Space Mono',monospace; font-size:0.95rem; color:#38bdf8; font-weight:700; margin-bottom:0.4rem;">👋 Quick Start Guide</div>
        <div style="font-size:0.85rem; color:{m_sub}; line-height:1.65;">
            NetSense has two ways to analyse network traffic — <strong style="color:{m_text};">Live Capture</strong> (watch your real network right now)
            and <strong style="color:{m_text};">Upload a PCAP file</strong> (analyse a saved recording). Pick whichever suits you below.
        </div>
    </div>

    <!-- ══ PATH A: LIVE CAPTURE ══ -->
    <div class="help-section-title">🔴 Path A — Live Network Capture</div>
    <div style="font-size:0.82rem; color:#64748b; margin-bottom:0.8rem; font-style:italic;">Use this to monitor your actual network traffic in real time.</div>

    <div class="help-step-card">
        <div class="help-step-num">1</div>
        <div class="help-step-body">
            <div class="help-step-title">Run the app with admin privileges</div>
            <div class="help-step-desc">
                Live packet capture requires root access. Open your terminal and run:<br><br>
                <code>sudo streamlit run app.py</code><br><br>
                Without <code>sudo</code>, Scapy cannot sniff packets and live capture will fail.
            </div>
        </div>
    </div>

    <div class="help-step-card">
        <div class="help-step-num">2</div>
        <div class="help-step-body">
            <div class="help-step-title">Load the model in the Sidebar</div>
            <div class="help-step-desc">
                On the left sidebar, check that the <strong>Model Path</strong> field says
                <code>tcp_udp_lstm_pytorch.pt</code> (or type the correct path to your <code>.pt</code> file).
                This is the trained LSTM — without it, classification won't work.
            </div>
        </div>
    </div>

    <div class="help-step-card">
        <div class="help-step-num">3</div>
        <div class="help-step-body">
            <div class="help-step-title">Go to the Live Monitoring tab</div>
            <div class="help-step-desc">
                Click the <strong>Live Monitoring</strong> tab at the top. In the dropdown, make sure
                <strong>🔴 Live Capture</strong> is selected (not PCAP Upload).
            </div>
        </div>
    </div>

    <div class="help-step-card">
        <div class="help-step-num">4</div>
        <div class="help-step-body">
            <div class="help-step-title">Click "Start Live Capture"</div>
            <div class="help-step-desc">
                Hit the <strong>▶ Start Live Capture</strong> button. NetSense will begin sniffing packets
                on your network interface in the background. You'll see a spinner — wait a few seconds
                while it collects at least 10 packets to form the first sequence.
            </div>
        </div>
    </div>

    <div class="help-step-card">
        <div class="help-step-num">5</div>
        <div class="help-step-body">
            <div class="help-step-title">Watch the Live Monitor panel update</div>
            <div class="help-step-desc">
                Once enough packets are captured, you'll see:
                <br>• The animated <strong>traffic pipe</strong> — colour shows congestion level
                <br>• The <strong>status badge</strong> — ✅ LOW / ⚠️ MEDIUM / 🛑 HIGH
                <br>• The <strong>Dynamic TCP Window Size</strong> (cwnd) — how wide the simulated TCP window is
                <br>• <strong>Metric cards</strong> showing total sequences classified so far
                <br>• The <strong>Last 5 Predictions</strong> table with raw probabilities
                <br><br>The page auto-refreshes every 2.5 seconds — just leave it open.
            </div>
        </div>
    </div>

    <div class="help-step-card">
        <div class="help-step-num">6</div>
        <div class="help-step-body">
            <div class="help-step-title">Switch to the Dashboard tab for detailed charts</div>
            <div class="help-step-desc">
                Click the <strong>Dashboard</strong> tab to see:
                <br>• <strong>TCP AIMD Congestion Window graph</strong> — the sawtooth pattern showing cwnd growing and collapsing
                <br>• <strong>Class Distribution pie chart</strong> — what % of traffic was Low / Medium / High
                <br>• <strong>Smoothed Traffic Timeline</strong> — how congestion evolved over time
                <br>• <strong>Probability Heatmap</strong> — the AI's confidence for each class across sequences
                <br>• <strong>Packet Length Distribution</strong> histogram
            </div>
        </div>
    </div>

    <div class="help-step-card">
        <div class="help-step-num">7</div>
        <div class="help-step-body">
            <div class="help-step-title">Stop capturing when done</div>
            <div class="help-step-desc">
                Click <strong>⏹ Stop Capture</strong> to halt the background sniffing thread.
                Your charts and predictions will remain visible — the dashboard won't clear until you start a new session.
            </div>
        </div>
    </div>

    <div class="help-warn">
        <strong>⚠️ Important:</strong> Live capture only works if your machine has a network interface available
        and Scapy is installed. If you see a Scapy error, install it with <code>pip install scapy</code>
        and re-run with <code>sudo</code>.
    </div>

    <!-- ══ PATH B: PCAP UPLOAD ══ -->
    <div class="help-section-title">📂 Path B — Upload a PCAP File</div>
    <div style="font-size:0.82rem; color:#64748b; margin-bottom:0.8rem; font-style:italic;">Use this if you have a pre-recorded <code>.pcap</code> or <code>.pcapng</code> file (e.g. from Wireshark).</div>

    <div class="help-step-card">
        <div class="help-step-num">1</div>
        <div class="help-step-body">
            <div class="help-step-title">Go to the Live Monitoring tab</div>
            <div class="help-step-desc">
                Click the <strong>Live Monitoring</strong> tab. In the mode dropdown, select
                <strong>📂 Upload PCAP File</strong>.
            </div>
        </div>
    </div>

    <div class="help-step-card">
        <div class="help-step-num">2</div>
        <div class="help-step-body">
            <div class="help-step-title">Upload your PCAP file</div>
            <div class="help-step-desc">
                Click the file uploader and select your <code>.pcap</code> or <code>.pcapng</code> file.
                NetSense will parse every packet in the file — no live internet needed.
                <br><br>
                <strong>Don't have a PCAP?</strong> Record one with Wireshark (free tool) — just start a capture,
                browse the web for 30 seconds, stop it, and save as <code>.pcap</code>.
            </div>
        </div>
    </div>

    <div class="help-step-card">
        <div class="help-step-num">3</div>
        <div class="help-step-body">
            <div class="help-step-title">Wait for processing</div>
            <div class="help-step-desc">
                NetSense will extract features from every packet, build sequences of 10, normalise them,
                and run the LSTM on all sequences at once. You'll see a
                <strong>✅ PCAP processed!</strong> confirmation when done.
            </div>
        </div>
    </div>

    <div class="help-step-card">
        <div class="help-step-num">4</div>
        <div class="help-step-body">
            <div class="help-step-title">Go to the Dashboard tab</div>
            <div class="help-step-desc">
                Click the <strong>Dashboard</strong> tab — all charts will be populated with results from
                your file. Everything is the same as live mode: AIMD graph, heatmap, pie chart, histogram.
            </div>
        </div>
    </div>

    <div class="help-tip">
        <strong>💡 Tip:</strong> PCAP mode is great for demonstrations and presentations — you get consistent,
        repeatable results every time without needing <code>sudo</code> or a live network.
    </div>

    <!-- ══ READING THE RESULTS ══ -->
    <div class="help-section-title">📊 Understanding What You See</div>

    <div style="margin-bottom:0.6rem;">
        <div class="help-badge blue">LOW</div>
        <span style="font-size:0.84rem; color:#94a3b8;">Network is clear. The LSTM detected low packet rate and small size variation.
        TCP window grows steadily — Slow Start or Congestion Avoidance phase.</span>
    </div>
    <div style="margin-bottom:0.6rem;">
        <div class="help-badge yellow">MEDIUM</div>
        <span style="font-size:0.84rem; color:#94a3b8;">Mild congestion. The AI detected elevated traffic intensity.
        Simulates 3 Duplicate ACKs — TCP window is halved (ssthresh = cwnd/2, cwnd = ssthresh).</span>
    </div>
    <div style="margin-bottom:1rem;">
        <div class="help-badge red">HIGH</div>
        <span style="font-size:0.84rem; color:#94a3b8;">Severe congestion. High packet rate with big rate changes detected.
        Simulates a Timeout — TCP window resets to 1 (full restart from Slow Start).</span>
    </div>

    <div class="help-tip">
        <strong>💡 AIMD Graph:</strong> The sawtooth pattern is <em>normal and expected</em>. TCP is designed to probe aggressively,
        detect congestion, cut back, then probe again. A perfectly flat line would mean no congestion — and also no growth.
    </div>

    <!-- ══ SIDEBAR SETTINGS ══ -->
    <div class="help-section-title">⚙️ Sidebar Settings Explained</div>

    <div class="help-faq">
        <div class="help-faq-q">Model Path (.pt)</div>
        <div class="help-faq-a">Path to your trained PyTorch LSTM file. Default: <strong>tcp_udp_lstm_pytorch.pt</strong> in the same folder as app.py. Change this if your model file has a different name or is in a subfolder.</div>
    </div>
    <div class="help-faq">
        <div class="help-faq-q">Sequence Timesteps (slider: 5–20)</div>
        <div class="help-faq-a">How many consecutive packets form one input sequence to the LSTM. Default is <strong>10</strong>. Higher values = the model looks further back in time but needs more packets before making its first prediction. Keep at 10 unless you retrained the model with a different value.</div>
    </div>

    <!-- ══ FAQ ══ -->
    <div class="help-section-title">❓ Common Questions</div>

    <div class="help-faq">
        <div class="help-faq-q">The page shows "waiting to accumulate packets" — what do I do?</div>
        <div class="help-faq-a">Just wait. NetSense needs at least <strong>10 packets</strong> (equal to the Sequence Timesteps value) before it can make its first prediction. If your network is quiet, try opening a browser and loading a website to generate traffic.</div>
    </div>
    <div class="help-faq">
        <div class="help-faq-q">Dashboard tab says "go to Live Monitoring first" — why?</div>
        <div class="help-faq-a">The Dashboard only shows results after the model has run. Either start a Live Capture and wait for predictions, or upload a PCAP file. Once predictions exist, switch to Dashboard.</div>
    </div>
    <div class="help-faq">
        <div class="help-faq-q">All my predictions are HIGH / all LOW — is something wrong?</div>
        <div class="help-faq-a">Not necessarily. If your network is very busy (downloading files, streaming video), HIGH is accurate. If it's idle, LOW is accurate. Try generating different traffic types and watch the predictions shift.</div>
    </div>
    <div class="help-faq">
        <div class="help-faq-q">Can I use this on Windows?</div>
        <div class="help-faq-a">Scapy works on Windows but requires <strong>Npcap</strong> to be installed (free from npcap.com). Run the terminal as Administrator instead of using <code>sudo</code>. PCAP upload mode works without any extra setup on Windows.</div>
    </div>

    <div style="text-align:center; padding:1.2rem 0 0.3rem; color:#64748b; font-family:'Space Mono',monospace; font-size:0.72rem;">
        NetSense · Built with PyTorch, Streamlit & Scapy · Computer Networks Project
    </div>
    </div>
    """)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
col_logo, col_title, col_opts = st.columns([0.5, 4.2, 5.3])

with col_title:
    st.markdown('<div class="hero-title">🌐 NetSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">LSTM-powered Network Traffic Classifier · TCP/UDP Analysis</div>', unsafe_allow_html=True)

with col_opts:
    st.markdown("<br>", unsafe_allow_html=True)
    b_learn, b_dev, b_help, b_dl = st.columns([0.8, 1.2, 0.8, 1.3])
    with b_learn:
        if st.button("📚 Learn", use_container_width=True):
            modal_learn()
    with b_dev:
        if st.button("👥 Developed by", use_container_width=True):
            modal_developed_by()
    with b_help:
        if st.button("❓ Help", use_container_width=True):
            modal_help()
    with b_dl:
        if 'preds' in st.session_state and st.session_state['preds'] is not None and len(st.session_state['preds']) > 0 and 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
            preds = list(st.session_state['preds'])
            df_raw = st.session_state['df_raw']
            probs = st.session_state.get('probs', [])
            total_pkts = len(df_raw)
            low = int(sum(1 for p in preds if p == 0))
            med = int(sum(1 for p in preds if p == 1))
            high = int(sum(1 for p in preds if p == 2))
            latest = preds[-1]
            if latest == 0:
                current_status = "LOW TRAFFIC (CLEAR)"
                window_kb = "64"
            elif latest == 1:
                current_status = "MEDIUM TRAFFIC"
                window_kb = "32"
            else:
                current_status = "HIGH CONGESTION!"
                window_kb = "8"
            
            if 'pdf_ready' in st.session_state and st.session_state['pdf_ready']:
                def reset_pdf_state():
                    st.session_state['pdf_ready'] = False

                st.download_button(
                    label="📥 Save PDF",
                    data=st.session_state['cached_pdf'],
                    file_name="netsense_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                    on_click=reset_pdf_state
                )
            else:
                if st.button("⚙️ PDF Report", use_container_width=True, type="primary"):
                    with st.spinner("Compiling PDF Graphs..."):
                        pdf_bytes = generate_pdf_report(preds, probs, df_raw, window_kb, current_status, low, med, high, total_pkts)
                        st.session_state['cached_pdf'] = pdf_bytes
                        st.session_state['pdf_ready'] = True
                    st.rerun()
        else:
            st.button("⚙️ PDF Report", use_container_width=True, disabled=True)

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
    st.markdown("### 🚀 How to Use NetSense")
    st.markdown("""
    **Step 1: Choose Mode**  
    Go to **Live Monitoring** and select **🔴 Live Capture** or **📂 Upload PCAP**.

    **Step 2: Start Data Flow**  
    Click **▶ Start** (needs root) or upload your file. Wait for the logic to accumulate **10 packets**.

    **Step 3: Monitor Live**  
    Watch the **Traffic Pipe** and real-time metrics update every 2.5s.

    **Step 4: Check Dashboard**  
    Switch to the **Dashboard** tab for the AIMD simulation and class heatmaps.

    **Step 5: Export PDF**  
    Click the **📄 Save PDF Report** button in the header once analysis is ready.
    """)


# ─────────────────────────────────────────
# BACKGROUND SNIFFER LOGIC
# ─────────────────────────────────────────
@st.cache_resource
def get_packet_buffer():
    return collections.deque(maxlen=2000)

packet_buffer = get_packet_buffer()

@st.cache_resource
def get_stop_event():
    return threading.Event()

stop_event = get_stop_event()

def packet_handler(pkt):
    try:
        if SCAPY_AVAILABLE:
            from scapy.all import IP, TCP, UDP
            if IP in pkt:
                length = len(pkt)
                proto = "TCP" if TCP in pkt else "UDP" if UDP in pkt else "OTHER"
                packet_buffer.append({
                    "Timestamp": time.time(),
                    "Length": length,
                    "Protocol": proto
                })
    except Exception:
        pass

def sniff_traffic():
    try:
        if not SCAPY_AVAILABLE:
            packet_buffer.append({"Error": "ScapyNotInstalled"})
            return
        from scapy.all import sniff
        sniff(prn=packet_handler, store=0, stop_filter=lambda x: stop_event.is_set())
    except PermissionError:
        packet_buffer.append({"Error": "PermissionError"})
    except Exception as e:
        packet_buffer.append({"Error": str(e)})

if 'capturing' not in st.session_state:
    st.session_state['capturing'] = False

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2 = st.tabs(["Live Monitoring", "Dashboard"])
mode = None 

# ══════════════════════════════════════════
# TAB 1 — LIVE MONITORING
# ══════════════════════════════════════════
with tab1:
    prev_mode = st.session_state.get('last_mode')
    mode = st.radio("Select Data Source", ["🔴 Live Capture (Npcap)", "📂 Upload PCAP File"], horizontal=True)
    st.session_state['last_mode'] = mode
    
    # Clear PDF cache if we switch modes
    if prev_mode and prev_mode != mode:
        st.session_state['pdf_ready'] = False
        st.session_state['cached_pdf'] = None
    
    if mode == "🔴 Live Capture (Npcap)":

        st.markdown('<div class="section-header">Live Network Traffic Interception</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Simulated real-time packet stream (cloud-safe mode).</div>', unsafe_allow_html=True)
    
        # ─────────────────────────────────────────
        # LOAD DATASET (ONCE)
        # ─────────────────────────────────────────
        @st.cache_data
        def load_stream_data():
            return pd.read_csv("perfect_traffic.csv")  # your dataset
    
        df_full = load_stream_data()
    
        # ─────────────────────────────────────────
        # SESSION INIT
        # ─────────────────────────────────────────
        if "stream_index" not in st.session_state:
            st.session_state.stream_index = 0
    
        if "live_buffer" not in st.session_state:
            st.session_state.live_buffer = []
    
        # ─────────────────────────────────────────
        # CONTROL BUTTONS (UNCHANGED UI)
        # ─────────────────────────────────────────
        col_a, col_b = st.columns([1, 1])
    
        with col_a:
            if not st.session_state['capturing']:
                if st.button("▶️ Start Live Capture"):
                    st.session_state['capturing'] = True
                    st.rerun()
            else:
                if st.button("⏸️ Pause Capture"):
                    st.session_state['capturing'] = False
                    st.rerun()
    
        with col_b:
            if st.button("🗑️ Clear Buffer & Dashboard"):
                st.session_state.live_buffer = []
                st.session_state.stream_index = 0
                for key in ['df_raw', 'df_proc', 'preds', 'probs']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
        # ─────────────────────────────────────────
        # 🔴 SIMULATED STREAM (REPLACES SCAPY)
        # ─────────────────────────────────────────
        timesteps = 10
        chunk_size = 5
    
        if st.session_state['capturing']:
    
            start = st.session_state.stream_index
            end = start + chunk_size
    
            df_chunk = df_full.iloc[start:end].copy()
    
            # loop stream
            st.session_state.stream_index = end % len(df_full)
    
            # add slight randomness → feels live
            if "Length" in df_chunk.columns:
                df_chunk["Length"] += np.random.randint(-5, 5, size=len(df_chunk))
    
            # append to buffer
            st.session_state.live_buffer.extend(df_chunk.to_dict("records"))
    
        # ─────────────────────────────────────────
        # 🔁 USE YOUR EXISTING PIPELINE (UNCHANGED)
        # ─────────────────────────────────────────
        packet_buffer = st.session_state.get("live_buffer", [])
    
        if st.session_state['capturing'] and len(packet_buffer) < (timesteps + 1):
            st.info(f"⏳ Streaming packets... {len(packet_buffer)}/{timesteps+1}")
    
        if len(packet_buffer) >= (timesteps + 1):
    
            with st.spinner("Monitoring live network..."):
    
                current_pkts = packet_buffer
    
                df_raw = pd.DataFrame(current_pkts)
                df_proc = preprocess(df_raw)
                X_seq, features = make_sequences(df_proc, timesteps)
    
                if len(X_seq) > 0:
                    try:
                        model = load_model(model_path)
                        preds, probs = predict(model, X_seq)
    
                        st.session_state['preds'] = preds
                        st.session_state['probs'] = probs
                        st.session_state['df_proc'] = df_proc
                        st.session_state['df_raw'] = df_raw
    
                    except Exception as e:
                        st.error(f"❌ Model prediction error: {e}")
                        st.stop()
    
            preds = st.session_state.get('preds', [])
    
            if not isinstance(preds, (list, np.ndarray)) or len(preds) == 0:
                st.warning("⏳ Waiting for predictions...")
                st.stop()
    
            recent_pred = preds[-1]
        
    elif mode == "📂 Upload PCAP File":

        st.markdown('<div class="section-header">PCAP File Analysis</div>', unsafe_allow_html=True)

        uploaded_pcap = st.file_uploader("Upload PCAP file", type=["pcap", "pcapng"])

        if uploaded_pcap is not None:
            # Check if this file is already processed to avoid redundant work
            if st.session_state.get('last_uploaded_file') != uploaded_pcap.name:
                with st.status(f"Parsing {uploaded_pcap.name}...") as status:
                    df_raw = parse_pcap(uploaded_pcap)
                    
                    if df_raw.empty:
                        st.error("No valid packets found in PCAP.")
                        st.stop()
                        
                    df_proc = preprocess(df_raw)
                    X_seq, features = make_sequences(df_proc, timesteps)
                    
                    if len(X_seq) == 0:
                        st.warning(f"Not enough packets (need >{timesteps}) to form a sequence.")
                        st.stop()
                        
                    status.update(label="Running AI Classifier...", state="running")
                    model = load_model(model_path)
                    preds, probs = predict(model, X_seq)
                    
                    st.session_state['preds'] = preds
                    st.session_state['probs'] = probs
                    st.session_state['df_proc'] = df_proc
                    st.session_state['df_raw'] = df_raw
                    st.session_state['last_uploaded_file'] = uploaded_pcap.name
                    st.session_state['pdf_ready'] = False # New data means new report needed
                    
                st.success(f"✅ {uploaded_pcap.name} processed successfully!")
                st.info("👉 Switch to the **Dashboard** tab for the full report.")
                st.rerun() # Force update header PDF button
            else:
                st.success(f"✅ {uploaded_pcap.name} is loaded.")
                if st.button("🗑️ Clear Data & Upload New"):
                    del st.session_state['last_uploaded_file']
                    for key in ['df_raw', 'df_proc', 'preds', 'probs', 'pdf_ready', 'cached_pdf']:
                        if key in st.session_state: del st.session_state[key]
                    st.rerun()

    
# ══════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════
with tab2:
    
    if 'preds' not in st.session_state:
        st.session_state['preds'] = []

    if 'probs' not in st.session_state:
        st.session_state['probs'] = []

    if 'df_proc' not in st.session_state:
        st.session_state['df_proc'] = None

    if 'df_raw' not in st.session_state:
        st.session_state['df_raw'] = None
    
    if 'preds' not in st.session_state or st.session_state['preds'] is None or len(st.session_state['preds']) == 0:
        if st.session_state.get('capturing', False):
            st.info("📡 Live capture is waiting to accumulate enough network packets. Please wait a few seconds while we build the first traffic sequence...")
        else:
            st.markdown('<div class="info-box">⬅️ Go to **Live Monitoring** tab and either:<br>'
            '• Start Live Capture 🔴<br>' 
            '• OR Upload a PCAP file 📂<br><br>'
            'Then return here for congestion analysis.</div>',unsafe_allow_html=True)
        st.stop()


    preds = st.session_state['preds']
    probs = st.session_state['probs']
    df_proc = st.session_state['df_proc']
    df_raw = st.session_state['df_raw']

    st.markdown('<div class="section-header">TCP Congestion Window (AIMD Simulation)</div>', unsafe_allow_html=True)

    # ── TCP AIMD Simulation Graph ──
    # We want the MOST RECENT 40 predictions so the graph scrolls!
    sim_length = min(len(preds), 40)
    sim_preds = preds[-sim_length:]
    
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
    fig_tcp.patch.set_facecolor(BG_COLOR)
    ax_tcp.set_facecolor(PLOT_CARD_BG)
    
    x_vals = np.arange(len(cwnd_history))
    ax_tcp.plot(x_vals, cwnd_history, color='#0ea5e9', linewidth=2.5, marker='o', markersize=5, markerfacecolor='#ef4444')
    
    for (x, y, label, ev_type) in events:
        ax_tcp.plot(x, y, marker='o', color='#ef4444', markersize=8)
        
        if label != "":
            y_offset = -6 if ev_type == '3ack' else 6
            x_offset = 1 if ev_type == 'timeout' else 0.5
            ax_tcp.annotate(label, (x, y), xytext=(x+x_offset, y+y_offset), color=TEXT_COLOR, fontsize=10, fontweight='bold')
            
            ax_tcp.vlines(x, ymin=0, ymax=y, colors=SUBTEXT_COLOR, linestyles='dotted', alpha=0.9, linewidth=1.5)
            ax_tcp.hlines(y, xmin=0, xmax=x, colors=SUBTEXT_COLOR, linestyles='dotted', alpha=0.9, linewidth=1.5)
        
    ax_tcp.set_xlabel("Transmission Round (Latest 40)", color=SUBTEXT_COLOR, fontsize=11, fontweight='bold')
    ax_tcp.set_ylabel("Congestion Window Size", color=SUBTEXT_COLOR, fontsize=11, fontweight='bold')
    
    y_max_bound = max(cwnd_history)
    y_max_bound = y_max_bound + 15 if y_max_bound > 0 else 40
    ax_tcp.set_ylim(0, y_max_bound)
    ax_tcp.set_xlim(0, len(cwnd_history))
    
    ax_tcp.set_xticks(np.arange(0, len(cwnd_history)+1, 2))
    ax_tcp.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax_tcp.spines['bottom'].set_color(PLOT_BORDER_COLOR)
    ax_tcp.spines['left'].set_color(PLOT_BORDER_COLOR)
    ax_tcp.spines['top'].set_visible(False)
    ax_tcp.spines['right'].set_visible(False)
    ax_tcp.grid(color=PLOT_BORDER_COLOR, linestyle='-', linewidth=0.3, alpha=0.5)
    
    st.pyplot(fig_tcp, clear_figure=True)
    plt.close()

    st.markdown('<br><div class="section-header">Traffic Overview</div>', unsafe_allow_html=True)
    
    # ── Row 1: Pie + Bar ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Class Distribution (Recent)**")
        # Base pie off last 200 for responsiveness
        recent_preds = preds[-200:]
        labels = [LABEL_NAMES[i] for i in range(3)]
        counts = [int(np.sum(recent_preds == i)) for i in range(3)]
        colors = ["#60a5fa", "#34d399", "#f87171"]

        fig1, ax1 = plt.subplots(figsize=(4.5, 4))
        fig1.patch.set_facecolor(BG_COLOR)
        ax1.set_facecolor(BG_COLOR)
        
        # Guard against empty pie charts pulling errors
        if sum(counts) > 0:
            wedges, texts, autotexts = ax1.pie(
                counts, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=140,
                textprops={'color': TEXT_COLOR, 'fontsize': 10},
                wedgeprops={'edgecolor': PLOT_CARD_BG, 'linewidth': 2}
            )
            for at in autotexts:
                at.set_color('#0b0f1a')
                at.set_fontweight('bold')
        ax1.set_title("Sequence Class Split", color='#94a3b8', fontsize=11)
        st.pyplot(fig1, clear_figure=True)
        plt.close()

        st.markdown("**Prediction vs Actual Timeline (Latest 80 + Forecast)**")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor(BG_COLOR)
        ax2.set_facecolor(PLOT_CARD_BG)

        # ─────────────────────────────
        # DATA
        # ─────────────────────────────
        trace_preds = preds[-80:]
        x = np.arange(len(trace_preds))

        # Actual (lagged or replace with real)
        actual = np.roll(trace_preds, 1)
        actual[0] = trace_preds[0]

        y_pred = np.array(trace_preds, dtype=float)
        y_actual = np.array(actual, dtype=float)

        # ─────────────────────────────
        # 🔮 FUTURE PREDICTION (next 10 steps)
        # ─────────────────────────────
        future_steps = 40

        # Simple forecasting: trend-based (you can upgrade later to LSTM rolling)
        last_trend = y_pred[-5:]  # recent trend
        future = []

        for i in range(future_steps):
            next_val = np.mean(last_trend)  # average trend
            next_val = np.clip(round(next_val), 0, 2)  # keep in [0,2]
            future.append(next_val)

            # update trend window
            last_trend = np.append(last_trend[1:], next_val)

        future = np.array(future)

        # Extend prediction
        y_pred_extended = np.concatenate([y_pred, future])
        x_extended = np.arange(len(y_pred_extended))

        # ─────────────────────────────
        # SMOOTHING
        # ─────────────────────────────
        def smooth(y, window=7):
            return np.convolve(y, np.ones(window)/window, mode='same')

        y_pred_smooth = smooth(y_pred_extended)
        y_actual_smooth = smooth(y_actual)

        # ─────────────────────────────
        # OFFSET (visual separation)
        # ─────────────────────────────
        y_pred_smooth += 0.12
        y_actual_smooth -= 0.12

        # ─────────────────────────────
        # ✨ PLOT
        # ─────────────────────────────

        # Actual (only till 80)
        ax2.plot(x, y_actual_smooth,
                color="#22c55e",
                linewidth=2.8,
                label="Actual Traffic")

        # Prediction (extended to 90)
        ax2.plot(x_extended, y_pred_smooth,
                color="#ef4444",
                linewidth=2.8,
                label="Predicted Traffic")

        # ─────────────────────────────
        # OPTIONAL: highlight forecast region 🔥
        # ─────────────────────────────
        ax2.axvspan(len(x)-1, len(x_extended),
                    color='#ef4444', alpha=0.08)

        # ─────────────────────────────
        # AXIS
        # ─────────────────────────────
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Low', 'Medium', 'High'], color=TEXT_COLOR)

        ax2.set_xlim(0, len(x_extended))

        ax2.set_xlabel("Recent Sequence Index", color=SUBTEXT_COLOR)
        ax2.set_ylabel("Traffic Level", color=SUBTEXT_COLOR)

        ax2.tick_params(colors=SUBTEXT_COLOR)

        # Style
        ax2.spines['bottom'].set_color(PLOT_BORDER_COLOR)
        ax2.spines['left'].set_color(PLOT_BORDER_COLOR)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        ax2.legend(facecolor=PLOT_CARD_BG, edgecolor='none', labelcolor=TEXT_COLOR)

        st.pyplot(fig2)
    # ── Probability Heatmap ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Probability Heatmap (Latest 60 sequences)</div>', unsafe_allow_html=True)

    fig5, ax5 = plt.subplots(figsize=(12, 2.5))
    fig5.patch.set_facecolor(BG_COLOR)
    ax5.set_facecolor(BG_COLOR)
    prob_sample = probs[-60:].T
    sns.heatmap(prob_sample, ax=ax5, cmap="YlOrRd", cbar=True,
                xticklabels=5, yticklabels=["Low", "Med", "High"],
                linewidths=0.3, linecolor=BG_COLOR)
    ax5.set_xlabel("Recent Sequence Index", color=SUBTEXT_COLOR)
    ax5.tick_params(colors=TEXT_COLOR)
    ax5.set_title("Class Probabilities Flow", color=SUBTEXT_COLOR)
    st.pyplot(fig5, clear_figure=True)
    plt.close()

    # ── Packet Length Distribution ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Packet Length Distribution</div>', unsafe_allow_html=True)

    fig6, ax6 = plt.subplots(figsize=(10, 3))
    fig6.patch.set_facecolor(BG_COLOR)
    ax6.set_facecolor(PLOT_CARD_BG)
    ax6.hist(df_raw['Length'].dropna(), bins=60, color='#818cf8', edgecolor=BG_COLOR, alpha=0.85)
    ax6.set_xlabel("Packet Length (bytes)", color=SUBTEXT_COLOR)
    ax6.set_ylabel("Count", color=SUBTEXT_COLOR)
    ax6.tick_params(colors=SUBTEXT_COLOR)
    ax6.spines['bottom'].set_color(PLOT_BORDER_COLOR)
    ax6.spines['left'].set_color(PLOT_BORDER_COLOR)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    st.pyplot(fig6, clear_figure=True)
    plt.close()

# ─────────────────────────────────────────
# GLOBAL AUTOREFRESH LOOP
# ─────────────────────────────────────────
# Placed at the very end so Streamlit successfully renders Tab 1 and Tab 2
# in the UI before it halts execution to rerun itself!
if st.session_state.get('capturing', False):
    time.sleep(2.5) # Increased to 2.5s to prevent Matplotlib 'MediaFileStorageError' race conditions
    st.rerun()
