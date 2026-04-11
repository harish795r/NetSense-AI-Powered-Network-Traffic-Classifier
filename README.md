# 🌐 NetSense — AI-Powered Network Traffic Classifier

> Real-time TCP/UDP network traffic classification using a stacked LSTM neural network, with live TCP AIMD congestion control simulation.

---

## 📌 Project Overview

**NetSense** is a Streamlit web application that monitors network traffic in real time (or from a recorded `.pcap` file), classifies it as **Low / Medium / High** congestion using a trained **PyTorch LSTM** model, and visually simulates how **TCP's AIMD congestion control** algorithm responds to those predictions.

Built as a Computer Networks course project by students of **VIT Chennai**, guided by **Dr. Swaminathan A**.

---

## 👥 Team

| Name | Roll No | GitHub |
|------|---------|--------|
| Kanika Rathore | 24BYB1080 | [@kanika-flow](https://github.com/kanika-flow) |
| R Harish | 24BYB1159 | [@harish795r](https://github.com/harish795r) |
| Akshaya H | 24BYB1124 | [@akshaya040806](https://github.com/akshaya040806) |

**Faculty Guide:** Dr. Swaminathan A — Faculty, Computer Networks

---

## ✨ Features

- **Live packet capture** via Scapy — sniffs real network traffic on your machine
- **PCAP file upload** — analyse pre-recorded Wireshark captures offline
- **LSTM inference** — 2-layer stacked PyTorch LSTM classifies every 10-packet window
- **TCP AIMD simulation** — congestion window (cwnd) grows and collapses in real time based on AI predictions
- **Animated traffic pipe** — visual indicator of current congestion state with colour-coded glow
- **Dashboard tab** — AIMD graph, probability heatmap, class distribution pie, smoothed traffic timeline, packet length histogram

---

## 🗂️ Project Structure

```
cnproject/
├── app.py                      # Main Streamlit application
├── app_test1.py                # Earlier prototype / test version
├── train.py                    # LSTM model training script
├── tcp_udp_lstm_pytorch.pt     # Trained model weights (PyTorch)
├── output1.csv                 # Primary training dataset (394k packets)
├── perfect_traffic.csv         # Synthetic clean traffic CSV 
├── wireshark_sample.csv        # Sample Wireshark export for testing
├── test_packets.csv            # Additional test packet CSV 
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9 or higher
- pip
- On Linux/macOS: root/sudo access (required for live packet capture)
- On Windows: [Npcap](https://npcap.com/) installed, terminal run as Administrator

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd cnproject
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
```
streamlit>=1.32.0
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.13.0
scapy>=2.5.0
```

### 3. Ensure the model file is present

The trained model `tcp_udp_lstm_pytorch.pt` must be in the same directory as `app.py`. If it's missing, retrain it:

```bash
python train.py
```

---

## 🚀 Running the App

### Live Capture Mode (requires sudo)

```bash
sudo streamlit run app.py
```

> **Why sudo?** Scapy needs raw socket access to sniff network packets. Without root privileges, live capture will fail.

### PCAP Upload Mode (no sudo needed)

```bash
streamlit run app.py
```

Then upload a `.pcap` or `.pcapng` file in the app.

---

## 🖥️ How to Use

### Path A — Live Network Capture

1. Run with `sudo streamlit run app.py`
2. In the **sidebar**, confirm the Model Path is `tcp_udp_lstm_pytorch.pt`
3. Click the **Live Monitoring** tab → select **🔴 Live Capture**
4. Click **▶ Start Live Capture**
5. Wait ~5 seconds for the first 10-packet sequence to accumulate
6. Watch the **traffic pipe**, **status badge**, and **metric cards** update automatically
7. Switch to the **Dashboard** tab for the AIMD graph and detailed charts
8. Click **⏹ Stop Capture** when done

### Path B — PCAP File Upload

1. Run with `streamlit run app.py` (no sudo needed)
2. Go to **Live Monitoring** tab → select **📂 Upload PCAP File**
3. Upload your `.pcap` or `.pcapng` file
4. Wait for the `✅ PCAP processed!` confirmation
5. Switch to the **Dashboard** tab to explore all charts

---

## 🧠 Model Architecture

The classifier is a **2-layer stacked LSTM** built with PyTorch:

```
Input: (batch, 10 timesteps, 5 features)
    ↓
LSTM Layer 1  — hidden size: 64
    ↓
Dropout (p=0.3)
    ↓
LSTM Layer 2  — hidden size: 32
    ↓
Dropout (p=0.2)  [applied to last timestep only]
    ↓
Linear (32 → 32) + ReLU
    ↓
Linear (32 → 3)
    ↓
Output: [P(Low), P(Medium), P(High)]
```

**Training details:**
- Dataset: `output1.csv` — ~394,000 real network packets
- Labels: assigned by tertile split on `packet_rate` (33rd / 66th percentile → Low / Medium / High)
- Train/test split: 80/20, stratified
- Optimiser: Adam (lr=0.001)
- Loss: CrossEntropyLoss with class-weight balancing
- Epochs: 15
- Batch size: 32
- Random seed: 42

---

## 📊 Input Features

For each packet, 5 features are extracted and grouped into overlapping windows of 10:

| Feature | Description |
|---------|-------------|
| `packet_count` | Always 1 per row — marks each individual packet |
| `avg_size` | Raw packet length in bytes |
| `size_variation` | Difference in length from the previous packet |
| `packet_rate` | Rolling 2-packet sum of lengths (traffic intensity proxy) |
| `rate_change` | Change in `packet_rate` between consecutive packets |

All features are normalised to **[0, 1]** using `MinMaxScaler` before inference.

---

## 🔧 TCP Congestion Control Simulation

NetSense simulates TCP's **AIMD (Additive Increase, Multiplicative Decrease)** algorithm based on each LSTM prediction:

| Prediction | Congestion Event Simulated | cwnd Behaviour |
|------------|---------------------------|----------------|
| 🔵 Low (0) | Clear — no congestion | Slow Start if `cwnd < ssthresh`, else +1 (Congestion Avoidance) |
| 🟡 Medium (1) | 3 Duplicate ACKs | `ssthresh = cwnd/2`, `cwnd = ssthresh` (Fast Retransmit) |
| 🔴 High (2) | Timeout | `ssthresh = cwnd/2`, `cwnd = 1` (full reset → Slow Start) |

The **Dashboard tab** plots the last 40 predictions as a live AIMD sawtooth graph, annotating each congestion event.

---

## 📁 Training Your Own Model

If you want to retrain on your own data:

1. Prepare a CSV with columns: `Timestamp`, `Protocol`, `Length`
2. Replace `output1.csv` with your file (update the filename in `train.py` line 26)
3. Run:

```bash
python train.py
```

The script will:
- Engineer 5 features from your data
- Auto-label using tertile splits on `packet_rate`
- Train the LSTM for 15 epochs
- Print a classification report + confusion matrix
- Save `tcp_udp_lstm_pytorch.pt` in the current directory

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Web framework | Streamlit ≥ 1.32 |
| Deep learning | PyTorch ≥ 2.0 |
| Packet capture | Scapy ≥ 2.5 |
| Data processing | Pandas, NumPy, scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| UI styling | Custom CSS (glassmorphism, Space Mono font) |

---

## ❓ Troubleshooting

**Live capture shows no data / "waiting for packets"**
→ Make sure you ran with `sudo`. Check that Scapy is installed: `pip install scapy`

**Model not found error**
→ Ensure `tcp_udp_lstm_pytorch.pt` is in the same folder as `app.py`, or update the Model Path in the sidebar.

**All predictions are the same class**
→ Normal if your network is consistently idle (all Low) or busy (all High). Try generating mixed traffic — browse the web, run a download, and watch predictions shift.

**Windows: permission error on capture**
→ Install [Npcap](https://npcap.com/) and run your terminal as Administrator.

**PCAP parse returns empty dataframe**
→ NetSense only processes packets with an IP layer. Make sure your PCAP contains IP-level traffic (not pure Ethernet or loopback-only captures).



---

*NetSense · VIT Chennai · Computer Networks Project*
