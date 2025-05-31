# UAV-Based Traffic Monitoring with Secure Data Transmission

## 📌 Overview

This project leverages a UAV (Unmanned Aerial Vehicle) to detect, count, and transmit vehicle data in real-time. Using on-device object detection and a secure MQTT communication channel with TLS, the system ensures low-latency, encrypted transmission of vehicle counts to a ground station dashboard.

## 🎯 Objectives

- Real-time vehicle detection using an edge-optimized YOLOv8 model (TFLite + INT8).
- Low-bandwidth communication using MQTT protocol.
- Secure transmission using TLS with fallback protocol switching for latency optimization.
- Dashboard visualization of traffic statistics using Streamlit.

## 🧠 Key Features

- ✅ **On-Device Inference:** Object detection runs on UAV, reducing reliance on cloud infrastructure.
- ✅ **TLS Secured MQTT:** Data is transmitted via MQTT with TLS authentication and encryption.
- ✅ **Adaptive Security Layer:** System dynamically chooses between TLS and lightweight fallback mode.
- ✅ **Streamlit Dashboard:** Easy-to-use web dashboard to view traffic data in real-time.

## 🔐 Security Architecture

We utilize:
- TLS v1.2 with Mosquitto broker for encrypted communication.
- Mutual certificate-based authentication.
- Adaptive fallback to authentication-only mode when resources are constrained (CPU/load/battery), reducing overhead.

## 📦 Folder Structure

```
UAV-Object-Detection-District4.0/
├── app.py # Streamlit dashboard
├── video_stream.py # Main object detection + MQTT code
├── yolov8n_saved_model/ # INT8 TFLite model
├── labels.txt # Class labels
```


## ⚙️ Setup Instructions

### Prerequisites

- Python 3.10+
- `paho-mqtt`, `opencv-python`, `numpy`, `streamlit`, `tensorflow` (lite)

### 1. Install Dependencies

#### ⚙️ Installation & Setup
First, clone the repository and navigate into it:

```bash

git clone https://github.com/kaleabnega/UAV-Object-Detection-District4.0.git
cd UAV-Object-Detection-District4.0
````

#### Set up a virtual environment (recommended):

```bash

python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
````

```bash
pip install -r requirements.txt
```

### 2. Run the Mosquitto Broker (locally)
Make sure mosquitto is installed and running with tls.conf configuration.
### 3. Run the Streamlit App
```bash
streamlit run app.py
```