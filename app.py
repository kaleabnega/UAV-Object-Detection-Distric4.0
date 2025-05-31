
import streamlit as st
import pandas as pd
import cv2
import time
from collections import deque
from video_stream import get_video_stream

# ─── Page Config ─────────────────────────────
st.set_page_config(page_title="Live Detection Dashboard", layout="wide")
st.title("📡 Real-time Object & Vehicle Detection")

# ─── Placeholders ────────────────────────────
frame_placeholder = st.empty()
chart_placeholder = st.empty()

# ─── History for Chart ───────────────────────
VEHICLES = ["car", "bus", "truck", "motorcycle", "bicycle"]

history = deque(maxlen=30)  # keep last 30 intervals
last_chart_update = time.time()

st.markdown("**Detected vehicle counts over time**")

# ─── Stream & Display ───────────────────────
for frame, counts in get_video_stream():
    # Display video frame
    frame_placeholder.image(frame, channels="RGB", use_container_width=True)

    # Build a record for this interval
    record = {v: counts.get(v, 0) for v in VEHICLES}
    record["timestamp"] = pd.Timestamp.now()
    history.append(record)

    # Update chart every 2 seconds
    if time.time() - last_chart_update > 2:
        df = pd.DataFrame(list(history)).set_index("timestamp")
        chart_placeholder.line_chart(df)
        last_chart_update = time.time()
