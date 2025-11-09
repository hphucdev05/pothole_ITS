import streamlit as st
import tempfile
import os
import cv2
import json
import firebase_admin
from firebase_admin import credentials, storage
from ultralytics import YOLO
import time
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="ITS - Pothole Detection", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)
speed_boost = st.sidebar.toggle("⚡ Boost speed (skip frames)", value=True)

# --- FIREBASE INIT ---
if "firebase" not in st.session_state:
    try:
        firebase_key = json.loads(st.secrets["FIREBASE_KEY"])
        cred = credentials.Certificate(firebase_key)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'storageBucket': f"{firebase_key['project_id']}.appspot.com"
            })
        st.session_state["firebase"] = True
        st.sidebar.success("✅ Firebase connected")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Firebase chưa cấu hình đúng: {e}")

# --- YOLO MODEL ---
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    model.to("cpu")  # Force CPU mode
    return model

model = load_model()

# --- MAIN ---
st.title("🕳️ Pothole Detection (Optimized CPU Version)")
uploaded_video = st.file_uploader("📹 Upload a road video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    frame_placeholder = st.empty()
    progress = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    detected_count = 0

    start_time = time.time()

    st.info("🚀 Detecting potholes... Please wait.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Skip frames for speed boost
        if speed_boost and frame_count % 2 != 0:
            continue

        # Resize smaller for faster inference
        frame_resized = cv2.resize(frame, (480, 270))

        # Predict
        results = model.predict(frame_resized, conf=confidence, imgsz=480, verbose=False)
        annotated = results[0].plot()

        # Count potholes
        detected_count += len(results[0].boxes)

        # Show every 3 frames (reduce Streamlit UI lag)
        if frame_count % 3 == 0:
            frame_placeholder.image(annotated, channels="BGR", use_column_width=True)

        progress.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    end_time = time.time()
    elapsed = end_time - start_time

    st.success(f"✅ Detection complete! {detected_count} potholes found in {frame_count} frames.")
    st.write(f"⏱️ Processing time: {elapsed:.1f}s (~{frame_count/elapsed:.1f} FPS)")

    # Upload summary to Firebase (optional)
    if "firebase" in st.session_state:
        try:
            bucket = storage.bucket()
            blob = bucket.blob(f"reports/{uploaded_video.name}_summary.txt")
            blob.upload_from_string(
                f"Potholes detected: {detected_count}\nFrames processed: {frame_count}\nFPS: {frame_count/elapsed:.1f}",
                content_type="text/plain"
            )
            st.success("📤 Uploaded detection summary to Firebase Storage!")
        except Exception as e:
            st.warning(f"⚠️ Firebase upload failed: {e}")

    os.remove(tfile.name)
