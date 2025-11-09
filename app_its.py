import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import time

st.set_page_config(page_title="Pothole Detection ITS", layout="wide")

st.title("🚗 Pothole Detection System (YOLOv8 - Streamlit)")
st.write("Upload a road video and watch potholes detected live — just like a camera feed!")

# Upload video
uploaded_video = st.file_uploader("🎥 Upload road video", type=["mp4", "mov", "avi", "mkv"])

# Confidence slider
conf = st.slider("Detection Confidence", 0.1, 1.0, 0.4, 0.05)

if uploaded_video is not None:
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    # Load YOLO model
    model = YOLO("best.pt")

    # Open video with OpenCV
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    fps_display = st.empty()
    prev_time = 0

    st.info("Detecting potholes... Press Stop to end.", icon="⚙️")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Resize for speed
        frame = cv2.resize(frame, (640, 360))

        # YOLO inference (stream-like)
        results = model.predict(frame, conf=conf, verbose=False)

        # Draw boxes
        annotated_frame = results[0].plot()

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        # Convert frame for Streamlit
        stframe.image(annotated_frame, channels="BGR", use_column_width=True)
        fps_display.markdown(f"**FPS:** {fps:.2f}")

    cap.release()
    st.success("✅ Detection complete!")
else:
    st.info("👆 Please upload a video to start detection.")
