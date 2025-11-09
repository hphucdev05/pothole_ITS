import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import json
import firebase_admin
from firebase_admin import credentials, storage, firestore
import datetime, os, time

# ----------------------------------------------------
# 1️⃣ Setup Streamlit sớm nhất
# ----------------------------------------------------
st.set_page_config(page_title="ITS - Pothole Detection", layout="wide")

st.title("🚗 Pothole Detection (YOLOv8 - Streamlit, CPU Optimized)")
st.write("Upload a road video and watch potholes detected live — optimized for CPU ⚙️")

# ----------------------------------------------------
# 2️⃣ Firebase (optional)
# ----------------------------------------------------
if "firebase" not in st.session_state:
    try:
        firebase_key = json.loads(st.secrets["FIREBASE_KEY"])
        cred = credentials.Certificate(firebase_key)
        firebase_admin.initialize_app(cred, {
            'storageBucket': f"{firebase_key['project_id']}.appspot.com"
        })
        st.session_state["firebase"] = True
        st.sidebar.success("Firebase connected ✅")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Firebase chưa cấu hình đúng: {e}")

# ----------------------------------------------------
# 3️⃣ Upload + Confidence
# ----------------------------------------------------
uploaded_video = st.file_uploader("🎥 Upload road video", type=["mp4", "mov", "avi", "mkv"])
conf = st.slider("Detection Confidence", 0.1, 1.0, 0.4, 0.05)

# ----------------------------------------------------
# 4️⃣ Load YOLO model (cache + force CPU)
# ----------------------------------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    model.to("cpu")
    return model

model = load_model()

# OpenCV optimization
cv2.setUseOptimized(True)
cv2.setNumThreads(2)

# ----------------------------------------------------
# 5️⃣ Detection loop (ultra-turbo)
# ----------------------------------------------------
if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    stframe = st.empty()
    fps_box = st.empty()
    info_box = st.empty()

    skip = 2   # skip every 2 frames
    frame_count = pothole_count = 0
    start = time.time()

    # Stream detection (no full load)
    for r in model.predict(video_path, conf=conf, stream=True, imgsz=320, verbose=False):
        frame_count += 1
        if frame_count % skip != 0:
            continue

        annotated_frame = r.plot()
        pothole_count += len(r.boxes)

        # Resize for display (faster render)
        annotated_frame = cv2.resize(annotated_frame, (480, 270))
        stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        fps = frame_count / (time.time() - start)
        fps_box.markdown(f"**⚡ FPS:** {fps:.2f}")
        info_box.info(f"Detected potholes: {pothole_count}")

    st.success(f"✅ Done! {pothole_count} potholes found in {frame_count} frames.")

    # ----------------------------------------------------
    # 6️⃣ Upload video + metadata to Firebase (optional)
    # ----------------------------------------------------
    try:
        bucket = storage.bucket()
        blob = bucket.blob(f"videos/{os.path.basename(video_path)}")
        blob.upload_from_filename(video_path)
        blob.make_public()
        video_url = blob.public_url

        db = firestore.client()
        db.collection("detections").add({
            "filename": os.path.basename(video_path),
            "potholes_detected": pothole_count,
            "frames": frame_count,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "video_url": video_url
        })

        st.success(f"☁️ Uploaded to Firebase! [Open video]({video_url})")
    except Exception as e:
        st.warning(f"Firebase upload failed: {e}")

    os.unlink(video_path)
else:
    st.info("👆 Please upload a video to start detection.")
