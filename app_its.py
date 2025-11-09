import streamlit as st
st.set_page_config(page_title="ITS - Pothole Detection", layout="wide")

import streamlit as st
from ultralytics import YOLO
import tempfile
import json
import firebase_admin
from firebase_admin import credentials, storage, firestore
import datetime
import os

# --------------------------
# 1️⃣ Firebase setup
# --------------------------
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

# --------------------------
# 2️⃣ Streamlit UI
# --------------------------
st.set_page_config(page_title="ITS - Pothole Detection", layout="wide")
st.title("🚗 Pothole Detection System (YOLOv8 - Streamlit, Firebase)")
st.write("Upload a road video and watch potholes detected live — just like a camera feed!")

uploaded_video = st.file_uploader("🎥 Upload road video", type=["mp4", "mov", "avi", "mkv"])
conf = st.slider("Detection Confidence", 0.1, 1.0, 0.4, 0.05)

# --------------------------
# 3️⃣ Cache YOLO model
# --------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --------------------------
# 4️⃣ Efficient YOLO stream processing
# --------------------------
if uploaded_video is not None:
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    stframe = st.empty()
    fps_display = st.empty()
    st.info("🔍 Detecting potholes... Please wait.", icon="⚙️")

    pothole_count = 0
    frame_count = 0

    # ✅ Stream mode — faster, smoother
    for r in model.predict(video_path, conf=conf, stream=True, verbose=False):
        annotated_frame = r.plot()
        pothole_count += len(r.boxes)
        frame_count += 1
        stframe.image(annotated_frame, channels="BGR", use_column_width=True)
        fps_display.markdown(f"**Processed frames:** {frame_count}")

    st.success(f"✅ Detection complete! {pothole_count} potholes found in {frame_count} frames.")

    # --------------------------
    # 5️⃣ Upload to Firebase
    # --------------------------
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
