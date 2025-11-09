import streamlit as st
import cv2, tempfile, json, os, datetime, time
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, storage, firestore
from concurrent.futures import ThreadPoolExecutor

# ----------------------------------------------------
# 1️⃣ Cấu hình cơ bản
# ----------------------------------------------------
st.set_page_config(page_title="ITS - Pothole Detection", layout="wide")
st.title("🚗 Pothole Detection (YOLOv8 - 8-Core CPU Optimized)")
st.write("⚙️ Optimized for CPU - parallel inference, skip frames, and low latency display.")

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
# 4️⃣ Load YOLO model (nhẹ và ép CPU)
# ----------------------------------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # model nhẹ nhất
    model.to("cpu")
    return model

model = load_model()

cv2.setUseOptimized(True)
cv2.setNumThreads(8)  # full 8 core

# ----------------------------------------------------
# 5️⃣ Ultra-fast Detection Loop (multi-threaded)
# ----------------------------------------------------
if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    fps_box = st.empty()
    info_box = st.empty()

    skip = 2           # skip every 2 frames
    frame_count = pothole_count = 0
    start = time.time()

    executor = ThreadPoolExecutor(max_workers=4)  # parallel inference
    futures = []

    def process_frame(frame):
        results = model.predict(frame, conf=conf, imgsz=320, verbose=False)[0]
        return results.plot(), len(results.boxes)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip != 0:
            continue

        frame_resized = cv2.resize(frame, (640, 360))
        futures.append(executor.submit(process_frame, frame_resized))

        # Giới hạn batch song song tránh nghẽn CPU
        if len(futures) > 2:
            done = futures.pop(0).result()
            annotated_frame, detections = done
            pothole_count += detections
            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

            fps = frame_count / (time.time() - start)
            fps_box.markdown(f"**⚡ FPS:** {fps:.2f}")
            info_box.info(f"Detected potholes: {pothole_count}")

    cap.release()
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
