import streamlit as st
import cv2
from ultralytics import YOLO
import base64
import subprocess
from pathlib import Path
import tempfile
import yaml
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
WEIGHTS_DIR = BASE_DIR / "weights"
PRED_DIR = BASE_DIR / "predictions_web"
IMG_OUT = PRED_DIR / "images"
LBL_OUT = PRED_DIR / "labels"
YAML_PATH = BASE_DIR / "yolo_params.yaml"

for directory in [PRED_DIR, IMG_OUT, LBL_OUT]:
    directory.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="SPACESIGHT", layout="wide", page_icon="🚀")


components.html("""
<!DOCTYPE html>
<html>
<head>
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;800&display=swap');
body {
  background: transparent;
  text-align: center;
  margin: 0;
  padding: 0;
  color: white;
  font-family: 'Poppins', sans-serif;
}
.main-title {
  font-size: 56px;
  font-weight: 800;
  background: linear-gradient(90deg, #00b4d8, #0077b6, #90e0ef, #caf0f8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 10px;
  animation: glowPulse 3s infinite alternate;
}
.sub-text {
  font-size: 30px;
  font-weight: 600;
  color: #ffffff;
  border-right: 3px solid #00b4d8;
  display: inline-block;
  white-space: nowrap;
  overflow: hidden;
}
@keyframes glowPulse {
  from { text-shadow: 0 0 10px rgba(0,180,216,0.5); }
  to { text-shadow: 0 0 25px rgba(0,180,216,1); }
}
</style>
</head>
<body>
  <h1 class="main-title">🚀 SPACESIGHT</h1>
  <h2 id="text" class="sub-text"></h2>

<script>
const words = [
  "is Intelligent 👁️",
  "is Real-Time 🚀",
  "is Adaptive 🧠",
  "is Accurate 🎯",
  "is Built for Space 🌌"
];
let wordIndex = 0;
let charIndex = 0;
let currentWord = "";
let deleting = false;
const textEl = document.getElementById("text");

function typeEffect() {
  currentWord = words[wordIndex];
  
  if (!deleting) {
    textEl.textContent = currentWord.substring(0, charIndex + 1);
    charIndex++;
    
    if (charIndex === currentWord.length) {
      deleting = true;
      setTimeout(typeEffect, 1200); // pause before deleting
      return;
    }
  } else {
    textEl.textContent = currentWord.substring(0, charIndex - 1);
    charIndex--;
    
    if (charIndex === 0) {
      deleting = false;
      wordIndex = (wordIndex + 1) % words.length;
    }
  }
  
  const speed = deleting ? 50 : 100;
  setTimeout(typeEffect, speed);
}

typeEffect();
</script>
</body>
</html>
""", height=250)


st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3 {
    text-align: center;
    font-weight: 800;
    text-shadow: 1px 1px 4px rgba(0,0,0,0.4);
}
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #1e293b, #0f172a);
    color: white;
    border-right: 2px solid rgba(255,255,255,0.1);
}
button, .stDownloadButton>button, .stButton>button {
    background: linear-gradient(135deg, #00b4d8, #0077b6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6em 1.2em;
    font-weight: 600;
    box-shadow: 0 5px 20px rgba(0, 119, 182, 0.4);
    transition: all 0.2s ease-in-out;
}
button:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #0077b6, #00b4d8);
}
.stFileUploader>div>div {
    background-color: rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px;
}
[data-testid="stMetricValue"] {
    color: #00b4d8;
    text-shadow: 0 0 8px rgba(0,180,216,0.6);
}
.stVideo {
    border-radius: 16px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.5);
    margin-bottom: 1em;
}
hr {
    border: 1px solid rgba(255,255,255,0.2);
    margin-top: 20px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

video_path = ASSETS_DIR / "video.mp4"
if video_path.exists():
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_b64 = base64.b64encode(video_bytes).decode()
    st.markdown(f"""
    <div style="display:flex; justify-content:center; align-items:center;">
        <video autoplay loop muted playsinline style="
            width:100%;
            max-width:1200px;
            border-radius:20px;
            box-shadow:0 6px 25px rgba(0,0,0,0.6);
            margin-bottom:1.5em;
            object-fit: cover;">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("🚨 Header video not found in assets folder!")

st.markdown("""
<h1>🚀 SPACESIGHT</h1>
<h2 style="color:#00b4d8;">Intelligent Object Detection and Tracking in Space Station Environments</h2>
<hr>
""", unsafe_allow_html=True)


AVAILABLE_MODELS = {
    "YOLOv8n (Nano)": WEIGHTS_DIR / "yolov8n.pt",
    "YOLOv8s (Small)": WEIGHTS_DIR / "yolov8s.pt",
    "YOLOv8m (Medium)": WEIGHTS_DIR / "yolov8m.pt",
    "YOLOv8l (Large)": WEIGHTS_DIR / "yolov8l.pt",
    "YOLOv8x (X-Large)": WEIGHTS_DIR / "yolov8x.pt",
    "Custom Model (DeepSORT Tracking)": WEIGHTS_DIR / "best.pt",
}

st.sidebar.header("🧠 Model Settings")
model_name = st.sidebar.selectbox("Select YOLO Model", list(AVAILABLE_MODELS.keys()))
model_path = AVAILABLE_MODELS[model_name]

if not model_path.exists():
    st.sidebar.error(f"⚠️ Model not found: {model_path}")
    st.stop()

model = YOLO(str(model_path))
tracker = DeepSort(max_age=30)  


task = st.sidebar.radio("Select Task", [
    "Batch Upload",
    "Video Inference",
    "Webcam Inference (DeepSORT Tracking)",
    "Test Set Browser",
    "Validation Metrics"
])


def save_predictions(result, filename):
    img_save = IMG_OUT / filename
    cv2.imwrite(str(img_save), result.plot())
    lbl_save = LBL_OUT / Path(filename).with_suffix(".txt")
    with open(lbl_save, 'w') as f:
        for box in result.boxes:
            cls_id = int(box.cls)
            x, y, w, h = box.xywh[0].tolist()
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    return img_save, lbl_save



if task == "Batch Upload":
    st.subheader("📸 Batch Image Detection")
    files = st.file_uploader("Upload one or more images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if files:
        for f in files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix)
            tmp.write(f.read())
            tmp.close()
            results = model.predict(source=tmp.name, conf=0.5)
            res = results[0]
            annotated = res.plot()
            st.image(annotated, caption=f.name, use_container_width=True)
            img_file, lbl_file = save_predictions(res, f.name)


elif task == "Webcam Inference (DeepSORT Tracking)":
    st.subheader("🎥 Live Object Tracking with Relative Position & Distance (YOLOv8 + DeepSORT)")

    reference_class = st.selectbox(
        "Select Reference Object (for relative positions):",
        ["person", "helmet", "oxygen_tank", "toolbox", "glove", "bottle"]
    )

    start = st.button("🚀 Start Tracking")
    stop = st.button("⛔ Stop")
    placeholder = st.empty()

    tracker = DeepSort(max_age=30)  
    class_colors = {}

    if start:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("⚠️ Cannot open webcam.")
        else:
            st.info("Tracking started... Press Stop to end.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break


                results = model(frame)
                res = results[0]
                detections = []

                for box in res.boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = box
                    w, h = x2 - x1, y2 - y1
                    label = res.names[int(cls)]
                    detections.append(([x1, y1, w, h], conf, label))

                tracks = tracker.update_tracks(detections, frame=frame)
                annotated = res.plot()

                tracked_objects = []
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    l, t, r, b = map(int, track.to_ltrb())
                    label = track.get_det_class() or "object"
                    cx, cy = int((l + r) / 2), int((t + b) / 2)

                    if label not in class_colors:
                        class_colors[label] = tuple(np.random.randint(0, 255, 3).tolist())
                    color = class_colors[label]

                    cv2.rectangle(annotated, (l, t), (r, b), color, 2)
                    cv2.circle(annotated, (cx, cy), 4, color, -1)
                    cv2.putText(annotated, f"{label} | ID:{track_id}", (l, t - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    tracked_objects.append({
                        "id": track_id,
                        "label": label,
                        "center": (cx, cy)
                    })

                ref_center = None
                for obj in tracked_objects:
                    if obj["label"] == reference_class:
                        ref_center = obj["center"]
                        break

                if ref_center:
                    x_ref, y_ref = ref_center
                    for obj in tracked_objects:
                        if obj["label"] != reference_class:
                            x, y = obj["center"]
                            dx, dy = x - x_ref, y - y_ref
                            distance = int(np.sqrt(dx ** 2 + dy ** 2))
                            horiz = "right" if dx > 30 else "left" if dx < -30 else "center"
                            vert = "below" if dy > 30 else "above" if dy < -30 else "same level"

                            rel_text = f"{obj['label']} ({obj['id']}): {distance}px {horiz} & {vert} of {reference_class}"
                            cv2.putText(annotated, rel_text, (20, 40 + 25 * tracked_objects.index(obj)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)


                            cv2.line(annotated, (x_ref, y_ref), (x, y),
                                     (0, 255, 255) if distance > 200 else (0, 0, 255), 2)

                placeholder.image(annotated, channels="BGR", use_column_width=True)
                if stop:
                    st.info("⛔ Tracking stopped.")
                    break

            cap.release()
            cv2.destroyAllWindows()



elif task == "Test Set Browser":
    st.subheader("🧪 Test Set Browser")
    if not YAML_PATH.exists():
        st.error("⚠️ yolo_params.yaml not found.")
    else:
        cfg = yaml.safe_load(open(YAML_PATH))
        img_dir = Path(cfg.get("test", "")) / "images"
        if not img_dir.exists():
            st.error("Test images folder missing.")
        else:
            imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            if imgs:
                idx = st.slider("Select Image", 0, len(imgs)-1, 0)
                img_path = imgs[idx]
                st.write(f"📷 {img_path.name}")
                res = model.predict(source=str(img_path), conf=0.5)[0]
                st.image(res.plot(), use_container_width=True)


elif task == "Validation Metrics":
    st.subheader("📊 Model Validation Metrics")
    if not YAML_PATH.exists():
        st.error("⚠️ yolo_params.yaml not found.")
    else:
        st.info("Running validation... please wait ⏳")
        metrics = model.val(data=str(YAML_PATH), split="test").box
        names = ["Precision", "Recall", "mAP50", "mAP50-95"]
        values = [float(v[0]) if hasattr(v, "__len__") else float(v)
                  for v in [metrics.p, metrics.r, metrics.map50, metrics.map]]
        col1, col2 = st.columns([1, 2])
        with col1:
            for name, val in zip(names, values):
                st.metric(label=name, value=f"{val:.3f}")
        with col2:
            fig, ax = plt.subplots()
            ax.bar(names, values, color="#00b4d8")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score")
            ax.set_title("Validation Metrics")
            st.pyplot(fig)


elif task == "Video Inference":
    st.subheader("🎬 Video Object Detection + 3D Scene Reconstruction")

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        suffix = Path(uploaded_video.name).suffix
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(uploaded_video.read())
        tfile.flush()
        tfile.close()

        st.info("🔍 Running YOLO detection...")
        output_dir = PRED_DIR / "video_pred"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = model.predict(
            source=tfile.name, conf=0.5, save=True,
            project=str(output_dir.parent), name=output_dir.name,
            exist_ok=True, vid_stride=2
        )

        pred_dir = Path(results[0].save_dir)
        pred_videos = list(pred_dir.rglob("*.mp4")) + list(pred_dir.rglob("*.avi"))
        if not pred_videos:
            st.error("⚠️ No output video generated.")
        else:
            final_output_path = pred_videos[0]
            with open(final_output_path, "rb") as vf:
                vb64 = base64.b64encode(vf.read()).decode()
            st.markdown(f"""
            <video autoplay loop muted playsinline style="
                width:100%;
                border-radius:16px;
                box-shadow:0 4px 20px rgba(0,0,0,0.4);
                margin-top:1em;">
                <source src="data:video/mp4;base64,{vb64}" type="video/mp4">
            </video>
            """, unsafe_allow_html=True)

            st.success("✅ Detection completed successfully!")

            st.markdown("---")
            st.subheader("🧩 3D Space Scene Reconstruction")

            x_coords, y_coords, z_coords, labels = [], [], [], []
            for res in results:
                for box in res.boxes:
                    cls_id = int(box.cls)
                    name = res.names[cls_id]
                    x, y, w, h = box.xywh[0].tolist()
                    fh, fw = res.orig_shape
                    depth = 1 / (h / fh + 0.01)
                    x_coords.append(x / fw)
                    y_coords.append(1 - y / fh)
                    z_coords.append(depth)
                    labels.append(name)

            if x_coords:
                fig = go.Figure(
                    data=[go.Scatter3d(
                        x=x_coords, y=y_coords, z=z_coords,
                        mode='markers+text', text=labels,
                        textposition="top center",
                        marker=dict(size=6, color=z_coords,
                                    colorscale='Viridis', opacity=0.9)
                    )]
                )
                fig.update_layout(
                    scene=dict(
                        xaxis_title='X', yaxis_title='Y',
                        zaxis_title='Depth (approx)',
                        zaxis=dict(autorange='reversed')),
                    margin=dict(l=0, r=0, b=0, t=0),
                    height=600, paper_bgcolor='rgba(0,0,0,0)',
                    scene_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No objects detected to visualize in 3D.")



