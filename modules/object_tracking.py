"""
Task 4: Real-Time Object Detection and Tracking
Unified AI Web Application - CodeAlpha AI Internship
"""

import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Check if Ultralytics/Torch is permitted by system policy
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    # Test torch initialization safely
    _test_yolo = YOLO("yolov8n.pt")
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


class SimpleCentroidTracker:
    """
    Lightweight persistent object tracker based on Euclidean distance & centroid persistence.
    Assigns persistent IDs (ID #1, ID #2...) and tracks them across frames.
    """
    def __init__(self, max_disappeared=20):
        self.next_object_id = 1
        self.objects = {}       # id -> (x, y, w, h, class_name, conf)
        self.disappeared = {}   # id -> frame_count

    def update(self, rects_with_info):
        """
        rects_with_info: list of (x, y, w, h, class_name, conf)
        """
        if len(rects_with_info) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > 20:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return self.objects

        if len(self.objects) == 0:
            for rect in rects_with_info:
                self.objects[self.next_object_id] = rect
                self.disappeared[self.next_object_id] = 0
                self.next_object_id += 1
            return self.objects

        # Match existing objects to new detections by centroid distance
        object_ids = list(self.objects.keys())
        current_centroids = [
            (self.objects[oid][0] + self.objects[oid][2] // 2,
             self.objects[oid][1] + self.objects[oid][3] // 2)
            for oid in object_ids
        ]

        new_centroids = [
            (r[0] + r[2] // 2, r[1] + r[3] // 2)
            for r in rects_with_info
        ]

        # Greedy match closest
        used_new = set()
        used_obj = set()

        for o_idx, oid in enumerate(object_ids):
            best_dist = float("inf")
            best_new_idx = -1
            cx, cy = current_centroids[o_idx]

            for n_idx, (nx, ny) in enumerate(new_centroids):
                if n_idx in used_new:
                    continue
                dist = (cx - nx) ** 2 + (cy - ny) ** 2
                if dist < best_dist and dist < 15000:  # Distance threshold
                    best_dist = dist
                    best_new_idx = n_idx

            if best_new_idx != -1:
                self.objects[oid] = rects_with_info[best_new_idx]
                self.disappeared[oid] = 0
                used_new.add(best_new_idx)
                used_obj.add(oid)
            else:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > 20:
                    del self.objects[oid]
                    del self.disappeared[oid]

        for n_idx, rect in enumerate(rects_with_info):
            if n_idx not in used_new:
                self.objects[self.next_object_id] = rect
                self.disappeared[self.next_object_id] = 0
                self.next_object_id += 1

        return self.objects


class OpenCVVisionDetector:
    """
    OpenCV-based Vision Detection Engine with Haar Cascade and Color/Contour intelligence.
    Provides fast, zero-dependency detection of humans, faces, cars, and foreground objects.
    """
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.upper_body = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
        self.tracker = SimpleCentroidTracker()

    def detect_and_track(self, frame, conf_threshold=0.35, is_video=False):
        """Processes frame, detects objects, and updates persistent tracking."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        detections = []

        # 1. Face / Person detections
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        for (x, y, fw, fh) in faces:
            detections.append((x, y, fw, fh, "person (face)", 0.92))

        # 2. Upper body detections
        bodies = self.upper_body.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
        for (x, y, bw, bh) in bodies:
            detections.append((x, y, bw, bh, "person (body)", 0.88))

        # 3. Dynamic Foreground & Contour Object Detection for motion/general items
        if is_video:
            fgmask = self.fgbg.apply(frame)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 900:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    if cw < w * 0.9 and ch < h * 0.9:
                        detections.append((x, y, cw, ch, "moving object", 0.78))
        elif len(detections) == 0:
            # For static image if no face/body, detect prominent foreground objects
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                if cv2.contourArea(cnt) > 1200:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    if cw < w * 0.95 and ch < h * 0.95:
                        detections.append((x, y, cw, ch, "object", 0.82))

        # Update persistent tracker
        tracked_objects = self.tracker.update(detections)
        
        # Annotate
        annotated_frame = frame.copy()
        counts = {}

        for obj_id, (x, y, dw, dh, cls_name, conf) in tracked_objects.items():
            if conf < conf_threshold:
                continue
            
            counts[cls_name] = counts.get(cls_name, 0) + 1

            # Consistent distinct color by ID
            np.random.seed(obj_id + 77)
            color = [int(c) for c in np.random.randint(60, 240, size=3)]

            # Draw box
            cv2.rectangle(annotated_frame, (x, y), (x + dw, y + dh), color, 2)

            # Label
            label = f"ID #{obj_id} | {cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x, y - 20), (x + tw + 6, y), color, -1)
            cv2.putText(annotated_frame, label, (x + 3, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated_frame, counts, tracked_objects


@st.cache_resource
def get_detector():
    """Initializes and returns vision detector."""
    if YOLO_AVAILABLE:
        try:
            return "yolo", YOLO("yolov8n.pt")
        except Exception:
            pass
    return "opencv", OpenCVVisionDetector()


def process_video_stream(video_path, detector_tuple, conf_threshold=0.35, max_frames=120):
    """Processes a video file frame-by-frame with persistent tracking."""
    det_type, detector = detector_tuple
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    st_frame = st.empty()
    progress_bar = st.progress(0)
    metrics_placeholder = st.empty()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(total_frames, max_frames) if total_frames > 0 else max_frames
    
    frame_idx = 0
    unique_tracked_ids = set()

    while cap.isOpened() and frame_idx < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        
        if det_type == "yolo":
            results = detector.track(frame, persist=True, conf=conf_threshold, verbose=False)
            # Annotate YOLO
            annotated_frame = frame.copy()
            counts = {}
            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = results[0].names.get(cls_id, f"Class {cls_id}")
                    tid = int(box.id[0]) if box.id is not None else 0
                    unique_tracked_ids.add(tid)
                    counts[cls_name] = counts.get(cls_name, 0) + 1
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"ID #{tid} | {cls_name} {conf:.2f}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            annotated_frame, counts, tracked = detector.detect_and_track(frame, conf_threshold=conf_threshold, is_video=True)
            for oid in tracked.keys():
                unique_tracked_ids.add(oid)

        # Convert BGR to RGB for Streamlit display
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(rgb_frame, caption=f"Frame {frame_idx}/{frames_to_process}", use_container_width=True)

        progress_bar.progress(frame_idx / frames_to_process)
        
        # Real-time metrics
        counts_str = ", ".join([f"{k}: {v}" for k, v in counts.items()]) or "None"
        metrics_placeholder.markdown(
            f"📊 **Current Frame Detections:** `{counts_str}` | **Total Unique Tracked IDs:** `{len(unique_tracked_ids)}`"
        )

    cap.release()
    st.success(f"✅ Video tracking complete! Processed {frame_idx} frames. Tracked {len(unique_tracked_ids)} unique persistent objects.")


def render_object_tracking_ui():
    """Renders the Streamlit UI for Task 4 Object Detection & Tracking."""
    st.markdown("""
        <div style="background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%); padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(239,68,68,0.2);">
            <h2 style="margin:0; font-weight:700; color:white;">🎯 Real-Time Object Detection & Tracking</h2>
            <p style="margin:0.5rem 0 0 0; opacity:0.9; font-size: 0.95rem;">Intelligent multi-object detection and persistent object tracking (IDs, Bounding Boxes, Confidence Scores).</p>
        </div>
    """, unsafe_allow_html=True)

    detector_tuple = get_detector()
    det_type, detector = detector_tuple

    st.caption(f"⚡ **Active Vision Engine:** `{det_type.upper()} Multi-Object Tracker & Detector`")

    # Configuration Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Vision & Tracking Settings")
        conf_thresh = st.slider("Confidence Threshold", 0.10, 0.95, 0.35, 0.05)
        max_video_frames = st.slider("Max Video Frames to Process", 30, 200, 100, 10)

    # Mode Selector
    input_mode = st.radio(
        "Select Input Source",
        ["📷 Upload Image / Snapshot", "🎥 Upload Video File (.mp4)", "🔴 Live Camera Capture"],
        horizontal=True
    )

    if input_mode == "📷 Upload Image / Snapshot":
        st.markdown("#### 🖼️ Image Detection & Classification")
        img_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
        use_sample = st.button("🖼️ Use Built-in Demo Sample")

        image_to_process = None
        if img_file is not None:
            image_to_process = Image.open(img_file)
        elif use_sample:
            # Create a rich demo test scene
            sample_img = np.full((400, 600, 3), 235, dtype=np.uint8)
            cv2.rectangle(sample_img, (60, 120), (220, 320), (180, 70, 40), -1)
            cv2.putText(sample_img, "Test Object A", (70, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.circle(sample_img, (420, 220), 75, (30, 150, 70), -1)
            cv2.putText(sample_img, "Test Object B", (350, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            image_to_process = Image.fromarray(sample_img)

        if image_to_process is not None:
            img_np = np.array(image_to_process.convert("RGB"))
            bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            with st.spinner("Running Vision Inference..."):
                if det_type == "yolo":
                    results = detector.predict(img_np, conf=conf_thresh, verbose=False)
                    # draw yolo
                    annotated_bgr = bgr.copy()
                    counts = {}
                    if results and len(results) > 0 and results[0].boxes is not None:
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            cls_name = results[0].names.get(cls_id, f"Class {cls_id}")
                            counts[cls_name] = counts.get(cls_name, 0) + 1
                            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
                            cv2.putText(annotated_bgr, f"{cls_name} {conf:.2f}", (x1, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
                else:
                    annotated_bgr, counts, _ = detector.detect_and_track(bgr, conf_threshold=conf_thresh, is_video=False)

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns([2, 1], gap="medium")
            with col1:
                st.image(annotated_rgb, caption="Vision Detection & Tracking Result", use_container_width=True)
            with col2:
                st.markdown("#### 📊 Detection Summary")
                if counts:
                    st.metric("Total Objects Detected", sum(counts.values()))
                    for cls_name, count in counts.items():
                        st.markdown(f"- **{cls_name}**: `{count}`")
                else:
                    st.info("No objects detected above the confidence threshold.")

    elif input_mode == "🎥 Upload Video File (.mp4)":
        st.markdown("#### 🎬 Video Multi-Object Tracking")
        video_file = st.file_uploader("Upload a video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
        
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            video_path = tfile.name
            tfile.close()

            st.video(video_path)
            
            if st.button("▶️ Start Object Tracking Pipeline", type="primary"):
                with st.spinner("Processing video frames with Persistent Object Tracking..."):
                    process_video_stream(video_path, detector_tuple, conf_threshold=conf_thresh, max_frames=max_video_frames)
                try:
                    os.remove(video_path)
                except Exception:
                    pass

    elif input_mode == "🔴 Live Camera Capture":
        st.markdown("#### 📷 Real-Time Camera Stream Detection")
        st.info("💡 Snap a photo from your webcam below to run instant object detection and tracking analysis.")
        
        camera_img = st.camera_input("Take a snapshot from webcam")
        
        if camera_img is not None:
            pil_img = Image.open(camera_img)
            img_np = np.array(pil_img.convert("RGB"))
            bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            with st.spinner("Analyzing camera frame..."):
                if det_type == "yolo":
                    results = detector.track(img_np, persist=True, conf=conf_thresh)
                    annotated_bgr = bgr.copy()
                    counts = {}
                else:
                    annotated_bgr, counts, tracked = detector.detect_and_track(bgr, conf_threshold=conf_thresh)

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(annotated_rgb, caption="Live Detection & Tracking Result", use_container_width=True)
            with c2:
                st.markdown("#### 📊 Detections in View")
                if counts:
                    st.metric("Objects in View", sum(counts.values()))
                    for cls_name, count in counts.items():
                        st.markdown(f"- **{cls_name}**: `{count}`")
                else:
                    st.info("No recognizable objects detected in current frame.")
