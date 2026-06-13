"""
app_deepface.py
===============
FALLBACK Streamlit app, used ONLY if `face_recognition` / dlib fails to
install on Windows. It provides the same functionality using DeepFace
(pretrained models, no training).

Key differences from app.py:
  * DeepFace does detection + embedding internally.
  * We use the default "VGG-Face" model. DeepFace's natural distance metric
    is cosine; the cosine "verification" threshold for VGG-Face is ~0.40,
    so the 0.5 Euclidean threshold from the dlib version does NOT transfer
    directly. We expose DISTANCE_THRESHOLD below and use cosine distance.

Install (only if needed):
    pip install deepface tf-keras
    pip install tensorflow-cpu

Enroll:
    python enroll_deepface.py        (see note at bottom) OR rely on the
    in-app embedding of known_faces/ at startup (this file does that).

Run:
    streamlit run app_deepface.py
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from deepface import DeepFace

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

MODEL_NAME = "VGG-Face"        # pretrained, downloaded automatically on first use
DETECTOR_BACKEND = "opencv"    # CPU-friendly detector
DISTANCE_METRIC = "cosine"

# Cosine distance threshold for VGG-Face. Smaller = more similar.
# DeepFace's recommended verification threshold for VGG-Face/cosine is ~0.40;
# we use 0.40 here. (This is the DeepFace analogue of the 0.5 Euclidean
# threshold used in the dlib version.)
DISTANCE_THRESHOLD = 0.40

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------
# Enrollment (done in-app to keep the fallback self-contained)
# --------------------------------------------------------------------------
@st.cache_resource
def load_known_embeddings():
    """
    Build {name: [embedding, ...]} by embedding every photo in known_faces/.
    Cached so it runs once per session.
    """
    known = {"encodings": [], "names": []}

    if not os.path.isdir(KNOWN_FACES_DIR):
        return known

    for person in sorted(os.listdir(KNOWN_FACES_DIR)):
        person_dir = os.path.join(KNOWN_FACES_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith(VALID_EXTENSIONS):
                continue
            img_path = os.path.join(person_dir, img_name)
            try:
                # represent() returns a list of detected faces with embeddings.
                reps = DeepFace.represent(
                    img_path=img_path,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=True,
                )
                if reps:
                    known["encodings"].append(np.array(reps[0]["embedding"]))
                    known["names"].append(person)
            except Exception as e:
                print(f"[WARN] Could not enroll {img_path}: {e}")

    return known


# --------------------------------------------------------------------------
# Attendance helpers (identical logic to app.py)
# --------------------------------------------------------------------------
def load_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        return pd.read_csv(ATTENDANCE_FILE)
    return pd.DataFrame(columns=["Name", "Date", "Time"])


def mark_attendance(name):
    df = load_attendance()
    today = datetime.now().strftime("%Y-%m-%d")
    already = (not df.empty
               and ((df["Name"] == name) & (df["Date"] == today)).any())
    if already:
        return False
    now_time = datetime.now().strftime("%H:%M:%S")
    new_row = pd.DataFrame([{"Name": name, "Date": today, "Time": now_time}])
    pd.concat([df, new_row], ignore_index=True).to_csv(ATTENDANCE_FILE, index=False)
    return True


def cosine_distance(a, b):
    """1 - cosine similarity. 0 = identical direction."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return 1.0 - sim


# --------------------------------------------------------------------------
# Recognition
# --------------------------------------------------------------------------
def recognize_faces(image_np, known):
    """
    Detect faces with DeepFace, embed each, match against known embeddings.
    Returns list of {"box": (top,right,bottom,left), "name", "distance"}.
    """
    results = []
    try:
        faces = DeepFace.extract_faces(
            img_path=image_np,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )
    except Exception:
        faces = []

    for face in faces:
        area = face.get("facial_area", {})
        x, y, w, h = area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0)
        # Convert DeepFace (x,y,w,h) to face_recognition-style (top,right,bottom,left).
        box = (y, x + w, y + h, x)

        name = "Unknown"
        best_distance = None
        try:
            reps = DeepFace.represent(
                img_path=image_np[y:y + h, x:x + w],
                model_name=MODEL_NAME,
                detector_backend="skip",  # already cropped
                enforce_detection=False,
            )
            if reps and known["encodings"]:
                emb = np.array(reps[0]["embedding"])
                dists = [cosine_distance(emb, k) for k in known["encodings"]]
                best_idx = int(np.argmin(dists))
                best_distance = float(dists[best_idx])
                if best_distance <= DISTANCE_THRESHOLD:
                    name = known["names"][best_idx]
        except Exception:
            pass

        results.append({"box": box, "name": name, "distance": best_distance})

    return results


def draw_results(image_np, results):
    image = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    for r in results:
        top, right, bottom, left = r["box"]
        recognized = r["name"] != "Unknown"
        color = (0, 200, 0) if recognized else (220, 0, 0)
        draw.rectangle([(left, top), (right, bottom)], outline=color, width=3)

        label = r["name"]
        if r["distance"] is not None and recognized:
            label += f" ({r['distance']:.2f})"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([(left, bottom), (left + tw + 10, bottom + th + 10)], fill=color)
        draw.text((left + 5, bottom + 5), label, fill=(255, 255, 255), font=font)

    return image


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Face Recognition Attendance (DeepFace)", layout="wide")
st.title("📸 Automated Attendance System (DeepFace fallback)")

with st.spinner("Loading model and enrolling known faces (first run downloads weights)..."):
    known = load_known_embeddings()

enrolled_names = sorted(set(known["names"]))
st.sidebar.header("Enrollment status")
st.sidebar.metric("People enrolled", len(enrolled_names))
st.sidebar.metric("Total face samples", len(known["encodings"]))
for n in enrolled_names:
    st.sidebar.write(f"- {n}")
st.sidebar.info(f"Model: {MODEL_NAME} | Metric: {DISTANCE_METRIC} "
                f"| Threshold: {DISTANCE_THRESHOLD}")

tab_mark, tab_records = st.tabs(["✅ Mark Attendance", "📋 View Records"])

with tab_mark:
    st.subheader("Upload a group / classroom photo")
    if not known["encodings"]:
        st.error("No enrolled faces. Add photos to known_faces/ and reload.")
    else:
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            pil_image = Image.open(uploaded).convert("RGB")
            image_np = np.array(pil_image)
            with st.spinner("Detecting and recognizing faces..."):
                results = recognize_faces(image_np, known)
                annotated = draw_results(image_np, results)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(annotated, caption="Detection result",
                         use_container_width=True)
            with col2:
                st.write(f"**Faces detected:** {len(results)}")
                recognized = [r for r in results if r["name"] != "Unknown"]
                newly, already = [], []
                for r in recognized:
                    (newly if mark_attendance(r["name"]) else already).append(r["name"])
                if newly:
                    st.success("Marked present: " + ", ".join(sorted(set(newly))))
                if already:
                    st.info("Already marked today: " + ", ".join(sorted(set(already))))
                if not recognized:
                    st.warning("No enrolled person recognized.")
                detail = [{"Face": i + 1, "Name": r["name"],
                           "Distance": round(r["distance"], 3)
                           if r["distance"] is not None else None}
                          for i, r in enumerate(results)]
                st.dataframe(pd.DataFrame(detail), use_container_width=True)

with tab_records:
    st.subheader("Attendance records")
    df = load_attendance()
    if df.empty:
        st.info("No attendance recorded yet.")
    else:
        st.dataframe(df.iloc[::-1].reset_index(drop=True), use_container_width=True)
        st.write(f"**Total records:** {len(df)}")
        st.download_button("⬇️ Download attendance.csv",
                           df.to_csv(index=False).encode("utf-8"),
                           "attendance.csv", "text/csv")

# NOTE: This fallback enrolls known_faces/ in-memory at startup (cached),
# so there is no separate encodings.pkl for the DeepFace version.
