"""
app.py
======
Streamlit front-end for the automated attendance system
(face_recognition / dlib version).

Pipeline per uploaded image:
  upload -> detect faces (HOG) -> embed (128-d ResNet) ->
  match vs enrolled embeddings (Euclidean distance, threshold 0.5) ->
  draw boxes -> log recognized people to attendance.csv (no duplicates/day).

Run with:
    streamlit run app.py
"""

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import face_recognition
from PIL import Image, ImageDraw, ImageFont

# Register a HEIC/HEIF decoder into Pillow so uploaded iPhone photos (.heic)
# can be opened by Image.open() just like JPEG/PNG.
from pillow_heif import register_heif_opener
register_heif_opener()

ENCODINGS_FILE = "encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"

# Recognition threshold on Euclidean distance between 128-d embeddings.
# face_recognition's pretrained model is tuned so that distances < ~0.6
# usually mean "same person". We use a stricter 0.5 to reduce false
# positives (wrongly marking an unknown person as someone enrolled),
# which is the costlier error for an attendance system.
RECOGNITION_THRESHOLD = 0.5


# --------------------------------------------------------------------------
# Data loading helpers
# --------------------------------------------------------------------------
@st.cache_data
def load_encodings():
    """Load the enrolled embeddings produced by enroll.py."""
    if not os.path.exists(ENCODINGS_FILE):
        return {"encodings": [], "names": []}
    with open(ENCODINGS_FILE, "rb") as f:
        return pickle.load(f)


def load_attendance():
    """Load attendance.csv, creating an empty frame if it does not exist."""
    if os.path.exists(ATTENDANCE_FILE):
        return pd.read_csv(ATTENDANCE_FILE)
    return pd.DataFrame(columns=["Name", "Date", "Time"])


def mark_attendance(name):
    """
    Append a row for `name` if they have NOT already been marked today.

    Duplicate prevention: we treat (Name, Date) as the unique key. If a row
    with this name and today's date already exists, we do nothing. This is
    what stops the same student being logged multiple times when their face
    appears in several uploaded photos on the same day.

    Returns True if a NEW record was written, False if it was a duplicate.
    """
    df = load_attendance()
    today = datetime.now().strftime("%Y-%m-%d")

    already_marked = (
        not df.empty
        and ((df["Name"] == name) & (df["Date"] == today)).any()
    )
    if already_marked:
        return False

    now_time = datetime.now().strftime("%H:%M:%S")
    new_row = pd.DataFrame([{"Name": name, "Date": today, "Time": now_time}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)
    return True


# --------------------------------------------------------------------------
# Core recognition pipeline
# --------------------------------------------------------------------------
def recognize_faces(image_np, data):
    """
    Detect and identify all faces in an RGB numpy image.

    Returns a list of dicts, one per detected face:
        {"box": (top, right, bottom, left), "name": str, "distance": float}
    """
    # HOG detector (CPU-friendly). For each face we then compute its embedding.
    face_locations = face_recognition.face_locations(image_np, model="hog")
    face_encodings = face_recognition.face_encodings(image_np, face_locations)

    known_encodings = data["encodings"]
    known_names = data["names"]

    results = []
    for box, encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        best_distance = None

        if known_encodings:
            # Euclidean (L2) distance between this face's embedding and every
            # enrolled embedding. Smaller distance = more similar face.
            distances = face_recognition.face_distance(known_encodings, encoding)
            best_idx = int(np.argmin(distances))
            best_distance = float(distances[best_idx])

            # Only accept the match if the closest enrolled face is within
            # our threshold; otherwise the face stays "Unknown".
            if best_distance <= RECOGNITION_THRESHOLD:
                name = known_names[best_idx]

        results.append({"box": box, "name": name, "distance": best_distance})

    return results


def draw_results(image_np, results):
    """Draw green boxes for recognized faces, red for unknown, with labels."""
    image = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    for r in results:
        top, right, bottom, left = r["box"]
        recognized = r["name"] != "Unknown"
        color = (0, 200, 0) if recognized else (220, 0, 0)  # green / red

        # Bounding box around the face.
        draw.rectangle([(left, top), (right, bottom)], outline=color, width=3)

        # Label text shown UNDER the box.
        label = r["name"]
        if r["distance"] is not None and recognized:
            label += f" ({r['distance']:.2f})"

        # Measure text so we can draw a filled background strip behind it.
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        draw.rectangle(
            [(left, bottom), (left + text_w + 10, bottom + text_h + 10)],
            fill=color,
        )
        draw.text((left + 5, bottom + 5), label, fill=(255, 255, 255), font=font)

    return image


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
st.title("📸 Automated Attendance System")
st.caption("Face detection + recognition with pretrained deep learning models "
           "(dlib / face_recognition)")

data = load_encodings()
enrolled_names = sorted(set(data["names"]))

# Sidebar: enrollment status.
st.sidebar.header("Enrollment status")
st.sidebar.metric("People enrolled", len(enrolled_names))
st.sidebar.metric("Total face samples", len(data["encodings"]))
if enrolled_names:
    st.sidebar.write("**Enrolled:**")
    for n in enrolled_names:
        st.sidebar.write(f"- {n}")
else:
    st.sidebar.warning("No one enrolled yet. Add photos to known_faces/ "
                       "then run `python enroll.py`.")
st.sidebar.info(f"Recognition threshold: {RECOGNITION_THRESHOLD} "
                "(Euclidean distance)")

def process_and_display(image_np, data):
    """
    Shared pipeline used by BOTH the upload tab and the live-camera tab:
    detect + recognize, draw boxes, mark attendance (with dedup), and render
    the results. Keeping this in one function means the webcam path reuses the
    exact same recognition logic as the upload path.
    """
    with st.spinner("Detecting and recognizing faces..."):
        results = recognize_faces(image_np, data)
        annotated = draw_results(image_np, results)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(annotated, caption="Detection result",
                 use_container_width=True)

    with col2:
        st.write(f"**Faces detected:** {len(results)}")

        recognized = [r for r in results if r["name"] != "Unknown"]
        unknown = [r for r in results if r["name"] == "Unknown"]

        # Mark attendance for each recognized person (dedup inside).
        newly_marked = []
        already_present = []
        for r in recognized:
            if mark_attendance(r["name"]):
                newly_marked.append(r["name"])
            else:
                already_present.append(r["name"])

        if newly_marked:
            st.success("Marked present: " + ", ".join(sorted(set(newly_marked))))
        if already_present:
            st.info("Already marked today: "
                    + ", ".join(sorted(set(already_present))))
        if unknown:
            st.warning(f"{len(unknown)} unrecognized face(s).")
        if not recognized:
            st.warning("No enrolled person recognized in this image.")

        # Per-face detail table.
        st.write("**Per-face results:**")
        detail = [
            {
                "Face": i + 1,
                "Name": r["name"],
                "Distance": round(r["distance"], 3)
                if r["distance"] is not None else None,
            }
            for i, r in enumerate(results)
        ]
        st.dataframe(pd.DataFrame(detail), use_container_width=True)


tab_mark, tab_live, tab_records = st.tabs(
    ["✅ Mark Attendance", "📷 Live Camera", "📋 View Records"]
)

# ---- Tab 1: Mark Attendance (upload) ------------------------------------
with tab_mark:
    st.subheader("Upload a group / classroom photo")

    if not data["encodings"]:
        st.error("No enrolled faces found. Run `python enroll.py` first.")
    else:
        uploaded = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png", "heic", "heif"]
        )

        if uploaded is not None:
            # Load upload as an RGB numpy array for face_recognition.
            pil_image = Image.open(uploaded).convert("RGB")
            image_np = np.array(pil_image)
            process_and_display(image_np, data)

# ---- Tab 2: Live Camera -------------------------------------------------
with tab_live:
    st.subheader("Capture attendance from the webcam")
    st.caption("Click the camera button to take a snapshot. Each capture runs "
               "the same detection + recognition pipeline and marks attendance.")

    if not data["encodings"]:
        st.error("No enrolled faces found. Run `python enroll.py` first.")
    else:
        # st.camera_input() opens the device webcam and returns a captured
        # photo (an UploadedFile, just like st.file_uploader). We use snapshot
        # capture rather than a continuous video stream because HOG detection
        # on every video frame is too slow on a CPU; a snapshot is reliable,
        # dependency-free, and reuses the identical recognition pipeline.
        camera_photo = st.camera_input("Take a photo")

        if camera_photo is not None:
            pil_image = Image.open(camera_photo).convert("RGB")
            image_np = np.array(pil_image)
            process_and_display(image_np, data)

# ---- Tab 2: View Records -------------------------------------------------
with tab_records:
    st.subheader("Attendance records")

    df = load_attendance()
    if df.empty:
        st.info("No attendance recorded yet.")
    else:
        # Show most recent first.
        st.dataframe(df.iloc[::-1].reset_index(drop=True),
                     use_container_width=True)
        st.write(f"**Total records:** {len(df)}")

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download attendance.csv",
            data=csv_bytes,
            file_name="attendance.csv",
            mime="text/csv",
        )
