# Automated Attendance System Using Face Recognition — Full Project Explanation

> A complete, plain-English walkthrough of what this project is, how every part works, how to run it, and how to defend it. Read this top-to-bottom and you will understand the entire system.

---

## 1. What this project is (in one paragraph)

This is an **automated attendance system** that recognizes people from photographs using **pretrained deep-learning face-recognition models** — no model training required. You enroll a handful of people by giving the system a few photos of each. Then you either **upload a group/classroom photo** or **take a live webcam snapshot**. The system finds every face in the image, identifies the people it knows, draws labelled boxes around them (green = recognized, red = unknown), and records each recognized person in an attendance spreadsheet (`attendance.csv`) with the date and time. It refuses to log the same person twice on the same day. Everything runs locally on an ordinary **CPU-only Windows machine**, with no database and no GPU.

---

## 2. The problem it solves

Taking attendance by hand wastes class time, is error-prone, and is easy to cheat (proxy attendance). A face-recognition system can mark an entire room from a single photo, is contactless, and is hard to fake. Modern face-recognition models are freely available **pretrained**, so we can build a working system without collecting a large dataset or training anything ourselves.

---

## 3. Core idea: how a computer "recognizes" a face

The system works in two conceptual steps, and it's important to keep them separate:

1. **Face detection** — *Where* are the faces in the image? This produces a rectangle (bounding box) around each face. We use a **HOG (Histogram of Oriented Gradients) + linear SVM** detector.

2. **Face recognition** — *Who* is each face? Each detected face is converted into a list of 128 numbers called an **embedding** (or "face encoding"). A pretrained deep neural network (a ResNet-style CNN) produces this embedding. The key property: **the same person's face always produces embeddings that are close together, and different people's embeddings are far apart.**

To identify a face, we measure the **Euclidean distance** between its embedding and the embeddings of everyone we enrolled. If the closest enrolled person is within a distance of **0.5**, we call it a match. Otherwise the face is labelled "Unknown."

That's the whole trick: recognition becomes a simple distance comparison in a 128-dimensional space.

---

## 4. The two phases of the system

### Phase A — Enrollment (done once, offline)

```
known_faces/<person_name>/*.jpg
        │
        ▼
   Detect the face (HOG)
        │
        ▼
   Compute the 128-d embedding (pretrained ResNet)
        │
        ▼
   Store (embedding, name) for every enrollment photo
        │
        ▼
   Save everything to  encodings.pkl
```

You put photos into folders named after each person, then run `enroll.py`. It builds a database file (`encodings.pkl`) holding every face embedding and the name it belongs to.

### Phase B — Recognition (runs every time you submit an image)

```
Upload a photo  OR  take a webcam snapshot
        │
        ▼
   Detect ALL faces (HOG)
        │
        ▼
   Compute a 128-d embedding for each face
        │
        ▼
   For each face: Euclidean distance to every enrolled embedding
        │
        ├─ closest distance ≤ 0.5  →  that person's name   (GREEN box)
        └─ otherwise                →  "Unknown"             (RED box)
        │
        ▼
   For each recognized person: write to attendance.csv
   (but only if they are not already recorded today)
```

---

## 5. Features

- **Enroll 5–10 people** from a few photos each.
- **Two input modes:**
  - **Upload** a group/classroom photo (`.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`).
  - **Live Camera** — take a webcam snapshot directly in the browser.
- **Detects every face** in the image, not just one.
- **Recognizes enrolled people** using 128-d embeddings + Euclidean distance with a 0.5 threshold.
- **Visual feedback:** green box + name (and match distance) for recognized faces; red box + "Unknown" for everyone else; labels drawn beneath each box.
- **Attendance logging** to `attendance.csv` with `Name, Date, Time`.
- **Duplicate prevention:** each person is logged at most once per calendar day.
- **HEIC/iPhone photo support** built in.
- **View Records tab:** browse the full attendance table and **download the CSV**.
- **Sidebar** showing how many people and face samples are enrolled.

---

## 6. The technology stack and why each piece was chosen

| Component | Library / tool | Why it was chosen |
|-----------|----------------|-------------------|
| Face detection + recognition | **`face_recognition`** (wraps **dlib**) | One-line detection and embedding; uses a pretrained ResNet with ~99.4% accuracy on the LFW benchmark; runs on CPU. |
| Pretrained model | **dlib ResNet (128-d)** | Strong, ready-to-use embeddings with no training. |
| Detector | **HOG + linear SVM** | Fast on CPU, no GPU needed, good for frontal faces. |
| Prebuilt dlib | **`dlib-bin`** | Ships a compiled dlib wheel so you don't need CMake / Visual Studio to install. |
| Web interface | **Streamlit** | Pure-Python web UI: file upload, webcam input, tabs, tables, and a download button with minimal code. |
| Image handling | **Pillow, OpenCV, NumPy** | Loading images, converting to arrays, drawing boxes and labels. |
| iPhone photo support | **pillow-heif** | Registers a HEIC/HEIF decoder into Pillow so `.heic` files open like JPEGs. |
| Tabular storage | **Pandas + CSV** | Lightweight, human-readable attendance log; no database to set up. |

---

## 7. Project structure — every file explained

```
attendance-system/
├── known_faces/              # YOU add photos here, one folder per person
│   └── .gitkeep
├── enroll.py                 # builds encodings.pkl from known_faces/
├── app.py                    # the main Streamlit app (3 tabs)
├── app_deepface.py           # fallback app if face_recognition won't install
├── requirements.txt          # dependencies
├── README.md                 # course-submission README
├── REPORT.md                 # full written report
├── PROJECT_EXPLANATION.md    # this document
├── .gitignore                # excludes encodings.pkl, attendance.csv, venv, etc.
└── sample_outputs/           # YOU add screenshots here
    └── .gitkeep
```

### `enroll.py` — building the face database
- Loops over every sub-folder of `known_faces/`. The **folder name is the person's name**.
- For each photo: registers HEIC support, detects the face with HOG, and computes its 128-d embedding.
- **Skips photos with no detectable face** (prints a warning) so one bad photo doesn't crash enrollment.
- Stores **multiple embeddings per person** (one per photo) rather than averaging them — this is more robust to differences in pose and lighting, because at recognition time we compare against the *closest* sample.
- Saves a dictionary `{"encodings": [...], "names": [...]}` to `encodings.pkl` and prints a summary.

### `app.py` — the main application (three tabs)
- **Loads `encodings.pkl`** at startup and shows the enrolled count in the sidebar.
- A shared function `process_and_display()` runs the full pipeline (detect → recognize → draw → log) so both input tabs behave identically.
- **Tab 1 — ✅ Mark Attendance:** upload an image; supports `.jpg/.jpeg/.png/.heic/.heif`.
- **Tab 2 — 📷 Live Camera:** uses `st.camera_input()` to take a webcam snapshot and run the same pipeline. (Snapshot rather than continuous video, because running HOG on every video frame is too slow on a CPU — a snapshot is reliable and needs no extra libraries.)
- **Tab 3 — 📋 View Records:** shows the attendance table (most recent first) and a **Download CSV** button.
- Key helper functions:
  - `recognize_faces()` — detection + embedding + nearest-match with the 0.5 threshold.
  - `mark_attendance()` — writes a row only if `(Name, Date)` isn't already present today.
  - `draw_results()` — draws green/red boxes and labels with Pillow.

### `app_deepface.py` — the fallback
- A second, complete version of the app that uses the **DeepFace** library instead of `face_recognition`/dlib. It exists in case dlib refuses to install on a given machine. It enrolls `known_faces/` in memory at startup (no separate `encodings.pkl`) and uses cosine distance with a 0.40 threshold tuned for the VGG-Face model.

### `requirements.txt`
- Lists the pure-wheel dependencies, and documents (in comments) the two manual install commands for dlib and `face_recognition` (explained in §8).

### `.gitignore`
- Keeps generated and machine-specific files out of the repo: `encodings.pkl`, `attendance.csv`, the virtual environment, IDE files, and OS junk.

---

## 8. Installation (Windows, the way that actually works)

> **Use Python 3.10 or 3.11.** Prebuilt `dlib-bin` wheels are reliable there. Python 3.13 may force a source build of dlib, which needs a full C++ compiler and often fails.

```powershell
# 1. Create and activate a virtual environment with Python 3.10
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1          # prompt should now start with (.venv)

# 2. Upgrade pip
python -m pip install --upgrade pip

# 3. Install the pure-wheel dependencies
pip install -r requirements.txt

# 4. Install the PREBUILT dlib (no CMake / Visual Studio needed)
pip install dlib-bin

# 5. Install face_recognition WITHOUT its dependencies
#    so pip never tries to COMPILE dlib from source
pip install --no-deps face_recognition_models face_recognition
```

**Why steps 4–5 are special (important — likely a viva question):**
`face_recognition`'s package metadata depends on the *source* package `dlib`. If you install it normally, pip tries to **compile dlib from source**, which fails on Windows without a working MSVC C++ compiler. The fix is `dlib-bin` (a precompiled drop-in that provides `import dlib`) plus installing `face_recognition` with `--no-deps` so pip doesn't try to rebuild dlib.

**Verify the install:**
```powershell
python -c "import dlib, face_recognition, streamlit, cv2, PIL, pandas, numpy; print('ALL IMPORTS OK')"
```

---

## 9. How to run it (step by step)

1. **Add photos.** Create one folder per person inside `known_faces/` and add 2–3 clear, front-facing photos each:
   ```
   known_faces/
     alice/  1.jpg  2.jpg
     bob/    1.heic
     carol/  1.png  2.png
   ```
2. **Build the database:**
   ```powershell
   python enroll.py
   ```
   You should see `[OK]` lines and a final `Saved N embedding(s) for M person(s)`.
3. **Launch the app:**
   ```powershell
   streamlit run app.py
   ```
   Your browser opens at `http://localhost:8501`.
4. **Mark attendance** — either upload a photo (Tab 1) or take a webcam snapshot (Tab 2). Allow camera access when the browser asks.
5. **View / download records** — Tab 3.

---

## 10. Key design decisions (defend these in your viva)

- **Why the `face_recognition` library?** It gives pretrained, high-accuracy embeddings with a one-line API, runs on CPU, and needs no training — ideal for the scope and timeframe.
- **Why HOG for detection?** Fast and dependency-light on CPU; accurate enough for the frontal, cooperative faces typical of attendance. CNN/MTCNN detectors are more accurate on hard poses but much heavier.
- **What is an embedding?** A 128-number vector summarising a face, produced by a CNN, where distance encodes identity similarity.
- **Why Euclidean distance?** The dlib model was trained with a triplet loss that optimises L2 distances, so Euclidean is the metric the embedding space was built for.
- **Why a 0.5 threshold?** dlib treats distances under ~0.6 as the same person; we tightened it to 0.5 to reduce **false positives**, because wrongly marking an unenrolled person present is the costlier error for attendance.
- **What is triplet loss?** A training objective using an anchor, a positive (same person), and a negative (different person); it pulls anchor–positive together and pushes anchor–negative apart by a margin, shaping the embedding space.
- **Why store multiple embeddings per person, not an average?** Averaging blurs pose/lighting variation; keeping individual samples lets us match against the closest one.
- **How does duplicate prevention work?** `(Name, Date)` is the unique key; before writing we check whether that person already has a row today and skip if so.
- **Why webcam snapshot, not live video?** HOG runs per-frame; on a CPU without a GPU, processing every video frame is too slow. A snapshot gives identical accuracy, stays responsive, and needs no extra dependencies.
- **Why no fine-tuning?** The pretrained model already generalises to unseen identities; fine-tuning needs a large dataset and GPU time and would risk overfitting to ~10 people.

---

## 11. Limitations (state these honestly)

- HOG misses small, strongly rotated, profile, or poorly lit faces.
- A single global threshold (0.5) cannot be optimal for every person and condition.
- No liveness / anti-spoofing — a photo of a photo could be accepted.
- Linear search over embeddings is fine for 5–10 people but does not scale to thousands without an index.
- Accuracy depends on the quality and number of enrollment photos.

---

## 12. Future work

- Swap HOG for a CNN/MTCNN detector for harder poses.
- Use an ArcFace-based model for higher accuracy.
- Add an approximate nearest-neighbour index (e.g. FAISS) to scale to large cohorts.
- Per-person adaptive thresholds and embedding averaging.
- Continuous live-video recognition (e.g. `streamlit-webrtc`) on more capable hardware.
- Liveness detection for anti-spoofing and a database-backed store.

---

## 13. References

1. F. Schroff, D. Kalenichenko, J. Philbin, "FaceNet: A Unified Embedding for Face Recognition and Clustering," *CVPR*, 2015.
2. J. Deng, J. Guo, N. Xue, S. Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," *CVPR*, 2019.
3. D. E. King, "Dlib-ml: A Machine Learning Toolkit," *JMLR*, vol. 10, pp. 1755–1758, 2009.
4. K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," *CVPR*, 2016.
5. N. Dalal, B. Triggs, "Histograms of Oriented Gradients for Human Detection," *CVPR*, 2005.
6. M. Turk, A. Pentland, "Eigenfaces for Recognition," *Journal of Cognitive Neuroscience*, vol. 3, no. 1, pp. 71–86, 1991.

---

*This system uses pretrained models for inference only; no model training or fine-tuning was performed. Built as a computer vision semester project.*
