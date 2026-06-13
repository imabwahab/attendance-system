# Automated Attendance System Using Face Recognition

> Mark classroom attendance automatically from a photo or live webcam snapshot using pretrained deep-learning face embeddings — no model training required.

<p align="center">
  <em>Computer Vision — Semester Project</em><br>
  <strong>6th Semester · Section 2E</strong>
</p>

---

## 📌 Project Information

| | |
|---|---|
| **Project Title** | Automated Attendance System Using Face Recognition with Pretrained Deep Learning Models |
| **Course** | Computer Vision |
| **Semester** | 6th Semester |
| **Section** | 2E |
| **Type** | Semester Project (Group of 2) |

### 👥 Team Members

| # | Name | Roll No. |
|---|------|----------|
| 1 | _Abdul Wahab_ | _F23BDOCS1E02055_ |
| 2 | _Muhammad Tauseef_ | _F23BDOCS1E02051_ |

---

## Overview

This project implements an **automated attendance system** based on **face detection and recognition** using **pretrained deep-learning models**. The user enrolls a small group of people (5–10) by providing a few photos of each — either by dropping images into folders or directly through the web interface. The system then accepts a **group/classroom photo** or a **live webcam snapshot**, detects every face present, recognizes the enrolled individuals, and logs each recognized person to a CSV attendance sheet with the date and time.

The recognition engine is the [`face_recognition`](https://github.com/ageitgey/face_recognition) library, which wraps **dlib**. Faces are located with a **HOG (Histogram of Oriented Gradients) + linear SVM** detector, and each face is converted into a **128-dimensional embedding** by a pretrained ResNet-style convolutional network trained with a triplet-loss objective. Identification is performed by comparing embeddings using **Euclidean distance** against a threshold of `0.5`.

The entire application runs on a **CPU-only Windows machine**, uses **no database** (attendance is stored in a plain CSV file), and is wrapped in a simple multi-tab **Streamlit** web interface. No models are trained or fine-tuned — all deep-learning weights are pretrained and used purely for inference.

---

## Key Features

- 👤 **Visual enrollment** — add a person directly in the app: type a name, upload photos, click Enroll. (A command-line `enroll.py` is also provided.)
- ✅ **Image upload** — mark attendance from a group/classroom photo.
- 📷 **Live camera** — capture a webcam snapshot in the browser and recognize on the spot.
- 🧠 **Pretrained deep embeddings** — 128-d face vectors via dlib's ResNet (~99.4% LFW accuracy), no training.
- 🟩 **Clear visual feedback** — green box + name for recognized faces, red box + "Unknown" otherwise.
- 📝 **Attendance logging** — writes `Name, Date, Time` to `attendance.csv`.
- 🚫 **Duplicate prevention** — each person is logged at most once per day.
- 📱 **HEIC support** — iPhone `.heic`/`.heif` photos work everywhere.
- 📋 **Records view** — browse the attendance table and download the CSV.

---

## Tech Stack

| Component | Choice | Justification |
|-----------|--------|---------------|
| Face detection + recognition | `face_recognition` (dlib) | Pretrained ResNet embeddings; one-line detection/encoding API; runs on CPU. |
| Deep model | dlib pretrained ResNet (128-d) | High-accuracy embeddings (~99.4% on LFW) without any training. |
| Detector | HOG + linear SVM | Fast and dependency-light on CPU; no GPU needed. |
| Prebuilt dlib | `dlib-bin` | Precompiled wheel — no CMake / Visual Studio required to install. |
| UI | Streamlit | Pure-Python web UI: upload, webcam, tabs, tables, and downloads with minimal code. |
| iPhone photos | `pillow-heif` | Registers a HEIC/HEIF decoder into Pillow. |
| Image handling | Pillow, OpenCV, NumPy | Loading, array conversion, and drawing bounding boxes/labels. |
| Tabular storage | Pandas + CSV | Lightweight, human-readable attendance log; no database to set up. |

---

## How It Works

```
              ┌──────────────────────────────────────────────┐
              │                 ENROLLMENT                    │
              │  known_faces/<name>/*.jpg  (or in-app upload)  │
              │        │                                      │
              │        ▼                                      │
              │  HOG face detect ─► 128-d embedding (ResNet)  │
              │        │                                      │
              │        ▼                                      │
              │   encodings.pkl  (names + embeddings)         │
              └──────────────────────────────────────────────┘

              ┌──────────────────────────────────────────────┐
              │                 RECOGNITION                   │
              │  Upload photo  OR  webcam snapshot            │
              │        │                                      │
              │        ▼                                      │
              │  HOG detect ALL faces                         │
              │        │                                      │
              │        ▼                                      │
              │  128-d embedding per face                     │
              │        │                                      │
              │        ▼                                      │
              │  Euclidean distance vs enrolled embeddings    │
              │        │                                      │
              │   dist ≤ 0.5 ? ──► recognized (green box)     │
              │        else   ──► Unknown     (red box)       │
              │        │                                      │
              │        ▼                                      │
              │  Log to attendance.csv (1 row/person/day)     │
              └──────────────────────────────────────────────┘
```

---

## Setup

### 1. Clone

```bash
git clone <your-repo-url>
cd attendance-system
```

### 2. Create a virtual environment

> **Use Python 3.10 or 3.11.** Prebuilt `dlib-bin` wheels are reliable there; Python 3.13 may force a source build of dlib that needs a full C++ toolchain.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install dependencies (in this exact order)

```powershell
# a) Pure-wheel dependencies
pip install -r requirements.txt

# b) Prebuilt dlib (no CMake / Visual Studio needed)
pip install dlib-bin

# c) face_recognition WITHOUT deps, so pip never tries to COMPILE dlib
pip install --no-deps face_recognition_models face_recognition
```

> **Why `--no-deps`?** `face_recognition`'s metadata depends on the *source* package `dlib`, so a normal install makes pip try to compile dlib from source — which fails on Windows without a working MSVC compiler. `dlib-bin` is a precompiled drop-in that satisfies `import dlib`, and `--no-deps` stops pip from rebuilding it.

**Verify the install:**
```powershell
python -c "import dlib, face_recognition, streamlit, cv2, PIL, pandas, numpy; print('ALL IMPORTS OK')"
```

### 4. Enroll people

**Option A — In the app (no files needed):** launch the app (step 5), open the **👤 Enroll Person** tab, enter a name, upload photos, and click Enroll.

**Option B — Command line:** add photos under `known_faces/<name>/` and run:
```powershell
python enroll.py
```

### 5. Run the app

```powershell
streamlit run app.py
```
Your browser opens at `http://localhost:8501`.

---

## Project Structure

```
attendance-system/
├── known_faces/              # enrollment photos, one folder per person
│   └── .gitkeep
├── enroll.py                 # CLI: builds encodings.pkl from known_faces/
├── app.py                    # Streamlit app (Enroll / Mark / Live / Records)
├── app_deepface.py           # fallback app (DeepFace) if dlib won't install
├── requirements.txt
├── README.md
├── REPORT.md                 # full written report
├── PROJECT_EXPLANATION.md    # plain-English walkthrough of the whole project
├── .gitignore
└── sample_outputs/           # screenshots
    └── .gitkeep
```

---

## Usage Walkthrough

1. **Enroll** people via the **👤 Enroll Person** tab (or `python enroll.py`). The sidebar shows the enrolled count.
2. **Mark attendance** from the **✅ Mark Attendance** tab (upload an image) or the **📷 Live Camera** tab (webcam snapshot). Allow camera access when prompted.
3. The app draws a **green box** around each recognized person (name + distance) and a **red box** around unknown faces.
4. Each recognized person is written to `attendance.csv` — **once per day** (same-day duplicates are ignored).
5. Open the **📋 View Records** tab to see the full table and **download `attendance.csv`**.

---

## Sample Results

> _Insert your screenshots here (saved under `sample_outputs/`)._

**Enrollment tab:**
![Enroll](sample_outputs/enroll_example.png)

**Detection on a group photo:**
![Detection result](sample_outputs/detection_example.png)

**Attendance records tab:**
![Attendance table](sample_outputs/records_example.png)

---

## Limitations

- **HOG detector** struggles with small, heavily rotated, or profile faces and low-light images.
- Recognition accuracy depends on enrollment photo quality; few/poor photos increase errors.
- A single fixed threshold (`0.5`) trades off false positives vs. false negatives globally.
- The live camera uses **snapshot capture**, not continuous video (HOG per-frame is too slow on CPU).
- No liveness/anti-spoofing — a photo of a photo could be accepted.
- Designed for small groups (5–10); linear search over embeddings does not scale to thousands without an index.

---

## Future Work

- Swap HOG for a CNN/MTCNN detector for harder poses (at higher CPU cost).
- Use an ArcFace-based model for higher discriminative accuracy.
- Approximate nearest-neighbor index (e.g., FAISS) to scale to thousands of identities.
- Per-person adaptive thresholds and embedding averaging.
- Continuous live-video recognition and database-backed storage.
- Liveness detection for anti-spoofing.

---

## References

1. F. Schroff, D. Kalenichenko, and J. Philbin, "FaceNet: A Unified Embedding for Face Recognition and Clustering," *CVPR*, 2015.
2. J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," *CVPR*, 2019.
3. D. E. King, "Dlib-ml: A Machine Learning Toolkit," *Journal of Machine Learning Research*, vol. 10, pp. 1755–1758, 2009.
4. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *CVPR*, 2016.
5. N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human Detection," *CVPR*, 2005.

---

*Built as a 6th-semester (Section 2E) Computer Vision project. Uses pretrained models for inference only; no training or fine-tuning was performed.*
