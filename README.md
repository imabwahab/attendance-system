# Automated Attendance System Using Face Recognition

> Mark classroom attendance automatically from a single group photo using pretrained deep-learning face embeddings — no model training required.

## Overview

This project implements an automated attendance system based on **face detection and face recognition** using **pretrained deep learning models**. A user enrolls a small set of people (5–10) by providing a few photos of each, then uploads a group/classroom photo. The system detects every face in the photo, recognizes the enrolled individuals, and logs each recognized person to a CSV attendance sheet with the date and time.

The recognition engine is the [`face_recognition`](https://github.com/ageitgey/face_recognition) library, which wraps **dlib**. Faces are located with a **HOG (Histogram of Oriented Gradients) + linear SVM** detector, and each face is converted into a **128-dimensional embedding** by a pretrained ResNet-style convolutional network trained with a triplet-loss objective. Identification is performed by comparing embeddings using **Euclidean distance** against a threshold of `0.5`.

The entire application runs on a **CPU-only Windows machine**, uses **no database** (attendance is stored in a plain CSV file), and is wrapped in a simple two-tab **Streamlit** web interface. No models are trained or fine-tuned — all deep learning weights are pretrained and used purely for inference.

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Face detection + recognition | `face_recognition` (dlib) | Battle-tested, pretrained ResNet embeddings; one-line detection/encoding API; runs on CPU. |
| Deep model | dlib pretrained ResNet (128-d) | High-accuracy embeddings (~99.4% on LFW) without any training. |
| Detector | HOG + linear SVM | Fast and dependency-light on CPU; no GPU needed. |
| UI | Streamlit | Pure-Python web UI; file upload, tabs, tables, and download buttons with minimal code. |
| Image handling | Pillow, OpenCV, NumPy | Loading, array conversion, and drawing bounding boxes/labels. |
| Tabular storage | Pandas + CSV | Lightweight, human-readable attendance log; no database to set up. |

## How It Works

```
              ┌──────────────────────────────────────────────┐
              │                 ENROLLMENT                    │
              │  known_faces/<name>/*.jpg                     │
              │        │                                      │
              │        ▼                                      │
              │  HOG face detect ─► 128-d embedding (ResNet)  │
              │        │                                      │
              │        ▼                                      │
              │   encodings.pkl  (names + embeddings)         │
              └──────────────────────────────────────────────┘

              ┌──────────────────────────────────────────────┐
              │                 RECOGNITION                   │
              │  Upload group photo (Streamlit)               │
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

## Setup

### 1. Clone

```bash
git clone <your-repo-url>
cd attendance-system
```

### 2. Create a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install dependencies

> **Use Python 3.10 or 3.11.** Prebuilt `dlib-bin` wheels are most reliable there; Python 3.13 may force a source build of dlib (which needs a full MSVC toolchain and often fails).

Install in this exact order:

```powershell
# a) Pure-wheel dependencies
pip install -r requirements.txt

# b) Prebuilt dlib (no CMake / Visual Studio needed)
pip install dlib-bin

# c) face_recognition WITHOUT deps, so pip never tries to COMPILE dlib.
#    dlib-bin already provides `import dlib`; --no-deps avoids the source build.
pip install --no-deps face_recognition_models face_recognition
```

> **Why `--no-deps`?** `face_recognition`'s metadata depends on the *source* package `dlib`, so a normal install makes pip try to compile dlib from source — which fails on Windows without a working MSVC compiler. `dlib-bin` is a precompiled drop-in that satisfies `import dlib`, and `--no-deps` stops pip from rebuilding it.

Verify the install:

```powershell
python -c "import face_recognition, streamlit, cv2, PIL, pandas, numpy; print('OK')"
```

### 4. Enroll people

Create one folder per person inside `known_faces/` and add a few clear, mostly-frontal photos:

```
known_faces/
  alice/  1.jpg  2.jpg
  bob/    1.jpg
  ...
```

Then build the embedding database:

```powershell
python enroll.py
```

### 5. Run the app

```powershell
streamlit run app.py
```

Your browser opens at `http://localhost:8501`.

## Project Structure

```
attendance-system/
├── known_faces/          # enrollment photos, one folder per person
│   └── .gitkeep
├── enroll.py             # builds encodings.pkl from known_faces/
├── app.py                # Streamlit app (face_recognition / dlib)
├── app_deepface.py       # fallback app (DeepFace) if dlib won't install
├── requirements.txt
├── README.md
├── REPORT.md             # full written report
├── .gitignore
└── sample_outputs/       # screenshots
    └── .gitkeep
```

## Usage Walkthrough

1. **Enroll** people (`python enroll.py`). The sidebar in the app shows the enrolled count.
2. Open the **Mark Attendance** tab and upload a group photo (`jpg/jpeg/png`).
3. The app draws a **green box** around each recognized person (with name + distance) and a **red box** around unknown faces.
4. Each recognized person is written to `attendance.csv` — **once per day** (duplicates on the same day are ignored).
5. Open the **View Records** tab to see the full attendance table and **download `attendance.csv`**.

## Sample Results

> _Insert your screenshots here (saved under `sample_outputs/`)._

**Detection on a group photo:**

![Detection result](sample_outputs/detection_example.png)

**Attendance records tab:**

![Attendance table](sample_outputs/records_example.png)

## Limitations

- **HOG detector** struggles with small, heavily rotated, or profile faces and low-light images.
- Recognition accuracy depends on enrollment photo quality; few/poor photos increase errors.
- A single fixed threshold (`0.5`) trades off false positives vs. false negatives globally.
- No liveness/anti-spoofing — a photo of a photo could be accepted (out of scope).
- Designed for small groups (5–10); linear search over embeddings does not scale to thousands without an index.

## Future Work

- Swap HOG for a CNN/MTCNN detector for harder poses (at higher CPU cost).
- Use an approximate nearest-neighbor index (e.g., FAISS) to scale to thousands of identities.
- Per-person adaptive thresholds and multiple-embedding averaging.
- Webcam/live-feed mode and database-backed storage.
- Liveness detection for anti-spoofing.

## References

1. F. Schroff, D. Kalenichenko, and J. Philbin, "FaceNet: A Unified Embedding for Face Recognition and Clustering," *CVPR*, 2015.
2. J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," *CVPR*, 2019.
3. D. E. King, "Dlib-ml: A Machine Learning Toolkit," *Journal of Machine Learning Research*, vol. 10, pp. 1755–1758, 2009.
4. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *CVPR*, 2016.
5. N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human Detection," *CVPR*, 2005.

---

*Built as a computer vision semester project. Uses pretrained models for inference only; no training or fine-tuning was performed.*
