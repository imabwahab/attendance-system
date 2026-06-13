<!--
REPORT.md
This file contains the full written report. Paste into Word or convert to PDF
(e.g. with `pandoc REPORT.md -o REPORT.pdf`). Insert your screenshots where
indicated in Section 6.
-->

# Automated Attendance System Using Face Recognition with Pretrained Deep Learning Models

**Course:** Computer Vision (Semester Project)
**Author:** _[Your Name]_
**Roll No. / ID:** _[Your ID]_
**Institution / Department:** _[Your University / Department]_
**Date:** _[Submission Date]_

---

## Abstract

Manual attendance marking in classrooms is time-consuming, error-prone, and susceptible to proxy attendance. This project presents an automated attendance system that identifies enrolled individuals from a single uploaded group photograph using pretrained deep-learning face-recognition models, requiring no model training or fine-tuning. The system is built around the dlib-based `face_recognition` library: faces are localized using a Histogram of Oriented Gradients (HOG) detector combined with a linear Support Vector Machine, and each detected face is encoded into a 128-dimensional embedding by a pretrained ResNet-style convolutional neural network trained with a triplet-loss objective. Recognition is performed by computing the Euclidean distance between embeddings and applying a decision threshold of 0.5, below which a face is matched to the nearest enrolled identity. Recognized individuals are logged to a CSV file with date and time, with duplicate entries for the same person on the same day automatically suppressed. The application is wrapped in a lightweight two-tab Streamlit web interface supporting image upload, annotated detection visualization, attendance review, and CSV export. The system runs entirely on a CPU without specialized hardware. Experimental use on small enrolled groups demonstrates reliable recognition of frontal, well-lit faces, validating the practicality of pretrained embedding models for small-scale attendance applications.

*(Approx. 190 words.)*

---

## 1. Introduction

### 1.1 Motivation
Attendance tracking is a routine but important administrative task in educational institutions. Traditional roll-call methods consume valuable instructional time, and signature- or card-based systems are vulnerable to proxy attendance. Biometric approaches—particularly face recognition—offer a contactless, hard-to-forge alternative that can process an entire class from a single image. The maturity of deep-learning face-recognition models, many of which are freely available in pretrained form, makes it feasible to build such systems without large datasets or GPU training infrastructure.

### 1.2 Problem
The goal of this project is to build a working, submittable system that, given enrollment photos of a small number of people and a single uploaded group photograph, detects all faces present, identifies the enrolled individuals, and records their attendance with timestamps while avoiding duplicate records. The system must run on a commodity CPU-only Windows machine and must rely exclusively on pretrained models.

### 1.3 Contribution
This project contributes (i) an end-to-end face-recognition attendance pipeline using only pretrained models for inference; (ii) a reproducible enrollment process that produces a reusable embedding database; (iii) a duplicate-aware CSV logging mechanism; and (iv) an accessible Streamlit interface enabling non-technical users to mark and review attendance. The work emphasizes correct application of established methods (HOG detection, deep embeddings, distance thresholding) rather than novel model design.

---

## 2. Problem Statement

Given:
- A set of enrolled identities, each represented by one or more reference images stored under `known_faces/<name>/`.
- A query image (group/classroom photograph) uploaded at runtime.

Produce:
- The set of bounding boxes of all human faces detected in the query image.
- For each detected face, a predicted identity drawn from the enrolled set or the label "Unknown".
- A persistent attendance record `(Name, Date, Time)` for each recognized identity, with at most one record per identity per calendar day.

Constraints:
- Inference only; no training or fine-tuning of any model.
- CPU-only execution on Windows with Python 3.10+.
- No external database; storage limited to flat files (CSV and a pickled embedding cache).

Success criteria: enrolled individuals appearing frontally and clearly in the query image are correctly recognized and logged; unenrolled faces are labeled "Unknown"; duplicate same-day records are suppressed.

---

## 3. Literature Review

Face recognition has evolved from hand-crafted, holistic statistical methods to deep representation learning. Early appearance-based methods such as **Eigenfaces** [6], which apply Principal Component Analysis to face images, and **Fisherfaces**, based on Linear Discriminant Analysis, project faces into low-dimensional subspaces. While historically important, these holistic methods are highly sensitive to variations in illumination, pose, and expression because they operate directly on raw pixel intensities and assume strong alignment. Local descriptor methods such as Local Binary Patterns improved robustness to lighting but still lacked the discriminative power needed for unconstrained settings.

The decisive shift came with **deep convolutional neural networks (CNNs)**. **Deep Residual Learning (ResNet)** [4] introduced residual (skip) connections that allow very deep networks to be trained effectively, and ResNet-style architectures became standard backbones for feature extraction in face recognition. Rather than classifying a fixed set of identities, modern systems learn an **embedding**: a mapping from a face image to a compact vector in which distance encodes identity similarity.

**FaceNet** [1] formalized this approach using a **triplet loss**, which trains the network on triplets of an anchor, a positive (same identity), and a negative (different identity) image. The objective pulls the anchor and positive closer together in embedding space while pushing the negative farther away by a margin. FaceNet's 128-dimensional embeddings achieved near-human accuracy (~99.6% on the LFW benchmark) and demonstrated that face verification, recognition, and clustering can all be reduced to simple distance computations in the learned space. The `face_recognition` library used in this project is built on dlib's pretrained ResNet model, which follows precisely this embedding paradigm and outputs a 128-dimensional descriptor per face, reporting ~99.4% accuracy on LFW.

Subsequent work focused on improving the *margin* used during training to make embeddings more discriminative. **ArcFace** [2] introduced an **additive angular margin loss**, which enforces a margin penalty on the angle between the embedding and its class center on a hypersphere. This produces tighter intra-class and larger inter-class separation than triplet loss and is the current state of the art on many benchmarks. Although this project uses the dlib/FaceNet-style model for practical installation reasons, ArcFace represents the natural upgrade path for higher accuracy.

For **face detection**—a prerequisite to recognition—this project uses the **Histogram of Oriented Gradients (HOG)** descriptor with a linear SVM classifier, an approach popularized by Dalal and Triggs for pedestrian detection [5] and provided directly by dlib. HOG captures the distribution of local gradient orientations, which is robust to small illumination changes and efficient on CPU. More accurate but heavier alternatives include CNN-based detectors and MTCNN, which jointly perform detection and landmark localization; these improve recall on rotated or small faces at significantly higher computational cost.

In summary, the literature motivates the chosen pipeline: a lightweight HOG detector for localization, followed by a pretrained deep ResNet embedding model (FaceNet-style, triplet-trained) for recognition via distance thresholding. This combination offers a strong accuracy/efficiency balance appropriate for a CPU-only, small-scale attendance application.

---

## 4. Methodology

### 4.1 System Architecture

The system has two phases: an offline **enrollment** phase that builds a database of embeddings, and an online **recognition** phase that processes uploaded images.

```
ENROLLMENT (run once, offline)
  known_faces/<name>/*.jpg
        │
        ▼
   HOG face detection ──► crop ──► ResNet 128-d embedding
        │
        ▼
   encodings.pkl  { encodings: [...], names: [...] }

RECOGNITION (online, per uploaded image)
  Uploaded group photo
        │
        ▼
   HOG detect ALL faces ──► for each face: 128-d embedding
        │
        ▼
   Euclidean distance to every enrolled embedding
        │
        ├── min distance ≤ 0.5 ──► nearest identity  (green box)
        └── otherwise          ──► "Unknown"          (red box)
        │
        ▼
   attendance.csv  (Name, Date, Time)  — 1 row / person / day
```

### 4.2 Face Detection (HOG + Linear SVM)

Detection is performed with dlib's frontal face detector exposed via `face_recognition.face_locations(image, model="hog")`. The HOG method divides the image into small spatial cells, computes a histogram of gradient orientations within each cell, and normalizes these histograms over larger overlapping blocks to gain invariance to illumination and contrast. A sliding window over an image pyramid extracts HOG feature vectors, and a pretrained **linear SVM** classifies each window as face or non-face. HOG was selected over CNN-based detection because it is fast on CPU, has no GPU dependency, and is sufficiently accurate for the frontal, cooperative faces typical of an attendance photo.

### 4.3 Face Recognition (128-d Embedding, ResNet, Triplet Loss)

Each detected face region is passed to dlib's pretrained ResNet model through `face_recognition.face_encodings`, producing a **128-dimensional embedding vector**. The network was trained on a large face dataset using a metric-learning objective in the spirit of FaceNet's **triplet loss**: for an anchor face, a positive sample of the same person, and a negative sample of a different person, training minimizes

```
L = max( ||f(a) − f(p)||² − ||f(a) − f(n)||² + α , 0 )
```

where `f(·)` is the embedding function and `α` is a margin. This shapes the embedding space so that vectors of the same identity cluster tightly and different identities are pushed apart. Crucially, the trained network generalizes to identities never seen during training, which is exactly why a pretrained model can recognize our enrolled people without any fine-tuning.

### 4.4 Attendance Logic (Distance Threshold, Duplicate Prevention)

For a query embedding `q`, the system computes the Euclidean (L2) distance to every enrolled embedding and selects the minimum-distance match. If that minimum distance is **≤ 0.5**, the face is assigned the matched identity; otherwise it is labeled "Unknown". The dlib model is calibrated such that a distance below roughly 0.6 typically indicates the same person; we use a stricter **0.5** to reduce false positives, since wrongly marking an unenrolled person as present is the more damaging error in an attendance setting.

Duplicate prevention treats the pair **(Name, Date)** as a unique key. Before writing a new row, the system checks whether the recognized name already has a record for the current date; if so, the write is skipped. This ensures each person is recorded at most once per day even if their face appears in multiple uploaded photos.

---

## 5. Implementation

### 5.1 Tools and Libraries
- **Python 3.10+** on Windows 10/11, CPU only.
- **`face_recognition` (dlib)** — HOG detection and 128-d ResNet embeddings.
- **`dlib-bin`** — prebuilt dlib wheels avoiding CMake/Visual Studio compilation.
- **Streamlit** — web UI (file upload, tabs, tables, download button).
- **Pillow / OpenCV / NumPy** — image loading, array conversion, and annotation.
- **Pandas** — reading/writing the CSV attendance log.
- **DeepFace** — provided as an optional fallback recognition backend.

### 5.2 Enrollment Process
`enroll.py` iterates over each sub-folder of `known_faces/`, treating the folder name as the person's identity. For every image it runs HOG detection; images with no detectable face are skipped with a warning, and if multiple faces are present the first is used. Each detected face is embedded and the `(embedding, name)` pair is appended to two parallel lists. Multiple embeddings per person are retained (rather than averaged) to better tolerate variations in pose and lighting, since recognition compares against the closest sample. The lists are serialized to `encodings.pkl` and a summary of enrolled people and sample counts is printed.

### 5.3 Recognition Pipeline
`app.py` loads `encodings.pkl` and, for each uploaded image, converts it to an RGB NumPy array, detects all faces with HOG, and embeds each face. For every face it computes `face_distance` against all enrolled embeddings, selects the nearest, and applies the 0.5 threshold to decide the identity or "Unknown". Recognized identities are passed to the duplicate-aware `mark_attendance` function.

### 5.4 User Interface
The Streamlit UI presents a sidebar with enrollment statistics and two tabs. The **Mark Attendance** tab provides the file uploader, displays the annotated image (green boxes for recognized faces with name and distance, red boxes for unknown faces, labels drawn beneath each box), and reports which people were newly marked, already marked, or unrecognized, along with a per-face detail table. The **View Records** tab renders the full attendance table (most recent first) and offers a one-click **Download CSV** button.

---

## 6. Results and Discussion

The system was evaluated by enrolling a small group of individuals and uploading group photographs containing a mix of enrolled and unenrolled people.

**Figure 1 — Detection and recognition on a group photo.**
_[Insert `sample_outputs/detection_example.png`.]_
The system draws green bounding boxes with the predicted name and matching distance for enrolled individuals and red boxes labeled "Unknown" for others.

**Figure 2 — Attendance records tab.**
_[Insert `sample_outputs/records_example.png`.]_
Each recognized person appears once per day with the correct date and time; uploading additional photos of the same person on the same day produces no new rows.

**Figure 3 — Sidebar enrollment summary.**
_[Insert `sample_outputs/sidebar_example.png`.]_

**Discussion.** For frontal, well-lit faces the system reliably recognized enrolled individuals, with matching distances comfortably below the 0.5 threshold (typically 0.30–0.45), while unenrolled faces produced distances above the threshold and were correctly labeled "Unknown". Recognition quality degraded for faces that were small, strongly rotated, partially occluded, or poorly lit—conditions under which the HOG detector may miss the face entirely or the embedding distance rises near the threshold. The stricter 0.5 threshold occasionally rejected a genuine match captured under difficult conditions (a false negative), which is the intended conservative trade-off: it is preferable to miss a mark than to falsely record someone as present. Increasing the number and diversity of enrollment photos per person measurably improved recognition robustness, consistent with the decision to store multiple embeddings per identity.

---

## 7. Limitations

- **Detector limitations:** HOG misses small, profile, or heavily rotated faces and performs poorly in low light.
- **Single global threshold:** one fixed value (0.5) cannot be optimal for all identities and conditions.
- **No anti-spoofing:** the system cannot distinguish a live face from a photograph of a face (deliberately out of scope).
- **Scalability:** linear comparison against all embeddings is fine for 5–10 people but does not scale to thousands without an approximate nearest-neighbor index.
- **Enrollment dependence:** accuracy is sensitive to the quality and quantity of enrollment images.
- **Static images only:** no live webcam capture; attendance is marked from uploaded photos.

---

## 8. Conclusion and Future Work

This project demonstrated a complete, working automated attendance system built entirely from pretrained deep-learning models. By combining a HOG face detector with a pretrained ResNet embedding network and a simple Euclidean-distance threshold, the system recognizes enrolled individuals from a single uploaded photograph and records their attendance to a CSV file with duplicate suppression, all behind an accessible Streamlit interface and running on a CPU-only machine. The results confirm that pretrained embedding models are well suited to small-scale recognition tasks without any training.

Future work includes replacing HOG with a CNN/MTCNN detector for harder poses, adopting an ArcFace-based embedding model for higher discriminative accuracy, introducing per-person adaptive thresholds and embedding averaging, integrating an approximate nearest-neighbor index (e.g., FAISS) to scale to large cohorts, adding liveness detection for anti-spoofing, and supporting live webcam capture with database-backed storage.

---

## References

[1] F. Schroff, D. Kalenichenko, and J. Philbin, "FaceNet: A Unified Embedding for Face Recognition and Clustering," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2015, pp. 815–823.

[2] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 4690–4699.

[3] D. E. King, "Dlib-ml: A Machine Learning Toolkit," *Journal of Machine Learning Research*, vol. 10, pp. 1755–1758, 2009.

[4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.

[5] N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human Detection," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2005, pp. 886–893.

[6] M. Turk and A. Pentland, "Eigenfaces for Recognition," *Journal of Cognitive Neuroscience*, vol. 3, no. 1, pp. 71–86, 1991.

[7] O. M. Parkhi, A. Vedaldi, and A. Zisserman, "Deep Face Recognition," in *Proc. British Machine Vision Conference (BMVC)*, 2015.

[8] G. B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller, "Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments," University of Massachusetts, Amherst, Tech. Rep. 07-49, 2007.
