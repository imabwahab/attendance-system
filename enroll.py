"""
enroll.py
=========
Builds the face database for the attendance system.

For every person folder inside `known_faces/`, this script:
  1. Reads each image in the folder.
  2. Detects the face (HOG detector) and computes a 128-d embedding
     using dlib's pretrained ResNet model (via face_recognition).
  3. Stores every embedding together with the person's name.
  4. Pickles the result to `encodings.pkl`, which app.py loads at runtime.

Run it once after adding/changing photos:

    python enroll.py

We store MULTIPLE embeddings per person (one per photo) instead of an
average. Keeping individual embeddings is more robust to pose/lighting
variation, because at recognition time we compare against the *closest*
sample rather than a blurred mean.
"""

import os
import pickle

import face_recognition

# Register a HEIC/HEIF decoder into Pillow so iPhone photos (.heic) can be
# read. face_recognition.load_image_file() opens images via Pillow, so once
# this is registered HEIC files work everywhere with no other code changes.
from pillow_heif import register_heif_opener
register_heif_opener()

KNOWN_FACES_DIR = "known_faces"
OUTPUT_FILE = "encodings.pkl"

# Image extensions we will attempt to enroll (HEIC/HEIF included for iPhone).
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic", ".heif")


def enroll():
    # Parallel lists: encodings[i] is the 128-d vector for the person named names[i].
    encodings = []
    names = []

    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"[ERROR] Folder '{KNOWN_FACES_DIR}' not found. Create it and add "
              f"one sub-folder per person, e.g. known_faces/alice/1.jpg")
        return

    # Each sub-folder of known_faces/ is one person; folder name = person name.
    people = sorted(
        d for d in os.listdir(KNOWN_FACES_DIR)
        if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))
    )

    if not people:
        print(f"[ERROR] No person folders found inside '{KNOWN_FACES_DIR}'.")
        return

    print(f"Found {len(people)} person folder(s): {', '.join(people)}\n")

    for person in people:
        person_dir = os.path.join(KNOWN_FACES_DIR, person)
        images = [f for f in os.listdir(person_dir)
                  if f.lower().endswith(VALID_EXTENSIONS)]

        if not images:
            print(f"[WARN] '{person}' has no images, skipping.")
            continue

        enrolled_for_person = 0
        for image_name in images:
            image_path = os.path.join(person_dir, image_name)

            # load_image_file returns an RGB numpy array (dlib expects RGB).
            image = face_recognition.load_image_file(image_path)

            # HOG detector: fast on CPU, no GPU needed. Good enough for
            # reasonably frontal enrollment photos.
            face_locations = face_recognition.face_locations(image, model="hog")

            if len(face_locations) == 0:
                # Gracefully skip photos where no face was found.
                print(f"  [WARN] No face detected in '{image_path}', skipping.")
                continue

            if len(face_locations) > 1:
                # Enrollment photos should contain exactly one person; if more
                # than one face is present we just use the first to stay simple.
                print(f"  [WARN] {len(face_locations)} faces in '{image_path}', "
                      f"using the first one.")

            # Compute the 128-d embedding for the (first) detected face.
            face_encs = face_recognition.face_encodings(
                image, known_face_locations=[face_locations[0]]
            )
            if not face_encs:
                print(f"  [WARN] Could not encode face in '{image_path}', skipping.")
                continue

            encodings.append(face_encs[0])
            names.append(person)
            enrolled_for_person += 1
            print(f"  [OK]   {image_path}")

        print(f"-> Enrolled {enrolled_for_person} image(s) for '{person}'\n")

    # Persist as a simple dict of two parallel lists.
    data = {"encodings": encodings, "names": names}
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)

    unique_people = sorted(set(names))
    print("=" * 50)
    print(f"Saved {len(encodings)} embedding(s) for "
          f"{len(unique_people)} person(s) to '{OUTPUT_FILE}'.")
    print(f"People enrolled: {', '.join(unique_people) if unique_people else '(none)'}")
    print("=" * 50)


if __name__ == "__main__":
    enroll()
