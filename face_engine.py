"""Face detection, enrollment and recognition.

Pipeline
--------
1. Enrollment  : capture ~20 photos of a student  -> dataset/<id>_<name>/*.jpg
2. Training    : for every photo, compute a 128-number "face embedding"
                 (face_recognition.face_encodings) -> encodings/encodings.pkl
3. Recognition : for a live frame, embed each detected face and compare it to
                 every known embedding. Closest match within tolerance wins.
"""
import pickle

import cv2
import numpy as np
import face_recognition

import config


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #
def _safe(name):
    return "".join(c for c in name if c.isalnum() or c in " _").strip().replace(" ", "_")


def _dirname(student_id, name):
    return f"{student_id}_{_safe(name)}"


def student_dataset_dir(student_id, name):
    d = config.DATASET_DIR / _dirname(student_id, name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def count_samples(student_id, name):
    d = config.DATASET_DIR / _dirname(student_id, name)
    return len(list(d.glob("*.jpg"))) if d.exists() else 0


def save_face_sample(student_id, name, frame_bgr):
    """Save one enrollment photo IF exactly one clear face is present.

    Returns the saved Path, or None when no face was found.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model=config.FACE_DETECTION_MODEL)
    if not boxes:
        return None
    # keep the largest face (the person closest to the camera)
    top, right, bottom, left = max(boxes, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))
    d = student_dataset_dir(student_id, name)
    idx = count_samples(student_id, name)
    path = d / f"{idx:03d}.jpg"
    # store the full frame (not just the crop) so re-training can re-detect with
    # whatever model/settings are configured later.
    cv2.imwrite(str(path), frame_bgr)
    return path


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train():
    """(Re)build encodings.pkl from every image in dataset/.

    Returns (num_encodings, num_students).
    """
    encodings, ids, names = [], [], []
    for student_dir in sorted(config.DATASET_DIR.iterdir()):
        if not student_dir.is_dir() or "_" not in student_dir.name:
            continue
        try:
            sid = int(student_dir.name.split("_", 1)[0])
        except ValueError:
            continue
        display_name = student_dir.name.split("_", 1)[1].replace("_", " ")
        for img_path in sorted(student_dir.glob("*.jpg")):
            image = face_recognition.load_image_file(str(img_path))
            boxes = face_recognition.face_locations(image, model=config.FACE_DETECTION_MODEL)
            for enc in face_recognition.face_encodings(image, boxes):
                encodings.append(enc)
                ids.append(sid)
                names.append(display_name)

    with open(config.ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "ids": ids, "names": names}, f)
    return len(encodings), len(set(ids))


# --------------------------------------------------------------------------- #
# Recognition
# --------------------------------------------------------------------------- #
class FaceRecognizer:
    def __init__(self):
        self.encodings, self.ids, self.names = [], [], []
        self.load()

    def load(self):
        """Load encodings.pkl into memory (call again after re-training)."""
        self.encodings, self.ids, self.names = [], [], []
        if config.ENCODINGS_FILE.exists():
            with open(config.ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
            self.encodings = data["encodings"]
            self.ids = data["ids"]
            self.names = data["names"]

    @property
    def is_trained(self):
        return len(self.encodings) > 0

    def recognize(self, frame_bgr):
        """Return a list of {box, student_id, name, distance} for a BGR frame.

        box is (top, right, bottom, left) in FULL-resolution coordinates.
        student_id is None and name is 'Unknown' when nothing matched.
        """
        if not self.encodings:
            return []

        scale = config.FRAME_DOWNSCALE
        small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model=config.FACE_DETECTION_MODEL)
        encs = face_recognition.face_encodings(rgb, boxes)

        known = np.array(self.encodings)
        inv = 1.0 / scale
        results = []
        for (top, right, bottom, left), enc in zip(boxes, encs):
            distances = np.linalg.norm(known - enc, axis=1)
            best = int(np.argmin(distances))
            student_id, name, dist = None, "Unknown", None
            if distances[best] <= config.FACE_MATCH_TOLERANCE:
                student_id = self.ids[best]
                name = self.names[best]
                dist = float(distances[best])
            results.append({
                "box": (int(top * inv), int(right * inv), int(bottom * inv), int(left * inv)),
                "student_id": student_id,
                "name": name,
                "distance": dist,
            })
        return results
