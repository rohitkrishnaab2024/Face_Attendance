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
    """Save one enrollment photo if a single, large, sharp face is present.

    Returns (saved_path_or_None, reason_message).
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model=config.FACE_DETECTION_MODEL)
    if not boxes:
        return None, "No face detected - face the camera."
    if len(boxes) > 1:
        return None, "More than one face in view."

    top, right, bottom, left = boxes[0]
    face_h = bottom - top
    if face_h < config.MIN_FACE_PIXELS:
        return None, "Move closer - your face is too small."

    # Laplacian variance is a cheap blur metric; low = blurry.
    crop = cv2.cvtColor(frame_bgr[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(crop, cv2.CV_64F).var() < config.MIN_SHARPNESS:
        return None, "Too blurry - hold still."

    d = student_dataset_dir(student_id, name)
    path = d / f"{count_samples(student_id, name):03d}.jpg"
    cv2.imwrite(str(path), frame_bgr)   # full frame, so training can re-detect
    return path, "Sample saved."


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
            # num_jitters re-samples each face a few times (small random shifts /
            # flips) and averages -> a more stable enrollment embedding.
            for enc in face_recognition.face_encodings(
                image, boxes, num_jitters=config.TRAIN_JITTERS
            ):
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

        Detection runs on a downscaled copy (fast); the face embedding is then
        computed on the FULL-resolution frame so it is directly comparable to the
        full-resolution embeddings built during training.
        """
        if not self.encodings:
            return []

        scale = config.FRAME_DOWNSCALE
        small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)
        boxes_small = face_recognition.face_locations(
            cv2.cvtColor(small, cv2.COLOR_BGR2RGB), model=config.FACE_DETECTION_MODEL
        )
        if not boxes_small:
            return []

        inv = 1.0 / scale
        h, w = frame_bgr.shape[:2]
        boxes_full = [
            (max(0, int(t * inv)), min(w, int(r * inv)),
             min(h, int(b * inv)), max(0, int(l * inv)))
            for (t, r, b, l) in boxes_small
        ]
        rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb_full, boxes_full, num_jitters=1)

        known = np.asarray(self.encodings)
        ids = np.asarray(self.ids)
        results = []
        for box, enc in zip(boxes_full, encs):
            distances = np.linalg.norm(known - enc, axis=1)
            student_id, name, dist = self._vote(distances, ids)
            results.append({
                "box": box,
                "student_id": student_id,
                "name": name or "Unknown",
                "distance": dist,
            })
        return results

    def _vote(self, distances, ids):
        """Among all enrolled samples within tolerance, pick the student with the
        most matching samples (ties broken by smallest distance). This is far more
        robust than trusting a single nearest neighbour."""
        within = np.where(distances <= config.FACE_MATCH_TOLERANCE)[0]
        if len(within) == 0:
            return None, None, None
        cand_ids = ids[within]
        best = None  # (votes, -min_distance, student_id, min_distance)
        for sid in set(cand_ids.tolist()):
            idx = within[cand_ids == sid]
            dmin = float(distances[idx].min())
            score = (len(idx), -dmin)
            if best is None or score > best[:2]:
                best = (len(idx), -dmin, sid, dmin)
        sid, dmin = best[2], best[3]
        return sid, self.names[self.ids.index(sid)], dmin
