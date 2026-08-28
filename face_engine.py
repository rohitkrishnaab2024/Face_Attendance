"""Face detection and recognition using InsightFace.

InsightFace bundles an SCRFD face detector and an ArcFace recognition model
(the `buffalo_l` pack). Each face becomes a 512-number, L2-normalised embedding,
so two faces are compared with **cosine similarity**: 1.0 = identical, higher =
more alike. A match is accepted when similarity >= FACE_SIM_THRESHOLD.

Pipeline
--------
1. Enrollment  : capture ~25 photos of a student -> dataset/<id>_<name>/*.jpg
2. Training    : embed every photo               -> encodings/encodings.pkl
3. Recognition : embed each face in a live frame -> nearest enrolled student
                 by cosine similarity, with multi-sample voting.

The public API (save_face_sample / train / FaceRecognizer.recognize) is
unchanged from the previous dlib-based engine, so the rest of the app did not
need to change. `recognize()` still returns a `distance` key (= 1 - similarity,
lower is better) plus a new `similarity` key.
"""
import pickle
import threading

import cv2
import numpy as np

import config

_APP = None
_APP_LOCK = threading.Lock()


def _app():
    """Lazily create the shared InsightFace model (first call downloads ~300 MB)."""
    global _APP
    if _APP is None:
        with _APP_LOCK:
            if _APP is None:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(
                    name=config.INSIGHTFACE_MODEL,
                    allowed_modules=["detection", "recognition"],
                    providers=["CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0,
                            det_size=(config.FACE_DET_SIZE, config.FACE_DET_SIZE))
                _APP = app
    return _APP


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


def _largest(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


# --------------------------------------------------------------------------- #
# Enrollment
# --------------------------------------------------------------------------- #
def save_face_sample(student_id, name, frame_bgr):
    """Save one enrollment photo if a single, large, confident face is present.

    Returns (saved_path_or_None, reason_message).
    """
    faces = _app().get(frame_bgr)
    if not faces:
        return None, "No face detected - face the camera."
    if len(faces) > 1:
        return None, "More than one face in view."

    f = faces[0]
    x1, y1, x2, y2 = f.bbox
    if (y2 - y1) < config.MIN_FACE_PIXELS:
        return None, "Move closer - your face is too small."
    if float(f.det_score) < config.MIN_DET_SCORE:
        return None, "Low-quality detection - fix lighting or face the camera."

    crop = cv2.cvtColor(
        frame_bgr[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)], cv2.COLOR_BGR2GRAY
    )
    if crop.size and cv2.Laplacian(crop, cv2.CV_64F).var() < config.MIN_SHARPNESS:
        return None, "Too blurry - hold still."

    d = student_dataset_dir(student_id, name)
    path = d / f"{count_samples(student_id, name):03d}.jpg"
    cv2.imwrite(str(path), frame_bgr)
    return path, "Sample saved."


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train():
    """(Re)build encodings.pkl from every image in dataset/.

    Returns (num_embeddings, num_students).
    """
    embeds, ids, names = [], [], []
    for student_dir in sorted(config.DATASET_DIR.iterdir()):
        if not student_dir.is_dir() or "_" not in student_dir.name:
            continue
        try:
            sid = int(student_dir.name.split("_", 1)[0])
        except ValueError:
            continue
        display_name = student_dir.name.split("_", 1)[1].replace("_", " ")
        for img_path in sorted(student_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            f = _largest(_app().get(img))
            if f is None:
                continue
            embeds.append(np.asarray(f.normed_embedding, dtype=np.float32))
            ids.append(sid)
            names.append(display_name)

    with open(config.ENCODINGS_FILE, "wb") as fh:
        pickle.dump({"encodings": embeds, "ids": ids, "names": names}, fh)
    return len(embeds), len(set(ids))


# --------------------------------------------------------------------------- #
# Recognition
# --------------------------------------------------------------------------- #
class FaceRecognizer:
    def __init__(self):
        self.encodings = np.zeros((0, 512), dtype=np.float32)
        self.ids, self.names = [], []
        self.load()

    def load(self):
        """Load encodings.pkl into memory (call again after re-training)."""
        self.encodings = np.zeros((0, 512), dtype=np.float32)
        self.ids, self.names = [], []
        if config.ENCODINGS_FILE.exists():
            with open(config.ENCODINGS_FILE, "rb") as fh:
                data = pickle.load(fh)
            if data["encodings"]:
                self.encodings = np.vstack(data["encodings"]).astype(np.float32)
            self.ids = list(data["ids"])
            self.names = list(data["names"])

    @property
    def is_trained(self):
        return len(self.ids) > 0

    def recognize(self, frame_bgr):
        """Return a list of {box, student_id, name, distance, similarity}.

        box is (top, right, bottom, left) in full-resolution pixels.
        student_id is None / name 'Unknown' when nothing matched.
        """
        if not self.is_trained:
            return []

        results = []
        ids = np.asarray(self.ids)
        for f in _app().get(frame_bgr):
            if float(f.det_score) < config.MIN_DET_SCORE:
                continue                            # ignore junk detections
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
            sims = self.encodings @ emb            # cosine sim (both L2-normalised)
            student_id, name, sim = self._vote(sims, ids)
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            results.append({
                "box": (y1, x2, y2, x1),
                "student_id": student_id,
                "name": name or "Unknown",
                "similarity": None if sim is None else round(float(sim), 3),
                "distance": None if sim is None else round(1.0 - float(sim), 3),
            })
        return results

    def _vote(self, sims, ids):
        """Score each student by the MEAN OF THEIR TOP-K similarities, then take
        the best student if that score clears the threshold.

        Averaging the k best samples is robust to one lucky/unlucky photo, while
        still ranking purely on similarity - unlike counting samples above the
        threshold, where a student with more enrolled photos could outvote a
        genuinely closer match.
        """
        best_sid, best_score = None, -1.0
        for sid in set(self.ids):
            s = sims[ids == sid]
            k = min(config.VOTE_TOP_K, len(s))
            score = float(np.sort(s)[-k:].mean())
            if score > best_score:
                best_score, best_sid = score, sid
        if best_sid is None or best_score < config.FACE_SIM_THRESHOLD:
            return None, None, None
        return best_sid, self.names[self.ids.index(best_sid)], best_score
