"""Central configuration. Everything is overridable with environment variables
(loaded from a .env file if present)."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional
    pass

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"        # raw face images, one folder per student
ENCODINGS_DIR = BASE_DIR / "encodings"    # trained face embeddings (pickle)
ENCODINGS_FILE = ENCODINGS_DIR / "encodings.pkl"
EXPORT_DIR = BASE_DIR / "exports"         # generated CSV reports
DB_PATH = BASE_DIR / "attendance.db"

for _d in (DATASET_DIR, ENCODINGS_DIR, EXPORT_DIR):
    _d.mkdir(exist_ok=True)

# ---- Face recognition (InsightFace / ArcFace) ------------------------------
# Model pack: "buffalo_l" (accurate, ~326 MB) or "buffalo_s" (smaller/faster).
# Downloaded automatically to ~/.insightface/models on first use.
INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
# Detector input size. 640 = good range; 320 = faster, shorter range.
FACE_DET_SIZE = int(os.getenv("FACE_DET_SIZE", "640"))
# Minimum cosine similarity (0-1) between embeddings to accept a match.
# ~0.30 is loose, ~0.40 is strict. 0.35 is a good default with voting.
FACE_SIM_THRESHOLD = float(os.getenv("FACE_SIM_THRESHOLD", "0.35"))
# A student is scored by the mean of their K most similar enrolled samples.
VOTE_TOP_K = int(os.getenv("VOTE_TOP_K", "3"))
# How many face images to capture per student during enrollment.
SAMPLES_PER_STUDENT = int(os.getenv("SAMPLES_PER_STUDENT", "25"))
# Reject enrollment frames whose face is too small / blurry / weakly detected.
MIN_FACE_PIXELS = int(os.getenv("MIN_FACE_PIXELS", "80"))
MIN_SHARPNESS = float(os.getenv("MIN_SHARPNESS", "40"))
MIN_DET_SCORE = float(os.getenv("MIN_DET_SCORE", "0.6"))
# Run recognition on every Nth captured frame (keeps the video smooth).
RECOGNITION_INTERVAL = int(os.getenv("RECOGNITION_INTERVAL", "5"))

# ---- Attendance rules ------------------------------------------------------
# After a student is marked, ignore that same face for this many seconds so one
# person standing in front of the camera does not create dozens of DB writes.
ATTENDANCE_COOLDOWN_SECONDS = int(os.getenv("ATTENDANCE_COOLDOWN_SECONDS", "300"))
# Anyone first seen after this clock time is stored with status "Late".
LATE_AFTER = os.getenv("LATE_AFTER", "09:15:00")

# ---- Camera ---------------------------------------------------------------
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# ---- Flask --------------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")

# ---- Email / SMTP (optional) --------------------------------------------
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "") or SMTP_USER
