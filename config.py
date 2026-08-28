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

# ---- Face recognition tuning -------------------------------------------------
# "hog" = fast, CPU friendly.  "cnn" = more accurate but needs a GPU to be usable.
FACE_DETECTION_MODEL = os.getenv("FACE_DETECTION_MODEL", "hog")
# Max face-embedding distance to still count as a match. Lower = stricter.
# 0.6 is the library default; 0.5 trades a few misses for far fewer false hits.
FACE_MATCH_TOLERANCE = float(os.getenv("FACE_MATCH_TOLERANCE", "0.5"))
# How many face images to capture per student during enrollment.
SAMPLES_PER_STUDENT = int(os.getenv("SAMPLES_PER_STUDENT", "20"))
# Run recognition on every Nth captured frame (keeps the video smooth).
RECOGNITION_INTERVAL = int(os.getenv("RECOGNITION_INTERVAL", "5"))
# Shrink frames before detection for speed (0.25 = quarter size).
FRAME_DOWNSCALE = float(os.getenv("FRAME_DOWNSCALE", "0.25"))

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
