# Smart Attendance System using Face Recognition

Automated classroom attendance: a webcam recognises enrolled students and writes
timestamped attendance rows into SQLite. Teachers use a Flask dashboard to enrol
students, run recognition, review records, and export/email reports.

**Stack:** Python · Flask · OpenCV · `face_recognition` (dlib) · SQLite

---

## How it works

```
Enroll            Train                     Recognise
------            -----                     ---------
webcam photos --> 128-d face embeddings --> live frame -> embed faces
dataset/<id>_..   encodings/encodings.pkl   -> nearest known embedding
                                            -> if distance <= tolerance: mark present
```

| File | Responsibility |
|------|----------------|
| `config.py`      | All tunables (tolerance, camera index, SMTP, paths). Reads `.env`. |
| `database.py`    | SQLite schema + queries. `UNIQUE(student_id, date)` = one mark per day. |
| `face_engine.py` | Save enrollment photos, `train()` embeddings, `FaceRecognizer.recognize()`. |
| `camera.py`      | One shared webcam. Background thread grabs frames, runs recognition, marks attendance. |
| `exporter.py`    | Build the CSV roster, email it via SMTP. |
| `app.py`         | Flask routes + MJPEG video feed. |
| `evaluate.py`    | Leave-one-out accuracy measurement over the enrolled dataset. |
| `templates/`, `static/` | The teacher dashboard UI. |

Data written at runtime (all git-ignored): `dataset/`, `encodings/`,
`exports/`, `attendance.db`.

---

## Setup (Windows)

A working virtual environment is already created at `.venv` (Python 3.13, all
deps installed and verified). To recreate it from scratch:

```powershell
cd "c:\Projects\Face Recognition"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Notes baked into `requirements.txt` (learned the hard way on Python 3.13):

- **`numpy>=2.1`** — numpy 1.26 has no 3.13 wheel and building it from source
  produces a broken install (`OverflowError` in `getlimits`).
- **`setuptools<80`** — `face_recognition_models` still imports `pkg_resources`,
  which setuptools 80+ removed. Without this you get
  *"Please install `face_recognition_models`"* even though it is installed.
- **`dlib==20.0.1`** — ships a prebuilt `cp313` Windows wheel, so no compiler is
  needed.

**If `dlib` has no wheel for your Python** (e.g. an even newer version), use
Python 3.11 or 3.12, or install a community wheel:
`pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.99-cp312-cp312-win_amd64.whl`
(match the `cpXX` tag to your Python version).

### Optional: email reports

```powershell
copy .env.example .env
# edit .env -> set SMTP_USER / SMTP_PASSWORD (Gmail: use an App Password)
```

---

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Open <http://127.0.0.1:5000>.

### Typical flow

1. **Students → Add student** — enter roll number + name (+ email for reports).
2. On the capture page, click **Auto-capture ×10** twice (~20 photos). Turn your
   head slightly between shots.
3. **Students → Train model** — builds `encodings/encodings.pkl`.
4. **Live Recognition → Start recognition** — matched faces get marked present.
   The same face is ignored for 5 minutes after marking (configurable).
5. **Records** — pick a date, review, add manual entries, **Download CSV** or
   **Send CSV by email**.

---

## Tuning (`.env` or environment variables)

| Variable | Default | Effect |
|----------|---------|--------|
| `FACE_MATCH_TOLERANCE` | `0.55` | Lower = stricter (fewer false matches, more misses). |
| `FACE_DETECTION_MODEL` | `hog` | `cnn` is more accurate but needs a GPU. |
| `SAMPLES_PER_STUDENT` | `25` | Photos captured per student. |
| `TRAIN_JITTERS` | `4` | Re-samples per photo when training (higher = slower, stabler). |
| `ATTENDANCE_COOLDOWN_SECONDS` | `300` | Re-mark suppression window per face. |
| `LATE_AFTER` | `09:15:00` | First-seen after this time → status `Late`. |
| `CAMERA_INDEX` | `0` | Change if the wrong camera opens. |
| `FRAME_DOWNSCALE` | `0.5` | Detection-only downscale. Smaller = faster, worse on far faces. |
| `MIN_FACE_PIXELS` / `MIN_SHARPNESS` | `80` / `40` | Enrollment frames below these are rejected. |

---

## Accuracy notes

If recognition is poor, work through this list in order:

1. **Enrollment quality is 80% of the result.** Capture 20–30 photos per student
   with the face filling a good part of the frame, in the lighting you'll
   actually use, turning the head slightly between shots (left/right/up/down,
   glasses on and off). Blurry / tiny / multi-face frames are now auto-rejected
   during capture.
2. **Re-train** (`Students → Train model`) after *any* change to students or
   photos. Recognition uses the last trained model only.
3. If people show as **Unknown**: raise `FACE_MATCH_TOLERANCE` to `0.58`–`0.6`.
   If the **wrong** person matches: lower it to `0.5` and capture more photos.
4. For a one-off, higher-accuracy enrollment set `FACE_DETECTION_MODEL=cnn`
   (slow on CPU, ~1–3 s/photo) while training, then switch back to `hog` for
   live use.

How the engine already helps:

- Detection runs on a half-size frame (fast); the **embedding is computed on the
  full-resolution frame**, matching how training embeddings are built.
- Training averages `TRAIN_JITTERS` (4) re-samples per photo for stabler
  embeddings.
- Recognition uses **multi-sample voting**: the student with the most enrolled
  samples within tolerance wins, not a single nearest neighbour.

`face_recognition` reports ~99.4% on the LFW benchmark; with good enrollment you
should comfortably clear 90% in a classroom.

### Measuring your accuracy

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

Runs **leave-one-out cross-validation** over your enrolled photos: each face is
hidden in turn and recognised against the rest, using the same voting logic as
the live system. Prints overall accuracy, per-student recall, a confusion list,
and mean match distance. Needs ≥ 2 students with ~15+ photos each.

Optional: drop photos of non-enrolled people into `eval_impostors/` and the
script also reports the **false-accept rate** (how often a stranger is let
through).

This measures enrollment-quality images, so it is a best case — live webcam is
usually a few points lower. Quote it honestly, e.g. *"~95% leave-one-out on an
N-person enrolled set; ~90% in live webcam testing."*

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Black video / "camera not ready" | Close other apps using the webcam; try `CAMERA_INDEX=1`. |
| `dlib` install fails | Use Python 3.11/3.12 or a prebuilt wheel (see Setup). |
| Everyone shows as "Unknown" | You didn't train, or tolerance is too low — try `0.55`. |
| Wrong person matched | Lower `FACE_MATCH_TOLERANCE`; capture more/better photos. |
| Email fails | Set `SMTP_USER`/`SMTP_PASSWORD`; for Gmail use an App Password, not your login. |
| Video feed frozen with `debug=True` | Already handled — `use_reloader=False` in `app.py`. |
| `Please install face_recognition_models` | `pip install "setuptools<80"` (see Setup notes). |
| numpy `OverflowError` on import | `pip install "numpy>=2.1"` (see Setup notes). |
