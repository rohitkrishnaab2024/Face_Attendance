# Smart Attendance System using Face Recognition

Automated classroom attendance: a webcam recognises enrolled students and writes
timestamped attendance rows into SQLite. Teachers use a Flask dashboard to enrol
students, run recognition, review records, and export/email reports.

**Stack:** Python · Flask · OpenCV · InsightFace (ArcFace / SCRFD) · SQLite

---

## How it works

```
Enroll            Train                      Recognise
------            -----                      ---------
webcam photos --> 512-d ArcFace embeddings --> live frame -> detect + embed faces
dataset/<id>_..   encodings/encodings.pkl    -> cosine similarity vs enrolled
                                             -> if sim >= threshold: mark present
```

InsightFace's `buffalo_l` pack (SCRFD detector + ArcFace recogniser, ONNX,
CPU-only here) replaces the older dlib `face_recognition`. Embeddings are 512-d
and L2-normalised, so matching is **cosine similarity** (higher = better), not
Euclidean distance. The `buffalo_l` model (~326 MB) downloads automatically to
`~/.insightface/models` on first run.

| File | Responsibility |
|------|----------------|
| `config.py`      | All tunables (similarity threshold, camera index, SMTP, paths). Reads `.env`. |
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

Notes:

- **`numpy>=2.1`** — numpy 1.26 has no Python 3.13 wheel and building it from
  source produces a broken install (`OverflowError` in `getlimits`).
- **`insightface==1.0.1` + `onnxruntime`** — ship prebuilt `cp313` wheels, so no
  compiler is needed. CPU inference only (no CUDA required).
- **First run downloads ~326 MB** (`buffalo_l` model) — needs internet once.

### Optional: email reports

```powershell
copy .env.example .env
# edit .env -> set SMTP_USER / SMTP_PASSWORD (Gmail: use an App Password)
# NOTE: put real secrets only in .env, never in .env.example (it's committed)
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
2. On the capture page, click **Auto-capture all 25** and follow the on-screen
   pose prompts. It stops automatically at the target.
3. **Students → Train model** — builds `encodings/encodings.pkl`. (Slower the
   first time — the model has to load.)
4. **Live Recognition → Start recognition** — matched faces get marked present.
   The same face is ignored for 5 minutes after marking (configurable).
5. **Records** — pick a date, review, add manual entries, **Download CSV** or
   **Send CSV by email**.

---

## Tuning (`.env` or environment variables)

| Variable | Default | Effect |
|----------|---------|--------|
| `FACE_SIM_THRESHOLD` | `0.35` | Min cosine similarity to accept a match. Higher = stricter. |
| `VOTE_TOP_K` | `3` | Student is scored by the mean of their K most similar samples. |
| `INSIGHTFACE_MODEL` | `buffalo_l` | `buffalo_s` is smaller/faster, slightly less accurate. |
| `FACE_DET_SIZE` | `640` | Detector input size. `320` is faster, shorter range. |
| `SAMPLES_PER_STUDENT` | `25` | Photos captured per student (auto-capture stops here). |
| `ATTENDANCE_COOLDOWN_SECONDS` | `300` | Re-mark suppression window per face. |
| `LATE_AFTER` | `09:15:00` | First-seen after this time → status `Late`. |
| `CAMERA_INDEX` | `0` | Change if the wrong camera opens. |
| `MIN_FACE_PIXELS` / `MIN_SHARPNESS` / `MIN_DET_SCORE` | `80` / `40` / `0.6` | Enrollment frames below these are rejected. |

---

## Accuracy notes

If recognition is poor, work through this list in order:

1. **Enrollment quality matters most.** Capture 20–30 photos per student with the
   face filling a good part of the frame, in the lighting you'll actually use,
   turning the head slightly between shots (left/right/up/down, glasses on and
   off). Blurry / tiny / multi-face / low-confidence frames are auto-rejected
   during capture.
2. **Re-train** (`Students → Train model`) after *any* change to students or
   photos. Recognition uses the last trained model only.
3. If people show as **Unknown**: lower `FACE_SIM_THRESHOLD` to `0.30`.
   If the **wrong** person matches: raise it to `0.40`–`0.45` and add photos.

How the engine helps:

- **ArcFace embeddings** (512-d) — state-of-the-art, ~99.8% on LFW, far more
  discriminative than the old 128-d dlib embeddings.
- **SCRFD detector** handles small, angled and partially-occluded faces well.
- Recognition scores each student by the **mean of their top-`VOTE_TOP_K` (3)
  similarities**, then accepts the best if it clears the threshold. Averaging the
  best few samples absorbs one unlucky photo, while still ranking purely on
  similarity — unlike counting samples above the threshold, where a student with
  more enrolled photos could outvote a genuinely closer match.
- Detections below `MIN_DET_SCORE` are ignored entirely.

With good enrollment you should clear 90% comfortably — often 95%+.

### Measured on this project's own dataset (3 students × 25 photos)

```
ACCURACY               : 100.0%   (75/75, leave-one-out)
Mean similarity (hits) : 0.876
Same-person similarity : mean 0.710, min 0.439
Cross-person similarity: mean 0.071, max 0.317
```

The gap between the worst same-person match (0.439) and the best cross-person
match (0.317) means the classes are **fully separable** — the 0.35 threshold sits
in empty space between them. Live webcam recognition scored 0.73–0.80 for the
enrolled subject against 0.07–0.20 for the other students.

### Measuring your accuracy

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

Runs **leave-one-out cross-validation** over your enrolled photos: each face is
hidden in turn and recognised against the rest, using the same voting logic as
the live system. Prints overall accuracy, per-student recall, a confusion list,
and mean match similarity. Needs ≥ 2 students with ~15+ photos each.

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
| First train/recognition hangs ~1 min | One-time `buffalo_l` model download (~326 MB). Needs internet. |
| Everyone shows as "Unknown" | You didn't train, or threshold too high — try `FACE_SIM_THRESHOLD=0.30`. |
| Wrong person matched | Raise `FACE_SIM_THRESHOLD` to `0.40`; capture more/better photos. |
| Email fails | Set `SMTP_USER`/`SMTP_PASSWORD`; for Gmail use an App Password, not your login. |
| Video feed frozen with `debug=True` | Already handled — `use_reloader=False` in `app.py`. |
| numpy `OverflowError` on import | `pip install "numpy>=2.1"`. |
| `onnxruntime` DLL load error | Install the [VC++ redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe). |
