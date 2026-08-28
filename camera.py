"""Single shared webcam.

Only ONE cv2.VideoCapture can own the webcam at a time, so this is a process-wide
singleton. A background thread grabs frames ONLY while the camera is actually
needed - i.e. while a browser is watching the MJPEG feed or recognition is on.
As soon as nobody needs it (all tabs closed, recognition stopped) the thread
releases the device after a short grace period, so the webcam light goes off.
Call Camera().stop() on process shutdown for a clean release.
"""
import threading
import time

import cv2

import config
import database
from face_engine import FaceRecognizer

# How long to keep the device open after the last viewer leaves.
IDLE_RELEASE_SECONDS = 3


class Camera:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._setup()
            return cls._instance

    def _setup(self):
        self.cap = None
        self.frame = None
        self.running = False
        self.thread = None

        self.recognizer = FaceRecognizer()
        self.recognition_on = False
        self.last_results = []
        self.recent_marks = []

        self._frame_no = 0
        self._last_seen = {}          # student_id -> unix time last marked
        self._viewers = 0             # active MJPEG streams
        self._last_need = 0.0         # last time the camera was needed

    # ---- lifecycle ------------------------------------------------------- #
    def start(self):
        """Ensure the background thread is running (it manages the device)."""
        with self._lock:
            if self.running:
                return
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop the thread and release the webcam. Safe to call multiple times."""
        self.running = False
        self.recognition_on = False
        t = self.thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2)
        self._release_device()
        self.thread = None

    def _open_device(self):
        # CAP_DSHOW avoids the slow MSMF backend on Windows
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
        return cap if cap.isOpened() else None

    def _release_device(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.frame = None
        self.last_results = []

    def _needed(self):
        return self._viewers > 0 or self.recognition_on

    # ---- capture loop -------------------------------------------------- #
    def _loop(self):
        while self.running:
            if self._needed():
                self._last_need = time.time()
                if self.cap is None:
                    self.cap = self._open_device()
                    if self.cap is None:
                        time.sleep(0.5)
                        continue
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.1)
                    continue
                self._frame_no += 1
                if self.recognition_on and self._frame_no % config.RECOGNITION_INTERVAL == 0:
                    try:
                        self.last_results = self.recognizer.recognize(frame)
                        self._register(self.last_results)
                    except Exception as exc:              # keep the stream alive
                        print("[camera] recognition error:", exc)
                self.frame = frame
                time.sleep(0.01)
            else:
                # nobody needs the camera -> release it after the grace period
                if self.cap is not None and time.time() - self._last_need > IDLE_RELEASE_SECONDS:
                    self._release_device()
                time.sleep(0.2)

        self._release_device()

    def _register(self, results):
        now = time.time()
        for r in results:
            sid = r["student_id"]
            if not sid:
                continue
            if now - self._last_seen.get(sid, 0) < config.ATTENDANCE_COOLDOWN_SECONDS:
                continue
            self._last_seen[sid] = now
            is_new = database.mark_attendance(sid, late_after=config.LATE_AFTER)
            self.recent_marks.insert(0, {
                "name": r["name"],
                "time": time.strftime("%H:%M:%S"),
                "status": "marked" if is_new else "already present",
            })
            del self.recent_marks[12:]

    # ---- frame output ------------------------------------------------ #
    def _annotate(self, frame):
        for r in self.last_results:
            top, right, bottom, left = r["box"]
            matched = r["student_id"] is not None
            color = (0, 180, 0) if matched else (0, 0, 220)
            label = r["name"]
            if r["distance"] is not None:
                label += f"  {r['distance']:.2f}"
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 24), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 5, bottom - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def stream(self):
        """Yield an MJPEG multipart stream. Registers as a viewer for its lifetime."""
        self.start()
        self._viewers += 1
        try:
            idle = 0
            while self.running:
                if self.frame is None:
                    time.sleep(0.05)
                    idle += 1
                    if idle > 100:            # ~5s with no frame -> give up
                        break
                    continue
                idle = 0
                frame = self.frame.copy()
                if self.recognition_on:
                    frame = self._annotate(frame)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + buf.tobytes() + b"\r\n")
                time.sleep(0.03)
        finally:
            self._viewers = max(0, self._viewers - 1)

    def snapshot(self):
        """Return the latest frame, briefly opening the camera if needed."""
        self.start()
        self._viewers += 1                    # count as a viewer while we wait
        try:
            for _ in range(60):               # wait up to ~3s for a frame
                if self.frame is not None:
                    return self.frame.copy()
                time.sleep(0.05)
            return None
        finally:
            self._viewers = max(0, self._viewers - 1)
