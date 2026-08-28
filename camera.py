"""Single shared webcam.

Only ONE cv2.VideoCapture can own the webcam at a time, so this is a process-wide
singleton. A background thread continuously grabs frames; Flask routes read the
latest frame for MJPEG streaming and, when recognition is enabled, the thread
also runs face recognition and writes attendance rows.
"""
import threading
import time

import cv2

import config
import database
from face_engine import FaceRecognizer


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
        self.last_results = []          # boxes from the most recent recognition pass
        self.recent_marks = []          # small feed shown on the Live page
        self._frame_no = 0
        self._last_seen = {}            # student_id -> unix time last marked

    # ---- lifecycle --------------------------------------------------------- #
    def start(self):
        with self._lock:
            if self.running:
                return
            # CAP_DSHOW avoids the slow MSMF backend on Windows
            self.cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def stop(self):
        with self._lock:
            self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        self.cap, self.frame, self.thread = None, None, None

    # ---- capture loop ---------------------------------------------------- #
    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            self._frame_no += 1
            if self.recognition_on and self._frame_no % config.RECOGNITION_INTERVAL == 0:
                try:
                    self.last_results = self.recognizer.recognize(frame)
                    self._register(self.last_results)
                except Exception as exc:                       # keep the stream alive
                    print("[camera] recognition error:", exc)
            self.frame = frame
            time.sleep(0.01)

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

    # ---- frame output -------------------------------------------------- #
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
        """Yield an MJPEG multipart stream forever."""
        while True:
            if self.frame is None:
                time.sleep(0.05)
                continue
            frame = self.frame.copy()
            if self.recognition_on:
                frame = self._annotate(frame)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
            time.sleep(0.03)

    def snapshot(self):
        return None if self.frame is None else self.frame.copy()
