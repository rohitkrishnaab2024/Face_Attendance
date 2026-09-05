"""Single shared webcam.

Only ONE cv2.VideoCapture can own the webcam at a time, so this is a process-wide
singleton. Two daemon threads run:

  capture thread     - grabs frames as fast as the webcam allows and publishes
                       the latest one. Nothing slow ever runs here, so the MJPEG
                       stream stays smooth.
  recognition thread - picks up the most recent frame and runs InsightFace on it
                       (~300-450 ms per pass on CPU), then publishes the boxes.

Keeping recognition OFF the capture thread is what stops the video stuttering:
previously every inference pass froze frame-grabbing for the whole ~450 ms.
The drawn boxes therefore lag the live image by up to one inference, which is
invisible at normal movement speed.

The camera is opened only while it is needed - a browser is watching the feed or
recognition is on - and released a few seconds after the last viewer leaves, so
the webcam light goes off. Call Camera().stop() on process shutdown.
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
        self.threads = []

        self.recognizer = FaceRecognizer()
        self.recognition_on = False
        self.last_results = []
        self.recent_marks = []

        self._last_seen = {}          # student_id -> unix time last marked
        self._viewers = 0             # active MJPEG streams
        self._last_need = 0.0         # last time the camera was needed
        self._infer_ms = 0.0          # most recent inference cost (ms)

    # ---- lifecycle ------------------------------------------------------- #
    def start(self):
        """Ensure the worker threads are running (they manage the device)."""
        with self._lock:
            if self.running:
                return
            self.running = True
            self.threads = [
                threading.Thread(target=self._capture_loop, daemon=True),
                threading.Thread(target=self._recognition_loop, daemon=True),
            ]
            for t in self.threads:
                t.start()

    def stop(self):
        """Stop the threads and release the webcam. Safe to call repeatedly."""
        self.running = False
        self.recognition_on = False
        for t in self.threads:
            if t.is_alive() and t is not threading.current_thread():
                t.join(timeout=2)
        self._release_device()
        self.threads = []

    def _open_device(self):
        # CAP_DSHOW avoids the slow MSMF backend on Windows
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not cap.isOpened():
            return None
        # a 1-frame buffer keeps the stream close to real time
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _release_device(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.frame = None
        self.last_results = []

    def _needed(self):
        return self._viewers > 0 or self.recognition_on

    # ---- capture thread (must stay fast) --------------------------------- #
    def _capture_loop(self):
        while self.running:
            if not self._needed():
                if self.cap is not None and time.time() - self._last_need > IDLE_RELEASE_SECONDS:
                    self._release_device()
                time.sleep(0.2)
                continue

            self._last_need = time.time()
            if self.cap is None:
                self.cap = self._open_device()
                if self.cap is None:
                    time.sleep(0.5)
                    continue

            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            self.frame = frame

        self._release_device()

    # ---- recognition thread (slow work lives here) ----------------------- #
    def _recognition_loop(self):
        last_id = None
        while self.running:
            if not self.recognition_on:
                self.last_results = []
                time.sleep(0.1)
                continue

            frame = self.frame
            if frame is None or id(frame) == last_id:
                time.sleep(0.02)              # nothing new to look at yet
                continue
            last_id = id(frame)

            try:
                t0 = time.perf_counter()
                results = self.recognizer.recognize(frame)
                self._infer_ms = (time.perf_counter() - t0) * 1000
                self.last_results = results
                self._register(results)
            except Exception as exc:          # never kill the thread
                print("[camera] recognition error:", exc)
                time.sleep(0.2)

            time.sleep(config.RECOGNITION_MIN_INTERVAL)

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

    # ---- frame output ---------------------------------------------------- #
    def _annotate(self, frame):
        for r in self.last_results:
            top, right, bottom, left = r["box"]
            matched = r["student_id"] is not None
            color = (0, 180, 0) if matched else (0, 0, 220)
            label = r["name"]
            sim = r.get("similarity")
            if sim is not None:
                label += "  {}%".format(int(sim * 100))
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 24), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 5, bottom - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def stream(self):
        """Yield an MJPEG multipart stream. Counts as a viewer for its lifetime."""
        self.start()
        self._viewers += 1
        try:
            idle = 0
            last_id = None
            while self.running:
                frame = self.frame
                if frame is None:
                    time.sleep(0.05)
                    idle += 1
                    if idle > 100:                # ~5s with no frame
                        break
                    continue
                if id(frame) == last_id:          # don't re-encode the same frame
                    time.sleep(0.005)
                    continue
                last_id, idle = id(frame), 0

                draw = self.recognition_on and self.last_results
                out = self._annotate(frame.copy()) if draw else frame
                ok, buf = cv2.imencode(
                    ".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, config.STREAM_JPEG_QUALITY]
                )
                if ok:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + buf.tobytes() + b"\r\n")
        finally:
            self._viewers = max(0, self._viewers - 1)

    def snapshot(self):
        """Return the latest frame, briefly opening the camera if needed."""
        self.start()
        self._viewers += 1
        try:
            for _ in range(60):                   # wait up to ~3s for a frame
                if self.frame is not None:
                    return self.frame.copy()
                time.sleep(0.05)
            return None
        finally:
            self._viewers = max(0, self._viewers - 1)
