"""Flask app: teacher dashboard for the face-recognition attendance system.

Run with:  python app.py     (then open http://127.0.0.1:5000)
"""
import atexit
import signal
from datetime import date as date_cls

from flask import (Flask, Response, flash, jsonify, redirect, render_template,
                   request, url_for)

import config
import database
import exporter
import face_engine
from camera import Camera

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
database.init_db()


def _shutdown_camera(*_):
    Camera().stop()


# Release the webcam when the process exits (Ctrl+C, kill, normal exit).
atexit.register(_shutdown_camera)
for _sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_sig, lambda s, f: (_shutdown_camera(), exit(0)))
    except (ValueError, OSError):        # not in main thread / unsupported
        pass


def _today():
    return date_cls.today().isoformat()


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
@app.route("/")
def dashboard():
    students = database.list_students()
    todays = database.attendance_for_date(_today())
    present_ids = {r["student_id"] for r in todays}
    absent = [s for s in students if s["id"] not in present_ids]
    return render_template(
        "dashboard.html",
        today=_today(),
        students=students,
        todays=todays,
        absent=absent,
        trained=Camera().recognizer.is_trained,
    )


# --------------------------------------------------------------------------- #
# Students & enrollment
# --------------------------------------------------------------------------- #
@app.route("/students")
def students():
    rows = []
    for s in database.list_students():
        rows.append({**dict(s), "samples": face_engine.count_samples(s["id"], s["name"])})
    return render_template("students.html", students=rows,
                           target=config.SAMPLES_PER_STUDENT)


@app.route("/students/add", methods=["GET", "POST"])
def add_student():
    if request.method == "POST":
        roll = request.form.get("roll_no", "").strip()
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip() or None
        if not roll or not name:
            flash("Roll number and name are both required.", "error")
            return redirect(url_for("add_student"))
        try:
            sid = database.add_student(roll, name, email)
        except Exception as exc:
            flash(f"Could not add student: {exc}", "error")
            return redirect(url_for("add_student"))
        flash(f"Created '{name}'. Now capture some face photos.", "success")
        return redirect(url_for("capture", student_id=sid))
    return render_template("add_student.html")


@app.route("/students/<int:student_id>/delete", methods=["POST"])
def delete_student(student_id):
    database.delete_student(student_id)
    flash("Student removed. Re-train the model to drop their face data.", "success")
    return redirect(url_for("students"))


@app.route("/students/<int:student_id>/capture")
def capture(student_id):
    student = database.get_student(student_id)
    if not student:
        flash("Student not found.", "error")
        return redirect(url_for("students"))
    cam = Camera()
    cam.start()
    cam.recognition_on = False
    return render_template(
        "capture.html",
        student=student,
        samples=face_engine.count_samples(student_id, student["name"]),
        target=config.SAMPLES_PER_STUDENT,
    )


@app.route("/api/capture/<int:student_id>", methods=["POST"])
def api_capture(student_id):
    student = database.get_student(student_id)
    if not student:
        return jsonify(error="student not found"), 404
    target = config.SAMPLES_PER_STUDENT
    count = face_engine.count_samples(student_id, student["name"])
    if count >= target:
        return jsonify(saved=False, done=True, count=count,
                       message=f"Enough samples ({count}). Go to Students -> Train model.")

    cam = Camera()
    cam.start()
    frame = cam.snapshot()
    if frame is None:
        return jsonify(saved=False, message="Camera not ready yet, try again.",
                       count=count), 503
    path, message = face_engine.save_face_sample(student_id, student["name"], frame)
    count = face_engine.count_samples(student_id, student["name"])
    return jsonify(saved=path is not None, done=count >= target,
                   message=message, count=count)


@app.route("/api/train", methods=["POST"])
def api_train():
    n_enc, n_students = face_engine.train()
    Camera().recognizer.load()
    return jsonify(encodings=n_enc, students=n_students)


# --------------------------------------------------------------------------- #
# Live recognition
# --------------------------------------------------------------------------- #
@app.route("/recognize")
def recognize_page():
    cam = Camera()
    cam.start()
    return render_template("recognize.html",
                           recognition_on=cam.recognition_on,
                           trained=cam.recognizer.is_trained)


@app.route("/video_feed")
def video_feed():
    cam = Camera()
    cam.start()
    return Response(cam.stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/recognition/<action>", methods=["POST"])
def recognition_toggle(action):
    cam = Camera()
    cam.start()
    cam.recognition_on = (action == "start")
    if cam.recognition_on:
        cam.recent_marks = []
    return jsonify(recognition_on=cam.recognition_on)


@app.route("/api/recent_marks")
def recent_marks():
    return jsonify(marks=Camera().recent_marks)


@app.route("/api/camera/release", methods=["POST"])
def camera_release():
    """Turn recognition off and let the webcam be released now."""
    Camera().recognition_on = False
    return jsonify(ok=True)


# --------------------------------------------------------------------------- #
# Attendance records
# --------------------------------------------------------------------------- #
@app.route("/attendance")
def attendance_view():
    d = request.args.get("date") or _today()
    rows = database.attendance_for_date(d)
    present_ids = {r["student_id"] for r in rows}
    absent = [s for s in database.list_students() if s["id"] not in present_ids]
    return render_template("attendance.html", date=d, rows=rows, absent=absent,
                           all_students=database.list_students())


@app.route("/attendance/manual", methods=["POST"])
def attendance_manual():
    d = request.form.get("date") or _today()
    student_id = int(request.form["student_id"])
    status = request.form.get("status", "Present")
    database.set_attendance(student_id, d, status=status)
    flash("Attendance updated.", "success")
    return redirect(url_for("attendance_view", date=d))


@app.route("/attendance/<int:att_id>/delete", methods=["POST"])
def attendance_delete(att_id):
    d = request.form.get("date") or _today()
    database.delete_attendance(att_id)
    flash("Entry deleted.", "success")
    return redirect(url_for("attendance_view", date=d))


@app.route("/attendance/export.csv")
def export_csv():
    d = request.args.get("date") or _today()
    return Response(
        exporter.attendance_csv(d),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance_{d}.csv"},
    )


@app.route("/attendance/email", methods=["POST"])
def email_report():
    d = request.form.get("date") or _today()
    recipient = request.form.get("recipient", "").strip()
    if not recipient:
        flash("Enter a recipient email address.", "error")
        return redirect(url_for("attendance_view", date=d))
    try:
        exporter.email_report(recipient, d)
        flash(f"Report for {d} emailed to {recipient}.", "success")
    except Exception as exc:
        flash(f"Email failed: {exc}", "error")
    return redirect(url_for("attendance_view", date=d))


if __name__ == "__main__":
    # use_reloader=False so the webcam is not opened twice
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True, use_reloader=False)
