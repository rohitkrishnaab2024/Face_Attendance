"""SQLite storage for students and attendance logs.

Two tables:
  students   - one row per enrolled person
  attendance - one row per student per day (UNIQUE(student_id, date))

The UNIQUE constraint is the core "timestamp-based validation": a student can
only be marked present once for a given calendar day, no matter how many times
their face is seen.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_no    TEXT UNIQUE NOT NULL,
    name       TEXT NOT NULL,
    email      TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    date       TEXT NOT NULL,               -- YYYY-MM-DD
    time_in    TEXT NOT NULL,               -- HH:MM:SS
    status     TEXT NOT NULL DEFAULT 'Present',
    UNIQUE(student_id, date),
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------- #
# Students
# --------------------------------------------------------------------------- #
def add_student(roll_no, name, email=None):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO students (roll_no, name, email, created_at) VALUES (?, ?, ?, ?)",
            (roll_no, name, email, datetime.now().isoformat(timespec="seconds")),
        )
        return cur.lastrowid


def list_students():
    with get_conn() as conn:
        return conn.execute("SELECT * FROM students ORDER BY name COLLATE NOCASE").fetchall()


def get_student(student_id):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()


def delete_student(student_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM students WHERE id = ?", (student_id,))


# --------------------------------------------------------------------------- #
# Attendance
# --------------------------------------------------------------------------- #
def mark_attendance(student_id, when=None, late_after=None):
    """Insert today's attendance row for a student.

    Returns True if a NEW row was created, False if the student was already
    marked for that day (the UNIQUE constraint rejects the duplicate).
    """
    when = when or datetime.now()
    date = when.strftime("%Y-%m-%d")
    time_in = when.strftime("%H:%M:%S")
    status = "Late" if (late_after and time_in > late_after) else "Present"
    with get_conn() as conn:
        try:
            conn.execute(
                "INSERT INTO attendance (student_id, date, time_in, status) VALUES (?, ?, ?, ?)",
                (student_id, date, time_in, status),
            )
            return True
        except sqlite3.IntegrityError:
            return False


def set_attendance(student_id, date, time_in=None, status="Present"):
    """Manual create-or-update used by the dashboard 'manage' controls."""
    time_in = time_in or datetime.now().strftime("%H:%M:%S")
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO attendance (student_id, date, time_in, status)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(student_id, date)
               DO UPDATE SET time_in = excluded.time_in, status = excluded.status""",
            (student_id, date, time_in, status),
        )


def delete_attendance(att_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM attendance WHERE id = ?", (att_id,))


def attendance_for_date(date):
    with get_conn() as conn:
        return conn.execute(
            """SELECT a.*, s.name, s.roll_no, s.email
               FROM attendance a
               JOIN students s ON s.id = a.student_id
               WHERE a.date = ?
               ORDER BY a.time_in""",
            (date,),
        ).fetchall()


def attendance_between(start, end):
    with get_conn() as conn:
        return conn.execute(
            """SELECT a.*, s.name, s.roll_no
               FROM attendance a
               JOIN students s ON s.id = a.student_id
               WHERE a.date BETWEEN ? AND ?
               ORDER BY a.date, a.time_in""",
            (start, end),
        ).fetchall()
