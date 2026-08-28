"""CSV generation and email delivery of attendance reports."""
import csv
import io
import smtplib
from email.message import EmailMessage

import config
import database


def attendance_csv(date):
    """Full class roster for a day: present/late rows plus 'Absent' for the rest."""
    marked = {r["student_id"]: r for r in database.attendance_for_date(date)}
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Date", "Roll No", "Name", "Status", "Time In"])
    for student in database.list_students():
        row = marked.get(student["id"])
        if row:
            writer.writerow([date, student["roll_no"], student["name"],
                             row["status"], row["time_in"]])
        else:
            writer.writerow([date, student["roll_no"], student["name"], "Absent", ""])
    return buf.getvalue()


def save_csv(date):
    path = config.EXPORT_DIR / f"attendance_{date}.csv"
    path.write_text(attendance_csv(date), encoding="utf-8")
    return path


def email_report(recipient, date):
    """Email the day's CSV as an attachment. Raises on misconfig / SMTP error."""
    if not (config.SMTP_USER and config.SMTP_PASSWORD):
        raise RuntimeError(
            "SMTP is not configured. Set SMTP_USER and SMTP_PASSWORD "
            "(see .env.example) before emailing reports."
        )
    msg = EmailMessage()
    msg["Subject"] = f"Attendance Report - {date}"
    msg["From"] = config.SMTP_FROM
    msg["To"] = recipient
    msg.set_content(f"Attendance report for {date} is attached as a CSV file.")
    msg.add_attachment(
        attendance_csv(date).encode("utf-8"),
        maintype="text", subtype="csv",
        filename=f"attendance_{date}.csv",
    )
    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=20) as server:
        server.starttls()
        server.login(config.SMTP_USER, config.SMTP_PASSWORD)
        server.send_message(msg)
