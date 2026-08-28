"""Measure face-recognition accuracy on your enrolled dataset.

Method: leave-one-out cross-validation. For every enrolled face image we hide
that one image, then recognise it against all the *other* enrolled faces using
the exact same voting logic the live system uses, and check whether the
predicted student is correct.

    python evaluate.py

Optional impostor test: put photos of people who are NOT enrolled into a folder
named  eval_impostors/  (any images). The script then also reports how often an
impostor is wrongly accepted as a real student (false accept rate).

Notes / honesty:
- This tests enrollment-quality images, so it is a best case. Live webcam
  numbers are usually a few points lower. Capture varied enrollment photos
  (angles, lighting) to keep the two close.
- Accuracy is only meaningful with >= 2 students and >= ~10 images each.
"""
import sys
from collections import defaultdict

import numpy as np
import face_recognition

import config


def _load_faces(root, labelled=True):
    """Return list of (student_id, name, encoding). id/name are None if not labelled."""
    items = []
    if not root.exists():
        return items
    dirs = sorted(p for p in root.iterdir() if p.is_dir()) if labelled else [root]
    for d in dirs:
        if labelled:
            if "_" not in d.name:
                continue
            try:
                sid = int(d.name.split("_", 1)[0])
            except ValueError:
                continue
            name = d.name.split("_", 1)[1].replace("_", " ")
            images = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
        else:
            sid, name = None, None
            images = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")) + sorted(d.glob("*.jpeg"))
        for img in images:
            image = face_recognition.load_image_file(str(img))
            boxes = face_recognition.face_locations(image, model=config.FACE_DETECTION_MODEL)
            for enc in face_recognition.face_encodings(image, boxes, num_jitters=1):
                items.append((sid, name, enc))
    return items


def _vote(distances, ids, tol):
    within = np.where(distances <= tol)[0]
    if len(within) == 0:
        return None
    cand = ids[within]
    best_id, best_score = None, None
    for sid in set(cand.tolist()):
        idx = within[cand == sid]
        score = (len(idx), -float(distances[idx].min()))
        if best_score is None or score > best_score:
            best_score, best_id = score, sid
    return best_id


def main():
    gallery = _load_faces(config.DATASET_DIR, labelled=True)
    students = sorted({sid for sid, _, _ in gallery})
    names = {sid: name for sid, name, _ in gallery}

    if len(students) < 2:
        print("Need at least 2 enrolled students to measure accuracy "
              f"(found {len(students)}). Enroll more people and re-run.")
        sys.exit(1)
    if len(gallery) < 2 * len(students):
        print(f"Warning: only {len(gallery)} face images for {len(students)} students "
              "- the number will be noisy. Aim for ~20 images each.\n")

    encs = np.array([e for _, _, e in gallery])
    ids = np.array([s for s, _, _ in gallery])
    tol = config.FACE_MATCH_TOLERANCE
    n = len(gallery)

    correct = wrong = unknown = 0
    per = defaultdict(lambda: [0, 0])          # sid -> [correct, total]
    confusion = defaultdict(int)               # (true, pred) -> count
    good_dists = []

    for i in range(n):
        true_id = ids[i]
        d = np.linalg.norm(encs - encs[i], axis=1)
        d[i] = np.inf                          # leave this image out
        pred = _vote(d, ids, tol)
        per[true_id][1] += 1
        if pred is None:
            unknown += 1
            confusion[(true_id, "Unknown")] += 1
        elif pred == true_id:
            correct += 1
            per[true_id][0] += 1
            good_dists.append(float(np.min(d[ids == true_id])))
            confusion[(true_id, pred)] += 1
        else:
            wrong += 1
            confusion[(true_id, pred)] += 1

    acc = correct / n
    print("=" * 52)
    print("  FACE RECOGNITION ACCURACY  (leave-one-out)")
    print("=" * 52)
    print(f"  Students enrolled     : {len(students)}")
    print(f"  Face images tested    : {n}")
    print(f"  Match tolerance       : {tol}")
    print("-" * 52)
    print(f"  Correct               : {correct:4d}   {correct/n:6.1%}")
    print(f"  Wrong person          : {wrong:4d}   {wrong/n:6.1%}")
    print(f"  Rejected as Unknown   : {unknown:4d}   {unknown/n:6.1%}")
    print("-" * 52)
    print(f"  ACCURACY              : {acc:6.1%}")
    if good_dists:
        print(f"  Mean distance (hits)  : {np.mean(good_dists):.3f}")
    print("-" * 52)
    print("  Per-student recall:")
    for sid in students:
        c, t = per[sid]
        print(f"    {names[sid][:22]:22s}  {c:3d}/{t:<3d}  {c/t:6.1%}")

    mistakes = {k: v for k, v in confusion.items() if k[0] != k[1]}
    if mistakes:
        print("-" * 52)
        print("  Misclassifications (true -> predicted):")
        for (t, p), cnt in sorted(mistakes.items(), key=lambda x: -x[1]):
            pname = "Unknown" if p == "Unknown" else names.get(p, p)
            print(f"    {names[t][:18]:18s} -> {str(pname)[:18]:18s}  x{cnt}")

    # ---- optional impostor test ----
    impostors = _load_faces(config.BASE_DIR / "eval_impostors", labelled=False)
    if impostors:
        falsely_accepted = 0
        for _, _, e in impostors:
            d = np.linalg.norm(encs - e, axis=1)
            if _vote(d, ids, tol) is not None:
                falsely_accepted += 1
        m = len(impostors)
        print("=" * 52)
        print(f"  Impostor faces tested : {m}")
        print(f"  Falsely accepted      : {falsely_accepted}   {falsely_accepted/m:6.1%}")
        print(f"  Correctly rejected    : {m - falsely_accepted}   {1 - falsely_accepted/m:6.1%}")

    print("=" * 52)
    if acc >= 0.90:
        print(f"  >= 90% target met ({acc:.1%}).")
    else:
        print(f"  Below 90% ({acc:.1%}). Try: more/varied photos per student,")
        print("  better lighting, or adjust FACE_MATCH_TOLERANCE in .env.")
    print("=" * 52)


if __name__ == "__main__":
    main()
