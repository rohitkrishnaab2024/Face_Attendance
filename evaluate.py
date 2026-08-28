"""Measure face-recognition accuracy on your enrolled dataset (InsightFace).

Method: leave-one-out cross-validation. For every enrolled face image we hide
that one image, recognise it against all the *other* enrolled faces using the
same cosine-similarity voting the live system uses, and check whether the
predicted student is correct.

    python evaluate.py

Optional impostor test: put photos of people who are NOT enrolled into a folder
named  eval_impostors/  . The script then also reports how often an impostor is
wrongly accepted (false accept rate).

Honesty notes:
- This tests enrollment-quality images, so it is a best case. Live webcam is
  usually a few points lower.
- Only meaningful with >= 2 students and ~15+ images each.
"""
import sys
from collections import defaultdict

import cv2
import numpy as np

import config
import face_engine


def _load(root, labelled=True):
    """Return list of (student_id, name, embedding)."""
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
        else:
            sid, name = None, None
        for img_path in sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.jpeg"))):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            f = face_engine._largest(face_engine._app().get(img))
            if f is None:
                continue
            items.append((sid, name, np.asarray(f.normed_embedding, dtype=np.float32)))
    return items


def _vote(sims, ids, thr):
    """Mirror of FaceRecognizer._vote: rank students by the mean of their top-K
    similarities, accept the best one if it clears the threshold."""
    best_id, best_score = None, -1.0
    for sid in set(ids.tolist()):
        s = sims[ids == sid]
        k = min(config.VOTE_TOP_K, len(s))
        score = float(np.sort(s)[-k:].mean())
        if score > best_score:
            best_score, best_id = score, sid
    return None if best_score < thr else best_id


def main():
    print("Loading InsightFace model and embedding the dataset...\n")
    gallery = _load(config.DATASET_DIR, labelled=True)
    students = sorted({s for s, _, _ in gallery})
    names = {s: n for s, n, _ in gallery}

    if len(students) < 2:
        print(f"Need at least 2 enrolled students (found {len(students)}).")
        sys.exit(1)
    if len(gallery) < 2 * len(students):
        print(f"Warning: only {len(gallery)} images for {len(students)} students - "
              "the number will be noisy.\n")

    embs = np.vstack([e for _, _, e in gallery]).astype(np.float32)
    ids = np.array([s for s, _, _ in gallery])
    thr = config.FACE_SIM_THRESHOLD
    n = len(gallery)

    correct = wrong = unknown = 0
    per = defaultdict(lambda: [0, 0])
    confusion = defaultdict(int)
    good_sims = []

    for i in range(n):
        true_id = ids[i]
        # leave image i out of the gallery entirely
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        sims, gal_ids = embs[mask] @ embs[i], ids[mask]

        pred = _vote(sims, gal_ids, thr)
        per[true_id][1] += 1
        if pred is None:
            unknown += 1
            confusion[(true_id, "Unknown")] += 1
        elif pred == true_id:
            correct += 1
            per[true_id][0] += 1
            good_sims.append(float(sims[gal_ids == true_id].max()))
            confusion[(true_id, pred)] += 1
        else:
            wrong += 1
            confusion[(true_id, pred)] += 1

    acc = correct / n
    print("=" * 52)
    print("  FACE RECOGNITION ACCURACY  (leave-one-out, InsightFace)")
    print("=" * 52)
    print(f"  Students enrolled     : {len(students)}")
    print(f"  Face images tested    : {n}")
    print(f"  Similarity threshold  : {thr}")
    print("-" * 52)
    print(f"  Correct               : {correct:4d}   {correct/n:6.1%}")
    print(f"  Wrong person          : {wrong:4d}   {wrong/n:6.1%}")
    print(f"  Rejected as Unknown   : {unknown:4d}   {unknown/n:6.1%}")
    print("-" * 52)
    print(f"  ACCURACY              : {acc:6.1%}")
    if good_sims:
        print(f"  Mean similarity (hits): {np.mean(good_sims):.3f}")
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

    impostors = _load(config.BASE_DIR / "eval_impostors", labelled=False)
    if impostors:
        fa = 0
        for _, _, e in impostors:
            if _vote(embs @ e, ids, thr) is not None:
                fa += 1
        m = len(impostors)
        print("=" * 52)
        print(f"  Impostor faces tested : {m}")
        print(f"  Falsely accepted      : {fa}   {fa/m:6.1%}")
        print(f"  Correctly rejected    : {m - fa}   {1 - fa/m:6.1%}")

    print("=" * 52)
    if acc >= 0.90:
        print(f"  >= 90% target met ({acc:.1%}).")
    else:
        print(f"  Below 90% ({acc:.1%}). Try more/varied photos per student,")
        print("  better lighting, or tune FACE_SIM_THRESHOLD in .env.")
    print("=" * 52)


if __name__ == "__main__":
    main()
