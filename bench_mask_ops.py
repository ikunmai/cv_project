"""Micro-benchmark: original set-based mask ops vs counting_rules fast versions.

Only measures the algorithmic cost of the mask comparisons that dominate the
temporal filter (``check_obj_persistence``) and ``get_new_objs``. Model forward
passes are not involved, so this runs on CPU.

    python bench_mask_ops.py --height 1080 --width 1920 --objects 100
"""
import argparse
import time

import numpy as np

import counting_rules as cr


def make_masks(n_objects, height, width, radius=14, seed=0, overlap=0.45):
    """Scatter overlapping blob masks, ~dense crystal field."""
    rng = np.random.default_rng(seed)
    side = int(np.sqrt(n_objects / max(overlap, 1e-6))) + 1
    centers = []
    for gy in range(side):
        for gx in range(side):
            if len(centers) == n_objects:
                break
            centers.append((int((gy + 0.5) * height / side + rng.normal(0, 6)),
                            int((gx + 0.5) * width / side + rng.normal(0, 6))))
    masks = []
    for cy, cx in centers:
        r = radius + int(rng.integers(-4, 5))
        y0, y1 = max(0, cy - r), min(height, cy + r)
        x0, x1 = max(0, cx - r), min(width, cx + r)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        keep = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        masks.append((yy[keep].astype(np.int64), xx[keep].astype(np.int64)))
    return masks


def original_pairwise_iou(short_term, sam_pred, w=3):
    """The nested set-intersection loop of check_obj_persistence."""
    n_frames = len(sam_pred)
    hits = 0
    for i in range(n_frames):
        for obj_j in sam_pred[i]["mask_ids"]:
            m_track = short_term[i][obj_j]
            for m in range(max(0, i - (w - 1)), i):
                a = set(zip(m_track[m][0], m_track[m][1]))
                for obj_k in sam_pred[m]["mask_ids"]:
                    mk = sam_pred[m]["mask_ids"][obj_k]
                    b = set(zip(mk[0], mk[1]))
                    inter = len(a & b)
                    if inter and inter / len(a | b) > 0.5:
                        hits += 1
                        break
    return hits


def fast_pairwise_iou(short_term, sam_pred, w=3):
    n_frames = len(sam_pred)
    hits = 0
    for i in range(n_frames):
        for obj_j in sam_pred[i]["mask_ids"]:
            m_track = short_term[i][obj_j]
            for m in range(max(0, i - (w - 1)), i):
                for obj_k in sam_pred[m]["mask_ids"]:
                    if cr.mask_iou(m_track[m], sam_pred[m]["mask_ids"][obj_k]) > 0.5:
                        hits += 1
                        break
    return hits


def original_get_new_objs(masks, tracked):
    union = set()
    for t in tracked:
        union |= set(zip(t[0], t[1]))
    out = 0
    for m in masks:
        s = set(zip(m[0], m[1]))
        out += len(s & union) == 0
    return out


def fast_get_new_objs(masks, tracked):
    idx = cr.FrameIndex(tracked)
    return sum(idx.packed_covered(m) == 0 for m in masks)


def timeit(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--objects", type=int, default=100)
    p.add_argument("--frames", type=int, default=5)
    args = p.parse_args()

    masks = make_masks(args.objects, args.height, args.width)
    area = np.mean([len(m[0]) for m in masks])
    sam_pred = {f: {"mask_ids": {k + 1: masks[k] for k in range(len(masks))},
                    "box_prompts": np.zeros((len(masks), 4))}
                for f in range(args.frames)}
    short_term = {
        f: {k + 1: {g: masks[(k + g) % len(masks)] for g in range(args.frames)}
            for k in range(len(masks))}
        for f in range(args.frames)
    }
    print(f"frame {args.height}x{args.width}, {args.objects} objects, "
          f"mean mask area {area:.0f} px, {args.frames} frames")

    t_set = timeit(lambda: original_pairwise_iou(short_term, sam_pred))
    t_fast = timeit(lambda: fast_pairwise_iou(short_term, sam_pred))
    print(f"pairwise IoU (temporal filter, w=3): original {t_set:.2f}s  "
          f"fast {t_fast:.2f}s  speedup {t_set / max(t_fast, 1e-9):.1f}x")

    tracked = masks[:len(masks) // 2]
    cand = masks[len(masks) // 2:]
    t_set = timeit(lambda: original_get_new_objs(cand, tracked))
    t_fast = timeit(lambda: fast_get_new_objs(cand, tracked))
    print(f"get_new_objs: original {t_set:.3f}s  fast {t_fast:.3f}s  "
          f"speedup {t_set / max(t_fast, 1e-9):.1f}x")

    t = timeit(lambda: cr.merge_tracks({k + 1: {f: masks[k] for f in range(args.frames)}
                                       for k in range(len(masks))},
                                      cr.CountConfig(merge=True)))
    print(f"merge_tracks on {args.objects} tracks: {t:.2f}s")


if __name__ == "__main__":
    main()
