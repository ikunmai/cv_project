"""Synthetic video-counting benchmark for the counting rules.

There is no GPU (and no VideoCount data) on the machine this was developed on, so
the end-to-end effect of the new rules cannot be measured there. This script
instead builds a *calibrated* simulation of the decision problem that stage 3
faces, and runs the *real* rule implementations from counting_rules.py on it:

* crystals appear over time and grow, in a dense field where SAM masks of
  neighbours overlap;
* CountGD misses objects, emits occasional near-duplicate boxes and a few
  background false positives;
* SAM 2 image masks are dilated versions of the object; video propagation keeps
  the identity most of the time but sometimes drifts, switches to a neighbour or
  loses the object, which is what makes duplicate tracks;
* the temporal filter drops detections that do not persist for w frames.

The baseline profile is calibrated so that its relative error matches the one
video the team has measured (ma2035_023: 100 predicted vs 111 ground truth,
9.91 %). The comparison between profiles below is therefore a mechanism study on
a fitted synthetic distribution, NOT a benchmark result -- real numbers must come
from replay_crystals.py on cached runs.

    python simulate_video_counting.py --seeds 20 --out results/sim_counting.csv
"""
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import counting_rules as cr


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------
@dataclass
class SimParams:
    n_objects: int = 111          # ground-truth count of the video
    n_frames: int = 37
    height: int = 300
    width: int = 500
    p_present_at_start: float = 1.0    # fraction of objects visible in frame 0
    coverage_target: float = 0.75      # fraction of the frame covered by crystal masks
    detect_floor: float = 0.365   # CountGD recall for the smallest crystal
    detect_slope: float = 0.62    # additional recall as the crystal grows
    detect_scale: float = 6.0     # radius (px) at which slope is mostly reached
    p_duplicate: float = 0.03     # extra near-duplicate box for a detected object
    p_fp_per_frame: float = 0.3   # background false positives per frame
    dilate: int = 2               # SAM mask is looser than the object
    p_hidden: float = 0.12        # fraction of objects with a short occlusion
    p_drift: float = 0.10         # propagation drifts a little
    p_switch: float = 0.01        # propagation slides onto a neighbour
    p_lost: float = 0.02          # propagation loses the object
    temporal_w: int = 3           # temporal filter window
    box_jitter: float = 2.0


class SyntheticVideo:
    def __init__(self, params: SimParams, seed: int):
        self.p = params
        self.rng = np.random.default_rng(seed)
        p, rng = params, self.rng
        H, W = p.height, p.width
        self.H, self.W = H, W

        n_start = int(round(p.p_present_at_start * p.n_objects))
        births = np.concatenate([
            np.zeros(n_start, dtype=int),
            rng.integers(1, p.n_frames, size=p.n_objects - n_start),
        ])
        # Dense but resolvable placement: grid cells with jitter.
        side = int(np.ceil(np.sqrt(p.n_objects * 1.3)))
        cells = [(gy, gx) for gy in range(side) for gx in range(side)]
        rng.shuffle(cells)
        self.birth = births
        # Radius is set so that the masks cover ``coverage_target`` of the frame:
        # this is what makes neighbour masks overlap and the strict zero-overlap
        # rule reject genuinely new objects.
        r_scale = np.sqrt(p.coverage_target * H * W / (p.n_objects * np.pi))
        self.cx, self.cy, self.r0 = [], [], []
        for k in range(p.n_objects):
            gy, gx = cells[k]
            self.cy.append(int((gy + 0.5) * H / side + rng.normal(0, H / side / 6)))
            self.cx.append(int((gx + 0.5) * W / side + rng.normal(0, W / side / 6)))
            self.r0.append(float(r_scale * rng.uniform(0.62, 1.35)))
        self.cx, self.cy, self.r0 = np.array(self.cx), np.array(self.cy), np.array(self.r0)

        self.alive = np.zeros((p.n_objects, p.n_frames), dtype=bool)
        for k in range(p.n_objects):
            self.alive[k, births[k]:] = True
        self.hidden = np.zeros_like(self.alive)
        for k in range(p.n_objects):
            if rng.random() < p.p_hidden:
                f0 = int(rng.integers(1, max(2, p.n_frames - 3)))
                self.hidden[k, f0:f0 + int(rng.integers(1, 4))] = True
        self.visible = self.alive & ~self.hidden

        # CountGD detection indicator: recall grows with the crystal radius as
        # the objects grow over the video, then the temporal filter keeps only
        # detections belonging to a run of at least w frames.
        prob = np.zeros((p.n_objects, p.n_frames))
        for k in range(p.n_objects):
            for f in range(p.n_frames):
                r = self.radius(k, f)
                prob[k, f] = p.detect_floor + p.detect_slope * (1 - np.exp(-r / p.detect_scale))
        self.detect_prob = np.clip(prob, 0, 0.97)
        self.detected = (rng.random((p.n_objects, p.n_frames)) < self.detect_prob) & self.visible
        self.detected = self._temporal_filter(self.detected)

        # Propagation states per (object, frame): 0 ok, 1 drift, 2 lose, 3 switch.
        u = rng.random((p.n_objects, p.n_frames))
        self.state = np.zeros((p.n_objects, p.n_frames), dtype=np.int8)
        self.state[u > 1 - p.p_lost] = 2
        self.state[(u <= 1 - p.p_lost) & (u > 1 - p.p_lost - p.p_switch)] = 3
        self.state[(u <= 1 - p.p_lost - p.p_switch) & (u > 1 - p.p_lost - p.p_switch - p.p_drift)] = 1

        # Propagation drift is pre-drawn per (object, frame) so that every rule
        # configuration sees exactly the same noisy propagation, whatever order
        # it visits the frames in.
        self.drift_x = rng.normal(0, 0.35, (p.n_objects, p.n_frames))
        self.drift_y = rng.normal(0, 0.35, (p.n_objects, p.n_frames))
        self._mask_cache = {}
        self._shift_cache = {}
        self._frame_cache = {}

    # -- geometry ----------------------------------------------------------
    def radius(self, k, f):
        span = max(1, self.p.n_frames - self.birth[k])
        grow = 1.0 + 0.6 * max(0, f - self.birth[k]) / span
        return self.r0[k] * grow

    def mask(self, k, f):
        key = (k, f)
        if key not in self._mask_cache:
            r = self.radius(k, f) + self.p.dilate
            y0, y1 = max(0, int(self.cy[k] - r)), min(self.H, int(self.cy[k] + r) + 1)
            x0, x1 = max(0, int(self.cx[k] - r)), min(self.W, int(self.cx[k] + r) + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            m = (yy - self.cy[k]) ** 2 + (xx - self.cx[k]) ** 2 <= r * r
            self._mask_cache[key] = (yy[m].astype(np.int64), xx[m].astype(np.int64))
        return self._mask_cache[key]

    def shifted(self, k, f):
        key = (k, f)
        if key not in self._shift_cache:
            rows, cols = self.mask(k, f)
            r = self.radius(k, f)
            dx, dy = self.drift_x[k, f] * r, self.drift_y[k, f] * r
            self._shift_cache[key] = (np.clip(rows + int(round(dy)), 0, self.H - 1).astype(np.int64),
                                      np.clip(cols + int(round(dx)), 0, self.W - 1).astype(np.int64))
        return self._shift_cache[key]

    def box(self, k, f):
        rows, cols = self.mask(k, f)
        j = self.p.box_jitter
        return np.array([cols.min() - j, rows.min() - j, cols.max() + j, rows.max() + j], dtype=np.float32)

    # -- detections --------------------------------------------------------
    def _temporal_filter(self, detected):
        w = self.p.temporal_w
        out = np.zeros_like(detected)
        for k in range(detected.shape[0]):
            f = 0
            line = detected[k]
            while f < len(line):
                if not line[f]:
                    f += 1
                    continue
                s = f
                while f + 1 < len(line) and line[f + 1]:
                    f += 1
                if f - s + 1 >= w:
                    out[k, s:f + 1] = True
                f += 1
        return out

    def frame(self, f):
        """Independent per-frame detections, in the pipeline's dict format."""
        if f in self._frame_cache:
            return self._frame_cache[f]
        masks, boxes, owners = {}, [], []
        for k in range(self.p.n_objects):
            if not self.detected[k, f]:
                continue
            masks[len(boxes) + 1] = self.mask(k, f)
            boxes.append(self.box(k, f))
            owners.append(k)
            if self.rng.random() < self.p.p_duplicate:
                masks[len(boxes) + 1] = self.shifted(k, f)
                b = self.box(k, f).copy()
                b += self.rng.normal(0, 4, 4).astype(np.float32)
                boxes.append(b)
                owners.append(k)
        n_fp = self.rng.poisson(self.p.p_fp_per_frame)
        for _ in range(n_fp):
            cxf, cyf = self.rng.uniform(0, self.W), self.rng.uniform(0, self.H)
            r = self.rng.uniform(4, 10)
            yy, xx = np.mgrid[int(cyf - r):int(cyf + r), int(cxf - r):int(cxf + r)]
            m = (yy - cyf) ** 2 + (xx - cxf) ** 2 <= r * r
            if m.sum() == 0:
                continue
            masks[len(boxes) + 1] = (yy[m].astype(np.int64), xx[m].astype(np.int64))
            boxes.append(np.array([cxf - r, cyf - r, cxf + r, cyf + r], dtype=np.float32))
            owners.append(-1)
        entry = {"mask_ids": masks, "box_prompts": np.array(boxes, dtype=np.float32).reshape(-1, 4),
                 "owners": owners}
        self._frame_cache[f] = entry
        return entry

    # -- SAM 2 video propagation ------------------------------------------
    def propagate(self, frame_idx, prompts, reverse=False):
        """prompts: {obj_id: box}. Yields (frame, {obj_id: mask})."""
        owner = {}
        for oid, box in prompts.items():
            owner[oid] = self._owner_of_box(box, frame_idx)
        frames = range(frame_idx, -1, -1) if reverse else range(frame_idx, self.p.n_frames)
        for f in frames:
            out = {}
            for oid, k in owner.items():
                if k is None or k < 0 or not self.visible[k, f]:
                    continue
                s = self.state[k, f]
                if s == 2:            # lost
                    continue
                if s == 3:            # switched to a neighbour
                    nbr = self._nearest(k, f)
                    if nbr is not None:
                        out[oid] = self.mask(nbr, f)
                    continue
                out[oid] = self.shifted(k, f) if s == 1 else self.mask(k, f)
            yield f, out

    def _owner_of_box(self, box, f):
        entry = self.frame(f)
        box = np.asarray(box, dtype=np.float32)
        d = np.abs(entry["box_prompts"] - box).sum(axis=1)
        idx = int(np.argmin(d)) if len(d) else -1
        return entry["owners"][idx] if idx >= 0 else None

    def _nearest(self, k, f):
        alive = np.where(self.visible[:, f])[0]
        if len(alive) <= 1:
            return None
        d = np.hypot(self.cx[alive] - self.cx[k], self.cy[alive] - self.cy[k])
        d[alive == k] = np.inf
        j = int(np.argmin(d))
        return int(alive[j]) if d[j] < 3 * np.median(self.r0) else None


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
CONFIGS = {
    "baseline": cr.PROFILES["baseline"],
    "seed_only": cr.PROFILES["seed_only"],
    "coverage": cr.PROFILES["coverage"],
    "balanced": cr.PROFILES["balanced"],
    "full": cr.PROFILES["full"],
}


def run_episode(params: SimParams, seed: int, configs):
    sim = SyntheticVideo(params, seed)
    frames = {f: sim.frame(f) for f in range(params.n_frames)}
    out = {}
    for name, cfg in configs.items():
        T, info = cr.track_and_count(frames, params.n_frames, sim.propagate, cfg)
        out[name] = {"count": len(T), **info}
    # Reference points: every object is detected at some point in most episodes,
    # so a perfect tracker could reach the full ground-truth count.
    out["_meta"] = {"oracle_detected": int(sim.detected.any(axis=1).sum()),
                    "seed_dets": len(frames[0]["box_prompts"]),
                    "best_dets": int(max(len(frames[f]["box_prompts"])
                                         for f in range(params.n_frames)))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=15, help="episodes at the calibrated condition")
    ap.add_argument("--sweep_seeds", type=int, default=4, help="episodes per sensitivity condition")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="results/sim_counting.csv")
    ap.add_argument("--gt", type=int, default=111)
    ap.add_argument("--figure", default="img/sim_counting.png")
    args = ap.parse_args()

    print("Synthetic mechanism study for the video counting rules")
    print("(calibrated to the one measured video ma2035_023: 111 -> 100).")
    print("Counts come from the real counting_rules.track_and_count, not from a reimplementation.\n")

    MAIN = (1.0, 0.01)                       # calibrated condition
    SWEEP = [(0.85, 0.005), (0.85, 0.02), (1.0, 0.005), (1.0, 0.02)]
    rows = []

    def episode(p_start, p_switch, seed):
        params = SimParams(p_present_at_start=p_start, p_switch=p_switch)
        res = run_episode(params, seed, CONFIGS)
        meta = res.pop("_meta")
        for name, r in res.items():
            rows.append({"p_start": p_start, "p_switch": p_switch, "seed": seed,
                         "config": name, "gt": args.gt, "pred": r["count"],
                         "abs_err": abs(r["count"] - args.gt),
                         "rel_err": abs(r["count"] - args.gt) / args.gt,
                         "merged_away": r.get("merged_away", 0),
                         "added": r["added"], "seed_frame": r["seed_frame"],
                         "oracle_detected": meta["oracle_detected"],
                         "seed_dets": meta["seed_dets"], "best_dets": meta["best_dets"]})

    for seed in range(args.start, args.start + args.seeds):
        episode(*MAIN, seed)
    for p_start, p_switch in SWEEP:
        for seed in range(args.start, args.start + args.sweep_seeds):
            episode(p_start, p_switch, seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")

    def subset(p_start, p_switch):
        return [r for r in rows if r["p_start"] == p_start and r["p_switch"] == p_switch]

    print(f"\nCalibrated condition (p_start={MAIN[0]}, p_switch={MAIN[1]}, "
          f"{args.seeds} episodes, gt={args.gt}):")
    print(f"{'config':>10} {'mean count':>10} {'bias':>6} {'MAE':>6} {'RMSE':>6} {'NAE':>7} {'merged':>7}")
    for name in CONFIGS:
        sel = [r for r in subset(*MAIN) if r["config"] == name]
        pred = np.array([r["pred"] for r in sel], float)
        err = np.abs(pred - args.gt)
        print(f"{name:>10} {pred.mean():10.1f} {pred.mean() - args.gt:+6.1f} {err.mean():6.2f} "
              f"{np.sqrt((err ** 2).mean()):6.2f} {np.mean(err / args.gt):7.4f} "
              f"{np.mean([r['merged_away'] for r in sel]):7.1f}")
    sel0 = [r for r in subset(*MAIN) if r["config"] == "baseline"]
    print(f"reference: seed-frame detections {np.mean([r['seed_dets'] for r in sel0]):.1f}, "
          f"densest frame {np.mean([r['best_dets'] for r in sel0]):.1f}, "
          f"objects detected at least once {np.mean([r['oracle_detected'] for r in sel0]):.1f}")

    print("\nSensitivity (mean NAE over the sweep conditions):")
    print(f"{'config':>10} " + " ".join(f"{f'{a}/{b}':>12}" for a, b in SWEEP))
    for name in CONFIGS:
        cells = []
        for a, b in SWEEP:
            sel = [r for r in subset(a, b) if r["config"] == name]
            cells.append(np.mean([r["rel_err"] for r in sel]))
        print(f"{name:>10} " + " ".join(f"{c:12.4f}" for c in cells))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
        labels = list(CONFIGS)
        data = [[r["rel_err"] * 100 for r in subset(*MAIN) if r["config"] == n] for n in labels]
        axes[0].boxplot(data, tick_labels=labels, showfliers=False)
        axes[0].axhline(9.91, color="0.4", ls="--", lw=1)
        axes[0].text(0.55, 10.3, "measured baseline 9.91%", fontsize=7, color="0.35")
        axes[0].set_ylabel("relative count error (%)")
        axes[0].set_title(f"calibrated condition ({args.seeds} episodes)", fontsize=9)
        axes[0].grid(alpha=0.3, axis="y")
        width = 0.8 / len(labels)
        for k, name in enumerate(labels):
            xs, ys = [], []
            for j, (a, b) in enumerate(SWEEP):
                sel = [r for r in subset(a, b) if r["config"] == name]
                xs.append(j + k * width)
                ys.append(np.mean([r["rel_err"] for r in sel]) * 100)
            axes[1].bar(xs, ys, width=width, label=name)
        axes[1].set_xticks([j + 0.4 - width / 2 for j in range(len(SWEEP))])
        axes[1].set_xticklabels([f"start {a}\nswitch {b}" for a, b in SWEEP], fontsize=7)
        axes[1].set_ylabel("mean relative error (%)")
        axes[1].set_title("sensitivity to scene parameters", fontsize=9)
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3, axis="y")
        fig.suptitle("Synthetic counting benchmark, real rule implementations", fontsize=10)
        fig.tight_layout()
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=160)
        print(f"wrote {args.figure}")
    except Exception as exc:
        print("figure skipped:", exc)


if __name__ == "__main__":
    main()
