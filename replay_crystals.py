"""Re-run stage 3 (seed + branch A + merge) from stage-1/2 caches, for many rule configs.

Stage 1 (CountGD + SAM 2 image predictor) and stage 2 (temporal filter) are the
expensive parts of a crystal video and do not depend on the counting rules.
``count_in_videos_pseudo_exemplars_v2.py --save_cache`` stores their outputs; this
script then runs only the propagation stage for every configuration in a sweep,
which takes a few minutes per configuration instead of a full run.

    # one full run per video with caching
    python count_in_videos_pseudo_exemplars_v2.py ... --save_cache caches

    # afterwards: sweep rules over all cached videos
    python replay_crystals.py --cache_dir caches --data_dir VideoCount/Crystals \
        --configs baseline,coverage,balanced,recall \
        --sam_checkpoint checkpoints/sam2.1_hiera_large.pt \
        --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
        --out results/replay_sweep.csv

Still needs a GPU: SAM 2 *video* propagation is part of stage 3. What is skipped
is CountGD and the per-frame temporal filter.
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

import counting_cache as cc
import counting_rules as cr


def build_config(name: str, overrides: dict) -> cr.CountConfig:
    base = cr.PROFILES[name]
    return base.with_(**{k: v for k, v in overrides.items() if v is not None}).validate()


def prepare_frames(meta: dict, temp_root: Path) -> Path:
    """Recreate the SAM 2 frame directory expected by the cache."""
    out = Path(tempfile.mkdtemp(prefix="replay_", dir=str(temp_root)))
    source = Path(meta["video_source"])
    for original, news in zip(meta["frame_original_names"], meta["frame_names"]):
        shutil.copyfile(source / original, out / news)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--data_dir", default="VideoCount/Crystals")
    p.add_argument("--configs", default="baseline,coverage,balanced,recall",
                   help="comma separated profile names, or profile:key=value:key=value")
    p.add_argument("--sam_checkpoint", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--sam_model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--which", default="filtered", choices=["filtered", "raw"])
    p.add_argument("--out", default="results/replay_sweep.csv")
    p.add_argument("--leave_one_out", action="store_true",
                   help="also report each video scored with the config chosen on the other videos")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # Imported lazily so that --help and the CSV code work on a CPU-only host.
    from sam2.build_sam import build_sam2_video_predictor

    def parse_config(text):
        parts = text.split(":")
        overrides = {}
        for kv in parts[1:]:
            k, v = kv.split("=", 1)
            overrides[k] = json.loads(v) if v.lower() in ("true", "false") else float(v)
        return parts[0], overrides

    configs = [parse_config(t) for t in args.configs.split(",") if t.strip()]
    gt_all = json.loads((Path(args.data_dir) / "anno" / "crystals-count-gt.json").read_text())

    predictor = build_sam2_video_predictor(args.sam_model_cfg, args.sam_checkpoint, device=args.device)

    rows = []
    temp_root = Path(tempfile.mkdtemp(prefix="replay_frames_"))
    for cache_path in sorted(Path(args.cache_dir).glob("*__*")):
        meta, frames = cc.load_cache(cache_path, which=args.which)
        video, text = meta["video"], meta["text"]
        if video not in gt_all or text not in gt_all[video]:
            print(f"skip {cache_path}: no ground truth for {video}/{text}", flush=True)
            continue
        gt = int(gt_all[video][text])
        if "video_source" in meta:
            frame_dir = prepare_frames(meta, temp_root)
        else:
            frame_dir = Path(meta["video_dir"])
        state = predictor.init_state(video_path=str(frame_dir))

        for name, overrides in configs:
            cfg = build_config(name, overrides)
            T, info = cr.track_and_count(frames, meta["n_frames"], _wrap(predictor, state), cfg)
            count = len(T)
            label = name + ("".join(f":{k}={v}" for k, v in overrides.items()) if overrides else "")
            rows.append({"video": video, "text": text, "config": label, "gt": gt,
                         "pred": count, "abs_err": abs(count - gt),
                         "rel_err": abs(count - gt) / gt,
                         "seed_frame": info["seed_frame"], "branches": info["branches"],
                         "added": info["added"], "merged_away": info.get("merged_away", 0)})
            print(f"{video:12s} {label:28s} gt={gt:4d} pred={count:4d} "
                  f"err={abs(count - gt):3d} ({100 * abs(count - gt) / gt:5.1f}%)", flush=True)
        del state
        shutil.rmtree(frame_dir, ignore_errors=True)

    if not rows:
        sys.exit("no cached videos found")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")

    print("\nconfig                          MAE    RMSE   NAE")
    for label in dict.fromkeys(r["config"] for r in rows):
        sel = [r for r in rows if r["config"] == label]
        err = np.array([r["abs_err"] for r in sel], dtype=float)
        rel = np.array([r["rel_err"] for r in sel], dtype=float)
        print(f"{label:30s} {err.mean():6.2f} {np.sqrt((err ** 2).mean()):6.2f} "
              f"{rel.mean():6.4f}")

    if args.leave_one_out:
        videos = sorted({r["video"] for r in rows})
        labels = list(dict.fromkeys(r["config"] for r in rows))
        held = []
        for v in videos:
            best, best_mae = None, float("inf")
            for label in labels:
                others = [r for r in rows if r["video"] != v and r["config"] == label]
                mae = np.mean([r["abs_err"] for r in others]) if others else float("inf")
                if mae < best_mae:
                    best, best_mae = label, mae
            mine = [r for r in rows if r["video"] == v and r["config"] == best][0]
            held.append(mine)
            print(f"[loo] {v}: chose {best} on the other {len(videos) - 1} videos -> "
                  f"err={mine['abs_err']} ({100 * mine['rel_err']:.1f}%)")
        err = np.array([r["abs_err"] for r in held], dtype=float)
        print(f"[loo] config chosen per video: MAE {err.mean():.2f}  "
              f"NAE {np.mean([r['rel_err'] for r in held]):.4f}")


def _wrap(predictor, state):
    def propagate(frame_idx, prompts, reverse=False):
        predictor.reset_state(state)
        for obj_id, box in prompts.items():
            predictor.add_new_points_or_box(inference_state=state, frame_idx=frame_idx,
                                            obj_id=obj_id, box=box)
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                state, start_frame_idx=frame_idx, reverse=reverse):
            masks = {}
            for ind, obj_id in enumerate(out_obj_ids):
                masks[int(obj_id)] = np.nonzero((out_mask_logits[ind] > 0.0).cpu().numpy().squeeze())
            yield out_frame_idx, masks
    return propagate


if __name__ == "__main__":
    main()
