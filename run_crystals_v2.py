"""Run crystal videos with the v2 counting rules, keeping the v1 runner's bookkeeping.

Same fixed settings as run_crystals_single_quality.py (SAM 2.1 Large, same
CountGD-Box weights, img batch 1, object batch 30, downsample 1, temporal filter
on with w=3, RGB conversion), plus the v2 rule profile and caching.

    # baseline and one improved profile on one video
    python run_crystals_v2.py --profile baseline --video ma2035_023
    python run_crystals_v2.py --profile balanced --video ma2035_023 --save_cache caches

    # every video in the ground-truth file, caching stage 1/2 for later sweeps
    python run_crystals_v2.py --all_videos --profile baseline --save_cache caches
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="ma2035_023")
    parser.add_argument("--all_videos", action="store_true")
    parser.add_argument("--profile", default="baseline")
    parser.add_argument("--rule", default=None, choices=["strict", "coverage"])
    parser.add_argument("--min_uncovered", type=float, default=None)
    parser.add_argument("--merge_tracks", action="store_true", default=None)
    parser.add_argument("--no_merge_tracks", dest="merge_tracks", action="store_false")
    parser.add_argument("--merge_iou", type=float, default=None)
    parser.add_argument("--handover", action="store_true", default=None)
    parser.add_argument("--handover_iou", type=float, default=None)
    parser.add_argument("--seed_policy", default=None, choices=["first", "most_detections"])
    parser.add_argument("--bidirectional", action="store_true", default=None)
    parser.add_argument("--no_bidirectional", dest="bidirectional", action="store_false")
    parser.add_argument("--exemplar_strategy", default="topk", choices=["topk", "diverse"])
    parser.add_argument("--exemplar_refresh_every", type=int, default=1)
    parser.add_argument("--save_cache", default="")
    parser.add_argument("--replay_cache", default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    import os
    os.chdir(root)
    data = Path("VideoCount/Crystals")
    gt = json.loads((data / "anno" / "crystals-count-gt.json").read_text())
    videos = sorted(gt) if args.all_videos else [args.video]
    for video in videos:
        if video not in gt:
            parser.error(f"Unknown video: {video}")
    for f in [Path("checkpoints/countgd_box.pth"), Path("checkpoints/sam2.1_hiera_large.pt")]:
        if not f.exists():
            raise FileNotFoundError(f)

    out = Path("results") / (f"crystals_v2_{args.profile}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    predicted = out / "predicted.json"
    commands = []
    for video in videos:
        for text in gt[video]:
            cmd = [sys.executable, "-u", "count_in_videos_pseudo_exemplars_v2.py",
                   "--video_dir", str(data / "frames" / video), "--input_text", text,
                   "--sam_checkpoint", "checkpoints/sam2.1_hiera_large.pt",
                   "--sam_model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml",
                   "--obj_batch_size", "30", "--img_batch_size", "1", "--downsample_factor", "1",
                   "--pretrain_model_path", "checkpoints/countgd_box.pth",
                   "--output_file", str(predicted), "--temp_dir", str(out / "temp_frames"),
                   "--temporal_filter", "--w", "3", "--convert_to_rgb",
                   "--profile", args.profile,
                   "--exemplar_strategy", args.exemplar_strategy,
                   "--exemplar_refresh_every", str(args.exemplar_refresh_every)]
            for name in ("rule", "min_uncovered", "merge_tracks", "merge_iou", "handover",
                         "handover_iou", "seed_policy", "bidirectional"):
                value = getattr(args, name)
                if value is not None:
                    if isinstance(value, bool):
                        cmd.append("--" + name if value else "--no_" + name.replace("--", ""))
                    else:
                        cmd += ["--" + name, str(value)]
            if args.save_cache:
                cmd += ["--save_cache", str(Path(args.save_cache).resolve())]
            if args.replay_cache:
                cmd += ["--replay_cache", str(Path(args.replay_cache).resolve())]
            commands.append(cmd)

    print(f"Videos: {videos}  Profile: {args.profile}  Results: {out}", flush=True)
    if args.dry_run:
        import shlex
        for cmd in commands:
            print(shlex.join(cmd))
        return
    out.mkdir(parents=True, exist_ok=False)
    predicted.write_text("{}")
    (out / "ground-truth.json").write_text(json.dumps({v: gt[v] for v in videos}, indent=2))
    files = ["count_in_videos_pseudo_exemplars_v2.py", "counting_rules.py", "counting_cache.py",
             Path(__file__).name]
    metadata = {"settings": vars(args), "commands": commands,
                "source_sha256": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}
    (out / "settings.json").write_text(json.dumps(metadata, indent=2))
    start = time.monotonic()
    try:
        with (out / "run.log").open("w", buffering=1) as log:
            for cmd in commands:
                with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1) as child:
                    for line in child.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    code = child.wait()
                    if code:
                        raise subprocess.CalledProcessError(code, cmd)
        result = json.loads(predicted.read_text())
        for video in videos:
            assert set(result[str(data / "frames" / video)]) == set(gt[video]), "Incomplete predictions"
        evaluation = subprocess.run([sys.executable, "evaluate_counting_accuracy.py",
                                     "--ground_truth", str(out / "ground-truth.json"),
                                     "--predicted", str(predicted),
                                     "--parent_dir", str(data / "frames")],
                                    capture_output=True, text=True, check=True)
        print(evaluation.stdout, flush=True)
        (out / "metrics.txt").write_text(evaluation.stdout)
        metadata["status"] = "complete"
    except BaseException as error:
        metadata["status"] = "incomplete"
        metadata["error"] = repr(error)
        raise
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - start
        (out / "settings.json").write_text(json.dumps(metadata, indent=2))
        print("Results saved to:", out, flush=True)


if __name__ == "__main__":
    main()
