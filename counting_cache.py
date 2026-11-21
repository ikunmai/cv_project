"""Persist the expensive per-frame outputs of stage 1/2 so stage 3 can be re-run.

Stage 1 (CountGD + SAM 2 image predictor) and stage 2 (temporal filter) cost most
of the wall-clock time of a crystal video, yet they do not depend on the rules
that decide the final count. Caching their outputs turns "try another rule
configuration" from a full run into a stage-3-only run, and makes the rule
sweep in ``replay_crystals.py`` practical.

Layout of one cache directory (``<cache_dir>/<video>__<text-slug>/``):

    meta.json    frame names, video directory, frame size, text, timing
    raw.npz      CountGD boxes/scores + independent SAM 2 masks per frame
    filtered.npz same, after the temporal filter (what stage 3 consumes)

Both npz files share one schema, see ``save_masks``.
"""
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "prompt"


def save_masks(path: Path, frames: Dict[int, dict]) -> None:
    """Save ``{frame_idx: {"mask_ids": {id: mask}, "box_prompts": boxes}}``.

    ``boxes`` may be a torch tensor, a list or an ndarray; only its values are
    stored. Mask pixel coordinates are stored as int32, which keeps a 37-frame,
    100-object crystal video around 30 MB.
    """
    path = Path(path)
    n_frames = max(frames) + 1 if frames else 0
    frame_box_ptr = np.zeros(n_frames + 1, dtype=np.int64)
    frame_obj_ptr = np.zeros(n_frames + 1, dtype=np.int64)
    boxes, scores, obj_ptr, pts = [], [], [0], []
    for f in range(n_frames):
        entry = frames.get(f) or {"mask_ids": {}, "box_prompts": []}
        b = np.asarray(entry["box_prompts"], dtype=np.float32).reshape(-1, 4)
        boxes.append(b)
        s = np.asarray(entry.get("box_scores", []), dtype=np.float32).ravel()
        if s.size != len(b):            # scores are optional; keep the schema aligned
            s = np.full(len(b), np.nan, dtype=np.float32)
        scores.append(s)
        frame_box_ptr[f + 1] = frame_box_ptr[f] + len(b)
        ids = sorted(entry["mask_ids"])
        assert ids == list(range(1, len(b) + 1)), (
            f"frame {f}: mask ids must be 1..N and match the box list, got {ids[:5]}... "
            f"({len(ids)} masks, {len(b)} boxes)")
        for obj_id in ids:
            rows, cols = entry["mask_ids"][obj_id]
            rows = np.asarray(rows, dtype=np.int32).ravel()
            cols = np.asarray(cols, dtype=np.int32).ravel()
            pts.append(np.stack([rows, cols], axis=1))
            obj_ptr.append(obj_ptr[-1] + rows.size)
        frame_obj_ptr[f + 1] = len(obj_ptr) - 1
    cat = lambda xs, shape: (np.concatenate(xs) if xs else np.empty(shape))  # noqa: E731
    np.savez_compressed(
        path,
        frame_box_ptr=frame_box_ptr,
        frame_obj_ptr=frame_obj_ptr,
        boxes=cat(boxes, (0, 4)).astype(np.float32),
        scores=cat(scores, (0,)).astype(np.float32),
        obj_ptr=np.asarray(obj_ptr, dtype=np.int64),
        pts=cat(pts, (0, 2)).astype(np.int32),
    )


def load_masks(path: Path, frame_indices=None) -> Dict[int, dict]:
    """Load a cache file back into the pipeline's dict format."""
    data = np.load(Path(path))
    frame_box_ptr = data["frame_box_ptr"]
    frame_obj_ptr = data["frame_obj_ptr"]
    boxes, scores, pts = data["boxes"], data["scores"], data["pts"]
    n_frames = len(frame_box_ptr) - 1
    wanted = range(n_frames) if frame_indices is None else frame_indices
    out = {}
    for f in wanted:
        b = boxes[frame_box_ptr[f]:frame_box_ptr[f + 1]]
        s = scores[frame_box_ptr[f]:frame_box_ptr[f + 1]]
        masks = {}
        lo, hi = frame_obj_ptr[f], frame_obj_ptr[f + 1]
        for k in range(hi - lo):
            p = pts[data["obj_ptr"][lo + k]:data["obj_ptr"][lo + k + 1]]
            masks[k + 1] = (p[:, 0].astype(np.int64), p[:, 1].astype(np.int64))
        out[f] = {"mask_ids": masks, "box_prompts": b, "box_scores": s}
    return out


def save_cache(cache_dir: Path, video: str, text: str, meta: dict,
               raw_frames: Dict[int, dict], filtered_frames: Dict[int, dict]) -> Path:
    cache_dir = Path(cache_dir) / f"{video}__{_slug(text)}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_masks(cache_dir / "raw.npz", raw_frames)
    save_masks(cache_dir / "filtered.npz", filtered_frames)
    meta = dict(meta, video=video, text=text)
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return cache_dir


def load_cache(cache_dir: Path, which: str = "filtered"):
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    frames = load_masks(cache_dir / f"{which}.npz")
    return meta, frames
