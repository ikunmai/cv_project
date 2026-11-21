"""Counting rules shared by the video counting scripts.

The original pipeline (count_in_videos_pseudo_exemplars.py) applies three rules
that decide the final count:

1. an object is "new" only if its SAM 2 mask shares *no pixel* with any mask that
   is already tracked in that frame (get_new_objs);
2. every propagated track is kept, even when two tracks describe the same object;
3. the count is len(T), the number of tracks.

This module re-implements those rules in a configurable, testable way:

* ``strict``      -- the original zero-overlap rule (default, bit-compatible);
* ``coverage``    -- a new mask is accepted when the fraction of its area that is
                     not covered by tracked masks is at least ``min_uncovered``.
                     Dense scenes rarely produce perfectly disjoint masks, so the
                     strict rule silently drops new objects there;
* ``merge_tracks`` -- union-find post-processing that fuses duplicate tracks
                     (co-visible pairs with high mask IoU, and hand-over pairs
                     where a track dies and another starts on the same object).

All functions are pure NumPy, operate on masks stored as ``np.nonzero`` tuples
``(rows, cols)`` (the format used throughout the repository), and never need a
GPU, so they can be unit-tested and replayed offline.
"""
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Mask = Tuple[np.ndarray, np.ndarray]
MaskDict = Dict[int, Mask]

# Masks are packed as row * _PACK + col. 2**20 covers frames up to one megapixel
# wide, which is far beyond any benchmark used here (Crystals frames are ~1920 px).
_PACK = 1 << 20


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------
def as_arrays(mask: Mask) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(rows, cols)`` as flat int64 arrays, accepting lists as well."""
    rows = np.asarray(mask[0], dtype=np.int64).ravel()
    cols = np.asarray(mask[1], dtype=np.int64).ravel()
    return rows, cols


def mask_area(mask: Mask) -> int:
    rows, _ = as_arrays(mask)
    return int(rows.size)


def pack_mask(mask: Mask) -> np.ndarray:
    rows, cols = as_arrays(mask)
    return rows * _PACK + cols


def unpack_mask(packed: np.ndarray) -> Mask:
    packed = np.asarray(packed, dtype=np.int64)
    return packed // _PACK, packed % _PACK


def mask_bbox(mask: Mask) -> Optional[Tuple[int, int, int, int]]:
    """Return ``(x0, y0, x1, y1)`` with x1/y1 exclusive, or None for empty masks."""
    rows, cols = as_arrays(mask)
    if rows.size == 0:
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def _bbox_overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def mask_iou(mask_a: Mask, mask_b: Mask) -> float:
    """Pixel IoU of two masks, identical in value to the original set-based code."""
    rows_a, cols_a = as_arrays(mask_a)
    rows_b, cols_b = as_arrays(mask_b)
    if rows_a.size == 0 or rows_b.size == 0:
        return 0.0
    # Cheap reject: masks whose bounding boxes do not touch cannot intersect.
    if (cols_a.max() < cols_b.min() or cols_b.max() < cols_a.min()
            or rows_a.max() < rows_b.min() or rows_b.max() < rows_a.min()):
        return 0.0
    inter = np.intersect1d(rows_a * _PACK + cols_a, rows_b * _PACK + cols_b,
                           assume_unique=True).size
    if inter == 0:
        return 0.0
    union = rows_a.size + rows_b.size - inter
    return float(inter) / float(union) if union else 0.0


def mask_containment(inner: Mask, outer: Mask) -> float:
    """Fraction of ``inner`` that lies inside ``outer``."""
    rows_i, cols_i = as_arrays(inner)
    if rows_i.size == 0:
        return 0.0
    rows_o, cols_o = as_arrays(outer)
    if rows_o.size == 0:
        return 0.0
    if (cols_i.max() < cols_o.min() or cols_o.max() < cols_i.min()
            or rows_i.max() < rows_o.min() or rows_o.max() < rows_i.min()):
        return 0.0
    inter = np.intersect1d(rows_i * _PACK + cols_i, rows_o * _PACK + cols_o,
                           assume_unique=True).size
    return float(inter) / float(rows_i.size)


class FrameIndex:
    """Union of all tracked masks in one frame, built once for fast queries.

    Replaces the per-pair Python ``set`` intersections of the original
    ``get_new_objs``/``check_obj_persistence`` with a sorted array lookup, which
    is one to two orders of magnitude faster on dense frames (see
    ``bench_mask_ops.py``).
    """

    def __init__(self, masks: Iterable[Mask]):
        packed = [pack_mask(m) for m in masks if mask_area(m) > 0]
        if packed:
            union = np.unique(np.concatenate(packed))
            self.union = union
            self.union_area = int(union.size)
        else:
            self.union = np.empty(0, dtype=np.int64)
            self.union_area = 0

    def packed_covered(self, mask: Mask) -> int:
        """Number of pixels of ``mask`` covered by the union."""
        packed = pack_mask(mask)
        if packed.size == 0 or self.union.size == 0:
            return 0
        return int(np.isin(packed, self.union, assume_unique=True).sum())

    def covered_fraction(self, mask: Mask) -> float:
        area = mask_area(mask)
        return self.packed_covered(mask) / area if area else 0.0


# ---------------------------------------------------------------------------
# Rule 1: is a candidate mask a new object?
# ---------------------------------------------------------------------------
@dataclass
class CountConfig:
    """All knobs that decide how many objects the video contains."""

    # New-object rule.
    rule: str = "strict"             # "strict" (original) | "coverage"
    min_uncovered: float = 0.5       # coverage rule: uncovered fraction of the candidate
    min_obj_area: int = 0            # pixels; 0 disables (original default)

    # Track de-duplication.
    merge: bool = False
    merge_iou: float = 0.5           # co-visible duplicate pair threshold
    merge_min_common: int = 2        # frames a duplicate pair must share
    handover: bool = False           # fuse "track dies, new track starts on it"
    handover_iou: float = 0.5
    handover_max_gap: int = 1        # missing frames tolerated between the two tracks

    # Track pruning.
    min_track_frames: int = 1        # drop tracks seen in fewer frames (1 = keep all)

    # Seeding and propagation direction.
    seed_policy: str = "first"       # "first" (v1) | "most_detections"
    bidirectional: bool = False      # also propagate the seed boxes backwards

    # SAM 2 propagation batching (keeps the v1 behaviour).
    obj_batch_size: int = 30

    def validate(self):
        if self.rule not in ("strict", "coverage"):
            raise ValueError(f"unknown rule: {self.rule}")
        if self.seed_policy not in ("first", "most_detections"):
            raise ValueError(f"unknown seed_policy: {self.seed_policy}")
        if not 0.0 <= self.min_uncovered <= 1.0:
            raise ValueError("min_uncovered must be in [0, 1]")
        if not 0.0 <= self.merge_iou <= 1.0 or not 0.0 <= self.handover_iou <= 1.0:
            raise ValueError("IoU thresholds must be in [0, 1]")
        if self.merge_min_common < 1:
            raise ValueError("merge_min_common must be >= 1")
        if self.min_track_frames < 1:
            raise ValueError("min_track_frames must be >= 1")
        return self

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def with_(self, **kwargs) -> "CountConfig":
        return replace(self, **kwargs)


#: Profiles used by the runners and the simulation. ``baseline`` is the code
#: path that produced the team's recorded numbers.
PROFILES: Dict[str, CountConfig] = {
    "baseline": CountConfig(rule="strict", merge=False, handover=False),
    "coverage": CountConfig(rule="coverage", min_uncovered=0.5, merge=False),
    "balanced": CountConfig(rule="coverage", min_uncovered=0.5,
                            merge=True, merge_iou=0.5, merge_min_common=2),
    "recall": CountConfig(rule="coverage", min_uncovered=0.35,
                          merge=True, merge_iou=0.45, merge_min_common=2,
                          handover=True, handover_iou=0.5),
    # Ablation: v1 rules, but seeded at the densest frame and propagated both ways.
    "seed_only": CountConfig(rule="strict", merge=False,
                             seed_policy="most_detections", bidirectional=True),
    # Seeding at the frame with the most detections and propagating backwards
    # too. Suits scenes where objects are present throughout but small at the
    # start (growing crystals); can miss objects only visible before the seed.
    "full": CountConfig(rule="coverage", min_uncovered=0.5,
                        merge=True, merge_iou=0.55, merge_min_common=2,
                        seed_policy="most_detections", bidirectional=True),
}


def is_new_object(candidate: Mask, index: FrameIndex, config: CountConfig) -> bool:
    """Decide whether ``candidate`` should start a new track."""
    area = mask_area(candidate)
    if area == 0 or area < config.min_obj_area:
        return False
    if config.rule == "strict":
        # Original get_new_objs: any shared pixel disqualifies the candidate.
        return index.packed_covered(candidate) == 0
    uncovered = 1.0 - index.covered_fraction(candidate)
    return uncovered >= config.min_uncovered


def select_new_objects(frame_pred: dict, tracked_masks: Sequence[Mask],
                       config: CountConfig) -> List[int]:
    """Indices of the independent per-frame masks that should start new tracks.

    ``frame_pred`` follows the repository layout:
    ``{"mask_ids": {obj_id: mask}, "box_prompts": tensor or list}``.
    ``tracked_masks`` are the masks already in T for this frame.
    """
    boxes = frame_pred.get("box_prompts")
    if boxes is None or len(boxes) == 0:
        return []
    index = FrameIndex(tracked_masks)
    box_index = {obj_id: k for k, obj_id in enumerate(sorted(frame_pred["mask_ids"]))}
    return [box_index[obj_id] for obj_id in sorted(frame_pred["mask_ids"])
            if is_new_object(frame_pred["mask_ids"][obj_id], index, config)]


# ---------------------------------------------------------------------------
# Rule 2: merge duplicate tracks
# ---------------------------------------------------------------------------
def track_frames(track: MaskDict) -> List[int]:
    return sorted(track)


def _common_frames(a: MaskDict, b: MaskDict) -> List[int]:
    return sorted(set(a).intersection(b))


def find_duplicate_pairs(T: Dict[int, MaskDict], config: CountConfig,
                         verbose: bool = False,
                         max_frames_checked: int = 5) -> List[Tuple[int, int, str, float]]:
    """Return ``(id_a, id_b, reason, score)`` for track pairs that are the same object.

    Every pair is pre-filtered by cached bounding boxes, and co-visibility is
    judged on at most ``max_frames_checked`` shared frames, so the quadratic
    pair loop stays cheap even with a few hundred tracks.
    """
    ids = sorted(T)
    bbox = {(o, f): mask_bbox(m) for o, track in T.items() for f, m in track.items()}
    pairs = []
    for i, a in enumerate(ids):
        fa = track_frames(T[a])
        if not fa:
            continue
        for b in ids[i + 1:]:
            fb = track_frames(T[b])
            if not fb:
                continue
            common = _common_frames(T[a], T[b])
            if len(common) >= config.merge_min_common:
                if len(common) > max_frames_checked:
                    pick = np.linspace(0, len(common) - 1, max_frames_checked).astype(int)
                    common = [common[j] for j in pick]
                ious = []
                for f in common:
                    ba, bb = bbox[(a, f)], bbox[(b, f)]
                    if ba is None or bb is None or not _bbox_overlaps(ba, bb):
                        ious.append(0.0)
                    else:
                        ious.append(mask_iou(T[a][f], T[b][f]))
                iou = float(np.mean(ious)) if ious else 0.0
                if iou >= config.merge_iou:
                    pairs.append((a, b, "covisible", iou))
                    continue
            if config.handover:
                # The later track starts after the earlier one ended.
                if fa[-1] < fb[0]:
                    (early, late), (fe, fl) = (a, b), (fa, fb)
                elif fb[-1] < fa[0]:
                    (early, late), (fe, fl) = (b, a), (fb, fa)
                else:
                    continue
                gap = fl[0] - fe[-1] - 1
                if 0 <= gap <= config.handover_max_gap:
                    be, bl = bbox[(early, fe[-1])], bbox[(late, fl[0])]
                    if be is None or bl is None or not _bbox_overlaps(be, bl):
                        continue
                    iou = mask_iou(T[early][fe[-1]], T[late][fl[0]])
                    if iou >= config.handover_iou:
                        pairs.append((a, b, "handover", iou))
    if verbose:
        for p in pairs:
            print(f"[merge] duplicate pair {p[0]}~{p[1]} ({p[2]} IoU={p[3]:.2f})")
    return pairs


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def merge_tracks(T: Dict[int, MaskDict], config: CountConfig,
                 verbose: bool = False) -> Tuple[Dict[int, MaskDict], dict]:
    """Fuse duplicate tracks; returns the merged ``T`` and a small report.

    When two tracks share a frame, the larger mask is kept: propagated masks of
    duplicate tracks are two observations of one object, and the larger one is
    the more complete segmentation.
    """
    T = {k: dict(v) for k, v in T.items() if v}
    pairs = find_duplicate_pairs(T, config, verbose=verbose)
    report = {"pairs": pairs, "merged_away": 0}
    if not pairs:
        return T, report
    uf = _UnionFind(list(T))
    for a, b, _, _ in pairs:
        uf.union(a, b)
    groups: Dict[int, List[int]] = {}
    for obj_id in T:
        groups.setdefault(uf.find(obj_id), []).append(obj_id)
    merged: Dict[int, MaskDict] = {}
    for root, members in groups.items():
        if len(members) == 1:
            merged[members[0]] = T[members[0]]
            continue
        # Keep the earliest id so logs stay comparable with the v1 output.
        keep = min(members)
        frames: MaskDict = {}
        for m in sorted(members):
            for f, mask in T[m].items():
                if f not in frames or mask_area(mask) > mask_area(frames[f]):
                    frames[f] = mask
        merged[keep] = frames
        report["merged_away"] += len(members) - 1
    return merged, report


def prune_tracks(T: Dict[int, MaskDict], config: CountConfig) -> Tuple[Dict[int, MaskDict], int]:
    if config.min_track_frames <= 1:
        return T, 0
    kept = {k: v for k, v in T.items() if len(v) >= config.min_track_frames}
    return kept, len(T) - len(kept)


# ---------------------------------------------------------------------------
# Rule 3: the stage-3 loop (seed + branch A), shared by the pipeline and the sim
# ---------------------------------------------------------------------------
Propagator = Iterable  # callable(frame_idx, {obj_id: box}) -> iterable[(frame, {obj_id: mask})]


def track_and_count(sam_pred_masks: Dict[int, dict],
                    n_frames: int,
                    propagate,
                    config: CountConfig,
                    verbose: bool = False) -> Tuple[Dict[int, MaskDict], dict]:
    """Stage 3 of CountVid: seed tracks, then admit uncovered objects, then merge.

    ``propagate(frame_idx, {obj_id: box}, reverse)`` must run SAM 2 video
    propagation from ``frame_idx`` (forward, or backwards when ``reverse`` is
    true) and yield ``(frame_idx, {obj_id: (rows, cols)})`` per frame, exactly
    like ``predictor.propagate_in_video`` wrapped by the caller. Keeping
    propagation behind a callback lets the same code run on GPUs and inside the
    simulator.
    """
    config.validate()
    T: Dict[int, MaskDict] = {}
    info = {"seed_frame": 0, "branches": 0, "added": 0, "candidates": 0}

    def n_boxes(f):
        return len(sam_pred_masks.get(f, {}).get("box_prompts", []))

    if config.seed_policy == "most_detections":
        i = int(max(range(n_frames), key=n_boxes)) if n_frames else 0
    else:
        i = 0
        for j in range(n_frames):
            if n_boxes(j) > 0:
                i = j
                break
    info["seed_frame"] = i
    seed_boxes = sam_pred_masks.get(i, {}).get("box_prompts", [])
    if len(seed_boxes) == 0:
        return T, info

    def _run_branch(frame_idx: int, boxes, first_id: int, reverse: bool = False):
        num_batches = int(np.ceil(len(boxes) / config.obj_batch_size))
        for batch_ind in range(num_batches):
            lo = batch_ind * config.obj_batch_size
            hi = min((batch_ind + 1) * config.obj_batch_size, len(boxes))
            prompts = {first_id + k: boxes[k] for k in range(lo, hi)}
            for out_frame, masks in propagate(frame_idx, prompts, reverse):
                for obj_id, mask in masks.items():
                    T.setdefault(obj_id, {})[out_frame] = mask

    # 1.-3. Seed from the first frame that has detections (or the densest one).
    _run_branch(i, list(seed_boxes), 1, reverse=False)
    if config.bidirectional and i > 0:
        _run_branch(i, list(seed_boxes), 1, reverse=True)

    # 4.-6. Walk forward and admit objects that the current tracks do not explain.
    while i < n_frames - 1:
        nxt = i + 1
        frame_pred = sam_pred_masks.get(nxt)
        if frame_pred is not None:
            tracked = [T[obj_id][nxt] for obj_id in T if nxt in T[obj_id]]
            idxs = select_new_objects(frame_pred, tracked, config)
            info["candidates"] += len(idxs)
            if idxs:
                boxes = [frame_pred["box_prompts"][k] for k in idxs]
                last_id = max(T) if T else 0
                _run_branch(nxt, boxes, last_id + 1)
                info["branches"] += 1
                info["added"] += len(boxes)
        i = nxt

    n_before = len(T)
    if config.merge or config.handover:
        T, merge_report = merge_tracks(T, config, verbose=verbose)
        info["merged_away"] = merge_report["merged_away"]
    T, pruned = prune_tracks(T, config)
    info["pruned"] = pruned
    info["tracks_before_post"] = n_before
    info["count"] = len(T)
    return T, info


# ---------------------------------------------------------------------------
# Rule 0: pseudo-exemplar selection (model side, NumPy implementation)
# ---------------------------------------------------------------------------
def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS in score order; returns kept indices."""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx0 = np.maximum(x0[i], x0[rest])
        yy0 = np.maximum(y0[i], y0[rest])
        xx1 = np.minimum(x1[i], x1[rest])
        yy1 = np.minimum(y1[i], y1[rest])
        inter = np.maximum(0.0, xx1 - xx0) * np.maximum(0.0, yy1 - yy0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def select_pseudo_exemplars(boxes_xyxy: np.ndarray, scores: np.ndarray,
                            num: int, strategy: str = "topk",
                            nms_iou: float = 0.4, grid: int = 4,
                            min_side: float = 2.0,
                            image_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Pick the box prompts used as pseudo-exemplars for the next frame.

    ``topk``       -- the original rule: highest scores first.
    ``diverse``    -- score-ordered NMS, then round-robin over a ``grid x grid``
                      spatial layout so the exemplars cover the image instead of
                      clustering on the most confident (and easiest) region.

    Returns an ``(n, 4)`` float array of xyxy boxes; ``n <= num``.
    """
    if num <= 0 or len(boxes_xyxy) == 0:
        return np.empty((0, 4), dtype=np.float32)
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(boxes_xyxy)) or not np.all(np.isfinite(scores)):
        ok = np.all(np.isfinite(boxes_xyxy), axis=1) & np.isfinite(scores)
        boxes_xyxy, scores = boxes_xyxy[ok], scores[ok]
        if len(boxes_xyxy) == 0:
            return np.empty((0, 4), dtype=np.float32)
    w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    valid = (w >= min_side) & (h >= min_side)
    boxes_xyxy, scores = boxes_xyxy[valid], scores[valid]
    if len(boxes_xyxy) == 0:
        return np.empty((0, 4), dtype=np.float32)
    if strategy == "topk":
        order = np.argsort(-scores)[:num]
        return boxes_xyxy[order].astype(np.float32)
    if strategy != "diverse":
        raise ValueError(f"unknown exemplar strategy: {strategy}")
    keep = nms_xyxy(boxes_xyxy, scores, nms_iou)
    boxes_xyxy, scores = boxes_xyxy[keep], scores[keep]
    if len(boxes_xyxy) <= num:
        return boxes_xyxy.astype(np.float32)
    if image_size is None:
        x_max = max(1.0, boxes_xyxy[:, 2].max())
        y_max = max(1.0, boxes_xyxy[:, 3].max())
    else:
        y_max, x_max = float(image_size[0]), float(image_size[1])
    # Score-ordered round robin over occupied grid cells.
    cx = 0.5 * (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / x_max
    cy = 0.5 * (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / y_max
    cell = np.clip((cy * grid).astype(int), 0, grid - 1) * grid + np.clip((cx * grid).astype(int), 0, grid - 1)
    order = np.argsort(-scores)
    picked, used_cells = [], set()
    for idx in order:                       # first pass: one per cell
        if cell[idx] not in used_cells:
            used_cells.add(cell[idx])
            picked.append(idx)
            if len(picked) == num:
                break
    if len(picked) < num:                   # second pass: fill remaining slots
        for idx in order:
            if idx not in picked:
                picked.append(idx)
                if len(picked) == num:
                    break
    return boxes_xyxy[np.asarray(picked)].astype(np.float32)
