"""Unit tests for counting_rules. Run: python -m pytest tests/ -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import counting_rules as cr  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def rect(y, x, h=6, w=8, H=120, W=160):
    rows, cols = np.mgrid[y:y + h, x:x + w]
    return rows.ravel().astype(np.int64), cols.ravel().astype(np.int64)


def original_compute_iou(mask1_indices, mask2_indices):
    """The set-based IoU used by the original pipeline."""
    set1 = set(zip(mask1_indices[0], mask1_indices[1]))
    set2 = set(zip(mask2_indices[0], mask2_indices[1]))
    inter = len(set1 & set2)
    if inter == 0:
        return 0
    union = len(set1 | set2)
    return inter / union if union else 0.0


def original_get_new_objs(frame_pred, T, frame_idx, min_obj_area=0):
    """Original get_new_objs with the tracked masks already resolved for the frame."""
    add = []
    if len(frame_pred["box_prompts"]) == 0:
        return add
    objs = [obj_id for obj_id in T if frame_idx in T[obj_id]]
    from functools import reduce
    coords = [set(zip(T[o][frame_idx][0], T[o][frame_idx][1])) for o in objs]
    union = reduce(lambda a, b: a | b, coords) if coords else set()
    for idx in range(len(frame_pred["box_prompts"])):
        mask = set(zip(frame_pred["mask_ids"][idx + 1][0], frame_pred["mask_ids"][idx + 1][1]))
        if len(mask) >= min_obj_area and len(mask & union) == 0:
            add.append(idx)
    return add


def original_stage3(sam_pred_masks, n_frames, propagate, obj_batch_size=30):
    """Verbatim port of the v1 stage-3 loop (including its batch-offset bug)."""
    T = {}
    i = 0
    for j in range(n_frames):
        if len(sam_pred_masks[j]["box_prompts"]) > 0:
            i = j
            break
    if len(sam_pred_masks[i]["box_prompts"]) == 0:
        return T
    num_batches = int(np.ceil(len(sam_pred_masks[i]["box_prompts"]) / obj_batch_size))
    for b in range(num_batches):
        prompts = {k + 1: sam_pred_masks[i]["box_prompts"][k]
                   for k in range(b * obj_batch_size,
                                  min((b + 1) * obj_batch_size, len(sam_pred_masks[i]["box_prompts"])))}
        for f, masks in propagate(i, prompts):
            for oid, m in masks.items():
                T.setdefault(oid, {})[f] = m
    while i < n_frames - 1:
        fp = sam_pred_masks[i + 1]
        new = original_get_new_objs(fp, T, i + 1)
        if new:
            boxes = [fp["box_prompts"][k] for k in new]
            n_batches = int(np.ceil(len(boxes) / obj_batch_size))
            for b in range(n_batches):
                last = max(T)
                prompts = {}
                # NB: original index bug reproduced on purpose.
                for ind in range(min(obj_batch_size, len(boxes) - b * obj_batch_size)):
                    prompts[last + ind + 1] = boxes[ind]
                for f, masks in propagate(i + 1, prompts):
                    for oid, m in masks.items():
                        T.setdefault(oid, {})[f] = m
        i += 1
    return T


# ---------------------------------------------------------------------------
# mask helpers
# ---------------------------------------------------------------------------
def test_iou_matches_original():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y1, x1, y2, x2 = rng.integers(0, 60, 4)
        a = rect(int(y1), int(x1), int(rng.integers(1, 20)), int(rng.integers(1, 20)))
        b = rect(int(y2), int(x2), int(rng.integers(1, 20)), int(rng.integers(1, 20)))
        assert abs(cr.mask_iou(a, b) - original_compute_iou(a, b)) < 1e-12


def test_iou_edge_cases():
    empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    assert cr.mask_iou(empty, rect(0, 0)) == 0.0
    assert cr.mask_iou(rect(0, 0), rect(0, 0)) == 1.0
    assert cr.mask_iou(rect(0, 0), rect(50, 60)) == 0.0


def test_frame_index_coverage():
    tracked = [rect(0, 0, 10, 10), rect(0, 20, 10, 10)]
    idx = cr.FrameIndex(tracked)
    assert idx.packed_covered(rect(0, 0, 10, 10)) == 100
    # 25 % of the candidate overlaps the first tracked mask (rows/cols 5..9).
    assert abs(idx.covered_fraction(rect(5, 5, 10, 10)) - 0.25) < 1e-9
    assert idx.packed_covered(rect(60, 60, 10, 10)) == 0


# ---------------------------------------------------------------------------
# new-object rule
# ---------------------------------------------------------------------------
def test_strict_rule_matches_original():
    rng = np.random.default_rng(1)
    H, W = 100, 120
    for trial in range(30):
        T = {o: {0: rect(int(rng.integers(0, 80)), int(rng.integers(0, 100)))}
             for o in range(1, 6)}
        frame = {"mask_ids": {}, "box_prompts": np.zeros((8, 4))}
        for k in range(1, 9):
            frame["mask_ids"][k] = rect(int(rng.integers(0, 80)), int(rng.integers(0, 100)),
                                        int(rng.integers(3, 20)), int(rng.integers(3, 20)))
        got = cr.select_new_objects(frame, [T[o][0] for o in T], cr.PROFILES["baseline"])
        want = original_get_new_objs(frame, T, 0)
        assert got == want, (trial, got, want)


def test_coverage_rule_admits_occluded_but_not_duplicate():
    tracked = [rect(0, 0, 20, 20)]
    cfg = cr.PROFILES["coverage"]
    # A duplicate shifted by 4 px: 64 % covered -> rejected.
    assert not cr.is_new_object(rect(4, 4, 20, 20), cr.FrameIndex(tracked), cfg)
    # A genuinely new object touching the tracked one by 25 % -> admitted.
    assert cr.is_new_object(rect(15, 15, 20, 20), cr.FrameIndex(tracked), cfg)
    # Strict rule rejects both.
    assert not cr.is_new_object(rect(15, 15, 20, 20), cr.FrameIndex(tracked),
                                cr.PROFILES["baseline"])


def test_min_area_respected():
    cfg = cr.CountConfig(rule="coverage", min_obj_area=50)
    assert not cr.is_new_object(rect(0, 0, 5, 5), cr.FrameIndex([]), cfg)
    assert cr.is_new_object(rect(0, 0, 10, 10), cr.FrameIndex([]), cfg)


# ---------------------------------------------------------------------------
# track merging
# ---------------------------------------------------------------------------
def test_merge_covisible_duplicates():
    obj = [rect(0, 0, 20, 20), rect(0, 1, 20, 20), rect(0, 0, 20, 21)]
    T = {
        1: {0: obj[0], 1: obj[1], 2: obj[2]},
        2: {0: rect(0, 1, 20, 20), 1: rect(0, 0, 20, 21), 2: rect(0, 0, 20, 20)},  # duplicate
        3: {0: rect(60, 60, 20, 20), 1: rect(60, 60, 20, 20), 2: rect(60, 60, 20, 20)},  # distinct
    }
    merged, report = cr.merge_tracks(T, cr.CountConfig(merge=True, merge_iou=0.5,
                                                       merge_min_common=2))
    assert len(merged) == 2
    assert report["merged_away"] == 1
    assert set(merged) == {1, 3}
    assert sorted(merged[1]) == [0, 1, 2]


def test_no_merge_below_common_frames():
    T = {1: {0: rect(0, 0, 20, 20), 1: rect(0, 0, 20, 20)},
         2: {0: rect(0, 0, 20, 20)}}  # only one shared frame
    merged, _ = cr.merge_tracks(T, cr.CountConfig(merge=True, merge_iou=0.5,
                                                  merge_min_common=2))
    assert len(merged) == 2


def test_handover_merge():
    T = {1: {0: rect(10, 10, 20, 20), 1: rect(10, 10, 20, 20)},
         2: {3: rect(10, 11, 20, 20), 4: rect(10, 10, 20, 20)}}  # reappears after a gap
    cfg = cr.CountConfig(merge=True, handover=True, handover_iou=0.5, handover_max_gap=1)
    merged, _ = cr.merge_tracks(T, cfg)
    assert len(merged) == 1
    assert sorted(merged[1]) == [0, 1, 3, 4]


def test_handover_gap_too_large():
    T = {1: {0: rect(10, 10, 20, 20)},
         2: {5: rect(10, 10, 20, 20)}}
    cfg = cr.CountConfig(merge=True, handover=True, handover_iou=0.5, handover_max_gap=1)
    merged, _ = cr.merge_tracks(T, cfg)
    assert len(merged) == 2


def test_prune_tracks():
    T = {1: {0: rect(0, 0), 1: rect(0, 0)}, 2: {0: rect(40, 40)}}
    kept, removed = cr.prune_tracks(T, cr.CountConfig(min_track_frames=2))
    assert set(kept) == {1} and removed == 1


# ---------------------------------------------------------------------------
# stage-3 loop
# ---------------------------------------------------------------------------
class FakePropagator:
    """Propagation model: a prompted box keeps yielding the mask of the object
    whose box it matches, from the prompt frame to the end of the video."""

    def __init__(self, n_frames, object_masks, box_to_object):
        self.n_frames = n_frames
        self.object_masks = object_masks  # list per frame of {object: mask}
        self.box_to_object = box_to_object  # box tuple -> object id

    def __call__(self, frame_idx, prompts, reverse=False):
        frames = range(frame_idx, -1, -1) if reverse else range(frame_idx, self.n_frames)
        for f in frames:
            out = {}
            for oid, box in prompts.items():
                obj = self.box_to_object[tuple(np.round(np.asarray(box, dtype=float), 3))]
                mask = self.object_masks[f].get(obj)
                if mask is not None:
                    out[oid] = mask
            yield f, out


def _toy_video(n_frames=6, H=100, W=120):
    """Objects 0 and 1 exist from frame 0; object 2 appears at frame 3."""
    object_masks = []
    for f in range(n_frames):
        frame = {0: rect(0, 0), 1: rect(0, 40)}
        if f >= 3:
            frame[2] = rect(50, 80)
        object_masks.append(frame)
    boxes = {0: np.array([0, 0, 8, 6], dtype=float),
             1: np.array([40, 0, 48, 6], dtype=float),
             2: np.array([80, 50, 88, 56], dtype=float)}
    box_to_object = {tuple(np.round(b, 3)): o for o, b in boxes.items()}
    return object_masks, boxes, box_to_object


def test_track_and_count_matches_original_loop():
    object_masks, boxes, b2o = _toy_video()
    n = len(object_masks)
    sam_pred = {}
    for f in range(n):
        ids = {k + 1: m for k, m in enumerate(object_masks[f].values())}
        sam_pred[f] = {"mask_ids": ids,
                       "box_prompts": [boxes[o] for o in object_masks[f]]}
    T_ref = original_stage3(sam_pred, n, FakePropagator(n, object_masks, b2o))
    T_new, info = cr.track_and_count(sam_pred, n,
                                     FakePropagator(n, object_masks, b2o),
                                     cr.PROFILES["baseline"])
    assert len(T_ref) == len(T_new) == 3
    assert info["count"] == 3
    assert {k: sorted(v) for k, v in T_ref.items()} == {k: sorted(v) for k, v in T_new.items()}


def test_branch_a_batch_offset_is_fixed():
    """More than obj_batch_size new objects in one frame must all be tracked once."""
    n_frames, H, W = 4, 200, 200
    n_new = 5
    object_masks = []
    for f in range(n_frames):
        frame = {0: rect(0, 0, 6, 6)}
        if f >= 1:
            for k in range(n_new):
                frame[1 + k] = rect(20 + 10 * k, 20, 6, 6)
        object_masks.append(frame)
    boxes = {0: np.array([0, 0, 6, 6], dtype=float)}
    for k in range(n_new):
        boxes[1 + k] = np.array([20, 20 + 10 * k, 26, 26 + 10 * k], dtype=float)
    box_to_object = {tuple(np.round(b, 3)): o for o, b in boxes.items()}
    sam_pred = {}
    for f in range(n_frames):
        sam_pred[f] = {"mask_ids": {k + 1: m for k, m in enumerate(object_masks[f].values())},
                       "box_prompts": [boxes[o] for o in object_masks[f]]}
    prop = FakePropagator(n_frames, object_masks, box_to_object)
    cfg = cr.PROFILES["baseline"].with_(obj_batch_size=2)
    T, info = cr.track_and_count(sam_pred, n_frames, prop, cfg)
    assert len(T) == 1 + n_new, (len(T), info)
    T_buggy = original_stage3(sam_pred, n_frames, prop, obj_batch_size=2)
    assert len(T_buggy) > len(T), "the v1 batch-offset bug should inflate the count"


def test_coverage_rule_finds_more_objects_in_dense_scene():
    """A new object overlapping a tracked mask is missed by strict, found by coverage."""
    n_frames = 4
    object_masks = []
    for f in range(n_frames):
        frame = {0: rect(0, 0, 20, 20)}
        if f >= 2:
            frame[1] = rect(10, 10, 20, 20)   # overlaps object 0 by 25 %
        object_masks.append(frame)
    boxes = {0: np.array([0, 0, 20, 20], dtype=float),
             1: np.array([10, 10, 30, 30], dtype=float)}
    box_to_object = {tuple(np.round(b, 3)): o for o, b in boxes.items()}
    sam_pred = {f: {"mask_ids": {k + 1: m for k, m in enumerate(ms.values())},
                    "box_prompts": [boxes[o] for o in ms]}
                for f, ms in enumerate(object_masks)}
    prop = FakePropagator(n_frames, object_masks, box_to_object)
    T_b, _ = cr.track_and_count(sam_pred, n_frames, prop, cr.PROFILES["baseline"])
    T_c, _ = cr.track_and_count(sam_pred, n_frames, prop, cr.PROFILES["coverage"])
    assert len(T_b) == 1
    assert len(T_c) == 2


# ---------------------------------------------------------------------------
# pseudo-exemplars
# ---------------------------------------------------------------------------
def test_topk_matches_original_order():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=float)
    scores = np.array([0.5, 0.9, 0.7])
    sel = cr.select_pseudo_exemplars(boxes, scores, num=2, strategy="topk")
    assert np.allclose(sel, boxes[[1, 2]])


def test_diverse_spreads_over_grid():
    # 40 near-duplicate boxes in one corner with the highest scores, plus
    # well-separated lower-score boxes elsewhere.
    boxes, scores = [], []
    for k in range(40):
        boxes.append([2 + 0.1 * k, 2 + 0.1 * k, 12 + 0.1 * k, 12 + 0.1 * k])
        scores.append(0.99)
    for k, (x, y) in enumerate([(80, 10), (10, 80), (80, 80), (45, 45)]):
        boxes.append([x, y, x + 10, y + 10])
        scores.append(0.6 - 0.05 * k)
    sel = cr.select_pseudo_exemplars(np.array(boxes), np.array(scores), num=4,
                                     strategy="diverse", nms_iou=0.4,
                                     image_size=(100, 100))
    assert len(sel) == 4
    centers = np.stack([(sel[:, 0] + sel[:, 2]) / 2, (sel[:, 1] + sel[:, 3]) / 2], 1)
    assert len({(int(c[0] // 25), int(c[1] // 25)) for c in centers}) >= 3


def test_exemplars_filter_tiny_and_bad_boxes():
    boxes = np.array([[0, 0, 1, 1], [10, 10, 30, 30], [np.nan, 0, 5, 5]])
    scores = np.array([0.9, 0.8, 0.95])
    sel = cr.select_pseudo_exemplars(boxes, scores, num=5, strategy="topk", min_side=2.0)
    assert len(sel) == 1


# ---------------------------------------------------------------------------
# seeding policy and bidirectional propagation
# ---------------------------------------------------------------------------
def test_seed_policy_most_detections():
    object_masks, boxes, b2o = _toy_video()
    n = len(object_masks)
    sam_pred = {f: {"mask_ids": {k + 1: m for k, m in enumerate(object_masks[f].values())},
                    "box_prompts": [boxes[o] for o in object_masks[f]]}
                for f in range(n)}
    prop = FakePropagator(n, object_masks, b2o)
    cfg = cr.PROFILES["baseline"].with_(seed_policy="most_detections")
    T, info = cr.track_and_count(sam_pred, n, prop, cfg)
    assert info["seed_frame"] == max(range(n), key=lambda f: len(sam_pred[f]["box_prompts"]))
    assert len(T) == 3


def test_bidirectional_recovers_earlier_frames():
    """Seeding on a later frame plus backward propagation fills earlier frames."""
    n = 5
    object_masks = [{0: rect(0, 0), 1: rect(0, 40)} for _ in range(n)]
    boxes = {0: np.array([0, 0, 8, 6], dtype=float),
             1: np.array([40, 0, 48, 6], dtype=float)}
    b2o = {tuple(np.round(b, 3)): o for o, b in boxes.items()}
    # Only the last frame has detections.
    sam_pred = {f: {"mask_ids": {}, "box_prompts": []} for f in range(n)}
    sam_pred[n - 1] = {"mask_ids": {1: object_masks[n - 1][0], 2: object_masks[n - 1][1]},
                       "box_prompts": [boxes[0], boxes[1]]}
    prop = FakePropagator(n, object_masks, b2o)
    cfg = cr.PROFILES["baseline"].with_(bidirectional=True)
    T, info = cr.track_and_count(sam_pred, n, prop, cfg)
    assert info["seed_frame"] == n - 1
    assert len(T) == 2
    for track in T.values():
        assert sorted(track) == list(range(n)), sorted(track)


# ---------------------------------------------------------------------------
# cache round-trip feeding stage 3
# ---------------------------------------------------------------------------
def test_cache_roundtrip_feeds_stage3(tmp_path):
    import counting_cache as cc

    object_masks, boxes, b2o = _toy_video()
    n = len(object_masks)
    sam_pred = {f: {"mask_ids": {k + 1: m for k, m in enumerate(object_masks[f].values())},
                    "box_prompts": [boxes[o] for o in object_masks[f]]}
                for f in range(n)}
    cache_dir = cc.save_cache(tmp_path, "video0", "white crystal",
                              {"frame_names": [f"{i:05d}.jpg" for i in range(n)],
                               "frame_original_names": [f"img_{i}.jpg" for i in range(n)],
                               "width": 120, "height": 100, "n_frames": n},
                              sam_pred, sam_pred)
    meta, loaded = cc.load_cache(cache_dir, which="filtered")
    assert meta["text"] == "white crystal"
    T, info = cr.track_and_count(loaded, meta["n_frames"],
                                 FakePropagator(n, object_masks, b2o),
                                 cr.PROFILES["baseline"])
    assert len(T) == 3
