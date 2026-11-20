"""Quality filtering for pseudo-exemplar prompts only; no temporal memory."""
import torch
from torchvision.ops import nms


def get_quality_pseudo_exemplars(model_output, image_size, box_threshold,
                                num_exemplars=10, *, min_confidence=0.30,
                                min_side=2.0, nms_iou=0.80):
    if not 0 <= min_confidence <= 1 or not 0 <= nms_iou <= 1 or min_side < 0:
        raise ValueError("Invalid quality filtering thresholds")
    if num_exemplars < 0:
        raise ValueError("num_exemplars must be nonnegative")
    logits = model_output["pred_logits"].sigmoid()[0]
    boxes = model_output["pred_boxes"][0]
    scores = logits.max(dim=-1).values
    valid = (torch.isfinite(boxes).all(dim=-1) & torch.isfinite(scores)
             & (scores > max(box_threshold, min_confidence)))
    boxes, scores = boxes[valid], scores[valid]
    height, width = image_size
    cx, cy = boxes[:, 0] * width, boxes[:, 1] * height
    bw, bh = boxes[:, 2] * width, boxes[:, 3] * height
    xyxy = torch.stack(((cx-bw/2).clamp(0, width),
                        (cy-bh/2).clamp(0, height),
                        (cx+bw/2).clamp(0, width),
                        (cy+bh/2).clamp(0, height)), dim=-1)
    # Sizes are measured on the transformed model-input image, not the original frame.
    valid_size = ((xyxy[:, 2]-xyxy[:, 0] >= min_side)
                  & (xyxy[:, 3]-xyxy[:, 1] >= min_side)
                  & (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1]))
    xyxy, scores = xyxy[valid_size], scores[valid_size]
    # An empty result stays empty: do not refill with rejected prompts.
    keep = nms(xyxy, scores, nms_iou)[:num_exemplars]
    selected = xyxy[keep]
    print(f"[quality] candidates={len(model_output['pred_boxes'][0])} "
          f"confidence_valid={len(boxes)} size_valid={len(xyxy)} "
          f"selected={len(selected)}", flush=True)
    return selected
