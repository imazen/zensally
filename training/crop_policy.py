"""Research-only crop candidates; learned scores cannot override face constraints.

Coordinates are continuous source-pixel boundaries, before rasterization.
An eventual renderer must preserve the protected regions in its actual sample
footprint too. This is not a new shipping crop API or a face detector.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Rect:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    @property
    def area(self):
        return self.width * self.height

    def valid(self):
        return all(math.isfinite(v) for v in (self.left, self.top, self.right, self.bottom)) and self.width > 0 and self.height > 0

    def contains(self, other):
        return self.left <= other.left and self.top <= other.top and self.right >= other.right and self.bottom >= other.bottom


@dataclass(frozen=True)
class CropPlan:
    kind: str
    mode: str
    source: Rect
    target_ratio: tuple[int, int] | None
    protected_faces: tuple[Rect, ...]
    candidates: tuple[Rect, ...]
    reason: str


def plan_crops(width, height, target_ratio=None, *, mode="minimal", protected_faces=(), faces_reviewed=False, anchors=5):
    """Minimal only changes framing when a ratio requires it; unknown faces fail closed.

    `faces_reviewed` is a trusted caller assertion about coverage, not a detector
    confidence threshold. Empty detector output alone does not establish it.
    All supplied faces matter equally. Infeasible requests use contain/padding.
    """
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("positive integer source dimensions required")
    if mode not in ("minimal", "maximal") or not isinstance(anchors, int) or anchors < 2:
        raise ValueError("invalid crop mode or anchor count")
    source = Rect(0., 0., float(width), float(height))
    faces = tuple(protected_faces)
    if any(not f.valid() or not source.contains(f) for f in faces):
        raise ValueError("protected faces must be valid visible source-space bounds")
    ratio = target_ratio or (width, height)
    if len(ratio) != 2 or any(not isinstance(v, int) or v <= 0 for v in ratio):
        raise ValueError("positive integer target ratio required")
    same_ratio = width * ratio[1] == height * ratio[0]
    if mode == "minimal" and (target_ratio is None or same_ratio):
        return CropPlan("unchanged", mode, source, target_ratio, faces, (source,), "ratio does not require cropping")
    if not faces_reviewed:
        return CropPlan("contain", mode, source, target_ratio, faces, (), "face coverage unverified")
    if width * ratio[1] <= height * ratio[0]:
        base_w, base_h = float(width), width * ratio[1] / ratio[0]
    else:
        base_w, base_h = height * ratio[0] / ratio[1], float(height)
    scales = (1.,) if mode == "minimal" else (1., .875, .75, .625, .5)
    candidates = []
    seen = set()
    for scale in scales:
        cw, ch = base_w * scale, base_h * scale
        lo_x = max([0.] + [f.right - cw for f in faces])
        hi_x = min([width - cw] + [f.left for f in faces])
        lo_y = max([0.] + [f.bottom - ch for f in faces])
        hi_y = min([height - ch] + [f.top for f in faces])
        if lo_x > hi_x or lo_y > hi_y:
            continue
        for iy in range(anchors):
            y = lo_y + (hi_y - lo_y) * iy / (anchors - 1)
            for ix in range(anchors):
                x = lo_x + (hi_x - lo_x) * ix / (anchors - 1)
                r = Rect(x, y, min(float(width), x + cw), min(float(height), y + ch))
                # No tolerance that could permit clipping even a tiny face edge.
                if r.valid() and source.contains(r) and all(r.contains(f) for f in faces) and r not in seen:
                    candidates.append(r)
                    seen.add(r)
    return CropPlan("candidates" if candidates else "contain", mode, source, target_ratio, faces, tuple(candidates), "rank legal compositions" if candidates else "target cannot contain all protected faces")


def choose_crop(plan, scores):
    """Final guard is independent of learned score, including NaNs/all rejected.

    Returns (kind, source-space rectangle). `contain` always retains the full
    source and requires an output canvas rather than distorting the image.
    """
    if plan.kind == "unchanged":
        return "unchanged", plan.source
    if len(scores) != len(plan.candidates):
        raise ValueError("one score per candidate required")
    legal = [(float(s), r.area, -i, r) for i, (r, s) in enumerate(zip(plan.candidates, scores)) if math.isfinite(float(s)) and r.valid() and plan.source.contains(r) and all(r.contains(f) for f in plan.protected_faces)]
    if plan.kind != "candidates" or not legal:
        return "contain", plan.source
    return "crop", max(legal, key=lambda v: v[:3])[3]
