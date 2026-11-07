"""Untrained research prototype: one encoder, ratio/mode-conditioned crop scores.

Reuse the owned w24 encoder. No saliency decoder or heatmap objective. Raw
scores rank alternative compositions; legal candidates are supplied by the
independent face-preserving policy. This is not an exported Rust model.
"""
import torch
from torch import nn
from torch.nn import functional as F
from model_v3ds import MicroSalNetV3ds


class CropRanker(nn.Module):
    def __init__(self, width=24, hidden=96):
        super().__init__()
        base = MicroSalNetV3ds(width=width)
        self.stem, self.enc1, self.enc2 = base.stem, base.enc1, base.enc2
        self.enc3, self.bottleneck = base.enc3, base.bottleneck
        channels = width * 3
        # Nine inside samples, four outside-band samples, one global context;
        # eight geometry/task values and six supplied face-context values.
        self.scorer = nn.Sequential(nn.Linear(channels * 14 + 14, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        yy, xx = torch.meshgrid((torch.arange(3) + .5) / 3, (torch.arange(3) + .5) / 3, indexing="ij")
        self.register_buffer("inside_grid", torch.stack((xx.flatten(), yy.flatten()), -1))

    def encode(self, image):
        return self.bottleneck(self.enc3(self.enc2(self.enc1(self.stem(image)))))

    def score(self, features, boxes, source_aspect, target_aspect, mode, face_context, legal, valid_rect):
        """Boxes/valid_rect: normalized source/canvas xyxy; source aspect: B.

        Target aspect/mode/legal are BxN, face_context is Bx6, boxes BxNx4.
        valid_rect (Bx4) locates aspect-fitted content in the encoded canvas.
        Face context: count, union xyxy, coverage-reviewed flag. It conditions
        composition but cannot make an illegal candidate legal.
        """
        batch, count, _ = boxes.shape
        lo, hi = boxes[..., :2], boxes[..., 2:]
        inside = lo[:, :, None] + (hi - lo)[:, :, None] * self.inside_grid
        x0, y0, x1, y1 = boxes.unbind(-1)
        # Midpoints of the disjoint discarded left/right/top/bottom bands.
        outside = torch.stack((torch.stack((x0 / 2, torch.full_like(y0, .5)), -1),
                               torch.stack(((1 + x1) / 2, torch.full_like(y0, .5)), -1),
                               torch.stack(((x0 + x1) / 2, y0 / 2), -1),
                               torch.stack(((x0 + x1) / 2, (1 + y1) / 2), -1)), -2)
        points = torch.cat((inside, outside), -2)
        vlo, vsize = valid_rect[:, :2], valid_rect[:, 2:] - valid_rect[:, :2]
        points = vlo[:, None, None] + points * vsize[:, None, None]
        grid = points.reshape(batch, count * 13, 1, 2) * 2 - 1
        sampled = F.grid_sample(features, grid, mode="bilinear", padding_mode="border", align_corners=False)
        sampled = sampled[..., 0].reshape(batch, features.shape[1], count, 13).permute(0, 2, 3, 1)
        outside_exists = torch.stack((x0 > 0, x1 < 1, y0 > 0, y1 < 1), -1)
        samples = torch.cat((sampled[:, :, :9], sampled[:, :, 9:] * outside_exists[..., None]), -2)
        # Fractional feature-cell coverage avoids dropping an extremely thin
        # letterboxed source whose content has no feature-cell center inside it.
        fh, fw = features.shape[-2:]
        gx = torch.arange(fw, device=features.device, dtype=features.dtype) / fw
        gy = torch.arange(fh, device=features.device, dtype=features.dtype) / fh
        wx = (torch.minimum(gx[None] + 1 / fw, valid_rect[:, 2:3]) - torch.maximum(gx[None], valid_rect[:, 0:1])).clamp_min(0)
        wy = (torch.minimum(gy[None] + 1 / fh, valid_rect[:, 3:4]) - torch.maximum(gy[None], valid_rect[:, 1:2])).clamp_min(0)
        weights = wy[:, :, None] * wx[:, None]
        global_context = (features * weights[:, None]).sum((-2, -1)) / weights.sum((-2, -1))[:, None].clamp_min(1e-12)
        global_context = global_context[:, None].expand(-1, count, -1)
        geom = torch.stack((x0, y0, x1, y1, (x1 - x0) * (y1 - y0),
                            source_aspect[:, None].expand(-1, count).log(), target_aspect.log(), mode), -1)
        face = face_context[:, None].expand(-1, count, -1)
        vector = torch.cat((samples.flatten(-2), global_context, geom, face), -1)
        scores = self.scorer(vector)[..., 0]
        # Recheck the protected union in the scorer interface as well. These
        # comparisons and the review flag have no learned weights.
        union = face_context[:, None, 1:5]
        contains_faces = (boxes[..., :2] <= union[..., :2]).all(-1) & (boxes[..., 2:] >= union[..., 2:]).all(-1)
        contains_faces = contains_faces | (face_context[:, None, 0] == 0)
        reviewed = face_context[:, None, 5] == 1
        valid_boxes = torch.isfinite(boxes).all(-1) & (boxes[..., :2] >= 0).all(-1) & (boxes[..., 2:] <= 1).all(-1) & (boxes[..., 2:] > boxes[..., :2]).all(-1)
        return scores.masked_fill(~(legal & reviewed & contains_faces & valid_boxes), -torch.inf)

    def forward(self, image, boxes, source_aspect, target_aspect, mode, face_context, legal, valid_rect):
        return self.score(self.encode(image), boxes, source_aspect, target_aspect, mode, face_context, legal, valid_rect)


def preference_loss(scores, pairs, weights, task_ids):
    """Bradley-Terry preference loss; pairs are (batch, preferred, other).

    Labels must compare the same source, requested aspect and mode. Ties and
    rejected/uncertain records require their own admission, not invented winners.
    """
    b, preferred, other = pairs.unbind(-1)
    if not bool((task_ids[b, preferred] == task_ids[b, other]).all()):
        raise ValueError("preferences must compare the same aspect/mode task")
    delta = scores[b, preferred] - scores[b, other]
    if weights.shape != delta.shape or not bool(torch.isfinite(delta).all()) or not bool(torch.isfinite(weights).all()) or not bool((weights >= 0).all()) or not bool((weights.sum() > 0)):
        raise ValueError("finite legal pairs and positive total label weight required")
    return (F.softplus(-delta) * weights).sum() / weights.sum()
