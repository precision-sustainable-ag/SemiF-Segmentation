"""torchvision RoIHeads with the mask branch replaced by PointRend.

The box branch is torchvision's, unchanged (forward below mirrors
torchvision.models.detection.roi_heads.RoIHeads.forward up to the mask
branch). The parent's own mask_roi_pool/mask_head/mask_predictor are left
None, so has_mask() is False and it never runs its mask branch; the PointRend
head runs instead, with access to the whole FPN feature dict -- which it
needs for the fine-grained point features.

Outputs match torchvision's: result["masks"] holds (N, 1, M, M) per-box
probabilities, which GeneralizedRCNNTransform.postprocess pastes into
(N, 1, H, W) image-sized masks.
"""

from __future__ import annotations

from typing import Optional

import torch
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss

from src.models.pointrend.mask_head import PointRendMaskHead


class PointRendRoIHeads(RoIHeads):
    def __init__(self, box_roi_pool, box_head, box_predictor, point_mask_head: PointRendMaskHead, **kwargs):
        """kwargs: RoIHeads' box settings (fg_iou_thresh, ..., detections_per_img)."""
        super().__init__(box_roi_pool, box_head, box_predictor, **kwargs)
        self.point_mask_head = point_mask_head

    @classmethod
    def from_roi_heads(cls, roi_heads: RoIHeads, point_mask_head: PointRendMaskHead) -> "PointRendRoIHeads":
        """Keep `roi_heads`' box branch (modules and settings), swap in PointRend for the mask branch."""
        return cls(
            roi_heads.box_roi_pool,
            roi_heads.box_head,
            roi_heads.box_predictor,
            point_mask_head,
            fg_iou_thresh=roi_heads.proposal_matcher.high_threshold,
            bg_iou_thresh=roi_heads.proposal_matcher.low_threshold,
            batch_size_per_image=roi_heads.fg_bg_sampler.batch_size_per_image,
            positive_fraction=roi_heads.fg_bg_sampler.positive_fraction,
            bbox_reg_weights=roi_heads.box_coder.weights,
            score_thresh=roi_heads.score_thresh,
            nms_thresh=roi_heads.nms_thresh,
            detections_per_img=roi_heads.detections_per_img,
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        targets: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        if targets is not None:
            for t in targets:
                if t["boxes"].dtype not in (torch.float, torch.double, torch.half):
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")

        # --- box branch (torchvision) ---
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = regression_targets = matched_idxs = None

        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(box_features)

        result: list[dict[str, torch.Tensor]] = []
        losses: dict[str, torch.Tensor] = {}
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            result = [{"boxes": b, "labels": l, "scores": s} for b, l, s in zip(boxes, labels, scores)]

        # --- mask branch (PointRend) ---
        agnostic = self.point_mask_head.class_agnostic
        if self.training:
            # only positive proposals, as in torchvision
            mask_proposals, pos_matched_idxs = [], []
            for img_proposals, img_labels, img_matched in zip(proposals, labels, matched_idxs):
                pos = torch.where(img_labels > 0)[0]
                mask_proposals.append(img_proposals[pos])
                pos_matched_idxs.append(img_matched[pos])
            mask_classes = None if agnostic else torch.cat(
                [t["labels"][idx] for t, idx in zip(targets, pos_matched_idxs)]
            )
            losses.update(self.point_mask_head.losses(
                features, mask_proposals, image_shapes,
                [t["masks"] for t in targets], pos_matched_idxs, mask_classes,
            ))
        else:
            boxes = [r["boxes"] for r in result]
            mask_classes = None if agnostic else torch.cat([r["labels"] for r in result])
            mask_logits = self.point_mask_head.inference(features, boxes, image_shapes, mask_classes)
            for r, probs in zip(result, mask_logits.sigmoid().split([len(b) for b in boxes])):
                r["masks"] = probs

        return result, losses
