"""Instance-proposal generators (SAM 3) as optional teachers for the instance
models: proposals, filtering, pseudo-labels. Importing this package does not
import SAM 3 itself."""

from src.models.proposals.base import ProposalGenerator
from src.models.proposals.filtering import FilterConfig, apply_filters, mask_iou, resolve_overlaps
from src.models.proposals.pseudo_labels import PseudoLabelConfig, create_pseudo_target
from src.models.proposals.sam3 import SAM3ProposalGenerator, load_sam3
from src.models.proposals.structures import InstanceProposal, ProposalResult, normalize_output

__all__ = [
    "FilterConfig",
    "InstanceProposal",
    "ProposalGenerator",
    "ProposalResult",
    "PseudoLabelConfig",
    "SAM3ProposalGenerator",
    "apply_filters",
    "create_pseudo_target",
    "load_sam3",
    "mask_iou",
    "normalize_output",
    "resolve_overlaps",
]
