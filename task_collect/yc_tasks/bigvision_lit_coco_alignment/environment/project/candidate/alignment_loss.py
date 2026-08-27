"""Starter objective: the pinned historical bidirectional contrastive loss."""

from big_vision import utils as _utils

_ORIGINAL_LOSS = _utils.bidirectional_contrastive_loss


def bidirectional_contrastive_loss(zimg, ztxt, t, mask=None, reduction=False):
    return _ORIGINAL_LOSS(zimg, ztxt, t, mask=mask, reduction=reduction)


__all__ = ["bidirectional_contrastive_loss"]

