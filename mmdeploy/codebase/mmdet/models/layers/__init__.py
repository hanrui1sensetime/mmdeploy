# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_nms import multiclass_nms
from . import matrix_nms  # noqa: F401, F403

__all__ = ['multiclass_nms']
