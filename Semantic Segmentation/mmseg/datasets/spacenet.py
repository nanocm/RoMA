# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SPACENETV1Dataset(CustomDataset):
    """SPACENETV1 dataset.

    In segmentation map annotation for SPACENETV1, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` and ``seg_map_suffix`` are both fixed to '.png'.
    """

    CLASSES = ('background', 'building')

    PALETTE = [[255, 255, 255], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(SPACENETV1Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
