# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.models.losses import *

from .dist_loss import (TokenDistilLoss, ScoreLoss, Reg2HMLoss)
from .foreground_distil_loss import (ForegroundTokenDistilLoss, 
                                     AdaptiveForegroundDistilLoss,
                                     DynamicForegroundDistilLoss)
from .local_global_loss import (LocalGlobalConsistencyLoss,
                                ProgressiveLocalGlobalLoss,
                                SimpleConsistencyLoss)
from .entropy_weighted_kd import (EntropyWeightedKDLoss,
                                  EntropyWeightedTokenKDLoss)
from .hd_distill_loss import (HardAwareDistillLoss,
                              DifficultyRegressionLoss)

__all__ = [
    'JointsMSELoss', 'JointsOHKMMSELoss', 'HeatmapLoss', 'AELoss',
    'MultiLossFactory', 'MeshLoss', 'GANLoss', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss', 
    'TokenDistilLoss', 'ScoreLoss', 'Reg2HMLoss',
    'ForegroundTokenDistilLoss', 'AdaptiveForegroundDistilLoss', 
    'DynamicForegroundDistilLoss',
    'LocalGlobalConsistencyLoss', 'ProgressiveLocalGlobalLoss',
    'SimpleConsistencyLoss', 'EntropyWeightedKDLoss',
    'EntropyWeightedTokenKDLoss',
    'HardAwareDistillLoss', 'DifficultyRegressionLoss'
]
