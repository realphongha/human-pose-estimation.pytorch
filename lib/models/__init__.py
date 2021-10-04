# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lib.models.pose_resnet
import lib.models.pose_hrnet
import lib.models.pose_hrnet_psa
import lib.models.pose_resnet_psa

from lib.models.pose_resnet import get_pose_net as pr
from lib.models.pose_hrnet import get_pose_net as ph
from lib.models.pose_hrnet_psa import get_pose_net as php
from lib.models.pose_resnet_psa import get_pose_net as prp

MODELS = {
    "pose_resnet": pr,
    "pose_hrnet": ph,
    "pose_hrnet_psa": php,
    "pose_resnet_psa": prp,
}