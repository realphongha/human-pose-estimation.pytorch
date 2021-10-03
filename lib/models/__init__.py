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

MODELS = {
    "pose_resnet": pose_resnet.get_pose_net,
    "pose_hrnet": pose_hrnet.get_pose_net,
    "pose_hrnet_psa": pose_hrnet_psa.get_pose_net,
    "pose_resnet_psa": pose_resnet_psa.get_pose_net,
}