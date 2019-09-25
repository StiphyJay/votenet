import torch
import torch.nn.functional as F

import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

class VoteConfig(object):
    def __init__(self):
        self.num_heading_bin = 4
        self.num_spatial_cls = self.num_heading_bin * 2

        self.normalized_rz = True
        self.max_r = 5.75
        self.max_z = 1.6
        self.max_theta = np.pi / self.num_heading_bin
        
        self.parse_r = F.softplus
        self.parse_z = F.softplus

        self.top_n_votes = 3

        assert self.top_n_votes <= self.num_spatial_cls
