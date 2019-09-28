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
    def __init__(self, 
                 num_heading_bin=4,
                 normalized_rz=True,
                 max_r=5.75,
                 max_z=1.6,
                 parse_r=F.softplus,
                 parse_z=F.softplus,
                 top_n_votes=3):
        
        self.num_heading_bin = num_heading_bin
        self.num_spatial_cls = self.num_heading_bin * 2

        self.normalized_rz = normalized_rz
        self.max_r = max_r
        self.max_z = max_z
        self.max_theta = np.pi / self.num_heading_bin
        
        self.parse_r = parse_r
        self.parse_z = parse_z

        self.top_n_votes = top_n_votes

        assert self.top_n_votes <= self.num_spatial_cls


class VoteConfig_Discrete(object):
    def __init__(self, 
                 num_heading_bin=4,
                 cut_points_r=(0.4, 1.2, 3.6),
                 cut_points_z=(-1.0, 0.0, 1.0),
                 fixed_votes_r=(0.2, 0.8, 2.4, 5.0),
                 fixed_votes_z=(-1.5, -0.5, 0.5, 1.5), 
                 top_n_votes=3):
        
        assert len(cut_points_r)+1 == len(fixed_votes_r)
        assert len(cut_points_z)+1 == len(fixed_votes_z)
        
        self.num_heading_bin = num_heading_bin
        self.num_spatial_cls = self.num_heading_bin * len(fixed_votes_r) * len(fixed_votes_z)

        # TODO: compute fixed votes in xyz
        self.fixed_votes = None # (num_spatial_cls, 3)
        assert len(self.fixed_votes) == self.num_spatial_cls

        self.top_n_votes = top_n_votes

        assert self.top_n_votes <= self.num_spatial_cls
