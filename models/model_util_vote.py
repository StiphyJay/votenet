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
                 num_vote_heading=4,
                 max_r=5.75,
                 max_z=1.6,
                 parse_r=F.relu,
                 parse_z=F.relu,
                 top_n_votes=3,
                 best_n_votes=768):
        
        self.num_vote_heading = num_vote_heading
        self.num_spatial_cls = self.num_vote_heading * 2

        self.max_r = max_r
        self.max_z = max_z
        self.max_theta = np.pi / self.num_vote_heading
        
        self.parse_r = parse_r
        self.parse_z = parse_z

        self.top_n_votes = top_n_votes
        self.best_n_votes = best_n_votes

        assert self.top_n_votes <= self.num_spatial_cls


class VoteConfigDistance(object):
    def __init__(self, 
                 num_vote_heading=4,
                 max_r=(2.0, 6.0),
                 max_z=(1.6,),
                 parse_r=F.relu,
                 parse_z=F.relu,
                 top_n_votes=3,
                 best_n_votes=768):

        self.num_vote_heading = num_vote_heading
        self.num_r = len(max_r)
        self.num_z = len(max_z)
        self.num_spatial_cls = self.num_r * self.num_z * 2 * self.num_vote_heading

        self.max_r = sorted(max_r)
        self.max_z = sorted(max_z)
        self.max_theta = np.pi / self.num_vote_heading
        
        self.parse_r = parse_r
        self.parse_z = parse_z

        self.top_n_votes = top_n_votes
        self.best_n_votes = best_n_votes

        assert self.top_n_votes <= self.num_spatial_cls