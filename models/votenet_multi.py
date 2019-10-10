# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_module import Pointnet2Backbone
from voting_module import VotingModuleMulti
from model_util_vote import VoteConfig
from proposal_module import ProposalModuleMulti
from dump_helper_multi import dump_results
from loss_helper_multi import get_loss


class VoteNetMulti(nn.Module):
    r"""
        Based on origin code by Charles et al., support real multi-vote.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_config: Config instance for supporting multi-voting
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, vote_config=VoteConfig(), 
                 sampling='vote_fps'):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_config = vote_config
        self.sampling=sampling

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModuleMulti(self.vote_config, 256)

        # Vote aggregation and detection  # TODO: if to change num_proposal
        self.pnet = ProposalModuleMulti(num_class, num_heading_bin, num_size_cluster,
                                        mean_size_arr, num_proposal, sampling, 
                                        vote_config=self.vote_config)

    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]

        end_points = self.backbone_net(inputs['point_clouds'], end_points)
                
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        
        xyz, features, spatial_score = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_spatial_score'] = spatial_score # (batch_size, num_seed, num_spatial_cls)

        # choose top_n_votes
        num_vote = self.vote_config.top_n_votes
        top_n_spatial_score, top_n_spatial_score_ind = torch.topk(spatial_score, num_vote, dim=2) # (batch_size, num_seed, num_vote)
        top_n_spatial_score_ind_expand = top_n_spatial_score_ind.unsqueeze(-1).repeat(1,1,1,3)
        xyz_top_n = torch.gather(xyz, 2, top_n_spatial_score_ind_expand) # (batch_size, num_seed, num_vote, 3)
        vote_feature_dim = features.size(-1)
        top_n_spatial_score_ind_expand = top_n_spatial_score_ind.unsqueeze(-1).repeat(1,1,1,vote_feature_dim)
        features_top_n = torch.gather(features, 2, top_n_spatial_score_ind_expand) # (batch_size, num_seed, num_vote, vote_feature_dim)

        xyz_top_n_reshape = xyz_top_n.view(batch_size, -1, 3).contiguous() # (batch_size, num_seed*num_vote, 3)
        features_top_n_reshape = features_top_n.view(batch_size, -1, vote_feature_dim).transpose(1, 2).contiguous() # (batch_size, vote_feature_dim, num_seed*num_vote)

        end_points['vote_xyz'] = xyz_top_n_reshape # (batch_size, num_seed*num_vote, 3)
        end_points['vote_features'] = features_top_n_reshape # (batch_size, vote_feature_dim, num_seed*num_vote)
        end_points['vote_spatial_score_top_n'] = top_n_spatial_score.view(batch_size, -1).contiguous() # (batch_size, num_seed*num_vote)

        end_points = self.pnet(xyz_top_n_reshape, features_top_n_reshape, end_points)

        return end_points


if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    from loss_helper import get_loss

    # Define model
    model_multi = VoteNetMulti(10,12,10,np.random.random((10,3)), num_proposal=256*3).cuda()
    
    try:
        # Define dataset
        TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=20000, use_v1=True)

        # prepare input
        sample = TRAIN_DATASET[5]
        inputs = {'point_clouds': torch.from_numpy(sample['point_clouds']).unsqueeze(0).cuda()}
    except:
        print('Dataset has not been prepared. Use a random sample.')
        inputs = {'point_clouds': torch.rand((20000,3)).unsqueeze(0).cuda()}

    end_points_multi = model_multi(inputs)
    
    print()
    for key in end_points_multi:
        print(key, end_points_multi[key])
    '''
    try:
        # Compute loss
        for key in sample:
            end_points[key] = torch.from_numpy(sample[key]).unsqueeze(0).cuda()
        loss, end_points = get_loss(end_points, DC)
        print('loss', loss)
        end_points['point_clouds'] = inputs['point_clouds']
        end_points['pred_mask'] = np.ones((1,128))
        dump_results(end_points, 'tmp', DC)
    except:
        print('Dataset has not been prepared. Skip loss and dump.')
    '''
    
    # TODO: get_loss_multi
