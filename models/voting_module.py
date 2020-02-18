# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Voting module: generate votes from XYZ and features of seed points.

Date: July, 2019
Author: Charles R. Qi and Or Litany
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class VotingModule(nn.Module):
    def __init__(self, vote_factor, seed_feature_dim):
        """ Votes generation from seed point features.

        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            vote_feature_dim: int
                number of channels of vote features
        """
        super().__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim # due to residual feature, in_dim has to be == out_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, (3+self.out_dim) * self.vote_factor, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)
        
    def forward(self, seed_xyz, seed_features):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
        """
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]
        num_vote = num_seed*self.vote_factor
        net = F.relu(self.bn1(self.conv1(seed_features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, (3+out_dim)*vote_factor, num_seed)
                
        net = net.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 3+self.out_dim)
        offset = net[:,:,:,0:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset
        vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)
        
        residual_features = net[:,:,:,3:] # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = seed_features.transpose(2,1).unsqueeze(2) + residual_features
        vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)
        vote_features = vote_features.transpose(2,1).contiguous()
        
        return vote_xyz, vote_features


class VotingModuleVar(nn.Module):
    def __init__(self, vote_factor, seed_feature_dim):
        """ Votes generation from seed point features.

        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            vote_feature_dim: int
                number of channels of vote features
        """
        super().__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim # due to residual feature, in_dim has to be == out_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, (6+self.out_dim) * self.vote_factor, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)
        
    def forward(self, seed_xyz, seed_features):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
            vote_var: (batch_size, num_seed*vote_factor, 3)
        """
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]
        num_vote = num_seed*self.vote_factor
        net = F.relu(self.bn1(self.conv1(seed_features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, (3+out_dim)*vote_factor, num_seed)
                
        net = net.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 6+self.out_dim)
        offset = net[:,:,:,0:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset
        vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)

        vote_var = net[:,:,:,3:6].contiguous().view(batch_size, num_vote, 3)

        residual_features = net[:,:,:,6:] # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = seed_features.transpose(2,1).unsqueeze(2) + residual_features
        vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)
        vote_features = vote_features.transpose(2,1).contiguous()
        
        return vote_xyz, vote_var, vote_features


class VotingModuleMulti(nn.Module):
    def __init__(self, config, seed_feature_dim):
        """ Votes generation from seed point features.

        Args:
            config: VoteConfig
                for testing different configs
            seed_feature_dim: int
                number of channels of seed point features
        """
        super().__init__()
        self.config = config
        self.num_vote_heading = self.config.num_vote_heading
        self.in_dim = seed_feature_dim

        start_theta = torch.arange(self.num_vote_heading, dtype=torch.float32) * 2 * self.config.max_theta
        self.register_buffer('start_theta', start_theta.repeat(2).view(1, 1, -1))
        
        # TODO: layer parameter
        self.out_vote_dim = 3 + self.in_dim + 1 # (r, z, theta) + (feature) + score
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, self.config.num_spatial_cls*self.out_vote_dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)
        
    def forward(self, seed_xyz, seed_features):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor

        Returns: Note that for the convenience of choosing top n vote, return shape is different
            vote_xyz: (batch_size, num_seed, num_spatial_cls, 3)
            vote_features: (batch_size, num_seed, num_spatial_cls, vote_feature_dim)
            vote_spatial_score: (batch_size, num_seed, num_spatial_cls)
        """
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]

        net = F.relu(self.bn1(self.conv1(seed_features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, num_spatial_cls*(3+out_dim+1), num_seed)
                
        net = net.transpose(2,1).view(batch_size, num_seed, -1, self.out_vote_dim)
        
        # parse offset
        offset_r = self.config.parse_r(net[:,:,:,0]) # (batch_size, num_seed, num_spatial_cls)
        offset_z = self.config.parse_z(net[:,:,:,1]) # (batch_size, num_seed, num_spatial_cls)
        
        offset_r *= self.config.max_r
        offset_z *= self.config.max_z
        
        offset_z[:,:,self.num_vote_heading:] *= -1.0
        
        res_theta = net[:,:,:,2] * self.config.max_theta
        offset_theta = res_theta + self.start_theta # (batch_size, num_seed, num_spatial_cls)

        offset_x = offset_r * torch.cos(offset_theta)
        offset_y = offset_r * torch.sin(offset_theta)
        offset_xyz = torch.cat((offset_x.unsqueeze(3), offset_y.unsqueeze(3), offset_z.unsqueeze(3)), dim=3) # (batch_size, num_seed, num_spatial_cls, 3)
        
        vote_xyz = seed_xyz.unsqueeze(2) + offset_xyz # (batch_size, num_seed, num_spatial_cls, 3)
        
        residual_features = net[:,:,:,3:-1] # (batch_size, num_seed, num_spatial_cls, out_dim)
        vote_features = seed_features.transpose(2,1).unsqueeze(2) + residual_features

        vote_spatial_score = net[:,:,:,-1] # (batch_size, num_seed, num_spatial_cls)
        
        return vote_xyz.contiguous(), vote_features.contiguous(), vote_spatial_score.contiguous()


class VotingModuleMultiDistance(nn.Module):
    def __init__(self, config, seed_feature_dim, no_feature_refine=False):
        """ Votes generation from seed point features.

        Args:
            config: VoteConfigDistance
                for testing different configs
            seed_feature_dim: int
                number of channels of seed point features
        """
        super().__init__()
        self.config = config
        self.num_vote_heading = self.config.num_vote_heading
        self.no_feature_refine = no_feature_refine
        self.in_dim = seed_feature_dim
        
        # buffer variable
        self.register_buffer('max_r', torch.tensor(self.config.max_r, dtype=torch.float32).view(1, -1, 1))
        self.register_buffer('max_z', torch.tensor(self.config.max_z, dtype=torch.float32).view(1, -1, 1))
        self.register_buffer('start_theta', torch.arange(self.num_vote_heading, dtype=torch.float32) * 2 * self.config.max_theta)
        
        # TODO: layer parameter
        self.out_vote_dim = 4 if self.no_feature_refine else 4 + self.in_dim # (r, z, theta) + (feature) + score
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, self.config.num_spatial_cls*self.out_vote_dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)
        
    def forward(self, seed_xyz, seed_features):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor

        Returns: Note that for the convenience of choosing top n vote, return shape is different
            vote_xyz: (batch_size, num_seed, num_spatial_cls, 3)
            vote_features: (batch_size, num_seed, num_spatial_cls, vote_feature_dim)
            vote_spatial_score: (batch_size, num_seed, num_spatial_cls)
        """
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]

        net = F.relu(self.bn1(self.conv1(seed_features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, num_spatial_cls*(3+out_dim+1), num_seed)
                
        net = net.transpose(2,1).contiguous().view(batch_size, num_seed, -1, self.out_vote_dim)

        # parse r
        offset_r = self.config.parse_r(net[..., 0].view(batch_size*num_seed, self.config.num_r, -1))
        offset_r *= self.max_r
        offset_r = offset_r.view(batch_size, num_seed, -1, 1)

        #parse z
        offset_z = self.config.parse_z(net[..., 1].view(-1, self.config.num_z, 2*self.num_vote_heading))
        offset_z *= self.max_z
        offset_z[..., self.num_vote_heading:] *= -1.0
        offset_z = offset_z.view(batch_size, num_seed, -1, 1)
        
        res_theta = net[..., 2].view(-1, self.num_vote_heading) * self.config.max_theta
        offset_theta = res_theta + self.start_theta
        offset_theta = offset_theta.view(batch_size, num_seed, -1, 1)

        offset_x = offset_r * torch.cos(offset_theta)
        offset_y = offset_r * torch.sin(offset_theta)
        offset_xyz = torch.cat((offset_x, offset_y, offset_z), dim=3) # (batch_size, num_seed, num_spatial_cls, 3)
        
        vote_xyz = seed_xyz.unsqueeze(2) + offset_xyz # (batch_size, num_seed, num_spatial_cls, 3)
        
        if self.no_feature_refine:
            vote_features = seed_features.transpose(2,1).unsqueeze(2) + \
                torch.zeros((batch_size, num_seed, self.config.num_spatial_cls, self.in_dim)).cuda(seed_features.device)
        else:
            residual_features = net[..., 3:-1]
            vote_features = seed_features.transpose(2,1).unsqueeze(2) + residual_features

        vote_spatial_score = net[..., -1]
        
        return vote_xyz.contiguous(), vote_features.contiguous(), vote_spatial_score.contiguous()


class VotingModuleDiscrete(nn.Module):
    def __init__(self, config, seed_feature_dim):
        """ Votes generation from seed point features.

        Args:
            config: VoteConfig_Discrete
                for testing different configs
            seed_feature_dim: int
                number of channels of seed point features
        """
        super().__init__()
        self.config = config
        self.num_spatial_cls = self.config.num_spatial_cls
        self.in_dim = seed_feature_dim
        
        # TODO: layer parameter
        self.out_vote_dim = self.in_dim + 1 # (feature) + score
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, self.num_spatial_cls*self.out_vote_dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)
        
    def forward(self, seed_xyz, seed_features):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor

        Returns: Note that for the convenience of choosing top n vote, return shape is different
            vote_xyz: (batch_size, num_seed, num_spatial_cls, 3)
            vote_features: (batch_size, num_seed, num_spatial_cls, vote_feature_dim)
            vote_spatial_score: (batch_size, num_seed, num_spatial_cls)
        """
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]

        net = F.relu(self.bn1(self.conv1(seed_features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, num_spatial_cls*(3+out_dim+1), num_seed)
                
        net = net.transpose(2,1).view(batch_size, num_seed, self.num_spatial_cls, self.out_vote_dim)

        fixed_votes = self.config.fixed_votes.clone().to(seed_xyz.device) # (num_spatial_cls, 3)
        vote_xyz = seed_xyz.unsqueeze(2) + fixed_votes.view(1, 1, -1, 3)

        residual_features = net[:,:,:,:-1] # (batch_size, num_seed, num_spatial_cls, out_dim)
        vote_features = seed_features.transpose(2,1).unsqueeze(2) + residual_features

        vote_spatial_score = net[:,:,:,-1] # (batch_size, num_seed, num_spatial_cls)
       
        return vote_xyz.contiguous(), vote_features.contiguous(), vote_spatial_score.contiguous()

# testing
if __name__=='__main__':
    net = VotingModule(2, 256).cuda()
    xyz, features = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda())
    print('xyz', xyz.shape)
    print('features', features.shape)

    from model_util_vote import VoteConfig
    net = VotingModuleMulti(VoteConfig(), 256).cuda()
    xyz, features, score = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda())
    print()
    print('xyz_multi', xyz.size())
    print('features_multi', features.size())
    print('score', score.size())