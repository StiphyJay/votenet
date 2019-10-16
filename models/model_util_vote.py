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


class VoteConfig_Discrete_Polar(object):
    def __init__(self, 
                 num_vote_heading=8,
                 fixed_votes_r=(0.25, 0.75, 1.5, 3.0),
                 fixed_votes_z=(-0.75, -0.25, 0.25, 0.75), 
                 top_n_votes=3,
                 best_n_votes=768):
        
        self.num_r_vote = len(fixed_votes_r)
        self.num_z_vote = len(fixed_votes_z)
        
        
        self.num_vote_heading = num_vote_heading
        self.num_spatial_cls = self.num_vote_heading * self.num_r_vote * self.num_z_vote

        # compute fixed votes in xyz
        start_theta = torch.arange(self.num_vote_heading, dtype=torch.float32) * (2 * np.pi) / self.num_vote_heading
        fixed_votes_r = torch.tensor(fixed_votes_r).view(1, -1)
        fixed_votes_x = torch.cos(start_theta).view(-1, 1) * fixed_votes_r
        fixed_votes_y = torch.sin(start_theta).view(-1, 1) * fixed_votes_r
        fixed_votes_z = torch.tensor(fixed_votes_z).view(1, 1, -1, 1).repeat(self.num_vote_heading, self.num_r_vote, 1, 1)
        self.fixed_votes = torch.cat((fixed_votes_x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.num_z_vote, 1), 
                                      fixed_votes_y.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.num_z_vote, 1),
                                      fixed_votes_z), dim=3) # (num_vote_heading, num_r_vote, num_z_vote, 3)
        self.fixed_votes = self.fixed_votes.view(-1, 3)

        self.top_n_votes = top_n_votes
        self.best_n_votes = best_n_votes

        assert self.top_n_votes <= self.num_spatial_cls


class VoteConfig_Discrete_Grid(object):
    def __init__(self,
                 fixed_votes_x=(-1.5, -0.75, -0.2, 0.2, 0.75, 1.5),
                 fixed_votes_y=(-1.5, -0.75, -0.2, 0.2, 0.75, 1.5),
                 fixed_votes_z=(-0.75, -0.25, 0.25, 0.75), 
                 top_n_votes=3,
                 best_n_votes=768):

        self.num_x_vote = len(fixed_votes_x)
        self.num_y_vote = len(fixed_votes_y)
        self.num_z_vote = len(fixed_votes_z)

        self.num_spatial_cls = self.num_x_vote * self.num_y_vote * self.num_z_vote

        # compute fixed votes in xyz
        fixed_votes_x = torch.tensor(fixed_votes_x).view(-1, 1, 1, 1).repeat(1, self.num_y_vote, self.num_z_vote, 1)
        fixed_votes_y = torch.tensor(fixed_votes_y).view(1, -1, 1, 1).repeat(self.num_x_vote, 1, self.num_z_vote, 1)
        fixed_votes_z = torch.tensor(fixed_votes_z).view(1, 1, -1, 1).repeat(self.num_x_vote, self.num_y_vote, 1, 1)
        self.fixed_votes = torch.cat((fixed_votes_x, fixed_votes_y, fixed_votes_z), dim=3) # (num_x_vote, num_y_vote, num_z_vote, 3)
        self.fixed_votes = self.fixed_votes.view(-1, 3)

        self.top_n_votes = top_n_votes
        self.best_n_votes = best_n_votes

        assert self.top_n_votes <= self.num_spatial_cls

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    vc1 = VoteConfig_Discrete_Polar()

    fig1 = plt.figure(figsize=(10, 10))
    ax1 = Axes3D(fig1)
    ax1.scatter3D(vc1.fixed_votes[:, 0], vc1.fixed_votes[:, 1], vc1.fixed_votes[:, 2], s=5, depthshade= False)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_zlim(-1.5, 1.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    vc2 = VoteConfig_Discrete_Grid()

    fig2 = plt.figure(figsize=(10, 10))
    ax2 = Axes3D(fig2)
    ax2.scatter3D(vc2.fixed_votes[:, 0], vc2.fixed_votes[:, 1], vc2.fixed_votes[:, 2], s=5, depthshade= False)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_zlim(-1.5, 1.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    plt.show()