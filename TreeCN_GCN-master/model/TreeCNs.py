import math
import torch
import torch.nn as nn
from model.TreeGradient import TreeGradient
from model.temporal import TemporalNet


# class TemporalNet(nn.Module):
#     def __init__(self, input_channels, output_channels, tmp_seq_length, tmp_levels, g_kernel, kernel_size, dropout):
#         super(TemporalNet, self).__init__()
#         self.TemporalConvNet = TemporalConvNet(input_channels, tmp_seq_length, tmp_levels, g_kernel=g_kernel,
#                                                kernel_size=kernel_size, dropout=dropout)
#         # self.linear = nn.Linear(num_channels[-1], output_size)
#
#         self.linear = nn.Linear(774, 10)
#
#     def forward(self, x):
#         """Inputs have to have dimension (N, C_in, L_in)"""
#         out = self.TemporalConvNet(x)  # input should have dimension (N, C, L)
#         return out


class TreeCNs(nn.Module):
    def __init__(
            self,
            num_nodes,
            spatial_channels,
            timesteps_output,
            max_node_number,
            g_kernel
    ):
        super(TreeCNs, self).__init__()
        self.spatial_channels = spatial_channels
        self.Theta1 = nn.Parameter(torch.FloatTensor(1, spatial_channels))
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.fully = nn.Linear(timesteps_output * (12 - 2), timesteps_output)
        self.fully_tmp = nn.Linear(16, 1)
        self.TreeGradient = TreeGradient(num_nodes=num_nodes, max_node_number=max_node_number)
        self.reset_parameters()
        self.TemporalNet = TemporalNet(input_channels=16, output_channels=3,
                                       g_kernel=g_kernel, kernel_size=2, dropout=0.05)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, NATree, X):
        tree_gradient = self.TreeGradient(NATree)
        lfs = torch.einsum("ij,jklm->kilm", [tree_gradient, X.permute(1, 0, 2, 3)])

        t2 = torch.tanh(torch.matmul(lfs, self.Theta1))
        # t2 = torch.matmul(lfs, self.Theta1)
        # t2 = self.fully_tmp(t2)
        t3 = self.TemporalNet(t2)
        out3 = self.batch_norm(t3)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))

        return out4
