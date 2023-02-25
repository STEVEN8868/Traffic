import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


'''
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size=1*kernel_size, stride=stride))
        self.conv1 = nn.Conv2d(n_inputs, 25, kernel_size=1 * 2)

        # self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(25, n_outputs, kernel_size))
        # self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1)
        # self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        # self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        # out = self.conv2(out)
        # out = self.relu2(out)
        # out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        # return self.relu(out)
        return out
'''


class TemporalBlock(nn.Module):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            kernel_size,
            stride,
            dilation,
            padding,
            dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv2d(n_inputs, n_outputs, (3, kernel_size), stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (3, kernel_size),
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv2d(n_inputs, n_outputs, (2 * padding + 1, 1), padding=padding)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class GaussianBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, g_kernel, kernel_size, stride, dropout=2):
        super(GaussianBlock, self).__init__()
        #  filters of shape (out_channels, in_channels, kH, kW)
        gkernel = torch.FloatTensor(g_kernel).unsqueeze(2).unsqueeze(2)
        self.weight = nn.Parameter(data=gkernel, requires_grad=False)
        # self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size=1*kernel_size, stride=stride))
        self.conv1 = nn.Conv2d(n_inputs, n_outputs, kernel_size=1 * kernel_size, stride=stride)

        # self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
        #                                    stride=stride, padding=padding, dilation=dilation))
        # self.chomp2 = Chomp1d(padding)
        # self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(dropout)

        # self.net = nn.Sequential(self.conv1, self.relu1)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        # self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = F.conv2d(x, self.weight)
        # out = self.conv1(x)
        # out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        # res = x if self.downsample is None else self.downsample(x)
        # return self.relu(out)
        return out


# class TemporalConvNet(nn.Module):
#     def __init__(self, tmp_input_channels, tmp_seq_length, tmp_levels, g_kernel, kernel_size=2, dropout=0.2):
#         super(TemporalConvNet, self).__init__()
#         layers = []
#         # num_levels = len(tmp_levels)
#         for i in range(tmp_levels):
#             # dilation_size = 2 ** i
#             # in_channels = num_inputs if i == 0 else tmp_levels[i-1]
#             # out_channels = tmp_levels[i]
#             # out_channels = tmp_seq_length - ((i + 1) * (kernel_size -1))
#             if i == 0:
#                 layers += [
#                     GaussianBlock(tmp_input_channels, tmp_input_channels, g_kernel=g_kernel, kernel_size=kernel_size,
#                                   stride=kernel_size, dropout=dropout)]
#             else:
#                 layers += [TemporalBlock(tmp_input_channels, tmp_input_channels, 2, stride=1, dropout=dropout)]
#
#         self.network = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.network(x.permute(0, 3, 1, 2))

class TemporalConvNet(nn.Module):
    def __init__(
            self,
            num_inputs,
            num_channels,
            kernel_size=2,
            dropout=0.2
    ):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            # in_channels = 1
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def cal_linear_num(layer_num, num_timesteps_input):
    result = num_timesteps_input + 4 * (2 ** layer_num - 1)
    return result


def cal_channel_size(layers, timesteps_input):
    channel_size = []
    for i in range(layers - 1):
        channel_size.append(timesteps_input)
    channel_size.append(timesteps_input - 2)
    return channel_size


class TemporalNet(nn.Module):
    def __init__(self, input_channels, output_channels, g_kernel, kernel_size, dropout):
        super(TemporalNet, self).__init__()
        self.GaussConvNet = GaussianBlock(input_channels, input_channels, g_kernel, kernel_size, kernel_size, dropout)
        tcn_layer = 5
        channel_size = cal_channel_size(tcn_layer, 12)
        self.TemporalConvNet = TemporalConvNet(1, channel_size, 3)
        linear_num = cal_linear_num(tcn_layer, 12)
        self.linear = nn.Linear(linear_num, output_channels)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        g_out = self.GaussConvNet(x)
        out = self.TemporalConvNet(g_out)  # input should have dimension (N, C, L)
        X = self.linear(out)
        X = X.permute(0, 2, 1, 3)
        return X
