import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import opt


class MyPool(nn.Module):
    def __init__(self, in_dim=2, out_dim=1):
        super(MyPool, self).__init__()

        self.data = {}
        # self.gru = nn.GRU(in_, d_hidden, 1, batch_first=True)
        self.linear = nn.Linear(in_dim, 1, bias=False)
        self.linear_i = nn.Linear(in_dim, out_dim)
        # self.relu_i = nn.ReLU()
        # self.linear2_i = nn.Linear(hide_dim, 1)

        # 文本
        self.linear_t = nn.Linear(in_dim, out_dim)
        # self.relu_t = nn.ReLU()
        # self.linear2_t = nn.Linear(hide_dim, 1)
        # 标准化
        # self.batch_norm = nn.BatchNorm1d(2)
        # self.batch_norm_ = nn.BatchNorm1d(2)

        # self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.device = opt.device

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        B, K, D = features.size()

        img_region = features.size(1)  # m
        a = torch.arange(img_region, dtype=torch.float32).unsqueeze(1).to(self.device)
        b = a.flip(dims=[0])
        cat = torch.cat((a, b), dim=1)
        # 标准化
        # cat = self.batch_norm(cat)
        theta = self.linear_i(cat)  # img_region*hide size
        # theta = self.relu_i(theta)
        # theta = self.linear2_i(theta)  # hide size*1
        theta = nn.functional.softmax(theta, dim=0)

        # pool_weights, mask = pool_weights(length, feature)

        features = features[:, :int(lengths.max()), :]
        # torch_abs = torch.abs(features)
        with torch.no_grad():
            for i in range(B):
                features[i] = sort_matrix_by_abs_values(features[i])
            # sorted_features = features.sort(torch_abs, descending=True)[0]

        pooled_features = (features * theta).sum(1)
        return pooled_features, theta


def sort_matrix_by_abs_values(matrix):
    # 获取矩阵的形状
    D, K = matrix.size()

    # 为每一行计算绝对值并进行排序，降序排列
    _, indices = torch.sort(matrix.abs(), dim=1, descending=True)

    # 使用gather函数根据索引重新组织数据
    sorted_matrix = torch.gather(matrix, 1, indices)

    return sorted_matrix
