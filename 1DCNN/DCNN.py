# -- coding: utf-8 --
# encoding: utf-8

'''
1DCNN模型的架构 用于处理表格数据
'''

import torch
from torch import functional as F
from torch import nn

class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size, channels = [256, 512, 512]):
        '''
        :parm num_features 输入数据的维度
        :parm num_targets 输出数据维度
        :parm hidden_size 隐藏层维度
        :parm channels 卷积层的通道数 列表格式数据
        '''
        super(Model, self).__init__()

        channel_1 = channels[0]
        channel_2 = channels[1]
        channel_3 = channels[2]

        channel_pool_1 = int(hidden_size / channel_1 / 2)
        channel_pool_2 = int(hidden_size / channel_1 / 2 / 2) * channel_3

        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.channel_3 = channel_3

        # Block1 dense层提高输入数据维度 为卷积核提供足够的“pixel”
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.Linear(num_features, hidden_size)

        # Conv1
        self.batch_norm_c1 = nn.BatchNorm1d(channel_1)
        self.dropout_c1 = nn.Dropout(0.1)
        self.conv1 = nn.Conv1d(channel_1, channel_2, kernel_size=5, stride=1, padding=2, bias=False)
        self.avg_pool_c1 = nn.AdaptiveAvgPool1d(output_size=channel_pool_1)

        # Conv2
        self.batch_norm_c2 = nn.BatchNorm1d(channel_2)
        self.dropout_c2 = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(channel_2, channel_2, kernel_size=3, stride=1, padding=1, bias=True)

        self.batch_norm_c2_1 = nn.BatchNorm1d(channel_2)
        self.dropout_c2_1 = nn.Dropout(0.3)
        self.conv2_1 = nn.Conv1d(channel_2, channel_2, kernel_size=3, stride=1, padding=1, bias=True)

        self.batch_norm_c2_2 = nn.BatchNorm1d(channel_2)
        self.dropout_c2_2 = nn.Dropout(0.2)
        self.conv2_2 = nn.Conv1d(channel_2, channel_3, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flatten = nn.Flatten()

        # Block3 输出层
        self.batch_norm3 = nn.BatchNorm1d(channel_pool_2)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(channel_pool_2, num_targets)

    def forward(self, x):
        '''
        x : [batch_size, num_features]
        '''
        # Block1
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x), alpha=0.06)   # batch_size hidden_size

        x = x.reshape(x.shape[0], self.channel_1, -1)  # batch_size  channel_1(256)  hidden_size / channel_1

        # Conv1
        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)               # conv1 size = 5  pad = 2
        x = F.relu(self.conv1(x))            # batch_size   channel_2(512)  (hidden_size / channel_1 + 2padding - kernel size)/2 + 1

        x = self.avg_pool_c1(x)              # batch_size   channel_2(512)   hidden_size / channel_1 / 2

        # Conv2
        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))          #  batch_size   channel_2(512)  hidden_size / channel_1 / 2
        conv2_out = x

        # Conv2_1
        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        conv2_1_out = F.relu(self.conv2_1(x))

        # Conv2_2
        x = self.batch_norm_c2_2(conv2_1_out)
        x = self.dropout_c2_2(x)
        conv2_2_out = F.relu(self.conv2_2(x))       # batch_size   channel_2(512)  hidden_size / channel_1 / 2
        conv_out = conv2_2_out * conv2_out

        conv_out = self.max_pool_c2(conv_out)          # kernel_size=4, stride=2, padding=1
                                         # batch_size   channel_2(512)  hidden_size / channel_1 / 4
        x = self.flatten(conv_out)

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        out = self.dense3(x)            # batch_size (hidden_size / channel_1 / 4) * channel_3(512)

        return out