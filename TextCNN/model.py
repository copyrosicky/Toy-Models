'''
TEXT CNN的模型架构
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Text(nn.Module):
    def __init__(self, embed_num, embed_dim, num_classes, out_channels,
                 kernel_size, dropout, in_channels = 1):
        '''
        :param embed_num embedding数目 voc_size
        :param embed_dim embedding维度
        :param num_classes 分类数目
        :param in_channels 输入通道数 默认为1
        :param out_channels  输出通道数目
        :param kernel_size 卷积核尺寸 列表格式 只输入卷积核的高度（宽度固定为embed_dim）
        :param dropout 列表格式

        '''
        super(CNN_Text, self).__init__()
        self.embed = nn.Embedding(embed_num, embed_dim)

        # TEXT CNN卷积核宽度必须和文本embedding size相同
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, (k, embed_dim)) for k in kernel_size])

        # 默认dropout 为 0.2
        self.dropout = nn.Dropout(dropout)

        # 输出的FC层
        self.fc = nn.Linear(len(kernel_size) * out_channels, num_classes)

    def forward(self, x):
        '''
        x : [batch_size, num_word, voc_size]
        '''
        # embedding部分
        embed_word = self.embed(x)  # batch_size num_word emb_size
        embed_word = embed_word.unsqueeze(1)  # batch_size  in_channels(文本数据始终为1) num_word emb_size

        # 卷积和1-max-pooling部分
        conv_out = [F.relu(conv(embed_word)).squeeze(3) for conv in self.convs]
        # [(batch_size, out_channels, num_word), ...]*len(kernel_size)

        max_pool_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_out]
        # [(batch_size, out_channels), ...]*len(kernel_size)
        max_pool_out = torch.cat(max_pool_out, 1)
        max_pool_out = self.dropout(max_pool_out)  # (batch_size, len(kernel_size)*out_channels)

        logit = self.fc(max_pool_out)  # (batch_size, num_classes)
        return logit
