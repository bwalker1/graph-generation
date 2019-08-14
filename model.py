from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from collections import OrderedDict
import math
import numpy as np
import time

# set up to work with or without cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def binary_cross_entropy_weight(y_pred, y, has_weight=False, weight_length=1, weight_max=10):
    '''

    :param y_pred:
    :param y:
    :param weight_length: how long until the end of sequence shall we add weight
    :param weight_value: the magnitude that the weight is enhanced
    :return:
    '''
    if has_weight:
        weight = torch.ones(y.size(0), y.size(1), y.size(2))
        weight_linear = torch.arange(1, weight_length + 1) / weight_length * weight_max
        weight_linear = weight_linear.view(1, weight_length, 1).repeat(y.size(0), 1, y.size(2))
        weight[:, -1 * weight_length:, :] = weight_linear
        loss = F.binary_cross_entropy(y_pred, y, weight=weight.to(device))
    else:
        loss = F.binary_cross_entropy(y_pred, y)
    return loss


def sample_sigmoid(y, sample, thresh=0.5, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y = F.sigmoid(y)
    # do sampling
    if sample:
        if sample_time > 1:
            y_result = Variable(torch.rand(y.size(0), y.size(1), y.size(2))).to(device)
            # loop over all batches
            for i in range(y_result.size(0)):
                # do 'multi_sample' times sampling
                for j in range(sample_time):
                    y_thresh = Variable(torch.rand(y.size(1), y.size(2))).to(device)
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data > 0).any():
                        break
                    # else:
                    #     print('all zero',j)
        else:
            y_thresh = Variable(torch.rand(y.size(0), y.size(1), y.size(2))).to(device)
            y_result = torch.gt(y, y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size(0), y.size(1), y.size(2)) * thresh).to(device)
        y_result = torch.gt(y, y_thresh).float()
    return y_result


def sample_sigmoid_supervised(y_pred, y, current, y_len, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y_pred: input
    :param y: supervision
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y_pred = F.sigmoid(y_pred)
    # do sampling
    y_result = Variable(torch.rand(y_pred.size(0), y_pred.size(1), y_pred.size(2))).to(device)
    # loop over all batches
    for i in range(y_result.size(0)):
        # using supervision
        if current < y_len[i]:
            while True:
                y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2))).to(device)
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                # print('current',current)
                # print('y_result',y_result[i].data)
                # print('y',y[i])
                y_diff = y_result[i].data - y[i]
                if (y_diff >= 0).all():
                    break
        # supervision done
        else:
            # do 'multi_sample' times sampling
            for j in range(sample_time):
                y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2))).to(device)
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                if (torch.sum(y_result[i]).data > 0).any():
                    break
    return y_result


def sample_sigmoid_supervised_simple(y_pred, y, current, y_len, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y_pred: input
    :param y: supervision
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y_pred = F.sigmoid(y_pred)
    # do sampling
    y_result = Variable(torch.rand(y_pred.size(0), y_pred.size(1), y_pred.size(2))).to(device)
    # loop over all batches
    for i in range(y_result.size(0)):
        # using supervision
        if current < y_len[i]:
            y_result[i] = y[i]
        # supervision done
        else:
            # do 'multi_sample' times sampling
            for j in range(sample_time):
                y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2))).to(device)
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                if (torch.sum(y_result[i]).data > 0).any():
                    break
    return y_result


# plain GRU model
class GRU_plain(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, graph_embedding_size=None, has_input=True,
                 has_output=False, is_encoder=False, output_size=None):
        super(GRU_plain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output
        self.is_encoder = is_encoder
        self.graph_embedding_size = graph_embedding_size

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )

        if self.is_encoder:
            assert graph_embedding_size is not None
            self.encode_net = nn.Sequential(nn.Linear(hidden_size, 2*hidden_size), nn.Sigmoid(), nn.Linear(2*hidden_size, 4*hidden_size), nn.Sigmoid(),
                                            nn.Linear(4*hidden_size, graph_embedding_size))

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None  # need initialize before forward run

        if graph_embedding_size is not None:
            self.use_Z = True
            self.hidden_net = nn.Sequential(nn.Linear(graph_embedding_size, self.num_layers * self.hidden_size),
                                            nn.Tanh(), nn.Linear(self.num_layers * self.hidden_size,
                                                                 self.num_layers * self.hidden_size), nn.Tanh(),
                                            nn.Linear(self.num_layers * self.hidden_size,
                                                      self.num_layers * self.hidden_size)
                                            )
        else:
            self.use_Z = False

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        return Variable(torch.ones(self.num_layers, batch_size, self.hidden_size)).to(device)

    def forward(self, input_raw, Z=None, pack=False, input_len=None):
        if self.has_input:
            input = self.input(input_raw)
            input = self.relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)

        if Z is not None and self.use_Z:
            if input_len is None:
                # need to provide input_len so we know batch size
                raise ValueError
            if not self.use_Z:
                # rnn was created without a graph embedding size and has no Z init network
                raise RuntimeError
            batch_size = len(input_len)
            # Run Z through the network and then reshape it accordingly
            self.hidden = self.hidden_net(Z).view(batch_size, self.num_layers, self.hidden_size)\
                              .transpose(0, 1).contiguous()

        output_raw, self.hidden = self.rnn(input, self.hidden)

        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.is_encoder:
            # print(output_raw[:,-1,:].size())
            output_raw = self.encode_net(output_raw[:, -1, :])
            # output_raw = output_raw[:,-1,0:self.graph_embedding_size]
        elif self.has_output:
            output_raw = self.output(output_raw)
        # return hidden state at each time step
        return output_raw
