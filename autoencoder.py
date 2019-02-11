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

from model import *


# This class represents an autoencoder that takes in a graph sequence and
# produces a new graph sequence from the same distribution of graphs (hopefully)
class GRUAutoencoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, graph_embedding_size,
                 output_size=None, hidden_size_rnn_output=None, embedding_size_rnn_output=None):
        super(GRUAutoencoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.graph_embedding_size = graph_embedding_size

        self.enc_hidden = None
        self.dec_hidden = None

        # stage 1: encoder

        self.encoder_rnn = GRU_plain(input_size, embedding_size, hidden_size, num_layers, has_input=False,
                                     has_output=True, is_encoder=True)

        # set up network to decode latent to new hidden state
        self.hidden_net = nn.Linear(graph_embedding_size, self.num_layers * self.hidden_size)
        self.decoder_rnn = GRU_plain(input_size=input_size, embedding_size=input_size,
                                     hidden_size=hidden_size, num_layers=num_layers,
                                     graph_embedding_size=graph_embedding_size, has_input=True,
                                     has_output=True, is_encoder=False,
                                     output_size=hidden_size_rnn_output).to(device)
        self.decoder_output = GRU_plain(input_size=1, embedding_size=embedding_size_rnn_output,
                                        hidden_size=hidden_size_rnn_output, num_layers=num_layers, has_input=True,
                                        has_output=True, output_size=1).to(device)

    def init_hidden(self, batch_size):
        pass

    def forward(self, x, output_x=False, pack=False, input_len=None, input_len_output=None, only_encode=False, decode_Z=None):
        # compute Z
        if decode_Z is not None:
            Z = decode_Z
        else:
            Z = self.encoder_rnn(x, pack=False, input_len=input_len)

        if only_encode:
            return Z
        else:
            # create a decoded network
            h = self.decoder_rnn(x, Z, pack=True, input_len=input_len)
            h = pack_padded_sequence(h, y_len, batch_first=True).data  # get packed hidden vector
            # reverse h...TODO: understand why
            idx = [i for i in range(h.size(0) - 1, -1, -1)]
            idx = Variable(torch.LongTensor(idx)).to(device)
            h = h.index_select(0, idx)
            hidden_null = Variable(torch.zeros(self.num_layers - 1, h.size(0), h.size(1))).to(device)
            self.decoder_output.hidden = torch.cat((h.view(1, h.size(0), h.size(1)), hidden_null),
                                                   dim=0)  # num_layers, batch_size, hidden_size
            y_pred = self.decoder_output(output_x, pack=True, input_len=input_len_output)
            y_pred = F.sigmoid(y_pred)

            return y_pred
