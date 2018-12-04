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
class GRU_Autoencoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, graph_embedding_size=None, has_input=True, has_output=False, output_size=None):
        super(GRU_Autoencoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output
        self.graph_embedding_size = graph_embedding_size

        self.enc_hidden = None
        self.dec_hidden = None

        # stage 1: encoder

        self.encode_net = nn.Sequential(nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,graph_embedding_size))

        # set up network to decode latent to new hidden state
        self.hidden_net = nn.Linear(graph_embedding_size,self.num_layers*self.hidden_size)

    def init_hidden(self,batch_size):
        pass

    def forward(self,x):
        # compute Z
        pass
