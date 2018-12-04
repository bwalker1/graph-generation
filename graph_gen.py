import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
from tensorboard_logger import configure, log_value
import scipy.misc
import time as tm
from utils import *
from model import *
from data import *
from args import *
import create_graphs
from graph_gen import *

# RNN is the graph-level RNN
# output is the output RNN
# Z is the embedding or a matrix of embeddings to generate multiple graphs
def graph_gen(args, rnn,output,Z,max_prev_node,max_num_node,test_batch_size):
    # generate graphs
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if Z is not None:
        Z = Variable(Z).to(device)
    else:
        rnn.hidden = rnn.init_hidden(batch_size=test_batch_size)

    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, max_prev_node)).to(device) # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,max_prev_node)).to(device)

    #
    h = rnn(x_step,Z,input_len = [0,]*test_batch_size)
    for i in range(max_num_node):
        
        # output.hidden = h.permute(1,0,2)
        hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(2))).to(device)
        output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
                                  dim=0)  # num_layers, batch_size, hidden_size
        x_step = Variable(torch.zeros(test_batch_size,1,max_prev_node)).to(device)

        output_x_step = Variable(torch.ones(test_batch_size,1,1)).to(device)
        for j in range(min(max_prev_node,i+1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
            x_step[:,:,j:j+1] = output_x_step
            output.hidden = Variable(output.hidden.data).to(device)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).to(device)
        
        h = rnn(x_step)

    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list
