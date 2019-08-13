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
from geomloss import SamplesLoss

from utils import *
from model import *
from data import *
from args import *
import create_graphs
from graph_gen import *
from collections import defaultdict

def test_autoencoder(args, rnn, data_loader):
    rnn.eval()

    arr = defaultdict(list)

    for batch_idx, data in enumerate(data_loader):
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        Z_unsorted = data['Z'].float()

        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        Z = torch.index_select(Z_unsorted, 0, sort_index)
        x = Variable(x).to(device)

        # feed the input graphs into the encoder to get the decoded sequence
        Z_pred = rnn(x, pack=False, input_len=y_len, only_encode=True)
        Z_pred = Z_pred.detach().cpu().numpy()
        N = Z_pred.shape[0]

        Nd = Z.shape[1]
        if Nd > 1:
            # one-hot classes
            labels = np.argmax(Z.numpy(), axis=1)
        else:
            # continuous label
            pass

        for i in range(Nd):
            arr[i].extend(Z_pred[labels == i])


    return {k: np.array(v) for k, v in arr.items()}




def train_autoencoder_epoch(epoch, args, rnn, data_loader,
                    optimizer_rnn,  scheduler_rnn):
    # set up to work with or without cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rnn.train()
    loss_sum = 0

    critic = nn.Sequential(nn.Linear(args.graph_embedding_size,4*args.graph_embedding_size),nn.ReLU(),
                           nn.Linear(4*args.graph_embedding_size, 8* args.graph_embedding_size), nn.ReLU(),
                           nn.Linear(8*args.graph_embedding_size, 4 * args.graph_embedding_size),nn.ReLU(),
                           nn.Linear(4*args.graph_embedding_size, 1))
    optimizer_critic = optim.Adam(critic.parameters(), lr=0.001)

    regularizer_loss_func = SamplesLoss(loss="sinkhorn")
    sigmoid = nn.Sigmoid()
    logsigmoid = nn.LogSigmoid()

    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        critic.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()

        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        # rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y, y_len, batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0), y_reshape.size(1), 1)

        output_x = torch.cat((torch.ones(y_reshape.size(0), 1, 1), y_reshape[:, 0:-1, 0:1]), dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin) - 1, 0, -1):
            count_temp = np.sum(output_y_len_bin[i:])  # count how many y_len is above i
            output_y_len.extend(
                [min(i, y.size(2))] * count_temp)  # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).to(device)
        #y = Variable(y).to(device)
        output_x = Variable(output_x).to(device)


        # feed the input graphs into the encoder to get the decoded sequence
        y_pred, Z_pred = rnn(x, output_x=output_x, pack=False, input_len=y_len, input_len_output=output_y_len)

        # compute the regularization term in the loss function
        # start with simple regularization to normal distribution (questionable results in literature)
        #Z_g = (torch.tensor(np.random.normal(size=Z_pred.shape), dtype=torch.float)).to(device)
        #regularizer_loss = 2*regularizer_loss_func(Z_pred, Z_g)
        regularizer_loss = Z_pred.abs().max(dim=1).sum()

        #D_pred = critic(Z_pred)
        #D_g = critic(Z_g)

        #regularizer_loss = (sigmoid(D_pred).sum())/len(D_pred)
        #critic_loss = logsigmoid(D_g).sum() - logsigmoid(D_pred).sum()

        # compute the reconstruction term in the loss function
        output_y = Variable(output_y).to(device)

        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]

        output_y = pack_padded_sequence(output_y, output_y_len, batch_first=True)
        output_y = pad_packed_sequence(output_y, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y) + regularizer_loss
        #critic_loss.backward(retain_graph=True)
        loss.backward()
        feature_dim = y.size(1) * y.size(2)
        loss_sum += loss.item() * feature_dim
        # update deterministic and lstm
        optimizer_rnn.step()
        scheduler_rnn.step()

        optimizer_critic.step()

        if epoch % args.epochs_log == 0 and batch_idx == 0:  # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs, loss.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))
    return loss_sum / (batch_idx + 1)



# train function for LSTM + VAE
def train_autoencoder(args, dataset_train, rnn):
    # get the filenames that we'll need for saving
    fns = filenames(args)
    # check if load existing model
    if args.load:

        fname = args.model_save_path + fns.fname + 'lstm_' + str(args.load_epoch) + '_cond=' + str(
            args.conditional) + '.dat'
        rnn.load_state_dict(torch.load(fname, map_location='cpu'))
        fname = args.model_save_path + fns.fname + 'output_' + str(args.load_epoch) + '_cond=' + str(
            args.conditional) + '.dat'

        args.lr = 0.00001
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1

    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch <= args.epochs:
        time_start = tm.time()
        # train_autoencoder_epoch(epoch, args, rnn, data_loader,
        #                     optimizer_rnn,  scheduler_rnn):
        train_autoencoder_epoch(epoch, args, rnn, dataset_train,
                                optimizer_rnn, scheduler_rnn)
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # possibly insert testing here

        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + fns.fname + 'autoencoder_' + str(epoch) + '_cond=' + str(
                    args.conditional) + '.dat'
                torch.save(rnn.state_dict(), fname)
        epoch += 1
    np.save(args.timing_save_path + fns.fname, time_all)
