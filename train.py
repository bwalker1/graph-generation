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


def train_rnn_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    # set up to work with or without cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rnn.train()
    output.train()
    loss_sum = 0

    use_Z = rnn.use_Z
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
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
        y = Variable(y).to(device)
        output_x = Variable(output_x).to(device)
        output_y = Variable(output_y).to(device)
        # print(output_y_len)
        # print('len',len(output_y_len))
        # print('y',y.size())
        # print('output_y',output_y.size())

        if use_Z:
            Z = data['Z'].float()
            Z = Variable(Z).to(device)
        else:
            rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
            Z = None

        # if using ground truth to train
        h = rnn(x, Z, pack=True, input_len=y_len)
        h = pack_padded_sequence(h, y_len, batch_first=True).data  # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).to(device)
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(1))).to(device)
        output.hidden = torch.cat((h.view(1, h.size(0), h.size(1)), hidden_null),
                                  dim=0)  # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y, output_y_len, batch_first=True)
        output_y = pad_packed_sequence(output_y, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()

        if epoch % args.epochs_log == 0 and batch_idx == 0:  # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs, loss.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        if args.use_tensorboard:
            log_value('loss_' + args.fname, loss.data[0], epoch * args.batch_ratio + batch_idx)
        feature_dim = y.size(1) * y.size(2)
        loss_sum += loss.item() * feature_dim
    return loss_sum / (batch_idx + 1)


def test_rnn_epoch(epoch, args, rnn, output, test_batch_size=16, Z_list=None):
    # rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # initialize testing Z
    # TODO: make this vary based on input Z
    if Z_list is None:
        Z = None
    else:
        Z = np.array(random.sample(Z_list, k=test_batch_size))

    return graph_gen(rnn, output, Z, int(args.max_prev_node), int(args.max_num_node), test_batch_size)


# train function for LSTM + VAE
def train(args, dataset_train, rnn, output, Z_list=None):
    # get the filenames that we'll need for saving
    fns = filenames(args)
    # check if load existing model
    if args.load:

        fname = args.model_save_path + fns.fname + 'lstm_' + str(args.load_epoch) + '_cond=' + str(
            args.conditional) + '.dat'
        rnn.load_state_dict(torch.load(fname, map_location='cpu'))
        fname = args.model_save_path + fns.fname + 'output_' + str(args.load_epoch) + '_cond=' + str(
            args.conditional) + '.dat'
        output.load_state_dict(torch.load(fname, map_location='cpu'))

        args.lr = 0.00001
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1

    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch <= args.epochs:
        time_start = tm.time()
        # train
        if 'GraphRNN_RNN' in args.note:
            train_rnn_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        else:
            raise RuntimeError
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:
            for sample_time in range(1, 4):
                G_pred = []
                while len(G_pred) < args.test_total_size:
                    if 'GraphRNN_RNN' in args.note:
                        G_pred_step = test_rnn_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,
                                                     Z_list=Z_list)
                    else:
                        raise RuntimeError
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = args.graph_save_path + fns.fname_pred + str(epoch) + '_' + str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)
                if 'GraphRNN_RNN' in args.note:
                    break
            print('test done, graphs saved')

        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + fns.fname + 'lstm_' + str(epoch) + '_cond=' + str(
                    args.conditional) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + fns.fname + 'output_' + str(epoch) + '_cond=' + str(
                    args.conditional) + '.dat'
                torch.save(output.state_dict(), fname)
        epoch += 1
    np.save(args.timing_save_path + fns.fname, time_all)


def train_encoder(args, dataset_train, rnn, Z_list):
    # get the filenames that we'll need for saving
    fns = filenames(args)
    # check if load existing model
    if args.load:
        epoch = args.load_epoch
        fname = args.model_save_path + fns.fname + 'rnn_' + str(epoch) + '_encoder.dat'
        rnn.load_state_dict(torch.load(fname, map_location='cpu'))

        args.lr = 0.0003

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
        # train
        train_rnn_encoder_epoch(epoch, args, rnn, dataset_train,
                                optimizer_rnn,
                                scheduler_rnn)
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        # ** todo: do some sort of testing

        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + fns.fname + 'rnn_' + str(epoch) + '_encoder.dat'
                torch.save(rnn.state_dict(), fname)
        epoch += 1
    np.save(args.timing_save_path + fns.fname, time_all)


def test_rnn_encoder(args, rnn, data_loader):
    rnn.eval()

    Zs = []
    ps = []

    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        # rnn.encode_net.requires_grad=False

        x_unsorted = data['x'].float()
        # print(data['id'])
        # print(x_unsorted[:,:,0])
        # print(data['id'])
        # print(torch.sum(data['x'],dim=[1,2]))
        # print(data['Z'])
        # exit(1)

        y_unsorted = data['y'].float()

        Z_unsorted = data['Z'].long()

        y_len_unsorted = data['len']

        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        # print("y_len_unsorted: ", y_len_unsorted)
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        # print("sort_index: ",sort_index)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)
        Z = torch.index_select(Z_unsorted, 0, sort_index)
        # print(torch.sum(x,dim=[1,2]))

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
        y = Variable(y).to(device)
        output_x = Variable(output_x).to(device)
        output_y = Variable(output_y).to(device)
        # print(output_y_len)
        # print('len',len(output_y_len))
        # print('y',y.size())
        # print('output_y',output_y.size())

        # if using ground truth to train
        Z_pred = rnn(x, pack=False, input_len=y_len)

        # use cross entropy loss

        # print(Z.size())
        Z = Z.to(device)
        # print(Z_pred)
        # print(Z)
        loss = nn.CrossEntropyLoss()(Z_pred, torch.max(Z, 1)[1])

        prob = list(np.array(nn.Softmax()(Z_pred).detach())[:, 1])
        correct = list(Z.detach().cpu().numpy()[:, 1])
        # print(prob)
        # print(correct)
        ps.extend(prob)
        Zs.extend(correct)

        # compute hard max accuracy
        class_pred = torch.max(Z_pred, 1)[1].long()
        class_true = torch.max(Z, 1)[1].long()
        # print(class_pred)
        # print(class_true)
        acc = torch.sum((class_true == class_pred).float()).item() / x_unsorted.shape[0]
        # update deterministic and lstm

        if batch_idx == 0:  # only output first batch's statistics
            print(
                'Testing: test loss: {:.6f}, test accuracy: {:.6f}, graph type: {}, num_layer: {}, hidden: {}, batch size: {}'.format(
                    loss.item(), acc, args.graph_type, args.num_layers, args.hidden_size_rnn, x_unsorted.shape[0]))

    # create histogram and ROC curve

    # ROC curve
    sort_index = sorted(range(len(ps)), key=lambda x: ps[x])
    # ps.sort()
    Zs = np.array(Zs)
    true_ps = np.array(ps)[Zs == 1]
    true_ps.sort()
    false_ps = np.array(ps)[Zs == 0]
    false_ps.sort()
    roc_y = [np.sum(true_ps > i) / len(true_ps) for i in false_ps]
    roc_x = [1 - i / len(false_ps) for i in range(len(false_ps))]

    plt.figure(num=1, figsize=(4, 3))
    plt.hist(true_ps, bins=np.linspace(0, 1, 101))
    plt.hist(false_ps, bins=np.linspace(0, 1, 101))
    plt.savefig('figures/hist.png', dpi=200)
    plt.close()

    roc_x.append(0)
    roc_y.append(0)

    plt.figure(num=2, figsize=(4, 3))
    plt.plot(roc_x, roc_y)
    plt.savefig('figures/roc.png', dpi=200)
    plt.close()
    # Zs = np.cumsum(np.array(Zs)[sort_index])/(len(Zs)/2)
    # print(Zs)


def train_rnn_encoder_epoch(epoch, args, rnn, data_loader,
                            optimizer_rnn, scheduler_rnn):
    # set up to work with or without cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rnn.train()
    loss_sum = 0

    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        # rnn.encode_net.requires_grad=False

        x_unsorted = data['x'].float()

        # print(torch.sum(data['x'],dim=[1,2]))
        # print(data['Z'])
        # exit(1)

        y_unsorted = data['y'].float()

        Z_unsorted = data['Z'].float()

        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)
        Z = torch.index_select(Z_unsorted, 0, sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y, y_len, batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx)

        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin) - 1, 0, -1):
            count_temp = np.sum(output_y_len_bin[i:])  # count how many y_len is above i
            output_y_len.extend(
                [min(i, y.size(2))] * count_temp)  # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).to(device)
        y = Variable(y).to(device)

        # if using ground truth to train
        Z_pred = rnn(x, pack=False, input_len=y_len)

        # use cross entropy loss
        # Z=Variable(Z).to(device)
        Z = Z.to(device)
        # print(Z_pred)
        # print(Z)
        # loss = nn.CrossEntropyLoss()(Z_pred,torch.max(Z,1)[1])
        loss = nn.MSELoss()(Z_pred, Z)
        loss.backward()

        # compute hard max accuracy
        class_pred = torch.max(Z_pred, 1)[1]
        class_true = torch.max(Z, 1)[1]
        acc = (torch.sum(class_true == class_pred).float()).item() / len(data['x'])
        # update deterministic and lstm
        optimizer_rnn.step()
        scheduler_rnn.step()

        if epoch % args.epochs_log == 0 and batch_idx == 0:  # only output first batch's statistics
            print(
                'Epoch: {}/{}, train loss: {:.6f}, training accuracy: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                    epoch, args.epochs, loss.item(), acc, args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        if args.use_tensorboard:
            log_value('loss_' + args.fname, loss.item(), epoch * args.batch_ratio + batch_idx)
        feature_dim = y.size(1) * y.size(2)
        loss_sum += loss.item() * feature_dim
    return loss_sum / (batch_idx + 1)
