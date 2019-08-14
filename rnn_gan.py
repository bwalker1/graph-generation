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


# modified RNN that does full (conditional) generation automatically in forward with Gumbel soft-max
class Generator(nn.Module):
    def __init__(self, max_num_node, max_prev_node, embedding_size_rnn, hidden_size_rnn, num_layers, graph_embedding_size,
                 hidden_size_rnn_output, embedding_size_rnn_output):
        super(Generator, self).__init__()
        self.max_prev_node = max_prev_node
        self.max_num_node = max_num_node
        self.num_layers = num_layers
        # the two RNNs used in the generation
        self.rnn = GRU_plain(input_size=max_prev_node, embedding_size=embedding_size_rnn,
                        hidden_size=hidden_size_rnn, num_layers=num_layers,
                        graph_embedding_size=graph_embedding_size, has_input=False,
                        has_output=True, is_encoder=False,
                        output_size=hidden_size_rnn_output).to(device)
        self.output = GRU_plain(input_size=1, embedding_size=embedding_size_rnn_output,
                               hidden_size=hidden_size_rnn_output, num_layers=num_layers, has_input=True,
                               has_output=True, output_size=1).to(device)

        #
        self.gumbel = torch.distributions.Gumbel(0.0, 1.0)

        # represents how soft our Gumbel softmax is
        self.tau = 1

        self.logsigmoid = nn.LogSigmoid()
        self.softmax = nn.Softmax(dim=1)

    def init_hidden(self, batch_size):
        self.rnn.hidden = self.rnn.init_hidden(batch_size=batch_size)
        self.output.hidden = self.output.init_hidden(batch_size=batch_size)

    def forward(self, Z):
        batch_size = Z.shape[0]

        y_pred = Variable(torch.zeros(batch_size, self.max_num_node, self.max_prev_node)).to(
            device)  # discrete prediction
        x_step = Variable(torch.ones(batch_size, 1, self.max_prev_node)).to(device)

        self.rnn.hidden = self.rnn.init_hidden(batch_size=batch_size)
        #
        h = self.rnn(x_step, Z, input_len=[0, ] * batch_size)
        for i in range(self.max_num_node):

            # output.hidden = h.permute(1,0,2)
            hidden_null = Variable(torch.zeros(self.num_layers - 1, h.size(0), h.size(2))).to(device)
            self.output.hidden = torch.cat((h.permute(1, 0, 2), hidden_null),
                                           dim=0)  # num_layers, batch_size, hidden_size
            x_step = Variable(torch.zeros(batch_size, 1, self.max_prev_node)).to(device)
            output_x_step = Variable(torch.ones(batch_size, 1, 1)).to(device)
            for j in range(min(self.max_prev_node, i + 1)):
                output_y_pred_step = self.output(output_x_step)
                logp = self.logsigmoid(output_y_pred_step)
                h = torch.cat((logp,-logp), dim=1)

                #output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
                # replace hard sample with a Gumbel trick softmax sample
                g = self.gumbel.sample(sample_shape=h.shape)
                output_x_step = self.softmax((h+g)/self.tau)[:,0:1,:]

                x_step[:, :, j:j + 1] = output_x_step
                self.output.hidden = Variable(self.output.hidden.data).to(device)
            y_pred[:, i:i + 1, :] = x_step
            self.rnn.hidden = Variable(self.rnn.hidden.data).to(device)

            h = self.rnn(x_step)

        return y_pred



def train_rnn_gan(args, generator, critic, dataloader):
    optimizer_generator = optim.Adam(list(generator.parameters()), lr=args.lr)
    optimizer_critic = optim.Adam(list(critic.parameters()), lr=args.lr)

    batch_size = args.batch_size

    logsigmoid = nn.LogSigmoid()

    for epoch in range(args.epochs):
        for batch_idx, data in enumerate(dataloader):
            generator.zero_grad()
            critic.zero_grad()

            generator.init_hidden(batch_size=batch_size)
            critic.hidden = critic.init_hidden(batch_size=2*batch_size)


            # generate latent space variables to feed into generator
            Z = Variable(torch.tensor(np.random.normal(size=[batch_size, args.graph_embedding_size]), dtype=torch.float))\
                .to(device)

            # generate the graphs
            y_pred = generator(Z)

            # sample sets of actual data
            y_true = data['y'].float()

            # construct variable encoding whether data is real
            a = torch.tensor(np.concatenate((np.zeros(batch_size), np.ones(batch_size))), dtype=torch.float)
            y = torch.cat((y_pred, y_true), dim=0)

            # run the critic
            log_a_pred = torch.flatten(critic(y))

            # compute the losses
            critic_loss = -(log_a_pred*a - log_a_pred*(1-a)).sum()/(batch_size)
            generator_loss = -logsigmoid(log_a_pred*(1-a)).sum()/batch_size

            critic_loss.backward(retain_graph=True)
            generator_loss.backward()

            optimizer_generator.step()
            optimizer_critic.step()

            if batch_idx==0:
                print(
                    'Epoch: {}/{}, generator loss: {:.6f}, critic loss: {:.6f}'.format(
                        epoch, args.epochs, generator_loss.item(), critic_loss.item()))


def train_rnn_gan_epoch(epoch, args, generator, optimizer_generator, critic, optimizer_critic):
    pass