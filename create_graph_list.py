def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from train import *
from args import *
from graph_gen import *
import sys
import argparse
import collections

if __name__=="__main__":
    args_default = Args()

    # handle command line argument
    parser = argparse.ArgumentParser()

    default_vars = vars(args_default)
    for k, v in default_vars.items():
        t = type(v)
        if t == bool:
            # boolean arguments should be handled differently than value arguments
            parser.add_argument('--' + k, dest=k, action='store_true')
            parser.add_argument('--no-' + k, dest=k, action='store_false')
            parser.set_defaults(**{k: v})
        else:
            parser.add_argument('--' + k, dest=k, default=v, type=t)

    args = parser.parse_args()

    if args.conditional:
        # put your desired Z here!
        Z = [1,0]
    else:
        Z = None

    # put how many graphs you want here!
    list_length = 10000

    if args.conditional:
        graph_embedding_size = 2
    else:
        graph_embedding_size = None

    if 'GraphRNN_VAE_conditional' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=False).to(device)
        output = MLP_VAE_conditional_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output,
                                           y_size=args.max_prev_node).to(device)
    elif 'GraphRNN_MLP' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=False).to(device)
        output = MLP_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output,
                           y_size=args.max_prev_node).to(device)
    elif 'GraphRNN_RNN' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers,
                        graph_embedding_size=graph_embedding_size, has_input=True,
                        has_output=True, output_size=args.hidden_size_rnn_output).to(device)
        output = GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
                           hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
                           has_output=True, output_size=1).to(device)
    else:
        raise RuntimeError

    fname = args.model_save_path + fns.fname + 'lstm_' + str(args.load_epoch) + '_cond=' + str(
        args.conditional) + '.dat'
    rnn.load_state_dict(torch.load(fname, map_location='cpu'))
    fname = args.model_save_path + fns.fname + 'output_' + str(args.load_epoch) + '_cond=' + str(
        args.conditional) + '.dat'
    output.load_state_dict(torch.load(fname, map_location='cpu'))


    # Generate a graph
    G = graph_gen(args, rnn, output, torch.Tensor(([Z] * list_length)), args.max_prev_node, args.max_num_node, list_length)