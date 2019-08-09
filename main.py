# this gets rid of annoying deprecation warnings when you import other people's stuff
# note: it also gets rid of potentially legitimate warnings
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

from train import *
from train_autoencoder import *
from args import *
from graph_gen import *
import sys
import argparse
import collections

import matplotlib.pyplot as plt

from autoencoder import *

if __name__ == '__main__':
    # set up to work with or without cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Main is using " + ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    # All necessary arguments are defined in args.py
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

    # parser.add_argument('--conditional', default=True)
    # parser.set_defaults(**vars(args_default))
    args = parser.parse_args()

    if args.conditional:
        print("Using conditional input")
    else:
        print("Not using conditional input")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('CUDA', args.cuda)
    fns = filenames(args)
    print('File name prefix', fns.fname)
    # check if necessary directories exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)

    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)
    if args.use_tensorboard:
        if args.clean_tensorboard:
            if os.path.isdir("tensorboard"):
                shutil.rmtree("tensorboard")
        configure("tensorboard/run" + time, flush_secs=5)

    generate_graphs = True
    if generate_graphs:
        graphs = create_graphs.create(args)
        draw_graph(graphs[0], 'Z=0')
        draw_graph(graphs[-1], 'Z=1')
        # plot_degree_distribution(graphs)
        # plt.show()

        # split datasets
        # random.seed(123)

        shuffle(graphs)
        graphs_len = len(graphs)
        # graphs_test = graphs[int(0.8 * graphs_len):]
        graphs_test = graphs[400:500] + graphs[900:1000]
        graphs_train = graphs[0:400] + graphs[500:900]
        shuffle(graphs_train)
        graphs_validate = graphs[0:int(0.2 * graphs_len)]

        if args.conditional:
            Z_list = [G.graph['Z'] for G in graphs]
        else:
            Z_list = None

        # if use pre-saved graphs
        # dir_input = "/dfs/scratch0/jiaxuany0/graphs/"
        # fname_test = dir_input + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
        #     args.hidden_size_rnn) + '_test_' + str(0) + '.dat'
        # graphs = load_graph_list(fname_test, is_real=True)
        # graphs_test = graphs[int(0.8 * graphs_len):]
        # graphs_train = graphs[0:int(0.8 * graphs_len)]
        # graphs_validate = graphs[int(0.2 * graphs_len):int(0.4 * graphs_len)]

        graph_validate_len = 0
        for graph in graphs_validate:
            graph_validate_len += graph.number_of_nodes()
        graph_validate_len /= len(graphs_validate)
        print('graph_validate_len', graph_validate_len)

        graph_test_len = 0
        for graph in graphs_test:
            graph_test_len += graph.number_of_nodes()
        graph_test_len /= len(graphs_test)
        print('graph_test_len', graph_test_len)

        args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
        max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
        min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

        # args.max_num_node = 2000
        # show graphs statistics
        print('total graph num: {}, training set: {}'.format(len(graphs), len(graphs_train)))
        print('max number node: {}'.format(args.max_num_node))
        print('max/min number edge: {}; {}'.format(max_num_edge, min_num_edge))
        print('max previous node: {}'.format(args.max_prev_node))

        # save ground truth graphs
        ## To get train and test set, after loading you need to manually slice
        save_graph_list(graphs, args.graph_save_path + fns.fname_train + '0.dat')
        save_graph_list(graphs, args.graph_save_path + fns.fname_test + '0.dat')
        print('train and test graphs saved at: ', args.graph_save_path + fns.fname_test + '0.dat')

        ### dataset initialization
        dataset = Graph_sequence_sampler_pytorch(graphs_train, max_prev_node=args.max_prev_node,
                                                 max_num_node=args.max_num_node, use_classes=args.conditional,
                                                 iteration=args.max_prev_node_iter)
        dataset_test = Graph_sequence_sampler_pytorch(graphs_test, max_prev_node=args.max_prev_node,
                                                      max_num_node=args.max_num_node, use_classes=args.conditional,
                                                      iteration=args.max_prev_node_iter)
        if args.max_prev_node is None:
            args.max_prev_node = dataset.max_prev_node
        sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
            [1.0 / len(dataset) for i in range(len(dataset))],
            num_samples=args.batch_size * args.batch_ratio, replacement=True)
        sample_strategy_test = torch.utils.data.sampler.WeightedRandomSampler(
            [1.0 / len(dataset_test) for i in range(len(dataset_test))],
            num_samples=args.test_batch_size, replacement=True)
        #sample_strategy_test = torch.utils.data.sampler.SequentialSampler(dataset_test)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                     sampler=sample_strategy)
        dataset_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size,
                                                          num_workers=args.num_workers,
                                                          sampler=sample_strategy_test)
    else:
        raise RuntimeError

    ### model initialization
    # check whether we're using conditional input
    if args.conditional:
        graph_embedding_size = args.graph_embedding_size
    else:
        graph_embedding_size = None

    if args.mode == "autoencoder":
        #(self, input_size, embedding_size, hidden_size, num_layers, graph_embedding_size, output_size=None, hidden_size_rnn_output=None, embedding_size_rnn_output=None)
        rnn = GRUAutoencoder(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                             hidden_size=args.hidden_size_rnn, num_layers=args.num_layers,
                             output_size=args.hidden_size_rnn_output,
                             graph_embedding_size=graph_embedding_size,hidden_size_rnn_output=args.hidden_size_rnn_output,
                             embedding_size_rnn_output=args.embedding_size_rnn_output).to(device)
    else:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers,
                        graph_embedding_size=graph_embedding_size, has_input=False,
                        has_output=True, is_encoder=args.train_encoder, output_size=args.hidden_size_rnn_output).to(device)
        if args.mode == "decoder":
            output = GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
                               hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
                               has_output=True, output_size=1).to(device)

    ### start training
    if args.train:
        if args.mode == "decoder":
            train(args, dataset_loader, rnn, output, Z_list)
        elif args.mode == "encoder":
            train_encoder(args, dataset_loader, rnn, Z_list)
        elif args.mode == "autoencoder":
            train_autoencoder(args, dataset_loader, rnn)

    # test phase
    if True:
        if args.mode == "encoder":
            test_rnn_encoder(args, rnn, dataset_loader_test)

    if args.make_graph_list:
        if args.mode == "encoder":
            if not args.train:
                # if we didn't just train, load something instead
                fname = args.model_save_path + fns.fname + 'lstm_' + str(args.load_epoch) + '_cond=' + str(
                    args.conditional) + '.dat'
                rnn.load_state_dict(torch.load(fname, map_location='cpu'))
                fname = args.model_save_path + fns.fname + 'output_' + str(args.load_epoch) + '_cond=' + str(
                    args.conditional) + '.dat'
                output.load_state_dict(torch.load(fname, map_location='cpu'))

            # how many to generate
            list_length = 1000
            # desired Z value (if you're using conditonal
            Z = torch.Tensor([[1, 0]] * list_length) if args.conditional else None
            # Generate a graph list
            G = graph_gen(args, rnn, output, Z, args.max_prev_node, args.max_num_node, list_length)
            # save the graphs
            save_graph_list(G, fns.fname_test2)
        elif args.mode == "autoencoder":
            # use our autoencoder to embed the test set
            out = test_autoencoder(args, rnn, dataset_loader_test)

            print(out)

            plt.switch_backend("agg")
            for list in out:
                plt.scatter(list[:, 0], list[:, 1])
            #plt.scatter(out[0][:,0], out[0][:,1])
            #plt.scatter(out[1][:,0], out[1][:,1])
            plt.savefig("figures/latent_encoding.png", dpi=300)
            plt.close()