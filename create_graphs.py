import networkx as nx
import numpy as np

from utils import *
from data import *
from sbm import *

def onehot(k,n):
    v = np.array([0,]*n)
    v[k]=1
    return v

def create(args):
### load datasets
    graphs=[]
    # synthetic graphs
    if args.graph_type=='ladder':
        graphs = []
        for i in range(100, 201):
            graphs.append(nx.ladder_graph(i))
        args.max_prev_node = 10
    elif args.graph_type=='ladder_small':
        graphs = []
        for i in range(2, 11):
            graphs.append(nx.ladder_graph(i))
        args.max_prev_node = 10
    elif args.graph_type=='tree':
        graphs = []
        for i in range(2,5):
            for j in range(3,5):
                graphs.append(nx.balanced_tree(i,j))
        args.max_prev_node = 256
    elif args.graph_type=='caveman':
        # graphs = []
        # for i in range(5,10):
        #     for j in range(5,25):
        #         for k in range(5):
        #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
        graphs = []
        for i in range(2, 3):
            for j in range(30, 81):
                for k in range(10):
                    graphs.append(caveman_special(i,j, p_edge=0.3))
        args.max_prev_node = 100
    elif args.graph_type=='caveman_small2':
        # graphs = []
        # for i in range(2,5):
        #     for j in range(2,6):
        #         for k in range(10):
        #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
        graphs = []
        for j in range(6, 11):
            for k in range(20):
                G = nx.relaxed_caveman_graph(2,int(1.5*j),p=0.15)
                G.graph['Z'] = np.array([1,0])
                graphs.append(G)
        args.max_prev_node = 20
    elif args.graph_type=='caveman_small':
        # graphs = []
        # for i in range(2,5):
        #     for j in range(2,6):
        #         for k in range(10):
        #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
        graphs = []
        for i in range(2, 3):
            for j in range(6, 11):
                for k in range(20):
                    graphs.append(caveman_special(i, j, p_edge=0.8)) # default 0.8
        args.max_prev_node = 20
    elif args.graph_type=='caveman_small3':
        # graphs = []
        # for i in range(2,5):
        #     for j in range(2,6):
        #         for k in range(10):
        #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
        graphs = []
        for j in range(6, 11):
            for k in range(20):
                #G = caveman_special(3,j,p_edge=1)
                G = nx.relaxed_caveman_graph(3,j,p=0.15)
                G.graph['Z'] = np.array([0,1])
                graphs.append(G)
        args.max_prev_node = 20
    elif args.graph_type=="caveman_sbm_mixed":
        graphs = []
        for k in range(100):
            G = nx.relaxed_caveman_graph(2,int(32),p=0.15)
            G.graph['Z'] = np.array([1,0,])
            G.graph['id'] = k
            graphs.append(G)
        #graphs += generateSetOfSBM([[64 // x]*x for x in [2]*500])
    elif args.graph_type=='er_ba_mixed':
        graphs = []
        for k in range(500):
            G = nx.random_regular_graph(4,50)
            G.graph['Z'] = np.array([1,0])
            graphs.append(G)
        for k in range(500):
            G = nx.barabasi_albert_graph(50,2)
            G.graph['Z'] = np.array([0,1])
            graphs.append(G)
        args.max_prev_node=50
    elif args.graph_type=='caveman_small_mixed':
        graphs = []
        for k in range(500):
            G = nx.relaxed_caveman_graph(2,int(18),p=0.15)
            G.graph['Z'] = np.array([1,0])
            G.graph['id'] = k
            graphs.append(G)
#         for k in range(50):
#             G = nx.relaxed_caveman_graph(3,int(12),p=0.15)
#             G.graph['Z'] = np.array([0,1])
#             graphs.append(G)
        for k in range(500):
            G = nx.relaxed_caveman_graph(3,int(12),p=0.15)
            G.graph['Z'] = np.array([0,1])
            G.graph['id'] = k
            graphs.append(G)
        args.max_prev_node = 30
    elif args.graph_type=='many_classes':
        for k in range(100):
            G = nx.relaxed_caveman_graph(2,int(30),p=0.15)
            G.graph['Z'] = onehot(0,10)
            graphs.append(G)
        for k in range(100):
            G = nx.relaxed_caveman_graph(3,int(20),p=0.15)
            G.graph['Z'] = onehot(1,10)
            graphs.append(G)
        for k in range(100):
            G = nx.relaxed_caveman_graph(4,int(15),p=0.15)
            G.graph['Z'] = onehot(2,10)
            graphs.append(G)
        for k in range(100):
            G = nx.relaxed_caveman_graph(5,int(12),p=0.15)
            G.graph['Z'] = onehot(3,10)
            graphs.append(G)
        for k in range(100):
            G = nx.ladder_graph(60)
            G.graph['Z'] = onehot(4,10)
        for k in range(100):
            G = nx.grid_2d_graph(10,6)
            G.graph['Z'] = onehot(5,10)
            graphs.append(G)
        for k in range(100):
            G = nx.random_regular_graph(4,60)
            G.graph['Z'] = onehot(6,10)
            graphs.append(G)
        for k in range(100):
            G = nx.barabasi_albert_graph(60,2)
            G.graph['Z'] = onehot(7,10)
            graphs.append(G)
        for k in range(100):
            G = nx.balanced_tree(2,6)
            G.graph['Z'] = onehot(7,10)
            graphs.append(G)
        for k in range(100):
            G = nx.balanced_tree(4,3)
            G.graph['Z'] = onehot(8,10)
            graphs.append(G)
        for k in range(100):
            G = nx.wheel_graph(60)
            G.graph['Z'] = onehot(9,10)
            graphs.append(G)

        args.max_prev_node = 50
    elif args.graph_type=='caveman_small_single':
        # graphs = []
        # for i in range(2,5):
        #     for j in range(2,6):
        #         for k in range(10):
        #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
        graphs = []
        for i in range(2, 3):
            for j in range(8, 9):
                for k in range(100):
                    graphs.append(caveman_special(i, j, p_edge=0.5))
        args.max_prev_node = 20
    elif args.graph_type.startswith('community'):
        num_communities = int(args.graph_type[-1])
        print('Creating dataset with ', num_communities, ' communities')
        c_sizes = np.random.choice([12, 13, 14, 15, 16, 17], num_communities)
        #c_sizes = [15] * num_communities
        for k in range(3000):
            graphs.append(n_community(c_sizes, p_inter=0.01))
        args.max_prev_node = 80
    elif args.graph_type=='grid':
        graphs = []
        for i in range(10,20):
            for j in range(10,20):
                graphs.append(nx.grid_2d_graph(i,j))
        args.max_prev_node = 40
    elif args.graph_type=='grid_small':
        graphs = []
        for i in range(2,5):
            for j in range(2,6):
                graphs.append(nx.grid_2d_graph(i,j))
        args.max_prev_node = 15
    elif args.graph_type=='barabasi':
        graphs = []
        for i in range(100,200):
             for j in range(4,5):
                 for k in range(5):
                    graphs.append(nx.barabasi_albert_graph(i,j))
        args.max_prev_node = 130
    elif args.graph_type=='barabasi_small':
        graphs = []
        for i in range(4,21):
             for j in range(3,4):
                 for k in range(10):
                    graphs.append(nx.barabasi_albert_graph(i,j))
        args.max_prev_node = 20
    elif args.graph_type=='grid_big':
        graphs = []
        for i in range(36, 46):
            for j in range(36, 46):
                graphs.append(nx.grid_2d_graph(i, j))
        args.max_prev_node = 90

    elif args.graph_type == 'sbm_large':
        N = 100
        graphs = generateSetOfSBM([[N // x]*x for x in [2,4]*500])

        args.max_prev_node = 80
        args.max_num_node = 100

    elif 'barabasi_noise' in args.graph_type:
        graphs = []
        for i in range(100,101):
            for j in range(4,5):
                for k in range(500):
                    graphs.append(nx.barabasi_albert_graph(i,j))
        graphs = perturb_new(graphs,p=args.noise/10.0)
        args.max_prev_node = 99

    # real graphs
    # elif args.graph_type == 'enzymes':
    #     graphs= Graph_load_batch(min_num_nodes=10, name='ENZYMES')
    #     args.max_prev_node = 25
    # elif args.graph_type == 'enzymes_small':
    #     graphs_raw = Graph_load_batch(min_num_nodes=10, name='ENZYMES')
    #     graphs = []
    #     for G in graphs_raw:
    #         if G.number_of_nodes()<=20:
    #             graphs.append(G)
    #     args.max_prev_node = 15
    # elif args.graph_type == 'protein':
    #     graphs = Graph_load_batch(min_num_nodes=20, name='PROTEINS_full')
    #     args.max_prev_node = 80
    # elif args.graph_type == 'DD':
    #     graphs = Graph_load_batch(min_num_nodes=100, max_num_nodes=500, name='DD',node_attributes=False,graph_labels=True)
    #     args.max_prev_node = 230
    # elif args.graph_type == 'citeseer':
    #     _, _, G = Graph_load(dataset='citeseer')
    #     G = max(nx.connected_component_subgraphs(G), key=len)
    #     G = nx.convert_node_labels_to_integers(G)
    #     graphs = []
    #     for i in range(G.number_of_nodes()):
    #         G_ego = nx.ego_graph(G, i, radius=3)
    #         if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
    #             graphs.append(G_ego)
    #     args.max_prev_node = 250
    # elif args.graph_type == 'citeseer_small':
    #     _, _, G = Graph_load(dataset='citeseer')
    #     G = max(nx.connected_component_subgraphs(G), key=len)
    #     G = nx.convert_node_labels_to_integers(G)
    #     graphs = []
    #     for i in range(G.number_of_nodes()):
    #         G_ego = nx.ego_graph(G, i, radius=1)
    #         if (G_ego.number_of_nodes() >= 4) and (G_ego.number_of_nodes() <= 20):
    #             graphs.append(G_ego)
    #     shuffle(graphs)
    #     graphs = graphs[0:200]
    #     args.max_prev_node = 15

    # kdd graphs
    elif args.graph_type == "mutag":
        G = load_kdd_graph("mutag")
        shuffle(G)

        args.max_prev_node = 10
        return G

    elif args.graph_type == "proteins":
        G = load_kdd_graph("proteins")
        shuffle(G)

        args.max_prev_node = 85
        return G

    return graphs
