import networkx as nx
from matplotlib import pyplot as plt
import random
import numpy as np

def generateRandomBlockMatrix(blockSize, n, cin = 5, cout = 0.5):
    # This function generates a random block matrix P

    # Input Parameters:
    # blockSize: the number of blocks(community) in the graph
    # probIntraCommunityRange: probability range of edges within the each community
    # probInterCommunityRange: probability range of edges across communities
    
    #probIntraCommunity = (cin/(n/blockSize - 1)) * np.diag(np.ones(blockSize))
    nnodes=n
    ep=.02
    avg_degree=8
    ncoms=blockSize
    #this formula assumes equal block sizes.  For fixed average degree,equal number of nodes per community, and ratio pout/pin=ep
    pin = (nnodes * avg_degree / (2.0)) / ((nnodes / float(ncoms)) * (nnodes / float(ncoms) - 1) / 2.0 * float(ncoms) + ep * (ncoms * (ncoms - 1) / 2.0) * (np.power(nnodes / (ncoms * 1.0), 2.0)))
    pout=ep*pin
    prob_mat = np.identity(ncoms) * pin + (np.ones((ncoms, ncoms)) - np.identity(ncoms)) * pout
    return prob_mat
    #probInterCommunity = (cout/((n/blockSize)*(blockSize-1))) * (1 - np.diag(np.ones(blockSize)))
    #P = probIntraCommunity + probInterCommunity
    #P = (P + np.transpose(P))/2 # make it symmetric
    #return P.tolist()


def generateSingleSBM(block, P):
    # generates a single SBM from given block sizes and block matrix P

    # Input Parameters:
    # block is an array of sizes of each block(community)
    # P is the block matrix
    g = nx.generators.community.stochastic_block_model(block, P)
    return g


def generateSetOfSBM(blockSizes):
    # Generates a collection of graph

    # Input Parameters:
    # blockSizes: This is a list of list. Each list contains block sizes of communities for a graph.
    G = []
    labelSet = {len(block) for block in blockSizes}
    maxLabel = len(labelSet)
    lookup = {k:v for v,k in enumerate(labelSet)}

    for block in blockSizes:
        P = generateRandomBlockMatrix(len(block),np.sum(block))
        g = generateSingleSBM(block, P)
        z = [0] * maxLabel
        z[lookup[len(block)]] = 1
        g.graph['Z'] = np.array(z)
        G.append(g)
    return G

if __name__ == '__main__':
    nGraphs = 5
    nNodes = random.sample(range(20, 50), nGraphs) # Choosing the number of nodes in a graph
    labels = random.sample(range(2, 8), nGraphs) # Choosing number of communities
    blocks = [[nNodes[i] // labels[i]] * labels[i] for i in range(0, nGraphs)]
    print(blocks)
    G = generateSetOfSBM(blocks)

    i = 0
    for g in G:
        print('\nNumber of nodes: ', len(g))
        print('\nNumber of communities: ', labels[i])
        nx.draw(g)
        plt.show()
        i = i + 1
