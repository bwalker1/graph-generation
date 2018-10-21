import networkx as nx
from matplotlib import pyplot as plt
import random
import numpy as np

def generateRandomBlockMatrix(blockSize, probIntraCommunityRange = [0.8, 1.0], probInterCommunityRange = [0.01, 0.04]):
    # This function generates a random block matrix P

    # Input Parameters:
    # blockSize: the number of blocks(community) in the graph
    # probIntraCommunityRange: probability range of edges within the each community
    # probInterCommunityRange: probability range of edges across communities

    probIntraCommunity = np.diag((probIntraCommunityRange[1] - probIntraCommunityRange[0]) * np.random.random_sample(blockSize) + probIntraCommunityRange[0])
    probInterCommunity = ((probInterCommunityRange[1] - probInterCommunityRange[0]) * np.random.random_sample((blockSize, blockSize)) + probInterCommunityRange[0]) * (1 - np.diag(np.ones(blockSize)))
    P = probIntraCommunity + probInterCommunity
    P = (P + np.transpose(P))/2 # make it symmetric
    return P.tolist()


def generateSingleSBM(block, P):
    # generates a single SBM from given block sizes and block matrix P

    # Input Parameters:
    # block is an array of sizes of each block(community)
    # P is the block matrix
    g = nx.generators.community.stochastic_block_model(block, P, seed = 0)
    return g


def generateSetOfSBM(blockSizes):
    # Generates a collection of graph

    # Input Parameters:
    # blockSizes: This is a list of list. Each list contains block sizes of communities for a graph.
    G = []
    labelSet = [len(block) for block in blockSizes]
    maxLabel = max(labelSet)

    for block in blockSizes:
        P = generateRandomBlockMatrix(len(block))
        g = generateSingleSBM(block, P)
        z = [0] * maxLabel
        z[len(block) - 1] = 1
        g.graph['Z'] = z
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
