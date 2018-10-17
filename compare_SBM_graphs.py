import networkx
from evaluate import eval_graph_lists
from utils import save_graph_list, draw_graph,load_graph_list
import itertools
import sys
import numpy as np
import os
import pandas as pd

def create_sbm_networkx(nnodes, avg_degree, ep, ncoms):

    pin = (nnodes * avg_degree / (2.0)) / ((nnodes / float(ncoms)) * (nnodes / float(ncoms) - 1) / 2.0 * float(ncoms) + ep * (ncoms * (ncoms - 1) / 2.0) * (
        np.power(nnodes / (ncoms * 1.0), 2.0)))
    pout=ep*pin
    prob_mat = np.identity(ncoms) * pin + (np.ones((ncoms, ncoms)) - np.identity(ncoms)) * pout
    eta=[int(nnodes / ncoms) for i in range(ncoms - 1)] #make sure these are ints
    eta+=[nnodes - sum(eta)]

    randsbm=networkx.stochastic_block_model(sizes=eta,p=prob_mat)
    return randsbm

def create_sbm_networks_list(num2create,nnodes,avg_degree,ep,ncoms):
    outlist=[]
    for i in range(num2create):
        outlist+=[create_sbm_networkx(nnodes,avg_degree,ep,ncoms)]
    return outlist


def main():

    num2create=200
    nnodes=2000
    c=10
    ep=.1

    sbm_list_outfiles=[]
    main_sbm_dir=os.path.join(".","dataset","SBM_sets")
    if not os.path.exists(main_sbm_dir):
        os.makedirs(main_sbm_dir)

    results_dir=os.path.join(".","results","SBM_tests")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    all_graph_lists={}

    mmd_df = pd.DataFrame(columns=['q1', 'q2', 'degree', 'clustering', 'orbits'])
    q_vals=[2,3,4,5]
    for q in q_vals:
        print("creating SBMs with {:d} coms".format(q))
        sbm_list_outfile=os.path.join(main_sbm_dir,"sbm_num{:d}_nnodes{:d}_c{:.3f}_ep{:.3f}_q{:d}.dat".format(num2create,
                                                                                                  nnodes,c,ep,q))
        cgraphlist=create_sbm_networks_list(num2create,nnodes,c,ep,q)
        all_graph_lists[q]=cgraphlist
        save_graph_list(cgraphlist,sbm_list_outfile)
        sbm_list_outfiles.append(sbm_list_outfile)
        #draw_graph_list(G_list=cgraphlist,row=4,col=3,fname=sbm_list_outfile)
        draw_graph(cgraphlist[0],"sbm_{:d}".format(q))


    mmd_df=pd.DataFrame(columns=['q1','q2','degree','clustering','orbits'])
    for q1,q2 in itertools.combinations_with_replacement(all_graph_lists.keys(),2):
        glist1=all_graph_lists[q1]
        if q2!=q1:
            glist2=all_graph_lists[q2]
        else: #create new list for self comparison
            glist2=create_sbm_networks_list(num2create,nnodes,c,ep,q2)
        mmd_degree, mmd_clustering, mmd_4orbits=eval_graph_lists(glist1,glist2)
        cind=mmd_df.shape[0]
        mmd_df.loc[cind,:]=[q1,q2,mmd_degree, mmd_clustering, mmd_4orbits]

    #print(mmd_df)
    mmd_df.to_csv((os.path.join(results_dir,"sbm_mmd_results_num{:d}_nnodes{:d}_c{:.3f}_ep{:.3f}.csv".format(
        num2create,nnodes,c,ep))))

    return 0

if __name__=="__main__":
    sys.exit(main())