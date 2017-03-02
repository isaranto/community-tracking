from __future__ import division
from muturank import Muturank_new
from synthetic import SyntheticDataConverter
from metrics import NMI, Omega, Bcubed
import networkx as nx
from itertools import combinations_with_replacement
import random
from tensor import TensorFact
import pickle
from collections import OrderedDict

if __name__=="__main__":
    path_test = "/home/lias/Dropbox/Msc/thesis/src/NEW/synthetic-data-generator/src/expand/"
    path_full = "data/synthetic/expand"
    sd = SyntheticDataConverter(path_test)
    number_of_dynamic_communities = len(sd.communities[0])
    nodes = sd.graphs[0].nodes()
    edges_1 = random.sample(list(combinations_with_replacement(nodes, 2)), 50)
    edges_2 = random.sample(list(combinations_with_replacement(nodes, 2)), 207)

    dict = {
        1: sd.graphs[0],
        2: sd.graphs[0],
        3: nx.Graph(edges_1),
        4: sd.graphs[0],
        5: sd.graphs[0]
    }
    edges = {
        0: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)],
        1: [(11, 12), (11, 13), (12, 13)],
        2: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)]
    }
    my_coms = {0: [[1,2,3,4],[5,6,7]],
               1:[[11,12,13]],
            2:[[1,2,3,4],[5,6,7]]
    }
    graphs = {}
    for i, edges in edges.items():
        graphs[i] = nx.Graph(edges)
    mutu = Muturank_new(sd.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='all',
                        clusters=number_of_dynamic_communities)
    with open('muturank.pickle', 'wb') as fp:
        pickle.dump(mutu, fp, protocol=pickle.HIGHEST_PROTOCOL)
    ground_truth = {i: [] for i in range(len(sd.communities.itervalues().next()))}
    for tf, coms in sd.communities.iteritems():
        for i, com in coms.iteritems():
            for node in com:
                ground_truth[i].append(str(node)+"-t"+str(tf))
    # for tf, coms in my_coms.iteritems():
    #     for i, com in coms:
    #         for node in com:
    #             ground_truth[i].append(str(node)+"-t"+str(tf))
    nmi_mutu = NMI(ground_truth, mutu.dynamic_com, evaluation_type="dynamic").results
    omega_mutu = Omega(ground_truth, mutu.dynamic_com)
    bcubed_mutu = Bcubed(ground_truth, mutu.dynamic_com)
    with open("results.txt", "w") as fp:
        fp.write("\nMuturank_NMI_score ")
        for key, val in nmi_mutu.items():
            fp.write(str(key) +" : " +str(val)+" ")
        fp.write("\nMuturank_Omega_score ")
        fp.write(" Omega = "+ str(omega_mutu.omega_score))
        fp.write("\nMuturank_Bcubed_score ")
        fp.write(" Precision = "+ str(bcubed_mutu.precision))
        fp.write(" Recall = "+ str(bcubed_mutu.recall))
        fp.write(" Fscore = "+ str(bcubed_mutu.fscore))
    fact = TensorFact(sd.graphs, num_of_coms=number_of_dynamic_communities, threshold=1e-4, seeds=1000)
    with open('gauvin.pickle', 'wb') as fp:
        pickle.dump(fact, fp, protocol=pickle.HIGHEST_PROTOCOL)
    nmi_gauvin = NMI(ground_truth, fact.dynamic_coms, evaluation_type="dynamic").results
    omega_gauvin = Omega(ground_truth, fact.dynamic_coms)
    bcubed_gauvin = Bcubed(ground_truth, fact.dynamic_coms)
    with open("results.txt", "a") as fp:
        fp.write("-------------------------------------------------------------------------------")
        fp.write("\nGauvin_NMI_score ")
        for key, val in nmi_gauvin.items():
            fp.write(str(key) +" : " +str(val)+" ")
        fp.write("\nGauvin_Omega_score ")
        fp.write(" Omega = "+ str(omega_gauvin.omega_score))
        fp.write("\nGauvin_Bcubed_score ")
        fp.write(" Precision = "+ str(bcubed_gauvin.precision))
        fp.write(" Recall = "+ str(bcubed_gauvin.recall))
        fp.write(" Fscore = "+ str(bcubed_gauvin.fscore))

    print ground_truth
    print fact.dynamic_coms