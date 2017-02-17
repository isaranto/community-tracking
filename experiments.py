from __future__ import division
from muturank import Muturank_new
from synthetic import SyntheticDataConverter
from metrics import NMI
import networkx as nx
from itertools import combinations_with_replacement
import random
from tensor import TensorFact
import pickle

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
    mutu = Muturank_new(sd.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='one',
                        clusters=number_of_dynamic_communities)
    with open('muturank.pickle', 'wb') as fp:
        pickle.dump(mutu, fp, protocol=pickle.HIGHEST_PROTOCOL)
    ground_truth = {i: [] for i in range(len(sd.communities.itervalues().next()))}
    for tf, coms in sd.communities.iteritems():
        for i, com in coms.iteritems():
            for node in com:
                ground_truth[i].append(str(node)+"-t"+str(tf))
    nmi_mutu = NMI(ground_truth, mutu.dynamic_com, evaluation_type="dynamic").results
    with open("results.txt", "w") as fp:
        fp.write("muturank_nmi_score ")
        for key, val in nmi_mutu.items():
            fp.write(str(key) +" : " +str(val)+" ")
    fact = TensorFact(sd.graphs, num_of_coms=number_of_dynamic_communities, threshold=1e-4)
    with open('gauvin.pickle', 'wb') as fp:
        pickle.dump(fact, fp, protocol=pickle.HIGHEST_PROTOCOL)
    nmi_gauvin = NMI(ground_truth, fact.dynamic_coms, evaluation_type="dynamic").results
    with open("results.txt", "a") as fp:
        fp.write("\nmuturank_gauvin_score ")
        for key, val in nmi_gauvin.items():
            fp.write(str(key) +" : " +str(val)+" ")