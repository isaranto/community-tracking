from __future__ import division
from muturank import Muturank_new
from synthetic import SyntheticDataConverter
from metrics import NMI
import networkx as nx
from itertools import combinations_with_replacement
import random
from tensor import TensorFact

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
    q = [1/mutu.tfs for i in range(mutu.tfs)]
    mutu.w = mutu.create_monorelational(mutu.q_new)
    mutu.dynamic_com = mutu.clustering()
    ground_truth = {i: [] for i in range(len(sd.communities.itervalues().next()))}
    for tf, coms in sd.communities.iteritems():
        for i, com in coms.iteritems():
            for node in com:
                ground_truth[i].append(str(node)+"-t"+str(tf))
    nmi_mutu = NMI(ground_truth, mutu.dynamic_com, evaluation_type="dynamic").results
    fact = TensorFact(sd.graphs, num_of_coms=5, seeds=1, threshold=1e-4)
    nmi_gauvin = NMI(ground_truth, fact.dynamic_coms, evaluation_type="dynamic").results
    print "muturank_nmi ", nmi_mutu
    print "gauvin_nmi ", nmi_gauvin
