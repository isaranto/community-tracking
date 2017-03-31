from __future__ import division
from muturank import Muturank_new
from synthetic import SyntheticDataConverter
from metrics import NMI, Omega, Bcubed
from dblp import dblp_loader
import networkx as nx
from itertools import combinations_with_replacement
import random
from tensor import TensorFact
import pickle
from collections import OrderedDict
import time
import json


class Data(object):
    def __init__(self, comms, graphs, timeFrames, number_of_dynamic_communities, dynamic_truth=[]):
        self.comms = comms
        self.graphs = graphs
        self.timeFrames = timeFrames
        self.number_of_dynamic_communities = number_of_dynamic_communities
        self.dynamic_truth = dynamic_truth


def object_decoder(obj, num):
    if 'type' in obj[num] and obj[num]['type'] == 'hand':
        edges = {int(tf): [(edge[0], edge[1]) for edge in edges] for tf, edges in obj[num]['edges'].iteritems()}
        graphs = {}
        for i, edges in edges.items():
            graphs[i] = nx.Graph(edges)
        comms = {int(tf): {int(id): com for id, com in coms.iteritems()} for tf, coms in obj[num]['comms'].iteritems() }
        print comms
        dynamic_coms = {int(id): [str(node) for node in com] for id, com in obj[num]['dynamic_truth'].iteritems()}
        return Data(comms, graphs, len(graphs), len(dynamic_coms), dynamic_coms)
    return obj


def evaluate(ground_truth, method, name):
    nmi = NMI(ground_truth, method.dynamic_coms, evaluation_type="dynamic").results
    omega = Omega(ground_truth, method.dynamic_coms)
    bcubed = Bcubed(ground_truth, method.dynamic_coms)
    with open("results.txt", "a") as fp:
        fp.write("\n-------------------------------------------------------------------------------")
        fp.write("\n "+name)
        fp.write("\nNMI_score ")
        for key, val in nmi.items():
            fp.write(str(key) +" : " +str(val)+" ")
        fp.write("\nOmega_score ")
        fp.write(" Omega = "+ str(omega.omega_score))
        fp.write("\nBcubed_score ")
        fp.write(" Precision = "+ str(bcubed.precision))
        fp.write(" Recall = "+ str(bcubed.recall))
        fp.write(" Fscore = "+ str(bcubed.fscore))
        try:
            fp.write("\n Q = "+ str(method.q_new))
        except Exception:
            pass

def run_experiments(data, ground_truth):
    # Run muturank - One connection
    mutu = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='one',
                        clusters=len(ground_truth), default_q=False)
    evaluate(ground_truth, mutu, "Muturank with one connection")

    # Muturank with all connections
    mutu = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='all',
                        clusters=len(ground_truth), default_q=False)
    evaluate(ground_truth, mutu, "Muturank with all connections")
    # Muturank with one connection - default q
    mutu = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='one',
                        clusters=len(ground_truth), default_q=True)
    evaluate(ground_truth, mutu, "Muturank with one connection - default q")
    # Muturank with all connections - default q
    mutu = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='all',
                        clusters=len(ground_truth), default_q=True)
    evaluate(ground_truth, mutu, "Muturank with all connections - default q")
    # NNTF
    fact = TensorFact(data.graphs, num_of_coms=len(ground_truth), threshold=1e-4, seeds=1)
    evaluate(ground_truth, fact, "NNTF")
    # GED
    import sys
    sys.path.insert(0, '../GED/')
    import preprocessing, Tracker
    start_time = time.time()
    from ged import GedWrite, ReadGEDResults
    ged_data = GedWrite(data)
    graphs = preprocessing.getGraphs(ged_data.fileName)
    tracker = Tracker.Tracker(graphs)
    tracker.compare_communities()
    #outfile = 'tmpfiles/ged_results.csv'
    outfile = '/home/lias/PycharmProjects/GED/dblp_ged_results.csv'
    with open(outfile, 'w')as f:
        for hypergraph in tracker.hypergraphs:
            hypergraph.calculateEvents(f)
    print "--- %s seconds ---" % (time.time() - start_time)
    ged = ReadGEDResults.ReadGEDResults(file_coms=ged_data.fileName, file_output=outfile)
    evaluate(ground_truth, ged, "GED")


def create_ground_truth(communities, number_of_dynamic_communities):
        ground_truth = {i: [] for i in range(number_of_dynamic_communities)}
        for tf, coms in communities.iteritems():
            for i, com in coms.iteritems():
                for node in com:
                    ground_truth[i].append(str(node)+"-t"+str(tf))
        return ground_truth


if __name__=="__main__":
    path_test = "/home/lias/Dropbox/Msc/thesis/src/NEW/synthetic-data-generator/src/expand/"
    path_full = "data/synthetic/expand"
    sd = SyntheticDataConverter(path_test)
    nodes = sd.graphs[0].nodes()
    edges_1 = random.sample(list(combinations_with_replacement(nodes, 2)), 50)
    edges_2 = random.sample(list(combinations_with_replacement(nodes, 2)), 207)
    #  ---------------------------------
    # Dynamic Netword Generator data (50 nodes/3 tfs)
    # number_of_dynamic_communities = len(sd.comms[0])
    # data = Data(sd.comms, sd.graphs, len(sd.graphs), len(sd.comms[0]))
    #  ---------------------------------
    #Dynamic Network Generator data (50 nodes/3 tfs) - Same network everywhere except one tf
    # dict = {
    #     0: sd.graphs[0],
    #     1: sd.graphs[0],
    #     2: sd.graphs[2],
    #     3: sd.graphs[0],
    #     4: sd.graphs[0]
    # }
    # comms = {0: sd.comms[0], 1: sd.comms[0],2: sd.comms[2], 3: sd.comms[0],4: sd.comms[0]}
    # number_of_dynamic_communities = len(sd.comms[0])
    # data = Data(comms, dict, len(dict), len(sd.comms[0]))
    # ---------------------------------
    # dblp = dblp_loader("data/dblp/my_dblp_data.json", start_year=2000, end_year=2004, coms='comp')
    # number_of_dynamic_communities = len(dblp.dynamic_coms)
    # data = Data(dblp.communities, dblp.graphs, len(dblp.graphs), len(dblp.dynamic_coms))
    # ground_truth = dblp.dynamic_coms
    # ---------------------------------
    with open("data/hand-drawn-data.json", mode='r') as fp:
        hand_drawn = json.load(fp)
    for i in range(len(hand_drawn)):
        data = object_decoder(hand_drawn, i)
        run_experiments(data, data.dynamic_truth)
