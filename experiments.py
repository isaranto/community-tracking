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
import time

class Data(object):
    def __init__(self, comms, graphs, timeFrames, number_of_dynamic_communities):
        self.comms = comms
        self.graphs = graphs
        self.timeFrames = timeFrames
        self.number_of_dynamic_communities = number_of_dynamic_communities

if __name__=="__main__":
    path_test = "/home/lias/Dropbox/Msc/thesis/src/NEW/synthetic-data-generator/src/expand/"
    path_full = "data/synthetic/expand"
    sd = SyntheticDataConverter(path_test)
    nodes = sd.graphs[0].nodes()
    edges_1 = random.sample(list(combinations_with_replacement(nodes, 2)), 50)
    edges_2 = random.sample(list(combinations_with_replacement(nodes, 2)), 207)

    #  ---------------------------------
    # Handwritten example
    # edges = {
    #     0: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)],
    #     1: [(11, 12), (11, 13), (12, 13)],
    #     2: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)]
    # }
    # my_coms = {0: {0: [1, 2, 3, 4], 1: [5, 6, 7]},
    #            1: {2: [11, 12, 13] },
    #            2: {0: [1, 2, 3, 4], 1: [5, 6, 7]}
    # }
    # graphs = {}
    # for i, edges in edges.items():
    #     graphs[i] = nx.Graph(edges)
    # number_of_dynamic_communities = 3
    # data = Data(my_coms, graphs, len(graphs), number_of_dynamic_communities)
    #  ---------------------------------
    # Dynamic Netword Generator data (50 nodes/3 tfs)
    number_of_dynamic_communities = len(sd.comms[0])
    data = Data(sd.comms, sd.graphs, len(sd.graphs), len(sd.comms[0]))
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
    ground_truth = {i: [] for i in range(number_of_dynamic_communities)}
    for tf, coms in data.comms.iteritems():
        for i, com in coms.iteritems():
            print tf,i,com
            for node in com:
                ground_truth[i].append(str(node)+"-t"+str(tf))
    print ground_truth
    # Run muturank - One connection
    mutu = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='one',
                        clusters=number_of_dynamic_communities, default_q=False)
    # with open('muturank.pickle', 'wb') as fp:
    #     pickle.dump(mutu, fp, protocol=pickle.HIGHEST_PROTOCOL)
    nmi_mutu = NMI(ground_truth, mutu.dynamic_com, evaluation_type="dynamic").results
    omega_mutu = Omega(ground_truth, mutu.dynamic_com)
    bcubed_mutu = Bcubed(ground_truth, mutu.dynamic_com)
    with open("results.txt", "w") as fp:
        fp.write("\n Muturank with one connection")
        fp.write("\nMuturank_NMI_score ")
        for key, val in nmi_mutu.items():
            fp.write(str(key) +" : " +str(val)+" ")
        fp.write("\nMuturank_Omega_score ")
        fp.write(" Omega = "+ str(omega_mutu.omega_score))
        fp.write("\nMuturank_Bcubed_score ")
        fp.write(" Precision = "+ str(bcubed_mutu.precision))
        fp.write(" Recall = "+ str(bcubed_mutu.recall))
        fp.write(" Fscore = "+ str(bcubed_mutu.fscore))
        fp.write("\n Q = "+ str(mutu.q_new))
    # Muturank with all connections
    mutu = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='all',
                        clusters=number_of_dynamic_communities, default_q=False)
    # with open('muturank.pickle', 'wb') as fp:
    #     pickle.dump(mutu, fp, protocol=pickle.HIGHEST_PROTOCOL)
    nmi_mutu = NMI(ground_truth, mutu.dynamic_com, evaluation_type="dynamic").results
    omega_mutu = Omega(ground_truth, mutu.dynamic_com)
    bcubed_mutu = Bcubed(ground_truth, mutu.dynamic_com)
    with open("results.txt", "a") as fp:
        fp.write("\n-------------------------------------------------------------------------------")
        fp.write("\n Muturank with all connections")
        fp.write("\nMuturank_NMI_score ")
        for key, val in nmi_mutu.items():
            fp.write(str(key) +" : " +str(val)+" ")
        fp.write("\nMuturank_Omega_score ")
        fp.write(" Omega = "+ str(omega_mutu.omega_score))
        fp.write("\nMuturank_Bcubed_score ")
        fp.write(" Precision = "+ str(bcubed_mutu.precision))
        fp.write(" Recall = "+ str(bcubed_mutu.recall))
        fp.write(" Fscore = "+ str(bcubed_mutu.fscore))
        fp.write("\n Q = "+ str(mutu.q_new))
    # Muturank with one connection - default q
    mutu = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='one',
                        clusters=number_of_dynamic_communities, default_q=True)
    # with open('muturank.pickle', 'wb') as fp:
    #     pickle.dump(mutu, fp, protocol=pickle.HIGHEST_PROTOCOL)
    nmi_mutu = NMI(ground_truth, mutu.dynamic_com, evaluation_type="dynamic").results
    omega_mutu = Omega(ground_truth, mutu.dynamic_com)
    bcubed_mutu = Bcubed(ground_truth, mutu.dynamic_com)
    with open("results.txt", "a") as fp:
        fp.write("\n-------------------------------------------------------------------------------")
        fp.write("\n Muturank with one connection - default q")
        fp.write("\nMuturank_NMI_score ")
        for key, val in nmi_mutu.items():
            fp.write(str(key) +" : " +str(val)+" ")
        fp.write("\nMuturank_Omega_score ")
        fp.write(" Omega = "+ str(omega_mutu.omega_score))
        fp.write("\nMuturank_Bcubed_score ")
        fp.write(" Precision = "+ str(bcubed_mutu.precision))
        fp.write(" Recall = "+ str(bcubed_mutu.recall))
        fp.write(" Fscore = "+ str(bcubed_mutu.fscore))
        fp.write("\n Q = "+ str(mutu.q_new))
    # Muturank with all connections - default q
    mutu = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='all',
                        clusters=number_of_dynamic_communities, default_q=True)
    # with open('muturank.pickle', 'wb') as fp:
    #     pickle.dump(mutu, fp, protocol=pickle.HIGHEST_PROTOCOL)
    nmi_mutu = NMI(ground_truth, mutu.dynamic_com, evaluation_type="dynamic").results
    omega_mutu = Omega(ground_truth, mutu.dynamic_com)
    bcubed_mutu = Bcubed(ground_truth, mutu.dynamic_com)
    with open("results.txt", "a") as fp:
        fp.write("\n-------------------------------------------------------------------------------")
        fp.write("\n Muturank with all connections - default q")
        fp.write("\nMuturank_NMI_score ")
        for key, val in nmi_mutu.items():
            fp.write(str(key) +" : " +str(val)+" ")
        fp.write("\nMuturank_Omega_score ")
        fp.write(" Omega = "+ str(omega_mutu.omega_score))
        fp.write("\nMuturank_Bcubed_score ")
        fp.write(" Precision = "+ str(bcubed_mutu.precision))
        fp.write(" Recall = "+ str(bcubed_mutu.recall))
        fp.write(" Fscore = "+ str(bcubed_mutu.fscore))
        fp.write("\n Q = "+ str(mutu.q_new))
    fact = TensorFact(data.graphs, num_of_coms=number_of_dynamic_communities, threshold=1e-4, seeds=1000)
    # with open('gauvin.pickle', 'wb') as fp:
    #     pickle.dump(fact, fp, protocol=pickle.HIGHEST_PROTOCOL)
    nmi_gauvin = NMI(ground_truth, fact.dynamic_coms, evaluation_type="dynamic").results
    omega_gauvin = Omega(ground_truth, fact.dynamic_coms)
    bcubed_gauvin = Bcubed(ground_truth, fact.dynamic_coms)
    with open("results.txt", "a") as fp:
        fp.write("\n-------------------------------------------------------------------------------")
        fp.write("\nGauvin_NMI_score ")
        for key, val in nmi_gauvin.items():
            fp.write(str(key) +" : " +str(val)+" ")
        fp.write("\nGauvin_Omega_score ")
        fp.write(" Omega = " + str(omega_gauvin.omega_score))
        fp.write("\nGauvin_Bcubed_score ")
        fp.write(" Precision = "+ str(bcubed_gauvin.precision))
        fp.write(" Recall = " + str(bcubed_gauvin.recall))
        fp.write(" Fscore = " + str(bcubed_gauvin.fscore))


    import sys
    sys.path.insert(0, '../GED/')
    import preprocessing, Tracker
    start_time = time.time()
    # graphs = preprocessing.getGraphs(sys.argv[1])
    #graphs = preprocessing.getGraphs("test_input_community_edges(pretty).json")
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
    print "read ged"
    ged = ReadGEDResults.ReadGEDResults(file_coms = ged_data.fileName, file_output = outfile)
    print "evaluate ged"
    nmi_ged = NMI(ground_truth, ged.dynamic_coms, evaluation_type="dynamic").results
    print "evaluate omega"
    omega_ged = Omega(ground_truth, ged.dynamic_coms)
    print "evaluate bcubed"
    bcubed_ged = Bcubed(ground_truth, ged.dynamic_coms)
    with open("results.txt", "a") as fp:
        fp.write("\n-------------------------------------------------------------------------------")
        fp.write("\nGED_NMI_score ")
        for key, val in nmi_ged.items():
            fp.write(str(key) +" : " +str(val)+" ")
        fp.write("\nGED_Omega_score ")
        fp.write(" Omega = " + str(omega_ged.omega_score))
        fp.write("\nGED_Bcubed_score ")
        fp.write(" Precision = "+ str(bcubed_ged.precision))
        fp.write(" Recall = " + str(bcubed_ged.recall))
        fp.write(" Fscore = " + str(bcubed_ged.fscore))
