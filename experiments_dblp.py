from __future__ import division
from muturank import Muturank_new
from synthetic import SyntheticDataConverter
from metrics import evaluate
from dblp import dblp_loader
import networkx as nx
from itertools import combinations_with_replacement
import random
from tensor import TensorFact
import pickle
from collections import OrderedDict
import time
import json
from tabulate import tabulate
import datetime
import pprint



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
        dynamic_coms = {int(id): [str(node) for node in com] for id, com in obj[num]['dynamic_truth'].iteritems()}
        return Data(comms, graphs, len(graphs), len(dynamic_coms), dynamic_coms)
    return obj


# def evaluate(results, ground_truth, method, name):
#     nmi = NMI(ground_truth, method.dynamic_coms, evaluation_type="sets").results
#     omega = Omega(ground_truth, method.dynamic_coms)
#     bcubed = Bcubed(ground_truth, method.dynamic_coms)
#     results["Method"].append(name)
#     results['NMI'].append(nmi['NMI<Max>'])
#     results['Omega'].append(omega.omega_score)
#     results['Bcubed-Precision'].append(bcubed.precision)
#     results['Bcubed-Recall'].append(bcubed.recall)
#     results['Bcubed-F1'].append(bcubed.fscore)
#     return results


def run_experiments(data, ground_truth, network_num):
    all_res = []
    # # Muturank with one connection - default q
    # mutu4 = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='one',
    #                      clusters=len(ground_truth), default_q=True)
    # all_res.append(
    #     evaluate.get_results(ground_truth, mutu4.dynamic_coms, "Muturank with one connection - default q", mutu4.tfs,
    #                          eval="dynamic", duration=mutu4.duration))
    # all_res.append(
    #     evaluate.get_results(ground_truth, mutu4.dynamic_coms, "Muturank with one connection - default q", mutu4.tfs,
    #                          eval="sets", duration=mutu4.duration))
    # all_res.append(
    #     evaluate.get_results(ground_truth, mutu4.dynamic_coms, "Muturank with one connection - default q", mutu4.tfs,
    #                          eval="per_tf", duration=mutu4.duration))
    # f = open(results_file, 'a')
    # f.write(tabulate(all_res, headers="keys", tablefmt="fancy_grid").encode('utf8') + "\n")
    # f.close()
    # Muturank with all connections - default q
    mutu5 = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='all',
                         clusters=len(ground_truth), default_q=True)
    all_res.append(evaluate.get_results(ground_truth, mutu5.dynamic_coms, "Muturank with all connections - default q"
                                        , mutu5.tfs,
                                        eval="dynamic", duration=mutu5.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu5.dynamic_coms, "Muturank with all connections - default "
                                                                          "q", mutu5.tfs,
                                        eval="sets", duration=mutu5.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu5.dynamic_coms, "Muturank with all connections - default "
                                                                          "q", mutu5.tfs,
                                        eval="per_tf", duration=mutu5.duration))
    f = open(results_file, 'a')
    f.write(tabulate(all_res, headers="keys", tablefmt="fancy_grid").encode('utf8') + "\n")
    f.close()
    # Muturank with next connection - default q
    mutu6 = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='next',
                         clusters=len(ground_truth), default_q=True)
    all_res.append(evaluate.get_results(ground_truth, mutu6.dynamic_coms, "Muturank with next connection - default q"
                                        , mutu6.tfs, eval="dynamic", duration=mutu6.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu6.dynamic_coms, "Muturank with next connection - default q",
                                        mutu6.tfs, eval="sets", duration=mutu6.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu6.dynamic_coms, "Muturank with next connection - default q",
                                        mutu6.tfs, eval="per_tf", duration=mutu6.duration))
    f = open(results_file, 'a')
    f.write(tabulate(all_res, headers="keys", tablefmt="fancy_grid").encode('utf8') + "\n")
    f.close()
    # NNTF
    fact = TensorFact(data.graphs, num_of_coms=len(ground_truth), threshold=1e-4, seeds=10, overlap=False)
    all_res.append(evaluate.get_results(ground_truth, fact.dynamic_coms, "NNTF", mutu6.tfs, eval="dynamic",
                                        duration=fact.duration))
    all_res.append(evaluate.get_results(ground_truth, fact.dynamic_coms, "NNTF", mutu6.tfs, eval="sets",
                                        duration=fact.duration))
    all_res.append(evaluate.get_results(ground_truth, fact.dynamic_coms, "NNTF", mutu6.tfs, eval="per_tf",
                                        duration=fact.duration))
    f = open(results_file, 'a')
    f.write(tabulate(all_res, headers="keys", tablefmt="fancy_grid").encode('utf8') + "\n")
    f.close()
    with open(results_file, 'a') as f:
        f.write("NNTF\n")
        f.write("Error: " + str(fact.error) + "Seed: " + str(fact.best_seed)+"\n")
        f.write("A\n")
        pprint.pprint(fact.A, stream=f, width=150)
        f.write("B\n")
        pprint.pprint(fact.B, stream=f, width=150)
        f.write("C\n")
        pprint.pprint(fact.C, stream=f, width=150)
        pprint.pprint(fact.dynamic_coms, stream=f, width=150)
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
    outfile = './results/GED-events-handdrawn-'+str(network_num)+'.csv'
    with open(outfile, 'w')as f:
        for hypergraph in tracker.hypergraphs:
            hypergraph.calculateEvents(f)
    ged_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print "--- %s seconds ---" % (ged_time)
    ged = ReadGEDResults.ReadGEDResults(file_coms=ged_data.fileName, file_output=outfile)
    with open(results_file, 'a') as f:
        f.write("GED\n")
        pprint.pprint(ged.dynamic_coms, stream=f, width=150)
    all_res.append(evaluate.get_results(ground_truth, ged.dynamic_coms, "GED", mutu6.tfs, eval="dynamic", duration=ged_time))
    all_res.append(evaluate.get_results(ground_truth, ged.dynamic_coms, "GED", mutu6.tfs, eval="sets", duration=ged_time))
    f = open(results_file, 'a')
    f.write(tabulate(all_res, headers="keys", tablefmt="fancy_grid").encode('utf8') + "\n")
    f.close()
    #all_res.append(evaluate.get_results(ground_truth, ged.dynamic_coms, "GED", mutu6.tfs, eval="per_tf"))
    # # Run muturank - One connection
    # mutu1 = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='one',
    #                      clusters=len(ground_truth), default_q=False)
    # all_res.append(evaluate.get_results(ground_truth, mutu1.dynamic_coms, "Muturank with one connection", mutu1.tfs,
    #                                     eval="dynamic", duration=mutu1.duration))
    # all_res.append(evaluate.get_results(ground_truth, mutu1.dynamic_coms, "Muturank with one connection", mutu1.tfs,
    #                                     eval="sets", duration=mutu1.duration))
    # all_res.append(evaluate.get_results(ground_truth, mutu1.dynamic_coms, "Muturank with one connection", mutu1.tfs,
    #                                     eval="per_tf", duration=mutu1.duration))
    # f = open(results_file, 'a')
    # f.write(tabulate(all_res, headers="keys", tablefmt="fancy_grid").encode('utf8') + "\n")
    # f.close()
    # muturank_res = OrderedDict()
    # muturank_res["tf/node"] = ['t' + str(tf) for tf in mutu1.tfs_list]
    # for i, node in enumerate(mutu1.node_ids):
    #     muturank_res[node] = [mutu1.p_new[tf * len(mutu1.node_ids) + i] for tf in range(mutu1.tfs)]
    # f = open(results_file, 'a')
    # f.write("ONE CONNECTION\n")
    # f.write(tabulate(muturank_res, headers="keys", tablefmt="fancy_grid").encode('utf8') + "\n")
    # f.write(tabulate(zip(['t' + str(tf) for tf in mutu1.tfs_list], mutu1.q_new), headers="keys",
    #                  tablefmt="fancy_grid").encode('utf8') + "\n")
    # f.close()

    # Muturank with all connections
    mutu2 = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='all',
                         clusters=len(ground_truth), default_q=False)
    all_res.append(evaluate.get_results(ground_truth, mutu2.dynamic_coms, "Muturank with all connections", mutu2.tfs,
                                        eval="dynamic", duration=mutu2.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu2.dynamic_coms, "Muturank with all connections", mutu2.tfs,
                                        eval="sets", duration=mutu2.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu2.dynamic_coms, "Muturank with all connections", mutu2.tfs,
                                        eval="per_tf", duration=mutu2.duration))
    f = open(results_file, 'a')
    f.write(tabulate(all_res, headers="keys", tablefmt="fancy_grid").encode('utf8') + "\n")
    f.close()
    muturank_res = OrderedDict()
    muturank_res["tf/node"] = ['t' + str(tf) for tf in mutu2.tfs_list]
    for i, node in enumerate(mutu2.node_ids):
        muturank_res[node] = [mutu2.p_new[tf * len(mutu2.node_ids) + i] for tf in range(mutu2.tfs)]
    f = open(results_file, 'a')
    f.write("ALL CONNECTIONS\n")
    f.write(tabulate(muturank_res, headers="keys", tablefmt="fancy_grid").encode('utf8') + "\n")
    f.write(tabulate(zip(['t' + str(tf) for tf in mutu2.tfs_list], mutu2.q_new), headers="keys",
                     tablefmt="fancy_grid").encode('utf8') + "\n")
    f.close()
    # Muturank with next connection
    mutu3 = Muturank_new(data.graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='next',
                         clusters=len(ground_truth), default_q=False)
    all_res.append(evaluate.get_results(ground_truth, mutu3.dynamic_coms, "Muturank with next connection", mutu3.tfs,
                                        eval="dynamic", duration=mutu3.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu3.dynamic_coms, "Muturank with next connection", mutu3.tfs,
                                        eval="sets", duration=mutu3.duration))
    all_res.append(evaluate.get_results(ground_truth, mutu3.dynamic_coms, "Muturank with next connection", mutu3.tfs,
                                        eval="per_tf", duration=mutu3.duration))
    f = open(results_file, 'a')
    f.write(tabulate(all_res, headers="keys", tablefmt="fancy_grid").encode('utf8') + "\n")
    f.close()
    muturank_res = OrderedDict()
    muturank_res["tf/node"] = ['t' + str(tf) for tf in mutu3.tfs_list]
    for i, node in enumerate(mutu3.node_ids):
        muturank_res[node] = [mutu3.p_new[tf * len(mutu3.node_ids) + i] for tf in range(mutu3.tfs)]
    f = open(results_file, 'a')
    f.write("NEXT CONNECTION\n")
    f.write(tabulate(muturank_res, headers="keys", tablefmt="fancy_grid").encode('utf8') + "\n")
    f.write(tabulate(zip(['t' + str(tf) for tf in mutu3.tfs_list], mutu3.q_new), headers="keys",
                     tablefmt="fancy_grid").encode('utf8') + "\n")
    f.write("GROUND TRUTH\n")
    pprint.pprint(ground_truth, stream=f, width=150)
    # f.write("ONE CONNECTION\n")
    # pprint.pprint(mutu1.dynamic_coms, stream=f, width=150)
    f.write("ALL CONNECTIONS\n")
    pprint.pprint(mutu2.dynamic_coms, stream=f, width=150)
    f.write("NEXT CONNECTION\n")
    pprint.pprint(mutu3.dynamic_coms, stream=f, width=150)
    f.close()
    return all_res


if __name__ == "__main__":
    from os.path import expanduser
    home = expanduser("~")
    path_full = home+"/Dropbox/Msc/thesis/data/dblp/1"
    results_file = "results_dblp_"+path_full.split("/")[-1]+".txt"
    print results_file
    dblp = dblp_loader(path_full+"/my_dblp_data.json", conf_file=path_full+"/confs.txt", start_year=2006,
                       end_year=2010, coms='comp')
    # for t in dblp.graphs.keys():
    #     print nx.number_of_nodes(dblp.graphs[t]),
    #     new_edges = []
    #     for conf, graph in dblp.conf_graphs[t].iteritems():
    #         new_edges.extend(graph.edges())
    #     print nx.number_of_nodes(nx.Graph(new_edges))
    # print(len(set(item for sublist in [g.nodes() for _, g in dblp.graphs.items()] for item in sublist)))
    for tf, year in enumerate(dblp.communities.keys()):
        print len(dblp.all_nodes[tf]), len(set([node for _, com in dblp.communities[year].iteritems() for node in com])), nx.number_of_nodes(dblp.graphs[tf])
        # print(len([node for _,com in dblp.communities[dblp.communities.keys()[t]].iteritems() for node in com])),
        # new_edges = []
        # for conf, graph in dblp.conf_graphs[dblp.conf_graphs.keys()[t]].iteritems():
        #     new_edges.extend(graph.edges())
        # print nx.number_of_nodes(nx.Graph(new_edges))
    print "Total",
    print(len([node for t in dblp.graphs.keys() for _, com in dblp.communities[dblp.communities.keys()[t]].iteritems() for node in com])),
    print(len(set(item for sublist in [g.nodes() for _, g in dblp.graphs.items()] for item in sublist)))
    print "# of dynamic communities", len(dblp.dynamic_coms)
    number_of_dynamic_communities = len(dblp.dynamic_coms)
    data = Data(dblp.communities, dblp.graphs, len(dblp.graphs), len(dblp.dynamic_coms), dblp.dynamic_coms)
    pprint.pprint(data.graphs.keys())
    #from plot import PlotGraphs
    #PlotGraphs(data.graphs, len(data.graphs), 'expand-contract', 100)
    all_res = run_experiments(data, data.dynamic_truth, 'dblp')
    results = OrderedDict()
    results["Method"] = []
    results['Eval'] = []
    results['NMI'] = []
    results['Omega'] = []
    results['Bcubed-Precision'] = []
    results['Bcubed-Recall'] = []
    results['Bcubed-F1'] = []
    results['Duration'] = []
    for res in all_res:
        for k, v in res.iteritems():
            results[k].extend(v)
    f = open(results_file, 'a')
    f.write(tabulate(results, headers="keys", tablefmt="fancy_grid").encode('utf8')+"\n")
    import pandas as pd
    df = pd.DataFrame.from_dict(results)
    del df["Duration"]
    f.write("\\begin{table}[h!] \n\centering \n\\begin{tabular}{ |p{4cm}||p{2cm}|p{3cm}|p{2cm}|p{2cm}|p{2cm}|} "
            "\n"
            "\hline \n\multicolumn{6}{|c|}{Evaluation comparison} \\\\\n\hline\n Method& NMI & Omega Index & "
            "BCubed Precision & BCubed Recall & BCubed F\\\\\n\hline\n")
    for index, row in df.iterrows():
        if row["Eval"] == "dynamic":
            del row["Eval"]
            f.write(str(row[0]))
            for item in row[1:]:
                f.write(" & " + str(item))
            f.write(str("\\\\")+"\n")
    f.write("\hline\n\end{tabular}\n\caption{Comparison of DBLP experiment } }"
            "\n\label{table:results-network}\n\end{table}\n")
    f.close()

