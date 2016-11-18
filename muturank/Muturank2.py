from __future__ import division
import networkx as nx
import numpy as np
import pandas as pd
from sktensor import sptensor
from copy import deepcopy, copy
from scipy import sparse
from sklearn.cluster import spectral_clustering

class Muturank2:
    def __init__(self, graphs, threshold, alpha, beta):
        self.graphs = graphs
        self.node_ids = list(set([node for i in graphs for node in nx.nodes(graphs[i])]))
        self.num_of_nodes = len(self.node_ids)
        self.tfs = len(self.graphs)
        # create a dict with {node_id : tensor_position} to be able to retrieve node_id
        self.node_pos = {node_id: i for i, node_id in enumerate(self.node_ids)}
        #self.a, self.o, self.r, self.sum_row, self.sum_time = self.create_sptensors()
        self.create_sptensors()
        self.e = threshold
        self.alpha = alpha
        self.beta = beta
        self.run_muturank()
        self.w = self.create_monorelational()
        self.w = self.add_time_edges(True)
        self.clustering()
        """print sum(self.p_new)
        print sum(self.q_new)
        print(len(self.p_new))
        print(len(self.q_new))"""
        # self.tensor= self.create_dense_tensors(graphs)
        # self.frame = self.create_dataframes(self.tensor)

    def create_sptensors(self):
        """
            Create a sparse tensor
            :param :
            :return:
            """
        tuples = []
        a = {}
        for i, (t, graph) in enumerate(self.graphs.iteritems()):
            #a[i] = sparse.csr_matrix((self.num_of_nodes*self.tfs, self.num_of_nodes*self.tfs), dtype=np.float32)
            a[i] = sparse.eye(self.num_of_nodes*self.tfs, dtype=np.float32,format="dok")
            for u, v in graph.edges_iter():
                # add self edges for nodes that exist
                a[i][i*self.num_of_nodes + self.node_pos[u], i*self.num_of_nodes + self.node_pos[u]] = 1
                a[i][i*self.num_of_nodes + self.node_pos[v], i*self.num_of_nodes + self.node_pos[v]] = 1
                # add edges - create symmetric matrix
                a[i][i*self.num_of_nodes + self.node_pos[u], i*self.num_of_nodes + self.node_pos[v]] = 1
                a[i][i*self.num_of_nodes + self.node_pos[v], i*self.num_of_nodes + self.node_pos[u]] = 1
        # add time edges
        a = self.add_time_edges(a, 'one')
        print a[1].toarray()
        o = deepcopy(a)
        r = deepcopy(a)
        sum_rows = sparse.csr_matrix((self.num_of_nodes*self.tfs, self.tfs), dtype=np.float32)
        for t in range(self.tfs):
            for i in range(self.num_of_nodes*self.tfs):
                #sum_rows[i, t] = a[t].sum(1)[i]
                pass
        for t in range(self.tfs):
            for i in range(self.num_of_nodes*self.tfs):
                sum_rows[i, t] = a[t].sum(1)[i]
                for j in range(i+1):
                    if a[t][i, j] != 0:
                        try:
                            # o[t][j,i] = a[t][j, i]/np.sum(a[t][j, :])
                            o[t][j, i] = a[t][j, i]/sum_rows[j, t]
                            if i != j:
                                # o[t][i, j] = a[t][i, j]/np.sum(a[t][i, :])
                                o[t][i, j] = a[t][i, j]/sum_rows[i, t]
                        except ZeroDivisionError:
                            pass
        print o[1].toarray()
        sum_time = sparse.csr_matrix((self.num_of_nodes*self.tfs, self.num_of_nodes*self.tfs), dtype=np.float32)
        for i in range(self.num_of_nodes*self.tfs):
            for j in range(self.num_of_nodes*self.tfs):
                for t in range(self.tfs):
                    if a[t][i, j] != 0:
                        sum_time[i, j] += a[t][i, j]
        for t in range(self.tfs):
            for i in range(self.num_of_nodes*self.tfs):
                for j in range(i+1):
                    if a[t][j, i] != 0:
                        r[t][j, i] = a[t][j, i]/sum_time[j, i]
                        r[t][i, j] = a[t][i, j]/sum_time[i, j]
        return a, o, r, sum_rows, sum_time

    def add_time_edges(self, a, connection):
        if connection == 'one':
            # connect only with previous and next timeframe
            for t in range(self.tfs):
                for i in range(a[t].shape[0]):
                    try:
                        if i < self.num_of_nodes:
                            a[t][i, i + self.num_of_nodes] = 1
                            a[t][i + self.num_of_nodes, i] = 1
                        elif i > self.num_of_nodes*(self.tfs-1):
                            a[t][i, i - self.num_of_nodes] = 1
                            a[t][i - self.num_of_nodes, i] = 1
                        else:
                            a[t][i, i + self.num_of_nodes] = 1
                            a[t][i + self.num_of_nodes, i] = 1
                            a[t][i, i - self.num_of_nodes] = 1
                            a[t][i - self.num_of_nodes, i] = 1
                    except IndexError:
                        pass
        if connection == 'all':
            # connect only with previous and next timeframe
            for t in range(self.tfs):
                for i in range(a[t].shape[0]):
                    for m in range(self.tfs):
                        for n in range(self.tfs):
                            try:
                                a[t][i+self.num_of_nodes*m, i+self.num_of_nodes*n] = 1
                                a[t][i+self.num_of_nodes*n, i+self.num_of_nodes*m] = 1
                            except IndexError:
                                pass

        return a


if __name__ == '__main__':
    edges = {
        0: [(1, 3), (1, 4), (2, 4)],
        1: [(1, 4), (3, 4), (1, 2)],
        2: [(1, 4), (3, 4), (1, 2)]
    }
    """
    edges = {
    0: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)],
    1: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7), (7, 8)],
    2: [(1, 2), (5, 6), (5, 8)]
    }
    edges = {
        0: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)],
        1: [(11, 12), (11, 13), (12, 13)],
        2: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)]
    }"""
    graphs = {}
    for i, edges in edges.items():
        graphs[i] = nx.Graph(edges)
    mutu = Muturank2(graphs, 1e-6, 0.85, 0.85)
    #print mutu.a[mutu.node_pos[1],mutu.node_pos[4],1]
    #print mutu.r