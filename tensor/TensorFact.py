import networkx as nx
import numpy as np
from sktensor import sptensor, dtensor, cp_als
import logging
import ncp

class TensorFact:
    def __init__(self, graphs):
        self.node_ids = list(set([node for i in range(len(graphs)) for node in nx.nodes(graphs[i])]))
        # create a dict with {node_id : tensor_position} to be able to retrieve node_id
        self.node_pos = {node_id: i for i, node_id in enumerate(self.node_ids)}
        #self.tensor = self.create_tensor(graphs, timeframes)
        self.tensor = self.get_non_empties(graphs)
        self.tensor_decomp()


    def create_tensor(self, graphs):
        n = 0

        nodes_id = {}
        for i, node_id in enumerate(self.node_ids):
            nodes_id[node_id] = i
        print nodes_id
        n = len(self.node_ids)
        s = len(graphs)
        tensor = np.zeros((n, n, s), dtype='float32')

        """for i, node in enumerate(nodes, 1):
            tensor[:, i, 0] = node
            tensor[:, 0, i] = node"""
        for i in range(s):
            for u, v in graphs[i].edges_iter():
                tensor[nodes_id[u], nodes_id[v], i] = 1
                tensor[nodes_id[v], nodes_id[u], i] = 1
        print tensor[nodes_id[13], nodes_id[12], 1]
        return tensor


    def get_non_empties(self, graphs):
        tuples = []
        # triplets = np.array([(u, v, t) for t in range(1, len(graphs)+1) for u, v in graphs[i].edges_iter()] +
                       # [(v, u, t) for t in range(1, len(graphs)+1) for u, v in graphs[i].edges_iter()])
        for i, graph in graphs.iteritems():
            for u, v in graph.edges_iter():
                tuples.append([self.node_pos[u], self.node_pos[v], i])
        triplets = np.array([(u, v, t) for u, v, t in tuples])
        T = sptensor(tuple(triplets.T), vals=np.ones(len(triplets)), shape=(len(self.node_ids), len(self.node_ids),
                                                                            len(graphs)),
                     dtype=int)
        return T

    def tensor_decomp(self):
        np.random.seed(4)
        X_approx_ks = ncp.nonnegative_tensor_factorization(self.tensor, 2, method='anls_bpp', stop_criterion=2)
        A = X_approx_ks.U[0]
        B = X_approx_ks.U[1]
        C = X_approx_ks.U[2]
        print A
        print B
        print C


if __name__ == '__main__':
    edges_old = {
        0: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)],
        1: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7), (7, 8)],
        2: [(1, 2), (5, 6), (5, 8)]
    }
    edges = {
        0: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)],
        1: [(11, 12),(11,13),(12,13)],
        2: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)]
    }
    graphs = {}
    for i, edges in edges.items():
        graphs[i] = nx.Graph(edges)
    fact = TensorFact(graphs)