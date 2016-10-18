from __future__ import division
import networkx as nx
import numpy as np
from sktensor import sptensor, dtensor, ktensor, cp_als
import ncp

class TensorFact:
    def __init__(self, graphs, num_of_coms):
        self.thres = 0
        self.node_ids = list(set([node for i in graphs for node in nx.nodes(graphs[i])]))
        # create a dict with {node_id : tensor_position} to be able to retrieve node_id
        self.node_pos = {node_id: i for i, node_id in enumerate(self.node_ids)}
        self.tensor = self.create_sptensor(graphs)
        """A_org = np.random.rand(len(self.node_ids), num_of_coms)
        B_org = np.random.rand(len(self.node_ids), num_of_coms)
        C_org = np.random.rand(len(graphs), num_of_coms)
        self.tensor = ktensor([A_org, B_org, C_org]).totensor()"""
        #self.tensor = self.create_dtensor(graphs)
        A, B, C = self.tensor_decomp(num_of_coms)
        self.comms = self.get_comms(A, B, C)
        self. timeline = self.get_timeline(C)
        print self.comms
        print self.timeline

    def create_dtensor(self, graphs):
        """
        Create a dense tensor
        :param graphs:
        :return:
        """
        n = 0
        nodes_id = {}
        n = len(self.node_ids)
        s = len(graphs)
        tensor = np.zeros((n, n, s), dtype='float32')
        for i in range(s):
            for u, v in graphs[i+1].edges_iter():
                tensor[self.node_pos[u], self.node_pos[v], i] = 1
                tensor[self.node_pos[v], self.node_pos[u], i] = 1
        return dtensor(tensor)


    def create_sptensor(self, graphs):
        """
        Create a sparse tensor
        :param graphs:
        :return:
        """
        tuples = []
        # triplets = np.array([(u, v, t) for t in range(1, len(graphs)+1) for u, v in graphs[i].edges_iter()] +
                       # [(v, u, t) for t in range(1, len(graphs)+1) for u, v in graphs[i].edges_iter()])
        for i, graph in graphs.iteritems():
            for u, v in graph.edges_iter():
                tuples.append([self.node_pos[u], self.node_pos[v], i-1])
                tuples.append([self.node_pos[v], self.node_pos[u], i-1])
        triplets = np.array([(u, v, t) for u, v, t in tuples])
        T = sptensor(tuple(triplets.T), vals=np.ones(len(triplets)), shape=(len(self.node_ids), len(self.node_ids),
                                                                            len(graphs)))
        return T

    def tensor_decomp(self, num_of_coms):
        # setting seed in order to reproduce experiment
        np.random.seed(4)
        X_approx_ks = ncp.nonnegative_tensor_factorization(self.tensor, num_of_coms, method='anls_bpp',
                                                           stop_criterion=2)
        A = X_approx_ks.U[0]
        B = X_approx_ks.U[1]
        C = X_approx_ks.U[2]
        error =(self.tensor - X_approx_ks.totensor()).norm() / self.tensor.norm()
        print error
        return A, B, C

    def get_comms(self, A, B ,C):
        """
        Return communities in a dict of the form { com_id : [list with node_ids that
        belong to the community}
        :param A:
        :param B:
        :param C:
        :return:
        """
        comms ={}
        for u in range(A.shape[0]):
            for c in range(A.shape[1]):
                if A[u, c] > self.thres:
                    try:
                        comms[c+1].append(self.node_ids[u])
                    except KeyError:
                        comms[c+1] = [self.node_ids[u]]
        return comms

    def get_timeline(self, C):
        timeline = {}
        for t in range(C.shape[0]):
            for c in range(C.shape[1]):
                if C[t, c] > self.thres:
                    try:
                        timeline[t+1].append(c+1)
                    except KeyError:
                        timeline[t+1] = [c+1]
        return timeline


if __name__ == '__main__':
    edges_old = {
        0: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)],
        1: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7), (7, 8)],
        2: [(1, 2), (5, 6), (5, 8)]
    }
    edges = {
        0: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)],
        1: [(11, 12), (11, 13), (12, 13)],
        2: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)]
    }
    graphs = {}
    for i, edges in edges.items():
        graphs[i+1] = nx.Graph(edges)
    fact = TensorFact(graphs, 2)


