from __future__ import division
import networkx as nx
import numpy as np
import pandas as pd
from sktensor import sptensor
from copy import deepcopy, copy
from scipy import sparse
from sklearn.cluster import spectral_clustering


class Muturank:
    def __init__(self, graphs, threshold, alpha, beta):
        self.graphs = graphs
        self.node_ids = list(set([node for i in graphs for node in nx.nodes(graphs[i])]))
        # create a dict with {node_id : tensor_position} to be able to retrieve node_id
        self.node_pos = {node_id: i for i, node_id in enumerate(self.node_ids)}
        # self.a, self.o, self.r = self.create_dtensors(graphs)
        self.a, self.o, self.r, self.sum_row, self.sum_time = self.create_sptensors()
        self.e = threshold
        self.alpha = alpha
        self.beta = beta
        self.run_muturank()
        self.create_monorelational()
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
        # TODO: add edges between timeframes
        for i, (t, graph) in enumerate(graphs.iteritems()):
            for u, v in graph.edges_iter():
                tuples.append([self.node_pos[u], self.node_pos[v], i])
                tuples.append([self.node_pos[v], self.node_pos[u], i])
        triplets = np.array([(u, v, t) for u, v, t in tuples])
        a = sptensor(tuple(triplets.T), vals=np.ones(len(triplets)), shape=(len(self.node_ids),
                                                                            len(self.node_ids),
                                                                            len(graphs)))
        o_values = []
        sum_rows = np.zeros((a.shape[0], a.shape[2]))
        for t in range(a.shape[2]):
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    # TODO : just add another for loop instead of : to access .sum()
                    # TODO : check sparse tensor performance and library
                    sum_rows[i, t] += a[i, j, t]
            for i in range(a.shape[0]):
                if sum_rows[i, t] != 0:
                    for j in range(i):
                        if a[i, j, t] != 0:
                            o_values.append(a[j, i, t]/sum_rows[j, t])
                            o_values.append(a[i, j, t]/sum_rows[i, t])
        o = sptensor(tuple(triplets.T), vals=o_values, shape=(len(self.node_ids),
                                                              len(self.node_ids),
                                                              len(graphs)))
        r_values = []
        sum_time = np.zeros((a.shape[0], a.shape[1]))
        for i in range(a.shape[0]):
            # OPTIMIZE: sum is a dense matrix/array. Should be sparse for memory
            for j in range(a.shape[1]):
                for t in range(a.shape[2]):
                    # TODO : just add another for loop instead of : to access .sum()
                    # TODO : check sparse tensor performance and library
                    if a[i, j, t] != 0:
                        sum_time[i, j] += a[i, j, t]
        for t in range(a.shape[2]):
            for i in range(a.shape[0]):
                for j in range(i):
                    if a[j, i, t] != 0:
                        r_values.append(a[j, i, t]/sum_time[j, i])
                        r_values.append(a[i, j, t]/sum_time[i, j])
        r = sptensor(tuple(triplets.T), vals=r_values, shape=(len(self.node_ids),
                                                              len(self.node_ids),
                                                              len(graphs)))
        return a, o, r, sum_rows, sum_time

    def create_dtensors(self, graphs):
        """
        construct two transition probability tensors O =[o i,j,d] and R =[r i,j,d]

        :param graphs:
        :return:
        """
        n = 0
        # set the number of nodes to the biggest graph found in the timeframes
        n = len(self.node_ids)
        s = len(graphs)
        a = np.zeros((s, n+1, n+1))
        for i, node in enumerate(self.node_ids, 1):
            a[:, i, 0] = node
            a[:, 0, i] = node
        for i in range(s):
            for u, v in graphs[i].edges_iter():
                a[i, u, v] = 1
                a[i, v, u] = 1
        o = np.copy(a)
        r = np.copy(a)
        for t in range(0, o.shape[0]):
            for j in range(1, o.shape[2]):
                for i in range(1, o.shape[1]):
                    if a[t, 1:, j].sum() != 0:
                        o[t, i, j] = o[t, i, j]/(a[t, 1:, j].sum())
        for i in range(1, r.shape[1]):
            for j in range(1, r.shape[2]):
                for t in range(0, r.shape[0]):
                    if a[:, i, j].sum() != 0:
                        r[t, i, j] = r[t, i, j]/(a[:, i, j].sum())
        return a, o, r

    def prob_t(self, d, j):
        # OPTIMIZE: calculate denominator once for both probabilities
        p = (self.q_new[d]*self.sum_row[j, d])/sum([self.q_new[d]*self.a[j, l, m]
                                                    for l in range(len(self.node_ids))
                                                    for m in range(len(self.graphs))])
        return p

    def prob_n(self, i, j):
        # OPTIMIZE: calculate denominator once for both probabilities
        p = sum([self.q_new[m]*self.a[i, j, m] for m in range(len(self.graphs))])/sum([self.q_new[m]*self.a[j, l, m]
                                                                                  for l in range(len(self.node_ids))
                                                                                  for m in range(len(self.graphs))])
        return p




    def run_muturank(self):
        """
        Input:
            A:      the affinity tensor
            e:      the convergence threshold
            p*,q*:  two prior distributions
            a,b:    two balancing parameters
        Output:
            p,q:    two equilibrium distributions
        :return:
        """
        t = 0
        # p_star = prior
        # p_old
        # p_new

        # initializing p_star and q_star with random probabilities
        # TODO: p* and q* should be 1/N and 1/m (the same goes for p0 and q0
        #p_star = np.random.dirichlet(np.ones(len(self.node_ids)))
        #q_star = np.random.dirichlet(np.ones(len(self.graphs)))
        p_star = [1/len(self.node_ids) for node in self.node_ids]
        q_star = [1/len(self.graphs) for tf in self.graphs]
        """
        alternatively we set q_star and q_star equal to 1/n
        p_star = np.ones(len(self.node_ids))/len(self.node_ids)
        q_star = np.ones(len(self.graphs))/len(self.graphs)
        """
        self.p_new= np.ones((len(p_star)))
        self.q_new = np.ones((len(q_star)))
        self.p_old = np.zeros((len(p_star)))
        self.q_old = np.zeros((len(q_star)))
        # while ||p(t)-p(t-1)||^2 + ||q(t) - q(t-1||^2 >=e
        while np.linalg.norm(self.p_new-self.p_old)**2 + np.linalg.norm(self.q_new-self.q_old)**2 > self.e:
            self.p_old = copy(self.p_new)
            self.q_old = copy(self.q_new)
            for i in range(len(self.node_ids)):
                self.p_new[i] = self.alpha *\
                               sum([self.p_old[j]*self.o[i, j, m]*self.prob_t(m, j)
                                    for j in range(len(self.node_ids))
                                    for m in range(len(self.graphs))])+(1-self.alpha)*p_star[i]
            for d in range(len(self.graphs)):
                self.q_new[d] = self.beta *\
                                sum([self.p_old[j]*self.r[i, j, d]* self.prob_n(i, j)
                                     for i in range(len(self.node_ids))
                                     for j in range(len(self.node_ids))])+(1-self.beta)*q_star[d]
            t += 1
        """checking the calculation of probabilities
        for j in range(len(self.node_ids)):
            print sum([self.prob_n(i, j) for i in range(len(self.node_ids))])

        for j in range(len(self.node_ids)):
            print sum([self.prob_t(d, j) for d in range(len(self.graphs))])"""
        return

    def create_monorelational(self):
        self.w = sparse.eye(len(self.node_ids), dtype=np.float32,format="dok")
        for i in range(len(self.node_ids)):
            for j in range((len(self.node_ids))):
                value = sum([self.q_new[d]*self.a[i, j, d] for d in range(len(self.graphs))])
                if value:
                    self.w[i, j] = value

    def clustering(self):
        clusters = spectral_clustering(self.w, n_clusters=2, n_init=10, eigen_solver='arpack')
        print clusters

    def create_dataframes(self, tensor):
        dataframes = {}
        for i in range(tensor.shape[0]):
            pass
        pd.DataFrame(data=tensor[1:, 1:],    # values
                     index=tensor[1:, 0],    # 1st column as index
                     columns=tensor[0, 1:])


if __name__ == '__main__':
    edges = {
        0: [(1, 3), (1, 4), (2, 4)],
        1: [(1, 4), (3, 4), (1, 2)]
    }
    """
    edges = {
    0: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)],
    1: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7), (7, 8)],
    2: [(1, 2), (5, 6), (5, 8)]
    }edges = {
        0: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)],
        1: [(11, 12), (11, 13), (12, 13)],
        2: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)]
    }"""
    graphs = {}
    for i, edges in edges.items():
        graphs[i] = nx.Graph(edges)
    mutu = Muturank(graphs, 1e-6, 0.85, 0.85)
    #print mutu.a[mutu.node_pos[1],mutu.node_pos[4],1]
    #print mutu.r
