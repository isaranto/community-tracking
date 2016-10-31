from __future__ import division
import networkx as nx
import numpy as np
import pandas as pd
from sktensor import sptensor
from copy import deepcopy


class Muturank:
    def __init__(self, graphs, threshold):
        self.graphs = graphs
        self.node_ids = list(set([node for i in graphs for node in nx.nodes(graphs[i])]))
        # create a dict with {node_id : tensor_position} to be able to retrieve node_id
        self.node_pos = {node_id: i for i, node_id in enumerate(self.node_ids)}
        # self.a, self.o, self.r = self.create_dtensors(graphs)
        self.a, self.o, self.r = self.create_sptensors()
        self.e = threshold
        # self.tensor= self.create_dense_tensors(graphs)
        # self.frame = self.create_dataframes(self.tensor)

    def create_sptensors(self):
        """
            Create a sparse tensor
            :param graphs:
            :return:
            """
        tuples = []
        # triplets = np.array([(u, v, t) for t in range(1, len(graphs)+1) for u, v in graphs[i].edges_iter()] +
        # [(v, u, t) for t in range(1, len(graphs)+1) for u, v in graphs[i].edges_iter()])
        for i, (t, graph) in enumerate(graphs.iteritems()):
            for u, v in graph.edges_iter():
                tuples.append([self.node_pos[u], self.node_pos[v], i])
                tuples.append([self.node_pos[v], self.node_pos[u], i])
        triplets = np.array([(u, v, t) for u, v, t in tuples])
        np.ones(len(triplets))
        a = sptensor(tuple(triplets.T), vals=np.ones(len(triplets)), shape=(len(self.node_ids),
                                                                            len(self.node_ids),
                                                                          len(graphs)))
        o_values = []
        for t in range(a.shape[2]):
            sum = []
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    # TODO : just add another for loop instead of : to access .sum()
                    # TODO : check sparse tensor performance and library
                    try:
                        sum[i] += a[i, j, t]
                    except IndexError:
                        sum.append(a[i, j, t])
            for i in range(a.shape[0]):
                if sum[i] != 0:
                    for j in range(i):
                        if a[i, j, t] != 0:
                            o_values.append(a[j, i, t]/sum[j])
                            o_values.append(a[i, j, t]/sum[i])
        o = sptensor(tuple(triplets.T), vals=o_values, shape=(len(self.node_ids),
                                                                            len(self.node_ids),
                                                                          len(graphs)))
        r_values = []
        sum = np.zeros((a.shape[0], a.shape[1]))
        for i in range(a.shape[0]):
            # TODO: sum is a dense matrix/array. Should be sparse for memory
            for j in range(a.shape[1]):
                for t in range(a.shape[2]):
                    # TODO : just add another for loop instead of : to access .sum()
                    # TODO : check sparse tensor performance and library
                    if a[i, j, t] != 0:
                        sum[i, j] += a[i, j, t]
        for t in range(a.shape[2]):
            for i in range(a.shape[0]):
                for j in range(i):
                    if a[j, i, t] != 0:
                        r_values.append(a[j, i, t]/sum[j, i])
                        r_values.append(a[i, j, t]/sum[i, j])
        r = sptensor(tuple(triplets.T), vals=r_values, shape=(len(self.node_ids),
                                                                            len(self.node_ids),
                                                                          len(graphs)))

        return a, o, r

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
                    if a[t, 1:, j].sum()!=0:
                        o[t, i, j] = o[t, i, j]/(a[t, 1:, j].sum())
        for i in range(1, r.shape[1]):
            for j in range(1, r.shape[2]):
                for t in range(0, r.shape[0]):
                    if a[:, i, j].sum()!=0:
                        r[t, i, j] = r[t, i, j]/(a[:, i, j].sum())
        return a, o, r

    def prob(self, i, j):
        pass

    def prob(self, d, j):
        pass


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
        p_new= np.ones((len(p_star)))
        q_new = np.ones((len(q_star)))
        p_old = np.zeros((len(p_star)))
        q_old = np.zeros((len(q_star)))
        # while ||p(t)-p(t-1)||^2 + ||q(t) - q(t-1||^2 >=e
        while np.linalg.norm(p_new-p_old)**2 + np.linalg.norm(q_new-q_old)**2 > self.e:
            p_old = p_new
            q_old = q_new
            for i in range(len(self.node_ids)):
                # TODO: calculate p_new
                p_new[i]= 0
            for d in range(len(self.graphs)):
                # TODO: calculate q_new
                q_new[d] = 0
            t += 1
        return p_new, q_new



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
    """ edges = {
    0: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)],
    1: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7), (7, 8)],
    2: [(1, 2), (5, 6), (5, 8)]
    }"""
    graphs = {}
    for i, edges in edges.items():
        graphs[i] = nx.Graph(edges)
    mutu = Muturank(graphs, 1e-6)
    #print mutu.a[mutu.node_pos[1],mutu.node_pos[4],1]
    #print mutu.r

