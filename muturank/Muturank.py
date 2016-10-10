from __future__ import division
import networkx as nx
import numpy as np
import pandas as pd



class Muturank:
    def __init__(self, graphs, threshold):
        self.graphs = graphs
        self.a, self.o, self.r = self.create_dense_tensors(graphs)
        self.e = threshold
        # self.tensor= self.create_dense_tensors(graphs)
        # self.frame = self.create_dataframes(self.tensor)

    def create_dense_tensors(self, graphs):
        """
        construct two transition probability tensors O =[O i,j,d] and R =[r i,j,d]

        :param graphs:
        :return:
        """
        n = 0
        # set the number of nodes to the biggest graph found in the timeframes
        self.nodes = list(set([node for i in range(len(graphs)) for node in nx.nodes(graphs[i])]))
        n = len(self.nodes)
        s = len(graphs)
        a = np.zeros((s, n+1, n+1))
        for i, node in enumerate(self.nodes, 1):
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
        p_star = np.random.dirichlet(np.ones(len(self.nodes)))
        q_star = np.random.dirichlet(np.ones(len(self.graphs)))
        """
        alternatively we set q_star and q_star equal to 1/n
        p_star = np.ones(len(self.nodes))/len(self.nodes)
        q_star = np.ones(len(self.graphs))/len(self.graphs)
        """
        p_new= np.ones((len(p_star)))
        q_new = np.ones((len(q_star)))
        p_old = np.zeros((len(p_star)))
        q_old = np.zeros((len(q_star)))
        # TODO: while ||p(t)-p(t-1)||^2 + ||q(t) - q(t-1||^2 >=e
        while np.linalg.norm(p_new-p_old)**2 + np.linalg.norm(q_new-q_old)**2 > self.e:
            p_old = p_new
            q_old = q_new
            for i in range(len(self.nodes)):
                # TODO: calculate p_new
                p_new[i]= 0
            for d in range(len(self.graphs)):
                # TODO: calculate q_new
                q_new[d] = 0
            t += 1
        return p_new, q_new



        # TODO:     { compute p(t+1) and q(t+1)
        # TODO:     t = t+1 }
        # TODO:
        # TODO:


    def create_dataframes(self, tensor):
        dataframes = {}
        for i in range(tensor.shape[0]):
            print i
        pd.DataFrame(data=tensor[1:, 1:],    # values
              index=tensor[1:, 0],    # 1st column as index
            columns=tensor[0, 1:])

    def create_sp_tensors(self, graphs):
        o = None
        r = None
        return o, r



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
    print Muturank(graphs).o
    print Muturank(graphs).r

