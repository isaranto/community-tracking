from __future__ import division
import networkx as nx
import numpy as np
import pandas as pd
from sktensor import sptensor
from copy import deepcopy, copy
from scipy import sparse
from sklearn.cluster import spectral_clustering
import time
np.set_printoptions(precision=3, linewidth=200)


class Muturank_new:
    def __init__(self, graphs, threshold, alpha, beta):
        self.graphs = graphs
        self.node_ids = list(set([node for i in graphs for node in nx.nodes(graphs[i])]))
        self.num_of_nodes = len(self.node_ids)
        self.tfs = len(self.graphs)
        # create a dict with {node_id : tensor_position} to be able to retrieve node_id
        self.node_pos = {node_id: i for i, node_id in enumerate(self.node_ids)}
        #self.a, self.o, self.r, self.sum_row, self.sum_time = self.create_sptensors()
        print "Creating tensors a, o ,r..."
        self.a, self.o, self.r, self.sum_row, self.sum_time = self.create_sptensors()
        self.e = threshold
        self.alpha = alpha
        self.beta = beta
        print "Running Muturank..."
        time1 = time.time()
        self.run_muturank()
        print "Muturank ran in ",time.time()-time1, " seconds"
        print "Creating monorelational network..."
        self.w = self.create_monorelational()
        print "Performing clustering on monorelational network..."
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
        a = self.add_time_edges(a, 'all')
        #print a[1].toarray()
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
        #print o[1].toarray()
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
        #print r[1].toarray()
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

    def prob_t(self, d, j):
        # OPTIMIZE: calculate denominator once for both probabilities
        p = (self.q_new[d]*self.sum_row[j, d])/sum([self.q_new[d]*self.a[m][j, l]
                                                    for l in range(self.num_of_nodes*self.tfs)
                                                    for m in range(self.tfs)])
        return p

    def prob_n(self, i, j):
        # OPTIMIZE: calculate denominator once for both probabilities
        p = sum([self.q_new[m]*self.a[m][i, j] for m in range(self.tfs)])/sum([self.q_new[m]*self.a[m][j, l]
                                                                                for l in range(self.num_of_nodes*self.tfs)
                                                                                for m in range(self.tfs)])
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
        p_star = [1/self.num_of_nodes*self.tfs for _ in range(self.num_of_nodes*self.tfs)]
        q_star = [1/self.tfs for _ in range(self.tfs)]
        """
        alternatively we set q_star and q_star equal to 1/n
        p_star = np.ones(len(self.node_ids))/len(self.node_ids)
        q_star = np.ones(len(self.graphs))/len(self.graphs)
        """
        self.p_new= np.ones((len(p_star)))
        self.q_new = np.ones((len(q_star)))
        self.p_old = np.zeros((len(p_star)))
        self.q_old = np.zeros((len(q_star)))# while ||p(t)-p(t-1)||^2 + ||q(t) - q(t-1||^2 >=e
        while np.linalg.norm(self.p_new-self.p_old)**2 + np.linalg.norm(self.q_new-self.q_old)**2 > self.e:
            self.p_old = copy(self.p_new)
            self.q_old = copy(self.q_new)
            for i in range(self.num_of_nodes*self.tfs):
                self.p_new[i] = self.alpha *\
                               sum([self.p_old[j]*self.o[d][i, j]*self.prob_t(d, j)
                                    for j in range(self.num_of_nodes*self.tfs)
                                    for d in range(self.tfs)])+(1-self.alpha)*p_star[i]
            for d in range(self.tfs):
                self.q_new[d] = self.beta *\
                                sum([self.p_old[j]*self.r[d][i, j]* self.prob_n(i, j)
                                     for i in range(self.num_of_nodes*self.tfs)
                                     for j in range(self.num_of_nodes*self.tfs)])+(1-self.beta)*q_star[d]
            t += 1
        """checking the calculation of probabilities
        for j in range(len(self.node_ids)):
            print sum([self.prob_n(i, j) for i in range(len(self.node_ids))])

        for j in range(len(self.node_ids)):
            print sum([self.prob_t(d, j) for d in range(len(self.graphs))])"""
        return

    def create_monorelational(self):
        w = sparse.eye(self.num_of_nodes*self.tfs, dtype=np.float32,format="dok")
        for i in range(self.num_of_nodes*self.tfs):
            for j in range(self.num_of_nodes*self.tfs):
                value = sum([self.q_new[d]*self.a[d][i, j] for d in range(self.tfs)])
                if value:
                    w[i, j] = value
        return w

    def clustering(self):
        clusters = spectral_clustering(self.w, n_clusters=3, n_init=10, eigen_solver='arpack')
        """com_time = {}
        for t in range(self.tfs):
            comms = {}
            for node in range(self.num_of_nodes):
                try:
                    comms[clusters[node + t*self.num_of_nodes]].append(self.node_ids[node])
                except KeyError:
                    comms[clusters[node + t*self.num_of_nodes]]= [self.node_ids[node]]
                #print self.node_ids[node], clusters[node + t*self.num_of_nodes]
            com_time[t] = comms"""
        comms = {}
        for n, c in enumerate(clusters):
            try:
                tf = n // self.num_of_nodes
                node = n % self.num_of_nodes
                comms[c].append(str(self.node_ids[node])+"-t"+str(tf))
            except KeyError:
                comms[c] = [str(self.node_ids[node])+"-t"+str(tf)]

        print clusters
        import pprint
        pprint.pprint(comms)


if __name__ == '__main__':
    """edges = {
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
    """
    edges = {
        0: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)],
        1: [(11, 12), (11, 13), (12, 13)],
        2: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)]
    }"""
    graphs = {}
    for i, edges in edges.items():
        graphs[i] = nx.Graph(edges)
    mutu = Muturank_new(graphs, 1e-6, 0.85, 0.85)
    #print mutu.a[mutu.node_pos[1],mutu.node_pos[4],1]
    #print mutu.r