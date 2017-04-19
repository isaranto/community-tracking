from __future__ import division
import networkx as nx
import numpy as np
import pandas as pd
from sktensor import sptensor
from copy import deepcopy, copy
from scipy import sparse
from sklearn.cluster import spectral_clustering
import time
import datetime
import random
np.set_printoptions(precision=3, linewidth=300, formatter={'float_kind': '{:.5f}'.format})


class Muturank_new:
    #@profile
    def __init__(self, graphs, threshold, alpha, beta, connection, clusters, default_q=False, random_state=0):
        start = time.time()
        self.random_state = random_state
        random.seed(self.random_state)
        self.graphs = graphs
        self.node_ids = list(set([node for i in graphs for node in nx.nodes(graphs[i])]))
        self.num_of_nodes = len(self.node_ids)
        self.tfs = len(self.graphs)
        self.tfs_list = self.graphs.keys()
        self.clusters = clusters
        # create a dict with {node_id : tensor_position} to be able to retrieve node_id
        self.node_pos = {node_id: i for i, node_id in enumerate(self.node_ids)}
        # self.a, self.o, self.r, self.sum_cols, self.sum_time = self.create_sptensors()
        print "Creating tensors a, o ,r..."
        time1 = time.time()
        self.a, self.o, self.r, self.sum_cols, self.sum_time = self.create_sptensors(connection)
        print "created tensors in  ", time.time()-time1, " seconds"
        self.e = threshold
        self.alpha = alpha
        self.beta = beta
        if default_q:
            self.q_new = [1/self.tfs for i in range(self.tfs)]
        else:
            time1 = time.time()
            print "Running Muturank..."
            self.run_muturank()
            print "Muturank ran in ", time.time()-time1, " seconds"
        print "Creating monorelational network..."
        time1 = time.time()
        self.w = self.create_monorelational()
        print "Created monorelational in ", time.time()-time1, " seconds"
        print "Performing clustering on monorelational network..."
        time1 = time.time()
        self.dynamic_coms = self.clustering()
        self.duration = str(datetime.timedelta(seconds=int(time.time() - start)))
        print "Performed clustering in ", time.time()-time1, " seconds"
        """print sum(self.p_new)
        print sum(self.q_new)
        print(len(self.p_new))
        print(len(self.q_new))"""
        # self.check_probs()
        # print self.w.toarray()
        # print self.q_new

    def create_adj_tensor(self, graphs):
        """
        This function is being used to convert a dictionary of graphs { timeframe# : networkx_graph}
        into a dictionary-tensor of the form { timeframe# : sparse_adjacency_matrix}
        :param graphs:
        :return:
        """
        temp = {}
        for i, (t, graph) in enumerate(graphs.iteritems()):
            irr_graph = self.irr_components(graph)
            # a[i] = sparse.csr_matrix((self.num_of_nodes*self.tfs, self.num_of_nodes*self.tfs), dtype=np.float64)
            temp[i] = sparse.eye(self.num_of_nodes*self.tfs, dtype=np.float64, format="dok")
            for u, v, d in irr_graph.edges_iter(data=True):
                # add self edges for nodes that exist
                temp[i][u, u] = 1
                temp[i][v, v] = 1
                # add edges - create symmetric matrix
                temp[i][u, v] = d['weight']
                temp[i][v, u] = d['weight']
        return temp

    #@profile
    def create_sptensors(self, connection):
        """
            Create tensors A, O and R
            Tensor a is initialized here, while function create_adj_tensor is being used to update a ,thus making it
            irreducible after the time_edges are added
            :param :
            :return:
            """
        tuples = []
        # create adjacency tensor from initial graphs
        a = {}
        for i, (t, graph) in enumerate(self.graphs.iteritems()):
            irr_graph = self.irr_components(graph)
            #a[i] = sparse.csr_matrix((self.num_of_nodes*self.tfs, self.num_of_nodes*self.tfs), dtype=np.float64)
            a[i] = sparse.eye(self.num_of_nodes*self.tfs, dtype=np.float64, format="csr")
            for u, v, d in irr_graph.edges_iter(data=True):
                # add self edges for nodes that exist
                a[i][i*self.num_of_nodes + self.node_pos[u], i*self.num_of_nodes + self.node_pos[u]] = 1
                a[i][i*self.num_of_nodes + self.node_pos[v], i*self.num_of_nodes + self.node_pos[v]] = 1
                # add edges - create symmetric matrix
                try:
                    a[i][i*self.num_of_nodes + self.node_pos[u], i*self.num_of_nodes + self.node_pos[v]] = d['weight']
                    a[i][i*self.num_of_nodes + self.node_pos[v], i*self.num_of_nodes + self.node_pos[u]] = d['weight']
                except KeyError:
                    a[i][i*self.num_of_nodes + self.node_pos[u], i*self.num_of_nodes + self.node_pos[v]] = 1
                    a[i][i*self.num_of_nodes + self.node_pos[v], i*self.num_of_nodes + self.node_pos[u]] = 1

        # add time edges
        print "Adding time edges"
        a = self.add_time_edges(a, connection)
        print "Making irreducible"
        # make irreducible again
        a = self.irr_components_time(a)
        from scipy.sparse import hstack
        self.a_i = np.empty((self.num_of_nodes*self.tfs,), dtype=sparse.lil_matrix)
        for i in range(self.num_of_nodes*self.tfs):
            self.a_i[i] = hstack(tuple(x.getcol(i) for x in a.values()), format='csr')
        print "Creating o"
        # sum_cols = sparse.csr_matrix((self.num_of_nodes*self.tfs, self.tfs), dtype=np.float64)
        sum_cols = sparse.lil_matrix((self.num_of_nodes*self.tfs, self.tfs), dtype=np.float64)
        # sum_cols = {}
        from sklearn.preprocessing import normalize
        o = {t: None for t in range(self.tfs)}
        for t in range(self.tfs):
            o[t] = normalize(a[t], norm='l1', axis=0)
            sum_cols[:, t] = a[t].sum(axis=1)
            """for j in range(self.num_of_nodes*self.tfs):
                # FIXME: column sum bottleneck
                #sum_cols[j, t] = a[t].sum(0)[0, j]
                #FIXME: matrix operation instead of for loop
                o[t][:, j] = a[t][:, j]/sum_cols[j, t]
                for i in range(j+1):
                    if a[t][i, j] != 0:
                        try:
                            # o[t][j,i] = a[t][j, i]/np.sum(a[t][j, :])
                            o[t][i, j] = a[t][i, j]/sum_cols[j, t]
                            if i != j:
                                # o[t][i, j] = a[t][i, j]/np.sum(a[t][i, :])
                                o[t][j, i] = a[t][j, i]/sum_cols[i, t]
                        except ZeroDivisionError:
                            pass"""
        print "Creating r"
        sum_time = sparse.csr_matrix((self.num_of_nodes*self.tfs, self.num_of_nodes*self.tfs), dtype=np.float64)
        # for i in range(self.num_of_nodes*self.tfs):
        #     for j in range(self.num_of_nodes*self.tfs):
        #         sum_time[i, j] = sum([a[t][i, j] for t in range(self.tfs)])
        r = {t: sparse.eye(self.num_of_nodes*self.tfs, dtype=np.float64, format="csr") for t in range(self.tfs)}
        for tf in range(self.tfs):
            #sum_time2 = (sum_time.tocsr() + a[tf].tocsr()).tolil()
            sum_time += a[tf]
        for t in range(self.tfs):
            array1, array2 = a[t].nonzero()
            for index in range(len(array1)):
                i = array1[index]
                j = array2[index]
                r[t][i, j] = a[t][i, j]/sum_time[i, j]
        return a, o, r, sum_cols, sum_time

    def add_time_edges(self, a, connection):
        """
        The function add the inter-timeframe edges proposed in Timerank, across all tensor slices
        :param a:
        :param connection:
        :return:
        """
        # check if node i exists in graph[timeframe]
        if connection == 'one':
            # connect only with previous and next timeframe
            for t in range(self.tfs):
                for i in range(a[t].shape[0]):
                    try:
                        # always check if the nodes exist in the timeframe
                        # if they do, add an edge, else add a weak edge
                        # in order to achieve irreducibility
                        tf = i // self.num_of_nodes
                        node = i % self.num_of_nodes
                        if self.graphs[self.tfs_list[tf]].has_node(self.node_ids[node]):
                            this = True
                        else:
                            this = False
                        if i < self.num_of_nodes:
                            if this and self.has_node(tf+1, node):
                                a[t][i, i + self.num_of_nodes] = 1
                                a[t][i + self.num_of_nodes, i] = 1
                            else:
                                a[t][i, i + self.num_of_nodes] = 1e-4
                                a[t][i + self.num_of_nodes, i] = 1e-4
                        elif i >= self.num_of_nodes*(self.tfs-1):
                            if this and self.has_node(tf-1, node):
                                a[t][i, i - self.num_of_nodes] = 1
                                a[t][i - self.num_of_nodes, i] = 1
                            else:
                                a[t][i, i - self.num_of_nodes] = 1e-4
                                a[t][i - self.num_of_nodes, i] = 1e-4
                        else:
                            if this and self.has_node(tf+1, node):
                                a[t][i, i + self.num_of_nodes] = 1
                                a[t][i + self.num_of_nodes, i] = 1
                            else:
                                a[t][i, i + self.num_of_nodes] = 1e-4
                                a[t][i + self.num_of_nodes, i] = 1e-4
                            if this and self.has_node(tf-1, node):
                                a[t][i, i - self.num_of_nodes] = 1
                                a[t][i - self.num_of_nodes, i] = 1
                            else:
                                a[t][i, i - self.num_of_nodes] = 1e-4
                                a[t][i - self.num_of_nodes, i] = 1e-4
                    except IndexError:
                        pass
        elif connection == 'next':
            # connect only all timeframes
            for t in range(self.tfs):
                for i in range(a[t].shape[0]):
                    try:
                        # always check if the nodes exist in the timeframe
                        # if they do, add an edge, else add a weak edge
                        # in order to achieve irreducibility
                        tf = i // self.num_of_nodes
                        node = i % self.num_of_nodes
                        if self.graphs[self.tfs_list[tf]].has_node(self.node_ids[node]):
                            this = True
                        else:
                            this = False
                        for d in range(1, self.tfs):
                            if self.has_node(t, node) and self.has_node(d, node):
                                a[t][i, i + self.num_of_nodes*d] = 1
                                a[t][i + self.num_of_nodes*d, i] = 1
                                break
                            else:
                                a[t][i, i + self.num_of_nodes*d] = 1e-4
                                a[t][i + self.num_of_nodes*d, i] = 1e-4
                    except IndexError:
                        pass
        elif connection == 'all':
            # connect only all timeframes
            for t in range(self.tfs):
                for i in range(a[t].shape[0]):
                    try:
                        # always check if the nodes exist in the timeframe
                        # if they do, add an edge, else add a weak edge
                        # in order to achieve irreducibility
                        tf = i // self.num_of_nodes
                        node = i % self.num_of_nodes
                        if self.graphs[self.tfs_list[tf]].has_node(self.node_ids[node]):
                            this = True
                        else:
                            this = False
                        for d in range(1, self.tfs):
                            if self.has_node(t, node) and self.has_node(d, node):
                                a[t][i, i + self.num_of_nodes*d] = 1
                                a[t][i + self.num_of_nodes*d, i] = 1
                            else:
                                a[t][i, i + self.num_of_nodes*d] = 1e-4
                                a[t][i + self.num_of_nodes*d, i] = 1e-4
                    except IndexError:
                        pass
        return a

    # @staticmethod
    def irr_components(self, graph):
        """
        get connected components per timeframe and add an edge between them (with weight 0.0001)
        :param graph:
        :return: the irreducible graph for this timeframe
        """
        """for u, v, d in graph.edges(data=True):
            d['weight'] = 1"""

        random.seed(self.random_state)
        nodes = []
        for comps in nx.connected_components(graph):
            nodes.append(random.choice(list(comps)))
        edges = []
        for i in range(len(nodes)-1):
            edges.append((nodes[i], nodes[i+1], 1e-4))
        graph.add_weighted_edges_from(edges)
        return graph

    def irr_components_time(self, a):
        """
        After the new representation (with the addition of N*T nodes for each timeframe) the new graphs are made
        irreducible once again.
        :param a:
        :return:
        """
        random.seed(self.random_state)
        for t in range(self.tfs):
            num, comps = sparse.csgraph.connected_components(a[t], directed=False)
            if num == 1:
                continue
            else:
                comp_dict = {i: [] for i in range(num)}
                for i, c in enumerate(comps):
                    comp_dict[c].append(i)
                nodes = []
                for comps in comp_dict.values():
                    nodes.append(random.choice(list(comps)))
                for i in range(len(nodes)-1):
                    a[t][nodes[i], nodes[i+1]] = 1e-4
                    a[t][nodes[i+1], nodes[i]] = 1e-4
        return a

    # def irr_components_time(self, a):
    #     """
    #
    #     :param a:
    #     :return:
    #     """
    #
    #     graphs = {}
    #     for t in range(self.tfs):
    #         edges = []
    #         for i in range(a[t].shape[0]):
    #             for j in range(a[t].shape[0]):
    #                 if a[t][i, j] != 0:
    #                     edges.append((i, j, a[t][i, j]))
    #         graphs[t] = nx.Graph()
    #         graphs[t].add_weighted_edges_from(edges)
    #     return self.create_adj_tensor(graphs)
    #@profile
    def prob_t(self, d, j, denom):
        """
        Calculation of probalities p_t (d|j)
        :param d:
        :param j:
        :param denom:
        :return:
        """
        p = (self.q_old[d]*self.sum_cols[j, d])/denom
        # np.sum([self.q_old[m]*self.a[m][j, l] for l in range(self.num_of_nodes*self.tfs) for m in range(self.tfs)])
        return p

    #@profile
    def prob_n(self, i, j, denom):
        """
        Calculation of probalities p_t (i|j)
        :param i:
        :param j:
        :param denom:
        :return:
        """
        #p = np.sum([self.q_old[m]*self.a[m][j, i] for m in range(self.tfs)])/denom
        p = self.a_i[j].getrow(i).dot(self.q_old)/denom
        # np.sum([self.q_old[m]*self.a[m][j, l] for l in range(self.num_of_nodes*self.tfs) for m in range(self.tfs)])
        return p

    #@profile
    def run_muturank(self):
        """
        Running the iterative process of muturank until convergence.
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
        # p_star = np.random.dirichlet(np.ones(len(self.node_ids)))
        # q_star = np.random.dirichlet(np.ones(len(self.graphs)))
        # p_star = [1/(self.num_of_nodes*self.tfs) for _ in range(self.num_of_nodes*self.tfs)]
        # q_star = [1/self.tfs for _ in range(self.tfs)]
        p_star = np.repeat(1/(self.num_of_nodes*self.tfs), self.num_of_nodes*self.tfs)
        q_star = np.repeat(1/self.tfs, self.tfs)
        self.p_new = np.repeat(1/(self.num_of_nodes*self.tfs), self.num_of_nodes*self.tfs)
        self.q_new = np.repeat(1/self.tfs, self.tfs)
        self.p_old = np.repeat(1/(self.num_of_nodes*self.tfs), self.num_of_nodes*self.tfs)
        self.q_old = np.repeat(1/self.tfs, self.tfs)
        # while ||p(t)-p(t-1)||^2 + ||q(t) - q(t-1||^2 >=e
        start = True
        while (np.linalg.norm(self.p_new-self.p_old)**2 + np.linalg.norm(self.q_new-self.q_old)**2 >= self.e) or (
                start):
            start = False
            self.p_old = copy(self.p_new)
            self.q_old = copy(self.q_new)
            # calculate prob denominators once
            denom = np.zeros(self.num_of_nodes*self.tfs, dtype=np.float64)
            # for i in range(self.num_of_nodes*self.tfs):
            #     denom[i] = np.sum([self.q_old[m]*self.a[m][i, l]
            #                       for l in range(self.num_of_nodes*self.tfs)
            #                       for m in range(self.tfs)])
            q_a = {}
            for i in range(self.num_of_nodes*self.tfs):
                denom[i] = np.sum([np.sum(self.q_old[m]*self.a[m][i, :]) for m in range(self.tfs)])
            prob_t = np.zeros((self.tfs, self.num_of_nodes*self.tfs), dtype=np.float64)
            proba_t = sparse.lil_matrix((self.tfs, self.num_of_nodes*self.tfs), dtype=np.float64)
            for j in range(proba_t.shape[1]):
                for d in range(self.tfs):
                    prob_t[d, j] = self.prob_t(d, j, denom[j])
            for i in range(self.num_of_nodes*self.tfs):
                # first = np.sum(self.p_old[j]*self.o[d][i, :]*proba_t[d, :].T)
                # self.p_new[i] = self.alpha *first + (1-self.alpha)*p_star[i]
                self.p_new[i] = self.alpha *\
                               np.sum([self.p_old[j]*self.o[d][i, j]*self.prob_t(d, j, denom[j])
                                      for j in range(self.num_of_nodes*self.tfs)
                                      for d in range(self.tfs)])+(1-self.alpha)*p_star[i]
            for d in range(self.tfs):
                self.q_new[d] = self.beta *\
                                np.sum([self.p_old[j]*self.r[d][i, j]*self.prob_n(i, j, denom[j])
                                       for i in range(self.num_of_nodes*self.tfs)
                                       for j in range(self.num_of_nodes*self.tfs)])+(1-self.beta)*q_star[d]
            # divide each element with the sum of the distribution in order to reduce the error
            self.p_new = self.p_new/np.sum(self.p_new)
            self.q_new = self.q_new/np.sum(self.q_new)
            t += 1
            print "EPOCH ", t
            print "error ",np.linalg.norm(self.p_new-self.p_old)**2 + np.linalg.norm(self.q_new-self.q_old)**2
            #self.check_probs()
            #checking the calculation of probabilities
        # for j in range(len(self.node_ids)):
        #     print sum([self.prob_n(i, j, denom[j]) for i in range(len(self.node_ids))])
        #
        # for j in range(len(self.node_ids)):
        #     print sum([self.prob_t(d, j, denom[j]) for d in range(len(self.graphs))])
        return

    #@profile
    def create_monorelational(self):
        """
        Creating the monorelational network from the weighted sum of all adjacency matrices.
        :return: w matrix with the final edge weights.
        """
        w = sparse.lil_matrix((self.num_of_nodes*self.tfs, self.num_of_nodes*self.tfs), dtype=np.float64)
        for d in range(self.tfs):
            w += self.q_new[d]*self.a[d]
        return w
    #@profile
    def clustering(self):
        """
        Applies the spectral clustering algorithm on the monorelational network (w matrix)
        :return: Dynamic communities
        """
        print "running spectral"
        # # TODO: how to obtain # of communities
        # from scipy.sparse.linalg import eigs
        # print np.all(eigs(self.w) > 0)
        # clusters = SpectralClustering(affinity='precomputed',n_clusters=self.clusters,
        #                                random_state=self.random_state, eigen_solver='arpack').fit_predict(self.w)
        clusters = spectral_clustering(self.w, n_clusters=self.clusters,
                                       random_state=self.random_state, eigen_solver='arpack')
        #clusters = k_means(self.w, n_clusters=self.clusters, n_init=10)
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
        print "saving communities"
        comms = {}
        com_time = {}
        for n, c in enumerate(clusters):
            try:
                tf = n // self.num_of_nodes
                node = n % self.num_of_nodes
                if self.has_node(tf, node):
                    comms[c].append(str(self.node_ids[node])+"-t"+str(tf))
                    #com_time[tf] = []
            except KeyError:
                comms[c] = [str(self.node_ids[node])+"-t"+str(tf)]
        return comms

    def check_probs(self):
        """
        Checkign the correctness of the implementation by checking if probability distributions sum to 1.
        :return:
        """
        if np.sum(self.p_new)!=1.0:
            print "p_new ", np.sum(self.p_new), self.p_new
        if np.sum(self.q_new) != 1.0:
            print "q_new ", np.sum(self.q_new), self.q_new
        denom = np.zeros(self.num_of_nodes*self.tfs)
        for i in range(self.num_of_nodes*self.tfs):
            denom[i] = np.sum([self.q_old[m]*self.a[m][i, l]
                           for l in range(self.num_of_nodes*self.tfs)
                           for m in range(self.tfs)])
        for j in range(self.num_of_nodes*self.tfs):
            sum = 0
            for d in range(self.tfs):
                sum += self.prob_t(d, j, denom[j])
            if sum != 1.0:
                print "prob_t is", sum, " for j=", j
        for j in range(self.num_of_nodes*self.tfs):
            sum=0
            for i in range(self.num_of_nodes*self.tfs):
                sum += self.prob_n(i, j, denom[j])
            if sum != 1.0:
                print "prob_n is", sum, " for i=", i

    def check_irreducibility(self, a):
        """
        Checks the tensor for irreducibility.
        :param a:
        :return:
        """
        check = True
        for _, graph in self.graphs.iteritems():
            check = check and nx.is_connected(graph)
        for t in range(self.tfs):
            edges = []
            for i in range(self.num_of_nodes*self.tfs):
                for j in range(self.num_of_nodes*self.tfs):
                    if a[t][i, j] != 0:
                        edges.append((i, j))
            graph = nx.Graph(edges)
            check = check and nx.is_connected(graph)
        return check

    def has_node(self, tf, node):
        """
        checks if graph in tf has the specific node by checking the corresponding node id.
        :param tf: timeframe
        :param node: position of node in matrix/tensor
        :return:
        """
        return self.graphs[self.tfs_list[tf]].has_node(self.node_ids[node])

    def check_irr_w(self):
        """
        Checks the matrix w for irreducibility.
        """
        edges = []
        count = 0
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[0]):
                if self.w[i, j] != 0:
                    edges.append((i, j))
                    count += 1
        graph = nx.Graph(edges)
        check = nx.is_connected(graph)
        return check


if __name__ == '__main__':
    """
    edges = {
        0: [(1, 3), (1, 4), (2, 4)],
        1: [(1, 4), (3, 4), (1, 2)],
        2: [(1, 4), (3, 4), (1, 2)]
    }
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
    }
    # edges = {
    #     0: [(1,1), (2,2), (3,3)],
    #     1: [(1,1), (2,2), (3,3)],
    #     2: [(1,1), (2,2), (3,3)]
    # }
    graphs = {}
    for i, edges in edges.items():
        graphs[i] = nx.Graph(edges)
    #mutu = Muturank_new(graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='one', clusters=3)
    # print mutu.a[mutu.node_pos[1],mutu.node_pos[4],1]
    # print mutu.r
    mutu = Muturank_new(graphs, threshold=1e-6, alpha=0.85, beta=0.85, connection='next', clusters=3,
                            default_q=False)
    print mutu.a[0].toarray()
    print mutu.q_new
    print mutu.p_new
    print mutu.duration