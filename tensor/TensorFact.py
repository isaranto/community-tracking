from __future__ import division
import networkx as nx
import numpy as np
from sktensor import sptensor, dtensor, ktensor, cp_als
import ncp
from copy import deepcopy
from scipy import sparse
from sklearn.cluster import spectral_clustering
import pprint


class TensorFact:
    def __init__(self, graphs, num_of_coms, threshold, seeds=20, overlap=True):
        self.overlap = overlap
        self.thres = threshold
        self.graphs = graphs
        self.add_self_edges()
        self.num_of_coms = num_of_coms
        self.tfs_list = self.graphs.keys()
        self.node_ids = list(set([node for i in graphs for node in nx.nodes(graphs[i])]))
        # create a dict with {node_id : tensor_position} to be able to retrieve node_id
        self.node_pos = {node_id: i for i, node_id in enumerate(self.node_ids)}
        self.tensor = self.create_sptensor(graphs)
        """A_org = np.random.rand(len(self.node_ids), num_of_coms)
        B_org = np.random.rand(len(self.node_ids), num_of_coms)
        C_org = np.random.rand(len(graphs), num_of_coms)
        self.tensor = ktensor([A_org, B_org, C_org]).totensor()"""
        # self.tensor = self.create_dtensor(graphs)
        self.A, self.B, self.C = self.nnfact_repeat(seeds)
        self.dynamic_coms = self.get_comms(self.A, self.B, self.C)
        self. timeline = self.get_timeline(self.C)
        print "communities: ",
        pprint.pprint(self.dynamic_coms, indent=2, width=80)
        print "communities in timeframes: com:[tfi,tfj...]  ",
        pprint.pprint(self.timeline, indent=2, width=80)
        self.get_fact_info(self.A)


    def add_self_edges(self):
        for i, graph in self.graphs.iteritems():
            for v in graph.nodes():
                graph.add_edge(v, v)

    def get_fact_info(self, factor):
        for j in range(factor.shape[1]):
            for i in range(factor.shape[0]):
                if factor[i, j] < 1e-6:
                    factor[i, j] = 'nan'
        mins = ('%f' % x for x in np.nanmin(factor, 0))
        maxs = ('%f' % x for x in np.nanmax(factor, 0))
        print "min values : ", [float(a) for a in mins]
        print "max values : ", [float(a) for a in maxs]


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
            for u, v in graphs[i].edges_iter():
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

    def nnfact_repeat(self, num_of_seeds):
        seed_list = np.random.randint(0, 4294967295, num_of_seeds)
        min_error = 1
        for seed in seed_list:
            A, B, C, error = self.tensor_decomp(seed)
            if error <= min_error:
                best_seed = seed
                min_error= error
        # run once with the custom clustering initialized A factor
        A, B, C, error = self.tensor_decomp(best_seed, random_init=False)
        # if it doesnt give better results, run with best seed
        if error > min_error:
            A, B, C, error = self.tensor_decomp(best_seed)
            self.error = error
            self.best_seed =best_seed
            print "Error = ", error, " seed: ", best_seed
        else:
            self.error = error
            self.best_seed = best_seed
            print "Error = ", error, "with Custom init", ", seed: ", best_seed
        print "A = \n", A, "\n B = \n", B, "\n C = \n", C
        return A, B, C

    def tensor_decomp(self, seed, random_init=True):
        # setting seed in order to reproduce experiment
        np.random.seed(seed)
        if random_init:
            A_init = np.random.rand(self.tensor.shape[0], self.num_of_coms)
            B_init = deepcopy(A_init)
            C_init = np.random.rand(self.tensor.shape[2], self.num_of_coms)
            Finit = [A_init, B_init, C_init]
        else:
            Finit = self.get_Finit(seed)

        #Finit = [np.random.rand(X.shape[i], r) for i in range(nWay)]
        X_approx_ks = ncp.nonnegative_tensor_factorization(self.tensor, self.num_of_coms, method='anls_bpp',
                                                           stop_criterion=2, init=Finit)
        A = X_approx_ks.U[0]
        B = X_approx_ks.U[1]
        C = X_approx_ks.U[2]
        error = (self.tensor - X_approx_ks.totensor()).norm() / self.tensor.norm()
        return A, B, C, error

    def get_comms(self, A, B, C):
        #FIXME : universal solution for communities

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
            if self.overlap:
                for c in range(A.shape[1]):
                    if A[u, c] > self.thres:
                        try:
                            comms[c].append(self.node_ids[u])
                        except KeyError:
                            comms[c] = [self.node_ids[u]]
            else:
                c = np.argmax(A[u, :])
                if A[u, c] > self.thres:
                        try:
                            comms[c].append(self.node_ids[u])
                        except KeyError:
                            comms[c] = [self.node_ids[u]]
        dynamic_coms ={}
        for i, com in comms.iteritems():
            for tf, G in self.graphs.iteritems():
                for node in com:
                    if G.has_node(node):
                        try:
                            dynamic_coms[i].append(str(node)+"-t"+str(tf))
                        except KeyError:
                            dynamic_coms[i] = [str(node)+"-t"+str(tf)]
        return dynamic_coms


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

    def aggregated_network_matrix(self):
        agg = sparse.eye(len(self.node_ids), dtype=np.float32,format="dok")
        for i in range(len(self.node_ids)):
            for j in range(i):
                if sum([self.tensor[i, j, t] for t in range(len(self.graphs))]):
                    agg[i, j] = 1
                    agg[j, i] = 1
        return agg

    def get_Finit(self, seed):
        agg_network = self.aggregated_network_matrix()
        # A_init = sparse.dok_matrix((len(self.node_ids),len(self.node_ids)), dtype=np.float32)
        A_init = np.zeros((len(self.node_ids), self.num_of_coms))
        clusters = spectral_clustering(agg_network, n_clusters=self.num_of_coms, n_init=10, eigen_solver='arpack',
                                       random_state=seed)
        for i, t in enumerate(clusters):
            A_init[i, t] = 1
        B_init = deepcopy(A_init)
        C_init = np.random.rand(self.tensor.shape[2], self.num_of_coms)
        Finit = [A_init, B_init, C_init]
        return Finit


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
        graphs[i] = nx.Graph(edges)
    fact = TensorFact(graphs, num_of_coms=3, seeds=1, threshold=1e-4)
    from metrics import Omega
    print Omega(fact.dynamic_coms, fact.dynamic_coms).omega_score
    from metrics import NMI
    print NMI(fact.dynamic_coms, fact.dynamic_coms).results
    print fact.dynamic_coms