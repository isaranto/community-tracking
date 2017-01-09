from __future__ import division
from itertools import combinations
from collections import Counter

class Omega:
    def __init__(self, comms1, comms2):
        self.nodes1 = self.get_node_assignment(comms1)
        self.nodes2 = self.get_node_assignment(comms2)
        self.nodes = list(set().union([node for i, com in comms2.iteritems() for node in com],
                                      [node for i, com in comms1.iteritems() for node in com]))
        J,K,N,obs,tuples1,tuples2 = self.observed()
        exp = self.expected(J,K,N,tuples1,tuples2)
        self.omega_score = self.calc_omega(obs, exp)
        print "obs = ", obs
        print "exp = ", exp
        print "Omega = ", self.omega_score


    def get_node_assignment(self,comms):
        """
        returns a dictionary with node-cluster assignments of the form {node_id :[cluster1, cluster_3]}
        :param comms:
        :return:
        """
        nodes = {}
        for i, com in comms.iteritems():
            for node in com:
                try:
                    nodes[node].append(i)
                except KeyError:
                    nodes[node] = [i]
        return nodes

    def num_of_common_clusters(self, u, v, nodes_dict):
        """
        return the number of clusters in which the pair u,v appears in the
        :param u:
        :param v:
        :param nodes_dict:
        :return:
        """
        try:
            sum = len(set(nodes_dict[u]) & set(nodes_dict[v]))
        except KeyError:
            sum = 0
        return sum

    def observed(self):
        N = 0
        tuples1 = {}
        J = 0
        for u, v in combinations(self.nodes, 2):
            N += 1
            n = self.num_of_common_clusters(u, v, self.nodes1)
            tuples1[(u, v)] = self.num_of_common_clusters(u, v, self.nodes1)
            J = n if n > J else J
        tuples2 = {}
        K = 0
        for u, v in combinations(self.nodes, 2):
            n = self.num_of_common_clusters(u, v, self.nodes2)
            tuples2[(u, v)] = self.num_of_common_clusters(u, v, self.nodes2)
            K = n if n > K else K
        obs = 0
        A = {j: 0 for j in range(min(J, K)+1)}
        for (u, v), n in tuples1.iteritems():
            try:
                if n == tuples2[(u, v)]:
                    A[n] += 1
            except KeyError:
                pass
        obs = sum(A[j]/N for j in range(min(J, K)+1))
        return J, K, N, obs, tuples1, tuples2

    def expected(self, J, K, N, tuples1, tuples2):
        N1 = Counter(tuples1.values())
        N2 = Counter(tuples2.values())
        exp = sum((N1[j]*N2[j])/(N**2) for j in range(min(J, K)+1))
        return exp

    def calc_omega(self, obs, exp):
        return (obs-exp)/(1-exp)

if __name__ == '__main__':
    comms1 = {1: [5, 6, 7], 2: [3, 4, 5], 3: [6, 7, 8]}
    comms2 = {1: [5, 6, 7], 2: [3, 4, 6], 3: [6, 7, 8]}
    comms3 = {1: [5, 6, 7], 2: [6, 7, 8], 3: [3, 4, 5]}
    omega = Omega(comms1, comms2)
