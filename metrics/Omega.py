from __future__ import division
from itertools import combinations
from collections import Counter


class Omega:
    def __init__(self, comms1, comms2):
        self.nodes1 = self.get_node_assignment(comms1)
        self.nodes2 = self.get_node_assignment(comms2)
        self.nodes = list(set().union([node for i, com in comms2.iteritems() for node in com],
                                      [node for i, com in comms1.iteritems() for node in com]))
        J, K, N, obs, tuples1, tuples2 = self.observed()
        exp = self.expected(J, K, N, tuples1, tuples2)
        self.omega_score = self.calc_omega(obs, exp)
        # print "obs = ", obs
        # print "exp = ", exp
        # print "Omega = ", self.omega_score

    def get_node_assignment(self, comms):
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
            _sum = len(set(nodes_dict[u]) & set(nodes_dict[v]))
        except KeyError:
            _sum = 0
        return _sum

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
        try:
            return (obs-exp)/(1-exp)
        except ZeroDivisionError:
            return 0

if __name__ == '__main__':
    comms1 = {1: [5, 6, 7], 2: [3, 4, 5], 3: [6, 7, 8]}
    comms2 = {1: [5, 6, 7], 2: [3, 4, 6], 3: [6, 7, 8]}
    comms3 = {1: [5, 6, 7], 2: [6, 7, 8], 3: [3, 4, 5]}
    comms4 = {0: ['1-t0', '2-t0', '3-t0', '4-t0', '1-t1', '2-t1',  '3-t1','4-t1', '1-t2','2-t2','3-t2','4-t2'],
              1: ['11-t1', '12-t1', '13-t1'],
              2: ['5-t2', '6-t2', '7-t2', '5-t0', '6-t0', '7-t0']}
    comms5 = {1: ['11-t1', '12-t1', '13-t1'],
              2: ['1-t0', '2-t0', '3-t0', '4-t0', '1-t1', '2-t1',  '3-t1','4-t1', '1-t2','2-t2','3-t2','4-t2'],
              3: ['5-t2', '6-t2', '7-t2', '5-t0', '6-t0', '7-t0']}
    omega = Omega(comms4, comms5)
    print omega.omega_score
