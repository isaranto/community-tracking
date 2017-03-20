from __future__ import division
from itertools import combinations
from collections import Counter
from subprocess import Popen, PIPE




class NMI:
    def __init__(self, comms1, comms2, evaluation_type="dynamic"):
        self.eval = evaluation_type
        self.write_files(comms1, comms2)
        res = self.execute_cpp()
        self.results = self.get_results(res)
        """self.nodes1 = self.get_node_assignment(comms1)
        self.nodes2 = self.get_node_assignment(comms2)
        self.nodes = list(set().union([node for i, com in comms2.iteritems() for node in com],
                                      [node for i, com in comms1.iteritems() for node in com]))"""

    def write_files(self, comms1, comms2):
        if self.eval == "sets":
            new_comms1 = {i: set() for i in comms1.keys()}
            for i, comm in comms1.iteritems():
                for node in comm:
                    new_comms1[i].add(node.split('-')[0])
            new_comms2 = {i: set() for i in comms2.keys()}
            for i, comm in comms2.iteritems():
                for node in comm:
                    new_comms2[i].add(node.split('-')[0])

        if self.eval == "dynamic":
            new_comms1 = comms1
            new_comms2 = comms2

        if self.eval == "per_tf":
            pass

        with open('/home/lias/PycharmProjects/community-tracking/metrics/nmi/file1.txt', 'w') as fp:
            for _, comm in new_comms1.iteritems():
                for node in comm:
                    fp.write(str(node))
                    fp.write(" ")
                fp.write("\n")
        with open('/home/lias/PycharmProjects/community-tracking/metrics/nmi/file2.txt', 'w') as fp:
            for _, comm in new_comms2.iteritems():
                for node in comm:
                    fp.write(str(node))
                    fp.write(" ")
                fp.write("\n")

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

    def execute_cpp(self):
        p = Popen(['/home/lias/PycharmProjects/community-tracking/metrics/nmi/onmi /home/lias/PycharmProjects/community-tracking/metrics/nmi/file1.txt /home/lias/PycharmProjects/community-tracking/metrics/nmi/file2.txt'], shell=True, stdout=PIPE, stdin=PIPE)
        result = []
        for ii in range(4):
            value = str(ii) + '\n'
            #value = bytes(value, 'UTF-8')  # Needed in Python 3.
            try:
                p.stdin.write(value)
            except IOError:
                pass
            p.stdin.flush()
            result.append(p.stdout.readline().strip())
        return result

    def get_results(self, results):
        res = {}
        for line in results:
            if line.split(":")[1] == "":
                continue
            res[line.split(":")[0]] = float(line.split(":")[1].strip())
        return res

if __name__ == '__main__':
    comms1 = {1: [5, 6, 7], 2: [3, 4, 5], 3: [6, 7, 8]}
    comms2 = {1: [5, 6, 7], 2: [3, 4, 6], 3: [6, 7, 8]}
    comms3 = {0: ['1-t0','2-t0', '3-t0','4-t0', '1-t1', '2-t1',  '3-t1','4-t1', '1-t2','2-t2','3-t2','4-t2'],
                1: ['11-t1', '12-t1', '13-t1'],
            2: ['5-t2', '6-t2', '7-t2','5-t0', '6-t0', '7-t0']}
    comms4 = {1: ['1-t0','2-t0', '3-t0','4-t0', '1-t1', '2-t1',  '3-t1','4-t1', '1-t2','2-t2','3-t2','4-t2'],
                2: ['11-t1', '12-t1', '13-t1'],
            3: ['5-t2', '6-t2', '7-t2'],
              4: ['5-t0', '6-t0', '7-t0']}
    nmi = NMI(comms3, comms4, evaluation_type="dynamic").results
    print nmi
