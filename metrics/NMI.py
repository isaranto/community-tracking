from __future__ import division
from itertools import combinations
from collections import Counter
from subprocess import Popen, PIPE




class NMI:
    def __init__(self, comms1, comms2):
        self.write_files(comms1, comms2)
        res = self.execute_cpp()
        self.results = self.get_results(res)
        """self.nodes1 = self.get_node_assignment(comms1)
        self.nodes2 = self.get_node_assignment(comms2)
        self.nodes = list(set().union([node for i, com in comms2.iteritems() for node in com],
                                      [node for i, com in comms1.iteritems() for node in com]))"""

    def write_files(self, comms1, comms2):
        with open('./nmi/file1.txt', 'w') as fp:
            for _, comm in comms1.iteritems():
                for node in comm:
                    fp.write(str(node))
                    fp.write(" ")
                fp.write("\n")
        with open('./nmi/file2.txt', 'w') as fp:
            for _, comm in comms2.iteritems():
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
        p = Popen(['./nmi/onmi ./nmi/file1.txt ./nmi/file2.txt'], shell=True, stdout=PIPE, stdin=PIPE)
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

if __name__=='__main__':
    comms1 = {1: [5, 6, 7], 2: [3, 4, 5], 3: [6, 7, 8]}
    comms2 = {1: [5, 6, 7], 2: [3, 4, 6], 3: [6, 7, 8]}
    nmi = NMI(comms1, comms2).results
    print nmi

