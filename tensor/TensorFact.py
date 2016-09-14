import networkx as nx
import numpy as np
from sktensor import dtensor, cp_als
import logging

class TensorFact:
    def __init__(self, graphs, timeframes):
        self.tensor = self.create_tensor(graphs, timeframes)
        self.tensor_decomp()

    def create_tensor(self, graphs, timeframes):
        # set the number of nodes to the biggest graph found in the timeframes
        nodes = list(set([node for i in range(1, timeframes+1) for node in nx.nodes(graphs[i])]))
        n = len(nodes)
        s = timeframes
        tensor = np.zeros((n+1, n+1, s), dtype='float32')
        for i, node in enumerate(nodes, 1):
            tensor[i, 0, :] = node
            tensor[0, i, :] = node
        for i in range(1, timeframes):
            for u, v in graphs[i].edges_iter():
                tensor[u, v, i] = 1
                tensor[v, u, i] = 1
        return tensor

    def tensor_decomp(self):
        logging.basicConfig(level=logging.DEBUG)
        T = dtensor(self.tensor)
        # Decompose tensor using CP-ALS
        p, fit, itr, exectimes = cp_als(T, 1, init='random')
        print p.shape, fit, itr