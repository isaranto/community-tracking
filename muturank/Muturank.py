from __future__ import division
import networkx as nx
import numpy as np
import pandas as pd


class Muturank:
    def __init__(self, graphs):
        self.tensor_o, self.tensor_r = self.create_dense_tensors(graphs)
        # self.tensor= self.create_dense_tensors(graphs)
        # self.frame = self.create_dataframes(self.tensor)

    def create_dense_tensors(self, graphs):
        """
        construct two transition probability tensors O =[O i,j,d] andR =[r i,j,d]

        :param graphs:
        :return:
        """
        n = 0
        # set the number of nodes to the biggest graph found in the timeframes
        nodes = list(set([node for i in range(len(graphs)) for node in nx.nodes(graphs[i])]))
        n = len(nodes)
        s = len(graphs)
        tensor_init = np.zeros((s, n+1, n+1))
        for i, node in enumerate(nodes, 1):
            tensor_init[:, i, 0] = node
            tensor_init[:, 0, i] = node
        for i in range(s):
            for u, v in graphs[i].edges_iter():
                tensor_init[i, u, v] = 1
                tensor_init[i, v, u] = 1
        tensor_o = np.copy(tensor_init)
        tensor_r = np.copy(tensor_init)
        for t in range(0, tensor_o.shape[0]):
            for j in range(1, tensor_o.shape[2]):
                for i in range(1, tensor_o.shape[1]):
                    if tensor_init[t, 1:, j].sum()!=0:
                        tensor_o[t, i, j] = tensor_o[t, i, j]/(tensor_init[t, 1:, j].sum())
        for i in range(1, tensor_r.shape[1]):
            for j in range(1, tensor_r.shape[2]):
                for t in range(0, tensor_r.shape[0]):
                    if tensor_init[:, i, j].sum()!=0:
                        tensor_r[t, i, j] = tensor_r[t, i, j]/(tensor_init[:, i, j].sum())
        return tensor_o, tensor_r

    def create_dataframes(self, tensor):
        dataframes = {}
        for i in range(tensor.shape[0]):
            print i
        pd.DataFrame(data=tensor[1:, 1:],    # values
              index=tensor[1:, 0],    # 1st column as index
            columns=tensor[0, 1:])

    def create_sp_tensors(self, graphs):
        tensor_o = None
        tensor_r = None
        return tensor_o, tensor_r



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
    print Muturank(graphs).tensor_o
