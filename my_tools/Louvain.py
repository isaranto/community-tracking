import networkx as nx
import igraph as ig
import louvain_igraph as louvain
import community


class Louvain:
    """
    Run the louvain method to detect communities in each timeframe.
    The output can be used as input for a dynamic community finding method e.g. GED.
    """
    def __init__(self, graphs):
        self.graphs = graphs
        self.communities = {i: {} for i in graphs.keys()}
        self.node_ids = list(set([node for i in graphs for node in nx.nodes(graphs[i])]))
        self.node_pos = {node_id: i for i, node_id in enumerate(self.node_ids)}
        self.get_communities()

    def get_communities(self):
        for tf, graph in self.graphs.iteritems():
            self.communities[tf] = self.run_louvain(graph)

    def run_louvain(self, G):
        com_dict = {}
        partition = community.best_partition(G)
        for node, c in partition.iteritems():
            try:
                com_dict[c].append(node)
            except KeyError:
                com_dict[c]=[node]
        print com_dict
        return com_dict

if __name__ == "__main__":
    edges = {
        0: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)],
        1: [(11, 12), (11, 13), (12, 13)],
        2: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)]
    }
    graphs = {i: nx.Graph(edges) for i, edges in edges.iteritems()}
    louvain = Louvain(graphs)
    #print list(list(nx.k_clique_communities(graphs[0], 2))[0])