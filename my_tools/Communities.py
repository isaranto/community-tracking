import networkx as nx
import community
class Communities:
    def __init__(self, graphs, type="louvain"):
        self.graphs = graphs
        self.communities = {i: {} for i in graphs.keys()}
        if type == "louvain":
            self.louvain()
        elif type == "klique":
            self.k_clique_communities()

    def k_clique_communities(self):
        for tf, graph in self.graphs.iteritems():
            coms = list(nx.k_clique_communities(graph, 2))
            self.communities[tf] = {i: list(com) for i, com in enumerate(coms)}

    def louvain(self):
        for tf, graph in self.graphs.iteritems():
            self.communities[tf] = self.run_louvain(graph)

    def run_louvain(self, G):
        com_dict = {}
        partition = community.best_partition(G)
        for node, c in partition.iteritems():
            try:
                com_dict[c].append(node)
            except KeyError:
                com_dict[c] = [node]
        for c in com_dict.keys():
            if not com_dict[c]:
                del com_dict[c]
        new_com_dict = {i: com for i, (c, com) in enumerate(com_dict.iteritems())}
        return new_com_dict




if __name__ == "__main__":
    edges = {
        0: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)],
        1: [(11, 12), (11, 13), (12, 13)],
        2: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)]
    }
    graphs = {i: nx.Graph(edges) for i, edges in edges.iteritems()}
    comz = Communities(graphs)
    print comz.communities
