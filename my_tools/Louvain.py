import networkx as nx
import igraph as ig
import louvain_igraph as louvain
import community

class Louvain:
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
        # comms = []
        # print self.node_pos
        # G_new = ig.Graph()
        # print G.nodes()
        # print G.edges()
        # G_new.add_vertices([self.node_pos[v] for v in G.nodes()])
        # try:
        #     G_new.add_edges([(self.node_pos[int(e1)], self.node_pos[int(e2)]) for (e1, e2) in G.edges()])
        # except Exception:
        #     print self.node_pos[int(e2)]
        #     raise Exception
        # opt = louvain.Optimiser()
        # partition = opt.find_partition(graph=G_new, partition_class=louvain.SignificanceVertexPartition)
        # partitions = list(partition)
        # for i, com in enumerate(partitions):
        #     for n, node in enumerate(com):
        #         partitions[i][n] = self.node_ids[node]
        # com_dict = {i: com for i, com in enumerate(partitions)}
        # print com_dict
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