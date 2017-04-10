import networkx as nx
import matplotlib.pyplot as plt


class PlotGraphs:
    def __init__(self, graphs, timeframes, type, node_size=500):
        self.draw_graphs(graphs, timeframes, type, node_size)

    def draw_graphs(self, graphs, timeframes, type, node_size):
        for i in range(timeframes):
            plt.figure()
            nx.draw(graphs[i], node_size=node_size, with_labels=True)
            plt.suptitle(type+"_Timeframe "+str(i), fontsize=14, fontweight='bold')
            plt.savefig("images/"+type+"_Timeframe_"+str(i)+".png")
