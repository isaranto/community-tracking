import networkx as nx
import matplotlib.pyplot as plt


class PlotGraphs:
    def __init__(self, graphs, timeframes, type):
        self.draw_graphs(graphs, timeframes, type)

    def draw_graphs(self, graphs, timeframes, type):
        for i in range(1, timeframes+1):
            plt.figure()
            nx.draw(graphs[i], node_size=500, with_labels=True)
            plt.suptitle(type+"_Timeframe "+str(i), fontsize=14, fontweight='bold')
            plt.savefig(type+"_Timeframe_"+str(i)+".png")
