import os
import networkx as nx
import json


class GedLoad:
    def __init__(self, fileName):
        fileName = "/home/lias/PycharmProjects/GED/test_input_community_edges.json"
        self.data = self.openFile(fileName)
        self.graphs, self.comms = self.getGraphs()

    def openFile(self, _file):
        try:
            with open(_file) as f:
                data = json.load(f)
        except IOError:
            print "Cannot load %s file. Expecting <input>.json file" % _file
            exit(1)
        return data


    def getGraphs(self):
        graphs = {}
        com_time = {}
        for timeIdx, window in enumerate(self.data['windows'], 1):
            comms = {}
            edges = []
            for com_index, community in enumerate(window['communities'], 1):
                comms[com_index] = []
                for e in community:
                    edges.append((e[0], e[1]))
                    comms[com_index].extend(e)
                comms[com_index] = list(set(comms[com_index]))
                #comms[com_index] = list(set([node for edge in community for node in edge]))
            graphs[timeIdx] = nx.Graph(edges)
            com_time[timeIdx] = comms

        return graphs, com_time
