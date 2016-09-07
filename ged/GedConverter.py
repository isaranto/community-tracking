import os
import networkx as nx
import json


class GedConverter:
    def __init__(self, filepath):
        self.filepath = filepath
        files = os.listdir(filepath)
        self.graphs = self.getGraphs(self.filepath)

    def createGraphs(self, data):
        graphs = {}
        # TODO load graphs for each timeframe

    def getGraphs(self, json_file):
        try:
            with open(json_file) as f:
                data = json.load(f)
        except IOError:
            print "Cannot load %s file. Expecting <input>.json file" % json_file
            exit(1)
        graphs = self.createGraphs(data)
        return graphs