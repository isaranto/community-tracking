from itertools import combinations_with_replacement
import json


class GedWrite:
    def __init__(self, data, fileName):
        self.fileName = fileName
        self.graphs= data.graphs
        self.comms = data.comms
        self.timeline = data.timeline
        self.timeframes = data.timeframes
        self.type = data.type
        self.write_data()

    def write_data(self):
        output = {}
        output["windows"] = []
        counter =0
        for tf in range(1, self.timeframes+1):
            communities = []
            # print tf, self.comms[tf]
            for id, comm in self.comms[tf].items():
                edges =[]
                # use all permutations of the possible edges in a community
                for u, v in combinations_with_replacement(comm, 2):
                    # if an edge exists add it to the community edges
                    if self.graphs[tf].get_edge_data(u,v) is not None:
                        edges.append([u, v])
                communities.append(edges)
            output["windows"].append({"communities":communities})
        with open(self.fileName, 'w') as f:
            json.dump(output, f, indent = 2)
