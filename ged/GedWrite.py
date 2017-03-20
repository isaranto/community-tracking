from itertools import combinations_with_replacement
import json


class GedWrite:
    def __init__(self, data, fileName='./data/temp_ged_communities.json'):
        self.fileName = fileName
        self.graphs= data.graphs
        self.comms = data.comms
        # self.timeLine = data.timeLine
        if isinstance(data.timeFrames, list):
            self.timeFrames = data.timeFrames
        else:
            self.timeFrames = range(data.timeFrames)
        #self.type = data.type
        self.write_data()

    def write_data(self):
        output = {"windows": []}
        #output["windows"] = []
        for tf in self.timeFrames:
            communities = []
            print "Now processing time frame ", tf
            for _id, comm in self.comms[tf].items():
                edges = []
                # use all permutations of the possible edges in a community
                for u, v in combinations_with_replacement(comm, 2):
                    # if an edge exists add it to the community edges
                    if self.graphs[tf].get_edge_data(u, v) is not None:
                        if not u == v:
                            edges.append([u, v])
                communities.append(edges)
            output["windows"].append({"communities": communities})
        with open(self.fileName, 'w') as f:
            json.dump(output, f, indent=2)
