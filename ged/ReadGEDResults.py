from ged import GedLoad
from collections import OrderedDict

class ReadGEDResults:
    """
    Convert results from GED output to universal format
    """
    def __init__(self, file_coms, file_output='../data/dblp_ged_results.csv'):
        self.coms = GedLoad(file_coms).comms
        self.dynamic_coms, self.cont = self.read_output(file_output)

    def read_output(self, _file):
        results = []
        with open(_file, 'r') as fp:
            for line in fp:
                results.append(line.strip().split(','))
        # cont = {i: {} for i in self.coms}
        cont = OrderedDict()
        for res in results:
            tf1, com1, tf2, com2, event = res
            if event in {'continuing', 'growing', 'shrinking'}:
                cont[(int(tf1), int(com1))] = (int(tf2), int(com2))
                cont[(int(tf2), int(com2))] = False
        dynamic_coms_list = []

        def dynamic_com(tf, com, new_com):
            if cont[tf, com]:
                    tf2, com2 = cont[tf, com]
                    for node in self.coms[tf2][com2]:
                        new_com.append(str(node)+'-t'+str(tf2))
                    cont[tf, com] = False
            return new_com

        for key, value in cont.iteritems():
            if value:
                tf1, com1 = key
                tf2, com2 = value
                new_com = []
                for node in self.coms[tf1][com1]:
                    new_com.append(str(node)+'-t'+str(tf1))
                for node in self.coms[tf2][com2]:
                    new_com.append(str(node)+'-t'+str(tf2))
                new_com = dynamic_com(tf2, com2, new_com)

                dynamic_coms_list.append(new_com)

        for tf, coms in self.coms.iteritems():
            for i, c in coms.iteritems():
                if (tf, i) in cont:
                    continue
                else:
                    new_com = []
                    for node in c:
                        new_com.append(str(node)+'-t'+str(tf))
                    dynamic_coms_list.append(new_com)
        dynamic_coms = {i: com for i, com in enumerate(dynamic_coms_list)}
        return dynamic_coms, cont



class Data(object):
    def __init__(self, comms, graphs, timeFrames, number_of_dynamic_communities, dynamic_truth=[]):
        self.comms = comms
        self.graphs = graphs
        self.timeFrames = timeFrames
        self.number_of_dynamic_communities = number_of_dynamic_communities
        self.dynamic_truth = dynamic_truth

if __name__ == '__main__':
    #ged = ReadGEDResults("/home/lias/PycharmProjects/GED/test_input_community_edges.json")
    # edges = {
    #     0: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)],
    #     1: [(11, 12), (11, 13), (12, 13)],
    #     2: [(1, 2), (1, 3), (1, 4), (3, 4), (5, 6), (6, 7), (5, 7)]
    # }
    # my_coms = {0: {0: [1, 2, 3, 4], 1: [5, 6, 7]},
    #            1: {2: [11, 12, 13] },
    #            2: {0: [1, 2, 3, 4], 1: [5, 6, 7]}
    # }
    # graphs = {}
    # import networkx as nx
    # for i, edges in edges.items():
    #     graphs[i] = nx.Graph(edges)
    # number_of_dynamic_communities = 3
    # data = Data(my_coms, graphs, len(graphs), number_of_dynamic_communities)
    import sys
    sys.path.insert(0, '../../GED/')
    import preprocessing, Tracker
    # from ged import GedWrite, ReadGEDResults
    # #ged_data = GedWrite(data)
    fileName = '../data/temp_ged_communities.json'
    # graphs = preprocessing.getGraphs(fileName)
    # tracker = Tracker.Tracker(graphs)
    # tracker.compare_communities()
    #outfile = 'tmpfiles/ged_results.csv'
    #outfile = '/home/lias/PycharmProjects/GED/dblp_ged_results.csv'
    outfile = '/home/lias/PycharmProjects/community-tracking/results/GED-events-handdrawn-9.csv'
    # with open(outfile, 'w')as f:
    #     for hypergraph in tracker.hypergraphs:
    #         hypergraph.calculateEvents(f)
    ged = ReadGEDResults(file_coms=fileName, file_output=outfile)
    #print data.dynamic_truth
    print ged.dynamic_coms
    print ged.cont
