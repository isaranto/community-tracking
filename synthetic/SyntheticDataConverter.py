import os
import networkx as nx

#TODO: get rid of false events
class SyntheticDataConverter:
    def __init__(self, filePath):
        if not filePath.endswith("/"):
            self.filePath = filePath+"/"
        else:
            self.filePath = filePath
        files = os.listdir(filePath)
        if any(item.startswith('expand') for item in files):
            self.type = 'expand-contract'
        elif any(item.startswith('birthdeath') for item in files):
            self.type = "birth-death"
        elif any(item.startswith('mergesplit') for item in files):
            self.type = "merge-split"
        elif any(item.startswith('hide') for item in files):
            self.type = "hide"

        self.edges_files = [item for item in sorted(files) if item.endswith('edges')]
        self.comm_files = [item for item in sorted(files) if item.endswith('comm')]
        self.event_file = [item for item in sorted(files) if item.endswith('events')]
        self.timeline_file = [item for item in sorted(files) if item.endswith('timeline')]
        self.timeFrames = len(self.edges_files)
        self.edges = self.get_edges()
        self.comms = self.get_comms()
        self.events = self.get_events()
        self.graphs = {}
        for i in range(self.timeFrames):
            self.graphs[int(i)] = nx.Graph(self.edges[i])
        self.add_self_edges()
        self.timeline = self.get_timeline()
        self.dynamic_truth = self.get_dynamic_coms()



    def get_edges(self):
        edge_time = {}
        for i, e_file in enumerate(self.edges_files):
            edge_time[i] = []
            with open(self.filePath+e_file, 'r') as fp:
                for line in fp:
                    edge_time[i].append((int(line.split()[0]), int(line.split()[1])))
        return edge_time

    def add_self_edges(self):
        for i, graph in self.graphs.iteritems():
            for v in graph.nodes():
                graph.add_edge(v, v)

    def get_comms(self):
        com_time = {}
        for timeFrame, c_file in enumerate(self.comm_files):
            with open(self.filePath+c_file, 'r') as fp:
                comms = {}
                for j, line in enumerate(fp):
                    comms[int(j)] = [int(node) for node in line.split()]
            com_time[int(timeFrame)] = comms
        return com_time

    def get_events(self):
        events = {}
        for i, e_file in enumerate(self.event_file):
            with open(self.filePath+e_file, 'r') as fp:
                for line in fp:
                    event = {}
                    e = line.strip().split(',')
                    event[int(e[2])] = e[1]
                    try:
                        events[int(e[0])].append(event)
                    except KeyError:
                        events[int(e[0])] = []
                        events[int(e[0])].append(event)
        return events

    def get_timeline(self):
        """
        Returns an ordered dictionary in the form dict[dynamic community][timeframe] = community

        :return:
        """
        dyn_communities = {}
        # TODO maybe we can use OrderedDict if appropriate
        for i, _file in enumerate(self.timeline_file):
            with open(self.filePath+_file, 'r') as fp:
                for line in fp:
                    timeline = {}
                    comm = int(line.split(":")[0].translate(None, "M"))-1
                    time_list = line.split(":")[1].strip().strip(",").split(",")
                    for value in time_list:
                        timeline[int(value.split("=")[0])-1] = int(value.split("=")[1])-1
                    dyn_communities[comm] = timeline
        return dyn_communities

    def get_dynamic_coms(self):
        dynamic_coms = {i: [] for i in self.timeline.keys()}
        for i, d_com_timeline in self.timeline.iteritems():
            new_com = []
            for tf, com_num in d_com_timeline.iteritems():
                for node in self.comms[tf][com_num]:
                    new_com.append("-t".join([str(node), str(tf)]))
            dynamic_coms[i] = new_com
        return dynamic_coms
