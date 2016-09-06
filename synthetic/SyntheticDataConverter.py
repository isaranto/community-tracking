import os
import networkx as nx


class SyntheticDataConverter:
    def __init__(self, filepath):
        self.filepath = filepath
        files = os.listdir(filepath)
        if any(item.startswith('expand') for item in files):
            self.type = 'expand-contract'
        elif any(item.startswith('birthdeath') for item in files):
            self.type = "birth-death"
        elif any(item.startswith('mergesplit') for item in files):
            self.type = "merge-split"

        self.edges_files = [item for item in sorted(files) if item.endswith('edges')]
        self.comm_files = [item for item in sorted(files) if item.endswith('comm')]
        self.event_file = [item for item in sorted(files) if item.endswith('events')]
        self.timeframes = len(self.edges_files)
        self.edges = self.get_edges()
        self.communities = self.get_comms()
        self.events = self.get_events()
        self.graphs = {}
        for i in range(1, self.timeframes+1):
            self.graphs[int(i)] = nx.Graph(self.edges[i])


    def get_edges(self):
        edge_time = {}
        for i, e_file in enumerate(self.edges_files, 1):
            edge_time[i] = []
            with open(self.filepath+e_file, 'r') as fp:
                for line in fp:
                    edge_time[i].append((int(line.split()[0]),int(line.split()[1])))
        return edge_time

    def get_comms(self):
        com_time = {}
        for i, c_file in enumerate(self.comm_files, 1):
            with open(self.filepath+c_file, 'r') as fp:
                comms = {}
                for j, line in enumerate(fp, 1):
                    comms[int(j)] = [int(node) for node in line.split()]
            com_time[int(i)] = comms
        return com_time

    def get_events(self):
        events = {}
        for i, e_file in enumerate(self.event_file, 1):
            with open(self.filepath+e_file, 'r') as fp:
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

