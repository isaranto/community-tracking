from __future__ import division
import json
import re
from itertools import combinations_with_replacement
import networkx as nx
import pprint


class dblp_parser:
    def __init__(self, json_file):
        with open(json_file, 'r') as fp:
            tmp = json.load(fp)
        self.data = self.parse_workshops(tmp)
        self.authors = self.get_authors()
        self.replace_names_with_ids()
        self.write_new_file()

    def parse_workshops(self, tmp):
        p = re.compile(r'conf/(.*)/.*$')
        workshops = {}
        count = 0
        total = 0
        for paper in tmp:
            total += 1
            m = p.match(paper[0])
            if m:
                count += 1
                name = m.group(1)
                authors = paper[1]
                year = paper[2]
                if year in workshops:
                    if name in workshops[year]:
                        workshops[year][name].append(authors)
                    else:
                        workshops[year][name] = []
                        workshops[year][name].append(authors)
                else:
                    workshops[year] = {}
                    workshops[year][name] = []
                    workshops[year][name].append(authors)
        print count, "/", total, " are confs. ", count/total*100, " %"
        return workshops

    def get_authors(self):
        authors = []
        for year, confs in self.data.items():
            for conf, papers in confs.items():
                for auth_list in papers:
                    authors.extend(auth_list)
        authors = list(set(authors))
        # auth_id = {}
        print len(authors), " # of authors"
        with open("../data/dblp/author_ids.txt", 'w')as fp:
            for j, author in enumerate(authors, 1):
                fp.write(str(j)+","+author.encode('utf-8')+"\n")
        auth_id = {}
        with open("../data/dblp/author_ids.txt", 'r') as fp:
            for line in fp:
                auth_id[line.split(",")[1].strip().decode('utf8')] = int(line.split(",")[0])
        return auth_id

    def replace_names_with_ids(self):
        for year, confs in self.data.items():
            for conf, papers in confs.items():
                for auth_list in papers:
                    for j, auth in enumerate(auth_list):
                        auth_list[j] = self.authors[auth]

    def write_new_file(self):
        with open('my_dblp_data.json', 'w')as fp:
            json.dump(self.data, fp, indent=2)


class dblp_loader:
    def __init__(self, _file, start_year, end_year, conf_file ='../data/dblp/confs.txt', coms='conf'):
        with open(_file, 'r')as fp:
            # load json and convert year-keys to int
            self.data = {int(k): v for k, v in json.load(fp).items()}
        self.conf_list = self.read_conf_list(conf_file)
        self.edges = self.get_edges(start_year, end_year)
        self.graphs = self.get_graphs(start_year, end_year)
        self.timeFrames = range(start_year, end_year+1)
        #self.communities = self.get_comms(start_year, end_year)
        conf_edges = self.get_conf_edges(start_year, end_year)
        self.conf_graphs = self.get_conf_graphs(conf_edges, start_year, end_year)
        if coms == 'conf':
            self.communities = self.get_conf_com(start_year, end_year)
        else:
            self.communities = self.get_cc_com(start_year, end_year)


    def get_edges(self, start_year, end_year):
        """
        Get the edges for the authors that include a paper
        in the conferences specified in the confs.txt
        :param start_year:
        :param end_year:
        :return:
        """
        edge_time = {}
        # for year, confs in self.data.iteritems():
        for year in range(start_year, end_year+1):
            edge_time[int(year)] = []
            for conf, paper_list in self.data[year].iteritems():
                if conf in self.conf_list:
                    for authors in paper_list:
                        for u, v in combinations_with_replacement(authors, 2):
                            edge_time[int(year)].append((u, v))
        return edge_time

    def get_conf_edges(self, start_year, end_year):
        """

        :param start_year:
        :param end_year:
        :return:
        """
        edge_time = {}
        # for year, confs in self.data.iteritems():
        for year in range(start_year, end_year+1):
            edge_time[int(year)] = {}
            for conf, paper_list in self.data[year].iteritems():
                if conf in self.conf_list:
                    for authors in paper_list:
                        for u, v in combinations_with_replacement(authors, 2):
                            try:
                                edge_time[int(year)][conf].append((u, v))
                            except:
                                edge_time[int(year)][conf] = [(u, v)]
        return edge_time

    def get_conf_graphs(self, conf_edges, start_year, end_year):
        graphs = {}
        for year, confs in conf_edges.iteritems():
            for conf, edges in confs.iteritems():
                try:
                    graphs[int(year)][conf] = nx.Graph(edges)
                except KeyError:
                    graphs[int(year)] = {conf: nx.Graph(edges)}
        return graphs

    """def get_comms(self, start_year, end_year):

        this is being used to get all conferences as communities
        :param start_year:
        :param end_year:
        :return:

        com_time = {}
        for year in range(start_year, end_year+1):
            # for year, confs in self.data.iteritems():
            comms = {}
            for j, (conf, papers) in enumerate(self.data[year].iteritems(), 1):
                com = [author for paper in papers for author in paper]
                # get rid of empty confs
                if len(com) == 0:
                    continue
                else:
                    comms[j] = com
            com_time[year] = comms
        return com_time"""

    def get_graphs(self, start_year, end_year):
        graphs = {}
        for year in range(start_year, end_year+1):
            graphs[year] = nx.Graph(self.edges[year])
        return graphs

    def read_conf_list(self, conf_file):
        conf_list = []
        with open(conf_file, 'r') as fp:
            for conf in fp:
                conf_list.append(conf.strip().lower())
        return conf_list


    def get_conf_com(self, start_year, end_year):
        """
        Returns the author communities for the conferences specified in confs.txt
        :param start_year:
        :param end_year:
        :return:
        """
        com_time = {}
        for year in range(start_year, end_year+1):
            # for year, confs in self.data.iteritems():
            comms = {}
            for j, conf in enumerate(self.conf_list, 1):
                com = []
                try:
                    for paper in self.data[year][conf]:
                        for author in paper:
                            com.append(author)
                    # com = [author for paper in self.data[year][conf] for author in paper]
                except KeyError:
                    # print(year, conf)
                    pass
                # get rid of empty confs
                if len(com) == 0:
                    continue
                else:
                    comms[j] = com
            com_time[year] = comms
        return com_time

    def get_cc_com(self, start_year, end_year):
        """
        get communities that correspond to connected components
        :param start_year:
        :param end_year:
        :return:
        """
        com_time = {}
        for year, confs in self.conf_graphs.iteritems():
            comms = {}
            com_id = 1
            for _, graph in confs.iteritems():
                for com in list(nx.connected_components(graph)):
                    if len(com) > 4:
                        comms[com_id] = list(com)
                        com_id += 1
            com_time[year] = comms
        return com_time

    def get_stats(self):
        length = {}
        for year, confs in dblp.data.iteritems():
            for conf, papers in confs.iteritems():
                conf_len = 0
                for paper in papers:
                    conf_len += len(paper)
                if conf_len > 0:
                    try:
                        length[conf][year] = conf_len
                    except KeyError:
                        length[conf] = {year: conf_len}
        confs = []
        for name, years in length.iteritems():
            parts_list = [parts for year, parts in years.iteritems() if year in range(2000, 2005)]
            if len(years) > 10 and all(parts_list[j] >= 1000 for j in range(len(parts_list))):
                    confs.append(name)
        return length


if __name__ == '__main__':
    filename = "../data/dblp/my_dblp_data.json"
    dblp = dblp_loader(filename, start_year=2000, end_year=2004, coms='components')
    # pprint.pprint(dblp.communities[2000])
    """stats = dblp.get_stats()
    pprint.pprint(dblp.data, indent=4, width=2)
    for i, graph in dblp.graphs.iteritems():
        print i, nx.number_connected_components(graph)
"""