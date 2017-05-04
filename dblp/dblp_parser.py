from __future__ import division
import json
import re
from itertools import combinations_with_replacement,combinations
import networkx as nx
import pprint
import pickle
import time
import numpy as np

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
    def __init__(self, _file, start_year, end_year, conf_file='../data/dblp/confs.txt',
                 coms='comp', new_file=None):
        with open(_file, 'r')as fp:
            # load json and convert year-keys to int
            self.data = {int(k): v for k, v in json.load(fp).items()}
        self.conf_list = self.read_conf_list(conf_file)
        self.edges = self.get_edges(start_year, end_year)
        self.graphs = self.get_graphs(start_year, end_year)
        self.timeFrames = len(range(start_year, end_year+1))
        # self.communities = self.get_comms(start_year, end_year)
        conf_edges = self.get_conf_edges(start_year, end_year)
        self.conf_graphs = self.get_conf_graphs(conf_edges)
        if coms == 'comp':
            # Connected components (within conferences) are communities
            self.communities, self.com_conf_map, self.all_nodes = self.get_cc_com()
            self.dynamic_coms = self.get_dynamic_coms()
        else:
            # conferences are communities
            self.communities = self.get_conf_com(start_year, end_year)
        if new_file:
            self.create_new_file(start_year, end_year, new_file)

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
                            except KeyError:
                                edge_time[int(year)][conf] = [(u, v)]
        return edge_time

    def get_conf_graphs(self, conf_edges):
        graphs = {}
        for tf, (year, confs) in enumerate(conf_edges.iteritems()):
            for conf, edges in confs.iteritems():
                try:
                    graphs[int(tf)][conf] = nx.Graph(edges)
                except KeyError:
                    graphs[int(tf)] = {conf: nx.Graph(edges)}
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
        for tf, year in enumerate(range(start_year, end_year+1)):
            graphs[tf] = nx.Graph(self.edges[year])
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
        for tf, year in enumerate(range(start_year, end_year+1)):
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
            com_time[tf] = comms
        return com_time

    def get_cc_com(self):
        """
        get communities that correspond to connected components
        :return:
        """
        all_nodes = {i: set() for i in self.conf_graphs.keys()}
        com_time = {}
        # keep a seperate dictionary with the com_id -> conference_name bindings
        conference = {}
        for year, confs in self.conf_graphs.iteritems():
            comms = {}
            conf_map = {}
            com_id = 1
            for conf_name, graph in confs.iteritems():
                for com in list(nx.connected_components(graph)):
                    if len(com) > 4:
                        comms[com_id] = list(com)
                        conf_map[com_id] = conf_name
                        com_id += 1
            com_time[year] = comms
            conference[year] = conf_map
        for i, coms in com_time.iteritems():
            for com in coms.values():
                all_nodes[i].update(com)
        return com_time, conference, all_nodes

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

    def create_new_file(self, start_year, end_year, new_file):
        """
        exports a new json file, filtering the initial json with start-end years and conferences
        :param start_year:
        :param end_year:
        :return:
        """
        filtered_data = {}
        for year, conf_dict in self.data.iteritems():
            if year in range(start_year, end_year+1):
                filtered_data[year]={}
                for conf, papers in conf_dict.iteritems():
                    if conf in self.conf_list:
                        filtered_data[year][conf] = papers
        import os
        print os.path.dirname(os.path.realpath(__file__))
        with open(new_file, 'w')as fp:
            json.dump(filtered_data, fp, indent=2)


    def get_dynamic_coms(self):
        """
        Track communities: if two communities from different timeframes belong to the same conference and they have
        at leat one common author/node then put them in the same dynamic community.
        :return: Dynamic communities
        """
        tracked_coms = []
        # 1.for each community compare with
        for y1, y2 in zip(self.communities.keys(), self.communities.keys()[1:]):
            for id1, com1 in self.communities[y1].iteritems():
                for id2, com2 in self.communities[y2].iteritems():
                    same_name = self.com_conf_map[y1][id1] == self.com_conf_map[y2][id2]
                    common_author = set(self.communities[y1][id1]) & set(self.communities[y2][id2])
                    if same_name and common_author:
                        tracked_coms.append([str(y1)+"-"+str(id1), str(y2)+"-"+str(id2)])
        def common_elements(_list):
            """
            check if there exists a list with common elements
            :return:
            """
            flag = False
            for com1, com2 in combinations(_list, 2):
                if set(com1).intersection(com2):
                    flag = True
            return flag
        while (common_elements(tracked_coms)):
            for com1, com2 in combinations(tracked_coms, 2):
                if set(com1).intersection(com2):
                    try:
                        tracked_coms.remove(com1)
                        tracked_coms.remove(com2)
                    except ValueError:
                        continue
                    tracked_coms.append(list(set(com1).union(com2)))
        dynamic_coms_list = []
        for d_com in tracked_coms:
            new_com = []
            for com in d_com:
                year = com.split("-")[0]
                com_id = com.split("-")[1]
                for node in self.communities[int(year)][int(com_id)]:
                    new_com.append(str(node)+"-t"+str(year))
            dynamic_coms_list.append(new_com)
        dynamic_coms = {i: com for i, com in enumerate(dynamic_coms_list)}
        return dynamic_coms




if __name__ == '__main__':
    # from os.path import expanduser
    # home = expanduser("~")
    # path_full = home + "/Dropbox/Msc/thesis/data/synthetic_generator/data/birth_death_data"
    filename = "../data/dblp/my_dblp_data.json"
    try:
        with open('../data/dblp/dblp.pkl', 'rb')as fp:
            print "loading..."
            start = time.time()
            dblp = pickle.load(fp)
            raise IOError
    except IOError:
        print "creating..."
        start = time.time()
        dblp = dblp_loader(filename, start_year=2000, end_year=2010, coms='comp', new_file="../data/dblp/filtered.json")
        with open('../data/dblp/dblp.pkl', 'wb')as fp:
            pickle.dump(dblp, fp, pickle.HIGHEST_PROTOCOL)
    import pprint
    # pprint.pprint(dblp.communities)
    # pprint.pprint(dblp.com_conf_map)
    # pprint.pprint(dblp.dynamic_coms)
    # pprint.pprint(dblp.communities[2000])
    stats = dblp.get_stats()
    pprint.pprint(dblp.conf_graphs, indent=4, width=2)
    for year, confs in dblp.conf_graphs.iteritems():
        comps = sum([nx.number_connected_components(g) for g in confs.values()])
        papers = sum([len(dblp.data[year][c]) for c in confs.keys()])
        print year, len(dblp.conf_graphs[year]), len(dblp.communities[year]), comps, papers
    for year, graph in dblp.graphs.iteritems():
        print year, ", nodes: ",nx.number_of_nodes(graph), ", edges: ",nx.number_of_edges(graph)
    # # TODO:  add some comments
    # conf_life = {}
    # for year, data in dblp.data.iteritems():
    #     if year>1990:
    #         for conf, _ in data.iteritems():
    #             try:
    #                 conf_life[conf].append(year)
    #             except KeyError:
    #                 conf_life[conf] = [year]
    #             except TypeError:
    #                 print conf, year
    # for conf in conf_life.keys():
    #     if len(conf_life[conf]) < 5:
    #         conf_life.pop(conf, None)
    # print len(conf_life)
    # biennial = []
    # triennial = []
    # annual = []
    # for conf in conf_life.keys():
    #     list = conf_life[conf]
    #     #if np.std(np.diff(list)) > 0.5:
    #     #    del conf_life[conf]
    #     #if list[-1]-list[0] < 10:
    #     #    del conf_life[conf]
    #     #else:
    #     diffs = np.diff(list)
    #     if np.all(diffs==1):
    #         annual.append(conf)
    #     elif np.all(diffs==2):
    #         biennial.append(conf)
    #     elif np.all(diffs==3):
    #         triennial.append(conf)
    #     else:
    #         del conf_life[conf]
    # print "annual", len(annual)
    # print "biennial", len(biennial)
    # print "triennial", len(triennial)
    # print "total", len(conf_life)
    # with open("../data/dblp/Mary/clean_confs.txt", "w") as fp:
    #     for conf in conf_life.keys():
    #         fp.write("%s\n" % conf)
    # with open("../data/dblp/Mary/annual_confs.txt", "w") as fp:
    #     for conf in annual:
    #         fp.write("%s\n" % conf)
    # with open("../data/dblp/Mary/biennial_confs.txt", "w") as fp:
    #     for conf in biennial:
    #         fp.write("%s\n" % conf)
    # with open("../data/dblp/Mary/conf_life_after_1990.json", 'w')as fp:
    #         json.dump(conf_life, fp, indent=2)
    # #new_dblp = dblp_loader(_file = filename, start_year=1990, end_year=2016, conf_file =
    # # "../data/dblp/Mary/clean_confs.txt")
    # #new_dblp.create_new_file(start_year=1990, end_year=2016, new_file="../data/dblp/Mary/filtered.json")
    # filtered_data = {}
    # for year, conf_dict in dblp.data.iteritems():
    #         if year in range(1990, 2016+1):
    #             filtered_data[year]={}
    #             for conf, papers in conf_dict.iteritems():
    #                 if conf in conf_life.keys():
    #                     filtered_data[year][conf] = papers
    # with open("../data/dblp/Mary/dblp_filtered_new.json", 'w') as fp:
    #         for year, conferences in filtered_data.iteritems():
    #             for conf_name, papers in conferences.iteritems():
    #                 fp.write("{")
    #                 record = '"year":"{0}", "conf_name": "{1}", "papers": {2}'.format(year, conf_name, papers)
    #                 fp.write(record)
    #                 fp.write("},\n")




