from __future__ import division
import json
import re
from itertools import combinations_with_replacement
import networkx as nx


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
            for i, author in enumerate(authors, 1):
                fp.write(str(i)+","+author.encode('utf-8')+"\n")
        auth_id = {}
        with open("../data/dblp/author_ids.txt", 'r') as fp:
            for line in fp:
                auth_id[line.split(",")[1].strip().decode('utf8')] = int(line.split(",")[0])
        return auth_id

    def replace_names_with_ids(self):
        for year, confs in self.data.items():
            for conf, papers in confs.items():
                for auth_list in papers:
                    for i, auth in enumerate(auth_list):
                        auth_list[i] = self.authors[auth]

    def write_new_file(self):
        with open('my_dblp_data.json', 'w')as fp:
            json.dump(self.data, fp, indent=2)


class dblp_loader:
    def __init__(self, filename, start_year=1959, end_year=2017):
        with open(filename, 'r')as fp:
            # load json and convert year-keys to int
            self.data = {int(k): v for k, v in json.load(fp).items()}
        self.edges = self.get_edges(start_year, end_year)
        self.graphs = self.get_graphs(start_year, end_year)
        self.timeFrames = range(start_year, end_year+1)
        #self.communities = self.get_comms(start_year, end_year)
        self.communities = self.test_com(start_year, end_year)

    def get_edges(self, start_year, end_year):
        edge_time = {}
        # for year, confs in self.data.iteritems():
        for year in range(start_year, end_year+1):
            edge_time[int(year)] = []
            for conf, auth_list in self.data[year].iteritems():
                for authors in auth_list:
                    for u, v in combinations_with_replacement(authors, 2):
                        edge_time[int(year)].append((u, v))
        return edge_time

    def get_comms(self, start_year, end_year):
        com_time = {}
        for year in range(start_year, end_year+1):
        # for year, confs in self.data.iteritems():
            comms = {}
            for i, (conf, papers) in enumerate(self.data[year].iteritems(), 1):
                com = [author for paper in papers for author in paper]
                # get rid of empty confs
                if len(com) == 0:
                    continue
                else:
                    comms[i] = com
            com_time[year] = comms
        return com_time

    def get_graphs(self, start_year, end_year):
        graphs = {}
        for year in range(start_year, end_year+1):
            graphs[year] = nx.Graph(self.edges[year])
        return graphs

    def test_com(self,start_year, end_year):
        conf_list = []
        with open('../data/dblp/confs.txt', 'r') as fp:
            for conf in fp:
                conf_list.append(conf.strip().lower())
        com_time = {}
        for year in range(start_year, end_year+1):
        # for year, confs in self.data.iteritems():
            comms = {}
            for i, conf in enumerate(conf_list, 1):
                com = []
                try:
                    for paper in self.data[year][conf]:
                        for author in paper:
                            com.append(author)
                    #com = [author for paper in self.data[year][conf] for author in paper]
                except:
                    #print(year, conf)
                    pass
                # get rid of empty confs
                if len(com) == 0:
                    continue
                else:
                    comms[i] = com
            com_time[year] = comms
        return com_time

    def get_stats(self):
        length = {}
        for year, confs in dblp.data.iteritems():
            for conf, papers in confs.iteritems():
                conf_len = 0
                for paper in papers:
                    for author in papers:
                        conf_len += 1
                if conf_len>0:
                    try:
                        length[conf][year] = conf_len
                    except KeyError:
                        length[conf]= {year:conf_len}
        confs =[]
        for name, years in length.iteritems():
            parts_list = [parts for year, parts in years.iteritems() if year in range(2000, 2005)]
            if len(years)> 10 and all(parts_list[i]>=1000 for i in range(len(parts_list))):
                    confs.append(name)
        return length



if __name__=='__main__':
    filename = "../data/dblp/my_dblp_data.json"
    dblp = dblp_loader(filename, start_year=2000, end_year=2004)
    stats = dblp.get_stats()
    print stats




