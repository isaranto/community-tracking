from ged import GedLoad

class ReadGEDResults:
    def __init__(self, file_coms, file_output='../data/dblp_ged_results.csv'):
        self.coms = GedLoad(file_coms).comms
        self.dynamic_coms = self.read_output(file_output)

    def read_output(self, _file):
        results =[]
        with open(_file, 'r') as fp:
            for line in fp:
                results.append(line.strip().split(','))
        count=0
        # cont = {i: {} for i in self.coms}
        cont ={}
        for res in results:
            if res[4] == 'continuing':
                cont[(int(res[0]),int(res[1]))] = (int(res[2]), int(res[3]))
                cont[(int(res[2]),int(res[3]))] = False
        dynamic_coms_list = []
        for key, value in cont.iteritems():
            if value:
                tf1, com1 = key
                tf2, com2 = key
                new_com = []
                for node in self.coms[tf1][com1]:
                    new_com.append(str(node)+'-t'+str(tf1))
                for node in self.coms[tf2][com2]:
                    new_com.append(str(node)+'-t'+str(tf2))
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
        dynamic_coms = { i : com for i, com in enumerate(dynamic_coms_list)}
        return dynamic_coms
