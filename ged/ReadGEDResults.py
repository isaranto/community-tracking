from ged import GedLoad


class ReadGEDResults:
    def __init__(self, file_coms, file_output='../data/dblp_ged_results.csv'):
        self.coms = GedLoad(file_coms).comms
        self.dynamic_coms = self.read_output(file_output)
        print self.dynamic_coms

    def read_output(self, _file):
        results = []
        with open(_file, 'r') as fp:
            for line in fp:
                results.append(line.strip().split(','))
        # cont = {i: {} for i in self.coms}
        cont = {}
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
        return dynamic_coms





if __name__ == '__main__':
    ged = ReadGEDResults("/home/lias/PycharmProjects/GED/test_input_community_edges.json")
    print type(ged.dynamic_coms)
