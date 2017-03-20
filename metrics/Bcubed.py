import bcubed

class Bcubed:
    def __init__(self, truth, coms):
        self.ground_truth = self.process_input(truth)
        self.communities = self.process_input(coms)
        self.precision = bcubed.precision(self.ground_truth, self.communities)
        self.recall = bcubed.recall(self.ground_truth, self.communities)
        self.fscore = bcubed.fscore(self.precision, self.recall)

    def process_input(self, coms):
        itemset = {}
        for id, com_list in coms.iteritems():
            for node in com_list:
                try:
                    itemset[node].add(id)
                except KeyError:
                    itemset[node] = set([id])
        print itemset
        return itemset


if __name__=='__main__':
    coms = {0: ['11-t1', '12-t1', '13-t1'],
            1: ['1-t0',
                '2-t0',
                '3-t0',
                '4-t0',
                '1-t1',
                '2-t1',
                '3-t1',
                '4-t1',
                '1-t2',
                '2-t2',
                '3-t2',
                '4-t2'],
            2: ['5-t0', '6-t0', '7-t0', '5-t2', '6-t2', '7-t2']}

    coms1 = {23: ['11-t1', '12-t1', '13-t1','1-t0'],
             50: ['1-t0',
                  '2-t0',
                  '3-t0',
                  '4-t0',
                  '1-t1',
                  '2-t1',
                  '3-t1',
                  '4-t1',
                  '1-t2',
                  '2-t2',
                  '3-t2',
                  '4-t2'],
             2: ['5-t0', '6-t0', '7-t0', '5-t2', '6-t2', '7-t2']}
    b = Bcubed(coms, coms1)
    print b.fscore
