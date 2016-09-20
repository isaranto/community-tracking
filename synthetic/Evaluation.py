from __future__ import division

class Evaluation:
    def __init__(self, sd):
        self.events = sd.events
        self.comms = sd.communities
        print "Now evaluating data..."
        self.type = sd.type
        self.filepath = sd.filepath
        if self.type=='expand-contract':
            self.count_faults()
        else:
            print "Evaluation for ", self.type, " events has not been added yet."

    def count_faults(self):
        faults = 0
        total =0
        for tf, events in self.events.iteritems():
            for event in events:
                for key, value in event.iteritems():
                    total +=1
                    old = self.comms[tf-1][key]
                    new = self.comms[tf][key]
                    change = (len(new)-len(old))/len(old)*100
                    if value=='expand' and change<10:
                        faults+=1
                    if value=='contract' and change>-10:
                        faults+=1


                    #print "Comm", key ,"has changed ", change, "% from tf",tf-1," to", tf, "ground truth" \
                                                                                           #" says" , event
            #print "tf ", tf-1, " to ", tf,
        eval =  str(faults/total*100)+ "% of events were wrong. Counted "+ str(faults)+ " in a total of "+str(total)+ " events"
        print eval
        with open(self.filepath+"evaluation.txt", 'w') as fp:
            fp.write(eval)
