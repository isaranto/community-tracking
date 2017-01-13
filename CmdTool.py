import cmd
from synthetic import SyntheticDataConverter, Evaluation
from dblp import dblp_parser, dblp_loader
from plot import PlotGraphs
from tensor import TensorFact
from ged import GedLoad, GedWrite
from muturank import Muturank, Muturank_new
import pprint


class CmdTool(cmd.Cmd):
    def __init__(self):
        cmd.Cmd.__init__(self)
        self.graphs = {}
        self.comms = {}
        self.timeFrames = 0
        self.timeline = {}
        self.type = ""
        print "Hi, choose one of the following options:"

    def do_print_data_info(self, e):
        print "timeframes :", [key for key in self.graphs]
        print "# of communities per timeframe", [len(comms) for tf, comms in self.comms.iteritems()]
        print self.type

    def do_clear_memory(self, e):
        CmdTool().cmdloop()

    def do_load_ged_data(self, filename):
        """Give a GED compatible json file as input
        :param give the filePath as parameter
        """
        if not filename:
            filename = "/home/lias/PycharmProjects/GED/test_input_community_edges.json"
        ged = GedLoad(filename)
        self.graphs = ged.graphs
        self.comms = ged.comms
        self.timeFrames = len(self.graphs)
        self.type = "GED"
        print "GED data have been loaded successfully!"

    def do_write_ged(self, filename):
        """Output a json file as needed for GED evaluation

        :param filename: the name of the file to be written
        :return:
        """
        GedWrite(self, filename)
        print "Data have been exported successfully!"

    def do_load_synthetic_data(self, filePath, evaluate=False):
        """
        Load synthetic data
        :param filePath: the path of the folder where
        :param evaluate: boolean operator whether to evaluate data or not
        :return:
        """
        if not filePath:
            # filePath = "/home/lias/Dropbox/Msc/thesis/src/NEW/synthetic-data-generator/src/expand/"
            filePath = "data/synthetic/expand"
        sd = SyntheticDataConverter(filePath)
        self.graphs = sd.graphs
        self.comms = sd.communities
        self.timeFrames = sd.timeFrames
        self.timeline = sd.get_timeline()
        self.type = sd.type
        print "Synthetic data have been successfully loaded!"
        if evaluate:
            Evaluation(sd)
        return

    def do_load_dblp(self, filename, start_year=2000, end_year=2004, conf_file ='data/dblp/confs.txt',
                     coms='components'):
        """
        Load dblp dataset between years given
        :param filename:
        :param start_year:
        :param end_year:
        :param coms: ground truth communities or conferences 'confs' or 'components'
        :return:
        """
        if not filename:
            filename = "data/dblp/my_dblp_data.json"
        dblp = dblp_loader(filename, start_year=start_year, end_year=end_year, conf_file=conf_file, coms=coms)
        self.graphs = dblp.graphs
        self.comms = dblp.communities
        self.timeFrames = dblp.timeFrames
        print "DBLP data have been successfully loaded!"
        return

    def do_nntfact(self, params):
        """

        :param params:
        :return:
        """
        k, seeds = params.split()
        # num_of_coms = len(set([c for _,comms in self.comms.iteritems() for c in comms ]))
        # set number of factors to the max number of communities in all timeframes
        if not k:
            k = max([len(comms) for tf, comms in self.comms.iteritems()])
        TensorFact(self.graphs, num_of_coms=int(k), threshold = 1e-4, seeds= int(seeds))
        return

    def do_create_muturank_tensor(self, connections):
        """

        :param connections:
        :return:
        """
        Muturank(self.graphs)
        print("Tensor for use with Muturank")
        return

    def do_run_Muturank(self, k, connection='one'):
        """

        :param connections:
        :return:
        """
        threshold = 1e-6
        alpha = 0.85
        beta= 0.85
        mutu = Muturank_new(self.graphs, threshold, alpha, beta, connection, clusters=int(k))
        return


    def do_plot_graphs(self, node_size):
        """

        :param e:
        :return:
        """
        PlotGraphs(self.graphs, self.timeFrames, self.type, node_size)
        return

    def do_exit(self, e):
        """
        exit the program
        """
        return True


if __name__ == '__main__':
    CmdTool().cmdloop()
