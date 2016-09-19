import cmd
from synthetic import SyntheticDataConverter, Evaluation
from plot import PlotGraphs
from tensor import TensorFact
from ged import GedLoad, GedWrite

class CmdTool(cmd.Cmd):
    def __init__(self):
        cmd.Cmd.__init__(self)
        self.graphs = {}
        self.comms = {}
        self.timeframes = 0
        self.timeline = {}
        self.type = ""
        print "Hi, choose one of the following options:"

    def do_load_ged_data(self, fileName):
        """Give a GED compatible json file as input
        :param give the filepath as parameter
        """
        if not fileName:
            fileName = "/home/lias/PycharmProjects/GED/test_input_community_edges.json"
        ged = GedLoad(fileName)
        self.graphs = ged.graphs
        self.comms = ged.comms
        self.timeframes = len(self.graphs)
        self.type = "GED"
        print "GED data have been loaded successfully!"

    def do_write_ged(self, fileName):
        """Output a json file as needed for GED evaluation

        :param fileName: the name of the file to be written
        :return:
        """
        GedWrite(self, fileName)
        print "Data have been exported successfully!"

    def do_load_synthetic_data(self, filepath, evaluate=True):
        """
        Load synthetic data
        :param filepath: the path of the folder where
        :param evaluate: boolean operator whether to evaluate data or not
        :return:
        """
        if not filepath:
            filepath = "/home/lias/Dropbox/Msc/thesis/src/NEW/synthetic-data-generator/src/expand/"
        sd = SyntheticDataConverter(filepath)
        self.graphs = sd.graphs
        self.comms = sd.communities
        self.timeframes = sd.timeframes
        self.timeline = sd.get_timeline()
        self.type = sd.type
        print "Synthetic data have been successfully loaded!"
        if evaluate:
            Evaluation(sd)
        return


    def do_create_tensor(self, e):
        """

        :return:
        """
        TensorFact(self.graphs, self.timeframes)
        return

    def do_plot_graphs(self, e):
        """

        :return:
        """
        PlotGraphs(self.graphs, self.timeframes, self.type)
        return

    def do_exit(self):
        """
        exit the program
        """
        return True


if __name__ == '__main__':
    CmdTool().cmdloop()