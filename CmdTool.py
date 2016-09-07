import cmd
from synthetic import SyntheticDataConverter
from plot import PlotGraphs

class CmdTool(cmd.Cmd):
    def __init__(self):
        cmd.Cmd.__init__(self)
        self.graphs = {}
        self.timeframes = 0
        self.type =""
        print "Hi, choose one of the following options:"

    def do_load_ged_data(self, fileName):
        """Give a GED compatible json file as input
        :param give the filepath as parameter
        """
        self.file = fileName
        print "got json file"

    def do_load_synthetic_data(self, filepath):
        """

        :param filepath:
        :return:
        """
        if not filepath:
            filepath = "/home/lias/Dropbox/Msc/thesis/src/NEW/synthetic-data-generator/src/expand/"
        sd= SyntheticDataConverter(filepath)
        self.graphs = sd.graphs
        self.timeframes = sd.timeframes
        self.type = sd.type
        print "Synthetic data have been successfully loaded!"
        return

    def do_load_ged_data(self, filepath):
        """

        :param filepath:
        :return:
        """
        print "GED data have been loaded successfully!!!"
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