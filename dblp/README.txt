===========================================================================================================
This software is being used to parse the DBLP dataset provided by http://dblp.uni-trier.de/xml.
Its part of my MSc thesis titles "Tracking the evolution of communities in dynamic social networks".
Its written in python using the networkx library for the representation of the co-authorship graphs. 

Ilias Sarantopoulos

===========================================================================================================
1)We use the code created by Erik Demaine (MIT) & MohammadTaghi Hajiaghayi (UMD) available at http://projects.csail.mit.edu/dnd/DBLP/DBLP2json.py 
in order to convert the file download from http://dblp.uni-trier.de/xml from XML to JSON.
Now the entries in the new file look like this:

["journals/acta/Saxena96", ["Sanjeev Saxena"], 1996],
["journals/acta/Simon83", ["Hans Ulrich Simon"], 1983],

2)There are two classes available in the dblp_parser.py file. The first one (dblp_parser) is used to parse the JSON file created with DBLP2json.py and output all the conferences into a new Json file. A new file is created "author_ids.txt" which is used to store the assignment of authors to ids.
the new Json file has the following structure: Year -> Conf -> [list of papers].
{
  year: {
    conference_name: [
      [
        author_id1
      ], 
      [
        author_id1, 
        author_id2,
        author_id3,
        author_id4
      ],..}

For each conference we store lists with the papers (in the example above the first paper has one author, while the second has four).

3) The second class in the dblp_parser.py file is name dblp_loader and is used to load the data we need for our experiments. The constructor needs the following arguments:
		 json file: the file extracted from the dblp_parser,
		 the start year: the first year for which we would like to get data (timeframe 0) ,
		 the end year: the last year (last timeframe). (We are extracting consecutive years as timeframes), 
		 the conferences file: a txt file which contains the conferences which we would like to extract(one per line), using their code name as used in the dblp file and website.
		 ground truth communities type: The type of ground truth communities we would like to use. There are two options available:
		 			'conf': for each year-timeframe we use each conference as a community.(a lot of disconnected components)
		 			'components': we extract the connected components from each conference, and we use as ground truth communities those that have more than 4 nodes.
	The data is being loaded into 2 python dictionaries (Hashmaps) - one for the graphs and one for the communities:
		The graphs dictionary stores networkx graph objects for each timeframe. Note that for each timeframe a co-authorship graph is constructed(authors that have written a paper together share an edge)
						{ 
							0: 1st_graph,
						  	1: 2nd_graph...
						}
		While the communities dictionary stores data using  the following structure:

						{
							year:
								com_id: [list of nodes in the community]
						}
