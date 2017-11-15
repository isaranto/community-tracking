# Tracking the Evolution of Communities in Dynamic Social Networks

This repository includes the code developed around the MSc Thesis:

> Ilias Sarantopoulos, “Tracking the Evolution of Communities in Dynamic Social Networks” MSc Thesis, Athens University of Economics
and Business, 2017  

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

All packages available through pip can be found in the file 'requirements.txt'. You can install them by running:

```
pip install -r requirements.txt
```
There are 3 more external packages used: 'GED' is the python repository available in 
[demokritos-github](https://github.com/iit-Demokritos/community-Tracking-GED "demokritos-github") for Group Evolution Discovery. 'omni' is the C++ package
for NMI evaluation developed by  and available in [onmi-Github-repo](https://github.com/aaronmcdaid/Overlapping-NMI "NMI-Github") and 'ncp' is the package available in [NNTF-Github-repo](https://github.com/panisson/ntf-school "NNTF") for Non Negative Tensor Factorization with the least alternate squares method.

### Running Experiments

Each of the scripts in the root directory that starts with "experiments_{dataset}" describes a full set of comparative experiments for a specific dataset. The user should change the path to the files where the dataset exists.
Running the script will produce a txt file "results_{dataset}" which will contain the evaluated results of each method (GED, NNTF, Muturank). 
