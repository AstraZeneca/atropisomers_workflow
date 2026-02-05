# Atropisomers workflow

This is a repository to support the publication **An Automated Computational Workflow for Quantifying Atropisomer Risk in Pharmaceutical Development** by *Elliot H. E. Farrar, Carlo Alberto Gaggioli, David Buttar and Simone Tomasi*. For queries, contact the corresponding author at *elliot.farrar@astrazeneca.com*.

Contents:
 - `analysis.ipynb`: Interactive notebook for workflow analysis.
 - `atropisomers.py`: Code for the atropisomers workflow.
 - `demo.ipynb`: A full worked example of the atropisomers workflow.
 - `example/`: Software outputs (xTB, CREST, CENSO, Gaussian) from the worked example.
 - `filter_benchmark.csv`: Structures and barriers for all compounds in the steric filter benchmark.
 - `filter_benchmark.ipynb`: Interactive notebook summarizing the data in the steric filter benchmark.
 - `literature_benchmark.csv`: Structures, barriers and sources for all compounds in the literature benchmark.
 - `literature_benchmark.ipynb`: Interactive notebook summarizing the data in the literature benchmark.
 - `requirements.txt`: Pip requirements for the atropisomers conda environment.


## Setup

### Clone the repository

```bash
git clone https://github.com/azu-rdit/atropisomers_workflow.git
```

### Install the conda environemnt

To build the conda environment for the interactive notebooks:
 - Create environment: `conda create --name atropisomers python=3.11.3`
 - Activate environment: `conda activate atropisomers`
 - Install pip requirements: `pip install -r requirements.txt`
 - Install cclib with conda: `conda install -c conda-forge openbabel cclib`
