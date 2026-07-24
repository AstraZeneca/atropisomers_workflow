# Atropisomers risk assessment workflow

This is a repository to support the publication **An Automated Computational Workflow for Quantifying Atropisomer Risk in Pharmaceutical Development** by *Elliot H. E. Farrar, Carlo Alberto Gaggioli, David Buttar and Simone Tomasi*. For queries, contact the corresponding author at *elliot.farrar@astrazeneca.com*.

<div align="center"> <img src="https://img.shields.io/badge/python-%3E=3.11-blue?logo=python"> <img src="https://img.shields.io/badge/Maturity%20Level-ML--0-red"> <img src="https://img.shields.io/badge/license-Apache%20License%202.0-blue"> </div>

## Overview

This repository contains an automated atropisomer risk assessment workflow that identifies hindered rotational axes likely to exhibit atropisomerism, evaluates their rotational barriers using quantum mechanics, assigns the [LaPlante atropisomer class](https://pubs.acs.org/doi/10.1021/jm200584g), and reports temperature- and solvent-dependent interconversion behaviour. The core class, `Atropisomers` (contained in `atropisomers/run.py`), operates from a SMILES string input and proceeds through substructure detection, steric filtering, conformer generation, conformer refinement, GIC scans, TS identification, barrier evaluation and results visualisation. The repository includes the complete cheminformatics and DFT barrier-evaluation pipeline, automated reporting and visualisation, and the supporting Slurm/QM interfacing utilities, including asyncio-based job scheduling and monitoring. Provided the required third-party QM software is available and executable paths are configured locally, the workflow can be run end-to-end directly from the repository with no manual transfer between stages by following the worked example available in `demo.ipynb`.

## Contents

 - `analysis.ipynb`: Interactive notebook containing workflow analysis and metrics computation.
 - `atropisomers/`: Code for the atropisomers workflow.
    - `.censorc`: Example CENSO configuration file used by the workflow.
    - `core.py`: Base workflow framework and shared infrastructure.
    - `red_conf_elim.py`: Iterative RMSD- and energy-based CREST conformer reduction script.
    - `run.py`: Main atropisomer workflow implementation and analysis logic.
    - `slurm.py`: Slurm job submission, execution, and progress-tracking utilities.
    - `utils.py`: Shared helper functions for parsing, structures, and workflow support.
 - `demo.ipynb`: A full worked example of the atropisomers workflow.
 - `example/`: Software outputs (xTB, CREST, CENSO, Gaussian) from the worked example.
 - `filter_benchmark.csv`: Structures and barriers for all compounds in the steric filter benchmark.
 - `filter_benchmark.ipynb`: Interactive notebook summarizing the data in the steric filter benchmark.
 - `literature_benchmark.csv`: Structures, barriers and sources for all compounds in the literature benchmark.
 - `literature_benchmark.ipynb`: Interactive notebook summarizing the data in the literature benchmark.
 - `qrnn_tbdft_benchmark.csv`: Structures and barriers for all compounds in the Balduf benchmark.
 - `requirements.txt`: Pip requirements for the atropisomers conda environment.
 - `substructures.ipynb`: Interactive notebook demonstrating substructure breadth.

Note that file paths in the code and example files have been replaced with placeholders and should be adapted to the local environment, where appropriate.

## Setup

### Clone the repository

```bash
git clone https://github.com/azu-rdit/atropisomers_workflow.git
```

### Install the conda environemnt

To build the conda environment for the interactive notebooks:
 - Create environment: `conda create --name atropisomers python=3.11.4`
 - Activate environment: `conda activate atropisomers`
 - Install pip requirements: `pip install -r requirements.txt`
 - Install cclib with conda: `conda install -c conda-forge openbabel cclib`

### Execute the demo notebook

Follow the steps in the interactive demonstration of the code using `jupyter notebook demo.ipynb`.
