# Atropisomers risk assessment workflow

This is a repository to support the publication **An Automated Computational Workflow for Quantifying Atropisomer Risk in Pharmaceutical Development** by *Elliot H. E. Farrar, Carlo Alberto Gaggioli, David Buttar and Simone Tomasi*. For queries, contact the corresponding author at *elliot.farrar@astrazeneca.com*.

<div align="center"> <img src="https://img.shields.io/badge/python-%3E=3.11-blue?logo=python"> <img src="https://img.shields.io/badge/Maturity%20Level-ML--0-red"> <img src="https://img.shields.io/badge/license-Apache%20License%202.0-blue"> </div>

## Overview

This repository contains an atropisomer risk assessment workflow that identifies hindered rotatable bonds likely to exhibit atropisomerism, prepares and analyzes rotational scans, and validates rotational transition states (TSs). This allows users to predict the rotational barrier and [LaPlante atropisomer class](https://pubs.acs.org/doi/10.1021/jm200584g) of the identified bonds using quantum mechanics and probe the temperature- and solvent-dependence of the racemization kinetics. The core class, `Atropisomers` (contained in `atropisomers.py`), operates from a SMILES string input and proceeds through:

 - **Substructure detection**: Screens the input structure for predefined SMARTS motifs associated with atropisomerism to identify rotational bonds.
 - **Steric filtering**: Evaluates the local steric environment of rotational bonds to capture hindered systems.
 - **GIC scan setup**: Constructs dihedral scans to probe the rotational axis in both directions.
 - **Scan parsing**: Parses Gaussian scan logs to identify local maximum corresponding to TS candidates.
 - **TS validation**: Parses Gaussian TS optimization logs to confirm dihedrals remain consistent with the intended rotational mode.
 - **Barrier correction**: Applies linear corrections derived from experimental benchmark regressions.
 - **Class assignment**: Assigns the LaPlante atropisomer class of the computed barrier.
 - **Interpretation**: Visualizes the temperature-dependance of the interconversion half-life.

A full, step-by-step interactive demonstration of the code, including a worked example, is available in `demo.ipynb`.

## Contents

 - `analysis.ipynb`: Interactive notebook for workflow analysis.
 - `atropisomers.py`: Code for the atropisomers workflow.
 - `demo.ipynb`: A full worked example of the atropisomers workflow.
 - `example/`: Software outputs (xTB, CREST, CENSO, Gaussian) from the worked example.
 - `filter_benchmark.csv`: Structures and barriers for all compounds in the steric filter benchmark.
 - `filter_benchmark.ipynb`: Interactive notebook summarizing the data in the steric filter benchmark.
 - `literature_benchmark.csv`: Structures, barriers and sources for all compounds in the literature benchmark.
 - `literature_benchmark.ipynb`: Interactive notebook summarizing the data in the literature benchmark.
 - `requirements.txt`: Pip requirements for the atropisomers conda environment.
 - `substructures.ipynb`: Interactive notebook demonstrating substructure breadth.

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

### Execute the demo notebook

Follow the steps in the interactive demonstration of the code in `demo.ipynb`.
