# Atropisomers Risk Assessment Workflow

This is a repository to support the publication **An Automated Computational Workflow for Quantifying Atropisomer Risk in Pharmaceutical Development** by *Elliot H. E. Farrar, Carlo Alberto Gaggioli, David Buttar and Simone Tomasi*. For queries, contact the corresponding author at *elliot.farrar@astrazeneca.com*.

## Overview

This repository contains an atropisomer risk assessment workflow that identifies hindered rotatable bonds likely to exhibit atropisomerism, prepares and analyzes rotational scans, and validates rotational transition states (TS). The core class, `AtropRun` (contained in `atropisomers.py`), operates from a SMILES string input and proceeds through:

 - **Substructure detection**: Screens the input structure for predefined SMARTS motifs associated with atropisomerism to identify rotational bonds.
 - **Steric filtering**: Evaluates the local steric environment of rotational bonds to capture sufficiently hindered systems.
 - **GIC scan setup**: Constructs Generalized Internal Coordinate (GIC) dihedral scans to probe the rotational axis in both directions.
 - **Scan parsing**: Parses Gaussian scan logs to identify local maximum corresponding to TS candidates.
 - **TS validation**: Parses Gaussian TS optimization logs to confirms that dihedrals remain consistent with the intended rotational mode.
 - **Barrier correction**: Applies linear corrections derived from experimental benchmark regressions.
 - **Visualization**: Visualizes identified atropisomeric bonds with the calculated barrier.

A full, step-by-step interactive demonstration of the code is available in `demo.ipynb`, including a worked example and illustrations of applicable substructures.

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
