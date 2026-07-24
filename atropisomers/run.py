import re
import math
import shutil
import asyncio
import subprocess
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from IPython.display import display
from collections import defaultdict
from itertools import combinations, product
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw

# Import base class to inherit from
from .core import Workflow

from .slurm import (
    ProgressBar,
    task,
)

from .utils import (
    parse_with_cclib,
    parse_scan_with_cclib,
    read_from_xyz,
    get_conformer_xyz,
)

# Import last to prevent C++ import errors
import pandas as pd

# Here, we define the SMARTS patterns and formal names for potential atropisomeric patterns. For each SMARTS pattern,
# we also define the minimum number of heavy substituents that are required around the rotatable bond for atropisomerism
# to be likely (based on the approximate steric bulk around the rotatable bond). These rules are based on a series of
# reference calculations and are aimed at capturing any barrier above ~12 kcal/mol. Any structures that violate the rules
# are deemed to have insufficient bulk for atropisomerism and are filtered out. We use 12 kcal/mol as a safe margin to
# catch as many Class 2 atropisomers as possible. For aryl rings, the number of non-heavy substituents is counted by how
# many of the ortho substituents are lone pairs or protons. For non-ring systems, such as benzamides or sulfoxides, we count
# the immediate neighbors of the atom in the rotatable bond instead. Thus, the maximum number of heavy atoms is always 4 for
# all substructures. Note that for some non-aryl systems, some substituents are heavy by definition (e.g., sulfoxides always
# have an =O atom). This is taken into account by the rules.

PATTERNS = {

    # Joined aryl-aryl ring systems
    "Biaryl (7-7)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1", # two 7-membered aromatics joined by a single bond
        "min_heavy_subs": 1, # minimum of 1/4 possible ring substituents
    },
    "Biaryl (7-6)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[*]1:[*]:[*]:[*]:[*]:[*]:1", # 7- and 6-membered rings joined by a single bond
        "min_heavy_subs": 1, # minimum of 1/4 possible ring substituents
    },
    "Biaryl (7-5)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[*]1:[*]:[*]:[*]:[*]:1", # 7- and 5-membered rings joined by a single bond
        "min_heavy_subs": 2, # minimum of 2/4 possible ring substituents
    },
    "Biaryl (6-6)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[*]1:[*]:[*]:[*]:[*]:[*]:1", # two 6-membered aromatic rings joined by a single bond
        "min_heavy_subs": 1, # minimum of 1/4 possible ring substituents
    },
    "Biaryl (6-5)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[*]1:[*]:[*]:[*]:[*]:1", # 6- and 5-membered rings joined by a single bond
        "min_heavy_subs": 2, # minimum of 2/4 possible ring substituents
    },
    "Biaryl (5-5)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:1-[*]1:[*]:[*]:[*]:[*]:1", # two 5-membered aromatic rings joined by a single bond
        "min_heavy_subs": 3, # minimum of 3/4 possible ring substituents
    },

    # Heteroatom-joined aryl-aryl ring systems
    # Includes diaryl ethers, thioethers (sulfoxides and sulfones) and amines
    "Diaryl ether (7-7)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[#8]-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 1, # minimum of 1/4 possible ring substituents
    },
    "Diaryl ether (7-6)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[#8]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 2, # minimum of 2/4 possible ring substituents
    },
    "Diaryl ether (7-5)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[#8]-[*]1:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 2, # minimum of 2/4 possible ring substituents
    },
    "Diaryl ether (6-6)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#8]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 2, # minimum of 2/4 possible ring substituents
    },
    "Diaryl ether (6-5)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#8]-[*]1:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # minimum of 3/4 possible ring substituents
    },
    "Diaryl ether (5-5)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:1-[#8]-[*]1:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # minimum of 3/4 possible ring substituents
    },
    "Diaryl thioether (7-7)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[#16]-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 1, # as for diaryl ethers
    },
    "Diaryl thioether (7-6)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#16]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 2, # as for diaryl ethers
    },
    "Diaryl thioether (7-5)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#16]-[*]1:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 2, # as for diaryl ethers
    },
    "Diaryl thioether (6-6)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#16]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 2, # as for diaryl ethers
    },
    "Diaryl thioether (6-5)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#16]-[*]1:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # as for diaryl ethers
    },
    "Diaryl thioether (5-5)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:1-[#16]-[*]1:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # as for diaryl ethers
    },
    "Diaryl amine (7-7)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 2, # minimum of 2/4 possible ring substituents
    },
    "Diaryl amine (7-6)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 2, # minimum of 2/4 possible ring substituents
    },
    "Diaryl amine (7-5)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#7]-[*]1:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 2, # minimum of 2/4 possible ring substituents
    },
    "Diaryl amine (6-6)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 2, # minimum of 2/4 possible ring substituents
    },
    "Diaryl amine (6-5)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#7]-[*]1:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 2, # minimum of 2/4 possible ring substituents
    },
    "Diaryl amine (5-5)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:1-[#7]-[*]1:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # minimum of 3/4 possible ring substituents
    },

    # Joined aryl-nonaryl ring systems
    "Aryl-aliphatic (7-7)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]~[*]~[*]1", # 7-membered aromatic and 7-membered non-aromatic joined by a single bond
        "min_heavy_subs": 0, # minimum of 0/4 possible ring substituents
    },
    "Aryl-aliphatic (7-6)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]~[*]1", # 7-membered aromatic and 6-membered non-aromatic joined by a single bond
        "min_heavy_subs": 0, # minimum of 0/4 possible ring substituents
    },
    "Aryl-aliphatic (6-7)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]~[*]~[*]1", # 6-membered aromatic and 7-membered non-aromatic joined by a single bond
        "min_heavy_subs": 0, # minimum of 0/4 possible ring substituents
    },
    "Aryl-aliphatic (7-5)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]~[*]1", # 7-membered aromatic and 5-membered non-aromatic joined by a single bond
        "min_heavy_subs": 1, # minimum of 1/4 possible ring substituents
    },
    "Aryl-aliphatic (5-7)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]~[*]~[*]1", # 5-membered aromatic and 7-membered non-aromatic joined by a single bond
        "min_heavy_subs": 1, # minimum of 1/4 possible ring substituents
    },
    "Aryl-aliphatic (6-6)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]~[*]1",
        "min_heavy_subs": 1, # minimum of 1/4 possible ring substituents
    },
    "Aryl-aliphatic (6-5)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]1",
        "min_heavy_subs": 2, # minimum of 2/4 possible ring substituents
    },
    "Aryl-aliphatic (5-6)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]~[*]1",
        "min_heavy_subs": 2, # minimum of 2/4 possible ring substituents
    },
    "Aryl-aliphatic (5-5)": {
        "smarts": "[*]1:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]1",
        "min_heavy_subs": 2, # minimum of 2/4 possible ring substituents
    },

    # Joined nonaryl-nonaryl ring systems
    "Bialiphatic (7-7)": {
        "smarts": "[!a]1~[!a]~[!a]~[!a]~[!a]~[!a]~[!a]1-[!a]2~[!a]~[!a]~[!a]~[!a]~[!a]~[!a]2",
        "min_heavy_subs": 0, # as for aryl-nonaryls
    },
    "Bialiphatic (7-6)": {
        "smarts": "[!a]1~[!a]~[!a]~[!a]~[!a]~[!a]~[!a]1-[!a]2~[!a]~[!a]~[!a]~[!a]~[!a]2",
        "min_heavy_subs": 0, # as for aryl-nonaryls
    },
    "Bialiphatic (7-5)": {
        "smarts": "[!a]1~[!a]~[!a]~[!a]~[!a]~[!a]~[!a]1-[!a]2~[!a]~[!a]~[!a]~[!a]~[!a]2",
        "min_heavy_subs": 1, # as for aryl-nonaryls
    },
    "Bialiphatic (6-6)": {
        "smarts": "[!a]1~[!a]~[!a]~[!a]~[!a]~[!a]1-[!a]2~[!a]~[!a]~[!a]~[!a]~[!a]2",
        "min_heavy_subs": 1, # as for aryl-nonaryls
    },
    "Bialiphatic (6-5)": {
        "smarts": "[!a]1~[!a]~[!a]~[!a]~[!a]~[!a]1-[!a]2~[!a]~[!a]~[!a]~[!a]2",
        "min_heavy_subs": 2, # as for aryl-nonaryls
    },
    "Bialiphatic (5-5)": {
        "smarts": "[!a]1~[!a]~[!a]~[!a]~[!a]1-[!a]2~[!a]~[!a]~[!a]~[!a]2",
        "min_heavy_subs": 2, # as for aryl-nonaryls
    },

    # Aryl-X systems
    # Includes benzamides, reverse benzamides, etc
    "Benzamide (7)": {
        "smarts": "[#7]-[#6](=[#8])-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # minimum of 3/4 ring and C substituents (both C substituents always heavy by defintion)
    },
    "Benzamide (6)": {
        "smarts": "[#7]-[#6](=[#8])-[*]1:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # minimum of 3/4 ring and C substituents (both C substituents always heavy by defintion)
    },
    "Benzamide (5)": {
        "smarts": "[#7]-[#6](=[#8])-[*]1:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # minimum of 3/4 ring and C substituents (both C substituents always heavy by defintion)
    },
    "Thiobenzamide (7)": {
        "smarts": "[#7]-[#6](=[#16])-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # as for benzamides
    },
    "Thiobenzamide (6)": {
        "smarts": "[#7]-[#6](=[#16])-[*]1:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # as for benzamides
    },
    "Thiobenzamide (5)": {
        "smarts": "[#7]-[#6](=[#16])-[*]1:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # as for benzamides
    },
    "Reverse benzamide (7)": {
        "smarts": "[#8]=[#6]-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 2, # minimum of 2/4 ring and N substituents (one N substituent always heavy by definition)
    },
    "Reverse benzamide (6)": {
        "smarts": "[#8]=[#6]-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # minimum of 3/4 ring and N substituents (one N substituent always heavy by definition)
    },
    "Reverse benzamide (5)": {
        "smarts": "[#8]=[#6]-[#7]-[*]1:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # minimum of 3/4 ring and N substituents (one N substituent always heavy by definition)
    },
    "Reverse thiobenzamide (7)": {
        "smarts": "[#16]=[#6]-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 2, # minimum of 2/4 ring and N substituents (one N substituent always heavy by definition)
    },
    "Reverse thiobenzamide (6)": {
        "smarts": "[#16]=[#6]-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # as for reverse benzamides
    },
    "Reverse thiobenzamide (5)": {
        "smarts": "[#16]=[#6]-[#7]-[*]1:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # as for reverse benzamides
    },
    "Sulfinamide (7)": {
        "smarts": "[#16](=[#8])-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 2, # as for reverse benzamides
    },
    "Sulfinamide (6)": {
        "smarts": "[#16](=[#8])-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # as for reverse benzamides
    },
    "Sulfinamide (5)": {
        "smarts": "[#16](=[#8])-[#7]-[*]1:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # as for reverse benzamides
    },
    "Sulfonamide (7)": {
        "smarts": "[#16](=[#8])(=[#8])-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 2, # as for reverse benzamides
    },
    "Sulfonamide (6)": {
        "smarts": "[#16](=[#8])(=[#8])-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # as for reverse benzamides
    },
    "Sulfonamide (5)": {
        "smarts": "[#16](=[#8])(=[#8])-[#7]-[*]1:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 3, # as for reverse benzamides
    },
    "Sulfoxide (7)": {
        "smarts": "[#16](=[#8])-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 4, # minimum of 4/4 ring and S substituents (one S substituent always heavy by definition)
    },
    "Sulfoxide (6)": {
        "smarts": "[#16](=[#8])-[*]1:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 4, # minimum of 4/4 ring and S substituents (one S substituent always heavy by definition)
    },
    "Sulfoxide (5)": {
        "smarts": "[#16](=[#8])-[*]1:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 4, # minimum of 4/4 ring and S substituents (one S substituent always heavy by definition)
    },
    "Sulfone (7)": {
        "smarts": "[#16](=[#8])(=[#8])-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 4, # minimum of 4/4 ring and S substituents (one S substituent always heavy by definition)
    },
    "Sulfone (6)": {
        "smarts": "[#16](=[#8])(=[#8])-[*]1:[*]:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 4, # minimum of 4/4 ring and S substituents (one S substituent always heavy by definition)
    },
    "Sulfone (5)": {
        "smarts": "[#16](=[#8])(=[#8])-[*]1:[*]:[*]:[*]:[*]:1",
        "min_heavy_subs": 4, # minimum of 4/4 ring and S substituents (one S substituent always heavy by definition)
    },

    # Amides
    "Amide": {
        "smarts": "[#7]-[#6](=[#8])",
        "min_heavy_subs": 2, # minimum of 2/4 C and N substituents (one C substituent always heavy by definition)
    },
    "Thioamide": {
        "smarts": "[#7]-[#6](=[#16])",
        "min_heavy_subs": 2, # as for amides
    },
}


class Atropisomers(Workflow):
    """
    A class for running the atropisomers workflow that inherits from .core.Workflow.

    The workflow has four main components:
     - `qm`: Embeds initial conformers of the reactant with RDKit and runs all QM steps (embedding, xTB, CREST, ELIM, CENSO).
     - `cheminformatics`: Identify atropisomeric bonds in the molecule.
     - `gauss`: Runs scans and TS searches, reactant and TS optimizations, and single point calculations with Gaussian.
     - `conclude`: Collects energies, calculates barriers, half-lives, and atropisomer class, and generates figures.

    Parameters:

        Gaussian settings:
            opt_method (str): Optimization method to use for QM calculations.
            opt_basis (str): Basis set to use for QM calculations.
            sp_method (str): Single point method to use for QM calculations.
            sp_basis (str): Basis set to use for single point QM calculations.

        Atropisomers settings:
            scan_steps (int): Number of steps in the dihedral scan.
            scan_stepsize (float): Step size in degrees for the dihedral scan.
            min_complete_scan_steps (int): Minimum number of steps required for a complete scan.
            min_scan_dihedral_change (float): Minimum dihedral change required for a valid scan TS.
            min_ts_displacement_percentage (float): Minimum percentage of displacement for a valid TS.
            max_ts_dihedral_change (float): Maximum dihedral change for a valid TS.
            max_ts_sp (int): Maximum number of TS conformers to run single point calculations on.
            ts_opt_energy_window (float): Energy window (in Hartrees) for accepting TS conformers for optimization.
            frac_ts_opt (float): Fraction of TS conformers to optimize.
            min_ts_opt (int): Minimum number of TS conformers to optimize.
            max_ts_opt (int): Maximum number of TS conformers to optimize.
            ts_frozen_extra (int): Extra number of frozen TS conformers to optimize.
            qh_entropy (str): Method for quasi-harmonic entropy correction to apply via Goodvibes ("Grimme" or "Truhlar").
            vib_freq_scale_factor (float): Vibrational frequency scaling factor to apply via Goodvibes.
            m (float): Slope for barrier regression.
            c (float): Intercept for barrier regression.
     
    """

    def __init__(
            self, 

            # Gaussian settings
            opt_method: str = "bp86 empiricaldispersion=GD3BJ",
            opt_basis: str = "def2svp",
            sp_method: str = "PW6B95D3",
            sp_basis: str = "def2tzvp",

            # Atropisomers settings
            scan_steps: int = 30,
            scan_stepsize: float = 0.2,
            min_complete_scan_steps: int = 6,
            min_scan_dihedral_change: float = 45.0,
            min_ts_displacement_percentage: float = 2.0,
            max_ts_dihedral_change: float = 45.0,
            max_ts_sp: int = 100,
            ts_opt_energy_window: float = 0.002,
            frac_ts_opt: float = 0.1,
            min_ts_opt: int = 5,
            max_ts_opt: int = 25,
            ts_frozen_extra: int = 5,
            qh_entropy: str = "Grimme",
            vib_freq_scale_factor: float = 1.0,
            m: float = 0.980,
            c: float = 1.458,  
            *args, 
            **kwargs):

        super().__init__(*args, **kwargs)

        # Initialize dictionaries to hold GIC dihedrals for each identified rotatable bond
        self.all_gic_1: dict = {}
        self.all_gic_2: dict = {}

        # Initialize a dictionary to hold the substructure pattern for each identified rotatable bond
        self.substructures: dict = {}

        # Initialize an attribute to hold central atoms for coupled torsions
        self.central_atoms: dict = {}

        # Define the structures dataframe
        self.structures = pd.Series(
            {
                "file_idx": "reactant_censo",
                "mol": self.mol,
                "confs": self.conf2mol(self.mol, self.num_confs, self.seed),
                "exported_confs": [],
                "stable_xyz": None,
                "substructure": "reactant",
                "E (hartree)": np.inf,
                "E_SP (hartree)": np.inf,
                "G_SP (hartree)": np.inf,
                "QH_G_SP (hartree)": np.inf,
                "ΔG activation (kcal/mol)": None,
                "ΔG activation_r (kcal/mol)": None,
            }
        ).to_frame().T

        # Define Gaussian method
        self.opt_method = opt_method
        self.opt_basis = opt_basis
        self.sp_method = sp_method
        self.sp_basis = sp_basis

        # Define thresholds for GIC scans
        self.scan_steps = scan_steps
        self.scan_stepsize = scan_stepsize
        self.min_complete_scan_steps = min_complete_scan_steps
        self.min_scan_dihedral_change = min_scan_dihedral_change

        # Define thresholds for determining TS validity
        self.min_ts_displacement_percentage = min_ts_displacement_percentage
        self.max_ts_dihedral_change = max_ts_dihedral_change

        # Define thresholds for processing conformers in the workflow
        self.max_ts_sp = max_ts_sp
        self.ts_opt_energy_window = ts_opt_energy_window
        self.frac_ts_opt = frac_ts_opt
        self.min_ts_opt = min_ts_opt
        self.max_ts_opt = max_ts_opt
        self.ts_frozen_extra = ts_frozen_extra

        # Define Goodvibes parameters
        self.qh_entropy = qh_entropy
        self.vib_freq_scale_factor = vib_freq_scale_factor

        # Define barrier regression thresholds
        self.m = m
        self.c = c

    # Cheminformatics helper functions

    def discover_ring_axis_atoms(self, mol, neighbors, match_indexes, axis_atoms):
        """
        Discover ring axis atoms in a molecular structure; this function identifies substituent atoms and lone pairs
        around a given set of neighboring atoms in a molecule. It distinguishes between substituents and lone pairs
        based on whether the neighboring atoms are part of a matched SMARTS index.

        Parameters:
            mol (rdkit.Chem.Mol): RDKit molecule object.
            neighbors (list): A list of indices representing neighboring atoms.
            match_indexes (set): A set of indices representing matched SMARTS atoms.
            axis_atoms (list): A list to which the indices of axis atoms will be appended.

        Returns:
            axis_atoms (list): Updated list of axis atom indices.
            lp_count (int): Count of lone pairs found.
            proton_count (int): Count of protons found.
        """

        # Initialize counts
        lp_count = 0
        proton_count = 0

        # Iterate over the neighboring atoms
        for idx in neighbors:
            atom = mol.GetAtomWithIdx(idx)
            is_exocyclic = False

            # If the neighbor is not part of the matched SMARTS indexes, record it as a substituent
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() not in match_indexes:
                    axis_atoms.append(neighbor.GetIdx())
                    is_exocyclic = True
                    # Add to the proton count if it's a hydrogen
                    if neighbor.GetSymbol() == "H":
                        proton_count += 1
                    break

            # If no substituent is found (break is not activated), we record e a lone pair. For example,
            # the original neighbor atom might be a heteroatom in the ring, and thus have no substituent.
            if not is_exocyclic:
                lp_count += 1
                axis_atoms.append(atom.GetIdx())
    
        return axis_atoms, lp_count, proton_count

    def get_ring_sizes(self, mol, atom_or_bond):
        """
        Returns the sizes of all rings that a given atom or bond is part of within a molecule.

        Parameters:
            mol (rdkit.Chem.Mol): RDKit molecule object (with explicit hydrogens).
            atom_or_bond (rdkit.Chem.Atom or rdkit.Chem.Bond): The atom or bond for which to determine ring membership and sizes.

        Returns:
            list of int: A list containing the sizes of all rings that include the specified atom or bond.
        """

        # Get ring information for the molecule to identify which atoms/bonds are part of rings and the size of those rings.
        ring_info = mol.GetRingInfo()
        bond_rings = ring_info.BondRings() # tuples of bond indexes per ring (SSSR)
        atom_rings = ring_info.AtomRings() # tuples of atom indexes per ring (SSSR)

        # Get all ring sizes for rings that the atom/bond is part of
        if isinstance(atom_or_bond, Chem.Bond):
            ring_sizes = [len(r) for r in bond_rings if atom_or_bond.GetIdx() in r]
        elif isinstance(atom_or_bond, Chem.Atom):
            ring_sizes = [len(r) for r in atom_rings if atom_or_bond.GetIdx() in r]
        else:
            raise ValueError("Input must be an RDKit Atom or Bond object")

        return ring_sizes

    def count_hydrogens(self, mol, axis_atoms):
        """
        Count the number of hydrogen atoms in a specified list of axis atoms.
        If the axis contains a duplicate atom, this is because there is no atom,
        i.e., a lone pair, and thus we do not count it as a hydrogen.

        Parameters:
            mol (rdkit.Chem.Mol): RDKit molecule object.
            axis_atoms (list): A list of indices representing the axis atoms.

        Returns:
            proton_count (int): The count of hydrogen atoms among the specified axis atoms.
        """

        # Initialize count
        proton_count = 0

        # Initialize tracker for duplicate atoms
        seen_atoms = set()

        # Count hydrogens
        for idx in axis_atoms:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "H" and idx not in seen_atoms:
                seen_atoms.add(idx)
                proton_count += 1

        return proton_count

    def is_atropisomeric(self, lone_pairs_1, lone_pairs_2, protons_1, protons_2, axis_1_atoms, axis_2_atoms, min_heavy_subs):
        """
        Determines if a molecular structure is potentially atropisomeric based on the local steric environment
        (number of lone pairs, number of protons, and the configuration of the atoms around the specified axes).
        We do not recommend a scan if the total number of heavy atom substituents does not exceed the threshold
        set for the substructure, or if there is insufficient bulk on the axes, as this indicates that the system
        is likely too unhindered for atropisomerism.

        Parameters:
            lone_pairs_1 (int): Number of lone pairs on the first unit.
            lone_pairs_2 (int): Number of lone pairs on the second unit.
            protons_1 (int): Number of protons on the first unit.
            protons_2 (int): Number of protons on the second unit.
            axis_1_atoms (list): List of atom indices for the first axis.
            axis_2_atoms (list): List of atom indices for the second axis.
            min_heavy_subs (int): Minimum number of heavy atom substituents required for atropisomerism.

        Returns:
            bool: True if the structure is considered atropisomeric, False otherwise.
        """

        # If steric filters are disabled, always return True
        if not self.filter:
            print(" - no steric filters applied.")
            return True

        # Calculate the number of heavy and non-heavy substituents. We can always assume the total number of substituents is 4.
        # For example, biaryls use the out-of-ring substituents, of which there are two, while non-aryl moieties, like amides,
        # sulfoxides and aliphatic rings, use the direct neighbors, of which there are two. Thus, there are always 4 total.
        total_substituents = 4 
        non_heavy_substituents = lone_pairs_1 + lone_pairs_2 + protons_1 + protons_2
        heavy_substituents = total_substituents - non_heavy_substituents

        # Do not recommend a scan if the number of heavy atoms does not exceed the required threshold
        if heavy_substituents < min_heavy_subs:
            print(f" - not enough heavy atom substituents (<{min_heavy_subs}): skipping the calculation")
            return False

        # Recommend a scan if there are sufficient substituents (sufficient bulk) on both axes
        elif len(axis_1_atoms) == 3 and len(axis_2_atoms) == 3:
            print(f" - sufficient bulk for atropisomerism: initialize GIC scan around bond {axis_1_atoms[0]}-{axis_2_atoms[0]} with atoms {axis_1_atoms} and {axis_2_atoms}")
            return True

        # If the axis atoms are not correctly defined, return False
        else:
            print(" - invalid pattern for atropisomerism (axes not identified): skipping the calculation")
            return False

    # Cheminformatics functions for each substructure

    def analyze_biring(self, mol, match_indexes, min_heavy_subs):
        """
        Analyzes a molecular structure with a rings (aryl or non-aryl) attached to one another to determine
        the atom indexes of the central rotatable bond, as well as the indexes of the atoms to use for scans.
        For aryl or aryl-like (consisting of sp2 atoms around the rotatable bond atom) these are the exocyclic
        second neighbors of the ring. Otherwise, these are the direct neighbors of the rotatable bond atom.
        These substituents are required for GIC scans of the rotatable bond in both directions. Additionally,
        this method tracks the number of lone pairs and protons around the rotatable bond and determines whether
        a GIC scan is necessary or whether atropisomerism is unlikely, for example due to lack of steric hindrance.

        Parameters:
            mol (rdkit.Chem.Mol): RDKit molecule object (with explicit hydrogens).
            match_indexes (list): Atom indices of the matched aryl-X SMARTS substructure.
            min_heavy_subs (int): Minimum number of heavy atom substituents required for atropisomerism.

        Returns:
            result (bool): True if the bond is considered hindered enough for atropisomerism.
            axis_1_atoms, axis_2_atoms (tuples): For each ring, these correspond to a tuple of three atom:
            the atom that is part of the rotatable bond and its two selected exocyclic second neighbors.
        """

        ### Step 1. Identify the rotatable bond and its neighbors

        # Identify all 5, 6 or 7-membered rings in the molecule
        ring_5_smarts = "[*]1~[*]~[*]~[*]~[*]1"
        ring_6_smarts = "[*]1~[*]~[*]~[*]~[*]~[*]1"
        ring_7_smarts = "[*]1~[*]~[*]~[*]~[*]~[*]~[*]1"
        smarts_5_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(ring_5_smarts))
        smarts_6_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(ring_6_smarts))
        smarts_7_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(ring_7_smarts))
        all_ring_smarts = smarts_5_matches + smarts_6_matches + smarts_7_matches

        # Keep only rings fully contained within the original SMARTS match
        included_ring_smarts = [m for m in all_ring_smarts if set(m).issubset(match_indexes)]

        # Initialize the rotatable bond as empty
        rotatable_bond_indexes = None

        # Iterate through every combination of match-included rings
        for i, j in combinations(range(len(included_ring_smarts)), 2):
            ring_i = included_ring_smarts[i]
            ring_j = included_ring_smarts[j]

            # Check if the two rings have an atom each that connect to form a bond between them
            for atom_i in ring_i:
                for atom_j in ring_j:
                    bond = mol.GetBondBetweenAtoms(atom_i, atom_j)

                    # If they do, we have identified the rotatable bond.
                    if bond is not None:
                        rotatable_bond_indexes = [atom_i, atom_j]
                        break

        # If no rotatable bond is found, return a pattern failure.
        if not rotatable_bond_indexes:
            print(" - invalid pattern for atropisomerism: skipping the calculation")
            return False, [], []

        # Check if the rotatable bond is part of another ring (e.g., through a fused ring system or macrocycle).
        # If so, only allow it to be part of larger rings (7-membered or more) as these may be flexible enough
        # to perform a GIC scan and identify a rotational transition state.
        ring_sizes = self.get_ring_sizes(mol, mol.GetBondBetweenAtoms(*rotatable_bond_indexes))
        if any(size <= 6 for size in ring_sizes):
            print(" - rotatable bond is part of a small fused ring (<= 6 members): skipping the calculation")
            return False, [], []

        # Define each atom in the rotatable bond
        atom_1 = mol.GetAtomWithIdx(rotatable_bond_indexes[0])
        atom_2 = mol.GetAtomWithIdx(rotatable_bond_indexes[1])

        # Identify if there are any non-ring substituents on the rotatable bond atoms (for non-aryl rings only)
        non_ring_sub_1, non_ring_sub_2 = None, None
        for neighbor in atom_1.GetNeighbors():
            if neighbor.GetIdx() not in match_indexes:
                non_ring_sub_1 = neighbor.GetIdx()
        for neighbor in atom_2.GetNeighbors():
            if neighbor.GetIdx() not in match_indexes:
                non_ring_sub_2 = neighbor.GetIdx()

        # Define the in-ring neighbors for each atom in the rotatable bond
        neighbors_1 = [n.GetIdx() for n in atom_1.GetNeighbors() if n.GetIdx() not in rotatable_bond_indexes and n.GetIdx() != non_ring_sub_1]
        neighbors_2 = [n.GetIdx() for n in atom_2.GetNeighbors() if n.GetIdx() not in rotatable_bond_indexes and n.GetIdx() != non_ring_sub_2]

        ### Step 2. Define the indexes of the axis atoms required for GIC scans
        # For each ring, we will check the valence of the in-ring neighbors of the atom in the rotatable bond.
        # If the neighbor is sp2, we can use the ortho substituent. Otherwise, we use the neighbors itself. If
        # no exocyclic neighbor is found (e.g. there is a heteroatom in the ring), we treat it as a possible lone
        # pair site, and use the in-ring atom (first neighbor) instead.

        # Initialize each axis with the relevant ring atom from the rotatable bond
        axis_1_atoms = [rotatable_bond_indexes[0]]
        axis_2_atoms = [rotatable_bond_indexes[1]]

        # Check the first axis
        lone_pairs_1, protons_1 = 0, 0
        for neighbor_idx in neighbors_1:
            neighbor = mol.GetAtomWithIdx(neighbor_idx)
            # If the neighbor is sp2, we can use the ortho substituent (if it exists)
            if neighbor.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                axis_atoms, lp, protons = self.discover_ring_axis_atoms(mol, [neighbor_idx], match_indexes, [])
                axis_1_atoms.extend(axis_atoms)
                protons_1 += protons
                lone_pairs_1 += lp
            # Otherwise, use the neighbor itself
            else:
                axis_1_atoms.append(neighbor_idx)
                
        # Check the second axis
        lone_pairs_2, protons_2 = 0, 0
        for neighbor_idx in neighbors_2:
            neighbor = mol.GetAtomWithIdx(neighbor_idx)
            # If the neighbor is sp2, we can use the ortho substituent (if it exists)
            if neighbor.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                axis_atoms, lp, protons = self.discover_ring_axis_atoms(mol, [neighbor_idx], match_indexes, [])
                axis_2_atoms.extend(axis_atoms)
                protons_2 += protons
                lone_pairs_2 += lp
            # Otherwise, use the neighbor itself
            else:
                axis_2_atoms.append(neighbor_idx)

        # Count the number of hydrogens on all GIC-defining atoms
        proton_count = self.count_hydrogens(mol, axis_1_atoms + axis_2_atoms)
        assert proton_count == (protons_1 + protons_2), "Mismatch in proton count"
        
        ### Step 3. Final analysis
    
        # 1-index the axis atoms
        axis_1_atoms = [idx + 1 for idx in axis_1_atoms]
        axis_2_atoms = [idx + 1 for idx in axis_2_atoms]

        # Determine whether an atropisomeric scan is appropriate
        result = self.is_atropisomeric(lone_pairs_1, lone_pairs_2, protons_1, protons_2, axis_1_atoms, axis_2_atoms, min_heavy_subs)
        if result:
            return result, axis_1_atoms, axis_2_atoms
        else:
            return False, [], []

    def analyze_diaryl(self, mol, match_indexes, min_heavy_subs):
        """
        Analyzes a molecular structure with two aryl rings attached to each other via a heteroatom,
        e.g., O, N, S, S=O or S(=O)(=O) (diaryl ethers, thioethers, amines, sulfoxides and sulfones)
        to determine the atom indexes of the central atom and the two adjacent rotatable bonds in the
        system, as well as the indexes of their exocyclic second neighbors (two on each ring). These
        substituents are required for GIC scans of the coupled torsion in both directions. Additionally,
        this method tracks the number of lone pairs and protons around the rotatable bonds and determines
        whether a GIC scan is necessary or whether atropisomerism is unlikely, for example due to lack of
        steric hindrance.

        Parameters:
            mol (rdkit.Chem.Mol): RDKit molecule object (with explicit hydrogens).
            match_indexes (list): Atom indices of the matched aryl-X SMARTS substructure.
            min_heavy_subs (int): Minimum number of heavy atom substituents required for atropisomerism.

        Returns:
            result (bool): True if the bond is considered hindered enough for atropisomerism.
            axis_1_atoms, axis_2_atoms (tuples): For each ring, these correspond to a tuple of three atom:
            the atom that is part of the rotatable bond and its two selected exocyclic second neighbors.
        """

        ### Step 1. Identify the rotatable bonds, the central atom, and their neighbors

        # Identify all 5, 6 or 7-membered rings in the molecule
        ring_5_smarts = "[*]1~[*]~[*]~[*]~[*]1"
        ring_6_smarts = "[*]1~[*]~[*]~[*]~[*]~[*]1"
        ring_7_smarts = "[*]1~[*]~[*]~[*]~[*]~[*]~[*]1"
        smarts_5_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(ring_5_smarts))
        smarts_6_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(ring_6_smarts))
        smarts_7_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(ring_7_smarts))
        all_ring_smarts = smarts_5_matches + smarts_6_matches + smarts_7_matches

        # Keep only rings fully contained within the original SMARTS match
        included_ring_smarts = [m for m in all_ring_smarts if set(m).issubset(match_indexes)]

        # Initialize the rotatable atoms/bonds as empty
        rotatable_bond_indexes = (None, None)
        rotatable_atom_indexes = []
        central_atom_idx = None

        # Iterate through every combination of match-included rings
        for i, j in combinations(range(len(included_ring_smarts)), 2):
            ring_i = included_ring_smarts[i]
            ring_j = included_ring_smarts[j]

            # Check if the two rings have an atom each that connects
            # to some central atom to form two bonds between them
            for atom_i in ring_i:
                for atom_j in ring_j:
                    for atom_c in match_indexes:
                        if atom_c != atom_i and atom_c != atom_j:
                            bond_1 = mol.GetBondBetweenAtoms(atom_i, atom_c)
                            bond_2 = mol.GetBondBetweenAtoms(atom_j, atom_c)

                            # Ensure both bonds are single bonds
                            if bond_1 is not None and bond_1.GetBondType() != Chem.rdchem.BondType.SINGLE:
                                bond_1 = None
                            if bond_2 is not None and bond_2.GetBondType() != Chem.rdchem.BondType.SINGLE:
                                bond_2 = None

                            # If they do, we have identified the rotatable bond
                            if bond_1 is not None and bond_2 is not None:
                                rotatable_atom_indexes = [atom_i, atom_j, atom_c]
                                rotatable_bond_indexes = [(atom_i, atom_c), (atom_c, atom_j)]
                                rotatable_bond_idx = (atom_i, atom_j)
                                central_atom_idx = atom_c
                                break

        # If two rotatable bonds between three unique atoms are not found, return a pattern failure
        if not rotatable_bond_indexes or len(rotatable_bond_indexes) != 2 or len(rotatable_atom_indexes) !=3:
            print(" - invalid pattern for atropisomerism: skipping the calculation")
            return False, [], []

        # Check if the rotatable bonds are part of another ring (e.g., through a fused ring system or macrocycle).
        # If so, only allow it to be part of larger rings (7-membered or more) as these may be flexible enough
        # to perform a GIC scan and identify a rotational transition state.
        for bond_idx in rotatable_bond_indexes:
            bond = mol.GetBondBetweenAtoms(*bond_idx)
            ring_sizes = self.get_ring_sizes(mol, bond)
            if any(size <= 6 for size in ring_sizes):
                print(" - rotatable bond is part of a small fused ring (<= 6 members): skipping the calculation")
                return False, [], []

        # If no central atom is found, return a pattern failure
        if central_atom_idx is None:
            print(" - invalid pattern for atropisomerism (no out-of-ring central atom): skipping the calculation")
            return False, [], []

        # Store the central atom with respect to the other atoms to use later
        self.central_atoms[tuple(rotatable_bond_idx)] = central_atom_idx

        # Define the in-ring neighbors for each atom in the rotatable bond (excluding the central atom)
        atom_1_idx = rotatable_bond_indexes[0][0]
        atom_2_idx = rotatable_bond_indexes[1][1]
        atom_1 = mol.GetAtomWithIdx(atom_1_idx)
        atom_2 = mol.GetAtomWithIdx(atom_2_idx)
        neighbors_1 = [n.GetIdx() for n in atom_1.GetNeighbors() if n.GetIdx() not in rotatable_bond_indexes and n.GetIdx() != central_atom_idx]
        neighbors_2 = [n.GetIdx() for n in atom_2.GetNeighbors() if n.GetIdx() not in rotatable_bond_indexes and n.GetIdx() != central_atom_idx]

        ### Step 2. Define the indexes of the axis atoms required for GIC scans

        # Initialize each axis with the relevant ring atom from the rotatable bond
        axis_1_atoms = [atom_1_idx]
        axis_2_atoms = [atom_2_idx]

        # Find the exocyclic second substituents of the first ring by searching through the central atom
        # neighbors and storing the first "exocyclic" (not in the biaryl, but can be in another ring) neighbor
        # found. If no exocyclic neighbor is found, treat as a possible lone pair site, and include the in-ring atom
        # (first neighbor). For example, if there is a heteroatom (e.g., O) in the ring, the exocyclic substituent
        # is the lone-pair and we include the atom index of the oxygen itself. Also count if it's a proton.
        axis_1_atoms, lone_pairs_1, protons_1 = self.discover_ring_axis_atoms(mol, neighbors_1, match_indexes, axis_1_atoms)
        axis_2_atoms, lone_pairs_2, protons_2 = self.discover_ring_axis_atoms(mol, neighbors_2, match_indexes, axis_2_atoms)

        # Count the number of hydrogens on all GIC-defining atoms
        proton_count = self.count_hydrogens(mol, axis_1_atoms + axis_2_atoms)
        assert proton_count == (protons_1 + protons_2), "Mismatch in proton count"
        
        ### Step 3. Final analysis

        # 1-index the axis atoms
        axis_1_atoms = [idx + 1 for idx in axis_1_atoms]
        axis_2_atoms = [idx + 1 for idx in axis_2_atoms]

        # Determine whether an atropisomeric scan is appropriate
        result = self.is_atropisomeric(lone_pairs_1, lone_pairs_2, protons_1, protons_2, axis_1_atoms, axis_2_atoms, min_heavy_subs)
        if result:
            return result, axis_1_atoms, axis_2_atoms
        else:
            return False, [], []

    def analyze_aryl_X(self, mol, match_indexes, min_heavy_subs):
        """
        Analyzes a molecular structure with an aryl ring attached to specific non-aromatic units; amides (benzamides),
        thioamides (thiobenzamide), their reversed forms (reversed (thio)benzamides), sulfones/sulfoxides, and
        sulfonamides/sulfinamides, to determine the atom indexes of the central rotatable bond, as well as the indexes
        of the two exocyclic second neighbors of the aromatic ring and the two direct neighbors of the out-of-ring atom
        (e.g., the =O, =S, or R-group). These substituents are required for GIC scans of the rotatable bond in both
        directions. Additionally, this method tracks the number of lone pairs and protons around the rotatable bond and
        determines whether a GIC scan is necessary or whether atropisomerism is unlikely, for example due to lack of
        steric hindrance.

        Parameters:
            mol (rdkit.Chem.Mol): RDKit molecule object (with explicit hydrogens).
            match_indexes (list): Atom indices of the matched aryl-X SMARTS substructure.
            min_heavy_subs (int): Minimum number of heavy atom substituents required for atropisomerism.

        Returns:
            result (bool): True if the bond is considered hindered enough for atropisomerism.
            in_ring_axis_atoms, out_ring_axis_atoms (tuples): For the aryl ring and non-aromatic unit, these
            correspond to a tuple of three atoms indexes: the atom that is part of the rotatable bond and its
            selected first or second neighbors.
        """

        ### Step 1. Identify the rotatable bond and its neighbors

        # Identify all substructures of the molecule corresponding to a 5/6/7-membered aromatic ring.
        aryl_5_smarts = "[*]1:[*]:[*]:[*]:[*]:1"
        aryl_6_smarts = "[*]1:[*]:[*]:[*]:[*]:[*]:1"
        aryl_7_smarts = "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1"
        smarts_5_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(aryl_5_smarts))
        smarts_6_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(aryl_6_smarts))
        smarts_7_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(aryl_7_smarts))
        all_ring_smarts = smarts_5_matches + smarts_6_matches + smarts_7_matches

        # Identify all substructures of the molecule corresponding to a 5/6/7-membered aromatic ring with one substituent.
        # For example, benzene returns six matches, each consisting of the ring atom indexes and the index of one of the hydrogens.
        aryl_5_sub_1_smarts = "[*]1:[*]:[*]:[*]:[*]:1-[*]"
        aryl_6_sub_1_smarts = "[*]1:[*]:[*]:[*]:[*]:[*]:1-[*]"
        aryl_7_sub_1_smarts = "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[*]"
        smarts_5_sub_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(aryl_5_sub_1_smarts))
        smarts_6_sub_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(aryl_6_sub_1_smarts))
        smarts_7_sub_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(aryl_7_sub_1_smarts))
        all_ring_sub_smarts = smarts_5_sub_matches + smarts_6_sub_matches + smarts_7_sub_matches

        # Keep only rings fully contained within the original SMARTS embedding. This should only be one ring for each type.
        included_ring = [m for m in all_ring_smarts if set(m).issubset(match_indexes)][0]
        included_sub_ring = [m for m in all_ring_sub_smarts if set(m).issubset(match_indexes)][0]

        # Find the out-of-ring atom of the rotatable bond by identifying the atom that is in included_sub_ring but not in included_ring.
        out_ring_atom_idx = list(set(included_sub_ring) - set(included_ring))[0]
        out_ring_atom = mol.GetAtomWithIdx(out_ring_atom_idx)

        # Find the in-ring atom of the rotatable bond by identifying the atom in included_sub_ring that is connected to the out-of-ring atom.
        in_ring_atom_idx = None
        for idx in included_sub_ring:
            if idx != out_ring_atom_idx and mol.GetBondBetweenAtoms(idx, out_ring_atom_idx) is not None:
                in_ring_atom_idx = idx
                break
        in_ring_atom = mol.GetAtomWithIdx(in_ring_atom_idx)

        # If no rotatable bond is found, return a pattern failure
        if out_ring_atom_idx is None or in_ring_atom_idx is None:
            print(" - invalid pattern for atropisomerism (no rotatable bond found): skipping the calculation")
            return False, [], []

        # Define the rotatable bond indexes
        rotatable_bond_indexes = [in_ring_atom_idx, out_ring_atom_idx]

        # Check if the rotatable bond is part of another ring (e.g., through a fused ring system or macrocycle).
        # If so, only allow it to be part of larger rings (7-membered or more) as these may be flexible enough
        # to perform a GIC scan and identify a rotational transition state.
        ring_sizes = self.get_ring_sizes(mol, mol.GetBondBetweenAtoms(*rotatable_bond_indexes))
        if any(size <= 6 for size in ring_sizes):
            print(" - rotatable bond is part of a small fused ring (<= 6 members): skipping the calculation")
            return False, [], []

        # Define the neighbors for each atom in the rotatable bond 
        atom_in_ring_neighbors = [n.GetIdx() for n in in_ring_atom.GetNeighbors() if n.GetIdx() not in rotatable_bond_indexes]
        atom_out_ring_neighbors = [n.GetIdx() for n in out_ring_atom.GetNeighbors() if n.GetIdx() not in rotatable_bond_indexes]

        # If an axis atom has been deprotonated, we identify only one neighbor. In this case, the external neighbor is
        # the lone pair, but this cannot use in the scan. Instead, we add the first neighbor to the axis again and record
        # it as an additional lone pair. Note that this will result in the two scans being identical for this axis.
        if len(atom_out_ring_neighbors) == 1:
            print(" - neighbor atom has been deprotonated: repeating one neighbor in the axis.")
            atom_out_ring_neighbors.append(atom_out_ring_neighbors[0])
            additional_lps = 1
        else:
            additional_lps = 0

        # If an axis atom has been protonated, we identify three neighbors. In this case, we identify a random proton
        # and remove it from the axis. There will always be at least one proton, since it was protonated.
        if len(atom_out_ring_neighbors) == 3:
            print(" - neighbor atom has been protonated: removing one proton from the axis.")
            for idx in atom_out_ring_neighbors:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == "H":
                    atom_out_ring_neighbors.remove(idx)
                    break

        # In the specific case of a sulfone, there will be three out-ring atom neighbors. In this
        # case, we identify the indexes of the two =O atoms, and remove one of them (randomly) from
        # the neighbors list. The scan is then performed using the other =O atom and the R-group.
        if len(atom_out_ring_neighbors) == 3 and out_ring_atom.GetSymbol() == "S":
            oxygen_neighbors = []
            for idx in atom_out_ring_neighbors:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == "O":
                    oxygen_neighbors.append(idx)
            # Remove an arbitrary =O from the neighbors list
            if len(oxygen_neighbors) == 2:
                atom_out_ring_neighbors.remove(oxygen_neighbors[0])
    
        ### Step 2. Define the indexes of the axis atoms required for GIC scans

        # Initialize each axis with the relevant atom from the rotatable bond
        in_ring_axis_atoms = [in_ring_atom.GetIdx()]
        out_ring_axis_atoms = [out_ring_atom.GetIdx()] 

        # Find the exocyclic second neighbours of the ring by searching through the central atom neighbors. If no exocyclic
        # neighbor is found, treat as a possible lone pair site, and include the in-ring atom. Also count if it's a proton.
        in_ring_axis_atoms, in_ring_lp, in_ring_protons = self.discover_ring_axis_atoms(mol, atom_in_ring_neighbors, match_indexes, in_ring_axis_atoms)

        # Add the neighbors of the out-of-ring (non-aromatic) unit to the axis. For benzamides and thiobenzamides 
        # this corresponds to the =O/=S atom and the N(R2) atom. For reverse benzamides and reverse thiobenzamides,
        # this corresponds to the R-substituent of the N and the carbon of the C=O or C=S bond. For sulfoxides, this
        # corresponds to the =O and the R-group.
        out_ring_axis_atoms.extend(atom_out_ring_neighbors)

        # Count how many of the out-of-ring substituents are hydrogens
        out_ring_protons = self.count_hydrogens(mol, out_ring_axis_atoms)

        # In theory, there cannot be lone pairs on these functional groups, unless they have been
        # deprotonated. To be sure, we can approximate the number of lone pairs on the out-of-ring
        # substituents by subtracting the number of identified substituents from three (since we know
        # we have a benzamide, reverse benzamide, or sulfoxide, we can assume the out-of-ring atom is
        # a heteroatom (O, S, N) with up to three bonds)). Lone pairs resulting from deprotonation
        # are accounted for via the `additional_lps` variable, which is added to the count.
        out_ring_lp = 3 - len(out_ring_axis_atoms) + additional_lps

        # Count the number of hydrogens on all GIC-defining atoms
        proton_count = self.count_hydrogens(mol, in_ring_axis_atoms + out_ring_axis_atoms)
        assert proton_count == (in_ring_protons + out_ring_protons), "Mismatch in proton count"

        ### Step 3. Final analysis

        # 1-index the axis atoms
        axis_1_atoms = [idx + 1 for idx in in_ring_axis_atoms]
        axis_2_atoms = [idx + 1 for idx in out_ring_axis_atoms]

        # Determine whether an atropisomeric scan is appropriate
        result = self.is_atropisomeric(in_ring_lp, out_ring_lp, in_ring_protons, out_ring_protons, axis_1_atoms, axis_2_atoms, min_heavy_subs)
        if result:
            return result, axis_1_atoms, axis_2_atoms
        else:
            return False, [], []

    def analyze_coupled_aryl_X(self, mol, match_indexes, min_heavy_subs):
        """
        Analyzes a molecular structure with an aryl ring attached to specific non-aromatic units; amides
        (benzamides), thioamides (thiobenzamide), and their reversed forms (reversed (thio)benzamides), to
        determine the atom indexes of the central atom and the two adjacent rotatable bonds in the system,
        as well as the indexes of the two exocyclic second neighbors of the aromatic ring and the two direct
        neighbors of the out-of-ring atom (e.g., the =O or R-group). These substituents are required for
        GIC scans of the coupled torsion in both directions. Additionally, this method tracks the number of
        lone pairs and protons around the rotatable bonds and determines whether a GIC scan is necessary or
        whether atropisomerism is unlikely, for example due to lack of steric hindrance.

        Parameters:
            mol (rdkit.Chem.Mol): RDKit molecule object (with explicit hydrogens).
            match_indexes (list): Atom indices of the matched aryl-X SMARTS substructure.
            min_heavy_subs (int): Minimum number of heavy atom substituents required for atropisomerism.

        Returns:
            result (bool): True if the bond is considered hindered enough for atropisomerism.
            in_ring_axis_atoms, out_ring_axis_atoms (tuples): For the aryl ring and non-aromatic unit, these
            correspond to a tuple of three atoms indexes: the atom that is part of the rotatable bond and its
            selected first or second neighbors.
        """

        ### Step 1. Identify the rotatable bonds, the central atom, and their neighbors

        # Identify all substructures of the molecule corresponding to a 5/6/7-membered aromatic ring.
        aryl_5_smarts = "[*]1:[*]:[*]:[*]:[*]:1"
        aryl_6_smarts = "[*]1:[*]:[*]:[*]:[*]:[*]:1"
        aryl_7_smarts = "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1"
        smarts_5_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(aryl_5_smarts))
        smarts_6_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(aryl_6_smarts))
        smarts_7_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(aryl_7_smarts))
        all_ring_smarts = smarts_5_matches + smarts_6_matches + smarts_7_matches

        # Identify all substructures of the molecule corresponding to the non-aromatic unit
        amide_smarts = "[#7]-[#6](=[#8])"
        thioamide_smarts = "[#7]-[#6](=[#16])"
        amide_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(amide_smarts))
        thioamide_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(thioamide_smarts))
        all_X_smarts = amide_matches + thioamide_matches

        # Keep only rings fully contained within the original SMARTS embedding. This should only be one ring for each type.
        included_ring = [m for m in all_ring_smarts if set(m).issubset(match_indexes)]
        included_X = [m for m in all_X_smarts if set(m).issubset(match_indexes)]
    
        # Initialize the rotatable atoms/bonds as empty
        rotatable_bond_indexes = (None, None)
        rotatable_atom_indexes = []
        in_ring_atom_idx = None
        out_ring_atom_idx = None
        central_atom_idx = None

        # Iterate through every combination of ring-X matches
        for ring, X in product(included_ring, included_X):

            # Check if the two units have a connection involving one atom from the ring and two from the X group
            for atom_i in ring:
                for atom_j in X:
                    for atom_c in match_indexes:
                        if atom_c != atom_i != atom_j and atom_c in X:
                            bond_1 = mol.GetBondBetweenAtoms(atom_i, atom_c)
                            bond_2 = mol.GetBondBetweenAtoms(atom_j, atom_c)

                            # Ensure both bonds are single bonds, to avoid capturing the C=O or C=S bond
                            if bond_1 is not None and bond_1.GetBondType() != Chem.rdchem.BondType.SINGLE:
                                bond_1 = None
                            if bond_2 is not None and bond_2.GetBondType() != Chem.rdchem.BondType.SINGLE:
                                bond_2 = None

                            # If they do, we have identified the rotatable bond.
                            if bond_1 is not None and bond_2 is not None:
                                print(f" - found rotatable bond between ring atom {atom_i} and amide atom {atom_j}, with central atom {atom_c}")
                                rotatable_atom_indexes = [atom_i, atom_j, atom_c]
                                rotatable_bond_indexes = [(atom_i, atom_c), (atom_c, atom_j)]
                                in_ring_atom_idx = atom_i
                                out_ring_atom_idx = atom_j
                                central_atom_idx = atom_c
                                break

        # If two rotatable bonds between three unique atoms are not found, return a pattern failure
        if not rotatable_bond_indexes or len(rotatable_bond_indexes) != 2 or len(rotatable_atom_indexes) !=3:
            print(" - invalid pattern for atropisomerism: skipping the calculation")
            return False, [], []

        # Check if the rotatable bonds are part of another ring (e.g., through a fused ring system or macrocycle).
        # If so, only allow it to be part of larger rings (7-membered or more) as these may be flexible enough
        # to perform a GIC scan and identify a rotational transition state.
        for bond_idx in rotatable_bond_indexes:
            bond = mol.GetBondBetweenAtoms(*bond_idx)
            ring_sizes = self.get_ring_sizes(mol, bond)
            if any(size <= 6 for size in ring_sizes):
                print(" - rotatable bond is part of a small fused ring (<= 6 members): skipping the calculation")
                return False, [], []

        # If no central atom is found, return a pattern failure
        if central_atom_idx is None:
            print(" - invalid pattern for atropisomerism (no out-of-ring central atom): skipping the calculation")
            return False, [], []

        # Store the central atom with respect to the other atoms to use later
        self.central_atoms[(in_ring_atom_idx, out_ring_atom_idx)] = central_atom_idx

        # Define the neighbors for each atom in the rotatable bond (excluding the central atom)
        in_ring_atom = mol.GetAtomWithIdx(in_ring_atom_idx)
        out_ring_atom = mol.GetAtomWithIdx(out_ring_atom_idx)
        atom_in_ring_neighbors = [n.GetIdx() for n in in_ring_atom.GetNeighbors() if n.GetIdx() not in rotatable_bond_indexes and n.GetIdx() != central_atom_idx]
        atom_out_ring_neighbors = [n.GetIdx() for n in out_ring_atom.GetNeighbors() if n.GetIdx() not in rotatable_bond_indexes and n.GetIdx() != central_atom_idx]

        # If an axis atom has been deprotonated, we identify only one neighbor. In this case, the external neighbor is
        # the lone pair, but this cannot use in the scan. Instead, we add the first neighbor to the axis again and record
        # it as an additional lone pair. Note that this will result in the two scans being identical for this axis.
        if len(atom_out_ring_neighbors) == 1:
            print(" - neighbor atom has been deprotonated: repeating one neighbor in the axis.")
            atom_out_ring_neighbors.append(atom_out_ring_neighbors[0])
            additional_lps = 1
        else:
            additional_lps = 0

        # If an axis atom has been protonated, we identify three neighbors. In this case, we identify a random proton
        # and remove it from the axis. There will always be at least one proton, since it was protonated.
        if len(atom_out_ring_neighbors) == 3:
            print(" - neighbor atom has been protonated: removing one proton from the axis.")
            for idx in atom_out_ring_neighbors:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == "H":
                    atom_out_ring_neighbors.remove(idx)
                    break

        ### Step 2. Define the indexes of the axis atoms required for GIC scans

        # Initialize each axis with the relevant atom from the rotatable bond
        in_ring_axis_atoms = [in_ring_atom.GetIdx()]
        out_ring_axis_atoms = [out_ring_atom.GetIdx()] 

        # Find the exocyclic second neighbours of the ring by searching through the central atom neighbors. If no exocyclic
        # neighbor is found, treat as a possible lone pair site, and include the in-ring atom. Also count if it's a proton.
        in_ring_axis_atoms, in_ring_lp, in_ring_protons = self.discover_ring_axis_atoms(mol, atom_in_ring_neighbors, match_indexes, in_ring_axis_atoms)

        # Add the neighbors of the out-of-ring (non-aromatic) unit to the axis. For benzamides and thiobenzamides 
        # this corresponds to the =O/=S atom and the N(R2) atom. For reverse benzamides and reverse thiobenzamides,
        # this corresponds to the R-substituent of the N and the carbon of the C=O or C=S bond.
        out_ring_axis_atoms.extend(atom_out_ring_neighbors)

        # Count how many of the out-of-ring substituents are hydrogens
        out_ring_protons = self.count_hydrogens(mol, out_ring_axis_atoms)

        # In theory, there cannot be lone pairs on these functional groups, unless they have been
        # deprotonated. To be sure, we can approximate the number of lone pairs on the out-of-ring
        # substituents by subtracting the number of identified substituents from three (since we know
        # we have a benzamide or reverse benzamide, we can assume the out-of-ring atom is a heteroatom
        # (O, S, N) with up to three bonds)). Lone pairs resulting from deprotonation are accounted
        # for via the `additional_lps` variable, which is added to the count.
        out_ring_lp = 3 - len(out_ring_axis_atoms) + additional_lps

        # Count the number of hydrogens on all GIC-defining atoms
        proton_count = self.count_hydrogens(mol, in_ring_axis_atoms + out_ring_axis_atoms)
        assert proton_count == (in_ring_protons + out_ring_protons), "Mismatch in proton count"
    
        ### Step 3. Final analysis

        # 1-index the axis atoms
        axis_1_atoms = [idx + 1 for idx in in_ring_axis_atoms]
        axis_2_atoms = [idx + 1 for idx in out_ring_axis_atoms]

        # Determine whether an atropisomeric scan is appropriate
        result = self.is_atropisomeric(in_ring_lp, out_ring_lp, in_ring_protons, out_ring_protons, axis_1_atoms, axis_2_atoms, min_heavy_subs)
        if result:
            return result, axis_1_atoms, axis_2_atoms
        else:
            return False, [], []

    def analyze_amide(self, mol, match_indexes, min_heavy_subs):
        """
        Analyzes an amide or thioamide and determines the atom indexes of the central rotatable bond,
        as well as the indexes of their four neighbors. These substituents are required for GIC scans
        of the rotatable bond in both directions. Additionally, this method tracks the number of lone
        pairs and protons around the rotatable bond and determines whether a GIC scan is necessary or
        whether atropisomerism is unlikely, for example due to lack of steric hindrance.

        Parameters:
            mol (rdkit.Chem.Mol): RDKit molecule object (with explicit hydrogens).
            match_indexes (list): Atom indices of the matched aryl-X SMARTS substructure.
            min_heavy_subs (int): Minimum number of heavy atom substituents required for atropisomerism.

        Returns:
            result (bool): True if the bond is considered hindered enough for atropisomerism.
            axis_1_atoms, axis_2_atoms (tuples): These correspond to a tuple of three atoms
            indexes: the atom that is part of the rotatable amide bond and its neighbours.
        """

        ### Step 1. Identify the rotatable bond and its neighbors

        # Identify the rotatable bond by finding the C and N atoms
        rotatable_bond_indexes = []
        for idx in match_indexes:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "C":
                c_atom_idx = idx
            elif atom.GetSymbol() == "N":
                n_atom_idx = idx
        rotatable_bond_indexes = [c_atom_idx, n_atom_idx]

        # Check if the rotatable bond is part of another ring (e.g., through a fused ring system or macrocycle).
        # If so, only allow it to be part of larger rings (7-membered or more) as these may be flexible enough
        # to perform a GIC scan and identify a rotational transition state
        ring_sizes = self.get_ring_sizes(mol, mol.GetBondBetweenAtoms(*rotatable_bond_indexes))
        if any(size <= 6 for size in ring_sizes):
            print(" - rotatable bond is part of a small fused ring (<= 6 members): skipping the calculation")
            return False, [], []

        # If no rotatable bond is found, return a pattern failure
        if not rotatable_bond_indexes:
            print(" - invalid pattern for atropisomerism (no rotatable bond found): skipping the calculation")
            return False, [], []

        # Define the neighbors for each atom in the rotatable bond 
        c_atom = mol.GetAtomWithIdx(c_atom_idx)
        n_atom = mol.GetAtomWithIdx(n_atom_idx)
        c_neighbors = [n.GetIdx() for n in c_atom.GetNeighbors() if n.GetIdx() not in rotatable_bond_indexes]
        n_neighbors = [n.GetIdx() for n in n_atom.GetNeighbors() if n.GetIdx() not in rotatable_bond_indexes]

        # If an axis atom has been deprotonated, we identify only one neighbor. In this case, the external neighbor is
        # the lone pair, but this cannot use in the scan. Instead, we add the first neighbor to the axis again and record
        # it as an additional lone pair. Note that this will result in the two scans being identical for this axis.
        if len(n_neighbors) == 1:
            print(" - neighbor atom has been deprotonated: repeating one neighbor in the axis.")
            n_neighbors.append(n_neighbors[0])
            additional_lps = 1
        else:
            additional_lps = 0

        # If an axis atom has been protonated, we identify three neighbors. In this case, we identify a random proton
        # and remove it from the axis. There will always be at least one proton, since it was protonated.
        if len(n_neighbors) == 3:
            print(" - neighbor atom has been protonated: removing one proton from the axis.")
            for idx in n_neighbors:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == "H":
                    n_neighbors.remove(idx)
                    break

        ### Step 2. Define the indexes of the axis atoms required for GIC scans
        
        # Initialize each axis with the relevant ring atom from the rotatable bond
        axis_c_atoms = [c_atom_idx]
        axis_n_atoms = [n_atom_idx]

        # Add the substituents of each atom in the rotatable bond to the axis atoms
        axis_c_atoms.extend(c_neighbors)
        axis_n_atoms.extend(n_neighbors)

        # Count the number of protons on each axis atom
        protons_c = self.count_hydrogens(mol, c_neighbors)
        protons_n = self.count_hydrogens(mol, n_neighbors)

        # In theory, there cannot be lone pairs on these functional groups, unless they have been
        # deprotonated. To be sure, we can approximate the number of lone pairs by subtracting the
        # number of identified substituents from two (since we know we have an amide, we can assume
        # the central atom is a carbon or nitrogen with up to three bonds)). Lone pairs resulting from
        # deprotonation are accounted for via the `additional_lps` variable, which is added to the count.
        lone_pairs_n = 2 - len(n_neighbors) + additional_lps
        lone_pairs_c = 2 - len(c_neighbors)

        # Count the number of hydrogens on all GIC-defining atoms
        proton_count = self.count_hydrogens(mol, axis_c_atoms + axis_n_atoms)
        assert proton_count == (protons_c + protons_n), "Mismatch in proton count"

        ### Step 3. Final analysis

        # 1-index the axis atoms
        axis_c_atoms = [idx + 1 for idx in axis_c_atoms]
        axis_n_atoms = [idx + 1 for idx in axis_n_atoms]
    
        # Final analysis: determine whether an atropisomeric scan is appropriate
        result = self.is_atropisomeric(lone_pairs_c, lone_pairs_n, protons_c, protons_n, axis_c_atoms, axis_n_atoms, min_heavy_subs)
        if result:
            return result, axis_c_atoms, axis_n_atoms
        else:
            return False, [], []

    # GIC helper functions 

    def get_gics(self, axis_1_atoms, axis_2_atoms):
        """
        Given two sets of three atoms, get the atoms required to define the 
        dihedrals to perform two GIC scans, one each in a different direction.

        Parameters:
            axis_1_atoms (list): A list of three atom indices for the first set of atoms.
            axis_2_atoms (list): A list of three atom indices for the second set of atoms.

        Returns:
            gic_1, gic_2 (tuples): Tuple containing two lists of GICs, where each list corresponds to 
            the dihedrals formed by the provided atom indices.
        """

        # Define the dihedrals
        dihedral1 = [
            axis_1_atoms[1],
            axis_1_atoms[0],
            axis_2_atoms[0],
            axis_2_atoms[1],
        ]
        dihedral2 = [
            axis_1_atoms[2],
            axis_1_atoms[0],
            axis_2_atoms[0],
            axis_2_atoms[2],
        ]

        dihedral3 = [
            axis_1_atoms[1],
            axis_1_atoms[0],
            axis_2_atoms[0],
            axis_2_atoms[2],
        ]
        dihedral4 = [
            axis_1_atoms[2],
            axis_1_atoms[0],
            axis_2_atoms[0],
            axis_2_atoms[1],
        ]

        # Define the GICs
        gic_1 = [[idx for idx in dihedral1], [idx for idx in dihedral2]]
        gic_2 = [[idx for idx in dihedral3], [idx for idx in dihedral4]]

        return gic_1, gic_2

    def remove_redundant_gics(self, mol, gic_dict):
        """
        This function removes sulfoxide and sulfone scans if they are part of a diaryl sulfone
        or diaryl sulfoxide. These scans are redundant, as we do not need to scan both the
        diaryl substructure as well as its constituent bonds individually.

        Parameters:
            mol (rdkit.Chem.Mol): RDKit molecule object.
            gic_dict (dict): A dictionary of central bond indices and their corresponding GICs.

        Returns:
            filtered_gics (dict): A filtered dictionary of bonds:GICs with redundant entries removed.
        """

        # Initialize a set of keys to remove
        keys_to_remove = set()

        # Create all combinations of two bonds
        pairs = list(combinations(gic_dict.keys(), 2))

        # Iterate all pairs of keys that share one atom
        for (pair1, pair2) in pairs:

            # Convert string indexes to lists of integers
            pair1 = [int(idx) for idx in pair1.split("_")]
            pair2 = [int(idx) for idx in pair2.split("_")]

            # Check if they share a central atom
            shared = set(pair1) & set(pair2)
            if len(shared) != 1:
                continue
            shared_atom = list(shared)[0]
            outer_atoms = tuple(sorted(list(set(pair1 + pair2) - {shared_atom})))

            # Check if the three atoms define a set of three connected atoms from two bonds
            atom_idxs = [int(outer_atoms[0])-1, int(shared_atom)-1, int(outer_atoms[1])-1]
            if (mol.GetBondBetweenAtoms(atom_idxs[0], atom_idxs[1]) is None or
                    mol.GetBondBetweenAtoms(atom_idxs[1], atom_idxs[2]) is None):
                continue

            # Define the key to search for the in the dictionary (smaller index first)
            if atom_idxs[0] < atom_idxs[2]:
                key = f"{atom_idxs[0]+1}_{atom_idxs[2]+1}"
            else:
                key = f"{atom_idxs[2]+1}_{atom_idxs[0]+1}"

            # If the key exists, mark the sub bonds for removal
            if key in gic_dict:
                keys_to_remove.add(f"{pair1[0]}_{pair1[1]}")
                keys_to_remove.add(f"{pair2[0]}_{pair2[1]}")

        # Report which keys are being removed
        if keys_to_remove:
            for key in keys_to_remove:
                if key in gic_dict:
                    print("Removing redundant scan around the following bond:", key)

        # Remove marked keys from the GIC dictionary
        filtered_gics = {key: gics for key, gics in gic_dict.items() if key not in keys_to_remove}

        return filtered_gics

    # Main cheminformatics function

    def cheminformatics(self, filter=True, coupled_benzamide=False):
        """
        Identifies potential atropisomeric bonds in the molecule, analyzes the bond and determines the input for subsequent GIC scans,
        one each in a different direction. A GIC scan is defined by one or more dihedrals, while a regular dihedral scan involves only
        a single dihedral angle. Thus, GIC scans offer flexibility to scan multiple or alternative dihedrals. This is especially useful
        for complex or unsymmetrical systems.

        Steps:
        1. Searches for structural motifs (e.g., biaryls, benzamides, etc) that can display atropisomerism using predefined SMARTS patterns.
        2. For each match, analyzes the substituent pattern of the identified substructure using the relevant cheminformatics functions to
           identify the rotatable bond and its substituents and determine if atropisomerism is likely (and thus if a GIC scan if appropriate).
        3. Records the atom indices required for GIC scans of each rotatable bond to the self.all_gic_1 and self.all_gic_2 dictionaries.
        4. De-duplicates results so each bond/dihedral is recorded only once.
        5. If no atropisomeric bonds are found, an error is raised.

        Parameters:
            filter (bool): Whether to apply steric filters to determine if a GIC scan is necessary.
            coupled_benzamide (bool): Whether to parse benzamides and reverse benzamides as coupled torsions.

        Returns:
            self.all_gic_1, self.all_gic_2 (dicts): Dictionaries mapping the central bond
            to the dihedral atoms required for forward and reverse GIC scans around it.
        """

        self.filter = filter

        # Iterate over each SMARTS pattern
        for pattern_name in PATTERNS.keys():

            # Define the pattern and non-heavy atom limit
            smarts = PATTERNS[pattern_name]["smarts"]
            min_heavy_subs = PATTERNS[pattern_name]["min_heavy_subs"]

            # Convert the SMARTS string to a mol object
            smarts_mol = Chem.MolFromSmarts(smarts)

            # Find all substructures in the parent molecule that match the SMARTS pattern. This returns
            # a tuple of tuples, where each tuple contains a group of atom indices (in the parent molecule)
            # that map onto the atoms of the SMARTS pattern in a successful substructure match.
            matches = self.protonated_mol.GetSubstructMatches(smarts_mol)
            
            # Iterate over the matches found in the current SMARTS pattern and apply the appropriate cheminformatics
            # function to analyze the substituent pattern and its local steric environment to determine if appropriate
            # for atropisomerism. The match will be processed differently depending on the type of matched pattern.
            for match_indexes in matches:
                print(f"\n{pattern_name} match identified:")

                # Analyze biaryl, aryl-aliphatic and bialiphatic matches
                if "Biaryl" in pattern_name or "Aryl-aliphatic" in pattern_name or "Bialiphatic" in pattern_name:
                    result, axis_1, axis_2 = self.analyze_biring(self.protonated_mol, match_indexes, min_heavy_subs)

                # Analyze diaryl ethers, thioethers and amines matches (also captures diaryl sulfoxides and sulfones)
                elif "Diaryl" in pattern_name:
                    result, axis_1, axis_2 = self.analyze_diaryl(self.protonated_mol, match_indexes, min_heavy_subs)
                
                # If enabled, analyze (thio)benzamides and reversed (thio)benzamides as coupled torsions
                elif coupled_benzamide and ("Benzamide" in pattern_name or "Thiobenzamide" in pattern_name or "Reverse benzamide" in pattern_name or "Reverse thiobenzamide" in pattern_name):
                    # The heavy atom threshold is only determined for single-bond benzamides, so we set it to 0 to allow all coupled benzamides
                    result, axis_1, axis_2 = self.analyze_coupled_aryl_X(self.protonated_mol, match_indexes, min_heavy_subs=0)

                # Analyze (thio)benzamide, reversed (thio)benzamide, aryl sulfoxide and sulfonamide matches
                elif "Benzamide" in pattern_name or "Thiobenzamide" in pattern_name or "Reverse benzamide" in pattern_name or "Reverse thiobenzamide" in pattern_name or "Sulfinamide" in pattern_name or "Sulfonamide" in pattern_name or "Sulfoxide" in pattern_name or "Sulfone" in pattern_name:
                    result, axis_1, axis_2 = self.analyze_aryl_X(self.protonated_mol, match_indexes, min_heavy_subs)

                # Analyze amides and thioamide matches
                elif pattern_name in ["Amide", "Thioamide"]:
                    result, axis_1, axis_2 = self.analyze_amide(self.protonated_mol, match_indexes, min_heavy_subs)
                # Flag unknown SMARTS patterns
                else:
                    print(f"Unknown SMARTS pattern {smarts} found. Skipping this match.")
                    continue

                # If a match is determined to have sufficient bulk in its substituents for atropisomerism, record the
                # rotatable bond, GICs and substructure type in the corresponding dictionaries. Two GICs are recorded,
                # corresponding to forwards and backwards scans of the rotatable bond. Sometimes a previous key can be
                # overwritten if the same central bond was already recognised, e.g, benzamide and aryl-aliphatic are often
                # recognised together. We allow this, to prevent duplicate GICs corresponding to the same central bond.
                if result:
        
                    # Define the central bond, always putting the lower index atom first. This avoids
                    # duplicate GICs corresponding to the same central bond but with different keys.
                    if axis_1[0] < axis_2[0]:
                        central_bond = f"{axis_1[0]}_{axis_2[0]}"
                    else:
                        central_bond = f"{axis_2[0]}_{axis_1[0]}"

                    # Record the GICs and substructures
                    gic_1, gic_2 = self.get_gics(axis_1, axis_2)
                    self.all_gic_1[central_bond] = gic_1
                    self.all_gic_2[central_bond] = gic_2
                    self.substructures[central_bond] = pattern_name

                    # Show bonds on the molecule
                    for atom in self.mol.GetAtoms():
                        for at in [axis_1[0], axis_2[0]]:
                            if atom.GetIdx() + 1 == at:
                                atom.SetProp("molAtomMapNumber", str(at))

                    # Show bonds and substituents on the protonated molecule
                    for atom in self.protonated_mol.GetAtoms():
                        for at in list(set(x for sub in gic_1 for x in sub)):
                            if atom.GetIdx() + 1 == at:
                                atom.SetProp("molAtomMapNumber", str(at))

        # Remove redundant GIC scans for sulfones and sulfoxides
        self.all_gic_1 = self.remove_redundant_gics(self.protonated_mol, self.all_gic_1)
        self.all_gic_2 = self.remove_redundant_gics(self.protonated_mol, self.all_gic_2)

        # Match the keys in substructures to the remaining GICs
        self.substructures = {k: v for k, v in self.substructures.items() if k in self.all_gic_1 or k in self.all_gic_2}

        # Report if no atropisomeric bonds are identified
        if not self.all_gic_1 and not self.all_gic_2:
            print("\nNo atropisomeric bonds were identified in the molecule. If you believe this is an error, please contact an expert.")
        else:
            # Summarise the rotatable bonds, dihedrals and substructures identified
            print("\nSummary of dihedrals for GIC scans:")
            print(" - scan 1:", self.all_gic_1)
            print(" - scan 2:", self.all_gic_2)
            print("\nSummary of identified substructures:")
            print(self.substructures)

            # Show the molecules with highlighted atoms
            print("\nRotatable bond:")
            display(Draw.MolToImage(self.mol))
            print("GIC atoms:")
            display(Draw.MolToImage(self.protonated_mol))

        return self.all_gic_1, self.all_gic_2

    # Gaussian helper functions

    def calc_dihedral(self, u1, u2, u3, u4):
        """
        Calculate the signed dihedral angle (in radians) defined by four 3D points.

        Parameters:
            u1, u2, u3, u4 (np.ndarray): Cartesian coordinates of the four atoms (shape: (3,)).

        Returns:
            float: Signed dihedral angle in radians.
        """

        # Build bond vectors
        a1 = u2 - u1
        a2 = u3 - u2
        a3 = u4 - u3

        # Normals to the two planes
        v1 = np.cross(a1, a2)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(a2, a3)
        v2 /= np.linalg.norm(v2)

        # Compute sign of angle using orientation
        porm = np.sign(np.dot(v1, a3))
        rad = np.arccos(np.dot(v1, v2))
        if porm != 0:
            rad *= porm  # Apply sign
        return rad

    def normalize_angle(self, angle):
        """
        Normalize an angle to be within the range of 0 to 180 degrees.

        Parameters:
            angle (float): The angle in degrees to be normalized.

        Returns:
            float: The normalized angle, which will be less than 180 degrees.
        """
        return angle if angle < 180 else 360 - angle

    def get_indexes_from_scan_inp(self, gjf_path):
        """
        Extract the GIC dihedral indices from a Gaussian scan input file.

        This function searches for lines of the form:
        bond1=D(i,j,k,l)
        bond2=D(i,j,k,l)

        Parameters:
            gjf_path (Path): The path to the Gaussian input scan file.

        Returns:
            indexes1 (list[int]): The first set of dihedral indices (1-indexed).
            indexes2 (list[int]): The second set of dihedral indices (1-indexed).
            constrained_atoms (list[int]): A sorted list of unique constrained atom indices derived from both sets (1-indexed).

        """

        # Define the regex pattern to find the constraint lines
        pattern = re.compile(r"bond\d+\s*=\s*D\((\d+),(\d+),(\d+),(\d+)\)")

        # Parse the Gaussian input file
        matches = []
        with open(gjf_path) as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    matches.append([int(x) for x in match.groups()])

        # Raise an error if no matches are found
        if not matches:
            raise ValueError(f"No GIC dihedrals found in {gjf_path}")

        # Define the lists
        indexes1 = matches[0]
        indexes2 = matches[1]

        # Combine both sets
        constrained_atoms = sorted(set(indexes1 + (indexes2)))

        return indexes1, indexes2, constrained_atoms

    def check_is_rotational_ts(self, log_path, scan_dir, xyz):
        """
        Check if the given log file corresponds to a transition state (TS) at the rotatable bond
        by comparing the relative magnitude of displacement vectors on the constrained atoms (those
        around the rotatable bond) compared to the rest of the molecule. Also checks that here isn't
        a significant change in the dihedral angles of the rotatable bond.

        Parameters:
            log_path (str): Path to the calculation log file to be parsed.
            scan_dir (str): Path to the scan directory containing constraint info.
            xyz (list): List of Cartesian coordinates for the molecule at different optimization steps.

        Returns:
            bool: True if the log file corresponds to a TS, False otherwise.
        """

        # Initialize counts for total magnitude of displacement vectors
        weights_sum = 0.0  # Summed over all atoms
        weights_sum_gic = 0.0  # Summed over constrained atoms

        # Retrieve dihedrals indexes and constrained atoms from the scan
        dihed_indexes1, dihed_indexes2, constrained_atoms = self.get_indexes_from_scan_inp(scan_dir / "dft_scan.gjf")

        # Parse the vibrational displacement vectors and take the first mode (imaginary frequency) for analysis
        _, _, _, displacements, _ = parse_with_cclib(log_path)
        im_freq_displacements = displacements[0]

        # Iterate over the displacements on each atom
        for i, disp in enumerate(im_freq_displacements):

            # Calculate the squared magnitude of the displacement vector for this atom
            disp_magnitude_squared = np.sum(disp ** 2)

            # Add to the total displacement magnitude
            weights_sum += disp_magnitude_squared

            # If this atom is in the constrained atoms, add to the constrained sum
            if (i + 1) in constrained_atoms:
                weights_sum_gic += disp_magnitude_squared

        # If there is no displacement at all, avoid a division by zero error and reject the TS.
        if weights_sum == 0:
            return False

        # If less than X% of total displacement is on the constrained atoms, we reject the TS.
        # In this case, the TS likely does not correspond to the desired rotation, but something else.
        if (weights_sum_gic / weights_sum)*100 < self.min_ts_displacement_percentage:
            return False

        # Additionally, check how the dihedrals have changed during optimization.
        dihed1_init = self.get_diheds(dihed_indexes1, xyz[0])
        dihed2_init = self.get_diheds(dihed_indexes2, xyz[0])
        dihed1_end = self.get_diheds(dihed_indexes1, xyz[-1])
        dihed2_end = self.get_diheds(dihed_indexes2, xyz[-1])
        dihed1_change = self.normalize_angle(abs(dihed1_init - dihed1_end))
        dihed2_change = self.normalize_angle(abs(dihed2_init - dihed2_end))
    
        # If either of the dihedral angles have deviated significantly (by at least X degrees) during optimization,
        # we reject the TS. In this case, the intended "flat" TS has likely relaxed towards the reactant geometry.
        if dihed1_change >= self.max_ts_dihedral_change or dihed2_change >= self.max_ts_dihedral_change:
            return False

        # If both pass, we likely have the correct rotational TS
        return True

    def ts_analysis(self, analysis_type):
        """
        Analyses TSs resulting from the initial scan or from the subsequent CREST optimisations,
        depending on the analysis type specified, and checks whether the TS corresponds to a true 
        rotational TS at the intended rotatable bond. Returns dictionaries mapping the paths of
        valid TS directories to the corresponding coordinates, SCF energies and free energies

        If directly after GIC scans, analyzes:
         - X_Y/opt_ts/max/g_opt_ts.log
         - X_Y/opt_ts/minus1/g_opt_ts.log
         - X_Y/opt_ts/plus1/g_opt_ts.log

        If after CREST search, analyzes:
         - X_Y/ts_conformers/dft_ts/n/g_opt_ts_confs.log 

        Parameters:
            analysis_type (str): Type of analysis, either 'scan' or 'crest'.
        
        Returns:
            ts_xyz_dict (dict): A dictionary mapping the directory of each TS calculation to its corresponding XYZ coordinates.
            ts_scf_energies_dict (dict): A dictionary mapping the directory of each TS calculation to its final SCF energy.
            ts_free_energies_dict (dict): A dictionary mapping the directory of each TS calculation to its free energy.

        """
        
        # Define the directory and log file patterns based on the analysis type
        if analysis_type == 'scan':
            sub_dir = "opt_ts"
            log_file_name = "g_opt_ts.log"
        elif analysis_type == 'crest':
            sub_dir = "ts_conformers/dft_ts"
            log_file_name = "g_opt_ts_confs.log"

        # Create list of directories with Gaussian TS output files to analyze
        gjf_dirs = []
        for idx in self.bonds_to_process:
            out_dir = self.job_dir / f"{idx}"
            # Get all TS optimisations: 
            #  - three conformers (max, minus1, plus1) after scan
            #  - one conformer after CREST
            for gjf in out_dir.glob(f"{sub_dir}/*/*.gjf"):
                gjf_dirs.append(gjf.parent)
                
        # Initialize dictionary to track processing status for each 
        # directory and mark each directory as unprocessed initially
        checks = {gjf_dir: False for gjf_dir in gjf_dirs}

        # Initialize dictionaries to hold geometries and energies
        ts_xyz_dict, ts_scf_energies_dict, ts_free_energies_dict = {}, {}, {}

        # Iterate until all optimisations have been processed
        while False in checks.values():
            for gjf_dir in gjf_dirs:

                # Define directory based on the analysis type
                if analysis_type == 'scan':
                    parent_dir = gjf_dir.parent.parent
                elif analysis_type == 'crest':
                    parent_dir = gjf_dir.parent.parent.parent

                if checks[gjf_dir] is False:

                    # Provide path to the Gaussian log file
                    log = gjf_dir / log_file_name

                    # If log file exists, extract relevant data
                    if log.exists():
                        
                        # Extract frequencies and geometries
                        scf_energies, free_energy, freqs, _, xyz_coords = parse_with_cclib(log)

                        # Skip this structure if properties could not be parsed
                        if not scf_energies or not free_energy or not freqs.any() or not xyz_coords:
                            self.pprint(gjf_dir, "ts_analysis", "Failed to parse log file, skipping.")
                            checks[gjf_dir] = True
                            continue

                        # Check for negative frequency (indicating a TS candidate)
                        if freqs[0] < 0:

                            # Check if the log file corresponds to a TS based on the main source
                            # of displacement and on the dihedral angles of the rotatable bond
                            scan_dir = parent_dir / "scan_dft_1"
                            is_ts = self.check_is_rotational_ts(log, scan_dir, xyz_coords)

                            # Store energies and coordinates if TS is valid
                            if is_ts:
                                ts_xyz_dict[gjf_dir] = xyz_coords[-1]
                                ts_scf_energies_dict[gjf_dir] = float(scf_energies[-1])
                                ts_free_energies_dict[gjf_dir] = float(free_energy)
        
                        # Mark directories related to this file index as processed
                        checks[gjf_dir] = True

        return ts_xyz_dict, ts_scf_energies_dict, ts_free_energies_dict

    def _fix_scientific_notation(self, s, decimals=10):
        """
        Convert scientific notation numbers in a string to fixed-point notation.

        Parameters:
            s (str): A string containing numbers in scientific notation.
            decimals (int): The number of decimal places to format the output. Default is 10.

        Returns:
            str: A string with numbers formatted in fixed-point notation.
        """

        numbers = s.split()

        # Format each number using fixed-point notation
        return ' '.join([f"{float(num):.{decimals}f}" for num in numbers])

    ## 1. g_opt_react (1 job)

    def export_g_opt_react(self, input_xyz, output_gjf, **kwargs):
        """
        Reads the lowest energy structure from the provided ensemble and generates a Gaussian input
        file for geometry optimization and frequency calculation at the specified output directory.

        Parameters:
            input_xyz (Path): Path to the XYZ file of the lowest energy structure from the ensemble.
            output_gjf (Path): Path to the output Gaussian input file.
            **kwargs (dict): Any keyword arguments accepted by `write_gaussian_input`, including:
                cpus (int): Number of processors for Gaussian job.   
                mem (str): Memory allocation for Gaussian job.

        Returns:
            None                 
        """
        
        # Parse the XYZ coordinates of the lowest energy conformer
        xyz_block, _, __ = get_conformer_xyz(input_xyz, 0)
        xyz_out = xyz_block[2:]

        # Get the coordinates in list format
        xyz = [' '.join(line.split()[1:]) for line in xyz_out]

        # Create the directory for the reactant gaussian optimization
        gjf_dir = output_gjf.parent
        gjf_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

        # Define the route card for the job
        route = f"opt=calcfc freq {self.opt_method} {self.opt_basis} {self.gauss_solv} IOP(1/8=2) nosymm"

        # Write the Gaussian input file
        self.write_gaussian_input(output_gjf, route, self.atom_types, xyz, kwargs["cpus"], kwargs["mem"])
        self.g_opt_react_inps = [output_gjf]

        # Initialize a row for the reactant optimization in self.structures, with default energy values
        substitutions = []
        substitutions += [
            pd.Series(
                {
                    "file_idx": gjf_dir.name,
                    "mol": None,
                    "confs": None,
                    "exported_confs": [],
                    "stable_xyz": None,
                    "substructure": "reactant",
                    "E (hartree)": np.inf,
                    "E_SP (hartree)": np.inf,
                    "G_SP (hartree)": np.inf,
                    "QH_G_SP (hartree)": np.inf,
                    "ΔG activation (kcal/mol)": None,
                    "ΔG activation_r (kcal/mol)": None,
                }
            )
            .to_frame()
            .T
        ]

        # Update self.structures dataframe
        self.structures = pd.concat(
            [self.structures, *substitutions], ignore_index=True
        )
        self.pprint("", "g_opt_react", "g_opt_react written successfully.")

    ## 2. g_scan (2 jobs per bond)

    def export_g_scan(self, input_xyz, scans_to_perform, repeat=False, **kwargs):
        """
        Reads the lowest energy structure from the provided ensemble and generates a Gaussian input
        file for each requested GIC scan (forward and/or backward) for the specified central bonds. 
        Creates the necessary directories for each scan and writes the corresponding Gaussian input files.

        Parameters:
            input_xyz (Path): Path to the XYZ file of the lowest energy structure from the ensemble.
            scans_to_perform (dict): Dictionary mapping central bond indices to scan directions.
            repeat (bool): Flag indicating whether to setup stricter repeat scans for requested indexes.
            **kwargs (dict): Any keyword arguments accepted by `write_gaussian_input`, including:
                cpus (int): Number of processors for Gaussian job.   
                mem (str): Memory allocation for Gaussian job.

        Returns:
            None
        """

        # Parse the XYZ coordinates of the lowest energy conformer
        xyz_block, _, __ = get_conformer_xyz(input_xyz, 0)
        xyz_out = xyz_block[2:]

        # Get the coordinates in list format
        xyz = [' '.join(line.split()[1:]) for line in xyz_out]

        # Define the route card for the scans
        route = f"opt=(addgic,loose) {self.opt_method} {self.opt_basis} {self.gauss_solv} IOP(1/8=2) nosymm"

        # Initialize a dict to track the Gaussian input files created
        self.g_scan_inps = {file_idx: [] for file_idx in scans_to_perform.keys()}

        # Iterate over each central bond in the requested scan dictionary
        for central_bond, requested in scans_to_perform.items():

            # Get the corresponding substructure pattern
            substructure = self.substructures.get(central_bond, "unknown")

            # Create a directory for this rotatable bond
            bond_dir = self.job_dir / central_bond
            bond_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

            # Determine if the repeat tag needs to be added
            tag = "_repeat" if repeat else ""

            # Create the forward GIC scan, if requested
            if requested["forward"]:

                # Create the subdirectory for the scan
                scan_dir_1 = self.job_dir / central_bond / f"scan_dft_1{tag}"
                scan_dir_1.mkdir(mode=0o775, parents=True, exist_ok=True)

                # Grab the GIC dihedrals from the GIC dict
                gic_idx_1 = self.all_gic_1[central_bond][0]
                gic_idx_2 = self.all_gic_1[central_bond][1]

                # Get the coordinates of the four atoms defining the dihedral
                a = np.array([float(x) for x in xyz[gic_idx_1[0] - 1].split()])
                b = np.array([float(x) for x in xyz[gic_idx_1[1] - 1].split()])
                c = np.array([float(x) for x in xyz[gic_idx_1[2] - 1].split()])
                d = np.array([float(x) for x in xyz[gic_idx_1[3] - 1].split()])

                # Get the dihedrals for the scan
                dihed = self.calc_dihedral(a, b, c, d) * 180 / np.pi

                # Create the tail list (for the scan direction and dihedral freezing)
                tail_list = []
                tail_list += [f"bond1=D({','.join(str(e) for e in gic_idx_1)})"]
                tail_list += [f"bond2=D({','.join(str(e) for e in gic_idx_2)})"]
                tail_list += ["coord=[bond1+bond2]"]
                if dihed >= 0:
                    tail_list += [f"coord(nsteps={self.scan_steps},stepsize={-self.scan_stepsize})"]
                else:                    
                    tail_list += [f"coord(nsteps={self.scan_steps},stepsize={self.scan_stepsize})"]

                # Write the Gaussian input file for the forward scan
                input_gjf = scan_dir_1 / "dft_scan.gjf"
                self.write_gaussian_input(input_gjf, route, self.atom_types, xyz, kwargs["cpus"], kwargs["mem"], tail_list)
                self.g_scan_inps[central_bond].append(input_gjf)

            # Create the backward GIC scan, if requested
            if requested["backward"]:

                # Create the subdirectory for the scan
                scan_dir_2 = self.job_dir / central_bond / f"scan_dft_2{tag}"
                scan_dir_2.mkdir(mode=0o775, parents=True, exist_ok=True)

                # Grab the GIC dihedrals from the GIC dict
                gic_idx_1 = self.all_gic_2[central_bond][0]
                gic_idx_2 = self.all_gic_2[central_bond][1]

                # Get the coordinates of the four atoms defining the dihedral
                a = np.array([float(x) for x in xyz[gic_idx_1[0] - 1].split()])
                b = np.array([float(x) for x in xyz[gic_idx_1[1] - 1].split()])
                c = np.array([float(x) for x in xyz[gic_idx_1[2] - 1].split()])
                d = np.array([float(x) for x in xyz[gic_idx_1[3] - 1].split()])

                # Get the dihedrals for the scan
                dihed = self.calc_dihedral(a, b, c, d) * 180 / np.pi

                # Create the tail list (for the scan direction and dihedral freezing)
                tail_list = []
                tail_list += [f"bond1=D({','.join(str(e) for e in gic_idx_1)})"]
                tail_list += [f"bond2=D({','.join(str(e) for e in gic_idx_2)})"]
                tail_list += ["coord=[bond1+bond2]"]
                if dihed >= 0:
                    tail_list += [f"coord(nsteps={self.scan_steps},stepsize={-self.scan_stepsize})"]
                else:                    
                    tail_list += [f"coord(nsteps={self.scan_steps},stepsize={self.scan_stepsize})"]

                # Write the Gaussian input file for the backward scan
                input_gjf = scan_dir_2 / "dft_scan.gjf"
                self.write_gaussian_input(input_gjf, route, self.atom_types, xyz, kwargs["cpus"], kwargs["mem"], tail_list)
                self.g_scan_inps[central_bond].append(input_gjf)

            # Add this rotatable bond to the structures dataframe
            substitutions = []
            substitutions += [
                pd.Series(
                    {
                        "file_idx": central_bond,
                        "mol": None,
                        "confs": None,
                        "exported_confs": [],
                        "stable_xyz": None,
                        "substructure": substructure,
                        "E (hartree)": np.inf,
                        "E_SP (hartree)": np.inf,
                        "G_SP (hartree)": np.inf,
                        "QH_G_SP (hartree)": np.inf,
                        "ΔG activation (kcal/mol)": None,
                        "ΔG activation_r (kcal/mol)": None,
                    }
                )
                .to_frame()
                .T
            ]

            # Update self.structures with default energy and Gibbs free energy values
            self.structures = pd.concat([self.structures, *substitutions], ignore_index=True)

        self.pprint("", "g_scan", "g_scan_gic(s) written successfully.")

    def get_diheds(self, dihedral_indexes, xyz_coords):
        """
        Calculates the dihedral angle given atom indices and their coordinates.

        Parameters:
            dihedral_indexes (list of int): List containing the indices of the four atoms that define the dihedral angle.
            xyz_coords (list of str): List of strings where each string contains the x, y, z coordinates of an atom.

        Returns:
            float: The dihedral angle in degrees.
        """

        # Get the coordinates of each atom in the dihedral list as an array
        a = np.array([float(i) for i in xyz_coords[dihedral_indexes[0] - 1].split()])
        b = np.array([float(i) for i in xyz_coords[dihedral_indexes[1] - 1].split()])
        c = np.array([float(i) for i in xyz_coords[dihedral_indexes[2] - 1].split()])
        d = np.array([float(i) for i in xyz_coords[dihedral_indexes[3] - 1].split()])

        # Calculate the dihedral
        dihedral = self.calc_dihedral(a, b, c, d) * 180 / np.pi

        return dihedral

    def scan_analysis(self):
        """
        Analyzes the Gaussian scan job directories, extracting energy and geometry data to identify
        the maximum energy points along the scan. The analysis is performed for each central bond
        in the molecule, and the results are stored in dictionaries for further processing. Checks
        are made to ensure that the correct geometry from the scan is taken, for example by checking
        that there has been a significant deviation (self.min_scan_dihedral_change degrees) in the
        dihedral angle from the starting structure, and by only recording the highest energy maxima
        along the scan. From this maximum point of the scan, the geometry one step forwards and one
        step backwards are also identified, resulting in max, minus1 and plus1 geometries for each
        scan. Afterwards, the lowest energy geometry in each category (max, minus1, plus1) are
        identified, regardless of which scan they originate from (forwards or backwards).

        Returns:
            max_scan_xyz_dict (dict): Dictionary mapping paths to the geometries of the maximum energy scan point.
            minus1_scan_xyz_dict (dict): Dictionary mapping paths to the geometries one step before the maximum scan point.
            plus1_scan_xyz_dict (dict): Dictionary mapping paths to the geometries one step after the maximum scan point.
        """

        # Gather the directories containing the completed scan output files
        # This may include failed scans, but these are filtered out later
        scan_dirs = []
        for idx in self.bonds_to_process:
            out_dir = self.job_dir / f"{idx}"
            for scan_dir in out_dir.glob(r"scan_dft_*/*.gjf"):
                scan_dirs.append(scan_dir.parent)

        # Initialize variables to store scan results
        max_scans = {}
        max_scans_minus1 = {}
        max_scans_plus1 = {}
        max_energies = {}

        # Iterate over the completed scan directories
        for scan_dir in scan_dirs:

            # Get the file index (central bond) from the directory name
            file_idx = scan_dir.parent.name

            # Initialize variables to store scan data
            coords_scan = []
            energy_scan = []

            # Check if the scan output exists
            log = scan_dir / "dft_scan.log"
            if log.exists():

                # Extract energies and geometries for each scan step from the log file
                energy_scan, coords_scan, initial_xyz = parse_scan_with_cclib(log)

                # Skip the scan if no energies could be parsed
                if not energy_scan:
                    self.pprint(file_idx, "scan_analysis", f"Failed to parse scan output ({scan_dir}), skipping.")
                    continue

                # Check that there are a sufficient number of energy data points (completed steps) in the scan.
                # If there are, identify the maximum point of the scan by searching for a local maximum within
                # the energy_scan array. If there are not, skip parsing the results for this scan.
                if len(energy_scan) > self.min_complete_scan_steps:

                    # Get the indexes of the two dihedral constraints from the scan
                    dihed_indexes1, dihed_indexes2, _ = self.get_indexes_from_scan_inp(scan_dir / "dft_scan.gjf")

                    # Get the dihedrals for the initial scan geometry
                    dihed1_init = self.get_diheds(dihed_indexes1, initial_xyz)
                    dihed2_init = self.get_diheds(dihed_indexes2, initial_xyz)

                    # This loop iterates over the energies recorded in energy_scan, excluding the first two and last three
                    # points to properly compare with neighboring conditions. It checks for local maxima by verifying that 
                    # the current point is higher in energy than the two energy values following it and preceding it.
                    for i in range(2, len(energy_scan) - 3):
                        if all(
                            [
                                float(energy_scan[i + 1]) < float(energy_scan[i]),
                                float(energy_scan[i + 2]) < float(energy_scan[i]),
                                float(energy_scan[i - 1]) < float(energy_scan[i]),
                                float(energy_scan[i - 2]) < float(energy_scan[i]),
                            ]
                        ):

                            # Get the dihedrals of the potential maximum point geometry
                            dihed1 = self.get_diheds(dihed_indexes1, coords_scan[i])
                            dihed2 = self.get_diheds(dihed_indexes2, coords_scan[i])

                            # Get the difference in dihedral angles compared to the initial geometry
                            dihed1_change = self.normalize_angle(abs(dihed1_init - dihed1))
                            dihed2_change = self.normalize_angle(abs(dihed2_init - dihed2))

                            # After identifying potential maximum candidates through energy comparisons, we check whether the
                            # dihedral angles (dihed1, dihed2) have deviated significantly (by at least the minimum scan dihedral
                            # change) from their initial values (dihed1_init, dihed2_init). We only accept the scan if one has.
                            sufficient_deviation = dihed1_change >= self.min_scan_dihedral_change or dihed2_change >= self.min_scan_dihedral_change

                            # If the dihedral angles have deviated sufficiently, and no point has been previously recorded for
                            # this scan, record the XYZ and energy values for that point. If a point has already been recorded,
                            # overwrite it if the new maximum point has a higher energy than the previously recorded maximum.
                            # We do this so we capture the highest energy maximum along the scan. Otherwise, we might record a
                            # small maximum that doesn't correspond to the true maximum.
                            if sufficient_deviation and (
                                scan_dir not in max_energies
                                or float(energy_scan[i]) > max_energies[scan_dir]
                                ):
      
                                # Record the XYZ coordinates of the maximum point and its neighbors
                                max_scans[scan_dir] = coords_scan[i]
                                max_scans_minus1[scan_dir] = coords_scan[i - 1]
                                max_scans_plus1[scan_dir] = coords_scan[i + 1]

                                # Record the energy of the maximum point
                                max_energies[scan_dir] = float(energy_scan[i])

        # Initialize dictionaries to store scan indices based on directory names
        max_scan_xyz_dict = {}
        minus1_scan_xyz_dict = {}
        plus1_scan_xyz_dict = {}

        # Create a dictionary with directories and their associated energies
        energies_dict = defaultdict(list)
        for scan_dir, energy in max_energies.items():
            energies_dict[scan_dir.parent.name] += [(scan_dir, energy)]

        # Create a dictionary to find the file with the smallest energy in each category
        max_energies_dict = {
            file_idx: min(values, key=lambda el: el[1])
            for file_idx, values in energies_dict.items()
        }

        # Populate scan indices based on the directory with the smallest energy
        for scan_dir, _ in max_energies_dict.values():
            max_scan_xyz_dict |= {scan_dir: max_scans[scan_dir]}
            minus1_scan_xyz_dict |= {scan_dir: max_scans_minus1[scan_dir]}
            plus1_scan_xyz_dict |= {scan_dir: max_scans_plus1[scan_dir]}

        return max_scan_xyz_dict, minus1_scan_xyz_dict, plus1_scan_xyz_dict

    ## 3. g_opt_ts_frozen (3 jobs per bond)

    def export_g_opt_ts_frozen(self, max_scan_xyz_dict, minus1_scan_xyz_dict, plus1_scan_xyz_dict, **kwargs):
        """
        Reads the provided scan geometries and creates three Gaussian input files for each rotatable bond,
        corresponding to the maximum energy point of the scan and its immediate neighbors (minus1 and plus1).
        The generated input files are frozen optimizations, meaning that certain dihedral angles are
        constrained during the optimization process:

        Parameters:
            max_scan_xyz_dict (dict): Dictionary mapping paths to the geometries of the maximum energy scan point.
            minus1_scan_xyz_dict (dict): Dictionary mapping paths to the geometries one step before the maximum scan point.
            plus1_scan_xyz_dict (dict): Dictionary mapping paths to the geometries one step after the maximum scan point.
            **kwargs (dict): Any keyword arguments accepted by `write_gaussian_input`, including:
                cpus (int): Number of processors for Gaussian job.   
                mem (str): Memory allocation for Gaussian job.

        Returns:
            None
        """

        # Create a dictionary matching the directory names to the corresponding scan XYZ dictionaries for easier iteration
        dirs_to_create = {
            "max": max_scan_xyz_dict,
            "minus1": minus1_scan_xyz_dict,
            "plus1": plus1_scan_xyz_dict
        }

        # Define the route card for these jobs
        route = f"opt=modredundant {self.opt_method} {self.opt_basis} {self.gauss_solv} IOP(1/8=2) nosymm"

        # Initialize a dict to track the Gaussian input files created
        self.g_opt_ts_frozen_inps = {file_idx: [] for file_idx in self.structures["file_idx"]}

        # Iterate over the three categories of frozen optimizations
        for dir_name, scans_xyz_dict in dirs_to_create.items():

            for scan_dir, xyz in scans_xyz_dict.items():

                # Get the file index
                file_idx = scan_dir.parent.name

                # Get the dihedral indexes from the scan input file
                dihed_indexes1, dihed_indexes2, _ = self.get_indexes_from_scan_inp(scan_dir / "dft_scan.gjf")

                # Create the tail list (for freezing the dihedrals in the input file)
                tail_list = []
                tail_list += [f"D {' '.join(str(e) for e in dihed_indexes1)} F"]
                tail_list += [f"D {' '.join(str(e) for e in dihed_indexes2)} F"]

                # Create the directory
                opt_frozen_dir = scan_dir.parent / "opt_frozen" / dir_name
                opt_frozen_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

                # Write the Gaussian input file
                input_gjf = opt_frozen_dir / "g_opt_frozen.gjf"
                self.write_gaussian_input(input_gjf, route, self.atom_types, xyz, kwargs["cpus"], kwargs["mem"], tail_list)
                self.g_opt_ts_frozen_inps[file_idx] += [input_gjf]

    ## 4. g_opt_ts (3 jobs per bond)
     
    def export_g_opt_ts(self, **kwargs):
        """
        Reads the frozen optimization output files and creates three Gaussian input files for each rotatable bond,
        corresponding to the maximum energy point of the scan and its immediate neighbors (minus1 and plus1). 
        The generated input files are unconstrained optimizations.
    
        Parameters:
            **kwargs (dict): Any keyword arguments accepted by `write_gaussian_input`, including:
                cpus (int): Number of processors for Gaussian job.   
                mem (str): Memory allocation for Gaussian job.
        """

        # Gather the scan directories for succesful scans
        scan_dirs = []
        for idx in self.bonds_to_process:
            if idx in self.successful_bonds:
                out_dir = self.job_dir / f"{idx}"
                scan_dirs.append(out_dir)

        # Initialize a dict to track the Gaussian input files created
        self.g_opt_ts_inps = {file_idx: [] for file_idx in self.structures["file_idx"]}

        for scan_dir in scan_dirs:

            # Get the file index
            file_idx = scan_dir.name

            # Get output geometries from the frozen optimization output files
            opt_frozen_dir1 = scan_dir / "opt_frozen" / "max" / "g_opt_frozen.log"
            opt_frozen_dir2 = scan_dir / "opt_frozen" / "minus1" / "g_opt_frozen.log"
            opt_frozen_dir3 = scan_dir / "opt_frozen" / "plus1" / "g_opt_frozen.log"
            _, _, _, _, frozen_xyz_coords1 = parse_with_cclib(opt_frozen_dir1)
            _, _, _, _, frozen_xyz_coords2 = parse_with_cclib(opt_frozen_dir2)
            _, _, _, _, frozen_xyz_coords3 = parse_with_cclib(opt_frozen_dir3)

            # Check the geometries were parsed correctly
            if frozen_xyz_coords1 is None:
                raise ValueError(f"Could not parse geometry for frozen optimization. Please check file: {opt_frozen_dir1}.")
            if frozen_xyz_coords2 is None:
                raise ValueError(f"Could not parse geometry for frozen optimization. Please check file: {opt_frozen_dir2}.")
            if frozen_xyz_coords3 is None:
                raise ValueError(f"Could not parse geometry for frozen optimization. Please check file: {opt_frozen_dir3}.")
        
            # Greate folders for the unconstrained TS optimizations
            opt_ts_dir1 = scan_dir / "opt_ts" / "max"
            opt_ts_dir2 = scan_dir / "opt_ts" / "minus1"
            opt_ts_dir3 = scan_dir / "opt_ts" / "plus1"
            opt_ts_dir1.mkdir(mode=0o775, parents=True, exist_ok=True)
            opt_ts_dir2.mkdir(mode=0o775, parents=True, exist_ok=True)
            opt_ts_dir3.mkdir(mode=0o775, parents=True, exist_ok=True)

            # Define the route card for these jobs
            route = f"opt=(ts,noeigen,recalcfc=15) {self.opt_method} {self.opt_basis} {self.gauss_solv} freq IOP(1/8=2) nosymm"

            # Create input files for the maximum geometry and the points following and preceding it
            input_gjf1 = opt_ts_dir1 / "g_opt_ts.gjf"
            input_gjf2 = opt_ts_dir2 / "g_opt_ts.gjf"
            input_gjf3 = opt_ts_dir3 / "g_opt_ts.gjf"
            self.write_gaussian_input(input_gjf1, route, self.atom_types, frozen_xyz_coords1[-1], kwargs["cpus"], kwargs["mem"])
            self.write_gaussian_input(input_gjf2, route, self.atom_types, frozen_xyz_coords2[-1], kwargs["cpus"], kwargs["mem"])
            self.write_gaussian_input(input_gjf3, route, self.atom_types, frozen_xyz_coords3[-1], kwargs["cpus"], kwargs["mem"])
            self.g_opt_ts_inps[file_idx] += [input_gjf1, input_gjf2, input_gjf3]

    ## 5. crest_ts (1 job per bond)

    def export_crest_ts(self, file_idx, ts_xyz_dict, ts_free_energies_dict):
        """
        Reads the lowest energy structure (max, minus1 or plus1) from the unconstrained
        TS optimizations and generates an input file for a CREST conformational search.

        Parameters:
            file_idx (str): The index of the bond for which the CREST input is being generated.
            ts_xyz_dict (dict): Dictionary mapping paths to the XYZ coordinates of the TS conformers.
            ts_free_energies_dict (dict): Dictionary mapping paths to the free energies of the TS conformers.

        Returns:
            None
        """

        # Only operate on successfully scanned bonds
        if file_idx in self.successful_bonds: 

            # Identify the TS paths and free energies for the current file index
            ts_free_energies_file_idx = {_path: _energy for _path, _energy in ts_free_energies_dict.items() if file_idx == _path.parents[1].name}

            # Identify the lowest energy unconstrained TS conformer from max, minus1 and plus1 
            min_dir = min(ts_free_energies_file_idx, key=lambda k: ts_free_energies_file_idx[k])

            # Create the CREST TS directory
            crest_ts_dir = min_dir.parent.parent / "ts_conformers" / "crest_ts"
            crest_ts_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

            # Write the coordinates into an .XYZ file
            self.write_xyz(crest_ts_dir / "geom_ts.xyz", self.atom_types, ts_xyz_dict[min_dir])

    ## 6. g_sp_ts_confs (X jobs per bond, depending on number of conformers)

    def export_g_sp_ts_confs(self, **kwargs):
        """
        Reads the CREST TS ensemble (or reduced ensemble, if it exists) and generates a Gaussian input file
        for a single point energy (SPE) calculation on each conformer, limited to self.max_ts_sp conformers.

        Parameters:
            **kwargs (dict): Any keyword arguments accepted by `write_gaussian_input`, including:
                cpus (int): Number of processors for Gaussian job.   
                mem (str): Memory allocation for Gaussian job.

        Returns:
            None
        """

        # Initialize a dict to track the Gaussian input files created
        self.g_sp_ts_confs_inps = {file_idx: [] for file_idx in self.structures["file_idx"]}

        # Iterate over the successful bond indexes
        for file_idx in self.bonds_to_process:
            if file_idx in self.successful_crest_bonds:

                # Check if reduced CREST ensemble exists, otherwise use original CREST ensemble
                ensemble = self.job_dir / f"{file_idx}" / "ts_conformers" / "crest_ts" / "crest_conformers_reduced.xyz"
                if not ensemble.exists():
                    ensemble = self.job_dir / f"{file_idx}" / "ts_conformers" / "crest_ts" / "crest_conformers.xyz"
                    if not ensemble.exists():
                        self.pprint(file_idx, "g_sp_ts_confs", "No crest_conformers.xyz available, skipping.")
                    else:
                        self.pprint(file_idx, "g_sp_ts_confs", f"Running CENSO job from {ensemble.name}.")
                else:
                    self.pprint(file_idx, "g_sp_ts_confs", f"Running CENSO job from {ensemble.name}.")

                # Get the number of conformers
                _, __, n_conformers = read_from_xyz(ensemble)

                # Create a folder for each conformation
                out_dir = self.job_dir / f"{file_idx}" / "ts_conformers" / "dft_sp"
                out_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

                # Write the Gaussian input file for a SPE on each conformer
                for i in range(n_conformers):

                    # Limit to X conformers
                    if i > self.max_ts_sp:
                        break

                    # Parse the XYZ coordinates of the ith conformer
                    xyz_block, _, __ = get_conformer_xyz(ensemble, i)
                    xyz_out = xyz_block[2:]

                    # Get the coordinates in list format
                    xyz = [' '.join(line.split()[1:]) for line in xyz_out]

                    # Define the output directory
                    out_dir_i = self.job_dir / f"{file_idx}" / "ts_conformers" / "dft_sp" / f"{i+1}"
                    out_dir_i.mkdir(mode=0o775, parents=True, exist_ok=True)

                    # Define the route card for these jobs
                    route = f"{self.opt_method} {self.opt_basis} {self.gauss_solv} nosymm"

                    # Write the Gaussian input file
                    input_gjf = out_dir_i / "g_sp_ts_confs.gjf"
                    self.write_gaussian_input(input_gjf, route, self.atom_types, xyz, kwargs["cpus"], kwargs["mem"])
                    self.g_sp_ts_confs_inps[file_idx] += [input_gjf]

    ## 7. g_opt_ts_confs_frozen (<=10 jobs per bond, depending on previous SPEs)    

    def export_g_opt_ts_confs_frozen(self, **kwargs):
        """
        Selects the X lowest energy TS conformers, where X is defined by self.conformers_to_optimize,
        based on the previous SPE calculations and creates a Gaussian input file for a constrained
        optimisation calculation on each. The constraints are atom-wise on each atom in the dihedral
        angles (6 in total).

        Parameters:
            **kwargs (dict): Any keyword arguments accepted by `write_gaussian_input`, including:
                cpus (int): Number of processors for Gaussian job.   
                mem (str): Memory allocation for Gaussian job.

        Returns:
            None
        """

        # Initialize a dict to track the Gaussian input files created
        self.g_opt_ts_confs_frozen_inps = {file_idx: [] for file_idx in self.structures["file_idx"]}

        # Iterate over the successful bond indexes
        for file_idx in self.bonds_to_process:
            if file_idx in self.successful_crest_bonds:

                # Get the directories containing completed SPE calculations
                current_dir = self.job_dir / f"{file_idx}"
                logs = list(current_dir.glob(r"ts_conformers/dft_sp/*/*.log"))
                out_dir = current_dir / "ts_conformers" / "dft_opt_frozen"
                out_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
                if not any(logs):
                    self.pprint(file_idx, "g_opt_ts_confs", "No sp log available, skipping.")
                    return
                
                # Get the energies and coordinates of the SPE calculations
                energies = []
                energy_xyz_sp_conf_ts = {}
                for log in logs:
                    scf_energies, _, _, _, xyz_coords = parse_with_cclib(log)
                    energy = float(scf_energies[-1])
                    energies.append(energy)
                    energy_xyz_sp_conf_ts[energy] = xyz_coords[-1]

                # Sort the energies
                energies = sorted(energies, reverse=False)

                # Get the number of constrained optimizations to initialize. We add self.ts_frozen_extra for some
                # leeway, since some will not surpass the energy threshold to go to unconstrained optimizations.
                calcs_to_initialize = self.conformers_to_optimize[file_idx] + self.ts_frozen_extra

                # Define the route card for these jobs
                route = f"opt=modredundant {self.opt_method} {self.opt_basis} {self.gauss_solv} IOP(1/8=2) nosymm"

                # Get the atoms to be constrained
                scan_dir = current_dir / "scan_dft_1"
                _, __, constrained_atoms = self.get_indexes_from_scan_inp(scan_dir / "dft_scan.gjf")

                # We constrain each atom individually, rather than the dihedral
                tail_list = []
                for i in range(len(constrained_atoms)):
                    tail_list += [f"{constrained_atoms[i]} F"]

                # Convert the N lowest energy SPE calculations to constrained optimizations
                jobs = 0
                for i in range(len(energies)):

                    # Stop after initializing the specified number of constrained optimizations
                    if jobs >= calcs_to_initialize:
                        break

                    # Only take negative energy conformers (these are likely to be true TS conformers)
                    if energies[i] < 0:
                        jobs += 1

                        # Get the coordinates in list format
                        xyz = energy_xyz_sp_conf_ts[energies[i]]

                        # Define the output directory
                        out_dir_i = out_dir / f"{i+1}"
                        out_dir_i.mkdir(mode=0o775, parents=True, exist_ok=True)

                        # Write the new Gaussian input file for the constrained optimization
                        input_gjf = out_dir_i / "g_opt_ts_confs.gjf"
                        self.write_gaussian_input(input_gjf, route, self.atom_types, xyz, kwargs["cpus"], kwargs["mem"], tail_list)
                        self.g_opt_ts_confs_frozen_inps[file_idx] += [input_gjf]

    ## 8. g_opt_ts_confs (1 job per bond)    

    def export_g_opt_ts_confs(self, ts_scan_scf_energies_dict, **kwargs):
        """
        Reads the constrained TS optimisation output files and compares their energies to the unconstrained
        TS optimisations from the scans. If any new conformer (from the constrained TS optimisation) is within
        X Hartree of the TSs from the scan, where X is defined by self.ts_opt_energy_window, a Gaussian input
        file for an unconstrained optimisation of the new conformer is created.

        Parameters:
            ts_scan_scf_energies_dict (dict): Dictionary containing energies of the unconstrained TS optimisations from the scans.
            **kwargs (dict): Any keyword arguments accepted by `write_gaussian_input`, including:
                cpus (int): Number of processors for Gaussian job.   
                mem (str): Memory allocation for Gaussian job.

        Returns:
            None
        """

        # Initialize a dict to track the Gaussian input files created
        self.g_opt_ts_confs_inps = {file_idx: [] for file_idx in self.structures["file_idx"]}

        # Iterate over the successful bond indexes
        for file_idx in self.bonds_to_process:
            if file_idx in self.successful_crest_bonds:

                # Specify location of frozen optimisation files
                current_dir = self.job_dir / f"{file_idx}"
                logs = list(current_dir.glob(r"ts_conformers/dft_opt_frozen/*/*.log"))

                # Create a new directory for storing the unconstrained TS optimisations
                # This will only be populated if the unconstrained TSs are stable enough
                out_dir = current_dir / "ts_conformers" / "dft_ts"
                out_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

                # Check the frozen optimizations exist
                if not any(logs):
                    self.pprint(
                        file_idx,
                        "g_opt_ts_confs analysis",
                        "No opt log available, skipping.",
                    )
                    return

                # Get energies of frozen optimisation from X_Y/ts_conformers/dft_opt_frozen
                energies = []
                energy_xyz_opt_conf_ts = {}
                for log in logs:
                    scf_energies, _, _, _, xyz_coords = parse_with_cclib(log)
                    energy = float(scf_energies[-1])
                    energies.append(energy)
                    energy_xyz_opt_conf_ts[energy] = xyz_coords[-1]

                # Sort the energies
                energies = sorted(energies, reverse=False)

                # Get the energy of the most stable unconstrained scan TS (for comparison)
                scan_TS_energies = [ts_scan_scf_energies_dict[_path] for _path in ts_scan_scf_energies_dict.keys() if file_idx == _path.parents[1].name]
                scan_TS_energies = sorted(scan_TS_energies, reverse=False)
                lowest_scan_TS_energy = scan_TS_energies[0]

                # Get the number of unconstrained optimizations to initialize
                calcs_to_initialize = self.conformers_to_optimize[file_idx]

                # Define the route card for these calculations
                route = f"opt=(ts,noeigen,recalcfc=15) {self.opt_method} {self.opt_basis} {self.gauss_solv} freq IOP(1/8=2) nosymm"

                # Convert the N lowest energy constrained calculations to unconstrained optimizations
                jobs = 0
                for i in range(len(energies)):
                    if jobs >= calcs_to_initialize:
                        break

                    # Compare the energy of the constrained TS to the energy of the most stable unconstrained scan TS
                    # If the new structure is X Hartrees more stable than the scan TS we setup a full TS optimization on it
                    energy = energies[i]
                    augmented_energy = energy + self.ts_opt_energy_window
                    if (augmented_energy < 0) and (augmented_energy < lowest_scan_TS_energy):
                        jobs += 1

                        # Get the coordinates in list format
                        xyz_out = energy_xyz_opt_conf_ts[energy]

                        # Define the output directory
                        out_dir_i = out_dir / f"{i+1}"
                        out_dir_i.mkdir(mode=0o775, parents=True, exist_ok=True)

                        # Write the Gaussian input file
                        input_gjf = out_dir_i / "g_opt_ts_confs.gjf"
                        self.write_gaussian_input(input_gjf, route, self.atom_types, xyz_out, kwargs["cpus"], kwargs["mem"])
                        self.g_opt_ts_confs_inps[file_idx] += [input_gjf]
                        
                # Report if no optimisations were created
                if jobs == 0:
                    self.pprint(file_idx, "g_opt_ts_confs", f"No TS conformers within {self.ts_opt_energy_window} Hartree of the scan TS, skipping.")

    def identify_lowest_energy_TS(self, ts_scan_free_energies_dict, ts_crest_free_energies_dict, ts_scan_xyz_dict, ts_crest_xyz_dict):
        """
        For each rotatable bond, identifies the overall lowest energy transition state (TS) from all
        TS conformers resulting from either the scan or CREST search and compiles them into a single
        dictionary, ts_lowest_xyz_dict. These TSs are used as the final TS for barrier calculations.

        Parameters:
            ts_scan_free_energies_dict (dict): Dictionary mapping paths to the free energies of the scan TSs.
            ts_crest_free_energies_dict (dict): Dictionary mapping paths to the free energies of the CREST TSs.
            ts_scan_xyz_dict (dict): Dictionary mapping paths to the coordinates of the scan TSs.
            ts_crest_xyz_dict (dict): Dictionary mapping paths to the coordinates of the CREST TSs.

        Returns:
            ts_lowest_xyz_dict (dict): Dictionary mapping paths to the coordinates of the lowest energy TS for each bond index.
        """

        # Initialize dictionary to hold the final TS index
        ts_lowest_xyz_dict = {}

        # Iterate over the successful bond indexes
        for file_idx in self.bonds_to_process:

            # Get the scan dictionary for this index
            scan_dict = {
                _path: _energy
                for _path, _energy in ts_scan_free_energies_dict.items()
                if file_idx == _path.parents[1].name
            }

            # Get the CREST dictionary for this index
            crest_dict = {
                _path: _energy
                for _path, _energy in ts_crest_free_energies_dict.items()
                if file_idx == _path.parents[2].name
            }

            # Combine the dictionaries
            combined_dict = {**scan_dict, **crest_dict}

            # Find the lowest energy TS and its coordinates
            lowest_TS_path = min(combined_dict, key=combined_dict.get) if combined_dict else None
            lowest_TS_xyz = ts_scan_xyz_dict.get(lowest_TS_path) or ts_crest_xyz_dict.get(lowest_TS_path)
            self.pprint(file_idx, "Final conformers", f"Lowest energy TS at {lowest_TS_path}")

            # Update the final TS dictionary with the lowest energy TS
            ts_lowest_xyz_dict.update({lowest_TS_path: lowest_TS_xyz})

        return ts_lowest_xyz_dict

    ## 9. g_sp_final (1 job + 1 job per bond)

    def export_g_sp_final(self, ts_lowest_xyz_dict, **kwargs):
        """
        Creates a new directory, sp_final, and populates it with the optimization .log files for the lowest
        energy reactant and TS structures for each rotatable bond, geometry.xyz files for visualisation, and
        generates Gaussian input files for single point energy (SPE) calculations on each. 

        Parameters:
            ts_lowest_xyz_dict (dict): A dictionary containing the index and geometry of the lowest energy TS.
            **kwargs (dict): Any keyword arguments accepted by `write_gaussian_input`, including:
                cpus (int): Number of processors for Gaussian job.   
                mem (str): Memory allocation for Gaussian job.

        Returns:
            None
        """

        # Define the route card for these jobs
        route = f"{self.sp_method} {self.sp_basis} {self.gauss_solv} nosymm integral=NoXCTest"

        # Initialize dicts to track the final log files and the Gaussian input files created
        self.g_opt_final_logs = {file_idx: None for file_idx in self.structures["file_idx"]}
        self.g_sp_final_inps = {file_idx: [] for file_idx in self.structures["file_idx"]}

        # Iterate over the structures in the dataframe
        for row_idx in self.structures.index:
            file_idx = self.structures.at[row_idx, "file_idx"]
            
            # Operate on the reactant file
            if file_idx == "reactant_gaussian":

                # Define the log file
                log_file = Path("g_opt_react.log")
                log_path = self.job_dir / "reactant_gaussian" / log_file

                # Check the log file exists
                if not log_path.exists():
                    self.pprint(
                        file_idx,
                        "sp_final",
                        "reactant geometry not available, skipping.",
                    )
                    return

                # Create the SPE directory
                spe_dir = self.job_dir / "sp_final" / "reactant"
                spe_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

                # Copy the Gaussian output to the SPE directory
                spe_log_path = spe_dir / log_file
                shutil.copy(str(log_path), str(spe_log_path))
                self.g_opt_final_logs[file_idx] = spe_log_path

                # Get the coordinates of the log file
                _, _, _, _, xyz_coords = parse_with_cclib(log_path)

                # Write the Gaussian input file for the SPE and the geometry.xyz file for visualisation in the GUI
                input_gjf = spe_dir / f"{log_file.stem}_sp.gjf"
                self.write_gaussian_input(input_gjf, route, self.atom_types, xyz_coords[-1], kwargs["cpus"], kwargs["mem"])
                self.write_xyz(spe_dir / "geometry.xyz", self.atom_types, xyz_coords[-1])
                self.g_sp_final_inps[file_idx] += [input_gjf]

            # Operate on successful TS files
            elif file_idx in self.successful_bonds:
    
                # Get the path to the lowest energy TS for this bond from ts_lowest_xyz_dict
                log_dir = next(
                    (_path for _path in ts_lowest_xyz_dict if file_idx in (_path.parents[1].name, _path.parents[2].name)),
                    None
                )

                # Define the log file
                log_file = Path("g_opt_ts.log") if "opt_ts" == log_dir.parent.name else Path("g_opt_ts_confs.log")
                log_path = log_dir / log_file

                # Check the log file exists
                if not log_path.exists():
                    self.pprint(
                        file_idx,
                        "sp_final",
                        "TS geometry not available, skipping.",
                    )
                    return

                # Create the SPE directory
                spe_dir = self.job_dir / "sp_final" / f"ts_{file_idx}"
                spe_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

                # Copy the Gaussian output to the SPE directory
                spe_log_path = spe_dir / log_file
                shutil.copy(str(log_path), str(spe_log_path))
                self.g_opt_final_logs[file_idx] = spe_log_path

                # Get the coordinates of the log file
                _, _, _, _, xyz_coords = parse_with_cclib(log_path)

                # Write the Gaussian input file for the SPE and the geometry.xyz file for visualisation in the GUI
                input_gjf = spe_dir / f"{log_file.stem}_sp.gjf"
                self.write_gaussian_input(input_gjf, route, self.atom_types, xyz_coords[-1], kwargs["cpus"], kwargs["mem"])
                self.write_xyz(spe_dir / "geometry.xyz", self.atom_types, xyz_coords[-1])
                self.g_sp_final_inps[file_idx] += [input_gjf]

    ## 10. Goodvibes

    def run_goodvibes(self, g_opt_final_logs):
        """
        Runs Goodvibes calculations for reactant and transition state (TS) structures to extract free energies.

        Parameters:
            g_opt_final_logs (dict): A dictionary containing the paths to the final Gaussian log files for reactant and TS structures.

        Returns:
            None
        """

        # Initialize list to store subprocess handles
        processes = []

        # Iterate over the final log files for reactant and TS structures
        for file_idx, log_path in g_opt_final_logs.items():

            # Skip non-existent log files
            if log_path is None:
                continue

            # Define the output goodvibes file path
            gv_out = log_path.parent / "goodvibes_output"

            # Run the goodvibes command in a subprocess if the output file doesn't already exist
            if not Path(gv_out).exists():
                p = subprocess.Popen(
                    f"python -m goodvibes --qs {self.qh_entropy} -v {self.vib_freq_scale_factor} --temp {self.temperature} {log_path} --spc sp > {gv_out}",
                    shell=True,
                )

                # Add the process to the list of running processes
                processes.append(p)
                self.pprint(file_idx, "goodvibes", f"Running Goodvibes on {log_path.name}")

            else:
                self.pprint(file_idx, "goodvibes", f"{gv_out.name} exists, skipping.")

        # Wait for all subprocesses to complete
        [p.wait() for p in processes]

    # Define the gauss function

    @task
    async def gauss(self):
        """
        Run the Gaussian calculations for the atropisomer optimization and analysis:
         - Optimizes the lowest CENSO reactant structure from `qm` (workflow does not wait for this calculation).
         - Performs a DFT scan on the reactant of the atropisomeric bond in both directions using the dihedrals from cheminformatics.
         - Optimizes three TS conformers (minus1, max, plus1) from the scan with constraints on the dihedrals.
         - Optimizes the three constrained TSs from scans without constraints.
         - Performs a CREST conformational search of one of the TSs from above.
         - Performs SPE calculations on each TS conformer from CREST to filter conformers.
         - Optimizes a selection of the lowest energy TS conformers from CREST with constraints on the dihedrals, and then without.
         - Identifies the lowest energy TS conformer overall resulting from either the initial scans or the CREST search.
         - Performs SPE calculations on the final reactant and lowest energy TS conformer.
         - Extracts free energies with Goodvibes for the final reactant and lowest energy TS conformer.
        """

        # Start a new output for Gaussian calculations
        self.add_output()

        # Define the progress bars
        g_opt_react_pbar = ProgressBar("g_opt_react")
        g_scan_pbar = ProgressBar("g_scan")
        g_opt_ts_frozen_pbar = ProgressBar("g_opt_ts_frozen")
        g_opt_ts_pbar = ProgressBar("g_opt_ts")
        crest_ts_pbar = ProgressBar("crest_ts")
        g_sp_ts_confs_pbar = ProgressBar("g_sp_ts_confs")
        g_opt_ts_confs_frozen_pbar = ProgressBar("g_opt_ts_confs_frozen")
        g_opt_ts_confs_pbar = ProgressBar("g_opt_ts_confs")
        g_sp_final_pbar = ProgressBar("g_sp_final")

        # Define the CENSO ensemble output for Gaussian optimization input
        censo_ensemble = self.job_dir / "reactant_censo" / "censo" / "enso_ensemble_part2.xyz"

        # From: reactant_censo/censo/enso_ensemble_part2.xyz
        # Create: reactant_gaussian/g_opt_react.gjf
        self.export_g_opt_react(
            input_xyz=censo_ensemble,
            output_gjf=self.job_dir / "reactant_gaussian" / "g_opt_react.gjf",
            **self.gauss_low_kwargs,
        )

        # Run: g_opt_react.gjf (but do not wait for it)
        g_opt_react_jobs = self.run_gaussian(
            row_idx=1,
            step_name="g_opt_react",
            pbar=g_opt_react_pbar, 
            input_gjfs=self.g_opt_react_inps, 
            **self.gauss_low_kwargs
            )
            
        # Unlike the other Gaussian jobs throughout this workflow, no future jobs rely on the
        # reactant optimisation, so we not not need to wait for it to finish before continuing
        # the workflow. As such, we cannot close the progress bar as normal after awaiting for
        # the job so we need to attach a callback to the jobs task to close the progress bar
        # upon completion.
        def cb(_):
            """Callback to close the pbar later once job finishes."""
            self.pprint("ALL", "g_opt_react", "All jobs done.", highlight=True)
            g_opt_react_pbar.close_pbar()

        # Add the callback to each job
        if g_opt_react_jobs:
            for job in g_opt_react_jobs:
                job.task.add_done_callback(cb)

        # Define the rotatable bonds to process from cheminformatics
        self.bonds_to_process = list(self.substructures.keys())

        # Create a dictionary of scans to perform (initially, this is a forward and backward scans for all rotatable bonds)
        scans_to_perform = {}
        for rotatable_bond in self.bonds_to_process:
            scans_to_perform[rotatable_bond] = {"forward": True, "backward": True}

        # From: reactant_censo/censo/enso_ensemble_part2.xyz
        # Create:
        #  - X_Y/scan_dft_1/dft_scan.gjf
        #  - X_Y/scan_dft_2/dft_scan.gjf 
        self.export_g_scan(censo_ensemble, scans_to_perform, **self.gauss_high_kwargs)

        # Run the DFT scans created above
        for row_idx in self.structures.index:
            if row_idx >= 2:
                file_idx = self.structures.at[row_idx, "file_idx"]
                self.run_gaussian(
                    row_idx=row_idx, 
                    step_name="g_scan", 
                    pbar=g_scan_pbar, 
                    input_gjfs=self.g_scan_inps[file_idx], 
                    **self.gauss_high_kwargs
                    )

        # Wait for the scans to finish
        await asyncio.gather(
            *[
                job.task
                for job in self.structures["g_scan_jobs"].sum()
                if job is not None
            ]
        )

        # Analyse scans to obtain dictionaries that contain the most stable max, minus1 and plus1 geometries
        max_scan_xyz_dict, minus1_scan_xyz_dict, plus1_scan_xyz_dict = self.scan_analysis() 
        self.pprint("ALL", "g_scan", "All jobs done.", highlight=True)
        g_scan_pbar.close_pbar()

        # Create a list of bonds that were successfully analyzed
        # Bonds whose scans could not be analyzed are dropped from this point
        self.successful_bonds = [
            bond
            for bond in self.bonds_to_process
            if any(
                bond == part
                for path in max_scan_xyz_dict
                for part in Path(path).parts
            )
        ]

        # From the lowest energy max, minus1 and plus1 geometries, create:
        #  - X_Y/opt_frozen/max/g_opt_frozen.gjf
        #  - X_Y/opt_frozen/minus1/g_opt_frozen.gjf
        #  - X_Y/opt_frozen/plus1/g_opt_frozen.gjf
        self.export_g_opt_ts_frozen(
            max_scan_xyz_dict,
            minus1_scan_xyz_dict,
            plus1_scan_xyz_dict,
            **self.gauss_high_kwargs,
        )

        # Run the frozen optimisations created above
        for row_idx in self.structures.index:
            if row_idx >= 2:
                file_idx = self.structures.at[row_idx, "file_idx"]
                self.run_gaussian(
                    row_idx=row_idx, 
                    step_name="g_opt_ts_frozen", 
                    pbar=g_opt_ts_frozen_pbar, 
                    input_gjfs=self.g_opt_ts_frozen_inps[file_idx], 
                    **self.gauss_high_kwargs
                )

        # Wait for the frozen optimisations to finish
        await asyncio.gather(
            *[
                job.task
                for job in self.structures["g_opt_ts_frozen_jobs"].sum()
                if job is not None
            ]
        )

        self.pprint("ALL", "g_opt_frozen", "All jobs done.", highlight=True)
        g_opt_ts_frozen_pbar.close_pbar()

        # From the frozen optimisations above, create:
        #  - X_Y/opt_ts/max/g_opt_ts.gjf
        #  - X_Y/opt_ts/minus1/g_opt_ts.gjf
        #  - X_Y/opt_ts/plus1/g_opt_ts.gjf
        self.export_g_opt_ts(**self.gauss_high_kwargs)

        # Run the unconstrained optimisations created above
        for row_idx in self.structures.index:
            if row_idx >= 2:
                file_idx = self.structures.at[row_idx, "file_idx"]
                self.run_gaussian(
                    row_idx=row_idx, 
                    step_name="g_opt_ts", 
                    pbar=g_opt_ts_pbar, 
                    input_gjfs=self.g_opt_ts_inps[file_idx], 
                    **self.gauss_high_kwargs
                )

        # Wait for the unconstrained optimisations to finish
        await asyncio.gather(
            *[
                job.task
                for job in self.structures["g_opt_ts_jobs"].sum()
                if job is not None
            ]
        )

        # Analyse TSs to obtain dictionaries that contain their SCF/free energies and geometries
        ts_scan_xyz_dict, ts_scan_scf_energies_dict, ts_scan_free_energies_dict = self.ts_analysis(analysis_type='scan')
        self.pprint("ALL", "g_opt_ts", "All jobs done.", highlight=True)
        g_opt_ts_pbar.close_pbar()

        # Create a list of bonds that were successfully analyzed
        # Bonds whose TSs could not be analyzed are dropped from this point
        self.successful_bonds = [
            bond
            for bond in self.bonds_to_process
            if any(
                bond == part
                for path in ts_scan_xyz_dict
                for part in Path(path).parts
            )
        ]

        # From: the lowest energy TS conformer from above
        # Create: a CREST input file: X_Y/ts_conformers/crest_ts/geom_ts.xyz
        for row_idx in self.structures.index:
            if row_idx >= 2:
                file_idx = self.structures.at[row_idx, "file_idx"]
                self.export_crest_ts(file_idx, ts_scan_xyz_dict, ts_scan_free_energies_dict)

        # Setup CREST jobs to generate the .xcontrol file for constraints
        for row_idx in self.structures.index:
            if row_idx >= 2:

                # Get the atoms to be constrained
                scan_input = self.job_dir / f"{file_idx}" / "scan_dft_1" / "dft_scan.gjf"
                _, __, constrained_atoms = self.get_indexes_from_scan_inp(scan_input)

                # Initialize the CREST job
                self.run_crest_xcontrol(
                    row_idx=row_idx,
                    step_name="crest_ts_xcontrol",
                    input_xyz=self.job_dir / f"{file_idx}" / "ts_conformers" / "crest_ts" / "geom_ts.xyz",
                    output=self.job_dir / f"{file_idx}" / "ts_conformers" / "crest_ts" / "crest_xcontrol.out",
                    constrained_atoms=constrained_atoms,
                    **self.crest_xcontrol_kwargs
                    )

        # Run CREST jobs and wait for the .xcontrol files to be created
        await asyncio.gather(
            *[
                job.task
                for job in self.structures["crest_ts_xcontrol_jobs"].sum()
                if job is not None
            ]
        )

        # Setup CREST jobs for the TS conformational search
        for row_idx in self.structures.index:
            file_idx = self.structures.at[row_idx, "file_idx"]
            input_xyz = Path(f"{self.job_dir}/{file_idx}/ts_conformers/crest_ts/geom_ts.xyz")
            output = Path(f"{self.job_dir}/{file_idx}/ts_conformers/crest_ts/crest_ts.out")
            xcontrol_file = Path(f"{self.job_dir}/{file_idx}/ts_conformers/crest_ts/.xcontrol.sample")
            if row_idx >= 2:
                self.run_crest(
                    row_idx=row_idx, 
                    step_name="crest_ts",
                    pbar=crest_ts_pbar,
                    input_xyz=input_xyz,
                    output=output,
                    xcontrol=xcontrol_file,
                    **self.crest_kwargs
                    )
            
        # Run CREST jobs and wait for them to finish
        await asyncio.gather(
            *[
                job.task
                for job in self.structures["crest_ts_jobs"].sum()
                if job is not None
            ]
        )

        self.pprint("ALL", "crest_ts", "All jobs done.", highlight=True)
        crest_ts_pbar.close_pbar()

        # Create a list of bonds that were successfully searched with CREST
        # Bonds whose TSs could not be searched are dropped from this point
        # We also determine the number of conformers to optimize later here
        self.successful_crest_bonds = []
        self.conformers_to_optimize = {}
        for row_idx in self.structures.index:
            if row_idx > 1:
                file_idx = self.structures.at[row_idx, "file_idx"]

                # Define the two output files from CREST
                crest_ts_output = self.job_dir / f"{file_idx}" / "ts_conformers/crest_ts/crest_ts.out"
                crest_ts_xyz = self.job_dir / f"{file_idx}" / "ts_conformers/crest_ts/crest_conformers.xyz"

                # Check if the job was successful from the .out file
                if crest_ts_output.exists():
                    with open(crest_ts_output, 'r') as f:
                        for line in f:
                            if "CREST terminated normally" in line:
                                self.successful_crest_bonds.append(file_idx)
                                break

                # If it was successful and has a crest_conformers.xyz file...
                if (file_idx in self.successful_crest_bonds) and crest_ts_xyz.exists():

                    # Check how many conformers were generated by CREST for each bond
                    _, _, num_confs = read_from_xyz(crest_ts_xyz)

                    # Determine how many TS conformers to optimize after the CREST TS search and SPEs.
                    # We optimize one-tenth the number of conformers, but guarantee its at least 5
                    # and no more than 25, to avoid situations where we optimize 50+ TS conformers.
                    target_conformers = int(num_confs * self.frac_ts_opt)
                    confs_to_opt = min(
                        max(self.min_ts_opt, target_conformers),
                        self.max_ts_opt,
                    )
                    self.conformers_to_optimize[file_idx] = confs_to_opt
                        
        # If any CREST search was successful, perform conformer elimination and subsequent optimizations
        if self.successful_crest_bonds:

            # Setup the redundant conformer eliminations on the CREST TS ensemble
            for row_idx in self.structures.index:
                file_idx = self.structures.at[row_idx, "file_idx"]
                if row_idx >= 2 and file_idx in self.successful_crest_bonds:
                    self.run_elim(
                        row_idx=row_idx, 
                        step_name="elim_ts",
                        input=Path(f"{self.job_dir}/{file_idx}/ts_conformers/crest_ts/crest_conformers.xyz"),
                        )

            # Report on the number of retained or eliminated conformers
            for row_idx in self.structures.index:
                file_idx = self.structures.at[row_idx, "file_idx"]
                if row_idx >= 2 and file_idx in self.successful_crest_bonds:
                    file_idx = self.structures.at[row_idx, "file_idx"]
                    self.report_elim(
                        row_idx, 
                        step_name="elim_ts", 
                        input=Path(f"{self.job_dir}/{file_idx}/ts_conformers/crest_ts/elim.out")
                    )

            # Report all elimination jobs as finished
            self.pprint("ALL", "elim_ts", "All jobs done.", highlight=True)

            # From: X_Y/ts_conformers/crest_ts/crest_conformers.xyz OR crest_conformers_reduced.xyz
            # Create: DFT SPE calculation on n TS conformers: X_Y/ts_conformers/dft_sp/n/g_sp_ts_confs.gjf
            self.export_g_sp_ts_confs(**self.gauss_low_kwargs)

            # Run the SPE jobs created above
            for row_idx in self.structures.index:
                if row_idx >= 2:
                    file_idx = self.structures.at[row_idx, "file_idx"]
                    self.run_gaussian(
                        row_idx=row_idx, 
                        step_name="g_sp_ts_confs", 
                        pbar=g_sp_ts_confs_pbar, 
                        input_gjfs=self.g_sp_ts_confs_inps[file_idx], 
                        **self.gauss_low_kwargs
                    )

            # Wait for the SPE jobs to finish
            await asyncio.gather(
                *[
                    job.task
                    for job in self.structures["g_sp_ts_confs_jobs"].sum()
                    if job is not None
                ]
            )

            self.pprint("ALL", "g_sp_ts_confs", "All jobs done.", highlight=True)
            g_sp_ts_confs_pbar.close_pbar()

            # From: the n lowest energy conformers from the SPEs above
            # Create: n constrained TS optimisations: X_Y/ts_conformers/dft_opt_frozen/n/g_opt_ts_confs.gjf
            self.export_g_opt_ts_confs_frozen(**self.gauss_high_kwargs)

            # Run the constrained TS optimisation jobs created above
            for row_idx in self.structures.index:
                if row_idx >= 2:
                    file_idx = self.structures.at[row_idx, "file_idx"]
                    self.run_gaussian(
                        row_idx=row_idx, 
                        step_name="g_opt_ts_confs_frozen", 
                        pbar=g_opt_ts_confs_frozen_pbar, 
                        input_gjfs=self.g_opt_ts_confs_frozen_inps[file_idx], 
                        **self.gauss_high_kwargs
                    )

            # Wait for the constrained TS optimisations to finish
            await asyncio.gather(
                *[
                    job.task
                    for job in self.structures["g_opt_ts_confs_frozen_jobs"].sum()
                    if job is not None
                ]
            )

            self.pprint("ALL", "g_opt_ts_confs", "All jobs done.", highlight=True)
            g_opt_ts_confs_frozen_pbar.close_pbar()

            # Compare the energies of the constrained TS optimizations resulting from CREST to
            # ts_scan_scf_energies_dict, the energies of the unconstrained TS optimisations resulting
            # from the initial scans. If the most stable constrained TS is within X Hartrees of the most
            # stable unconstrained scan TS, create: X_Y/ts_conformers/dft_ts/n/g_opt_ts_confs.gjf
            self.export_g_opt_ts_confs(ts_scan_scf_energies_dict, **self.gauss_high_kwargs)

            # Run the unconstrained optimisation(s) created above (if any exist)
            for row_idx in self.structures.index:
                if row_idx >= 2:
                    file_idx = self.structures.at[row_idx, "file_idx"]
                    self.run_gaussian(
                        row_idx=row_idx, 
                        step_name="g_opt_ts_confs", 
                        pbar=g_opt_ts_confs_pbar, 
                        input_gjfs=self.g_opt_ts_confs_inps[file_idx], 
                        **self.gauss_high_kwargs
                    )

            # Wait for the unconstrained TS optimisation(s) to finish
            await asyncio.gather(
                *[
                    job.task
                    for job in self.structures["g_opt_ts_confs_jobs"].sum()
                    if job is not None
                ]
            )

            # Analyse TSs to obtain dictionaries that contain their free energies and geometries
            ts_crest_xyz_dict, _, ts_crest_free_energies_dict = self.ts_analysis(analysis_type='crest')

            self.pprint("ALL", "g_opt_ts_confs", "All jobs done.", highlight=True)
            g_opt_ts_confs_pbar.close_pbar()

        # If CREST search was unsuccessful, skip conformer elimination and subsequent optimizations
        else:
            ts_crest_free_energies_dict = {}
            ts_crest_xyz_dict = {}
        
        # Raise error if no valid TSs were identified via scans or CREST
        if not ts_scan_xyz_dict and not ts_crest_xyz_dict:
            raise ValueError("No valid TSs were identified during analysis. Please check the log files and contact an expert if you believe this is an error.")

        # From both sets of TS optimisations (from the scans and from the CREST search)
        # identify which is the most stable overall to use as the final TS for barriers
        ts_lowest_xyz_dict = self.identify_lowest_energy_TS(ts_scan_free_energies_dict, ts_crest_free_energies_dict, ts_scan_xyz_dict, ts_crest_xyz_dict)
        self.pprint("reactant_gaussian", "Final conformers", f"Lowest energy reactant at reactant_gaussian")
        self.pprint("ALL", "Final conformers", "All final structures identified.", highlight=True)

        # Create a list of bonds that had the lowest TS successfully identified
        # Bonds whose lowest TS could not be identified are dropped from this point
        self.successful_bonds = [
            bond
            for bond in self.bonds_to_process
            if any(
                bond == part
                for path in ts_lowest_xyz_dict
                for part in Path(path).parts
            )
        ]

        # Wait for the original reactant gaussian calculation to finish
        g_opt_react_jobs = self.structures.at[1, "g_opt_react_jobs"]
        if g_opt_react_jobs:
            await asyncio.gather(
                *[
                    job.task
                    for job in g_opt_react_jobs
                    if job is not None
                ]
            )

        # Move reactant and TS optimization files to sp_final and create SPE inputs
        self.export_g_sp_final(ts_lowest_xyz_dict, **self.gauss_low_kwargs)

        # Run the SPE jobs created above and wait for results
        for row_idx in self.structures.index:
            if row_idx >= 1:
                file_idx = self.structures.at[row_idx, "file_idx"]
                self.run_gaussian(
                    row_idx=row_idx,
                    step_name="g_sp_final",
                    pbar=g_sp_final_pbar,
                    input_gjfs=self.g_sp_final_inps[file_idx],
                    **self.gauss_low_kwargs
                )

        # Wait for the SPE jobs to finish
        await asyncio.gather(
            *[
                job.task
                for job in self.structures["g_sp_final_jobs"].sum()
                if job is not None
            ]
        )
        self.pprint("ALL", "g_sp_final", "All jobs done.", highlight=True)
        g_sp_final_pbar.close_pbar()

        # Run goodvibes on the reactant and TS calculations
        self.run_goodvibes(self.g_opt_final_logs)
        self.pprint("ALL", "goodvibes", "All jobs done.", highlight=True)

        # Report all jobs done
        self.pprint("ALL", "ALL", "All jobs done.", highlight=True)

        # Remove job columns from dataframe
        self.structures = self.structures.drop(
            [
                "g_opt_react_jobs",
                "g_scan_jobs",
                "g_opt_ts_frozen_jobs",
                "g_opt_ts_jobs",
                "crest_ts_xcontrol_jobs",
                "crest_ts_jobs",
                "g_sp_ts_confs_jobs",
                "g_opt_ts_confs_frozen_jobs",
                "g_opt_ts_confs_jobs",
                "g_sp_final_jobs",
            ],
            axis=1,
            errors="ignore", # ignore nonexistent job columns
        )

    # Define the conclude functions

    def assign_class(self, barrier):
        """
        Assigns the LaPlante atropisomer class based on the value of the barrier.

        Parameters:
            barrier (float): The barrier value to classify.

        Returns:
            int: The assigned class.
        """
        return 3 if barrier > 30 else 1 if barrier < 20 else 2

    def generate_image(self):
        """
        Generates an image with annotated barriers for the molecule.
        """

        import cairosvg
        from IPython.display import SVG
        from rdkit.Chem.Draw import rdMolDraw2D

        # Setup drawing options
        width, height = (400,400)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().annotationFontScale = 1.0

        # Prepare highlight and annotation mappings
        highlight_atoms = []
        highlight_bonds = []

        # Iterate over each atropisomeric bond
        for row_idx in self.structures.index:
            file_idx = self.structures.at[row_idx, "file_idx"]

            # Only annotate successful bond indexes
            if file_idx in self.successful_indexes:
                
                # Get the corrected barrier
                barrier = round(self.structures.at[row_idx, "ΔG activation_r (kcal/mol)"], 1)

                # Get the two atoms of the bond
                idx_temp = file_idx.split()
                idx1 = int(idx_temp[1].split("_")[0]) - 1
                idx2 = int(idx_temp[1].split("_")[1]) - 1

                # Check if there is a central atom associated with this bond
                # This will be the case for diaryl ethers, amines, etc
                central_atom = self.central_atoms.get((int(idx1), int(idx2)))
                if central_atom is None:
                    central_atom = self.central_atoms.get((int(idx2), int(idx1)))
                
                # For most situations...
                if not central_atom:

                    # Get the rotatable bond object
                    bond = self.structures.at[0, "mol"].GetBondBetweenAtoms(idx1, idx2)

                    # Annotate the barrier
                    bond.SetProp("bondNote", str(barrier))

                    # Store the bond information
                    highlight_atoms.extend([idx1, idx2])
                    highlight_bonds.append(bond.GetIdx())

                # For diaryls...
                else:

                    # Get the central atom and both bonds connected to it
                    c_atom = self.structures.at[0, "mol"].GetAtomWithIdx(central_atom)
                    bond1 = self.structures.at[0, "mol"].GetBondBetweenAtoms(idx1, central_atom)
                    bond2 = self.structures.at[0, "mol"].GetBondBetweenAtoms(idx2, central_atom)

                    # Annotate the barrier
                    c_atom.SetProp("atomNote", str(barrier))

                    # Store the bond information
                    highlight_atoms.extend([idx1, central_atom, idx2])
                    highlight_bonds.append(bond1.GetIdx())
                    highlight_bonds.append(bond2.GetIdx())

        # Get the molecule
        mol = self.structures.at[0, "mol"]

        # Highlight atropisomeric bonds
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightBonds=highlight_bonds,
        )

        # Draw the molecule
        drawer.FinishDrawing()
        image = drawer.GetDrawingText()

        # Save the image as a PNG
        cairosvg.svg2png(
            bytestring=SVG(image).data, dpi=3000, output_width=width, output_height=height,
            write_to=f'{self.results_dir}/structure.png'
        )

    def simplify_dataframe(self, df):
        """
        Simplify the results DataFrame by removing unnecessary columns, standardizing
        naming conventions, converting data types, and cleaning bond index values.

        Parameters:
            df (pandas.DataFrame): Input DataFrame containing conformer/energy results.

        Returns:
            pandas.DataFrame: Simplified DataFrame.

        """

        # Define columns to remove
        del_columns = ["mol",
                       "confs",
                       "index",
                       "ROMol",
                       "exported_confs",
                       "stable_xyz",
                       "crest_exists",
                       "crest_equivalent",
                       "censo_part1_exists",
                       "censo_part1_equivalent",
                       "censo_part2_exists",
                       "censo_part2_equivalent",
                       "censo_part3_exists",
                       "censo_part3_equivalent",
                       "E (hartree)",
                       "E_SP (hartree)",
                       "G_SP (hartree)",
                       "QH_G_SP (hartree)",
                       "ΔG activation (kcal/mol)",
                       ]
        to_remove = [col for col in del_columns if col in df]

        # Remove some columns
        df.drop(columns=to_remove, inplace=True)
        
        # Remove the initial structure
        df.drop(index=0, inplace=True)

        # Rename some parameters
        df = df.rename(
            columns={
                "file_idx":"Bond index",
                "substructure": "Substructure",
                "ΔG activation (kcal/mol)": "Raw ΔG of activation (kcal/mol)",
                "ΔG activation_r (kcal/mol)": "ΔG of activation (kcal/mol)",
            }
        )
        
        # Convert the columns to the correct types
        df['ΔG of activation (kcal/mol)'] = df['ΔG of activation (kcal/mol)'].astype(float)
        df['ΔG of activation (kcal/mol)'] = df['ΔG of activation (kcal/mol)'].round(2)

        # Split the bond index column to keep only the relevant part
        bond_index_column = df['Bond index'] 
        def split_name(x):
            x = x.split()[1]
            return x
        new_bond_index_column = bond_index_column.apply(split_name)
        df['Bond index'] = new_bond_index_column

        return df

    def conclude(self):
        """
        Concludes the workflow:
         - Collects energies and free energies for each structure. 
         - Calculates the free energy barrier to rotation, half-life of interconversion and atropisomer class for each rotational axis.
         - Scales the barrier using a linear regression equation derived from a benchmark of experimental vs. computed rotational barriers. 
         - Generates and saves images depicting the structure with barriers and atropisomer bonds annotated.

        """

        # Create the results directory
        self.results_dir = Path(self.job_dir/'results')
        self.results_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

        # Iterate over the structures in the dataframe
        for row_idx in self.structures.index:
            file_idx = self.structures.at[row_idx, "file_idx"]

            # Operate on the Gaussian reactant
            if file_idx == "reactant_gaussian":
                gv_out = self.job_dir / "sp_final" / "reactant" / "goodvibes_output" 

            # Operate on successful rotatable bond indexes
            elif file_idx in self.successful_bonds:
                gv_out = self.job_dir / "sp_final"/ f"ts_{file_idx}" / "goodvibes_output"

            # Skip other structures (e.g., failed bonds)
            else:
                continue

            # Check the goodvibes output exists
            if not gv_out.exists():
                raise FileNotFoundError(f"Goodvibes output file not found for {file_idx}. Please check the log files.")

            # Initialize variables
            warning = None
            energies_line = None

            # Attempt to extract goodvibes energies
            with open(gv_out,'r') as file:
                for line in file:

                    # Check for a warning
                    if "Warning!" in line:
                        warning = line.strip()
                        break

                    # Find the line that contains the headers
                    if "Structure" in line:
                        file.readline() # Skip the *** line
                        energies_line = file.readline() # Get the energies line
                        break

            # Report if Goodvibes encountered a warning for this structure
            if warning:
                self.pprint(file_idx, "goodvibes", f"Warning encountered in goodvibes output, skipping: {warning}")

            # Extract relevant energies and populate the dataframe
            elif energies_line:
                parts = energies_line.split()
                self.structures.at[row_idx, "E (hartree)"] = round(float(parts[3]), 6) # Optimization energy (E)
                self.structures.at[row_idx, "E_SP (hartree)"] = round(float(parts[2]), 6) # SPE energy (E_SPC)
                self.structures.at[row_idx, "G_SP (hartree)"] = round(float(parts[8]), 6) # SPE-corrected free energy (G(T)_SPC)
                self.structures.at[row_idx, "QH_G_SP (hartree)"] = round(float(parts[9]), 6) # Quasiharmonic SPE-corrected free energy (qh-G(T)_SPC)

            else:
                self.pprint(file_idx, "goodvibes", f"Could not extract energies from goodvibes output, skipping.")

        # Copy the reactant structure to the new row and drop the original reactant row
        self.structures.at[1, "mol"] = self.structures.at[0, "mol"]
        self.structures = self.structures.drop(index=0,axis=0).reset_index()

        # Iterate over the structures in the dataframe
        for row_idx in self.structures.index:
            file_idx = self.structures.at[row_idx, "file_idx"]

            # Operate on successful rotatable bond indexes
            if file_idx in self.successful_bonds:

                # Calculate the free energy barrier in kcal/mol
                DG = (self.structures.at[row_idx, "QH_G_SP (hartree)"] - self.structures.at[0, "QH_G_SP (hartree)"]) * 627.5095
                self.structures.at[row_idx, "ΔG activation (kcal/mol)"] = round(DG, 2)

                # Correct the free energy barrier with the regression
                DG_r = (self.structures.at[row_idx, "ΔG activation (kcal/mol)"] * self.m) + self.c
                self.structures.at[row_idx, "ΔG activation_r (kcal/mol)"] = round(DG_r, 2)

                # Calculate the atropisomer class
                self.structures.at[row_idx, "Class"] = self.assign_class(DG_r)

        # Label first structure in the DataFrame as "reactant" and all subsequent 
        # structures as transition states with their corresponding file_idx values
        self.successful_indexes = []
        for row_idx in self.structures.index:
            file_idx = self.structures.at[row_idx, "file_idx"]
            if row_idx == 0:
                self.structures.iloc[row_idx, 1] = "reactant"
            elif file_idx in self.successful_bonds:
                new_index = f"ts {file_idx}"
                self.structures.iloc[row_idx, 1] = new_index
                self.successful_indexes.append(new_index)

        # Generate and save the annotated structure image
        self.generate_image()

        # Remove failed bond indexes from self.structures
        failed_indexes = [idx for idx in self.structures["file_idx"] if idx not in self.successful_indexes and idx != "reactant"]
        self.structures = self.structures[~self.structures['file_idx'].isin(failed_indexes)]

        # Save the results dataframe as a .csv file
        self.structures.to_csv(self.results_dir / "energies.csv", index=False)

        # Simplify the results and save that too
        simple_results = self.simplify_dataframe(self.structures)
        simple_results.to_csv(self.results_dir / "results.csv", index=False)

    # Define the plotting functions

    def calculate_half_life(self, T, G):
        """
        Calculates the half life of an atropisomer at a given temperature and activation barrier:
         - The Eyring equation relates the rate constant to the activation free energy barrier (ΔG‡) and temperature (T)
         - The half-life of interconversion (t₁/₂) is related to the rate constant (k) by: t₁/₂ = ln(2) / k
         - Combining these equations allows calculation of t₁/₂ from ΔG‡ and T.

        Parameters:
            T (float): The temperature (°C).
            G (float): The activation free energy barrier (kcal/mol).

        Returns:
            float: The computed half-life (seconds).

        """

        # Define constants
        kB = 1.3806488E-23  #  Boltzmann constant
        h = 6.62606957E-34  # Planck's constant

        # Calculate half-life
        return math.log(2) / (2 * (kB * (273.15 + T) / h) * math.exp(-G * 1000 / (1.987 * (273.15 + T))))

    def convert_units(self, array):
        """
        Converts the input array from seconds to the unit specified by self.half_life_units.

        Supported units are:
            - 'seconds' (no conversion)
            - 'minutes'
            - 'hours'
            - 'days'
            - 'years'

        Parameters:
            array (numpy.ndarray): A numpy array containing half-life values in seconds.

        Returns:
            numpy.ndarray: A numpy array containing half-life values in the specified units.
        """

        if self.half_life_units == 'seconds':
            return array # default
        elif self.half_life_units == 'minutes':
            return array / self.seconds_in_minute
        elif self.half_life_units == 'hours':
            return array / self.seconds_in_hour
        elif self.half_life_units == 'days':
            return array / self.seconds_in_day
        elif self.half_life_units == 'years':
            return array / self.seconds_in_year

    def format_number(self, num):
        """
        Formats a number as a string, using scientific notation for very large or very small values.
        Scientific notation is used if the absolute value of the number is greater than or equal to
        1e4, or less than 1e-3 (excluding zero). Otherwise, returns the number with two decimal places.

        Args:
            num (float): The number to format.

        Returns:
            str: The formatted number as a string.
        """

        # Use scientific notation
        if abs(num) >= 1e4 or (abs(num) < 1e-3 and num != 0):
            return f"{num:.2e}"

        # Use standard float representation
        else:
            return f"{num:.2f}"

    def proportion_through_range(self, min_value, max_value, value):
        """
        Given a minimum and maximum value defining a range, and a value within that range, this method
        returns the proportion (between 0 and 1) that the value represents within the range.

        Parameters:
            min_value (float): The minimum value of the range.
            max_value (float): The maximum value of the range.
            value (float): The value for which to calculate the proportion. Must satisfy min_value <= value <= max_value.

        Returns:
            float: The proportion of the value within the range [min_value, max_value].

        Raises:
            ValueError: If the value is not within the specified range.
        """

        # Ensure that the value is within the range
        if not (min_value <= value <= max_value):
            raise ValueError("The value must be within the range [min_value, max_value].")

        # Calculate the range
        range_value = max_value - min_value

        # Calculate the distance from the minimum value
        distance_from_min = value - min_value

        return (distance_from_min / range_value)

    def add_constant_half_life_line(self, fig, half_life):
        """
        Adds a contour line representing a constant half-life to a Plotly figure and calculates the angle of the line.

        Parameters:
            fig (plotly figure): The Plotly figure to annotate.
            half_life (float): The value of the half-life to be represented as a contour line.

        Returns:
            plotly figure: The updated Plotly figure.
            angle (float): The angle (in degrees) of the contour line.
        """

        # Add a contour line at the specified half-life value
        fig.add_trace(go.Contour(
            z=self.Z,
            x=self.T[0],
            y=self.G[:, 0],
            contours=dict(
                type='constraint',
                operation='=',
                value=half_life
            ),
            line=dict(
                color='RGB(100,100,100)',
                dash='dash',
                width=1,
            ),
            showlegend=False, # hide legend
            hoverinfo='skip', # don't change the hover info
            ),
        )

        # Determine the angle of the line assuming it can be represented by y = mx + c
        x0, x1 = self.T[0][0], self.T[0][-1]
        y0, y1 = self.G[0, 0], self.G[-1, 0]

        # Calculate the slope (m) of the line
        slope = (y1 - y0) / (x1 - x0) if x1 != x0 else float('inf')

        # Calculate the angle (in degrees) using the arctangent function
        angle = np.degrees(np.arctan(slope)) if x1 != x0 else 90.0

        return fig, angle

    def annotate_constant_half_life_line(self, fig, text, x, y, angle):
        """
        Annotates a constant half-life line on a Plotly figure with a custom text label.

        Parameters:
            fig (plotly figure): The Plotly figure to annotate.
            text (str): The text to display as the annotation.
            x (float): The x-coordinate for the annotation.
            y (float): The y-coordinate for the annotation.
            angle (float): The angle (in degrees) to rotate the annotation text.

        Returns:
            plotly figure: The updated Plotly figure.
        """

        # Add text at the specified coordinates
        fig.add_annotation(
            x=x, # x-coordinate of text
            y=y, # y-coordinate of text
            text=text, # the text to display
            showarrow=False, # whether to show an arrow pointing to the text
            font=dict(color='RGB(240,240,240)', size=13), # font settings
            xanchor='center', # horizontal alignment of the text
            yanchor='middle', # vertical alignment of the text
            textangle=angle  # angle of the text
        )
        return fig

    def add_class_boundary_line(self, fig, boundary):
        """
        Adds a horizontal boundary line to the given Plotly figure at the specified y-value.

        Parameters:
            fig (plotly figure): The Plotly figure to annotate.
            boundary (float): The y-value at which to draw the horizontal boundary line.

        Returns:
            plotly figure: The updated Plotly figure.
        """

        # Add a horizontal line at the specified boundary value
        fig.add_shape(
            type='line',
            x0=np.min(self.T), x1=np.max(self.T),
            y0=boundary, y1=boundary,
            line=dict(
                color='black',
                dash='dot',
                width=1
                ),
            showlegend=False, # hide legend
            )

        return fig

    def annotate_class_boundary_line(self, fig, text, subtext, x, y):
        """
        Adds two text annotations to a Plotly figure at specified coordinates to label a class boundary line.

        Parameters:
            fig (plotly figure): The Plotly figure to annotate.
            text (str): The main annotation text to display at the specified (x, y) coordinates.
            subtext (str): The secondary annotation text to display slightly below the main text.
            x (float): The x-coordinate for both annotations.
            y (float): The y-coordinate for the main annotation text. The secondary text is placed at (x, y-3).

        Returns:
            plotly figure: The updated Plotly figure.
        """

        # Add the main annotation text at the specified coordinates
        fig.add_annotation(
            x=x, # x-coordinate of the text
            y=y, # y-coordinate of the text
            text=text, # the text to display
            showarrow=False,  # whether to show an arrow pointing to the text
            font=dict(color='black', size=13.5), # font settings
            xanchor='center',  # horizontal alignment of the text
            yanchor='middle',  # vertical alignment of the text
        )

        # Add the secondary annotation text slightly below the main text
        fig.add_annotation(
            x=x,  # x-coordinate of the text
            y=y-3,  # y-coordinate of the text
            text=subtext, # the text to display
            showarrow=False, # whether to show an arrow pointing to the text
            font=dict(color='black', size=11), # font settings
            xanchor='center', # horizontal alignment of the text
            yanchor='middle', # vertical alignment of the text
        )

        return fig

    def add_prediction(self, fig, T, G):
        """
        Adds a prediction marker to the given Plotly figure.

        Parameters:
            fig (plotly figure): The Plotly figure to annotate.
            T (float or int): The x-coordinate (temperature) for the prediction marker.
            G (float or int): The y-coordinate (Gibbs free energy barrier) for the prediction marker.

        Returns:
            plotly figure: The updated Plotly figure.
        """

        # Add a marker at the specified coordinates
        fig.add_trace(go.Scatter(
            x=[T],
            y=[G],
            mode='markers',
            marker=dict(size=11, color='black', symbol='star'),
            showlegend=False, # hide legend
            name="Prediction",
        ))

        return fig

    def generate_half_life_plot(self, barrier, temperature, half_life_units):
        """
        Calculates half-life and atropisomer class information. Creates a contour plot of the half-life of interconversion.

        Params:
            results: a dataframe containing the bond indexes and rotational barriers.
            temperature: a float indicating the temperature specified by the user.
            half_life_units: a string indicating the units for the half-life.
        """

        # Define parameters
        self.temperature = temperature
        self.half_life_units = half_life_units

        # Define regions of interest for plotting
        self.seconds_in_minute = 60
        self.seconds_in_hour = 3600
        self.seconds_in_day = 86400
        self.seconds_in_year = 3.154e+7

        # Generate a grid of values for T and G
        T = np.linspace(0, 250, 100) # range for temperature
        G = np.linspace(0, 55, 100) # range for barriers
        self.T, self.G = np.meshgrid(T, G)

        # Vectorize the half-life equation function for array inputs
        self.Z = np.vectorize(self.calculate_half_life)(self.T, self.G)

        # Convert the time units if necessary
        self.Z_format = self.convert_units(self.Z)

        # Convert self.Z to scientific notation (for hover template)
        format_vectorized = np.vectorize(self.format_number)
        self.Z_format = format_vectorized(self.Z_format)

        # Get the min and max values in self.Z (for color scale)
        self.min, self.max = np.min(self.Z), np.max(self.Z)

        # Define colors for the scale
        green = "green" # green for free interconversion
        red = "RGB(185,0,0)" # red in the problematic region
        yellow = "RGB(255,255,0)" # yellow close to boundaries
        blue = "RGB(60,110,170)" # blue for limited interconversion

        # Define a custom colorscale based on the custom regions
        # The first value is a float between 0 and 1 that represents
        # the proportion of the colour scale to be a particular colour
        colorscale = [
            [0.0, green],
            [self.proportion_through_range(self.min, self.max, 25), yellow],
            [self.proportion_through_range(self.min, self.max, 1000), red],
            [self.proportion_through_range(self.min, self.max, 2*self.seconds_in_year), yellow],
            [self.proportion_through_range(self.min, self.max, 100*self.seconds_in_year), blue],
            [self.proportion_through_range(self.min, self.max, self.max), blue]
        ]

        # Create the contour plot
        fig = go.Figure(go.Contour(
            z=self.Z, # half-life
            x=self.T[0], # temperature
            y=self.G[:, 0], # barrier
            colorscale=colorscale, # colour scale
            showscale=False, # hide the legend
            contours=dict(coloring='heatmap'),
            line=dict(color='rgba(0,0,0,0)'), # make contour lines transparent
            customdata=self.Z_format, # pass Z to the hover template
            hovertemplate=(
                'Temperature: %{x:.1f} °C<br>' +
                'Barrier: %{y:.1f} kcal.mol<sup>-1</sup><br> ' +
                'Half-life: %{customdata}<extra></extra> ' + self.half_life_units
            )
        ))

        # Identify critical regions
        fig, angle_a = self.add_constant_half_life_line(fig, 1000)
        fig, angle_b = self.add_constant_half_life_line(fig, self.seconds_in_day)
        fig, angle_c = self.add_constant_half_life_line(fig, self.seconds_in_year)

        # Annotate critical regions
        fig = self.annotate_constant_half_life_line(fig, "1 year", 74.2, 31.9, -angle_a-5)
        fig = self.annotate_constant_half_life_line(fig, "24 hours", 106.8, 33.2, -angle_b-5)
        fig = self.annotate_constant_half_life_line(fig, "1000 seconds", 142.8, 32.5, -angle_c-2.5)

        # Add atropisomer class boundaries
        fig = self.add_class_boundary_line(fig, 20) # class 1-2 boundary
        fig = self.add_class_boundary_line(fig, 30) # class 2-3 boundary

        # Annotate critical regions
        fig = self.annotate_class_boundary_line(fig, "<b>Class 3</b>", "Atropisomer - limited<br>interconversion (at RT)", 223, 35)
        fig = self.annotate_class_boundary_line(fig, "<b>Class 2</b>", "Atropisomer - interconverts<br>at problematic timescale", 223, 27.5)
        fig = self.annotate_class_boundary_line(fig, "<b>Class 1</b>", "Rapid interconversion - exists<br>as equilibrating mixture", 223, 17.5)
    
        # Plot the computed barrier
        if barrier is not None:
            fig = self.add_prediction(fig, self.temperature, barrier)

        # Update figure layout
        fig.update_layout(
            xaxis=dict(
                title_text='Temperature (°C)',
                tickcolor='white',
                linecolor='white',
            ),
            yaxis=dict(
                title_text='Rotational barrier (kcal.mol<sup>-1</sup>)',
                tickcolor='white',
                linecolor='white',
            ),
            width=1000, # set the width in pixels
            height=700,  # set the height in pixels
        )

        # Display figure
        fig.show()
