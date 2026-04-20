import math
import cclib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IPython.display import SVG
from itertools import combinations, product
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw

class Atropisomers():
    """
    A class to run the atropisomers workflow. 
    
    The following operations are included:
     - Process SMILES strings and identify potential atropisomeric axes and the dihedral angles required for scanning.
     - Apply steric filters to determine if atropisomerism is likely.
     - Identify candidate TS structures from Gaussian scans.
     - Confirm valid rotational TSs after Gaussian optimization.

    """

    def __init__(self, smiles):
        """
        Parameters:
            smiles (str): The SMILES representation of the molecule.
        """

        # Initialize the molecule with RDKit
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)

        # Get number of atoms and charge
        self.num_atoms = self.mol.GetNumAtoms()
        self.charge = Chem.GetFormalCharge(self.mol)

        # Initialize dictionaries to hold GIC dihedrals for each identified rotatable bond
        self.all_gic_1: dict = {}
        self.all_gic_2: dict = {}

        # Initialize a dictionary to hold the substructure pattern for each identified rotatable bond
        self.substructures: dict = {}

        # Initialize an attribute to hold central atoms for coupled torsions
        self.central_atoms: dict = {}
      
    # Cheminformatics helper functions

    def discover_ring_axis_atoms(self, mol, neighbors, match_indexes, axis_atoms):
        """
        Discover ring axis atoms in a molecular structure; this function identifies substituent atoms and lone pairs
        around a given set of neighboring atoms in a molecule. It distinguishes between substituents and lone pairs
        based on whether the neighboring atoms are part of a matched SMARTS index.

        Parameters:
            mol (rdkit.Chem.Mol): the RDKit molecule object.
            neighbors (list): A list of indices representing neighboring atoms.
            match_indexes (set): A set of indices representing matched SMARTS atoms.
            axis_atoms (list): A list to which the indices of axis atoms will be appended.

        Returns:
            tuple: A tuple containing:
                - list: Updated list of axis atom indices.
                - int: Count of lone pairs found.
                - int: Count of protons found.
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
            mol (rdkit.Chem.Mol): The RDKit molecule object (with explicit hydrogens).
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

        Parameters:
            mol (rdkit.Chem.Mol): the RDKit molecule object.
            axis_atoms (list): A list of indices representing the axis atoms.

        Returns:
            proton_count (int): The count of hydrogen atoms among the specified axis atoms.
        """

        # Initialize count
        proton_count = 0

        # Count hydrogens
        for idx in axis_atoms:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "H":
                proton_count += 1

        return proton_count

    def is_atropisomeric(self, lp_1, lp_2, protons_1, protons_2, axis_1_atoms, axis_2_atoms, non_heavy_limit):
        """
        Determines if a molecular structure is potentially atropisomeric based on the local steric environment
        (number of lone pairs, number of protons, and the configuration of the atoms around the specified axes).
        We do not recommend a scan if the total number of lone pairs or protons exceeds the limit set for the
        substructure, or if there is insufficient bulk on the axes, as this indicates that the system is likely
        too unhindered for atropisomerism.

        Parameters:
            lp_1 (int): Number of lone pairs on the first unit.
            lp_2 (int): Number of lone pairs on the second unit.
            protons_1 (int): Number of protons on the first unit.
            protons_2 (int): Number of protons on the second unit.
            axis_1_atoms (list): List of atom indices for the first axis.
            axis_2_atoms (list): List of atom indices for the second axis.
            non_heavy_limit (int): Maximum number of lone pair or protons substituents allowed in total for atropisomerism.

        Returns:
            bool: True if the structure is considered atropisomeric, False otherwise.
        """

        # If steric filters are disabled, always return True
        if not self.filter:
            print(" - no steric filters applied.")
            return True

        # Do not recommend a scan if the number of non-heavy atoms exceeds the limit
        if lp_1 + lp_2 + protons_1 + protons_2 > non_heavy_limit:
            print(f" - too many hydrogens and/or lone pairs (>{non_heavy_limit}): skipping the calculation")
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

    def analyze_biring(self, mol, match_indexes, non_heavy_limit):
        """
        Analyzes a molecular structure with a rings (aryl or non-aryl) attached to one another to determine
        the atom indexes of the central rotatable bond, as well as the indexes of the atoms to use for scans.
        For aryl or aryl-like (consisting of sp2 atoms around the rotatable bond atom) these are the exocyclic
        second neighbors of the ring. Otherwise, these are the direct neighbors of the rotatable bond atom.
        These substituents are required for GIC scans of the rotatable bond in both directions. Additionally,
        this method tracks the number of lone pairs and protons around the rotatable bond and determines whether
        a GIC scan is necessary or whether atropisomerism is unlikely, for example due to lack of steric hindrance.

        Parameters:
            mol (rdkit.Chem.Mol): The RDKit molecule object (with explicit hydrogens).
            match_indexes (list): The atom indices of the matched bi-ring SMARTS substructure.
            max_non_heavy_subs (int): Maximum number of proton or lone pair substituents allowed for atropisomerism.

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
        lp_1, protons_1 = 0, 0
        for neighbor_idx in neighbors_1:
            neighbor = mol.GetAtomWithIdx(neighbor_idx)
            # If the neighbor is sp2, we can use the ortho substituent (if it exists)
            if neighbor.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                axis_atoms, lp, protons = self.discover_ring_axis_atoms(mol, [neighbor_idx], match_indexes, [])
                axis_1_atoms.extend(axis_atoms)
                protons_1 += protons
                lp_1 += lp
            # Otherwise, use the neighbor itself
            else:
                axis_1_atoms.append(neighbor_idx)
                
        # Check the second axis
        lp_2, protons_2 = 0, 0
        for neighbor_idx in neighbors_2:
            neighbor = mol.GetAtomWithIdx(neighbor_idx)
            # If the neighbor is sp2, we can use the ortho substituent (if it exists)
            if neighbor.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                axis_atoms, lp, protons = self.discover_ring_axis_atoms(mol, [neighbor_idx], match_indexes, [])
                axis_2_atoms.extend(axis_atoms)
                protons_2 += protons
                lp_2 += lp
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
        result = self.is_atropisomeric(lp_1, lp_2, protons_1, protons_2, axis_1_atoms, axis_2_atoms, non_heavy_limit)
        if result:
            return result, axis_1_atoms, axis_2_atoms
        else:
            return False, [], []

    def analyze_diaryl(self, mol, match_indexes, non_heavy_limit):
        """
        Analyzes a molecular structure with two aryl rings attached to each other via a heteroaatom,
        e.g., O, N, S, S=O or S(=O)(=O) (diaryl ethers, thioethers, amines, sulfoxides and sulfones)
        to determine the atom indexes of the central atom and the two adjacent rotatable bonds in the
        system, as well as the indexes of their exocyclic second neighbors (two on each ring). These
        substituents are required for GIC scans of the coupled torsion in both directions. Additionally,
        this method tracks the number of lone pairs and protons around the rotatable bonds and determines
        whether a GIC scan is necessary or whether atropisomerism is unlikely, for example due to lack of
        steric hindrance.

        Parameters:
            mol (rdkit.Chem.Mol): The RDKit molecule object (with explicit hydrogens).
            match_indexes (list): The atom indices of the matched diaryl SMARTS substructure.
            max_non_heavy_subs (int): Maximum number of proton or lone pair substituents allowed for atropisomerism.

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
        if central_atom_idx == None:
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
        axis_1_atoms, lp_1, protons_1 = self.discover_ring_axis_atoms(mol, neighbors_1, match_indexes, axis_1_atoms)
        axis_2_atoms, lp_2, protons_2 = self.discover_ring_axis_atoms(mol, neighbors_2, match_indexes, axis_2_atoms)

        # Count the number of hydrogens on all GIC-defining atoms
        proton_count = self.count_hydrogens(mol, axis_1_atoms + axis_2_atoms)
        assert proton_count == (protons_1 + protons_2), "Mismatch in proton count"
        
        ### Step 3. Final analysis

        # 1-index the axis atoms
        axis_1_atoms = [idx + 1 for idx in axis_1_atoms]
        axis_2_atoms = [idx + 1 for idx in axis_2_atoms]

        # Determine whether an atropisomeric scan is appropriate
        result = self.is_atropisomeric(lp_1, lp_2, protons_1, protons_2, axis_1_atoms, axis_2_atoms, non_heavy_limit)
        if result:
            return result, axis_1_atoms, axis_2_atoms
        else:
            return False, [], []

    def analyze_aryl_X(self, mol, match_indexes, non_heavy_limit):
        """
        Analyzes a molecular structure with an aryl ring attached to specific non-aromatic units; amides (bezamides),
        thioamides (thiobenzamide), their reversed forms (reversed (thio)benzamides), sulfones/sulfoxides, and
        sulfonamides/sulfinamides, to determine the atom indexes of the central rotatable bond, as well as the indexes
        of the two exocyclic second neighbors of the aromatic ring and the two direct neighbors of the out-of-ring atom
        (e.g., the =O, =S, or R-group). These substituents are required for GIC scans of the rotatable bond in both
        directions. Additionally, this method tracks the number of lone pairs and protons around the rotatable bond and
        determines whether a GIC scan is necessary or whether atropisomerism is unlikely, for example due to lack of
        steric hindrance.

        Parameters:
            mol (rdkit.Chem.Mol): The RDKit molecule object (with explicit hydrogens).
            match_indexes (list): The atom indices of the matched aryl-X SMARTS substructure.
            max_non_heavy_subs (int): Maximum number of proton or lone pair substituents allowed for atropisomerism.

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
        if not out_ring_atom_idx or not in_ring_atom_idx:
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

        # In theory, there cannot be lone pairs on these functional groups, but to be sure we can
        # approximate the number of lone pairs on the out-of-ring substituents by subtracting the
        # number of identified substituents from three (since we know we have a benzamide, reverse
        # benzamide, or sulfoxide, we can assume the out-of-ring atom is a heteroatom (O, S, N) with
        # up to three bonds)).
        out_ring_lp = 3 - len(out_ring_axis_atoms)

        # Count the number of hydrogens on all GIC-defining atoms
        proton_count = self.count_hydrogens(mol, in_ring_axis_atoms + out_ring_axis_atoms)
        assert proton_count == (in_ring_protons + out_ring_protons), "Mismatch in proton count"

        ### Step 3. Final analysis

        # 1-index the axis atoms
        axis_1_atoms = [idx + 1 for idx in in_ring_axis_atoms]
        axis_2_atoms = [idx + 1 for idx in out_ring_axis_atoms]

        # Determine whether an atropisomeric scan is appropriate
        result = self.is_atropisomeric(in_ring_lp, out_ring_lp, in_ring_protons, out_ring_protons, axis_1_atoms, axis_2_atoms, non_heavy_limit)
        if result:
            return result, axis_1_atoms, axis_2_atoms
        else:
            return False, [], []

    def analyze_coupled_aryl_X(self, mol, match_indexes, non_heavy_limit):
        """
        Analyzes a molecular structure with an aryl ring attached to specific non-aromatic units; amides
        (bezamides), thioamides (thiobenzamide), and their reversed forms (reversed (thio)benzamides), to
        determine the atom indexes of the central atom and the two adjacent rotatable bonds in the system,
        as well as the indexes of the two exocyclic second neighbors of the aromatic ring and the two direct
        neighbors of the out-of-ring atom (e.g., the =O or R-group). These substituents are required for
        GIC scans of the coupled torsion in both directions. Additionally, this method tracks the number of
        lone pairs and protons around the rotatable bonds and determines whether a GIC scan is necessary or
        whether atropisomerism is unlikely, for example due to lack of steric hindrance.

        Parameters:
            mol (rdkit.Chem.Mol): The RDKit molecule object (with explicit hydrogens).
            match_indexes (list): The atom indices of the matched aryl-X SMARTS substructure.
            max_non_heavy_subs (int): Maximum number of proton or lone pair substituents allowed for atropisomerism.

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

        # Identify all substructures of the molecule corresponding to an amide or thioamide
        amide_smarts = "[#7]-[#6](=[#8])"
        thioamide_smarts = "[#7]-[#6](=[#16])"
        sulfoxide_smarts = "[*]-[#16](=[#8])"
        sulfone_smarts = "[*]-[#16](=[#8])(=[#8])"
        sulfinamide_smarts = "[#16](=[#8])-[#7]"
        sulfonamide_smarts = "[#16](=[#8])(=[#8])"
        amide_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(amide_smarts))
        thioamide_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(thioamide_smarts))
        sulfoxide_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(sulfoxide_smarts))#
        sulfone_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(sulfone_smarts))
        sulfinamide_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(sulfinamide_smarts))
        sulfonamide_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(sulfonamide_smarts))
        all_X_smarts = amide_matches + thioamide_matches + sulfoxide_matches + sulfone_matches + sulfinamide_matches + sulfonamide_matches

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
                                rotatable_atom_indexes = [atom_i, atom_j, atom_c]
                                rotatable_bond_indexes = [(atom_i, atom_c), (atom_c, atom_j)]
                                rotatable_bond_idx = (atom_i, atom_j)
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
        if central_atom_idx == None:
            print(" - invalid pattern for atropisomerism (no out-of-ring central atom): skipping the calculation")
            return False, [], []

        # Store the central atom with respect to the other atoms to use later
        self.central_atoms[(in_ring_atom_idx, out_ring_atom_idx)] = central_atom_idx

        # Define the neighbors for each atom in the rotatable bond (excluding the central atom)
        in_ring_atom = mol.GetAtomWithIdx(in_ring_atom_idx)
        out_ring_atom = mol.GetAtomWithIdx(out_ring_atom_idx)
        atom_in_ring_neighbors = [n.GetIdx() for n in in_ring_atom.GetNeighbors() if n.GetIdx() not in rotatable_bond_indexes and n.GetIdx() != central_atom_idx]
        atom_out_ring_neighbors = [n.GetIdx() for n in out_ring_atom.GetNeighbors() if n.GetIdx() not in rotatable_bond_indexes and n.GetIdx() != central_atom_idx]

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

        # In theory, there cannot be lone pairs on these functional groups, but to be sure we can
        # approximate the number of lone pairs on the out-of-ring substituents by subtracting the
        # number of identified substituents from three (since we know we have a benzamide, reverse
        # benzamide, or sulfoxide, we can assume the out-of-ring atom is a heteroatom (O, S, N) with
        # up to three bonds)).
        out_ring_lp = 3 - len(out_ring_axis_atoms)

        # Count the number of hydrogens on all GIC-defining atoms
        proton_count = self.count_hydrogens(mol, in_ring_axis_atoms + out_ring_axis_atoms)
        assert proton_count == (in_ring_protons + out_ring_protons), "Mismatch in proton count"
    
        ### Step 3. Final analysis

        # 1-index the axis atoms
        axis_1_atoms = [idx + 1 for idx in in_ring_axis_atoms]
        axis_2_atoms = [idx + 1 for idx in out_ring_axis_atoms]

        # Determine whether an atropisomeric scan is appropriate
        result = self.is_atropisomeric(in_ring_lp, out_ring_lp, in_ring_protons, out_ring_protons, axis_1_atoms, axis_2_atoms, non_heavy_limit)
        if result:
            return result, axis_1_atoms, axis_2_atoms
        else:
            return False, [], []

    def analyze_amide(self, mol, match_indexes, non_heavy_limit):
        """
        Analyzes an amide or thioamide and determines the atom indexes of the central rotatable bond,
        as well as the indexes of their four neighbors. These substituents are required for GIC scans
        of the rotatable bond in both directions. Additionally, this method tracks the number of lone
        pairs and protons around the rotatable bond and determines whether a GIC scan is necessary or
        whether atropisomerism is unlikely, for example due to lack of steric hindrance.

        Parameters:
            mol (rdkit.Chem.Mol): The RDKit molecule object (with explicit hydrogens).
            match_indexes (list): The atom indices of the matched amide SMARTS substructure.
            max_non_heavy_subs (int): Maximum number of proton or lone pair substituents allowed for atropisomerism.

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
                rotatable_bond_indexes.append(idx)
            elif atom.GetSymbol() == "N":
                rotatable_bond_indexes.append(idx)

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
        atom_1 = mol.GetAtomWithIdx(rotatable_bond_indexes[0])
        atom_2 = mol.GetAtomWithIdx(rotatable_bond_indexes[1])
        neighbors_1 = [n.GetIdx() for n in atom_1.GetNeighbors() if n.GetIdx() not in rotatable_bond_indexes]
        neighbors_2 = [n.GetIdx() for n in atom_2.GetNeighbors() if n.GetIdx() not in rotatable_bond_indexes]

        ### Step 2. Define the indexes of the axis atoms required for GIC scans
        
        # Initialize each axis with the relevant ring atom from the rotatable bond
        axis_1_atoms = [rotatable_bond_indexes[0]]
        axis_2_atoms = [rotatable_bond_indexes[1]]

        # Add the substituents of each atom in the rotatable bond to the axis atoms
        axis_1_atoms.extend(neighbors_1)
        axis_2_atoms.extend(neighbors_2)

        # Count the number of protons on each axis atom
        protons_1 = self.count_hydrogens(mol, neighbors_1)
        protons_2 = self.count_hydrogens(mol, neighbors_2)

        # In theory, there cannot be lone pairs on these functional groups, but to be sure we can
        # approximate the number of lone pairs on by subtracting the number of identified substituents
        # from two (since we know we have an amide, we can assume the central atom is a carbon or
        # nitrogen with up to three bonds))
        lp_1 = 2 - len(neighbors_1)
        lp_2 = 2 - len(neighbors_2)

        # Count the number of hydrogens on all GIC-defining atoms
        proton_count = self.count_hydrogens(mol, axis_1_atoms + axis_2_atoms)
        assert proton_count == (protons_1 + protons_2), "Mismatch in proton count"

        ### Step 3. Final analysis

        # 1-index the axis atoms
        axis_1_atoms = [idx + 1 for idx in axis_1_atoms]
        axis_2_atoms = [idx + 1 for idx in axis_2_atoms]

        # Final analysis: determine whether an atropisomeric scan is appropriate
        result = self.is_atropisomeric(lp_1, lp_2, protons_1, protons_2, axis_1_atoms, axis_2_atoms, non_heavy_limit)
        if result:
            return result, axis_1_atoms, axis_2_atoms
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
            mol (rdkit.Chem.Mol): The RDKit molecule object
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
        filtered_gics = {k: v for k, v in gic_dict.items() if k not in keys_to_remove}

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
           identify the rotable bond and its substituents and determine if atropisomerism is likely (and thus if a GIC scan if appropriate).
        3. Records the atom indices required for GIC scans of each rotatable bond to the self.all_gic_1 and self.all_gic_2 dictionaries.
        4. De-duplicates results so each bond/dihedral is recorded only once.
        5. If no atropisomeric bonds are found, an error is raised.

        Parameters:
            filter (bool): Whether to apply steric filters to determine if a GIC scan is necessary.
            coupled_benzamide (bool): Whether to parse benzamides and reverse benzamides as coupled torsions.
        """

        self.filter = filter

        ### 1. Define SMARTS patterns and formal names for potential atropisomeric patterns
        # For each SMARTS pattern, also define the maximum number of non-heavy substituents that are allowed around the rotatable bond
        # for atropisomerism to be likely (based on the approximate steric bulk around the rotatable bond). These rules are based on a
        # series of reference calculations and are aimed at capturing any barrier above ~12 kcal/mol. Any structures that violate the
        # rules are deemed to have insufficient bulk for atropisomerism and are filtered out. We use 12 kcal/mol as a safe margin to
        # catch as many Class 2 atropisomers as possible. For aryl rings, we are counting how many of the ortho substituents are lone
        # pairs or protons. For non-ring systems, such as benzamides or sulfoxides, we are counting the immediate neighbors of the atom
        # in the rotatable bond. For some of these non-aryl systems, some substituents are heavy by definition (e.g., sulfoxides always
        # have an =O atom). This is taken into account by the rules.
        
        PATTERNS = {

            # Joined aryl-aryl ring systems
            "Biaryl (7-7)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1", # two 7-membered aromatics joined by a single bond
                "non_heavy_limit": 3, # minimum of 1/4 possible ring substituents
            },
            "Biaryl (7-6)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[*]1:[*]:[*]:[*]:[*]:[*]:1", # 7- and 6-membered rings joined by a single bond
                "non_heavy_limit": 3, # minimum of 1/4 possible ring substituents
            },
            "Biaryl (7-5)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[*]1:[*]:[*]:[*]:[*]:1", # 7- and 5-membered rings joined by a single bond
                "non_heavy_limit": 2, # minimum of 2/4 possible ring substituents
            },
            "Biaryl (6-6)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[*]1:[*]:[*]:[*]:[*]:[*]:1", # two 6-membered aromatic rings joined by a single bond
                "non_heavy_limit": 3, # minimum of 1/4 possible ring substituents
            },
            "Biaryl (6-5)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[*]1:[*]:[*]:[*]:[*]:1", # 6- and 5-membered rings joined by a single bond
                "non_heavy_limit": 2, # minimum of 2/4 possible ring substituents
            },
            "Biaryl (5-5)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:1-[*]1:[*]:[*]:[*]:[*]:1", # two 5-membered aromatic rings joined by a single bond
                "non_heavy_limit": 1, # minimum of 3/4 possible ring substituents
            },

            # Heteroatom-joined aryl-aryl ring systems
            # Includes diaryl ethers, thioethers (sulfoxides and sulfones) and amines
            "Diaryl ether (7-7)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[#8]-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 3, # minimum of 1/4 possible ring substituents
            },
            "Diaryl ether (7-6)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[#8]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 2, # minimum of 2/4 possible ring substituents
            },
            "Diaryl ether (7-5)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[#8]-[*]1:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 2, # minimum of 2/4 possible ring substituents
            },
            "Diaryl ether (6-6)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#8]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 2, # minimum of 2/4 possible ring substituents
            },
            "Diaryl ether (6-5)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#8]-[*]1:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # minimum of 3/4 possible ring substituents
            },
            "Diaryl ether (5-5)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:1-[#8]-[*]1:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # minimum of 3/4 possible ring substituents
            },
            "Diaryl thioether (7-7)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[#16]-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 3, # as for diaryl ethers
            },
            "Diaryl thioether (7-6)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#16]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 2, # as for diaryl ethers
            },
            "Diaryl thioether (7-5)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#16]-[*]1:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 2, # as for diaryl ethers
            },
            "Diaryl thioether (6-6)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#16]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 2, # as for diaryl ethers
            },
            "Diaryl thioether (6-5)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#16]-[*]1:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # as for diaryl ethers
            },
            "Diaryl thioether (5-5)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:1-[#16]-[*]1:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # as for diaryl ethers
            },
            "Diaryl amine (7-7)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 2, # minimum of 2/4 possible ring substituents
            },
            "Diaryl amine (7-6)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 2, # minimum of 2/4 possible ring substituents
            },
            "Diaryl amine (7-5)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#7]-[*]1:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 2, # minimum of 2/4 possible ring substituents
            },
            "Diaryl amine (6-6)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 2, # minimum of 2/4 possible ring substituents
            },
            "Diaryl amine (6-5)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[#7]-[*]1:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 2, # minimum of 2/4 possible ring substituents
            },
            "Diaryl amine (5-5)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:1-[#7]-[*]1:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # minimum of 3/4 possible ring substituents
            },

            # Joined aryl-nonaryl ring systems
            "Aryl-aliphatic (7-7)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]~[*]~[*]1", # 7-membered aromatic and 7-membered non-aromatic joined by a single bond
                "non_heavy_limit": 4, # minimum of 0/4 possible ring substituents
            },
            "Aryl-aliphatic (7-6)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]~[*]1", # 7-membered aromatic and 6-membered non-aromatic joined by a single bond
                "non_heavy_limit": 4, # minimum of 0/4 possible ring substituents
            },
            "Aryl-aliphatic (6-7)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]~[*]~[*]1", # 6-membered aromatic and 7-membered non-aromatic joined by a single bond
                "non_heavy_limit": 4, # minimum of 0/4 possible ring substituents
            },
            "Aryl-aliphatic (7-5)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]~[*]1", # 7-membered aromatic and 5-membered non-aromatic joined by a single bond
                "non_heavy_limit": 3, # minimum of 1/4 possible ring substituents
            },
            "Aryl-aliphatic (5-7)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]~[*]~[*]1", # 5-membered aromatic and 7-membered non-aromatic joined by a single bond
                "non_heavy_limit": 3, # minimum of 1/4 possible ring substituents
            },
            "Aryl-aliphatic (6-6)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]~[*]1",
                "non_heavy_limit": 3, # minimum of 1/4 possible ring substituents
            },
            "Aryl-aliphatic (6-5)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]1",
                "non_heavy_limit": 2, # minimum of 2/4 possible ring substituents
            },
            "Aryl-aliphatic (5-6)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]~[*]1",
                "non_heavy_limit": 2, # minimum of 2/4 possible ring substituents
            },
            "Aryl-aliphatic (5-5)": {
                "smarts": "[*]1:[*]:[*]:[*]:[*]:1-[!a]1~[*]~[*]~[*]~[*]1",
                "non_heavy_limit": 2, # minimum of 2/4 possible ring substituents
            },

            # Joined nonaryl-nonaryl ring systems
            "Bialiphatic (7-7)": {
                "smarts": "[!a]1~[!a]~[!a]~[!a]~[!a]~[!a]~[!a]1-[!a]2~[!a]~[!a]~[!a]~[!a]~[!a]~[!a]2",
                "non_heavy_limit": 4, # as for aryl-nonaryls
            },
            "Bialiphatic (7-6)": {
                "smarts": "[!a]1~[!a]~[!a]~[!a]~[!a]~[!a]~[!a]1-[!a]2~[!a]~[!a]~[!a]~[!a]~[!a]2",
                "non_heavy_limit": 4, # as for aryl-nonaryls
            },
            "Bialiphatic (7-5)": {
                "smarts": "[!a]1~[!a]~[!a]~[!a]~[!a]~[!a]~[!a]1-[!a]2~[!a]~[!a]~[!a]~[!a]~[!a]2",
                "non_heavy_limit": 3, # as for aryl-nonaryls
            },
            "Bialiphatic (6-6)": {
                "smarts": "[!a]1~[!a]~[!a]~[!a]~[!a]~[!a]1-[!a]2~[!a]~[!a]~[!a]~[!a]~[!a]2",
                "non_heavy_limit": 3, # as for aryl-nonaryls
            },
            "Bialiphatic (6-5)": {
                "smarts": "[!a]1~[!a]~[!a]~[!a]~[!a]~[!a]1-[!a]2~[!a]~[!a]~[!a]~[!a]2",
                "non_heavy_limit": 2, # as for aryl-nonaryls
            },
            "Bialiphatic (5-5)": {
                "smarts": "[!a]1~[!a]~[!a]~[!a]~[!a]1-[!a]2~[!a]~[!a]~[!a]~[!a]2",
                "non_heavy_limit": 2, # as for aryl-nonaryls
            },

            # Aryl-X systems
            # Includes benzamides, reverse benzamides, etc
            "Benzamide (7)": {
                "smarts": "[#7]-[#6](=[#8])-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # minimum of 3/4 ring and C substituents (both C substituents always heavy by defintion)
            },
            "Benzamide (6)": {
                "smarts": "[#7]-[#6](=[#8])-[*]1:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # minimum of 3/4 ring and C substituents (both C substituents always heavy by defintion)
            },
            "Benzamide (5)": {
                "smarts": "[#7]-[#6](=[#8])-[*]1:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # minimum of 3/4 ring and C substituents (both C substituents always heavy by defintion)
            },
            "Thiobenzamide (7)": {
                "smarts": "[#7]-[#6](=[#16])-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # as for benzamides
            },
            "Thiobenzamide (6)": {
                "smarts": "[#7]-[#6](=[#16])-[*]1:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # as for benzamides
            },
            "Thiobenzamide (5)": {
                "smarts": "[#7]-[#6](=[#16])-[*]1:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # as for benzamides
            },
            "Reverse benzamide (7)": {
                "smarts": "[#8]=[#6]-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 2, # minimum of 2/4 ring and N substituents (one N substituent always heavy by definition)
            },
            "Reverse benzamide (6)": {
                "smarts": "[#8]=[#6]-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # minimum of 3/4 ring and N substituents (one N substituent always heavy by definition)
            },
            "Reverse benzamide (5)": {
                "smarts": "[#8]=[#6]-[#7]-[*]1:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # minimum of 3/4 ring and N substituents (one N substituent always heavy by definition)
            },
            "Reverse thiobenzamide (7)": {
                "smarts": "[#16]=[#6]-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 2, # minimum of 2/4 ring and N substituents (one N substituent always heavy by definition)
            },
            "Reverse thiobenzamide (6)": {
                "smarts": "[#16]=[#6]-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # as for reverse benzamides
            },
            "Reverse thiobenzamide (5)": {
                "smarts": "[#16]=[#6]-[#7]-[*]1:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # as for reverse benzamides
            },
            "Sulfinamide (7)": {
                "smarts": "[#16](=[#8])-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 2, # as for reverse benzamides
            },
            "Sulfinamide (6)": {
                "smarts": "[#16](=[#8])-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # as for reverse benzamides
            },
            "Sulfinamide (5)": {
                "smarts": "[#16](=[#8])-[#7]-[*]1:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # as for reverse benzamides
            },
            "Sulfonamide (7)": {
                "smarts": "[#16](=[#8])(=[#8])-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 2, # as for reverse benzamides
            },
            "Sulfonamide (6)": {
                "smarts": "[#16](=[#8])(=[#8])-[#7]-[*]1:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # as for reverse benzamides
            },
            "Sulfonamide (5)": {
                "smarts": "[#16](=[#8])(=[#8])-[#7]-[*]1:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 1, # as for reverse benzamides
            },
            "Sulfoxide (7)": {
                "smarts": "[#16](=[#8])-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 0, # minimum of 4/4 ring and S substituents (one S substituent always heavy by definition)
            },
            "Sulfoxide (6)": {
                "smarts": "[#16](=[#8])-[*]1:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 0, # minimum of 4/4 ring and S substituents (one S substituent always heavy by definition)
            },
            "Sulfoxide (5)": {
                "smarts": "[#16](=[#8])-[*]1:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 0, # minimum of 4/4 ring and S substituents (one S substituent always heavy by definition)
            },
            "Sulfone (7)": {
                "smarts": "[#16](=[#8])(=[#8])-[*]1:[*]:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 0, # minimum of 4/4 ring and S substituents (one S substituent always heavy by definition)
            },
            "Sulfone (6)": {
                "smarts": "[#16](=[#8])(=[#8])-[*]1:[*]:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 0, # minimum of 4/4 ring and S substituents (one S substituent always heavy by definition)
            },
            "Sulfone (5)": {
                "smarts": "[#16](=[#8])(=[#8])-[*]1:[*]:[*]:[*]:[*]:1",
                "non_heavy_limit": 0, # minimum of 4/4 ring and S substituents (one S substituent always heavy by definition)
            },

            # Amides
            "Amide": {
                "smarts": "[#7]-[#6](=[#8])",
                "non_heavy_limit": 2, # minimum of 2/4 C and N substituents (one C substituent always heavy by definition)
            },
            "Thioamide": {
                "smarts": "[#7]-[#6](=[#16])",
                "non_heavy_limit": 2, # as for amides
            },
        }

        ### 2. Identify and analyze potential atropisomeric bonds

        # Add explicit hydrogens to the molecule
        protonated_mol = Chem.AddHs(self.mol)

        # Iterate over each SMARTS pattern
        for pattern_name in PATTERNS.keys():

            # Define the pattern and non-heavy atom limit
            smarts = PATTERNS[pattern_name]["smarts"]
            non_heavy_limit = PATTERNS[pattern_name]["non_heavy_limit"]

            # Convert the SMARTS string to a mol object
            smarts_mol = Chem.MolFromSmarts(smarts)

            # Find all substructures in the parent molecule that match the SMARTS pattern. This returns
            # a tuple of tuples, where each tuple contains a group of atom indices (in the parent molecule)
            # that map onto the atoms of the SMARTS pattern in a successful substructure match.
            matches = protonated_mol.GetSubstructMatches(smarts_mol)
            
            # Loop over the matches found in the current SMARTS pattern and apply the appropriate cheminformatics
            # function to analyze the substituent pattern and its local steric environment to determine if appropriate
            # for atropisomerism. The match will be processed differently depending on the type of matched pattern.
            for match_indexes in matches:
                print(f"{pattern_name} match identified:")

                # Analyze biaryl, aryl-aliphatic and bialiphatic matches
                if "Biaryl" in pattern_name or "Aryl-aliphatic" in pattern_name or "Bialiphatic" in pattern_name:
                    result, axis_1, axis_2 = self.analyze_biring(protonated_mol, match_indexes, non_heavy_limit)

                # Analyze diaryl ethers, thioethers and amines matches (also captures diaryl sulfoxides and sulfones)
                elif "Diaryl" in pattern_name:
                    result, axis_1, axis_2 = self.analyze_diaryl(protonated_mol, match_indexes, non_heavy_limit)
                
                # If enabled, analyze (thio)benzamides and reversed (thio)benzamides as coupled torsions
                elif coupled_benzamide and ("Benzamide" in pattern_name or "Thiobenzamide" in pattern_name or "Reverse benzamide" in pattern_name or "Reverse thiobenzamide" in pattern_name):
                    # The non-heavy limit is determined for single-bond benzamides, so we set it to 5 to allow all coupled benzamides
                    result, axis_1, axis_2 = self.analyze_coupled_aryl_X(protonated_mol, match_indexes, non_heavy_limit=5)

                # Analyze (thio)benzamide, reversed (thio)benzamide, aryl sulfoxide and sulfonamide matches
                elif "Benzamide" in pattern_name or "Thiobenzamide" in pattern_name or "Reverse benzamide" in pattern_name or "Reverse thiobenzamide" in pattern_name or "Sulfinamide" in pattern_name or "Sulfonamide" in pattern_name or "Sulfoxide" in pattern_name or "Sulfone" in pattern_name:
                    result, axis_1, axis_2 = self.analyze_aryl_X(protonated_mol, match_indexes, non_heavy_limit)

                # Analyze amides and thioamide matches
                elif pattern_name in ["Amide", "Thioamide"]:
                    result, axis_1, axis_2 = self.analyze_amide(protonated_mol, match_indexes, non_heavy_limit)

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

                    # Record the GICs and pattern name
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
                    for atom in protonated_mol.GetAtoms():
                        for at in list(set(x for sub in gic_1 for x in sub)):
                            if atom.GetIdx() + 1 == at:
                                atom.SetProp("molAtomMapNumber", str(at))

        # Remove redundant GIC scans for sulfones and sulfoxides
        self.all_gic_1 = self.remove_redundant_gics(protonated_mol, self.all_gic_1)
        self.all_gic_2 = self.remove_redundant_gics(protonated_mol, self.all_gic_2)

        # Match the keys in substructures to the remaining GICs
        self.substructures = {k: v for k, v in self.substructures.items() if k in self.all_gic_1 or k in self.all_gic_2}

        # Report if no atropisomeric bonds are identified
        if not self.all_gic_1 and not self.all_gic_2:
            print("\nNo atropisomeric bonds were identified in the molecule. If you believe this is an error, please contact an expert.")
            display(Draw.MolToImage(self.mol))

        else:
            # Summarise the rotatable bonds, dihedrals and substructures identified
            print("\nSummary of dihedrals for GIC scans:")
            print(" - scan 1:", self.all_gic_1)
            print(" - scan 2:", self.all_gic_2)
            print("\nSummary of identified substructures:")
            print(f" - {self.substructures}")
            
            # Show the molecules with highlighted atoms
            print("\nRotatable bond:")
            display(Draw.MolToImage(self.mol))
            print("GIC atoms:")
            display(Draw.MolToImage(protonated_mol))

        # Return rotatable bond indexes
        return self.substructures.keys()

    # Scan analysis functions

    def calc_dihedral(self, u1, u2, u3, u4):
        """
        Calculate the signed dihedral angle (in radians) defined by four 3D points.

        Args:
            u1, u2, u3, u4 (np.ndarray): Cartesian coordinates of the four atoms (shape: (3,)).

        Returns:
            rad (float): The signed dihedral angle in radians.
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
    
    def get_diheds(self, dihedral_indexes, xyz_coords):
        """
        Calculate the dihedral angle given the indices of the atoms and their coordinates.

        Parameters:
            dihedral_indexes (list of int): A list containing the indices of the four atoms that define the dihedral angle.
            xyz_coords (list of str): A list of strings where each string contains the x, y, z coordinates of an atom.

        Returns:
            dihedral (float): The dihedral angle in degrees.
        """

        # Get the coordinates of each atom in the dihedral list
        a_temp = xyz_coords[dihedral_indexes[0] - 1].split()
        a_temp = [float(i) for i in a_temp]
        b_temp = xyz_coords[dihedral_indexes[1] - 1].split()
        b_temp = [float(i) for i in b_temp]
        c_temp = xyz_coords[dihedral_indexes[2] - 1].split()
        c_temp = [float(i) for i in c_temp]
        d_temp = xyz_coords[dihedral_indexes[3] - 1].split()
        d_temp = [float(i) for i in d_temp]

        # Convert to arrays
        a = np.array(a_temp)
        b = np.array(b_temp)
        c = np.array(c_temp)
        d = np.array(d_temp)

        # Calculate the dihedral
        dihedral = self.calc_dihedral(a, b, c, d) * 180 / np.pi

        return dihedral

    def normalize_angle(self, angle):
        """
        Normalize an angle to be within the range of 0 to 180°.

        Parameters:
            angle (float): The angle in degrees to be normalized.

        Returns:
            float: The normalized angle, which will be less than 180°.
        """
        return angle if angle < 180 else 360 - angle

    def scan_analysis(self, scan_dict):
        """
        Analyzes a set of Gaussian scan jobs to determine the maximum energy point and neighboring points 
        for each. Only the highest energy maxima along the scan is recorded to exclude unwanted rotations
        of neighbouring groups, and checks are made to ensure that the there has been a significant deviation
        (>45°) in the dihedral angle from the starting structure. From the maximum point of the scans, the
        neighboring geometries (one scan step forwards and one scan step backwards) are also identified.
        Finally, the lowest energy geometry in each category (max, minus1, plus1) over all scans are
        identified. These correspond to candidate rotational TS structures.

        Parameters:
            scan_dict (dict): A dictionary mapping scan log file paths to their corresponding GIC dihedral indices.

        Returns:
            max_xyz: Maximum-energy geometry of the lowest energy scan.
            minus1_xyz: Geometry one step before the maximum-energy geometry of the lowest energy scan.
            plus1_xyz: Geometry one step after the maximum-energy geometry of the lowest energy scan.
        """

        # Initialize variables to store scan results
        max_xyzs = {}
        minus1_xyzs = {}
        plus1_xyzs = {}
        max_energies = {}

        # Loop over the scan dictionary
        for scan_log, gics in scan_dict.items():

            # Initialize variables to store scan data
            coords_scan = []
            energy_scan = []
            xyz_lines = (-1, -1)
            xyz_coords = []
            coords_init = []
            energy = None

            # Check if the scan output exists
            if scan_log.exists():

                # Extract energies and geometries from the scan log file
                with open(scan_log) as f:
                    for i, line in enumerate(f):

                        # Find the lines containing each geometry block. There are several optimizations per scan step.
                        if "Input orientation:" in line:
                            xyz_lines = (i + 5, i + 5 + self.num_atoms)

                            # Initialize an empty set of coordinates when there is a new geometry block
                            xyz_coords = []
                            energy = None

                        # Extract the coordinates for the current geometry block
                        if i >= xyz_lines[0] and i < xyz_lines[1]:
                            xyz_coords += [" ".join(line.split()[-3:])]

                            # Record the initial coordinates (once only)
                            if not coords_init:
                                coords_init = xyz_coords

                        # Extract the energy for the current geometry block
                        if "SCF Done:" in line:
                            energy = float(line.split()[4])

                        # Only record the energy and geometry for the final optimized geometry of each scan step
                        if "Optimization completed." in line:
                            coords_scan += [xyz_coords]
                            energy_scan += [energy]

                            # Reset for new optimisation
                            xyz_lines = (-1, -1)
                            xyz_coords = []
                            energy = None

                # Define steps for plotting
                steps = list(range(len(energy_scan)))

                # Find index of maximum energy in energy_scan
                max_index = np.argmax(energy_scan)

                # Initialize the matplotlib figure
                fig, ax = plt.subplots(figsize=(6, 4))

                # Plot data
                ax.plot(steps, energy_scan, marker='o', color='tab:blue', markersize=7)

                # Highlight max points
                for i in [max_index-1, max_index, max_index+1]:
                    ax.plot(steps[i], energy_scan[i], marker='o', color='tab:red', markersize=8, label=f'Point {i}')

                # Sett labels and title
                ax.set_title(scan_log)
                ax.set_xlabel("Scan Step", fontsize=10)
                ax.set_ylabel("Energy / Hartrees", fontsize=10)

                # Show the plot
                plt.show()

                # Check that there are a sufficient number of energy data points (completed steps) in the scan.
                # If there are, identify the maximum point of the scan by searching for a local maximum within
                # the energy_scan array. If there are not, skip parsing the results for this scan.
                if len(energy_scan) > 6:

                    # Get the indexes of the two dihedral constraints
                    dihed_indexes1, dihed_indexes2 = gics

                    # Get the dihedrals of the initial geometry
                    dihed1_init = self.get_diheds(dihed_indexes1, coords_init)
                    dihed2_init = self.get_diheds(dihed_indexes2, coords_init)

                    # This loop iterates over the energies recorded in energy_scan, excluding the first two and last three
                    # points to properly compare with neighboring conditions. It checks for local maxima by verifying that 
                    # the current point (energy_scan[i]) is greater than the energy value following it (energy_scan[i + 1]) 
                    # while being preceded by an ascending order (i.e., the current point is greater than energy_scan[i-1]).
                    for i in range(2, len(energy_scan) - 3):
                        if all(
                            [
                                float(energy_scan[i + 1])
                                < float(energy_scan[i]),
                                float(energy_scan[i + 2])
                                < float(energy_scan[i + 1]),
                                float(energy_scan[i + 3])
                                < float(energy_scan[i + 2]),
                                float(energy_scan[i - 1])
                                < float(energy_scan[i]),
                            ]
                        ):

                            # Get the dihedrals of the potential maximum point geometry
                            dihed1 = self.get_diheds(dihed_indexes1, coords_scan[i])
                            dihed2 = self.get_diheds(dihed_indexes2, coords_scan[i])

                            # Get the difference in dihedral angles compared to the initial geometry
                            dihed1_change = self.normalize_angle(abs(dihed1_init - dihed1))
                            dihed2_change = self.normalize_angle(abs(dihed2_init - dihed2))

                            # After identifying potential maximum candidates through energy comparisons, we check whether
                            # the dihedral angles (dihed1, dihed2) have deviated significantly (by at least 45 degrees) 
                            # from their initial values (dihed1_init, dihed2_init). We only accept the scan if one has.
                            sufficient_deviation = dihed1_change >= 45.0 or dihed2_change >= 45.0

                            # If the dihedral angles have deviated sufficiently, and no point has been previously recorded for
                            # this scan, record the XYZ and energy values for that point. If a point has already been recorded,
                            # overwrite it if the new maximum point has a higher energy than the previously recorded maximum.
                            # We do this so we capture the highest energy maximum along the scan. Otherwise, we might record a
                            # small maximum that doesn't correspond to the intended atropisomeric rotation.
                            if sufficient_deviation and (
                                scan_log not in max_energies
                                or float(energy_scan[i]) > max_energies[scan_log]
                                ):
        
                                # Record the XYZ coordinates of the maximum point and its neighbors
                                max_xyzs[scan_log] = coords_scan[i]
                                minus1_xyzs[scan_log] = coords_scan[i - 1]
                                plus1_xyzs[scan_log] = coords_scan[i + 1]

                                # Record the energy of the maximum point
                                max_energies[scan_log] = float(energy_scan[i])

        # Find which scan has the lowest energy maxima
        lowest_key = min(max_energies, key=max_energies.get)
        print(f"Lowest energy maximum found in scan: {lowest_key}")

        # Return the coordiantes of the points for that scan
        max_xyz = max_xyzs[lowest_key]
        minus1_xyz = minus1_xyzs[lowest_key]
        plus1_xyz = plus1_xyzs[lowest_key]

        return max_xyz, minus1_xyz, plus1_xyz

    # TS analysis functions

    def parse_with_cclib(self, logfile):
        """
        Parse the log file using cclib and extract relevant data.

        Args:
            logfile (str): Path to the log file.

        Returns:
            free_energy (float): The free energy in Hartrees.
            frequencies (list): List of vibrational frequencies.
            formatted_coords (list): List of formatted atomic coordinates.
        """

        # Check logfile exists
        if not Path(logfile).exists():
            raise FileNotFoundError(f"Log file not found: {logfile}")

        # Parse log file with cclib
        data = cclib.io.ccread(logfile)

        # Extract the free energy (Hartrees)
        try:
            free_energy = data.freeenergy
        except:
            free_energy = None
        
        # Extract the vibrational frequencies
        try:
            frequencies = data.vibfreqs
        except:
            frequencies = None

        # Extract atomic coordinates
        try:
            all_coords = data.atomcoords
    
            # Convert each set of coordinates into a space-separated string
            # to match the xyz format convention throughout the workflow
            formatted_coords = []
            for coords in all_coords:
                coordinates = [' '.join(map(str, row)) for row in coords]
                formatted_coords.append(coordinates)
        except:
            formatted_coords = None

        return free_energy, frequencies, formatted_coords

    def check_is_rotational_ts(self, logfile, xyz_coords, ts_index):
        """
        Check if a TS optimization output corresponds to a TS at the rotatable bond defined by the GICs dihedrals:
         - Compare relative magnitude of displacement vectors on the constrained atoms compared to the rest of the molecule.
         - Check that there isn't a significant change (>45°) in the dihedral angles of the rotatable bond.

        Args:
            logfile (str): Path to the calculation logfile to be parsed.
            xyz_coords (list of str): List of XYZ coordinates at each optimization step.
            ts_index (int): The index of the substructure in the substructures list.

        Returns:
            bool: True if the logfile corresponds to a TS, False otherwise.
        """

        # Initialize counts for total magnitude of displacement vectors
        weights_sum = 0.0  # Summed over all atoms
        weights_sum_gic = 0.0  # Summed over constrained atoms

        # Get the indexes of the dihedral constraints
        dihed_indexes1a, dihed_indexes2a = self.all_gic_1[ts_index]
        dihed_indexes1b, dihed_indexes2b = self.all_gic_1[ts_index]

        # Define list of all constrained atoms
        constrained_atoms = sorted(set(dihed_indexes1a + dihed_indexes2a))

        # Read the log file
        with open(logfile) as f_out:
            for line in f_out:

                # Look for the marker indicating the start of atom displacement vectors
                if "Atom  AN" in line:

                    # Read each line corresponding to an atom
                    for j in range(self.num_atoms):
                        line1 = next(f_out)
                        parts = line1.split()

                        # Skip lines that don't have enough columns
                        if len(parts) < 5:
                            continue

                        # Extract x, y, z displacement components (columns 2, 3, 4) and square them
                        x2 = float(parts[2])**2
                        y2 = float(parts[3])**2
                        z2 = float(parts[4])**2

                        # Add the squared components to the running total for all atoms
                        weights_sum += x2 + y2 + z2

                        # If the atom is in the constraints index, also add to the constrained atom sum
                        if (j + 1) in constrained_atoms:
                            weights_sum_gic += x2 + y2 + z2
    
                    break # All atom vectors processed; no need to continue reading

        # Calculate the proportion of total displacement of the constrained atoms
        displacement_proportion = (weights_sum_gic / weights_sum)*100
        print(f"Displacement %: {displacement_proportion:.2f}")

        # If less than 2% of total displacement is on the constrained atoms, we reject the TS.
        # In this case, the TS likely does not correspond to the desired rotation.
        if displacement_proportion < 2.0:
            return False

        # Check how the dihedrals changed during optimization
        dihed1_init = self.get_diheds(dihed_indexes1a, xyz_coords[0])
        dihed2_init = self.get_diheds(dihed_indexes2a, xyz_coords[0])
        dihed1_end = self.get_diheds(dihed_indexes1a, xyz_coords[-1])
        dihed2_end = self.get_diheds(dihed_indexes2a, xyz_coords[-1])
        dihed1a_change = self.normalize_angle(abs(dihed1_init - dihed1_end))
        dihed2a_change = self.normalize_angle(abs(dihed2_init - dihed2_end))

        # Repeat for the other set of dihedrals
        dihed1_init = self.get_diheds(dihed_indexes1b, xyz_coords[0])
        dihed2_init = self.get_diheds(dihed_indexes2b, xyz_coords[0])
        dihed1_end = self.get_diheds(dihed_indexes1b, xyz_coords[-1])
        dihed2_end = self.get_diheds(dihed_indexes2b, xyz_coords[-1])
        dihed1b_change = self.normalize_angle(abs(dihed1_init - dihed1_end))
        dihed2b_change = self.normalize_angle(abs(dihed2_init - dihed2_end))

        # Report the dihedral changes
        print(f"Dihedral changes: {dihed1a_change:.2f}°, {dihed2a_change:.2f}°, {dihed1b_change:.2f}°, {dihed2b_change:.2f}°")
    
        # If either any of the dihedral angles have deviated significantly (>45°) during optimization,
        # we reject the TS. In this case, the TS has likely relaxed towards the ground state geometry.
        if dihed1a_change >= 45.0 or dihed2a_change >= 45.0 or dihed1b_change >= 45.0 or dihed2b_change >= 45.0:
            return False

        # If both pass, we likely have the correct rotational TS
        return True

    def ts_analysis(self, ts_logs, ts_index):
        """
        Analyzes a set of Gaussian TS optimization jobs to collect energies and geometries and validate TS candidates.

        Args:
            ts_logs (list): A list of paths to TS log files.
            ts_index (int): The index of the substructure in the substructures list.
        
        Returns:
            ts_xyzs (dict): A dictionary mapping valid TS log files to their corresponding optimized geometries.
            ts_energies (dict): A dictionary mapping valid TS log files to their corresponding free energies.
        """

        # Initialize a dictionary to track processing status for each TS
        checks = {ts_log: False for ts_log in ts_logs}

        # Initialize dictionaries to hold geometries and energies
        ts_xyzs, ts_energies = {}, {}

        # Loop until all TS files have been processed
        while False in checks.values():
            for log in ts_logs:

                # Process only unprocessed TS files
                if checks[log] is False:
                    print(f"Log file: {log}")
    
                    # If log file exists, extract energies, frequencies and geometries
                    if log.exists():
                        energy, freqs, xyz_coords = self.parse_with_cclib(log)

                        # Check for negative frequency (indicating a TS candidate)
                        if freqs[0] < 0:

                            # Check if the log file corresponds to a TS based on the main source
                            # of displacement and on the dihedral angles of the rotatable bond
                            is_ts = self.check_is_rotational_ts(log, xyz_coords, ts_index)

                            # Store energies and coordinates if TS is valid
                            if is_ts:
                                print("TS validation passed.\n")
                                ts_xyzs[log] = xyz_coords[-1] # Optimized geometry only
                                ts_energies[log] = float(energy)
        
                        # Mark directories related to this file index as processed
                        checks[log] = True

        return ts_xyzs, ts_energies

    # Results functions

    def extract_free_energy(self, gv_file):
        """
        Extracts the free energy value from a given goodvibes data file.

        Args:
            gv_file (str): The path to the goodvibes data file.

        Returns:
            free_energy (float): The free energy value extracted from the file.
        """
        with open(gv_file,'r') as gv_out:
            for line in gv_out:
                if "Structure" in line:
                    headers = gv_out.readline()
                    energies = gv_out.readline()
                    free_energy = float(energies.split()[-1])
                    break
        return free_energy

    def correct_barrier(self, barrier, ts_index, overall=False):
            """
            Corrects the energy barrier based on the substructure type and regression parameters from the literature
            benchmark. If no substructure-specific regression exists, the parameters for the full benchmark are used.

            Parameters:
                barrier (float): The original energy barrier value to be corrected.
                ts_index (int): The index of the substructure in the substructures list.
                overall (bool): If True, applies the overall correction regardless of substructure type.

            Returns:
                float: The corrected energy barrier value based on the substructure type.
            """

            # Apply overall correction if requested
            if overall:
                return 0.980*barrier + 1.458

            # Else, get the substructure of the index
            substructure = self.substructures[ts_index]

            # Apply substructure-specific correction
            if "Biaryl" in substructure:
                return 0.997*barrier + 0.658
            elif "Diaryl" in substructure:
                return 1.238*barrier - 3.377
            elif "Benzamide" in substructure or "Thiobenzamide" in substructure:
                return 1.041*barrier + 0.453
            elif "Reverse benzamide" in substructure or "Reverse thiobenzamide" in substructure:
                return 0.918*barrier + 2.776
            elif "Sulfoxide" in substructure or "Sulfone" in substructure:
                return 0.852*barrier + 4.286
            elif substructure in ["Amide", "Thioamide"]:
                return 0.867*barrier + 3.107
            elif "Aryl-aliphatic" in substructure or "Bialiphatic" in substructure:
                return 1.230*barrier - 3.080

            # Apply overall correction if no specific one exists
            else:
                return 0.980*barrier + 1.458

    def visualize_barrier(self, bond_index, barrier):
        """
        Visualize the molecule with the rotatable bond and barrier annotated.

        Parameters:
            bond_index (int): The index of the rotatable bond.
            barrier (float): The computed barrier of the bond.
        """
        # Setup drawing options
        width, height = (400,400)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().annotationFontScale = 1.0

        # Prepare highlight and annotation mappings
        highlight_atoms = []
        highlight_bonds = []

        # Get rotatable bond indexes
        idx1, idx2 = self.all_gic_1[bond_index][0][1:3]

        # 0-index for RDKit
        idx1 -= 1
        idx2 -= 1

        # Get the rotatable bond object
        bond = self.mol.GetBondBetweenAtoms(idx1, idx2)

        # Annotate the barrier
        bond.SetProp("bondNote", str(barrier))

        # Store the bond information
        highlight_atoms.extend([idx1, idx2])
        highlight_bonds.append(bond.GetIdx())

        # Highlight atropisomeric bonds
        drawer.DrawMolecule(
            self.mol,
            highlightAtoms=highlight_atoms,
            highlightBonds=highlight_bonds,
        )

        # Draw the molecule
        drawer.FinishDrawing()
        image = drawer.GetDrawingText()

        # Display molecule
        display(SVG(image))

    def assign_class(self, barrier):
        """
        Assigns the LaPlante atropisomer class based on the value of the barrier.

        Parameters:
            barrier (float): The barrier value to classify.

        Returns:
            int: The assigned class.
        """
        return 3 if barrier > 30 else 1 if barrier < 20 else 2
        
    # Plotting functions

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
