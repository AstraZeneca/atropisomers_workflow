import os
import re
import cclib.io
import subprocess
import numpy as np
from pathlib import Path
from rdkit.Chem import AllChem as Chem

# Functions for equivalence checks

def convert_xyz_to_sdf(xyz, sdf):
    """
    Convert an XYZ file to an SDF file using Open Babel.

    Parameters:
        xyz (str): Path to the input XYZ file.
        sdf (str): Path to the output SDF file.

    Returns:
        None
    """

    # Convert XYZ to SDF with openbabel subprocess
    cmd = f'obabel -ixyz {xyz} -osdf -xf > {sdf}'
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate()

def _connectivity_only(mol):
    """
    Return an RDKit Mol object with all bond orders set to single bonds.

    Parameters:
        mol (rdkit.Chem.Mol): The original molecule.

    Returns:
        rdkit.Chem.Mol: A new molecule with all bond orders set to single.
    """
    mol = Chem.RWMol(mol) # copy to avoid modifying original
    for bond in mol.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
    return mol

def _remove_radicals(mol):
    """
    Return an RDKit Mol object with all radical electrons set to zero.

    Parameters:
        mol (rdkit.Chem.Mol): The original molecule.

    Returns:
        rdkit.Chem.Mol: A new molecule with all radical electrons set to zero.

    """
    mol = Chem.Mol(mol) # copy to avoid modifying original
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            atom.SetNumRadicalElectrons(0)
    return mol

def _remove_charges(mol):
    """
    Return an RDKit Mol object with all formal charges set to zero.

    Parameters:
        mol (rdkit.Chem.Mol): The original molecule.

    Returns:
        rdkit.Chem.Mol: A new molecule with all formal charges set to zero.
    """
    mol = Chem.Mol(mol) # copy to avoid modifying original
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)
    return mol

def compare_connectivity(mol1, mol2):
    """
    Check if two molecular structures have identical connectivity, regardless of bond order, aromaticity, radicals, etc. 

    Parameters:
        mol1 (rdkit.Chem.Mol): The first molecular structure.
        mol2 (rdkit.Chem.Mol): The second molecular structure.
        sanitize (bool): If True, sanitize the molecules when removing hydrogens.
        ignore_hydrogens (bool): If True, ignore hydrogen atoms in the comparison.

    Returns:
        bool: True if the connectivity of the two molecules is identical, False otherwise.
    """

    # Reduce to connectivity-only and remove radicals and charges
    g1, g2 = _connectivity_only(mol1), _connectivity_only(mol2)
    g1, g2 = _remove_radicals(g1), _remove_radicals(g2)
    g1, g2 = _remove_charges(g1), _remove_charges(g2)

    # Check they are the same
    return g1.GetNumAtoms() == g2.GetNumAtoms() and g1.HasSubstructMatch(g2) and g2.HasSubstructMatch(g1)

def compare_equivalence(sdf1, sdf2, fallback_check=True):
    """
    Compares two molecular structures represented in SDF format to determine if they are equivalent.

    RDkit is used as it is purpose-built to detect identity of molecular graphs regardless of atom numbering,
    resonance forms, or file/internal atom ordering. RDKits canonical SMILES is robust for most connectivity,
    resonance, and simple tautomer issues, meaning changes in connectivity due to rearrangement of double
    bonds within aromatics, migration of a cation from one oxygen to another within a carboxylate group,
    or other common examples, are accounted for.

    By including the `isomericSmiles` tag in SMILES generation, the output string encodes not only
    the atomic connectivity, but also the specific configuration of stereocenters (such as R/S chirality)
    and the geometry of double bonds (E/Z or cis/trans). This ensures that structures differing in
    stereochemistry will result in different SMILES strings and thus be recognized as non-equivalent.

    Parameters:
        sdf1 (str or Path): The file path to the first SDF file.
        sdf2 (str or Path): The file path to the second SDF file.
        fallback_check (bool): If True, perform a secondary check on connectivity only if the canonical SMILES comparison fails.

    Returns:
        bool: True if the structures are equivalent, otherwise False.
    """

    # Check that the SDF files exists
    if not Path(sdf1).exists():
        raise FileNotFoundError(f"The SDF file '{sdf1}' does not exist.")
    if not Path(sdf2).exists():
        raise FileNotFoundError(f"The SDF file '{sdf2}' does not exist.")
    
    # Check that the SDF files are not empty
    if Path(sdf1).stat().st_size == 0:
        raise ValueError(f"The SDF file '{sdf1}' is empty. Check that openbabel is installed and that the xyz file is valid.")
    if Path(sdf2).stat().st_size == 0:
        raise ValueError(f"The SDF file '{sdf2}' is empty. Check that openbabel is installed and that the xyz file is valid.")

    # Load the molecules from the SDF files.
    mol1 = Chem.MolFromMolFile(str(sdf1), removeHs=False)
    mol2 = Chem.MolFromMolFile(str(sdf2), removeHs=False)

    # If it fails, try without sanitization. For some anion/cation
    # intermediates, the sanitization fails but the molecule is ok.
    if not mol1 or not mol2:
        mol1 = Chem.MolFromMolFile(str(sdf1), removeHs=False, sanitize=False)
        mol2 = Chem.MolFromMolFile(str(sdf2), removeHs=False, sanitize=False)

    # Check if both molecules are valid
    if not mol1 or not mol2:
        return False

    # Generate canonical SMILES for both molecules
    smi1 = Chem.MolToSmiles(mol1, canonical=True, isomericSmiles=True)
    smi2 = Chem.MolToSmiles(mol2, canonical=True, isomericSmiles=True)

    # Compare the SMILES representations
    if smi1 == smi2:
        return True

    # If failed, check pure connectivity (ignoring bond orders, aromaticity, and radicals).
    # This is required because RDKit sometimes recognises two structures as different when
    # they are identical connectivity-wise. For example, this usually occurs when radicals
    # or charges are formally located on different atoms, causing aromaticity or single-double
    # bonding patterns to change. This is particularly prevalant for cations and anions.
    if fallback_check:
        if compare_connectivity(mol1, mol2):
            return True

    return False

# Functions for extracting data from QM outputs

def get_charge(smiles):
    """
    Calculate the total formal charge of a molecule from its SMILES string.

    Args:
        smiles (str): The SMILES representation of the molecule.

    Returns:
        int: The total formal charge of the molecule.
    """

    # Convert SMILES to RDKit Mol object
    mol = Chem.MolFromSmiles(smiles)

    # Check if the molecule was successfully created
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Calculate the total formal charge by summing the formal charges of all atoms
    total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    return total_charge

def get_xtb_energy(xyz_path):
    """
    Extract the xTB energy value from an XYZ file.

    This function opens an XYZ file, skips the first line (atom count), and parses
    the second line as metadata, returning the second whitespace-separated token as
    a floating-point energy value.

    Args:
        xyz_path (str | Path): Path to the XYZ file.

    Returns:
        float: The parsed xTB energy.
    """

    # Check XYZ file exists
    if not Path(xyz_path).exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")
    
    # Parse the energy from the second line of the XYZ file
    with xyz_path.open() as f:
        next(f)
        return float(next(f).split()[1])

def get_crest_energy(crest_best):
    """
    Extracts the energy of the CREST best output.

    Parameters:
        crest_best (str): Path to the CREST best output file.

    Returns:
        float: The energy value extracted from the CREST best output.
    """

    # Get energy of CREST best
    with open(crest_best) as out:
        out.readline()
        gibbs = float(out.readline().split()[0])

    return gibbs

def get_censo_energy(censo_out, solvent):
    """
    Extracts the energy of CENSO parts from the given CENSO output file.

    Parameters:
        censo_out (str): The path to the CENSO output file.
        solvent (bool): True if a solvent is used, False if gas phase.

    Returns:
        list: A list of energy values for each CENSO part. The list has a length
        of 4, where each element corresponds to the energy of a specific part.
    """

    # Initialize Gibbs list with infinite values for each part
    gibbs = [np.inf] * 4

    # Set regex based on whether solvent is used or not.
    pattern = re.compile(r"<<==part(\d)==")
    if solvent:
        pos = (2, 4, 4, 4)
    else:
        pos = (2, 3, 3, 3)

    # Get energy of CENSO parts
    with open(censo_out, "r") as f:
        for line in f:
            if match := pattern.findall(line):
                part = int(match[0])
                gibbs[part] = float(line.split()[pos[part]])

    return gibbs

def parse_scan_with_cclib(log_path):
    """
    Parse a scan QM log/out (e.g. Gaussian) file using cclib and extract relevant data.

    Args:
        log_path (str): Path to the log file.

    Returns:
        scf_energies (list): List of SCF energies (in Hartrees) for each step.
        formatted_coords (list): List of lists containing formatted atomic coordinates for each step.
        formatted_input_coord (list): List of strings containing the formatted atomic coordinates of the initial input structure.
    """

    # Check logfile exists
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    # Parse log file with cclib
    data = cclib.io.ccread(log_path)

    # Extract the SCF energies
    try:
        scf_energies_ev = data.scanenergies
    except:
        scf_energies_ev = None

    # Convert energies to Hartrees if successfully extracted
    if scf_energies_ev is not None:
        scf_energies = [float(e) / 27.2114079527 for e in scf_energies_ev]
    else:        
        scf_energies = None

    # If extraction failed, attempt to parse energies manually
    if scf_energies is None:
        scf_energies = []
        with open(log_path) as f:
            for line in f:
                if "SCF Done:" in line:
                    energy = float(line.split()[4])
                if "Optimization completed." in line:
                    scf_energies += [energy]

        # Set to none if no energies were found
        if scf_energies == []:
            scf_energies = None

    # Extract scan coordinates
    try:
        all_coords = data.converged_geometries
        # Convert each set of coordinates into a space-separated string
        # to match the xyz format convention throughout the workflow
        formatted_coords = []
        for coords in all_coords:
            coordinates = [' '.join(map(str, row)) for row in coords]
            formatted_coords.append(coordinates)
    except:
        formatted_coords = None

    # Extract the initial input coordinates
    try:
        input_coord = data.atomcoords[0]
        formatted_input_coord = [' '.join(map(str, row)) for row in input_coord]
    except:
        formatted_input_coord = None

    return scf_energies, formatted_coords, formatted_input_coord

def parse_with_cclib(log_path):
    """
    Parse a QM output file (e.g. Gaussian) file using cclib and extract relevant data.

    Args:
        log_path (str | Path): Path to the log file.

    Returns:
        scf_energies (list): List of SCF energies (in Hartrees) for each step.
        free_energy (float): Free energy (in Hartrees) of the system.
        frequencies (list): List of vibrational frequencies (in cm^-1).
        displacements (list): List of vibrational displacements.
        formatted_coords (list): List of lists containing formatted atomic coordinates for each step.
    """

    # Check logfile exists
    if not Path(log_path).exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    # Parse log file with cclib
    data = cclib.io.ccread(log_path)

    # Extract the SCF energies
    try:
        scf_energies_ev = data.scfenergies
    except:
        scf_energies_ev = []

    # Convert to Hartrees if energy was successfully extracted
    if scf_energies_ev is not None:
        scf_energies = [float(e) / 27.2114079527 for e in scf_energies_ev]
    else:        
        scf_energies = None

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

    # Extract the vibrational displacements
    try:
        displacements = data.vibdisps
    except:
        displacements = None

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

    return scf_energies, free_energy, frequencies, displacements, formatted_coords

def read_from_xyz(xyz_path):
    """
    Counts the number of lines, number of conformers, and number of atoms per structure from .XYZ ensemble.

    Parameters:
        xyz_path (str | Path): Path to the .xyz file.

    Returns:
        lines (list): List of all lines in the .xyz file.
        n_atoms (int): Number of atoms per conformer.
        n_conformers (int): Number of conformers in the ensemble.
    """

    # Check if the file exists
    if not Path(xyz_path).exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

    # Read .xyz ensemble
    with open(xyz_path, "r") as f:
        lines = f.readlines()

    # Check if the file is empty
    if not lines:
        raise ValueError(f"XYZ file is empty: {xyz_path}")

    # Get number of atoms and conformers
    n_atoms = int(lines[0])
    n_conformers = int(len(lines)/(n_atoms+2))

    return lines, n_atoms, n_conformers

def get_conformer_xyz(xyz_path, conformer_index=0):
    """
    Extracts XYZ coordinates of the specified conformer (by index) from an .xyz ensemble file.

    Parameters:
        xyz_path (str | Path): Path to the .xyz ensemble file.
        conformer_index (int): Index of desired conformer (0-based; 0 is first conformer). Defaults to 0.

    Returns:
        conformer_lines (list): Lines of the XYZ block for the selected conformer (including atom and coordinates).
        conformer_index (int): Index of the conformer that was successfully retrieved.
        n_conformers (int): Total number of conformers in the ensemble.
        
    """

    # Check if the file exists
    if not Path(xyz_path).exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

    # Get ensemble details and all lines
    lines, n_atoms, n_conformers = read_from_xyz(xyz_path)

    # Check if lines were successfully read
    if lines is None:
        return None, None, None

    # Set conformer index to the max if it exceeds the limit
    if conformer_index < 0 or conformer_index >= n_conformers:
        conformer_index = n_conformers - 1

    # Each conformer's block: atom count line + comment line + lines of atoms
    block_size = n_atoms + 2
    start_line = conformer_index * block_size
    end_line = start_line + block_size

    # Grab only the coordinate lines ("atom X Y Z"), skipping atom count & comment
    conformer_lines = lines[start_line:end_line]
    
    return conformer_lines, conformer_index, n_conformers
