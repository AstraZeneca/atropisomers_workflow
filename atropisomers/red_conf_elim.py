import argparse
import numpy as np
from spyrmsd import rmsd
from pathlib import Path

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

def read_from_xyz_lines(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse atom labels and Cartesian coordinates from XYZ-style text lines.

    Args:
        lines (list[str]): Sequence of XYZ atom lines, e.g. "C 0.0 1.2 -0.4".

    Returns:
        nums (np.ndarray): Array of atomic symbols (e.g., ['C', 'H', 'O']).
        coords (np.ndarray): Array of Cartesian coordinates with shape (N, 3), where N is the number of atoms.
    """

    # Initialize lists to hold atomic symbols and coordinates
    nums = []
    coords = []

    # Iterate through each line
    for line in lines:
        
        # Split the line into components, filtering out any empty strings
        s = line.split(' ')
        s = [x for x in s if x != '']

        # Append the atomic symbol and coordinates to their respective lists
        nums.append(s[0])
        coords.append([float(i) for i in s[1:]])

    # Convert lists to numpy arrays
    nums = np.array(nums)
    coords = np.array(coords)

    return nums, coords

def calc_rmsd(path1, path2, method):
    """
    Compute the RMSD-like distance between two molecular structures read from XYZ files.

    Parameters:
        path1 (str): Path to the first XYZ file.
        path2 (str): Path to the second XYZ file.
        method (str): Method for RMSD calculation. Supported options are:
            - 'RMSD': Standard RMSD calculation.
            - 'HRMSD': Hungarian RMSD calculation.

    Returns:
        float: The computed RMSD value between the two structures.

    """

    # Read atomic symbols and coordinates from the two XYZ files
    nums1, coords1 = read_from_xyz_lines(path1)
    nums2, coords2 = read_from_xyz_lines(path2)

    # Calculate RMSD based on the specified method
    if method == 'RMSD':
        RMSD = rmsd.rmsd(coords1,coords2,nums1,nums2,center=True,minimize=True)
    elif method == 'HRMSD':
        RMSD = rmsd.hrmsd(coords1,coords2,nums1,nums2)

    return RMSD

def red_conf_elim(ensemble, rmsd_threshold=0.5, energy_threshold=5, verbose=True):
    """
    Perform redundant conformer elimination on a CREST conformer ensemble, removing conformers that are
    geometrically or energetically redundant based on the specified energy and/or RMSD thresholds, then
    write a reduced ensemble. The first conformer is assumed to be the reference (lowest-energy) structure.

    Parameters:
        ensemble (str): Path to the input XYZ ensemble file.
        rmsd_threshold (float): RMSD threshold for retaining conformers.
        energy_threshold (float): Energy threshold in kcal/mol for retaining conformers relative to the first conformer.
        verbose (bool): Whether to print messages.

    """

    # Read the .xyz ensemble
    with open(ensemble, "r") as f:
        lines = f.readlines()

    # Get number of atoms and conformers
    n_atoms = int(lines[0])
    n_conformers = int(len(lines)/(n_atoms+2))

    if verbose:
        print(f"Performing redundant conformer elimination for {ensemble} ({n_conformers} conformers, {n_atoms} atoms)\n")

    # Initialize lists to hold conformers, energies, and relative energies
    conformers, energies, rel_energies = [], [], []

    # Iterate through each conformer in the ensemble
    for i in range(n_conformers):

        # Extract the conformer lines and energy from the ensemble
        conformer = lines[2+(i*(n_atoms+2)):2+n_atoms+(i*(n_atoms+2))]
        conformers.append(conformer)
        energy = lines[1+(i*(n_atoms+2))]
        energies.append(energy)

        # Also calculate relative energies in kcal/mol if an energy threshold is
        # provided, assuming the first conformer is the lowest energy conformer
        if energy_threshold:
            rel_energy = (float(energy) - float(energies[0]))*627.5
            rel_energies.append(rel_energy)

    num_original = len(conformers)

    # Filter conformers based on energy threshold if provided
    if energy_threshold:

        # Find conformers above the energy threshold
        idxs = []
        for i, energy in enumerate(energies):
            rel_energy = rel_energies[i]
            if rel_energy > energy_threshold:
                idxs.append(i)
        
        # Remove conformers above the energy threshold
        conformers = [j for i, j in enumerate(conformers) if i not in idxs]
        energies = [j for i, j in enumerate(energies) if i not in idxs]
        if verbose:
            print(f'{len(idxs)} conformers removed by energy threshold')
    
    else:
        if verbose:
            print('Energy filtering switched off')

    # Filter conformers based on RMSD threshold if provided
    if rmsd_threshold:

        # Initialize lists to hold retained conformers and their energies
        retained_confs = []
        retained_energies = []

        # Iterate through each conformer
        for i in range(len(conformers)):

            # Keep the first conformer by default
            if i == 0:
                retained_confs.append(conformers[i])
                retained_energies.append(energies[i])

            # Compare other conformers to the retained ones based on RMSD
            else:
                conf = conformers[i]
                RMSDs = []
                for ret_conf in retained_confs:

                    # Calculate Hungarian RMSD
                    RMSD = calc_rmsd(conf, ret_conf, 'HRMSD')
                    RMSDs.append(RMSD)
                    
                    # Alternatively, we can calculate with CREST like:
                    # crest --rmsd struc1.xyz struc2.xyz | tail -1cond

                # Check if any retained structures are under threshold
                check = [i for i in RMSDs if i < rmsd_threshold]

                # If not, retain the conformer and its energy
                if len(check) == 0:
                    retained_confs.append(conformers[i])
                    retained_energies.append(energies[i])

        if verbose:
            print(f'{len(conformers) - len(retained_confs)} conformers removed by RMSD threshold')

    else:
        if verbose:
            print('RMSD filtering switched off')
        retained_confs = conformers
        retained_energies = energies

    # Report overall statistics
    if verbose:
        print(f'\nConformers Omitted: {num_original - len(retained_confs)}')
        print(f'Conformers Retained: {len(retained_confs)}')
    
    # Define output file name
    reduced_ensemble = f"{ensemble.split('.')[0]}_reduced.xyz"

    # Write the retained conformers and their energies to a new .xyz file
    with open(reduced_ensemble, 'w') as f:
        for i in range(len(retained_confs)):
            conformer = retained_confs[i]
            energy = retained_energies[i]
            f.write(f'  {n_atoms}\n')
            f.write(f'{energy}')
            for line in conformer:
                f.write(f'{line}')

    if verbose:
        print(f"\nWriting new file to {reduced_ensemble}")


def main(ensemble, num_confs, rmsd_threshold=0.0, rmsd_increment=0.05, energy_threshold=5.0):
    """
    Reduce a CREST conformer ensemble iteratively by increasing the RMSD cutoff in small increments
    and using a constant energy threshold to remove conformers that are geometrically or energetically
    redundant, stopping once the reduced ensemble has fewer than the specified number of conformers.
    The first conformer is assumed to be the reference (lowest-energy) structure. The reduced ensemble
    is written to a new .xyz file.

    A small increment is recommended so we don't overshoot and remove many more conformers that requested.
    Note, however, that it is still possible to remove more conformers than requested because of the way
    the conformers might cluster.

    Parameters:
        ensemble (str): Path to the input XYZ ensemble file.
        num_confs (int): Target number of conformers to reduce below.
        rmsd_threshold (float): Starting RMSD threshold for retaining conformers.
        rmsd_increment (float): Increment to increase RMSD threshold by each iteration.
        energy_threshold (float): Energy threshold in kcal/mol for retaining conformers relative to the first conformer.

    """

    # Define output file name
    reduced_ensemble = f"{ensemble.split('.')[0]}_reduced.xyz"

    # Raise an error if the increment is set to 0
    if rmsd_increment <= 0:
        raise ValueError("RMSD increment must be greater than 0.")

    # Get the number of conformers in the original ensemble
    _, __, n_conformers = read_from_xyz(ensemble)

    # Only reduce the ensemble if the number of conformers is too high
    if n_conformers > num_confs:

        # Using a constant energy threshold, we will continually reduce the number of conformers
        # by scaling up the RMSD threshold by the specified increment each time, until we reduce
        # the ensemble to less than or equal to the desired number.
        while n_conformers > num_confs:
            rmsd_threshold += rmsd_increment
            red_conf_elim(ensemble, rmsd_threshold, energy_threshold, verbose=False)
            _, __, n_conformers = read_from_xyz(reduced_ensemble)

        # Run it a final time but printing results
        red_conf_elim(ensemble, rmsd_threshold, energy_threshold, verbose=True)

    else:
        print("Number of conformers acceptable - no elimination performed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
        Redundant conformer elimination loop for .XYZ ensembles.
        This codes loops over the standard redundant conformer elimination tool
        with increasingly larger thresholds until the number of conformers in
        the reduced ensemble is less than the specified amount.
        """)
    parser.add_argument('xyz', help="An .xyz ensemble, e.g., crest_conformers.xyz")
    parser.add_argument('num_confs', nargs='?', type=float, default=40, help="The target number of conformers to reduce below.")
    parser.add_argument('rmsd_threshold', nargs='?', type=float, default=0.0, help="Starting RMSD retain threshold.")
    parser.add_argument('rmsd_increment', nargs='?', type=float, default=0.05, help="Increment to increase RMSD threshold by each iteration.")
    parser.add_argument('energy_threshold', nargs='?', type=float, default=5, help="Energy retain threshold in kcal/mol (use 0 to switch off).")
    args = parser.parse_args()
    main(args.xyz, args.num_confs, args.rmsd_threshold, args.rmsd_increment, args.energy_threshold)
