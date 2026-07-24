import shutil
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
from itertools import zip_longest
from IPython.display import display
from ipywidgets.widgets import Output
from rdkit.Chem import AllChem as Chem
from textwrap import wrap

from .slurm import (
    Job,
    ProgressBar,
    xtb_batch,
    crest,
    censo,
    gaussian,
    task,
)

from .utils import (
    convert_xyz_to_sdf,
    compare_equivalence,
    get_conformer_xyz,
    get_xtb_energy,
    get_crest_energy,
    get_censo_energy,
    read_from_xyz,
    get_charge,
)

# Import last to prevent C++ import errors
import pandas as pd

class Workflow:
    """
    Base class for QM workflows that initializes the job settings, molecular settings, program settings, and Slurm settings,
    and provides methods for logging, extracting energies, and running the core workflow steps (embedding, xTB, CREST, ELIM, CENSO).

    Parameters:

        Job settings:
            job_name (str): Slurm job name, used when instantiating QM workflow.
            job_dir (str | Path): Path to the folder where the job files should be stored.
            tmp_dir (str | Path): Path to the temporary (scratch) folder where jobs run.

        Program settings:
            bashrc (str | Path): Path to `.bashrc` for configuring Slurm environment.
            xtb (str | Path): Path to `xtb` binary.
            crest (str | Path): Path to `crest` binary.
            elim (str | Path): Path to `red_conf_elim.py` script.
            censo (str | Path): Path to `censo` binary.
            censorc (str | Path): Path to `.censorc` file for configuring CENSO execution parameters.
            cosmotherm (str | Path): `COSMOtherm` package to load.
            gaussian (str | Path): Path to `Gaussian` binary.

        Molecular settings:
            smiles (str): SMILES string of the molecule.
            mult (int): Spin multiplicity of the molecule. The workflow is only validated for singlets.
            xtb_solv (str): Solvent to use for xTB calculations (defaults to gas phase).
            censo_solv (str): Solvent to use for CENSO calculations (defaults to gas phase).
            gauss_solv (str): Solvent to use for Gaussian calculations (defaults to gas phase).
            temperature (float): Temperature for the calculations (default: 298.15 K).
            seed (int): Random seed to use when generating conformers.
            num_confs (int): Number of conformers to generate for the molecule (see `conf2mol`).
            elim_confs (float): How many conformations to reduce the ensemble to before running CENSO.
            crest_force_constant (float): Force constant for CREST calculations (default: 5.0).

        Slurm:
            account (str | None): Slurm account to use for the job. If `None`, the user's default account is used.
            xtb_cpus (int): Number of CPUs to allocate for xTB jobs.
            xtb_mem_per_cpu (str): Memory per CPU to allocate for xTB jobs (e.g., "1GB").
            xtb_partition (str): Slurm partition to use for xTB jobs.
            xtb_time (str): Time limit for xTB jobs (e.g., "1:00:00").
            crest_cpus (int): Number of CPUs to allocate for CREST jobs.
            crest_mem_per_cpu (str): Memory per CPU to allocate for CREST jobs (e.g., "1GB").
            crest_partition (str): Slurm partition to use for CREST jobs.
            crest_time (str): Time limit for CREST jobs (e.g., "20:00:00").
            censo_cpus (int): Number of CPUs to allocate for CENSO jobs.
            censo_mem_per_cpu (str): Memory per CPU to allocate for CENSO jobs (e.g., "4GB").
            censo_partition (str): Slurm partition to use for CENSO jobs.
            censo_time (str): Time limit for CENSO jobs (e.g., "48:00:00").
            gauss_low_cpus (int): Number of CPUs to allocate for low-level Gaussian jobs.
            gauss_low_mem (str): Memory to allocate for low-level Gaussian jobs (e.g., "8000MB").
            gauss_low_partition (str): Slurm partition to use for low-level Gaussian jobs.
            gauss_low_time (str): Time limit for low-level Gaussian jobs (e.g., "20:00:00").
            gauss_high_cpus (int): Number of CPUs to allocate for high-level Gaussian jobs.
            gauss_high_mem (str): Memory to allocate for high-level Gaussian jobs (e.g., "12000MB").
            gauss_high_partition (str): Slurm partition to use for high-level Gaussian jobs.
            gauss_high_time (str): Time limit for high-level Gaussian jobs (e.g., "40:00:00").
    """

    def __init__(
        self,

        # Job settings
        job_name: str,
        job_dir: str | Path,
        tmp_dir: str | Path,

        # Program settings
        bashrc: str | Path,
        xtb: str | Path,
        crest: str | Path,
        elim: str | Path,
        censo: str | Path,
        censorc: str | Path,
        cosmotherm: str | Path,
        gaussian: str | Path,

        # Molecular settings
        smiles: str,
        mult: int = 1,
        xtb_solv: str = "",
        censo_solv: str = "",
        gauss_solv: str = "",
        temperature: float = 298.15,
        seed: int = 42,
        num_confs: int = 20,
        elim_confs: int = 50,
        crest_force_constant: float = 5.0,

        # Slurm settings
        account: str | None = None,
        xtb_cpus: int = 4,
        xtb_mem_per_cpu: str = "1GB",
        xtb_partition: str = "PARTITION_NAME",
        xtb_time: str = "1:00:00",
        crest_cpus: int = 8,
        crest_mem_per_cpu: str = "1GB",
        crest_partition: str = "PARTITION_NAME",
        crest_time: str = "20:00:00",
        censo_cpus: int = 32,
        censo_mem_per_cpu: str = "4GB",
        censo_partition: str = "PARTITION_NAME",
        censo_time: str = "48:00:00",
        gauss_low_cpus: int = 8,
        gauss_low_mem: str = "8GB",
        gauss_low_partition: str = "PARTITION_NAME",
        gauss_low_time: str = "20:00:00",
        gauss_high_cpus: int = 8,
        gauss_high_mem: str = "12GB",
        gauss_high_partition: str = "PARTITION_NAME",
        gauss_high_time: str = "40:00:00",

    ) -> None:

        # Define job name and directories
        self.job_name = job_name
        self.job_dir = Path(job_dir)
        self.tmp_dir = Path(tmp_dir)
        self.log_dir = self.job_dir / "logs"

        # Create the job, scratch and logging directories
        self.job_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
        self.tmp_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
        self.log_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

        # Define the molecule
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(self.smiles) # Render with RDKit
        self.protonated_mol = Chem.AddHs(self.mol) # Add explicit protons
        self.charge = get_charge(smiles) # Obtain formal charge
        self.mult = int(mult)
        self.unpaired = int(self.mult-1) # Unpaired electrons is multiplicity-1 (from 2S+1)
        self.atom_types = [atom.GetSymbol() for atom in self.protonated_mol.GetAtoms()] # List of atom types

        # Define other job parameters
        self.xtb_solv = xtb_solv
        self.censo_solv = censo_solv
        self.gauss_solv = gauss_solv
        self.temperature = float(temperature)
        self.seed = seed
        self.num_confs = int(num_confs)
        self.elim_confs = int(elim_confs)
        self.crest_force_constant = crest_force_constant

        # Define xTB kwargs
        self.xtb_kwargs = dict(
            bashrc=bashrc, 
            xtb=xtb, 
            cpus=xtb_cpus,
            mem_per_cpu=xtb_mem_per_cpu, 
            partition=xtb_partition, 
            account=account, 
            time=xtb_time,
        )

        # Define CREST xcontrol kwargs (using shorter xTB resources)
        self.crest_xcontrol_kwargs = dict(
            bashrc=bashrc, 
            xtb=xtb, 
            crest=crest, 
            cpus=xtb_cpus, 
            mem_per_cpu=xtb_mem_per_cpu,
            partition=xtb_partition,
            account=account, 
            time=xtb_time
        )

        # Define CREST kwargs
        self.crest_kwargs = dict(
            bashrc=bashrc, 
            xtb=xtb, 
            crest=crest,
            cpus=crest_cpus,
            mem_per_cpu=crest_mem_per_cpu,
            partition=crest_partition,
            account=account,
            time=crest_time,
        )

        # Define elimination script
        self.elim = elim
    
        # Define CENSO kwargs
        self.censo_kwargs = dict(
            bashrc=bashrc,
            censo=censo,
            censorc=censorc,
            cosmotherm=cosmotherm,
            cpus=censo_cpus,
            mem_per_cpu=censo_mem_per_cpu,
            partition=censo_partition,
            account=account,
            time=censo_time,
        )
        
        # Define low-level Gaussian kwargs
        self.gauss_low_kwargs = dict(
            bashrc=bashrc,
            gauss=gaussian,
            cpus=gauss_low_cpus,
            mem=gauss_low_mem, 
            partition=gauss_low_partition,
            account=account, 
            time=gauss_low_time,
            )
        
        # Define high-level Gaussian kwargs
        self.gauss_high_kwargs = dict(
            bashrc=bashrc,
            gauss=gaussian,
            cpus=gauss_high_cpus,
            mem=gauss_high_mem, 
            partition=gauss_high_partition,
            account=account, 
            time=gauss_high_time,
            )

        # Initialize dictionaries to hold QM energies
        self.censo_dict = {}
        self.crest_best_dict = {}

        # Define a list to hold Output widgets for logging
        self.OUTPUT_WIDGETS = []

    # Define the functions for logging

    def add_output(self) -> None:
        """
        Adds a new Output widget to the list. Use it to split the current Output.
        """

        # Create a new Output widget
        out = Output()

        # Display it in the notebook
        display(out)

        # Add it to the current widget list
        self.OUTPUT_WIDGETS += [out]

    def pprint(
        self,
        col1: ...,
        col2: ...,
        col3: ...,
        highlight: bool = False,
        save: bool = True,
        filename: str = "workflow.log",
    ) -> None:
        """
        Prints a formatted log message to the current Output widget, optionally highlighting
        it in red and saving it to a file, defaulting to "self.log_dir/workflow.log". The first
        column (column 0) is always the current system time.

        Parameters:
            col1 (...): Stringable contents in 2nd column. Max width 12ch.
            col2 (...): Stringable contents in 3rd column. Max width 12ch.
            col3 (...): Stringable contents in 4th column. Max width 60ch.
            highlight (bool): Whether to use normal `stdout` output or red `stderr` output.
            save (bool): Whether to write to file in current directory.
            filename (str): Filename to write to at `self.log_dir`.
        """

        # Get the current system time for the first column
        now = datetime.now()

        # Iterate over the wrapped lines of each column
        for c0, c1, c2, c3 in zip_longest(
            wrap(now.strftime("%Y-%m-%d %H:%M:%S"), 20),
            wrap(str(col1), 12),
            wrap(str(col2), 12),
            wrap(str(col3), 60),
            fillvalue="",
        ):

            # Format the line with fixed-width columns
            line = f"{c0:<20} | {c1:<12} | {c2:<12} | {c3:<60}\n"

            # Get the last Output widget in the list
            current = self.OUTPUT_WIDGETS[-1]

            # Print the line to the appropriate output stream based on the highlight flag
            if highlight:
                current.append_stderr(line)
            else:
                current.append_stdout(line)

            # Save the line to the specified log file if requested
            if save:
                with open(self.log_dir / filename, "a") as f:
                    f.write(line)

    # Define the functions for the core `qm` workflow (embedding, xTB, CREST, ELIM, CENSO)

    def conf2mol(
        self,
        mol: Chem.rdchem.Mol,
        num_confs: int,
        seed: int,
    ) -> tuple[Chem.rdchem.Mol, ...]:
        """
        Generates separate conformers given a molecule, each with one different set of atom positions.

        It is possible for the resulting ensemble to have less conformers than requested, see:
        https://greglandrum.github.io/rdkit-blog/posts/2023-05-17-understanding-confgen-errors.html.

        In such cases, we repeat embedding with more attempts and raise an error if none are produced.

        Parameters:
            mol (Chem.rdchem.Mol): The input molecule for which conformers are to be generated.
            num_confs (int): The number of conformers to generate.
            seed (int): The random seed for reproducibility.

        Returns:
            tuple[Chem.rdchem.Mol, ...]: A tuple containing the generated conformers as separate RDKit Mol objects.

        """

        # Add explicit protons
        positions = Chem.AddHs(mol)

        # Generate conformers using RDKit's EmbedMultipleConfs function
        conf_ids = Chem.EmbedMultipleConfs(
            positions,
            numConfs=num_confs,
            maxAttempts=10000,
            randomSeed=seed,
        )

        # If none were generated, try embedding random coordinates
        if len(set(conf_ids)) != num_confs:
            positions.RemoveAllConformers()
            params = Chem.EmbedParameters()
            params.useRandomCoords = True
            conf_ids = Chem.EmbedMultipleConfs(
                positions, 
                numConfs=num_confs, 
                params=params,
            )

            # Attempt to optimize the conformers with UFF to resolve any bad contacts
            for conf_id in conf_ids:
                try:
                    Chem.UFFOptimizeMolecule(positions, confId=int(conf_id), maxIters=500)
                except Exception:
                    pass

        # End the workflow if no conformers could be generated
        if len(set(conf_ids)) == 0:
            raise RuntimeError("No RDKit conformers could be generated - exiting workflow. Please check your input molecule.")
        
        # Warn if less were generated than requested (but proceed)
        elif len(set(conf_ids)) != num_confs:
            self.pprint("ALL", "conf2mol", f"Number of conformers generated ({len(set(conf_ids))}) does not match number specified ({num_confs}).")

        # Create separate molecules for each conformer
        confs = []
        for position in positions.GetConformers():
            conf = Chem.AddHs(mol)
            conf.RemoveAllConformers()
            conf.AddConformer(position)
            confs.append(conf)
        return tuple(confs)
    
    def export_confs(self) -> None:
        """
        Export molecule conformers to the specified directory.

        This method exports the molecule conformers to the directory specified by `self.job_dir`.
        Each conformer is saved as an XYZ and an SDF file in a subdirectory named after the corresponding file index.

        Example directory: `/job_name/file_idx/confs/0.xyz` and `/job_name/file_idx/confs/0.sdf`
        
        Returns:
            None
        """
        
        # Create the job directory if it doesn't exist
        self.job_dir.mkdir(parents=True, exist_ok=True)

        # Iterate over the rows in the structures DataFrame
        for row_idx in self.structures.index:

            # Get the file index and conformers for the current row
            file_idx = self.structures.at[row_idx, "file_idx"]
            confs = self.structures.at[row_idx, "confs"]

            # Initialize a list to collect exported conformers
            exported_confs = []

            # Create the conformer directories
            confs_dir = self.job_dir / f"{file_idx}/confs"
            confs_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

            # Enumerate over the conformers
            for i, conf in enumerate(confs):

                # Define file names
                xyz = confs_dir / f"{i}.xyz"
                sdf = confs_dir / f"{i}.sdf"
                exported_confs.append(xyz)

                # Generate an XYZ file
                if not xyz.exists():
                    Chem.MolToXYZFile(conf, str(xyz))

                # Generate an SDF file (for equivalence checks)
                if not sdf.exists():
                    convert_xyz_to_sdf(xyz, sdf)

            # Record the exported XYZ in self.structures
            self.structures.at[row_idx, "exported_confs"] = exported_confs

            self.pprint(file_idx, "export", f"Exported {len(confs)} molecule conformers to {self.job_dir}.")

    def run_xtb(
        self,
        row_idx: int,
        step_name: str,
        pbar: ProgressBar,
        input_xyzs: list,
        **kwargs,
        ) -> list[Job]:
        """
        Runs batched xTB jobs for a specific row in the structures DataFrame.

        Parameters:
            row_idx (int): Row index of the internal state table, self.structures, to process.
            step_name (str): Label used to identify the workflow step.
            pbar (ProgressBar): The progress bar to update during job execution.
            input_xyzs (list): List of paths to the XYZ files to run with xTB.
            **kwargs (dict): Any keyword arguments accepted by `slurm.xtb`.

        Returns:
            list[Job]: List of `Job` objects representing the xTB jobs created.
        """

        # Get the file index for this row
        file_idx = self.structures.at[row_idx, "file_idx"]

        # Define the job column name in self.structures for this step
        job_column = f"{step_name}_jobs"

        # Ensure that a job list column exists in self.structures
        if job_column not in self.structures:
            self.structures[job_column] = pd.Series(
                [[] for _ in range(len(self.structures))]
            )

        # Define the list of jobs for this row and step
        jobs = self.structures.at[row_idx, job_column]

        # Clear any existing jobs for this step
        jobs.clear()

        # Ensure that there are conformers to process
        if not any(input_xyzs):
            self.pprint(file_idx, step_name, "No conformers available, skipping.")
            return jobs

        # Create a list of outputs for each xTB job
        xtb_outs = [
            xyz.parents[1] / "xtb" / xyz.stem / "xtb.out"
            for xyz in input_xyzs
        ]

        # Combine inputs and outputs into a dictionary
        xyz_out_dict = dict(zip(input_xyzs, xtb_outs))

        # Identify which calculations already have an output
        # - To check if an xTB job has been attempted we look for `xtb.out` (this is present whether it fails or succeeds)
        # - To check if an xTB job has completed successfully we look for `.xtboptok` (this is only present on success)
        # - we use the former, to avoid continuously rerunning xTB jobs that repeatedly fail
        inputs_to_run = {xyz: out for xyz, out in xyz_out_dict.items() if not (self.job_dir / out).exists()}

        # Setup the job if any outputs don't exist
        if inputs_to_run:
            job = xtb_batch(
                inputs=list(inputs_to_run.keys()),
                outputs=list(inputs_to_run.values()),
                slurm_name=f"{self.job_name}_batch_{step_name}",
                job_dir=self.job_dir,
                tmp_dir=self.tmp_dir,
                log_dir=self.log_dir,
                pbar=pbar,
                charge=self.charge,
                unpaired=self.unpaired,
                solvent=self.xtb_solv,
                temperature=self.temperature,
                **kwargs,
            )
            jobs += [job]

        # Output check
        if not jobs:
            self.pprint(file_idx, step_name, ".xtboptok exists for all conformers, skipping.")
        else:
            self.pprint(file_idx, step_name, f"{len(jobs)} job(s) added with {len(inputs_to_run)} conformers.")
        return jobs

    def find_stable_xtb(
        self,
        row_idx: int,
        step_name: str,
        input_xyzs: list,
        ):
        """
        Find the most stable xTB conformer with correct connectivity and stereochemistry for a given row index.
        Note that carbanions are particularly prone to changing connectivity during optimisations.

        Parameters:
            row_idx (int): Row index of the internal state table, self.structures, to process.
            step_name (str): Label used to identify the workflow step.
            input_xyzs (list): List of paths to the XYZ files to run with xTB.

        Returns:
            None
        """

        # Get the file index for this row
        file_idx = self.structures.at[row_idx, "file_idx"]

        # Create a list of output XYZ files for each xTB job
        xtb_xyzs = [
            xyz.parents[1] / "xtb" / xyz.stem / "xtbopt.xyz"
            for xyz in input_xyzs
        ]

        # Check if xTB outputs exist
        if not any(xtb_xyzs):
            self.pprint(file_idx, step_name, "No xtbopt.xyz available, skipping.")
            return

        # Create a dictionary to store the energies of the xTB conformers
        xtb_energies = {}

        # Iterate over the xTB .xyz files
        for xyz in xtb_xyzs:

            # Generate an SDF file for the output
            output_sdf = xyz.parent / f"xtbopt.sdf"
            convert_xyz_to_sdf(xyz, output_sdf)

            # Check structure is equivalent to the input
            input_sdf = xyz.parent.parent.parent / "confs" / f"{xyz.parent.name}.sdf"
            equivalent = compare_equivalence(input_sdf, output_sdf)

            # Only consider conformers that retained the correct connectivity and stereochemistry after xTB optimisation
            if equivalent:

                # Extract the energy
                energy = get_xtb_energy(xyz)
                xtb_energies[xyz] = energy

        # Find the most stable xTB conformer
        if xtb_energies:
            stable_xyz = min(xtb_energies, key=xtb_energies.get)
            self.pprint(file_idx, step_name, f"Lowest energy xTB conformer: {stable_xyz}")

        # Otherwise, if no conformers retained the correct connectivity and stereochemistry after
        # xTB, use a conformer from the original embedding (arbitrarily conformer 0.xyz)
        else:
            stable_xyz = self.job_dir / f"{file_idx}" / "confs" / "0.xyz"
            self.pprint(file_idx, step_name, f"No xTB conformers retained connectivity or stereochemistry. Using: {stable_xyz}")

        # Record the stable_xyz path in self.structures
        self.structures.at[row_idx, "stable_xyz"] = Path(stable_xyz)

    def update_fc(self, filename):
        """
        Updates the force constant in the xcontrol file to `self.crest_force_constant`.

        Parameters:
            filename (str): The path to the file.

        Returns:
            None
        """

        # Read the file
        with open(filename) as file:
            lines = file.readlines()

        # Replace the second line with the new force constant
        new_line = f"  force constant={self.crest_force_constant}\n"
        lines[2] = new_line

        # Write the updated lines back to the file
        with open(filename, "w") as file:
            for line in lines:
                file.write(line)

    def run_crest(
        self,
        row_idx: int,
        step_name: str,
        pbar: ProgressBar,
        input_xyz: str,
        output: str,
        xcontrol: Path | None = None,
        **kwargs,
    ) -> list[Job]:
        """
        Runs a CREST job for a specific row in the structures DataFrame.

        Parameters:
            row_idx (int): Row index of the internal state table, self.structures, to process.
            step_name (str): Label used to identify the workflow step.
            pbar (ProgressBar): The progress bar to update during job execution.
            input_xyz (Path): Path to the XYZ file to run with CREST.
            output (Path): Path to the intended CREST output.
            xcontrol (Path, optional): Path to the .xcontrol file for constrained CREST runs. If provided, the CREST job will be run with constraints.
            **kwargs (dict): Any keyword arguments accepted by `slurm.crest`.

        Returns:
            list[Job]: List of `Job` objects representing the CREST jobs created.

        """

        # Get the file index for this row
        file_idx = self.structures.at[row_idx, "file_idx"]

        # Define the job column name in self.structures for this step
        job_column = f"{step_name}_jobs"

        # Ensure that a job list column exists in self.structures
        if job_column not in self.structures:
            self.structures[job_column] = pd.Series(
                [[] for _ in range(len(self.structures))]
            )

        # Define the list of jobs for this row and step
        jobs = self.structures.at[row_idx, job_column]

        # Clear any existing jobs for this step
        jobs.clear()

        # If an xcontrol file is specified, run a search with constraints
        if xcontrol is not None:

            if not xcontrol.exists():
                self.pprint(file_idx, step_name, "xcontrol file does not exist, skipping.",)
                return jobs
            
            # Change xcontrol force constant from 0.5 to 5.0 to prevent movement during CREST
            self.update_fc(xcontrol)

            # Define the constraints to add to the CREST command
            constraints = " --cinp .xcontrol.sample "
        
        else:
            constraints = ""

        # Report if the xyz doesn't exist
        if not input_xyz.exists():
            self.pprint(file_idx, step_name, f"Source geometry ({input_xyz}) does not exist, skipping.",)
            return jobs

        # Setup the CREST job from the chosen conformer
        if not output.exists():
                job = crest(
                    input=input_xyz,
                    output=output,
                    slurm_name=f"{self.job_name}_{step_name}",
                    job_dir=self.job_dir,
                    tmp_dir=self.tmp_dir,
                    log_dir=self.log_dir,
                    pbar=pbar,
                    charge=self.charge,
                    unpaired=self.unpaired,
                    solvent=self.xtb_solv,
                    temperature=self.temperature,
                    constraints=constraints,
                    **kwargs,
                )
                jobs += [job]

        # Output check
        if not jobs:
            self.pprint(file_idx, step_name, "CREST output exists, skipping.")
        else:
            self.pprint(file_idx, step_name, f"{len(jobs)} job(s) added.")
        return jobs

    def run_crest_xcontrol(
        self,
        row_idx: int,
        step_name: str,
        input_xyz: str,
        output: str,
        constrained_atoms: list[int],
        **kwargs,
    ) -> list[Job]:
        """
        Runs a CREST job to generate a .xcontrol_sample file for a specific row in the structures DataFrame.
        These files are used to constrain the CREST conformational search to specific atoms or bonds.

        Parameters:
            row_idx (int): Row index of the internal state table, self.structures, to process.
            step_name (str): Label used to identify the workflow step.
            input_xyz (Path): Path to the XYZ file to run with CREST.
            output (Path): Path to the intended CREST output.
            constrained_atoms (list[int]): The list of atom indices to be constrained.
            **kwargs (dict): Any keyword arguments accepted by `slurm.crest`.

        Returns:
            list[Job]: List of `Job` objects representing the CREST jobs created.
        """

        # Get the file index for this row
        file_idx = self.structures.at[row_idx, "file_idx"]

        # Define the job column name in self.structures for this step
        job_column = f"{step_name}_jobs"

        # Ensure that a job list column exists in self.structures
        if job_column not in self.structures:
            self.structures[job_column] = pd.Series(
                [[] for _ in range(len(self.structures))]
            )

        # Define the list of jobs for this row and step
        jobs = self.structures.at[row_idx, job_column]

        # Clear any existing jobs for this step
        jobs.clear()

        # Report if the xyz doesn't exist
        if not input_xyz.exists():
            self.pprint(file_idx, step_name, f"Source geometry ({input_xyz}) does not exist, skipping.",)
            return jobs

        # Generate the constraint
        constraint_str = ""
        for i in range(len(constrained_atoms)):
            constraint_str += str(constrained_atoms[i])
            if i + 1 != len(constrained_atoms):
                constraint_str += ","

        # Define the .xcontrol file path
        xcontrol_file = output.parent / ".xcontrol.sample"

        # Generate the .xcontrol file if it doesn't exist
        if not xcontrol_file.exists():
            job = crest(
                input=input_xyz,
                output=output,
                slurm_name=f"{self.job_name}_{step_name}",
                job_dir=self.job_dir,
                tmp_dir=self.tmp_dir,
                log_dir=self.log_dir,
                charge=self.charge,
                unpaired=self.unpaired,
                solvent=self.xtb_solv,
                temperature=self.temperature,
                constraints=f" --constrain {constraint_str} ",
                **kwargs,
            )
            jobs += [job]

        # Output check
        if not jobs:
            self.pprint(file_idx, step_name, "CREST output exists, skipping.")
        else:
            self.pprint(file_idx, step_name, f"{len(jobs)} job(s) added.")
        return jobs

    def run_elim(
        self,
        row_idx: int,
        step_name: str,
        input: str,
    ) -> None:
        """
        Runs a redundant conformer elimination job for a specific row in the structures DataFrame.

        Parameters:
            row_idx (int): Row index of the internal state table, self.structures, to process.
            step_name (str): Label used to identify the workflow step.
            input (Path): Path to the XYZ ensemble to reduce.

        Returns:
            None
        """

        # Initialize list to store subprocess handles
        processes = []

        # Get the file index for this row
        file_idx = self.structures.at[row_idx, "file_idx"]

        # Define the outputs
        elim_output = input.parent / "elim.out"
        reduced_ensemble = input.parent / "crest_conformers_reduced.xyz"

        # Check if input ensemble exists
        if input.exists():

            # Check if both the outputs already exists
            if reduced_ensemble.exists() and elim_output.exists():
                self.pprint(file_idx, step_name, "crest_conformers_reduced.xyz and elim.out already exist, skipping.")
                return

            else:

                # If the number of conformers is already less than or equal to the
                # threshold, skip job and populate `elim.out` with a default message
                _, __, n_conformers = read_from_xyz(input)
                if n_conformers <= self.elim_confs:
                    with open(elim_output, 'w') as file:
                        self.pprint(file_idx, step_name, "Number of conformers acceptable - no elimination performed.")
                        file.write("Number of conformers acceptable - no elimination performed.\n")
                    return
                
                # Otherwise setup the job
                else:
                    p = subprocess.Popen(
                        f"python {self.elim} {input} {self.elim_confs} > {elim_output}",
                        shell=True,
                    )

                    # Add the process to the list of running processes
                    processes.append(p)
                    self.pprint(file_idx, step_name, f"Running elimination on {input.name}")

        # Wait for all subprocesses to complete
        [p.wait() for p in processes]

        # Output check
        if processes:
            self.pprint(file_idx, step_name, f"{len(processes)} process(es) added.")

    def _index_containing_substring(self, the_list, substring):
        """
        For a list of strings, get the index of the element containing a substring.

        Parameters:
            the_list (list): A list of strings to search through.
            substring (str): The substring to search for within the list elements.
        
        Returns:
            int or None: The index of the first element containing the substring, or None if not found.
        """
        for i, s in enumerate(the_list):
            if substring in s:
                return i
        return None

    def report_elim(
        self,
        row_idx: int,
        step_name: str,
        input: str,
    ):
        """
        Report the results of elimination for a specific row index.

        Parameters:
            row_idx (int): Row index of the internal state table, self.structures, to process.
            step_name (str): Label used to identify the workflow step.
            input (Path): Path to the elimination output file to report.

        Returns:
            None
        """

        # Get the file index for this row
        file_idx = self.structures.at[row_idx, "file_idx"]

        # Read elimination output
        with open(input, "r") as f:
            lines = f.readlines()
        idx1 = self._index_containing_substring(lines, "Conformers Omitted")
        idx2 = self._index_containing_substring(lines, "Conformers Retained")

        # Report results of elimination
        if idx1 is not None and idx2 is not None:
            omitted = lines[idx1].split(" ")[-1]
            retained = lines[idx2].split(" ")[-1]
            self.pprint(file_idx, step_name, f"{omitted}conformers omitted and {retained}retained.")

        # Report jobs as finished
        self.pprint(file_idx, step_name, "All jobs done.")

    def run_censo(
        self,
        row_idx: int,
        step_name: str,
        pbar: ProgressBar,
        input: str,
        output: str,
        **kwargs,
    ) -> list[Job]:
        """
        Runs a CENSO job for a specific row in the structures DataFrame.

        Parameters:
            row_idx (int): Row index of the internal state table, self.structures, to process.
            step_name (str): Label used to identify the workflow step.
            pbar (ProgressBar): The progress bar to update during job execution.
            input (Path): Path to the XYZ ensemble to run with CENSO.
            output (Path): Path to the intended CENSO output.
            **kwargs: Any keyword arguments accepted by `slurm.censo`.

        Returns:
            list[Job]: List of `Job` objects representing the CENSO jobs created.
        """

        # Get the file index for this row
        file_idx = self.structures.at[row_idx, "file_idx"]

        # Define the job column name in self.structures for this step
        job_column = f"{step_name}_jobs"

        # Ensure that a job list column exists in self.structures
        if job_column not in self.structures:
            self.structures[job_column] = pd.Series(
                [[] for _ in range(len(self.structures))]
            )

        # Define the list of jobs for this row and step
        jobs = self.structures.at[row_idx, job_column]

        # Clear any existing jobs for this step
        jobs.clear()

        # Report which ensemble is being used for CENSO
        self.pprint(file_idx, step_name, f"Running CENSO job from {input.name}.")

        # Setup the job (if output doesn't exist)
        if not output.exists():
            job = censo(
                input=input,
                output=output,
                slurm_name=f"{self.job_name}_{step_name}",
                job_dir=self.job_dir,
                tmp_dir=self.tmp_dir,
                log_dir=self.log_dir,
                pbar=pbar,
                charge=self.charge,
                unpaired=self.unpaired,
                solvent=self.censo_solv,
                temperature=self.temperature,
                **kwargs,
            )
            jobs += [job]

        # Output check
        if not jobs:
            self.pprint(file_idx, step_name, "Logs indicate success, skipping.")
        else:
            self.pprint(file_idx, step_name, f"{len(jobs)} job(s) added.")
        return jobs

    def write_gaussian_input(self, input_gjf, route, atom_list, xyz_list, cpus, mem, tail_list=[]):
        """
        Generates and writes a Gaussian input file (.GJF) with specified parameters.

        Parameters:
            input_gjf (str): Path to the output Gaussian input file.
            route (str): The Gaussian route section specifying the calculation type and options.
            atom_list (list[str]): A list of atomic symbols corresponding to each atom in the molecule.
            xyz_list (list[str]): A list of strings, each containing the x, y, z coordinates for the corresponding atom in `atom_list`.
            cpus (int): The number of CPU cores to request for the Gaussian calculation.
            mem (str): The amount of memory to request for the Gaussian calculation (e.g., "4GB").
            tail_list (list[str], optional): A list of additional lines to append to the end of the file (e.g., constraints).

        Returns:
            None
        """

        # Define the .GJF template
        gjf = f"""%mem={mem}
%nprocshared={cpus}
# {route}

{self.job_name}

{self.charge} {self.mult}
"""

        # Ensure atom_list and xyz_list have the same length
        if len(atom_list) != len(xyz_list):
            raise ValueError("atom_list and xyz_list must have the same length.")

        # Add the atom types and coordinates
        for i in range(len(atom_list)):

            # Fix scientific notation if present in the coordinates
            xyz_fixed = self._fix_scientific_notation(xyz_list[i])
            gjf += f"{atom_list[i]} {xyz_fixed}\n"

        # Put an empty line on the end
        gjf += "\n"

        if tail_list:

            # Add the tail lines (e.g. for freezing dihedrals)
            for i in range(len(tail_list)):
                gjf += f"{tail_list[i]}\n"

            # Put an empty line on the end
            gjf += "\n"

        # Write the .GJF file
        with open(input_gjf, "w") as f:
            f.write(gjf)

    def write_xyz(self, input_xyz, atom_list, xyz_list, xyz_name=""):
        """
        Generates and writes an XYZ file with specified parameters.

        Parameters:
            input_xyz (str): Path to the output XYZ file.
            atom_list (list[str]): A list of atomic symbols corresponding to each atom in the molecule.
            xyz_list (list[str]): A list of strings, each containing the x, y, z coordinates for the corresponding atom in `atom_list`.
            xyz_name (str, optional): A name or comment line for the XYZ file. Default is an empty string.

        Returns:
            None

        """

        # Define the .XYZ template
        xyz = f"""
{len(atom_list)}
{xyz_name}
"""
        # Add the atom types and coordinates
        for i in range(len(atom_list)):

            # Fix scientific notation if present in the coordinates
            xyz_fixed = self._fix_scientific_notation(xyz_list[i])
            xyz += f"{atom_list[i]} {xyz_fixed}\n"

        # Write the .XYZ file
        with open(input_xyz, "w") as f:
            f.write(xyz)

    def run_gaussian(
        self, 
        row_idx: int,
        step_name: str,
        pbar: ProgressBar,
        input_gjfs: list,
        **kwargs,
        ) -> list[Job]:
        """
        Runs a Gaussian job for a specific row in the structures DataFrame.

        Parameters:
            row_idx (int): Row index of the internal state table, self.structures, to process.
            step_name (str): Label used to identify the workflow step.
            pbar (ProgressBar): The progress bar to update during job execution.
            input_gjfs (list): List of paths to the GJF files to run with Gaussian.
            **kwargs (dict): Any keyword arguments accepted by `slurm.gaussian`.

        Returns:
            list[Job]: List of `Job` objects representing the Gaussian jobs created.
        """

        # Get the file index for this row
        file_idx = self.structures.at[row_idx, "file_idx"]

        # Define the job column name in self.structures for this step
        job_column = f"{step_name}_jobs"

        # Ensure that a job list column exists in self.structures
        if job_column not in self.structures:
            self.structures[job_column] = pd.Series(
                [[] for _ in range(len(self.structures))]
            )

        # Define the list of jobs for this row and step
        jobs = self.structures.at[row_idx, job_column]

        # Clear any existing jobs for this step
        jobs.clear()

        # Check that all required Gaussian input files exist. If not, return the empty job list
        if not all(gjf.exists() for gjf in input_gjfs):
            self.pprint(file_idx, step_name, "Required Gaussian input files are missing, skipping.")
            return jobs

        # Define the job slurm name
        slurm_name = f"{self.job_name}_{step_name}"

        # Iterate over each input Gaussian file
        for gjf in input_gjfs:

            # Define the outputs
            out_dir = gjf.parent
            log_file = f"{gjf.stem}.log"
            log_path = out_dir / log_file

            # If the expected log file doesn't exist, submit the job via slurm.
            # If it does exist, we skip the job to avoid duplication.
            if not log_path.exists():
                job = gaussian(
                    input=gjf,
                    output=log_path,
                    slurm_name=slurm_name,
                    job_dir=self.job_dir,
                    tmp_dir=self.tmp_dir,
                    log_dir=self.log_dir,
                    pbar=pbar,
                    **kwargs,
                )

                # Append the job object to the list of jobs for this row and step
                jobs.append(job)

        # Report on the number of jobs submitted for this step and file index. 
        # If no jobs were submitted, the expected output log files already exist.
        if not jobs:
            self.pprint(file_idx, step_name, "Output exists, skipping.")
        else:
            self.pprint(file_idx, step_name, f"{len(jobs)} job(s) added.")

        return jobs

    def extract_energies_with_equivalence_check(self, row_idx, file_idx, program):
        """
        Checks that connectivity and stereochemistry (equivalence) were retained for a specific
        row and file index for the given QM program ('crest' or 'censo_px' where x = the part
        of CENSO to check. If the structure has the correct connectivity and stereochemistry
        (i.e., is equivalent) compared to the original xTB input, energies are extracted for
        the specified program.

        Parameters:
            row_idx (int): Row index of the internal state table, self.structures, to process.
            file_idx (int): File index of the internal state table, self.structures, to process.
            program (str): Program name ('crest' or 'censo_px').

        Returns:
            bool: True if energies were successfully extracted, False otherwise.
        """

        # Define the current directory
        current_dir = self.job_dir / str(file_idx)

        # Define the relevant XYZ ensemble
        if program == 'crest':
            xyz_ensemble = current_dir / "crest" / "crest_conformers.xyz"
        elif 'censo' in program:
            part = program.split('_')[-1]
            xyz_ensemble = current_dir / "censo" / f"enso_ensemble_{part}.xyz"

        # Check that the XYZ ensemble exists
        if xyz_ensemble.exists():
            self.structures.at[row_idx, f"{program}_exists"] = True

            # Get the xTB conformer (this has same connectivity as original embedding)
            # The SDF already exists from the equivalence checks performed after xTB
            stable_xyz = self.structures.at[row_idx, "stable_xyz"]
            input_sdf = str(stable_xyz).replace(".xyz", ".sdf")

            # Get the XYZ conformer to compare against (this already exists for
            # CREST) but must be generated for CENSO from the XYZ ensemble
            if program == 'crest':
                xyz_best = current_dir / "crest" / "crest_best.xyz"
            elif 'censo' in program:

                # Parse the XYZ coordinates of the lowest energy conformer
                xyz_block, _, __ = get_conformer_xyz(xyz_ensemble, 0)

                # Write the conformer to its own .xyz file
                xyz_best = current_dir / "censo" / f"most_stable_{part}.xyz"
                with open(xyz_best, "w") as file_out:
                    for line in xyz_block:
                        file_out.write(line)

            # Generate an SDF file for the best XYZ file
            output_sdf = str(xyz_best).replace(".xyz", ".sdf")
            if not Path(output_sdf).exists():
                convert_xyz_to_sdf(xyz_best, output_sdf)

            # Check and record if structure is equivalent to the input xTB geometry
            equivalent = compare_equivalence(input_sdf, output_sdf)
            self.structures.at[row_idx, f"{program}_equivalent"] = equivalent

            # If equivalent, collect energies
            if equivalent:
                self.pprint(file_idx, program, f'Connectivity ok.') 

                # Get energies of CREST
                if program == 'crest':
                    crest_best = current_dir / "crest" / "crest_best.xyz"
                    if crest_best.exists():
                        gibbs = get_crest_energy(crest_best)
                        self.crest_best_dict[row_idx] = gibbs

                # Get energies of all CENSO parts
                elif 'censo' in program:
                    censo_out = current_dir / "censo" / "censo.out"
                    if censo_out.exists():
                        gibbs = get_censo_energy(censo_out, False if self.censo_solv == "" else True)
                        self.censo_dict[row_idx] = gibbs

                return True
            
            # If not equivalent, report a change in connectivity or stereochemistry
            else:
                self.pprint(file_idx, program, f'Connectivity or stereochemistry has changed - please check the structure.')
                raise ValueError(f"Connectivity or stereochemistry has changed for {program} - please check the structure.")

        else:
            self.pprint(file_idx, program, f'The XYZ ensemble ({xyz_ensemble}) does not exist.')
            self.structures.at[row_idx, f"{program}_exists"] = False
            return False

    @task
    async def qm(
        self, 
        elim: bool = True,
        cleanup: bool = True,
    ) -> None:
        """
        Quantum mechanics workflow:
        - Embed input SMILES into a number of 3D conformers with RDKit (reactant_censo/confs). 
        - Optimize these RDKit conformers with xTB (reactant_censo/xtb). 
        - Search the lowest energy conformer with CREST (reactant_censo/crest). 
        - Refine the CREST ensemble with CENSO (reactant_censo/censo). 
        - Collect energies and check equivalence after CENSO.

        Parameters:
            elim (bool): Whether to perform redundant conformer elimination before CENSO.
            cleanup (bool): Whether to remove the temporary working directory after all jobs are finished.

        Returns:
            None
        """

        # Start a new output for QM workflow
        self.add_output()

        # Export conformers
        self.export_confs()

        # Define progress bars
        xtb_pbar = ProgressBar("xtb_react")
        crest_pbar = ProgressBar("crest_react")
        censo_pbar = ProgressBar("censo_react")

        # Split output to prevent pbars overwriting self.pprint log
        self.add_output()

        async def per_row(row_idx):

            # Get the file index for this row
            file_idx = self.structures.at[row_idx, "file_idx"]

            # Run the xTB jobs (batched)
            xtb_jobs = self.run_xtb(
                row_idx=row_idx, 
                step_name="xtb_react",
                pbar=xtb_pbar, 
                input_xyzs=self.structures.at[row_idx, "exported_confs"],
                **self.xtb_kwargs
                )

            # Wait for the xTB jobs to finish
            await asyncio.gather(*[job.task for job in xtb_jobs])
            self.pprint(file_idx, "xtb_react", "All jobs done.")

            # Report all xTB jobs as finished and complete progress bar
            all_xtb_jobs = self.structures["xtb_react_jobs"].sum()
            if all(isinstance(job, Job) and job.task.done() for job in all_xtb_jobs):
                if not xtb_pbar.disable:
                    self.pprint("ALL", "xtb_react", "All jobs done.", highlight=True)
                    xtb_pbar.close_pbar()

            # Find the most stable XYZ geometry to provide to CREST
            self.find_stable_xtb(
                row_idx,
                step_name="xtb_react",
                input_xyzs=self.structures.at[row_idx, "exported_confs"]
                )

            # Run the CREST jobs
            crest_jobs = self.run_crest(
                row_idx=row_idx,
                step_name="crest_react",
                pbar=crest_pbar,
                input_xyz=self.structures.at[row_idx, "stable_xyz"],
                output=Path(f"{self.job_dir}/{file_idx}/crest/crest.out"),
                **self.crest_kwargs
                )

            # Wait for the CREST jobs to finish
            await asyncio.gather(*[job.task for job in crest_jobs])
            self.pprint(file_idx, "crest_react", "All jobs done.")

            # Report all CREST jobs as finished and complete progress bar
            all_crest_jobs = self.structures["crest_react_jobs"].sum()
            if all(isinstance(job, Job) and job.task.done() for job in all_crest_jobs):
                if not crest_pbar.disable:
                    self.pprint("ALL", "crest_react", "All jobs done.", highlight=True)
                    crest_pbar.close_pbar()

            # Check that the conformers retained connectivity and stereochemistry from original embedding to CREST
            crest_equivalence = self.extract_energies_with_equivalence_check(row_idx, file_idx, program='crest')

            # Only continue with CENSO workflow if equivalence was retained from CREST
            if not crest_equivalence:
                self.pprint(file_idx, "crest_react", f'CREST best geometry has different connectivity or stereochemistry, no further calculations will be done for this idx.')
            else:
    
                if elim:

                    # Run the elimination jobs
                    self.run_elim(
                        row_idx=row_idx, 
                        step_name="elim_react",
                        input=Path(f"{self.job_dir}/{file_idx}/crest/crest_conformers.xyz"),
                        )

                    # Report on the number of retained or eliminated conformers
                    self.report_elim(
                        row_idx, 
                        step_name="elim_react", 
                        input=Path(f"{self.job_dir}/{file_idx}/crest/elim.out")
                    )
                        
                    # Report all elimination jobs as finished
                    self.pprint("ALL", "elim_react", "All jobs done.", highlight=True)

                # Define the CENSO input
                reduced_ensemble = Path(f"{self.job_dir}/{file_idx}/crest/crest_conformers_reduced.xyz")
                full_ensemble = Path(f"{self.job_dir}/{file_idx}/crest/crest_conformers.xyz")
                censo_input = reduced_ensemble if reduced_ensemble.exists() else full_ensemble
                censo_output = Path(f"{self.job_dir}/{file_idx}/censo/censo.out")

                # Run the CENSO jobs
                censo_jobs = self.run_censo(
                    row_idx=row_idx, 
                    step_name="censo_react",
                    pbar=censo_pbar,
                    input=censo_input,
                    output=censo_output,
                    **self.censo_kwargs
                    )

                # Wait for the CENSO jobs to finish
                await asyncio.gather(*[job.task for job in censo_jobs])
                self.pprint(file_idx, "censo_react", "All jobs done.")

                # Report all CENSO jobs as finished and complete progress bar
                all_censo_jobs = self.structures["censo_react_jobs"].sum()
                if all(isinstance(job, Job) and job.task.done() for job in all_censo_jobs):
                    if not censo_pbar.disable:
                        self.pprint("ALL", "censo_react", "All jobs done.", highlight=True)
                        censo_pbar.close_pbar()

                # Check which parts CENSO are active for the current workflow
                p1_active, p2_active, p3_active = False, False, False
                censo_out = self.job_dir / f"{file_idx}/censo/censo.out"
                with open(censo_out, 'r') as file:
                    for line in file:
                        if "part1:" in line:
                            if "   on" in line:
                                p1_active = True
                        if "part2:" in line:
                            if "   on" in line:
                                p2_active = True
                        if "part3:" in line:
                            if "   on" in line:
                                p3_active = True

                # Check that the conformers retained connectivity from original embedding to CENSO
                if p1_active:
                    self.extract_energies_with_equivalence_check(row_idx, file_idx, program='censo_part1')
                if p2_active:
                    self.extract_energies_with_equivalence_check(row_idx, file_idx, program='censo_part2')
                if p3_active:
                    self.extract_energies_with_equivalence_check(row_idx, file_idx, program='censo_part3')

        await asyncio.gather(*[per_row(row_idx) for row_idx in self.structures.index])

        # Check that all CENSO jobs are done
        censo_jobs = self.structures["censo_react_jobs"].sum()
        if len(censo_jobs) == 0:
            self.pprint("ALL", "censo_react", "No jobs found.")

        # Attempt to remove the temporary scratch directory 
        if cleanup and self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
            self.pprint("ALL", "ALL", "Removed temporary job directory.")

        # Remove job columns from dataframe
        self.structures = self.structures.drop(
            [
                "xtb_react_jobs",
                "crest_react_jobs",
                "censo_react_jobs",
            ],
            axis=1,
            errors="ignore", # ignore nonexistent job columns
        )
