import os
import asyncio
import functools
import shutil
from collections.abc import Callable
from tqdm.auto import tqdm
from pathlib import Path

### Default bash scripts for each software type
# Note that `ulimit -s unlimited` is required to prevent stack overflow errors with some softwares

# ---
# XTB (batch)
XTB1 = """#!/bin/bash
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={time}
#SBATCH --output={slurm_output}
#SBATCH --partition {partition}
#SBATCH --hint=nomultithread
{account}

source {bashrc}

ulimit -s unlimited
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_STACKSIZE="$SLURM_MEM_PER_CPU"M
"""

XTB2 = """
cd "{output_dir}"
echo {xyz}
{xtb} {xyz} --opt --etemp {temperature} --chrg {charge} --uhf {unpaired} {solvent} -P "$SLURM_CPUS_PER_TASK" > {output}
"""

# ---
# CREST
CREST = """#!/bin/bash
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={time}
#SBATCH --output={slurm_output}
#SBATCH --partition {partition}
#SBATCH --hint=nomultithread
{account}

source {bashrc}

export PATH={xtb}:$PATH

ulimit -s unlimited
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_STACKSIZE="$SLURM_MEM_PER_CPU"M

echo {xyz}
{crest} {xyz} {constraints} -v2i -gfn2 --etemp {temperature} --chrg {charge} --uhf {unpaired} {solvent} --noreftopo --opt tight -squick --T "$SLURM_CPUS_PER_TASK" > {output}
"""


# ---
# CENSO
CENSO = """#!/bin/bash
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={time}
#SBATCH --output={slurm_output}
#SBATCH --partition {partition}
{account}

source {bashrc}

export PARA_ARCH=SMP

module load {cosmotherm}

ulimit -s unlimited
export PARNODES={cpus_per_calc}
export SMPCPUS={cpus_per_calc}
export OMP_NUM_THREADS={cpus_per_calc}
export OMP_STACKSIZE="$SLURM_MEM_PER_CPU"M
export TURBOTMPDIR=""

echo {xyz}
{censo} -inprc {censorc} -inp {xyz} --charge {charge} --unpaired {unpaired} {solvent} --temperature {temperature} -P {parallel_calcs} -O {cpus_per_calc} > {output}
"""

# ---
# Gaussian
GAUSS = """#!/bin/bash
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={slurm_output}
#SBATCH --partition {partition}
{account}

source {bashrc}

echo {gjf}
{gauss} < {gjf} > {output}
"""


### Define functions

def check_executable(exe_path):

    # Check the program executable exists
    if not Path(exe_path).exists():
        raise FileNotFoundError(f"Executable not found: {exe_path}")
    
    # Check the program is executable
    if not os.access(exe_path, os.X_OK):
        raise PermissionError(f"Cannot access executable. Check permissions for:  {exe_path}")

def task(func):
    """
    Decorator that converts a function into an asyncio task.
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return asyncio.create_task(func(*args, **kwargs))

    return wrapped

def _callback(job, **kwargs) -> None:
    """
    Default callback function that does nothing, to be used as a placeholder.

    Parameters:
        job: The job object.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """
    return

class Job:
    """
    Class that defines a Slurm job. It is used to store information about the job and to submit it to Slurm.
    Jobs should not be reused after completion.

    Attributes:
        job_name (str): Job name.
        job_id (int | None): Job ID.
        cmd (str | None): `sbatch` command executed in shell.
        script (str | None): Contents of script executed by `sbatch`.
        exitcode (int | None): Exit code of `sbatch` when execution is completed.
    """

    job_name: str
    job_id: int | None
    cmd: str | None
    script: str | None
    exitcode: int | None

    def __init__(
        self,
        job_name: str = "",
        job_dir: Path = None,
        tmp_dir: Path = None,
        log_dir: Path = None,
    ) -> None:
        """
        Prepares basic information for job creation.

        Parameters:
            job_name (str): Job name.
            job_dir (Path): Absolute path to working directory, used in `--chdir`.
            tmp_dir (Path): Absolute path to the temporary (scratch) working directory, used in `--chdir`.
            log_dir (Path): Absolute path to Slurm log directory.
        """

        # Store instance variables
        self.job_name = job_name
        self.job_dir = job_dir
        self.tmp_dir = tmp_dir
        self.log_dir = log_dir
        self.cmd = None
        self.script = None
        self.exitcode = None
        self.task = None
        self.id = None

        # Create the job, scratch, and logging directories
        self.job_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
        self.tmp_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
        self.log_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

    @task
    async def _sbatch(
        self,
        script: str,
        wait: bool,
        use_tmp_dir: bool,
        callback: Callable,
        callback_kwargs: dict,
    ):
        """
        Runs an sbatch script with some preconfigured arguments.
        Note that the job will keep running even if the shell is closed (e.g. if Python is forcibly terminated).

        Parameters:
            script (str): Contents of the sbatch script.
            wait (bool): Adds the `--wait` argument, which makes the function wait until the job is complete before returning.
            use_tmp_dir (bool): Whether to use the temporary directory instead of the current working directory for the job.
            callback ((self, **kwargs) -> self): Callback function that is executed once the shell has closed. The first argument is always the current instance.
            callback_kwargs (dict): Any additional keyword arguments to be used by the callback.

        Returns:
            A Task that returns the callback function when the when the corresponding job is complete.
        """

        # Prevent re-submission of the same job
        if self.exitcode is not None:
            return

        # Define sbatch script and command
        self.script = script
        self.cmd = "sbatch --parsable"

        # Add job name if provided
        if self.job_name:
            self.cmd += f" --job-name {self.job_name}"

        # Define working directory
        dir = self.dir
        if use_tmp_dir:
            dir = self.tmp_dir
        dir.mkdir(mode=0o775, parents=True, exist_ok=True)

        # Add the working directory to the commaand
        self.cmd += f" --chdir {dir}"

        # Wait the command, if requested
        if wait:
            self.cmd += " --wait"

        # Run the command in a subprocess and wait for it to complete
        process = await asyncio.create_subprocess_shell(
            self.cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        # Ensure that stdin and stdout are not None before writing to them
        assert process.stdin is not None and process.stdout is not None

        # Write the script to stdin and wait for the process to complete
        process.stdin.write(self.script.encode())

        # Wait for the process to complete and capture the output
        process.stdin.write_eof()

        # Read the output and error streams
        self.exitcode = await process.wait()

        # Return the callback function
        return callback(self, **callback_kwargs)

    def sbatch(
        self,
        script: str,
        wait: bool = True,
        use_tmp_dir: bool = False,
        callback: Callable = _callback,
        callback_kwargs: dict = {},
    ):
        """
        Wraps `_sbatch` with default arguments, and stores the created Task.
        """

        self.task = self._sbatch(
            script=script,
            wait=wait,
            use_tmp_dir=use_tmp_dir,
            callback=callback,
            callback_kwargs=callback_kwargs,
        )

        return self
    
class ProgressBar(tqdm):
    """
    A custom progress bar for tracking `Job` completions, based on `tqdm`. 
    It is designed to be used in Jupyter notebooks and can be updated as jobs are completed.
    """

    def __init__(self, title: str) -> None:
        """
        Creates a placeholder progress bar.

        Parameters
            title (str): Title of the progress bar.
        """

        # Store instance variables
        self.title = title

        # Initialize the progress bar with custom settings
        super().__init__(
            desc=title,
            total=0,
            initial=1,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| Pending...",
            unit="job",
            colour='yellow',
        )

        # Try to access the HTML representation of the progress bar and set its style to default
        try:
            _, html, _ = self.container.children 
            html.bar_style = ""
        except Exception:
            pass

    def close_pbar(self) -> None:
        """
        Updates the status of the progress bar and closes it.
        """

        # Update the progress bar to indicate that all jobs have been completed
        if self.total == 0 and self.initial == 1:
            self.colour = "red"
            self.bar_format = "{l_bar}{bar}| No jobs allocated!"
        self.close()

    def update_total(self, n: int) -> int:
        """Updates the status of the progress bar and the total number of `Job`s.

        Parameters
            n (int): Number to increase the total by.

        Returns:
            The new total.
        """

        # Update the progress bar to indicate that jobs are being allocated
        if self.total == 0 and self.initial == 1:
            self.reset(0)
            self.initial = 0
            self.bar_format = (
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}+{remaining}, {rate_inv_fmt}]"
            )
            self.colour = None
            self.refresh()

        # Update the total number of jobs
        self.total += n

        # Try to access the HTML representation of the progress bar and set its maximum value to the new total
        try:
            _, html, _ = self.container.children
            html.max = self.total
        except Exception:
            pass

        # Refresh the progress bar to reflect the new total
        self.refresh()

        # Return the new total
        return self.total

def xtb_batch(
    inputs: list[str | Path],
    outputs: list[str | Path],
    slurm_name: str,
    job_dir: str | Path,
    tmp_dir: str | Path,
    log_dir: str | Path,
    charge: int,
    unpaired: int,
    solvent: str = "",
    temperature: float = 298.15,
    cpus: int = 4,
    mem_per_cpu: str = "1GB",
    partition: str = "PARTITION_NAME",
    account: str = None,
    time: str = "1:00:00",
    wait: bool = True,
    pbar: ProgressBar | None = None,
    scriptfile: str | Path = "",
    bashrc: str | Path = "",
    xtb: str | Path = "",
) -> Job:

    """
    A preconfigured function that runs many `xTB` jobs in batch. This is the preferred option to reduce the number of calls made to slurm.
    The function uses the `XTB1` and `XTB2` format strings. You can override this behaviour by providing your own script file.
    See the `XTB1` and `XTB2` strings to see the available placeholder variables.

    Parameters
        inputs (list[str | Path]): List of paths to the input `xyz` files.
        outputs (list[str | Path]): List of paths to the output files.
        slurm_name (str): Name of the Slurm job.
        job_dir (str | Path): Path to the working directory where the files are stored.
        tmp_dir (str | Path): Path to the scratch directory where the jobs are run.
        log_dir (str | Path): Path to the log directory where the Slurm output files are stored.
        charge (int): Charge of the molecule.
        unpaired (int): Number of unpaired electrons in the molecule (multiplicity-1).
        solvent (str): Solvent to use for the calculation.
        temperature (float): Temperature to use for the calculation.
        cpus (int): Number of CPUs allocated.
        mem_per_cpu (str): Memory allocated per CPU.
        partition (str): Slurm partition to use.
        account (str): Slurm account to use. If not provided, it will use the running user's default account.
        time (str): Maximum slurm time allowed.
        wait (bool): Whether the function should track job submission or job completion.
        pbar (ProgressBar | None): Progress bar to update on completion.
        scriptfile (str | Path): Path to an alternative script file, overriding the default script.
        bashrc (str | Path): Path to `.bashrc` for configuring Slurm environment.
        xtb (str | Path): Path to `xtb` binary.

    Returns:
        A Job containing a task for the Slurm job. If `wait=False`, the task will complete when
        the job is submitted, otherwise it will complete when the job is completed.
    """

    # Check xTB is executable
    check_executable(xtb)

    # Parse arguments as paths
    inputs = [Path(xyz) for xyz in inputs]
    outputs = [Path(output) for output in outputs]
    job_dir = Path(job_dir)
    tmp_dir = Path(tmp_dir)
    log_dir = Path(log_dir)
    scriptfile = Path(scriptfile)
    bashrc = Path(bashrc)
    xtb = Path(xtb)

    # Initialize an instance of the job object
    job = Job(slurm_name, job_dir, tmp_dir, log_dir)

    # Define a callback function to update the progress bar
    def callback(job: Job, pbar: ProgressBar) -> Job:
        if pbar is not None:
            pbar.update(1)
        return job

    # Initialize the sbatch script
    script = XTB1

    # Replace the script with the contents of the provided scriptfile if it exists
    if scriptfile.is_file():
        script = scriptfile.read_text()

    # Define the account command, if specified
    account_cmd = f"#SBATCH --account {account}" if account else ''

    # Initialize the sbatch script by replacing placeholders
    script = script.format(
        bashrc=bashrc,
        slurm_output=job.log_dir / "%j.out",
        cpus=cpus,
        mem_per_cpu=mem_per_cpu,
        partition=partition,
        account=account_cmd,
        time=time,
    )
    
    # Loop through the input files and add an xTB command to the sbatch script for each
    for xyz, output in zip(inputs, outputs):

        # Create the output directory
        job.dir = output.parent
        job.dir.mkdir(mode=0o775, parents=True, exist_ok=True)

        # Initialize the sbatch script by replacing placeholders
        script += XTB2.format(
            xtb=xtb,
            xyz=xyz,
            output=output,
            output_dir=job.dir,
            charge=charge,
            unpaired=unpaired,
            solvent=solvent,
            temperature=temperature,
        )

    # Update the progress bar total if it is not None
    if pbar is not None:
        pbar.update_total(1)

    # Write the sbatch script for debugging
    (job.dir / "xtb.sbatch").write_text(script)

    # Submit the job to Slurm and return the Job object
    return job.sbatch(
        script,
        wait=wait,
        callback=callback,
        callback_kwargs=dict(pbar=pbar),
    )

def crest(
    input: str | Path,
    output: str | Path,
    slurm_name: str,
    job_dir: str | Path,
    tmp_dir: str | Path,
    log_dir: str | Path,
    charge: int,
    unpaired: int,
    solvent: str = "",
    temperature: float = 298.15,
    constraints: str = "",
    cpus: int = 8,
    mem_per_cpu: str = "1GB",
    partition: str = "PARTITION_NAME",
    account: str = None,
    time: str = "10:00:00",
    wait: bool = True,
    pbar: ProgressBar | None = None,
    scriptfile: str | Path = "",
    bashrc: str | Path = "",
    crest: str | Path = "",
    xtb: str | Path = "",
) -> Job:
    """
    A preconfigured function that runs a single `CREST` job.
    The function uses the `CREST` format strings. You can override this behaviour by providing your own script file.
    See the `CREST` string to see the available placeholder variables.

    Parameters
        input (str | Path): Path to the input `xyz` file.
        output (str | Path): Path to the output file.
        slurm_name (str): Name of the Slurm job.
        job_dir (str | Path): Path to the working directory where the files are stored.
        tmp_dir (str | Path): Path to the scratch directory where the jobs are run.
        log_dir (str | Path): Path to the log directory where the Slurm output files are stored.
        charge (int): Charge of the molecule.
        unpaired (int): Number of unpaired electrons in the molecule (multiplicity-1).
        solvent (str): Solvent to use for the calculation.
        temperature (float): Temperature to use for the calculation.
        constraints (str): Path to a constraints file for the calculation.
        cpus (int): Number of CPUs allocated.
        mem_per_cpu (str): Memory allocated per CPU.
        partition (str): Slurm partition to use.
        account (str): Slurm account to use. If not provided, it will use the running user's default account.
        time (str): Maximum slurm time allowed.
        wait (bool): Whether the function should track job submission or job completion.
        pbar (ProgressBar | None): Progress bar to update on completion.
        scriptfile (str | Path): Path to an alternative script file, overriding the default script.
        bashrc (str | Path): Path to `.bashrc` for configuring Slurm environment.
        crest (str | Path): Path to `crest` binary.
        xtb (str | Path): Path to `xtb` binary.

    Returns:
        A Job containing a task for the Slurm job. If `wait=False`, the task will complete when
        the job is submitted, otherwise it will complete when the job is completed.
    """

    # Check CREST is executable
    check_executable(crest)

    # Parse arguments as paths
    input = Path(input)
    output = Path(output)
    job_dir = Path(job_dir)
    tmp_dir = Path(tmp_dir)
    log_dir = Path(log_dir)
    scriptfile = Path(scriptfile)
    bashrc = Path(bashrc)
    crest = Path(crest)
    xtb = Path(xtb)

    # Initialize an instance of the job object
    job = Job(slurm_name, job_dir, tmp_dir, log_dir)

    # Create the output directory
    job.dir = output.parent
    job.dir.mkdir(mode=0o775, parents=True, exist_ok=True)

    # Define a callback function to update the progress bar
    def callback(job: Job, pbar: ProgressBar) -> Job:
        if pbar is not None:
            pbar.update(1)
        return job

    # Initialize the SBATCH script
    script = CREST

    # Replace the script with the contents of the provided scriptfile if it exists
    if scriptfile.is_file():
        script = scriptfile.read_text()

    # Define the account command, if specified
    account_cmd = f"#SBATCH --account {account}" if account else ''

    # Initialize the SBATCH script by replacing placeholders
    script = script.format(
        xtb=xtb.parent,
        crest=crest,
        bashrc=bashrc,
        slurm_output=job.log_dir / "%j.out",
        xyz=input,
        output=output,
        charge=charge,
        unpaired=unpaired,
        solvent=solvent,
        temperature=temperature,
        constraints=constraints,
        cpus=cpus,
        mem_per_cpu=mem_per_cpu,
        partition=partition,
        account=account_cmd,
        time=time,
    )

    # Update the progress bar total if it is not None
    if pbar is not None:
        pbar.update_total(1)

    # Write the sbatch script for debugging
    (job.dir / "crest.sbatch").write_text(script)

    # Submit the job to Slurm and return the Job object
    return job.sbatch(
        script,
        wait=wait,
        callback=callback,
        callback_kwargs=dict(pbar=pbar),
    )

def censo(
    input: str | Path,
    output: str | Path,
    slurm_name: str,
    job_dir: str | Path,
    tmp_dir: str | Path,
    log_dir: str | Path,
    charge: int,
    unpaired: int,
    solvent: str = "",
    temperature: float = 298.15,
    cpus: int = 32,
    mem_per_cpu: str = "4GB",
    partition: str = "PARTITION_NAME",
    account: str = None,
    time: str = "48:00:00",
    parallel_calcs: int = 16,
    wait: bool = True,
    pbar: ProgressBar | None = None,
    scriptfile: str | Path = "",
    bashrc: str | Path = "",
    censo: str | Path = "",
    censorc: str | Path = "",
    cosmotherm: str | Path = "",
) -> Job | None:
    """
    A preconfigured function that runs a single `CENSO` job.
    You have to provide your own `.censorc` file to configure other CENSO execution parameters.
    The function uses the `CENSO` format strings. You can override this behaviour by providing your own script file.
    See the `CENSO` string to see the available placeholder variables.
    
    Parameters
        input (str | Path): Path to the input `xyz` file.
        output (str | Path): Path to the output file.
        slurm_name (str): Name of the Slurm job.
        job_dir (str | Path): Path to the working directory where the files are stored.
        tmp_dir (str | Path): Path to the scratch directory where the jobs are run.
        log_dir (str | Path): Path to the log directory where the Slurm output files are stored.
        charge (int): Charge of the molecule.
        unpaired (int): Number of unpaired electrons in the molecule (multiplicity-1).
        solvent (str): Solvent to use for the calculation.
        temperature (float): Temperature to use for the calculation.
        cpus (int): Number of CPUs allocated.
        mem_per_cpu (str): Memory allocated per CPU.
        partition (str): Slurm partition to use.
        account (str): Slurm account to use. If not provided, it will use the running user's default account.
        time (str): Maximum slurm time allowed.
        parallel_calcs (int): Number of parallel calculations to run. Must divide into `cpus`.
        wait (bool): Whether the function should track job submission or job completion.
        pbar (ProgressBar | None): Progress bar to update on completion.
        scriptfile (str | Path): Path to an alternative script file, overriding the default script.
        bashrc (str | Path): Path to `.bashrc` for configuring Slurm environment.
        censo (str | Path): Path to `censo` binary.
        censorc (str | Path): Path to `.censorc` file for configuring CENSO execution parameters.
        cosmotherm (str | Path): `COSMOtherm` package to load.

    Returns:
        A Job containing a task for the Slurm job. If `wait=False`, the task will complete when
        the job is submitted, otherwise it will complete when the job is completed.
        If `parallel_calcs` does not divide into `cpus`, don't do anything.
    """

    # Check CENSO is executable
    check_executable(censo)

    # Ensure that the number of CPUs is divisible by the number of parallel calculations
    if cpus % parallel_calcs != 0:
        raise ValueError(f"The number of CPUs allocated ({cpus}) is not divisible by the number of parallel calculations ({parallel_calcs}), skipping.")

    # Parse arguments as paths
    input = Path(input)
    output = Path(output)
    job_dir = Path(job_dir)
    tmp_dir = Path(tmp_dir)
    log_dir = Path(log_dir)
    scriptfile = Path(scriptfile)
    bashrc = Path(bashrc)
    censo = Path(censo)
    censorc = Path(censorc)
    cosmotherm = Path(cosmotherm)

    # Initialize an instance of the job object
    job = Job(slurm_name, job_dir, tmp_dir, log_dir)

    # Create the output directory
    job.dir = output.parent
    job.dir.mkdir(mode=0o775, parents=True, exist_ok=True)

    # Define a callback function to update the progress bar
    def callback(job: Job, pbar: ProgressBar) -> Job:
        if pbar is not None:
            pbar.update(1)

        # Copy CENSO files to the job directory
        for enso_ensemble in job.tmp_dir.glob(r"**/enso_ensemble_part[1-3].xyz"):
            try:
                shutil.copy2(enso_ensemble, job.dir)
            except Exception:
                pass

        return job

    # Initialize the SBATCH script
    script = CENSO

    # Replace the script with the contents of the provided scriptfile if it exists
    if scriptfile.is_file():
        script = scriptfile.read_text()

    # Define the account command, if specified
    account_cmd = f"#SBATCH --account {account}" if account else ''

    # Initialize the SBATCH script by replacing placeholders
    script = script.format(
        censo=censo,
        cosmotherm=cosmotherm,
        bashrc=bashrc,
        censorc=censorc,
        slurm_output=job.log_dir / "%j.out",
        xyz=input,
        output=output,
        charge=charge,
        unpaired=unpaired,
        solvent=solvent,
        temperature=temperature,
        cpus=cpus,
        parallel_calcs=parallel_calcs,
        cpus_per_calc=int(cpus / parallel_calcs),
        mem_per_cpu=mem_per_cpu,
        partition=partition,
        account=account_cmd,
        time=time,
    )

    # Update the progress bar total if it is not None
    if pbar is not None:
        pbar.update_total(1)

    # Write the sbatch script for debugging
    (job.dir / "censo.sbatch").write_text(script)

    # Submit the job to Slurm and return the Job object
    return job.sbatch(
        script,
        wait=wait,
        use_tmp_dir=True,
        callback=callback,
        callback_kwargs=dict(pbar=pbar),
    )

def gaussian(
    input: str | Path,
    output: str | Path,
    slurm_name: str,
    job_dir: str | Path,
    tmp_dir: str | Path,
    log_dir: str | Path,
    cpus: int = 4,
    mem: str = "6000MB",
    partition: str = "PARTITION_NAME",
    account: str = None,
    time: str = "20:00:00",
    wait: bool = True,
    pbar: ProgressBar | None = None,
    scriptfile: str | Path = "",
    bashrc: str | Path = "",
    gauss: str | Path = ""
) -> Job:
    """
    A preconfigured function that runs a single `Gaussian` job.
    Input parameters are passed to `Gaussian` through the `gjf` file, which should be populated in advance.
    The function uses the `Gaussian` format strings. You can override this behaviour by providing your own script file.
    See the `Gaussian` string to see the available placeholder variables.
    
    Parameters
        input (str | Path): Path to the input `gjf` file.
        output (str | Path): Path to the output file.
        slurm_name (str): Name of the Slurm job.
        job_dir (str | Path): Path to the working directory where the files are stored.
        tmp_dir (str | Path): Path to the scratch directory where the jobs are run.
        log_dir (str | Path): Path to the log directory where the Slurm output files are stored.
        cpus (int): Number of CPUs allocated.
        mem (str): Memory allocated per CPU.
        partition (str): Slurm partition to use.
        account (str): Slurm account to use. If not provided, it will use the running user's default account.
        time (str): Maximum slurm time allowed.
        wait (bool): Whether the function should track job submission or job completion.
        pbar (ProgressBar | None): Progress bar to update on completion.
        scriptfile (str | Path): Path to an alternative script file, overriding the default script.
        bashrc (str | Path): Path to `.bashrc` for configuring Slurm environment.
        gauss (str | Path): Path to `Gaussian` binary.

    Returns:
        A Job containing a task for the Slurm job. If `wait=False`, the task will complete when
        the job is submitted, otherwise it will complete when the job is completed.
    """

    # Check Gaussian is executable
    check_executable(gauss)

    # Parse arguments as paths
    input = Path(input)
    output = Path(output)
    job_dir = Path(job_dir)
    tmp_dir = Path(tmp_dir)
    log_dir = Path(log_dir)
    scriptfile = Path(scriptfile)
    bashrc = Path(bashrc)
    gauss = Path(gauss)

    # Initialize an instance of the job object
    job = Job(slurm_name, job_dir, tmp_dir, log_dir)

    # Create the output directory
    job.dir = output.parent
    job.dir.mkdir(mode=0o775, parents=True, exist_ok=True)

    # Define a callback function to update the progress bar
    def callback(job: Job, pbar: ProgressBar) -> Job:
        if pbar is not None:
            pbar.update(1)
        return job

    # Initialize the SBATCH script
    script = GAUSS

    # Replace the script with the contents of the provided scriptfile if it exists
    if scriptfile.is_file():
        script = scriptfile.read_text()

    # Define the account command, if specified
    account_cmd = f"#SBATCH --account {account}" if account else ''

    # Initialize the SBATCH script by replacing placeholders
    script = script.format(
        gauss=gauss,
        bashrc=bashrc,
        slurm_output=job.log_dir / "%j.out",
        gjf=input,
        output=output,
        cpus=cpus,
        mem=mem,
        partition=partition,
        account=account_cmd,
        time=time,
    )

    # Update the progress bar total if it is not None
    if pbar is not None:
        pbar.update_total(1)

    # Write the sbatch script for debugging
    (job.dir / "gaussian.sbatch").write_text(script)

    # Submit the job to Slurm and return the Job object
    return job.sbatch(
        script,
        wait=wait,
        callback=callback,
        callback_kwargs=dict(pbar=pbar),
    )
