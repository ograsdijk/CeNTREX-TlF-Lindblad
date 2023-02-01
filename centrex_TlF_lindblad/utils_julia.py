from pathlib import Path
from typing import List

from julia import Main

__all__ = ["initialize_julia", "generate_ode_fun_julia"]


def initialize_julia(nprocs: int, verbose: bool = True):
    """
    Function to initialize Julia over nprocs processes.
    Creates nprocs processes and loads the necessary Julia
    packages.

    Args:
        nprocs (int): number of Julia processes to initialize.
    """
    Main.eval(
        """
        using Logging: global_logger
        using TerminalLoggers: TerminalLogger
        global_logger(TerminalLogger())

        using Distributed
        using ProgressMeter
    """
    )

    if Main.eval("nprocs()") < nprocs:
        Main.eval(f"addprocs({nprocs}-nprocs())")

    if Main.eval("nprocs()") > nprocs:
        procs = Main.eval("procs()")
        procs = procs[nprocs:]
        Main.eval(f"rmprocs({procs})")

    Main.eval(
        """
        @everywhere begin
            using LinearAlgebra
            using Trapz
            using DifferentialEquations
        end
    """
    )
    # loading common julia functions from julia_common.jl
    path = Path(__file__).parent / "julia_common.jl"
    Main.eval(f'include(raw"{path}")')

    if verbose:
        print(f"Initialized Julia with {nprocs} processes")


def generate_ode_fun_julia(preamble: str, code_lines: List[str]) -> str:
    """
    Generate the ODE function from the preamble and code lines
    generated in Python.

    Args:
        preamble (str): preamble of the ODE function initializing the
                        function variable definitions.
        code_lines (list): list of strings, each line is a generated
                            line of Julia code for part of the ODE.

    Returns:
        str : function definition of the ODE
    """
    ode_fun = preamble
    for cline in code_lines:
        ode_fun += "\t\t" + cline + "\n"
    ode_fun += "\t end \n \t nothing \n end"
    Main.eval(f"@everywhere {ode_fun}")
    return ode_fun
