from dataclasses import dataclass
from numbers import Number
from typing import List, Literal, Optional, Sequence, Tuple, Union, overload

import numpy as np
import numpy.typing as npt
import sympy as smp
from julia import Main

from .ode_parameters import odeParameters

numeric = Union[int, float, complex]

__all__ = [
    "setup_initial_condition_scan",
    "setup_parameter_scan_1D",
    "setup_ratio_calculation",
    "setup_state_integral_calculation",
    "setup_parameter_scan_ND",
    "setup_discrete_callback_terminate",
    "setup_problem",
    "solve_problem",
    "get_results",
    "do_simulation_single",
    "setup_problem_parameter_scan",
    "solve_problem_parameter_scan",
    "get_results_parameter_scan",
]


def setup_initial_condition_scan(
    values: Union[List[Number], npt.NDArray[Union[np.int_, np.float_, np.complex_]]]
) -> None:
    Main.params = values
    Main.eval("@everywhere params = $params")
    Main.eval(
        """
    @everywhere function prob_func(prob,i,repeat)
        remake(prob,u0=params[i])
    end
    """
    )


def setup_parameter_scan_1D(
    odePar: odeParameters,
    parameters: Union[str, List[str]],
    values: Union[List[Number], npt.NDArray[Union[np.int_, np.float_, np.complex_]]],
) -> None:
    """
    Convenience function for setting up a 1D parameter scan.
    Scan can be performed over multiple parameters simultaneously,
    but only for the same value for each parameter. For different
    values per parameter see setup_parameter_scan_zipped.

    Args:
        odePar (odeParameters): object containing all the parameters
                                for the ODE system
        parameters (str, list): parameter or list of parameters to
                                scan over
        values (list, np.ndarray): values to scan the parameter(s)
                                    over.
    """
    # check if parameters is a list, get indices of the parameters
    # as defined in odePar
    if isinstance(parameters, (list, tuple)):
        indices = [odePar.get_index_parameter(par) for par in parameters]
    else:
        indices = [odePar.get_index_parameter(parameters)]

    # generate the parameter sequence for the prob_func function
    pars = list(odePar.p)
    for idx in indices:
        pars[idx] = "params[i]"
    _pars = "[" + ",".join([str(p) for p in pars]) + "]"

    # generate prob_func which remakes the ODE problem
    # for each different parameter set
    setup_initial_condition_scan(values)
    Main.eval(
        f"""
    @everywhere function prob_func(prob,i,repeat)
        remake(prob, p = {_pars})
    end
    """
    )


def setup_parameter_scan_zipped(
    odePar: odeParameters,
    parameters: Union[str, List[str]],
    values: Union[List[Number], npt.NDArray[Union[np.int_, np.float_, np.complex_]]],
) -> None:
    """
    Convenience function for initializing a 1D parameter scan over
    multiple parameters, with each parameter scanning over a different
    set of parameters.

    Args:
        odePar (odeParameters): object containing all the parameters
                                for the OBE system.
        parameters (list): list of parameters to scan over
        values (list, np.ndarray): list/array of values to scan over.
    """
    # get the indices of each parameter that is scanned over,
    # as defined in odePars. If a parameter is not scanned over,
    # use the variable definition
    pars = list(odePar.p)
    for idN, parameter in enumerate(parameters):
        if isinstance(parameter, (list, tuple)):
            indices = [odePar.get_index_parameter(par) for par in parameter]
        else:
            indices = [odePar.get_index_parameter(parameter)]
        for idx in indices:
            pars[idx] = f"params[i,{idN+1}]"
    params = np.array(list(zip(*values)))

    _pars = "[" + ",".join([str(p) for p in pars]) + "]"

    # generate prob_func which remakes the ODE problem for
    # each different parameter set
    setup_initial_condition_scan(params)
    Main.eval(
        f"""
    @everywhere function prob_func(prob, i, repeat)
        remake(prob, p = {_pars})
    end
    """
    )


def setup_parameter_scan_ND(
    odePar: odeParameters,
    parameters: Union[str, List[str]],
    values: Union[List[Number], npt.NDArray[Union[np.int_, np.float_, np.complex_]]],
) -> None:
    """
    Convenience function for generating an ND parameter scan.
    For each parameter a list or np.ndarray of values is supplied,
    and each possible combination between all parameters is simulated.

    Args:
        odePar (odeParameters): object containing all the parameters for
                                the OBE system.
        parameters (list, np.ndarray): strs of parameters to scan over.
        values (list, np.ndarray): list or np.ndarray of values to scan over
                                    for each parameter
    """
    # create all possible combinations between parameter values with
    # meshgrid
    params = np.array(np.meshgrid(*values)).T.reshape(-1, len(values))

    setup_parameter_scan_zipped(odePar, parameters, params.T)


def setup_ratio_calculation(
    states: Union[Sequence[int], Sequence[Sequence[int]]],
    output_func: Optional[str] = None,
) -> str:
    if output_func is None:
        output_func = "output_func"
    cmd = ""
    if isinstance(states[0], (list, np.ndarray, tuple)):
        for state in states:
            cmd += (
                f"sum(real(diag(sol.u[end])[{state}]))/"
                f"sum(real(diag(sol.u[1])[{state}])), "
            )
        cmd = cmd.strip(", ")
        cmd = "[" + cmd + "]"
    else:
        cmd = (
            f"sum(real(diag(sol.u[end])[{states}]))/sum(real(diag(sol.u[1])[{states}]))"
        )

    Main.eval(
        f"""
    @everywhere function {output_func}(sol,i)
        if size(sol.u)[1] == 1
            return NaN, false
        else
            val = {cmd}
            return val, false
        end
    end"""
    )
    return output_func


def setup_state_integral_calculation(
    states: Sequence[int],
    output_func: Optional[str] = None,
    nphotons: bool = False,
    Γ: Optional[float] = None,
) -> str:
    """Setup an integration output_function for an EnsembleProblem.
    Uses trapezoidal integration to integrate the states.

    Args:
        states (list): list of state indices to integrate
        nphotons (bool, optional): flag to calculate the number of photons,
                                    e.g. normalize with Γ
        Γ (float, optional): decay rate in 2π Hz (rad/s), not necessary if already
                                loaded into Julia globals
    """
    if output_func is None:
        _output_func = "output_func"
    else:
        _output_func = output_func
    if nphotons & Main.eval("@isdefined Γ"):
        Main.eval(
            f"""
        @everywhere function {_output_func}(sol,i)
            return Γ.*trapz(sol.t, [real(sum(diag(sol.u[j])[{states}])) for j in 1:size(sol)[3]]), false
        end"""
        )
    else:
        if nphotons:
            assert (
                Γ is not None
            ), "Γ not defined as a global in Julia and not supplied to function"
            Main.eval(f"@everywhere Γ = {Γ}")
            Main.eval(
                f"""
            @everywhere function {_output_func}(sol,i)
                return {Γ}.*trapz(sol.t, [real(sum(diag(sol.u[j])[{states}])) for j in 1:size(sol)[3]]), false
            end"""
            )
        else:
            Main.eval(
                f"""
            @everywhere function {_output_func}(sol,i)
                return trapz(sol.t, [real(sum(diag(sol.u[j])[{states}])) for j in 1:size(sol)[3]]), false
            end"""
            )
    return _output_func


def setup_discrete_callback_terminate(
    odepars: odeParameters, stop_expression: str, callback_name: Optional[str] = None
) -> str:
    # parse expression string to sympy equation
    expression = smp.parsing.sympy_parser.parse_expr(stop_expression)
    # extract symbols in expression and convert to a list of strings
    symbols_in_expression = list(expression.free_symbols)
    symbols_in_expression = [str(sym) for sym in symbols_in_expression]
    # check if all symbols are parameters of the ODE
    odepars.check_symbols_in_parameters(symbols_in_expression)

    # remove t
    symbols_in_expression.remove("t")
    # get indices of symbols
    indices = [
        odepars.get_index_parameter(sym, mode="julia") for sym in symbols_in_expression
    ]
    for idx, sym in zip(indices, symbols_in_expression):
        stop_expression = stop_expression.replace(str(sym), f"integrator.p[{idx}]")
    if callback_name is None:
        _callback_name = "cb"
    else:
        _callback_name = callback_name
    Main.eval(
        f"""
        @everywhere condition(u,t,integrator) = {stop_expression}
        @everywhere affect!(integrator) = terminate!(integrator)
        {_callback_name} = DiscreteCallback(condition, affect!)
    """
    )
    return _callback_name


def setup_problem(
    odepars: odeParameters,
    tspan: Union[List[float], Tuple[float]],
    ρ: npt.NDArray[np.complex_],
    problem_name="prob",
) -> None:
    odepars.generate_p_julia()
    Main.ρ = ρ
    Main.tspan = tspan
    assert Main.eval(
        "@isdefined Lindblad_rhs!"
    ), "Lindblad function is not defined in Julia"
    Main.eval(
        f"""
        {problem_name} = ODEProblem(Lindblad_rhs!,ρ,tspan,p)
    """
    )


def setup_problem_parameter_scan(
    odepars: odeParameters,
    tspan: List[float],
    ρ: npt.NDArray[np.complex_],
    parameters: List[str],
    values: Union[List[Number], npt.NDArray[Union[np.int_, np.float_, np.complex_]]],
    dimensions: int = 1,
    problem_name: str = "prob",
    output_func: Optional[str] = None,
    zipped: bool = False,
) -> str:
    setup_problem(odepars, tspan, ρ, problem_name)
    if dimensions == 1:
        if zipped:
            setup_parameter_scan_zipped(odepars, parameters, values)
        else:
            setup_parameter_scan_1D(odepars, parameters, values)
    else:
        setup_parameter_scan_ND(odepars, parameters, values)
    if output_func is not None:
        Main.eval(
            f"""
            ens_{problem_name} = EnsembleProblem({problem_name},
                                                    prob_func = prob_func,
                                                    output_func = {output_func}
                                                )
        """
        )
    else:
        Main.eval(
            f"""
            ens_{problem_name} = EnsembleProblem({problem_name},
                                                    prob_func = prob_func)
        """
        )
    return f"ens_{problem_name}"


def solve_problem(
    method: str = "Tsit5()",
    abstol: float = 1e-7,
    reltol: float = 1e-4,
    dt: float = 1e-8,
    callback: Optional[str] = None,
    problem_name: str = "prob",
    progress: bool = False,
    saveat: Optional[Union[List[float], npt.NDArray[np.float_]]] = None,
    dtmin: Optional[int] = None,
    maxiters: int = 100_000,
) -> None:
    force_dtmin = "false" if dtmin is None else "true"
    _dtmin = "nothing" if dtmin is None else str(dtmin)
    _saveat = saveat if saveat is not None else "[]"

    if callback is not None:
        Main.eval(
            f"""
            sol = solve({problem_name}, {method}, abstol = {abstol},
                        reltol = {reltol}, dt = {dt},
                        progress = {str(progress).lower()},
                        callback = {callback}, saveat = {_saveat},
                        dtmin = {_dtmin}, maxiters = {maxiters},
                        force_dtmin = {force_dtmin}
                    )
        """
        )
    else:
        Main.eval(
            f"""
            sol = solve({problem_name}, {method}, abstol = {abstol},
                        reltol = {reltol}, dt = {dt},
                        progress = {str(progress).lower()}, saveat = {_saveat},
                        dtmin = {_dtmin}, maxiters = {maxiters},
                        force_dtmin = {force_dtmin}
                    )
        """
        )


def solve_problem_parameter_scan(
    method: str = "Tsit5()",
    distributed_method: str = "EnsembleDistributed()",
    abstol: float = 1e-7,
    reltol: float = 1e-4,
    dt: float = 1e-8,
    save_everystep: bool = True,
    callback: Optional[str] = None,
    ensemble_problem_name: str = "ens_prob",
    trajectories: Optional[int] = None,
    saveat: Optional[Union[List[float], npt.NDArray[np.float_]]] = None,
) -> None:
    _trajectores = "size(params)[1]" if trajectories is None else str(trajectories)
    _saveat = "[]" if saveat is None else str(saveat)
    if callback is not None:
        Main.eval(
            f"""
            sol = solve({ensemble_problem_name}, {method}, {distributed_method},
                        abstol = {abstol}, reltol = {reltol}, dt = {dt},
                        trajectories = {_trajectores}, callback = {callback},
                        save_everystep = {str(save_everystep).lower()},
                        saveat = {_saveat}
                    )
        """
        )
    else:
        Main.eval(
            f"""
            sol = solve({ensemble_problem_name}, {method}, {distributed_method},
                        abstol = {abstol}, reltol = {reltol}, dt = {dt},
                        trajectories = {_trajectores},
                        save_everystep = {str(save_everystep).lower()},
                        saveat = {_saveat}
                    )
        """
        )


@dataclass
class OBEResult:
    t: npt.NDArray[np.float_]
    y: npt.NDArray[np.complex_]


def get_results() -> OBEResult:
    """Retrieve the results of a single trajectory OBE simulation solution.

    Returns:
        tuple: tuple containing the timestamps and an n x m numpy arra, where
                n is the number of states, and m the number of timesteps
    """
    results = np.real(np.einsum("jji->ji", np.array(Main.eval("sol[:]")).T))
    t = Main.eval("sol.t")
    return OBEResult(t, results)


@overload
def get_results_parameter_scan(scan_values: Literal[None]) -> npt.NDArray[np.complex_]:
    ...


@overload
def get_results_parameter_scan() -> npt.NDArray[np.complex_]:
    ...


@overload
def get_results_parameter_scan(
    scan_values: Union[
        List[Number],
        npt.NDArray[np.int_],
        npt.NDArray[np.float_],
        npt.NDArray[np.complex_],
    ]
) -> Tuple[
    npt.NDArray[np.complex_], npt.NDArray[np.complex_], npt.NDArray[np.complex_]
]:
    ...


def get_results_parameter_scan(
    scan_values: Optional[
        Union[
            List[Number],
            npt.NDArray[np.int_],
            npt.NDArray[np.float_],
            npt.NDArray[np.complex_],
        ]
    ] = None
) -> Union[
    npt.NDArray[np.complex_],
    Tuple[npt.NDArray[np.complex_], npt.NDArray[np.complex_]],
    Tuple[npt.NDArray[np.complex_], npt.NDArray[np.complex_], npt.NDArray[np.complex_]],
]:
    results = np.array(Main.eval("sol.u"))
    if scan_values is not None:
        _scan_values = scan_values
        if isinstance(_scan_values[0], (list, np.ndarray)):
            # this will always give a valid length for len(val) but mypy throws an error
            results = results.reshape([len(val) for val in _scan_values])  # type: ignore
            X, Y = np.meshgrid(*_scan_values)
            return X, Y, results.T
        else:
            return scan_values, results
    else:
        return results


def do_simulation_single(
    odepars: odeParameters,
    tspan: Union[List[float], Tuple[float]],
    ρ: npt.NDArray[np.complex_],
    terminate_expression: Optional[str] = None,
    dt: float = 1e-8,
    saveat: Optional[Union[List[float], npt.NDArray[np.float_]]] = None,
    dtmin: Optional[int] = None,
    maxiters: int = 100_000,
) -> OBEResult:
    """Perform a single trajectory solve of the OBE equations for a specified
    TlF system.

    Args:
        odepars (odeParameters): object containing the ODE parameters used in
        the solver
        tspan (list, tuple): time range to solve for
        terminate_expression (str, optional): Expression that determines when to
                                            stop integration. Defaults to None.
        saveat (array or float, optional): save solution at timesteps given by
                                            saveat, either a list or every
                                            saveat
        dtmin (float, optional): minimum dt allowed for adaptive timestepping
        maxiters (float, optional): maximum number of steps allowed

    Returns:
        tuple: tuple containing the timestamps and an n x m numpy array, where
                n is the number of states, and m the number of timesteps
    """
    callback_flag = False
    if terminate_expression is not None:
        setup_discrete_callback_terminate(odepars, terminate_expression)
        callback_flag = True
    setup_problem(odepars, tspan, ρ)
    if callback_flag:
        solve_problem(
            callback="cb", saveat=saveat, dtmin=dtmin, dt=dt, maxiters=maxiters
        )
    else:
        solve_problem(saveat=saveat, dtmin=dtmin, dt=dt, maxiters=maxiters)
    return get_results()
