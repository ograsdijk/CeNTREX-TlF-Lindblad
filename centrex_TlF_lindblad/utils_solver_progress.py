from typing import Optional, Union, List
import numpy as np
import numpy.typing as npt
from julia import Main

__all__ = ["solve_problem_parameter_scan_progress"]


def solve_problem_parameter_scan_progress(
    method: str = "Tsit5()",
    distributed_method: str = "EnsembleDistributed()",
    abstol: float = 1e-7,
    reltol: float = 1e-4,
    save_everystep: bool = True,
    callback: Optional[str] = None,
    problem_name: str = "prob",
    ensemble_problem_name: str = "ens_prob",
    trajectories: Optional[int] = None,
    output_func: Optional[str] = None,
    saveat: Optional[Union[List[float], npt.NDArray[np.float_]]] = None,
):
    _trajectories = "size(params)[1]" if trajectories is None else trajectories
    _callback = "nothing" if callback is None else callback
    _saveat = "[]" if saveat is None else str(saveat)

    if output_func is None:
        Main.eval(
            """
            @everywhere function output_func_progress(sol, i)
                put!(channel, 1)
                sol, false
            end
        """
        )
    else:
        Main.eval(
            f"""
            @everywhere function output_func_progress(sol, i)
                put!(channel, 1)
                a,b = {output_func}(sol, i)
                return a,b
            end
        """
        )
    Main.eval(
        f"""
        {ensemble_problem_name} = EnsembleProblem({problem_name},
                                                prob_func = prob_func,
                                                output_func = output_func_progress
                                            )
    """
    )

    Main.eval(
        """
        if !@isdefined channel
            const channel = RemoteChannel(()->Channel{Int}(1))
            @everywhere const channel = $channel
        end
    """
    )

    Main.eval(
        f"""
        progress = Progress({_trajectories}, showspeed = true)
        @sync sol = begin
            @async begin
                tasksdone = 0
                while tasksdone < {_trajectories}
                    tasksdone += take!(channel)
                    update!(progress, tasksdone)
                end
            end
            @async begin
                @time global sol = solve({ensemble_problem_name}, {method},
                            {distributed_method}, trajectories={_trajectories},
                            abstol = {abstol}, reltol = {reltol},
                            callback = {_callback},
                            save_everystep = {str(save_everystep).lower()},
                            saveat = {_saveat})
            end
    end
    """
    )
