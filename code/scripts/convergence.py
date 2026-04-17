import argparse

import numpy as np
import pandas as pd

from geometric_fv import schemes
from geometric_fv.config import (
    BoundaryConfig,
    IterationConfig,
    MeshConfig,
    ReconstConfig,
    SolverConfig,
)
from geometric_fv.enums import BCType, LimiterType, SlopeType
from geometric_fv.experiments import STUDY_REGISTRY
from geometric_fv.problems import BurgersSmooth
from geometric_fv.utils import calculate_norms


def run_experiment(name: str, dt_dx_override: float = None, t_final: float = 0.2):
    """Executes the full convergence study for a single named experiment."""
    experiment = STUDY_REGISTRY[name]
    dt_dx_target = dt_dx_override if dt_dx_override is not None else experiment.default_dt_dx

    resolutions = [50 * 2**n for n in range(4)]

    prob = BurgersSmooth()
    print(f"Shock formation time: {prob.t_shock:.4f}")

    # Calculate synchronized refinement parameters
    ncells_base = resolutions[0]
    dx_base = (prob.x_max - prob.x_min) / ncells_base
    dt_base = dt_dx_target * dx_base
    nsteps_base = int(np.round(t_final / dt_base))

    results = []

    print(f"Experiment: {name} | dt/dx_target={dt_dx_target} | t={t_final}")

    for ncells in resolutions:
        # Calculate refinement factor relative to the base resolution
        refine_factor = ncells // ncells_base
        nsteps = nsteps_base * refine_factor

        # Define mesh
        mesh_cfg = MeshConfig.from_problem(prob, ncells=ncells)
        mesh = mesh_cfg.create_mesh()
        dx = mesh.dx[0]

        # Deerive dt to maintain constant dt/dx
        dt_actual = t_final / nsteps
        dt_dx_actual = dt_actual / dx

        config = SolverConfig(
            mesh=mesh_cfg,
            boundary=BoundaryConfig(bc_type=BCType.CONSTANT_EXTEND),
            reconst=ReconstConfig(**experiment.reconst_kwargs),
            iteration=IterationConfig(tol=1e-10, maxiter=50),
            equation=prob.equation,
            dt_dx=dt_dx_actual,
        )

        scheme = experiment.scheme_class(config=config)
        state = scheme.init_state(prob.u0)

        print(f"Running ncells={ncells:4d}, dt_dx={dt_dx_actual:.4f}, nsteps={nsteps}")

        # 4. Time Integration Loop
        for _ in range(nsteps):
            scheme.apply_bc(state)
            scheme.sweep(state)
            state.u_old[:] = state.u_new[:]

        # 5. Error Calculation (Outside the loop!)
        u_numerical = state.u_new[scheme.nghost : -scheme.nghost]
        u_exact = prob.exact(mesh.centers, t_final)

        errors = calculate_norms(u_numerical, u_exact, dx)
        errors["ncells"] = ncells
        errors["dx"] = dx
        results.append(errors)

    # Analysis
    df = pd.DataFrame(results)

    # Calculate Convergence Order: log(E_i / E_{i+1}) / log(dx_i / dx_{i+1})
    for norm in ["L1", "L2", "Linf"]:
        err = df[norm].values
        dxs = df["dx"].values
        order = np.log(err[1:] / err[:-1]) / np.log(dxs[1:] / dxs[:-1])
        df[f"Order_{norm}"] = np.concatenate([[np.nan], order])

    print(f"\n--- Convergence Results ({name}, t={t_final}) ---")
    cols = ["ncells", "L1", "Order_L1", "L2", "Order_L2", "Linf", "Order_Linf"]
    print(
        df[cols].to_string(
            index=False, float_format=lambda x: f"{x:.2e}" if x < 1e-1 else f"{x:.2f}"
        )
    )

    # --- Save Results ---
    # The 'results/' directory should be created by the Makefile
    csv_path = f"results/conv_{name}_cfl{dt_dx_target}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV results saved to '{csv_path}'")
    return df


def run_study():
    parser = argparse.ArgumentParser(description="Run convergence study.")
    parser.add_argument(
        "--experiment",
        nargs="+",
        default="burgers-implup",
        choices=list(STUDY_REGISTRY.keys()) + ["all"],
        help="Experiment name(s) from STUDY_REGISTRY or 'all'",
    )
    parser.add_argument("--dt_dx", type=float, default=None, help="Override dt/dx ratio")
    parser.add_argument(
        "--t_final", type=float, default=0.2, help="Final simulation time"
    )
    args = parser.parse_args()

    names = args.experiment
    if "all" in names:
        names = list(STUDY_REGISTRY.keys())

    for name in names:
        run_experiment(name, args.dt_dx, args.t_final)

if __name__ == "__main__":
    run_study()
