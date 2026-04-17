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


def run_study():
    parser = argparse.ArgumentParser(description="Run convergence study.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="burgers-implup",
        choices=list(STUDY_REGISTRY.keys()),
        help="Experiment name from STUDY_REGISTRY",
    )
    parser.add_argument("--dt_dx", type=float, default=None, help="Override dt/dx ratio")
    parser.add_argument(
        "--t_final", type=float, default=0.2, help="Final simulation time"
    )
    args = parser.parse_args()

    # Experiment selection
    experiment = STUDY_REGISTRY[args.experiment]
    dt_dx_target = args.dt_dx if args.dt_dx is not None else experiment.default_dt_dx

    x_min = 0.0
    x_max = 1.0
    resolutions = [50 * 2**n for n in range(6)]

    # Calculate synchronized refinement parameters
    ncells_base = resolutions[0]
    dx_base = (x_max - x_min) / ncells_base
    # Rounding base steps to the nearest integer to get as close as possible to
    # dt_dx_target
    dt_base = dt_dx_target * dx_base
    nsteps_base = int(np.round(args.t_final / dt_base))

    prob = BurgersSmooth(x_min=x_min, x_max=x_max)
    print(f"Shock formation time: {prob.t_shock:.4f}")

    results = []

    print(f"Experiment: {args.experiment} | dt/dx_target={dt_dx_target} | t={args.t_final}")

    for ncells in resolutions:
        # Calculate refinement factor relative to the base resolution
        refine_factor = ncells // ncells_base
        nsteps = nsteps_base * refine_factor

        # Define mesh
        mesh_cfg = MeshConfig(x_min=x_min, x_max=x_max, ncells=ncells)
        mesh = mesh_cfg.create_mesh()
        dx = mesh.dx[0]

        # Deerive dt to maintain constant dt/dx
        dt_actual = args.t_final / nsteps
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
        u_exact = prob.exact(mesh.centers, args.t_final)

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

    print(f"\n--- Convergence Results ({args.experiment}, t={args.t_final}) ---")
    cols = ["ncells", "L1", "Order_L1", "L2", "Order_L2", "Linf", "Order_Linf"]
    print(
        df[cols].to_string(
            index=False, float_format=lambda x: f"{x:.2e}" if x < 1e-1 else f"{x:.2f}"
        )
    )

    # --- Save Results ---
    # The 'results/' directory should be created by the Makefile
    csv_path = f"results/conv_{args.experiment}_cfl{dt_dx_target}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV results saved to '{csv_path}'")


if __name__ == "__main__":
    run_study()
