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
from geometric_fv.problems import BurgersSmooth
from geometric_fv.utils import calculate_norms


def run_study():
    parser = argparse.ArgumentParser(description="Run convergence study.")
    parser.add_argument(
        "--scheme", type=str, default="Lozano", help="Scheme class name"
    )
    parser.add_argument("--dt_dx", type=float, default=0.8, help="Target dt/dx ratio")
    parser.add_argument(
        "--t_final", type=float, default=0.2, help="Final simulation time"
    )
    args = parser.parse_args()

    # 1. Dynamic Scheme Selection
    # This looks up the class in the 'schemes' module by its string name
    try:
        SchemeClass = getattr(schemes, args.scheme)
    except AttributeError:
        available = [s for s in dir(schemes) if not s.startswith("_")]
        raise ValueError(f"Scheme '{args.scheme}' not found. Available: {available}")

    x_min = 0.0
    x_max = 1.0
    resolutions = [50 * 2**n for n in range(4)]

    prob = BurgersSmooth(x_min=x_min, x_max=x_max)
    print(f"Shock formation time: {prob.t_shock:.4f}")

    results = []

    print(f"Scheme: {args.scheme} | dt/dx={args.dt_dx} | t={args.t_final}")

    for ncells in resolutions:
        # Define mesh
        mesh_cfg = MeshConfig(x_min=x_min, x_max=x_max, ncells=ncells)
        mesh = mesh_cfg.create_mesh()
        dx = mesh.dx[0]

        # Adjust dt
        dt_ideal = args.dt_dx * dx
        nsteps = int(np.ceil(args.t_final / dt_ideal))
        dt_actual = args.t_final / nsteps
        dt_dx_actual = dt_actual / dx

        config = SolverConfig(
            mesh=mesh_cfg,
            boundary=BoundaryConfig(bc_type=BCType.CONSTANT_EXTEND),
            reconst=ReconstConfig(
                slope_type=SlopeType.BOX,
                limiter_type=LimiterType.NONE,
            ),
            iteration=IterationConfig(tol=1e-10, maxiter=50),
            equation=prob.equation,
            dt_dx=dt_dx_actual,
        )

        scheme = SchemeClass(config=config)
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

    print("\n--- Convergence Results (Burgers Smooth, t=0.2) ---")
    cols = ["ncells", "L1", "Order_L1", "L2", "Order_L2", "Linf", "Order_Linf"]
    print(
        df[cols].to_string(
            index=False, float_format=lambda x: f"{x:.2e}" if x < 1e-1 else f"{x:.2f}"
        )
    )

    # --- Save Results ---
    # The 'results/' directory should be created by the Makefile
    csv_path = f"results/conv_{args.scheme}_cfl{args.dt_dx}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV results saved to '{csv_path}'")


if __name__ == "__main__":
    run_study()
