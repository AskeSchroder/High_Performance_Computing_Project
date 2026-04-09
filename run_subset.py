import argparse
from os.path import join
import time

import numpy as np


LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
GRID_SIZE = 512


def load_building_ids(load_dir):
    with open(join(load_dir, "building_ids.txt"), "r") as f:
        return f.read().splitlines()


def load_data(load_dir, bid):
    """
    Loads one building:
      - domain: 512x512 initial temperature grid
      - interior mask: 512x512 boolean mask
    Pads the domain to 514x514 to match the reference implementation.
    """
    u = np.zeros((GRID_SIZE + 2, GRID_SIZE + 2), dtype=np.float64)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(bool)
    return u, interior_mask


@profile
def jacobi(u, interior_mask, max_iter=20_000, atol=1e-4):
    """
    Reference-style Jacobi solver.
    """
    u = np.copy(u)

    for _ in range(max_iter):
        u_new = 0.25 * (
            u[1:-1, :-2] +   # left
            u[1:-1, 2:] +    # right
            u[:-2, 1:-1] +   # up
            u[2:, 1:-1]      # down
        )

        u_old_interior = u[1:-1, 1:-1][interior_mask]
        u_new_interior = u_new[interior_mask]

        delta = np.abs(u_old_interior - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break

    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]

    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.mean(u_interior > 18.0) * 100.0
    pct_below_15 = np.mean(u_interior < 15.0) * 100.0

    return {
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "pct_above_18": pct_above_18,
        "pct_below_15": pct_below_15,
    }


def main():
    parser = argparse.ArgumentParser(description="Run wall-heating simulation on first N floorplans.")
    parser.add_argument("N", type=int, nargs="?", default=1, help="Number of floorplans to process")
    parser.add_argument("--max-iter", type=int, default=20_000, help="Maximum Jacobi iterations")
    parser.add_argument("--atol", type=float, default=1e-4, help="Convergence tolerance")
    parser.add_argument("--time", action="store_true", help="Print timing information")
    args = parser.parse_args()

    building_ids = load_building_ids(LOAD_DIR)[:args.N]

    if args.time:
        t0 = time.perf_counter()

    print("building_id,mean_temp,std_temp,pct_above_18,pct_below_15")

    for bid in building_ids:
        u0, interior_mask = load_data(LOAD_DIR, bid)
        u = jacobi(u0, interior_mask, max_iter=args.max_iter, atol=args.atol)
        stats = summary_stats(u, interior_mask)

        print(
            f"{bid},"
            f"{stats['mean_temp']},"
            f"{stats['std_temp']},"
            f"{stats['pct_above_18']},"
            f"{stats['pct_below_15']}"
        )

    if args.time:
        t1 = time.perf_counter()
        print(f"# Total runtime: {t1 - t0:.3f} seconds")


if __name__ == "__main__":
    main()