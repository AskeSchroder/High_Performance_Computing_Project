import argparse
from os.path import join
import time

import numpy as np
from numba import njit

LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
GRID_SIZE = 512


def load_building_ids(load_dir):
    with open(join(load_dir, "building_ids.txt"), "r") as f:
        return f.read().splitlines()


def load_data(load_dir, bid):
    u = np.zeros((GRID_SIZE + 2, GRID_SIZE + 2), dtype=np.float64)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(np.bool_)
    return u, interior_mask


@njit(cache=True)
def jacobi_numba(u0, interior_mask, max_iter, atol):
    """
    Jacobi solver using explicit loops and double buffering.
    u0:           shape (514, 514)
    interior_mask shape (512, 512)
    """
    u = u0.copy()
    u_new = u0.copy()

    nrows, ncols = u.shape

    for _ in range(max_iter):
        delta = 0.0

        # Loop over padded grid interior: 1..512 in both directions
        for i in range(1, nrows - 1):
            ii = i - 1  # corresponding row in 512x512 mask

            # j as inner loop is cache-friendly for row-major arrays
            for j in range(1, ncols - 1):
                jj = j - 1

                if interior_mask[ii, jj]:
                    new_val = 0.25 * (
                        u[i, j - 1] +
                        u[i, j + 1] +
                        u[i - 1, j] +
                        u[i + 1, j]
                    )

                    diff = abs(new_val - u[i, j])
                    if diff > delta:
                        delta = diff

                    u_new[i, j] = new_val

        u, u_new = u_new, u

        if delta < atol:
            break

    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]

    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.mean(u_interior > 18.0) * 100.0
    pct_below_15 = np.mean(u_interior < 15.0) * 100.0

    return mean_temp, std_temp, pct_above_18, pct_below_15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=1)
    parser.add_argument("--max-iter", type=int, default=20_000)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()

    building_ids = load_building_ids(LOAD_DIR)[:args.N]

    if args.time:
        t0 = time.perf_counter()

    print("building_id,mean_temp,std_temp,pct_above_18,pct_below_15")

    for k, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)

        # First call includes JIT compilation overhead
        u = jacobi_numba(u0, interior_mask, args.max_iter, args.atol)

        mean_temp, std_temp, pct_above_18, pct_below_15 = summary_stats(u, interior_mask)

        print(f"{bid},{mean_temp},{std_temp},{pct_above_18},{pct_below_15}")

    if args.time:
        t1 = time.perf_counter()
        print(f"# Total runtime: {t1 - t0:.3f} seconds")


if __name__ == "__main__":
    main()