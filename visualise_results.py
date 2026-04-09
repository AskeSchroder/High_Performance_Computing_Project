import argparse
from os.path import join
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    u = u0.copy()
    u_new = u0.copy()
    nrows, ncols = u.shape

    for _ in range(max_iter):
        delta = 0.0

        for i in range(1, nrows - 1):
            ii = i - 1
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
                else:
                    u_new[i, j] = u[i, j]

        u, u_new = u_new, u

        if delta < atol:
            break

    return u


def make_plot(bid, u0, interior_mask, u_final, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(u0[1:-1, 1:-1], origin="lower")
    axes[0].set_title(f"Initial grid\nBuilding {bid}")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(interior_mask, origin="lower")
    axes[1].set_title("Interior mask")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(u_final[1:-1, 1:-1], origin="lower")
    axes[2].set_title("Final temperature")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.tight_layout()
    out_path = join(out_dir, f"{bid}_visualization.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=3, help="Number of floorplans to visualize")
    parser.add_argument("--max-iter", type=int, default=20_000)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--outdir", type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    building_ids = load_building_ids(LOAD_DIR)[:args.N]

    # warm-up compile on first building
    first_u0, first_mask = load_data(LOAD_DIR, building_ids[0])
    _ = jacobi_numba(first_u0, first_mask, args.max_iter, args.atol)

    for bid in building_ids:
        u0, interior_mask = load_data(LOAD_DIR, bid)
        u_final = jacobi_numba(u0, interior_mask, args.max_iter, args.atol)
        make_plot(bid, u0, interior_mask, u_final, args.outdir)


if __name__ == "__main__":
    main()