"""Generate random wafer maps for testing the upload feature.

Usage:
    python make_test_wafers.py
    python make_test_wafers.py --count 3 --seed 7 --out test_wafers
    python make_test_wafers.py --data Wafer_Map_Datasets.npz   (use the real wafer outline)

Each sample is written as .npy, .csv and .png. A few deliberately invalid files
are written to <out>/invalid so you can check the error messages.
manifest.csv lists the pattern each file was drawn to resemble. That is the
generator's intent, not a guarantee of what a model will predict.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

N = 52
PALETTE = {0: (27, 39, 64), 1: (53, 196, 107), 2: (239, 68, 68)}  # blank, pass, fail
CELL_PIXELS = 10

# Patterns to draw. Each entry is a list of basic defects.
SAMPLES = [
    ["Normal"],
    ["Center"], ["Donut"], ["Edge_Loc"], ["Edge_Ring"], ["Loc"],
    ["Near_Full"], ["Scratch"], ["Random"],
    ["Center", "Edge_Loc"], ["Center", "Scratch"], ["Donut", "Loc"],
    ["Edge_Ring", "Scratch"], ["Center", "Edge_Ring", "Scratch"],
    ["Donut", "Edge_Loc", "Loc"], ["Center", "Loc", "Edge_Loc", "Scratch"],
]


def wafer_outline(data_path):
    """Boolean mask of die positions. Uses the dataset if given, else a circle."""
    if data_path and Path(data_path).exists():
        with np.load(data_path) as data:
            wafers = data["arr_0"]
        step = max(1, len(wafers) // 2000)
        return (wafers[::step] != 0).mean(axis=0) > 0.5

    yy, xx = np.mgrid[:N, :N]
    return np.hypot(yy - (N - 1) / 2, xx - (N - 1) / 2) <= 25


def draw(grid, mask, kind, rng):
    """Turn dies into failures (value 2) for one basic defect type."""
    yy, xx = np.mgrid[:N, :N]
    cy, cx = (N - 1) / 2, (N - 1) / 2
    radius = np.hypot(yy - cy, xx - cx)
    angle = np.arctan2(yy - cy, xx - cx)
    shape = np.zeros((N, N), bool)

    if kind == "Center":
        shape = radius <= rng.uniform(4, 8)

    elif kind == "Donut":
        inner = rng.uniform(4, 7)
        shape = (radius >= inner) & (radius <= inner + rng.uniform(3, 5))

    elif kind == "Edge_Ring":
        shape = radius >= 25 - rng.uniform(2.5, 4)

    elif kind == "Edge_Loc":
        start = rng.uniform(-np.pi, np.pi)
        width = rng.uniform(0.6, 1.2)
        diff = np.angle(np.exp(1j * (angle - start)))
        shape = (radius >= 25 - rng.uniform(3, 5)) & (np.abs(diff) <= width / 2)

    elif kind == "Loc":
        theta = rng.uniform(-np.pi, np.pi)
        dist = rng.uniform(8, 18)
        py, px = cy + dist * np.sin(theta), cx + dist * np.cos(theta)
        blob = np.hypot(yy - py, xx - px) <= rng.uniform(3.5, 6)
        shape = blob & (rng.random((N, N)) < 0.9)

    elif kind == "Scratch":
        p0 = rng.uniform(5, 47, size=2)
        p1 = rng.uniform(5, 47, size=2)
        bend = rng.uniform(-6, 6)
        for t in np.linspace(0, 1, 160):
            y = p0[0] + (p1[0] - p0[0]) * t + bend * np.sin(np.pi * t)
            x = p0[1] + (p1[1] - p0[1]) * t
            iy, ix = int(round(y)), int(round(x))
            if 0 <= iy < N and 0 <= ix < N:
                shape[iy, ix] = True

    elif kind == "Near_Full":
        shape = rng.random((N, N)) < rng.uniform(0.7, 0.9)

    elif kind == "Random":
        shape = rng.random((N, N)) < rng.uniform(0.3, 0.45)

    grid[mask & shape] = 2
    return grid


def make_wafer(kinds, mask, rng):
    grid = np.where(mask, 1, 0).astype(np.int8)

    # Background scatter, present even on Normal wafers
    scatter = rng.uniform(0.02, 0.07)
    grid[mask & (rng.random((N, N)) < scatter)] = 2

    for kind in kinds:
        if kind != "Normal":
            draw(grid, mask, kind, rng)

    return grid


def save_png(grid, path):
    rgb = np.zeros((N, N, 3), np.uint8)
    for value, color in PALETTE.items():
        rgb[grid == value] = color

    image = Image.fromarray(rgb).resize(
        (N * CELL_PIXELS, N * CELL_PIXELS), Image.NEAREST
    )
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="test_wafers")
    parser.add_argument("--count", type=int, default=1, help="samples per pattern")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", default="Wafer_Map_Datasets.npz")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "invalid").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    mask = wafer_outline(args.data)
    manifest = []
    number = 0

    for kinds in SAMPLES:
        for _ in range(args.count):
            number += 1
            grid = make_wafer(kinds, mask, rng)
            stem = f"wafer_{number:02d}_{'_'.join(kinds).lower()}"

            np.save(out / f"{stem}.npy", grid)
            np.savetxt(out / f"{stem}.csv", grid, fmt="%d", delimiter=",")
            save_png(grid, out / f"{stem}.png")

            manifest.append({
                "file": stem,
                "drawn_pattern": "+".join(kinds),
                "failed_dies": int((grid == 2).sum()),
                "fail_rate_percent": round(100 * (grid == 2).sum() / mask.sum(), 1),
            })

    # Files that the app should reject with a clear message
    np.save(out / "invalid" / "invalid_shape_50x50.npy", np.ones((50, 50), np.int8))
    bad = make_wafer(["Center"], mask, rng)
    bad[10, 10] = 3
    np.save(out / "invalid" / "invalid_value_3.npy", bad)
    (out / "invalid" / "invalid_text.csv").write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    pd.DataFrame(manifest).to_csv(out / "manifest.csv", index=False)
    print(f"Wrote {number} wafers (.npy, .csv, .png) and 3 invalid files to {out}/")


if __name__ == "__main__":
    main()
