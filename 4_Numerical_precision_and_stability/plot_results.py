import csv
from pathlib import Path

import matplotlib.pyplot as plt


folder = Path(__file__).parent
csv_path = folder / "results.csv"
plot_path = folder / "performance_plots.png"

with csv_path.open("r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

if not rows:
    raise ValueError("results.csv is empty.")

fig, (time_ax, speed_ax) = plt.subplots(1, 2, figsize=(13, 5))

colors = {
    "matvec": "tab:blue",
    "matmul": "tab:orange",
}
styles = {
    "float32": "-",
    "float64": "--",
}

for dtype in ("float32", "float64"):
    dtype_rows = sorted(
        [row for row in rows if row["dtype"] == dtype],
        key=lambda row: int(row["N"]),
    )

    if not dtype_rows:
        continue

    sizes = [int(row["N"]) for row in dtype_rows]

    for operation, label in (("matvec", "Matrix-vector"), ("matmul", "Matrix-matrix")):
        times = [float(row[f"{operation}_time_s"]) for row in dtype_rows]
        speeds = [float(row[f"{operation}_gflops"]) for row in dtype_rows]

        time_ax.plot(
            sizes,
            times,
            marker="o",
            color=colors[operation],
            linestyle=styles[dtype],
            label=f"{label} {dtype}",
        )
        speed_ax.plot(
            sizes,
            speeds,
            marker="o",
            color=colors[operation],
            linestyle=styles[dtype],
            label=f"{label} {dtype}",
        )

time_ax.set_title("Execution time")
time_ax.set_xlabel("Matrix size N")
time_ax.set_ylabel("Time (seconds)")
time_ax.set_xscale("log", base=2)
time_ax.set_yscale("log")
time_ax.grid(True, which="both", linestyle=":")

speed_ax.set_title("Computational performance")
speed_ax.set_xlabel("Matrix size N")
speed_ax.set_ylabel("Performance (GFLOP/s)")
speed_ax.set_xscale("log", base=2)
speed_ax.grid(True, which="both", linestyle=":")

time_ax.legend(fontsize=8)
speed_ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(plot_path, dpi=200)

print(f"Plots saved to: {plot_path}")