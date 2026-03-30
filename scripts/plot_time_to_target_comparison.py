"""Generate comparison figures for the separated PyTorch-to-JAX time-to-target runs."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cache_dir = Path(tempfile.gettempdir()) / "qho_pinn_matplotlib"
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))

import matplotlib.pyplot as plt
import numpy as np


RUNS = [
    {
        "label": "physics_only",
        "display": "Physics Only",
        "pytorch_targets": ROOT / "artifacts" / "hpc_pytorch_targets_jz_ptgt_1523131" / "pytorch_targets.json",
        "jax_summary": ROOT / "artifacts" / "hpc_jax_time_to_target_jz_jt2t_1524156" / "jax_time_to_target_summary.csv",
    },
    {
        "label": "physics_plus_data_64",
        "display": "Physics + Data (64)",
        "pytorch_targets": ROOT / "artifacts" / "hpc_pytorch_targets_jz_ptgt_1523161" / "pytorch_targets.json",
        "jax_summary": ROOT / "artifacts" / "hpc_jax_time_to_target_jz_jt2t_1524357" / "jax_time_to_target_summary.csv",
    },
]


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_row(path: Path) -> dict[str, float | int | str]:
    """Load a one-row CSV summary."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    parsed: dict[str, float | int | str] = {}
    for key, value in row.items():
        try:
            if value.isdigit():
                parsed[key] = int(value)
            else:
                parsed[key] = float(value)
        except ValueError:
            parsed[key] = value
    return parsed


def load_summary() -> list[dict[str, float | int | str]]:
    """Load the curated time-to-target runs and derived metrics."""
    rows = []
    for run in RUNS:
        pytorch_payload = load_json(run["pytorch_targets"])
        jax_summary = load_csv_row(run["jax_summary"])
        pytorch_rows = pytorch_payload["reference_runs"]
        pytorch_mean_total = float(np.mean([row["training_seconds"] for row in pytorch_rows]))
        pytorch_median_total = float(np.median([row["training_seconds"] for row in pytorch_rows]))
        rows.append(
            {
                "label": run["label"],
                "display": run["display"],
                "pytorch_mean_total": pytorch_mean_total,
                "pytorch_median_total": pytorch_median_total,
                "pytorch_l2_target": float(pytorch_payload["targets"]["relative_l2_error_target"]),
                "pytorch_energy_target": float(pytorch_payload["targets"]["absolute_energy_error_target"]),
                "jax_success_l2": int(jax_summary["successful_runs_l2"]),
                "jax_success_energy": int(jax_summary["successful_runs_energy"]),
                "jax_success_both": int(jax_summary["successful_runs_both"]),
                "jax_runs": int(jax_summary["measured_runs"]),
                "jax_time_l2_median": float(jax_summary["time_to_l2_target_median"]),
                "jax_time_energy_median": float(jax_summary["time_to_energy_target_median"]),
                "jax_time_both_median": float(jax_summary["time_to_both_targets_median"]),
                "jax_final_l2_median": float(jax_summary["relative_l2_error_median"]),
                "jax_final_energy_median": float(jax_summary["absolute_energy_error_median"]),
            }
        )
    return rows


def plot_success_rates(rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    """Plot JAX success rates against the PyTorch-derived targets."""
    labels = [str(row["display"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.24
    metrics = [
        ("jax_success_l2", "L2"),
        ("jax_success_energy", "Energy"),
        ("jax_success_both", "Both"),
    ]
    colors = ["#2a6f97", "#d17a22", "#3a7d44"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for idx, ((key, title), color) in enumerate(zip(metrics, colors)):
        values = [100.0 * float(row[key]) / float(row["jax_runs"]) for row in rows]
        bars = ax.bar(x + (idx - 1) * width, values, width=width, color=color, label=title)
        for bar, value, row in zip(bars, values, rows):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1.0,
                f"{int(row[key])}/{int(row['jax_runs'])}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_ylabel("Success Rate (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 112)
    ax.set_title("JAX Success Rate Against PyTorch Median Targets")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "time_to_target_success_rates.png", dpi=250)
    plt.close(fig)


def plot_time_cost(rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    """Compare PyTorch full-run time against JAX median time-to-target."""
    labels = [str(row["display"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    pytorch_values = [float(row["pytorch_mean_total"]) for row in rows]
    jax_values = [float(row["jax_time_both_median"]) for row in rows]

    bars_pt = ax.bar(x - width / 2, pytorch_values, width=width, color="#1f77b4", label="PyTorch mean full training time")
    bars_jax = ax.bar(x + width / 2, jax_values, width=width, color="#e24a33", label="JAX median time to both targets")

    for bars, values in ((bars_pt, pytorch_values), (bars_jax, jax_values)):
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.2f}s", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Seconds")
    ax.set_xticks(x, labels)
    ax.set_title("Cost to Reach the PyTorch-Derived Target")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "time_to_target_cost.png", dpi=250)
    plt.close(fig)


def write_manifest(rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    """Write the metrics used for the figures."""
    with (output_dir / "time_to_target_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def main() -> None:
    """Generate the comparison figures and manifest."""
    rows = load_summary()
    figure_dir = ROOT / "results" / "quantum_oscillator" / "time_to_target_comparison"
    analysis_dir = ROOT / "outputs" / "quantum_oscillator" / "analysis" / "time_to_target_comparison"
    asset_dir = ROOT / "assets" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    plot_success_rates(rows, figure_dir)
    plot_time_cost(rows, figure_dir)
    write_manifest(rows, analysis_dir)

    for name in ("time_to_target_success_rates.png", "time_to_target_cost.png"):
        (asset_dir / name).write_bytes((figure_dir / name).read_bytes())

    print(f"Figures written to: {figure_dir}")
    print(f"Copied curated figures to: {asset_dir}")
    print(f"Manifest written to: {analysis_dir}")


if __name__ == "__main__":
    main()
