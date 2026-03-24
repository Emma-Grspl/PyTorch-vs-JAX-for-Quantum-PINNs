from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.quantum_pinn.config import load_config, resolve_framework_config
from src.quantum_pinn.io import ensure_dir, write_json
from src.quantum_pinn.runner import run_jax_once


def main() -> None:
    try:
        from src.quantum_pinn.jax.trainer import JAXTrainer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "JAX is not installed in the current environment. "
            "Install `jax` before running `scripts/train_jax.py`."
        ) from exc
    from src.quantum_pinn.plotting import plot_prediction, plot_training_history

    config_path = ROOT / "config" / "quantum_oscillator.yaml"
    config = resolve_framework_config(load_config(config_path), "jax")

    output_dir = ensure_dir(ROOT / config["experiment"]["output_dir"] / "jax")
    results_dir = ensure_dir(ROOT / config["experiment"]["results_dir"] / "jax")

    metrics, history, params = run_jax_once(config, int(config["experiment"]["seed"]))

    write_json(results_dir / "metrics.json", metrics)
    from src.quantum_pinn.problem import reference_solution
    from src.quantum_pinn.metrics import align_sign
    from src.quantum_pinn.jax.model import build_activation, mlp_forward
    import jax.numpy as jnp

    x_eval, psi_exact, _ = reference_solution(config["problem"])
    activation = build_activation(config["model"]["activation"])
    psi_pred = mlp_forward(
        params["network"],
        jnp.asarray(x_eval.reshape(-1, 1), dtype=jnp.float32),
        activation,
    ).squeeze(-1)
    psi_pred = align_sign(np.asarray(psi_pred, dtype=np.float32), psi_exact)
    plot_prediction(
        x=x_eval,
        reference=psi_exact,
        prediction=psi_pred,
        path=output_dir / "prediction.png",
        title="Quantum Harmonic Oscillator Ground State - JAX PINN",
    )
    plot_training_history(history, output_dir / "losses.png")

    print("JAX training completed.")
    print(metrics)


if __name__ == "__main__":
    main()
