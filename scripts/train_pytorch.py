from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.quantum_pinn.benchmark import build_run_metadata, count_pytorch_parameters
from src.quantum_pinn.config import load_config, resolve_framework_config
from src.quantum_pinn.io import ensure_dir, write_json
from src.quantum_pinn.runner import run_pytorch_once


def main() -> None:
    from src.quantum_pinn.plotting import plot_prediction, plot_training_history

    config_path = ROOT / "config" / "quantum_oscillator.yaml"
    config = resolve_framework_config(load_config(config_path), "pytorch")

    seed = int(config["experiment"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = ensure_dir(ROOT / config["experiment"]["output_dir"] / "pytorch")
    results_dir = ensure_dir(ROOT / config["experiment"]["results_dir"] / "pytorch")

    metrics, history, model = run_pytorch_once(config, seed)

    torch.save(model.state_dict(), results_dir / "model.pt")
    write_json(results_dir / "metrics.json", metrics)
    from src.quantum_pinn.problem import reference_solution
    from src.quantum_pinn.metrics import align_sign
    x_eval, psi_exact, _ = reference_solution(config["problem"])
    trainer_prediction = model(torch.tensor(x_eval.reshape(-1, 1), dtype=torch.float32)).detach().cpu().numpy().squeeze(-1)
    psi_pred = align_sign(trainer_prediction, psi_exact)
    plot_prediction(
        x=x_eval,
        reference=psi_exact,
        prediction=psi_pred,
        path=output_dir / "prediction.png",
        title="Quantum Harmonic Oscillator Ground State - PyTorch PINN",
    )
    plot_training_history(history, output_dir / "losses.png")

    print("PyTorch training completed.")
    print(metrics)


if __name__ == "__main__":
    main()
