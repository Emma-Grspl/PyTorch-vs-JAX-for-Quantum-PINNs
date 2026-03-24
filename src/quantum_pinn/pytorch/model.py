from __future__ import annotations

import torch
from torch import nn


def build_activation(name: str) -> nn.Module:
    activations = {
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


class QuantumPINN(nn.Module):
    def __init__(self, hidden_layers: list[int], activation: str, energy_init: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = 1

        for width in hidden_layers:
            layers.append(nn.Linear(in_features, width))
            layers.append(build_activation(activation))
            in_features = width

        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)
        self.energy = nn.Parameter(torch.tensor(float(energy_init), dtype=torch.float32))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

