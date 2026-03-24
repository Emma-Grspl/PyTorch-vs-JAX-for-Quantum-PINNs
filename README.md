# Quantum Harmonic Oscillator PINN Benchmark

Ce dépôt est maintenant structuré pour comparer **PyTorch** et **JAX** sur un même problème PINN:
la résolution de l'état fondamental de l'oscillateur harmonique quantique 1D.

Le cadre est volontairement **sans données**. L'apprentissage repose uniquement sur:
- le résidu de l'équation de Schrödinger stationnaire,
- les conditions aux bords,
- la normalisation de la fonction d'onde,
- une contrainte de symétrie adaptée à l'état fondamental.

La solution analytique sert uniquement à **évaluer** la précision après entraînement.

## Structure

Le code principal est sous [`src/quantum_pinn`](/Users/emma.grospellier/Projects/Project1_Pytorch_Jax_Hybrid/src/quantum_pinn).

- [`src/quantum_pinn/problem.py`](/Users/emma.grospellier/Projects/Project1_Pytorch_Jax_Hybrid/src/quantum_pinn/problem.py): solution analytique et définition du problème quantique.
- [`src/quantum_pinn/pytorch`](/Users/emma.grospellier/Projects/Project1_Pytorch_Jax_Hybrid/src/quantum_pinn/pytorch): implémentation PyTorch.
- [`src/quantum_pinn/jax`](/Users/emma.grospellier/Projects/Project1_Pytorch_Jax_Hybrid/src/quantum_pinn/jax): implémentation JAX.
- [`src/quantum_pinn/metrics.py`](/Users/emma.grospellier/Projects/Project1_Pytorch_Jax_Hybrid/src/quantum_pinn/metrics.py): métriques de comparaison.
- [`src/quantum_pinn/plotting.py`](/Users/emma.grospellier/Projects/Project1_Pytorch_Jax_Hybrid/src/quantum_pinn/plotting.py): figures de sortie.
- [`config/quantum_oscillator.yaml`](/Users/emma.grospellier/Projects/Project1_Pytorch_Jax_Hybrid/config/quantum_oscillator.yaml): configuration unique du benchmark.
- [`scripts/train_pytorch.py`](/Users/emma.grospellier/Projects/Project1_Pytorch_Jax_Hybrid/scripts/train_pytorch.py): entraînement PyTorch.
- [`scripts/train_jax.py`](/Users/emma.grospellier/Projects/Project1_Pytorch_Jax_Hybrid/scripts/train_jax.py): entraînement JAX.
- [`scripts/compare_frameworks.py`](/Users/emma.grospellier/Projects/Project1_Pytorch_Jax_Hybrid/scripts/compare_frameworks.py): comparaison finale.

## Entraînement

Avec l'environnement local existant:

```bash
./project1/bin/python scripts/train_pytorch.py
```

Pour JAX, il faut d'abord installer `jax` dans l'environnement. Ensuite:

```bash
./project1/bin/python scripts/train_jax.py
./project1/bin/python scripts/compare_frameworks.py
```

## Sorties du benchmark

Chaque entraînement écrit un `metrics.json` dans `results/quantum_oscillator/<framework>/`.
Le script de comparaison agrège ces résultats dans `results/quantum_oscillator/benchmark_summary.csv`.

## Remarques

- Le benchmark actuel est centré sur l'état fondamental `n = 0`. La structure permet ensuite d'étendre aux états excités via `state_index`.
