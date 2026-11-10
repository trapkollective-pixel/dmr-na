# Dual-Mirrored Recursive Neural Architecture (DMR-NA)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24097297.svg)](https://doi.org/10.5281/zenodo.24097297)

An alternative sequence-learning topology that enforces mathematical state boundedness and gradient retention via reciprocal cross-coupling and continuous digital root reduction dynamics ($f(n) = 1 + ((n - 1) \pmod 9)$).

## Features
- **Harmonic Digital Root Gate:** Infinitely differentiable Fourier-series surrogate mapping latent states onto compact modular residue manifolds.
- **Reciprocal Symmetry:** Dual-mirrored fields (Primary $\mathcal{P}$ and Mirrored $\mathcal{M}$) establishing dynamic equilibrium without explicit normalization layers.
- **Gradient Preservation:** Strict analytical lower bounds preventing vanishing or exploding gradients over deep horizons.

## Project Structure
- `dmr_na/`: Core PyTorch cells and model definitions.
- `benchmarks/`: Synthetic benchmark test suites (adding problem).
- `tests/`: Pytest suite verifying stability bounds and contraction mapping.
- `paper/`: Compiled academic preprint PDF.

## Author
**Lloyd C. Woods**  
Independent Researcher, Tulsa, OK  
Email: `trapkollective@gmail.com`
