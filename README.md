# WQU-Gizmo: Subspace-Constrained Quantum Portfolio Selection
**Team Gizmo — IBM Qiskit Fall Fest Hackathon 2026**

## Team Members
- **Akhilesh Varma** (`akhverm@gmail.com`)
- **Abhay Gupta** (`guptasanjay6283@gmail.com`)

## Overview
Project Gizmo presents an end-to-end framework for constrained Markowitz portfolio optimization on gate-model quantum computers. By constraining the variational quantum evolution strictly to the symmetric excitation subspace using Dicke states and XY-mixers, Gizmo achieves 100% feasibility across all shots, eliminating soft-penalty hyperparameter tuning and reducing quantum resource requirements via linear alternating SWAP networks on IBM heavy-hex topologies.

## Key Modules
- `gizmo.data`: Market data ingestion with live S&P 500 download and offline factor-model fallback.
- `gizmo.classical`: Exact combinatorial ground-truth evaluation over all binomial combinations.
- `gizmo.hamiltonian`: Mapping Markowitz mean-variance optimization to Ising Hamiltonians.
- `gizmo.subspace`: Dicke state initialization and excitation-preserving XY-mixers.
- `gizmo.swap_network`: Linear alternating SWAP networks for all-to-all covariance mapping in linear depth.
- `gizmo.algorithms`: Naive vs Subspace QAOA circuit generation and optimization routines.

## Documentation & Presentation
- Formal Paper: `docs/gizmo_paper.pdf` (`docs/gizmo_paper.tex`)
- Presentation Deck: `docs/gizmo_slides.pdf` (`docs/gizmo_slides.tex`)
- Executed Showcase Notebook: `gizmo_portfolio_showcase.ipynb`
