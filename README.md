
# 🚚 Diffusion + Mamba Routing

### Neural Optimization for CVRP, VRPTW, PDVRP, MDVRP, Dynamic VRP

**A modular research framework for scalable neural routing solvers.**

---

## Overview

This repo contains my research codebase for **Diffusion-based latent search + Mamba SSM refinement** for Vehicle Routing Problems (VRPs).

The goal is to build **a unified, general-purpose neural routing solver** that can:

* generate diverse, high-quality routing structures via **diffusion priors**,
* enforce feasibility through **lightweight Mamba refinement**,
* support multiple VRP families via **modular constraint heads**,
* scale training + inference on **Cloud TPU v4/v5e/v6e**,
* evaluate against classical + neural baselines (LKH, OR-Tools, POMO, DiffCO).

This repo is the implementation for my papers on:

* CVRP
* VRPTW
* PDVRP
* MDVRP
* Dynamic VRP

Each problem simply plugs in a different **Constraint Module**, everything else stays shared and elegant.

---

##  Repository Structure

```
routing/
│
├── core/
│   ├── diffusion/        # score model, DDIM sampler, noise schedules
│   ├── mamba/            # refinement blocks, state-space updates
│   ├── constraints/      # CVRP, VRPTW, PDVRP, MDVRP, Dynamic VRP constraint heads
│   ├── data/             # dataset loaders + synthetic generators
│   ├── utils/            # metrics, logging, distributed utils, etc.
│   └── models/           # shared encoders, embeddings, latent reps
│
├── configs/              # YAML configs for experiments + sweeps
├── experiments/          # training runs + results (ignored by Git)
├── scripts/              # launch scripts for TPU / Slurm / local
├── main_train.py         # high-level train loop
├── main_infer.py         # evaluation + sampling
└── README.md
```

Everything is **plug-n-play**:

* swap constraint → solve new VRP
* swap diffusion backbone → try new priors
* swap refinement block → test different SSMs
* change config → full hyperparam sweep

---

## Method Summary

### Diffusion Prior

Learns the geometry of feasible routing structures in a continuous latent space.

### Mamba Refinement

Performs fast, differentiable constraint-aware correction steps to push samples toward feasibility.

### Constraint Modules

Each VRP variant provides:

* violation functions,
* projection operators,
* constraint embeddings.

This keeps the solver **general** but the constraints **specific**.

---

## Training

Supports **JAX/Flax** and **PyTorch** backends.

### Start a training run:

```bash
python main_train.py --config configs/cvrp.yaml
```

### TPU (v4/v5e/v6e) launch example:

```bash
python scripts/launch_tpu.py --zone europe-west4-b --tpu v5e-64 --config configs/cvrp.yaml
```

Distributed training uses:

* `jax.pjit` / `nnx` for Flax
* `pjrt` runtime for TPUs
* automatic mixed precision (bf16)

---

## 📊 Logging with Weights & Biases

Enable W&B by adding in your config:

```yaml
wandb:
  enabled: true
  project: diffusion-mamba-routing
  entity: YOUR_USERNAME
```

Training automatically logs:

* losses
* violation curves
* sample trajectories
* inference metrics
* run configs
* TPU topology info

You also get **hyperparameter sweeps** via:

```bash
wandb sweep configs/sweep.yaml
```

---

## 🧪 Baselines Included

* OR-Tools
* LKH-3 (optional external dependency)
* POMO
* Attention Model (Kool et al.)
* Diffusion CO

Used for benchmarking across CVRP100/200, VRPTW, PDVRP, etc.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Or if using JAX + TPU:

```bash
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

---

## 📄 Papers (WIP)

📄 **CVRP** — *Diffusion + Mamba for Capacitated Vehicle Routing*

📄 **VRPTW** — *Temporal Diffusion Models for Time-Window Routing*

📄 **PDVRP** — *Diffusion Optimization with Pickup–Delivery Constraints*

📄 **MDVRP** — *Multi-Depot Routing Priors + Mamba Refinement*

📄 **Dynamic VRP** — *Online Diffusion Routing with Predictive SSMs*

Each paper will have:

* its own folder
* training configs
* experiment scripts
* results tables

---

## 💬 Contact

For questions or collaboration:
**[ritwikareddykancharla@gmail.com](mailto:ritwikareddykancharla@gmail.com)**

---
