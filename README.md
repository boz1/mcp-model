# Decentralized Building Simulator (db-sims)

Simulator for studying geographic location choice incentives in decentralized block building.

Agents (builders) learn to pick regions via reinforcement learning (EMA-Softmax or UCB). Rewards are shared among builders who cover a source using a configurable sharing rule (e.g., `equal_split`). The simulator tracks how the population distributes across regions over time and how efficiently sources are covered.

## Quick Start

```bash
pip install -r requirements.txt
python run.py configs/ema_baseline.yaml
```

To run all configs in a directory:

```bash
python run.py configs/
```

## Usage

```
python run.py <config.yaml> [config2.yaml ...] | <configs_dir/> [--poa] [--poa-method {brute_force,greedy}]
```

### PoA Analysis

```bash
# Exact optimal welfare (feasible for small numbers of regions or builders)
python run.py configs/ema_baseline.yaml --poa

# Greedy approximation (faster for large numbers of regions or builders)
python run.py configs/ema_baseline.yaml --poa --poa-method greedy
```

Results and plots are saved to `results/`.

## Key Parameters

| Parameter | Description |
|---|---|
| `policy_type` | `"EMA"` or `"UCB"` |
| `eta`, `beta_reg` | EMA learning rate and softmax temperature |
| `alpha` | UCB exploration bonus |
| `cost_c` | Migration cost |
| `n_builders` | Number of concurrent builders per slot |
| `n_slots` | Number of simulation slots |

## Metrics

- **Inequality**: Gini, entropy, HHI across regions/sources/population
- **Value-capture**: per-region share, top-1/top-3 concentration
- **Volatility**: L1 change in distributions between slots
- **Price of Anarchy**: `optimal_welfare / actual_welfare` (≥ 1; 1 = socially optimal). Optimal welfare assumes one builder per source.

All metrics are time series; pass any subset to `compare_experiments(results, metrics=[...])`.

## Project Structure

```
sim/
  simulator.py         — core simulator (policies, propagation model, tracking)
  config.py            — ExperimentConfig, load_config
  datasets.py          — GCP latency data loading
analysis/
  experiment_runner.py — runner, plots
  result.py            — ExperimentResult
  plotting.py          — comparison and detail plots
  poa.py               — Price of Anarchy computation
configs/               — YAML experiment configs
run.py                 — CLI entrypoint
```

## References

- Paper: [arXiv:2509.21475v2](https://arxiv.org/pdf/2509.21475v2)
- Original repo: [geographical-decentralization-simulation](https://github.com/syang-ng/geographical-decentralization-simulation)
