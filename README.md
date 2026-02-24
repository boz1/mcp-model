# Decentralized Building Simulator (db-sims)

Simulator for studying geographic location choice incentives in decentralized block building.

Agents (proposers) learn to pick regions and information sources via reinforcement learning (EMA-Softmax or UCB). Rewards are shared among proposers using a source according to a configurable policy — either distance-ranked (`rank_weighted`) or flat (`equal_split`). The simulator tracks how the population distributes across regions over time and how efficiently sources are covered.

## Quick Start

```bash
python my_experiments.py
```

Edit `my_experiments.py` to define experiments. Shared topology lives in module-level constants (`REGIONS`, `SOURCES`, `K`, `BASE`); each `ExperimentConfig` unpacks `**BASE` and overrides only what changes.

## Key Parameters

| Parameter | Description |
|---|---|
| `sharing_policy` | `"rank_weighted"` or `"equal_split"` |
| `policy_type` | `"EMA"` or `"UCB"` |
| `eta`, `beta_reg`, `beta_src` | EMA learning rate and softmax temperatures |
| `alpha` | UCB exploration bonus |
| `cost_c` | Migration cost |
| `K` | Concurrent proposers per slot |

## Metrics

- **Inequality**: Gini, entropy, HHI across regions/sources/population
- **Value-capture**: per-region share, top-1/top-3 concentration
- **Volatility**: L1 change in distributions between slots
- **Price of Anarchy**: `optimal_welfare / actual_welfare` (≥ 1; 1 = socially optimal). Optimal welfare assumes one proposer per source.

All metrics are time series; pass any subset to `compare_experiments(results, metrics=[...])`.

## Files

- `mcp_simulator.py` — core simulator (Omega mechanism, policies, tracking)
- `experiment_runner.py` — config, runner, comparison plots, detail plots
- `my_experiments.py` — experiment definitions (edit this)

## References

- Paper: [arXiv:2509.21475v2](https://arxiv.org/pdf/2509.21475v2)
- Original repo: [geographical-decentralization-simulation](https://github.com/syang-ng/geographical-decentralization-simulation)
