# Decentralized Building Simulator (db-sims)

A simulator for studying location choice incentives in distributed and decentralized block building regimes.

## Overview

This simulator models block builders (agents) that:

- Choose regions (geographic locations) and information sources
- Learn optimal strategies through reinforcement learning (EMA-Softmax or UCB)
- Receive rewards based on distance-ranked sharing with concentration penalties
- Start from an even distribution and migrate over time

The simulator is designed to study how different block building regimes (e.g., multiple concurrent proposers, distributed building networks, decentralized ordering protocols) affect the geographic distribution of builders and the resulting centralization dynamics.

## Key Features

- **Distance-ranked sharing**: Closer regions to sources get higher rank weights (w_j = 2^-j)
- **Concentration penalty**: Crowded regions split rewards, incentivizing diversity
- **Persistent proposer distribution**: Track how proposers migrate between regions over time
- **Multiple learning policies**: EMA-Softmax and UCB
- **Comprehensive metrics**: Gini, entropy, HHI, value-capture, volatility, and convergence analysis
- **Economic decentralization**: Track value-capture distribution and concentration across regions
- **Easy experimentation**: Centralized configuration and comparison tools

## Quick Start

### Run the default experiments:

```bash
python my_experiments.py
```

This will:

- Run 4 different experiment configurations
- Compare EMA vs UCB policies, exploration levels, and migration costs
- Generate plots showing how metrics evolve over time
- Save results to `experiment_results/` directory

### Customize your experiments:

Edit `my_experiments.py` and modify the `define_experiments()` function:

```python
def define_experiments():
    experiments = []

    experiments.append(ExperimentConfig(
        name="my_experiment",

        # Geography
        n_regions=5,
        region_names=["West", "CentralWest", "Central", "CentralEast", "East"],

        # Information sources: (name, value, home_region)
        sources_config=[
            ("LowValue", 8.0, 0),      # At West (distance 0)
            ("MedValue", 12.0, 2),     # At Central (distance 0)
            ("HighValue", 18.0, 4)     # At East (distance 0)
        ],

        # Policy
        policy_type="EMA",  # or "UCB"
        eta=0.12,           # Learning rate
        beta_reg=1.5,       # Region selection temperature (lower = more exploration)
        beta_src=2.5,       # Source selection temperature
        cost_c=0.2,         # Migration cost

        # Simulation
        n_proposers=80,     # Total number of proposers
        K=8,                # Concurrent proposers per slot
        n_slots=10000,      # Simulation length
        seed=42
    ))

    return experiments
```

## Available Metrics

All metrics are computed over time and can be plotted/compared across experiments.

### Metric Categories

**Inequality/Diversity**:
- `proposer_dist_gini`, `region_gini`, `source_gini` - Gini coefficient (0=equal, 1=concentrated)
- `proposer_dist_entropy`, `region_entropy`, `source_entropy` - Normalized entropy (0=concentrated, 1=uniform)

**Concentration (HHI)**:
- `proposer_dist_hhi` - Population concentration across regions
- `region_hhi` - Active region selection concentration per slot
- `source_hhi` - Source selection concentration
- `value_share_hhi` - Economic concentration (value-capture)

**Value-Capture (Economic Decentralization)**:
- `value_share_top1` - Share of total value captured by top region
- `value_share_top3` - Share captured by top 3 regions
- `value_share_entropy` - Diversity of value distribution

**Stability/Volatility**:
- `region_volatility` - L1 change in region selection distribution between slots
- `proposer_dist_volatility` - L1 change in population distribution
- `value_share_volatility` - L1 change in value-capture distribution

**Performance**:
- `reward` - Average reward per proposer

### Usage

```python
# Compare experiments
compare_experiments(
    results,
    metrics=["proposer_dist_hhi", "value_share_hhi", "region_volatility", "reward"],
    save_plots=True
)

# Access data
result.proposer_dist_hhi_over_time  # Time series
result.value_capture_by_region      # (n_slots, n_regions) array
result.value_share_distribution     # (n_slots, n_regions) array

# Analysis functions
result.compute_average_volatility('region', window=2000)
result.compute_time_to_convergence(metric_series, window=500, epsilon=0.01)
```

**Interpretation**:
- **Lower** is better: Gini, HHI, Volatility (more equal/stable)
- **Higher** is better: Entropy (more diverse)

## File Structure

### Core Library

- **`mcp_simulator.py`** - Core simulator with Omega mechanism, policies, and tracking
- **`experiment_runner.py`** - Experiment framework with configuration, comparison, and visualization

### Experiments & Demos

- **`my_experiments.py`** - Main experiment file (edit this for your experiments)

### Documentation

- **`README.md`** - This file

## References

This simulator was inspired by research on geographic decentralization in blockchain systems:

- Paper: [arXiv:2509.21475v2](https://arxiv.org/pdf/2509.21475v2)
- Original repo: [geographical-decentralization-simulation](https://github.com/syang-ng/geographical-decentralization-simulation)

The simulator has been generalized to study location choice incentives across various distributed and decentralized block building regimes.

## License

Open source - feel free to modify and extend for your research.
