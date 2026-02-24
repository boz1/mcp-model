#!/usr/bin/env python3
"""
My Experiments - Simple template for running custom experiments.

Edit the configurations below and run this script to perform your experiments.
"""

from experiment_runner import (
    ExperimentConfig,
    run_experiment,
    compare_experiments,
    print_comparison_table,
    plot_experiment_details,
    plot_network_setup,
)


# ============================================================================
# SHARED NETWORK TOPOLOGY
# One source per region, all with the same value.  K = number of sources so
# the socially optimal allocation is exactly one proposer per source.
# ============================================================================

REGIONS = ["West", "CentralWest", "Central", "CentralEast", "East"]
SOURCE_VALUE = 10.0  # identical across all sources

SOURCES = [(f"Src_{name}", SOURCE_VALUE, i) for i, name in enumerate(REGIONS)]
K = len(SOURCES)  # concurrent proposers per slot = number of sources

# Params shared by every experiment (override per-experiment as needed)
BASE = dict(
    n_regions=len(REGIONS),
    region_names=REGIONS,
    sources_config=SOURCES,
    K=K,
    n_proposers=80,
    n_slots=10000,
    seed=42,
)


# ============================================================================
# CONFIGURE YOUR EXPERIMENTS HERE
# ============================================================================


def define_experiments():
    """Define all experiments you want to run."""

    experiments = []

    # -- Equal-split policy experiments --

    # Experiment 1: EMA baseline (equal split)
    experiments.append(ExperimentConfig(
        name="ema_equal_baseline",
        **BASE,
        policy_type="EMA",
        eta=0.12,
        beta_reg=1.5,
        beta_src=2.5,
        cost_c=0.2,
        sharing_policy="equal_split",
    ))

    # Experiment 2: EMA high exploration (equal split)
    experiments.append(ExperimentConfig(
        name="ema_equal_highexplore",
        **BASE,
        policy_type="EMA",
        eta=0.12,
        beta_reg=1.0,
        beta_src=1.5,
        cost_c=0.2,
        sharing_policy="equal_split",
    ))

    # Experiment 3: UCB (equal split)
    experiments.append(ExperimentConfig(
        name="ucb_equal_baseline",
        **BASE,
        policy_type="UCB",
        alpha=2.0,
        sharing_policy="equal_split",
    ))

    # Experiment 4: UCB high exploration (equal split)
    experiments.append(ExperimentConfig(
        name="ucb_equal_highexplore",
        **BASE,
        policy_type="UCB",
        alpha=5.0,
        sharing_policy="equal_split",
    ))

    return experiments


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Run all experiments and compare results."""

    print("=" * 70)
    print("Running Custom Experiments")
    print("=" * 70)

    # Define experiments
    configs = define_experiments()

    # Visualize network setup (all experiments share the same topology)
    print("\n\nGenerating network setup visualization...")
    plot_network_setup(configs[0], save_plots=True)

    # Run all experiments
    results = []
    for config in configs:
        result = run_experiment(config, verbose=True)
        results.append(result)
        print(f"✓ Completed: {config.name}")

    # Print comparison table
    print("\n\n")
    print(f"Total experiments completed: {len(results)}")
    print(f"Experiment names: {[r.config.name for r in results]}")
    print_comparison_table(results)

    # Plot comparisons
    print("\n\nGenerating comparison plots...")
    print(f"Plotting {len(results)} experiments: {[r.config.name for r in results]}")

    compare_experiments(
        results,
        metrics=[
            "proposer_dist_hhi",
            "proposer_dist_entropy",
            "value_share_hhi",
            "value_share_top1",
            "region_volatility",
            "reward",
            "poa",
        ],
        save_plots=True,
    )

    # Plot individual experiment details
    print("\n\nGenerating detailed plots for each experiment...")
    for result in results:
        print(f"\nPlotting details for: {result.config.name}")
        plot_experiment_details(result, save_plots=True)

    print("\n" + "=" * 70)
    print("All experiments complete!")
    print(f"Results saved to: experiment_results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
