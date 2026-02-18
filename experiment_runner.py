#!/usr/bin/env python3
"""
Experiment Runner: Configurable simulator for studying location choice in decentralized building.

This script provides:
1. Centralized configuration for all experiment parameters
2. Easy-to-run experiments with different settings
3. Comparison tools to analyze metrics over time across experiments

Supports various distributed/decentralized block building regimes.
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import json
from pathlib import Path
from mcp_simulator import (
    Region, Source, Proposer, MCPSimulator,
    EMASoftmaxPolicy, UCBPolicy
)

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not found. Install with 'pip install matplotlib' for visualizations.")


# ============================================================================
# CONFIGURATION SECTION - Edit parameters here
# ============================================================================

@dataclass
class ExperimentConfig:
    """All experiment parameters in one place."""

    # Experiment identification
    name: str = "default_experiment"

    # Regions configuration
    n_regions: int = 5
    region_names: List[str] = None  # If None, will auto-generate

    # Sources configuration
    # Each source: (name, value, home_region)
    sources_config: List[tuple] = None

    # Policy configuration
    policy_type: str = "EMA"  # "EMA" or "UCB"

    # EMA policy parameters
    eta: float = 0.12
    beta_reg: float = 1.5
    beta_src: float = 2.5
    cost_c: float = 0.2

    # UCB policy parameters
    alpha: float = 2.0

    # Simulation parameters
    n_proposers: int = 80
    K: int = 8  # Concurrent proposers per slot
    n_slots: int = 10000
    seed: int = 42

    # Output configuration
    save_results: bool = True
    results_dir: str = "experiment_results"

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.region_names is None:
            self.region_names = [f"Region_{i}" for i in range(self.n_regions)]

        if self.sources_config is None:
            # Default: 3 sources spread across regions
            self.sources_config = [
                ("SourceA", 8.0, 0),
                ("SourceB", 12.0, self.n_regions // 2),
                ("SourceC", 18.0, self.n_regions - 1)
            ]


# ============================================================================
# EXPERIMENT PRESETS - Quick configurations for common scenarios
# ============================================================================

def get_preset_config(preset_name: str) -> ExperimentConfig:
    """Get a predefined experiment configuration."""

    presets = {
        "small_uniform": ExperimentConfig(
            name="small_uniform",
            n_regions=3,
            region_names=["West", "Central", "East"],
            sources_config=[
                ("Oracle1", 8.0, 0),
                ("Oracle2", 8.0, 1),
                ("Oracle3", 8.0, 2)
            ],
            policy_type="EMA",
            n_proposers=60,
            K=6,
            n_slots=5000
        ),

        "large_diverse": ExperimentConfig(
            name="large_diverse",
            n_regions=5,
            region_names=["West", "CentralWest", "Central", "CentralEast", "East"],
            sources_config=[
                ("FastOracle", 8.0, 0),
                ("BalancedOracle", 12.0, 2),
                ("PremiumOracle", 18.0, 4)
            ],
            policy_type="EMA",
            eta=0.12,
            beta_reg=1.5,
            beta_src=2.5,
            n_proposers=80,
            K=8,
            n_slots=10000
        ),

        "ucb_exploration": ExperimentConfig(
            name="ucb_exploration",
            n_regions=5,
            sources_config=[
                ("Source1", 10.0, 0),
                ("Source2", 15.0, 2),
                ("Source3", 20.0, 4)
            ],
            policy_type="UCB",
            alpha=2.0,
            n_proposers=80,
            K=8,
            n_slots=10000
        ),

        "high_migration_cost": ExperimentConfig(
            name="high_migration_cost",
            n_regions=5,
            sources_config=[
                ("Source1", 10.0, 0),
                ("Source2", 15.0, 4)
            ],
            policy_type="EMA",
            eta=0.1,
            beta_reg=2.0,
            beta_src=2.0,
            cost_c=1.0,  # High migration cost
            n_proposers=80,
            K=8,
            n_slots=10000
        )
    }

    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

    return presets[preset_name]


# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

class ExperimentResult:
    """Store results from a single experiment."""

    def __init__(self, config: ExperimentConfig, simulator: MCPSimulator):
        self.config = config
        self.simulator = simulator
        self.stats = simulator.get_statistics()

        # Time series data
        self.region_counts = np.array(simulator.region_counts_history)
        self.source_counts = np.array(simulator.source_counts_history)
        self.proposer_distribution = np.array(simulator.proposer_distribution_history)
        self.rewards = [np.mean(r) if r else 0 for r in simulator.reward_history]
        self.region_reward_pairs = simulator.region_reward_pairs_history

        # Compute time-series metrics
        self._compute_time_series_metrics()

    def _compute_time_series_metrics(self):
        """Compute Gini, entropy, HHI, volatility, value-capture, etc. over time."""

        # Helper functions
        def gini(x):
            sorted_x = np.sort(x)
            n = len(x)
            if np.sum(x) == 0:
                return 0.0
            cumsum = np.cumsum(sorted_x)
            return (2 * np.sum((np.arange(1, n+1)) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n

        def entropy(counts):
            """Normalized entropy: H(p) / log(n)"""
            total = np.sum(counts)
            if total == 0:
                return 0.0
            probs = counts / total
            probs = probs[probs > 0]
            n = len(counts)
            return -np.sum(probs * np.log(probs)) / np.log(n) if len(probs) > 0 and n > 1 else 0.0

        def hhi(counts):
            """Herfindahl-Hirschman Index: sum of squared shares"""
            total = np.sum(counts)
            if total == 0:
                return 0.0
            shares = counts / total
            return np.sum(shares ** 2)

        def top_k_concentration(counts, k):
            """Top-k concentration: sum of top k shares"""
            total = np.sum(counts)
            if total == 0:
                return 0.0
            shares = counts / total
            sorted_shares = np.sort(shares)[::-1]  # Descending order
            return np.sum(sorted_shares[:k])

        def l1_distance(p1, p2):
            """L1 distance between two distributions"""
            return np.sum(np.abs(p1 - p2))

        # Initialize metric lists
        # Existing metrics
        self.region_gini_over_time = []
        self.region_entropy_over_time = []
        self.source_gini_over_time = []
        self.source_entropy_over_time = []
        self.proposer_dist_gini_over_time = []
        self.proposer_dist_entropy_over_time = []

        # A) HHI metrics
        self.region_hhi_over_time = []  # Active region selection HHI
        self.proposer_dist_hhi_over_time = []  # Population distribution HHI
        self.source_hhi_over_time = []  # Source selection HHI

        # B) Value-capture metrics
        self.value_capture_by_region = []  # C_r(n) - total value captured per region
        self.value_share_distribution = []  # q_r(n) - value shares
        self.value_share_hhi_over_time = []  # HHI(q(n))
        self.value_share_entropy_over_time = []  # H(q(n))
        self.value_share_top1_over_time = []  # Top-1 concentration
        self.value_share_top3_over_time = []  # Top-3 concentration

        # C) Volatility metrics
        self.region_volatility_over_time = []  # L1 change in active region selection
        self.proposer_dist_volatility_over_time = []  # L1 change in population distribution
        self.value_share_volatility_over_time = []  # L1 change in value shares

        # Previous distributions for computing volatility
        prev_region_shares = None
        prev_proposer_shares = None
        prev_value_shares = None

        n_regions = self.config.n_regions
        n_sources = len(self.config.sources_config)

        # Compute for each time step
        for t in range(len(self.region_counts)):
            # === Existing metrics ===
            self.region_gini_over_time.append(gini(self.region_counts[t]))
            self.region_entropy_over_time.append(entropy(self.region_counts[t]))
            self.source_gini_over_time.append(gini(self.source_counts[t]))
            self.source_entropy_over_time.append(entropy(self.source_counts[t]))
            self.proposer_dist_gini_over_time.append(gini(self.proposer_distribution[t]))
            self.proposer_dist_entropy_over_time.append(entropy(self.proposer_distribution[t]))

            # === A) HHI metrics ===
            self.region_hhi_over_time.append(hhi(self.region_counts[t]))
            self.proposer_dist_hhi_over_time.append(hhi(self.proposer_distribution[t]))
            self.source_hhi_over_time.append(hhi(self.source_counts[t]))

            # === B) Value-capture metrics ===
            # Compute C_r(n) - total value captured per region
            value_by_region = np.zeros(n_regions)
            for region_id, reward in self.region_reward_pairs[t]:
                value_by_region[region_id] += reward

            self.value_capture_by_region.append(value_by_region.copy())

            # Compute value shares q_r(n)
            total_value = np.sum(value_by_region)
            if total_value > 0:
                value_shares = value_by_region / total_value
            else:
                value_shares = np.zeros(n_regions)

            self.value_share_distribution.append(value_shares.copy())
            self.value_share_hhi_over_time.append(hhi(value_by_region))
            self.value_share_entropy_over_time.append(entropy(value_by_region))
            self.value_share_top1_over_time.append(top_k_concentration(value_by_region, 1))
            self.value_share_top3_over_time.append(top_k_concentration(value_by_region, min(3, n_regions)))

            # === C) Volatility metrics ===
            # Convert counts to shares for volatility computation
            region_total = np.sum(self.region_counts[t])
            if region_total > 0:
                current_region_shares = self.region_counts[t] / region_total
            else:
                current_region_shares = np.zeros(n_regions)

            proposer_total = np.sum(self.proposer_distribution[t])
            if proposer_total > 0:
                current_proposer_shares = self.proposer_distribution[t] / proposer_total
            else:
                current_proposer_shares = np.zeros(n_regions)

            # Compute L1 distances
            if prev_region_shares is not None:
                self.region_volatility_over_time.append(l1_distance(current_region_shares, prev_region_shares))
            else:
                self.region_volatility_over_time.append(0.0)

            if prev_proposer_shares is not None:
                self.proposer_dist_volatility_over_time.append(l1_distance(current_proposer_shares, prev_proposer_shares))
            else:
                self.proposer_dist_volatility_over_time.append(0.0)

            if prev_value_shares is not None:
                self.value_share_volatility_over_time.append(l1_distance(value_shares, prev_value_shares))
            else:
                self.value_share_volatility_over_time.append(0.0)

            # Update previous distributions
            prev_region_shares = current_region_shares
            prev_proposer_shares = current_proposer_shares
            prev_value_shares = value_shares

        # Convert lists to numpy arrays for easier manipulation
        self.value_capture_by_region = np.array(self.value_capture_by_region)
        self.value_share_distribution = np.array(self.value_share_distribution)

    def compute_average_volatility(self, metric_name: str, window: int = 2000) -> float:
        """
        Compute average volatility over the last 'window' slots.

        Args:
            metric_name: One of 'region', 'proposer_dist', 'value_share'
            window: Number of slots to average over (default: 2000)

        Returns:
            Average volatility over the window
        """
        if metric_name == 'region':
            data = self.region_volatility_over_time
        elif metric_name == 'proposer_dist':
            data = self.proposer_dist_volatility_over_time
        elif metric_name == 'value_share':
            data = self.value_share_volatility_over_time
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

        if len(data) == 0:
            return 0.0

        # Take last 'window' points or all if fewer
        recent_data = data[-window:]
        return np.mean(recent_data)

    def compute_time_to_convergence(self, metric_series: np.ndarray,
                                    window: int = 500,
                                    epsilon: float = 0.01) -> int:
        """
        Compute time-to-convergence for a metric series.

        Convergence is defined as the first time n after which the metric stays
        within epsilon of its rolling mean for the next 'window' slots.

        T = min{n : |m(t) - mean(m[n:n+W])| <= epsilon for all t in [n, n+W]}

        Args:
            metric_series: Time series of metric values
            window: Rolling window size W (default: 500)
            epsilon: Convergence threshold (default: 0.01)

        Returns:
            Convergence time T, or -1 if never converges
        """
        n_slots = len(metric_series)

        if n_slots < window:
            return -1  # Not enough data

        # Check each potential convergence point
        for n in range(n_slots - window):
            # Compute rolling mean over window [n, n+W]
            window_data = metric_series[n:n+window]
            rolling_mean = np.mean(window_data)

            # Check if all points in window are within epsilon of mean
            deviations = np.abs(window_data - rolling_mean)
            if np.all(deviations <= epsilon):
                return n

        return -1  # Never converged

    def save(self, filepath: Optional[str] = None):
        """Save results to disk."""
        if filepath is None:
            results_dir = Path(self.config.results_dir)
            results_dir.mkdir(exist_ok=True)
            filepath = results_dir / f"{self.config.name}_results.npz"

        # Save numpy arrays and metadata
        np.savez(
            filepath,
            region_counts=self.region_counts,
            source_counts=self.source_counts,
            proposer_distribution=self.proposer_distribution,
            rewards=np.array(self.rewards),
            # Existing metrics
            region_gini_over_time=np.array(self.region_gini_over_time),
            region_entropy_over_time=np.array(self.region_entropy_over_time),
            source_gini_over_time=np.array(self.source_gini_over_time),
            source_entropy_over_time=np.array(self.source_entropy_over_time),
            proposer_dist_gini_over_time=np.array(self.proposer_dist_gini_over_time),
            proposer_dist_entropy_over_time=np.array(self.proposer_dist_entropy_over_time),
            # HHI metrics
            region_hhi_over_time=np.array(self.region_hhi_over_time),
            proposer_dist_hhi_over_time=np.array(self.proposer_dist_hhi_over_time),
            source_hhi_over_time=np.array(self.source_hhi_over_time),
            # Value-capture metrics
            value_capture_by_region=self.value_capture_by_region,
            value_share_distribution=self.value_share_distribution,
            value_share_hhi_over_time=np.array(self.value_share_hhi_over_time),
            value_share_entropy_over_time=np.array(self.value_share_entropy_over_time),
            value_share_top1_over_time=np.array(self.value_share_top1_over_time),
            value_share_top3_over_time=np.array(self.value_share_top3_over_time),
            # Volatility metrics
            region_volatility_over_time=np.array(self.region_volatility_over_time),
            proposer_dist_volatility_over_time=np.array(self.proposer_dist_volatility_over_time),
            value_share_volatility_over_time=np.array(self.value_share_volatility_over_time),
            # Metadata
            config=np.array([asdict(self.config)], dtype=object),
            stats=np.array([self.stats], dtype=object)
        )

        print(f"Results saved to: {filepath}")
        return filepath

    @staticmethod
    def load(filepath: str) -> 'ExperimentResult':
        """Load results from disk."""
        data = np.load(filepath, allow_pickle=True)

        # Reconstruct config
        config_dict = data['config'].item()
        config = ExperimentConfig(**config_dict)

        # Create a minimal result object
        result = object.__new__(ExperimentResult)
        result.config = config
        result.stats = data['stats'].item()
        result.region_counts = data['region_counts']
        result.source_counts = data['source_counts']
        result.proposer_distribution = data['proposer_distribution']
        result.rewards = list(data['rewards'])
        # Existing metrics
        result.region_gini_over_time = list(data['region_gini_over_time'])
        result.region_entropy_over_time = list(data['region_entropy_over_time'])
        result.source_gini_over_time = list(data['source_gini_over_time'])
        result.source_entropy_over_time = list(data['source_entropy_over_time'])
        result.proposer_dist_gini_over_time = list(data['proposer_dist_gini_over_time'])
        result.proposer_dist_entropy_over_time = list(data['proposer_dist_entropy_over_time'])
        # HHI metrics
        result.region_hhi_over_time = list(data.get('region_hhi_over_time', []))
        result.proposer_dist_hhi_over_time = list(data.get('proposer_dist_hhi_over_time', []))
        result.source_hhi_over_time = list(data.get('source_hhi_over_time', []))
        # Value-capture metrics
        result.value_capture_by_region = data.get('value_capture_by_region', np.array([]))
        result.value_share_distribution = data.get('value_share_distribution', np.array([]))
        result.value_share_hhi_over_time = list(data.get('value_share_hhi_over_time', []))
        result.value_share_entropy_over_time = list(data.get('value_share_entropy_over_time', []))
        result.value_share_top1_over_time = list(data.get('value_share_top1_over_time', []))
        result.value_share_top3_over_time = list(data.get('value_share_top3_over_time', []))
        # Volatility metrics
        result.region_volatility_over_time = list(data.get('region_volatility_over_time', []))
        result.proposer_dist_volatility_over_time = list(data.get('proposer_dist_volatility_over_time', []))
        result.value_share_volatility_over_time = list(data.get('value_share_volatility_over_time', []))

        return result


def create_scenario_from_config(config: ExperimentConfig):
    """Create regions, sources, and distance matrix from config."""
    # Create regions
    regions = [Region(i, config.region_names[i]) for i in range(config.n_regions)]

    # Create sources
    sources = []
    for i, (name, value, home_region) in enumerate(config.sources_config):
        sources.append(Source(i, name, value, home_region))

    # Create distance matrix
    distance_matrix = np.zeros((config.n_regions, len(sources)))
    for r in range(config.n_regions):
        for i, source in enumerate(sources):
            distance_matrix[r, i] = abs(r - source.home_region)

    return regions, sources, distance_matrix


def run_experiment(config: ExperimentConfig, verbose: bool = True) -> ExperimentResult:
    """Run a single experiment with given configuration."""

    if verbose:
        print(f"\n{'='*70}")
        print(f"Running Experiment: {config.name}")
        print(f"{'='*70}")
        print(f"Policy: {config.policy_type}")
        print(f"Regions: {config.n_regions}, Sources: {len(config.sources_config)}")
        print(f"Proposers: {config.n_proposers}, K: {config.K}, Slots: {config.n_slots}")

    # Create scenario
    regions, sources, distance_matrix = create_scenario_from_config(config)

    if verbose:
        print(f"\nSources: {[(s.name, f'V={s.value}', f'home={s.home_region}') for s in sources]}")
        print(f"\nDistance Matrix:")
        print(distance_matrix)

    # Create proposers
    proposers = []
    for i in range(config.n_proposers):
        if config.policy_type == "EMA":
            policy = EMASoftmaxPolicy(
                config.n_regions, len(sources),
                eta=config.eta,
                beta_reg=config.beta_reg,
                beta_src=config.beta_src,
                cost_c=config.cost_c
            )
        elif config.policy_type == "UCB":
            policy = UCBPolicy(config.n_regions, len(sources), alpha=config.alpha)
        else:
            raise ValueError(f"Unknown policy: {config.policy_type}")

        proposers.append(Proposer(i, policy))

    # Create and run simulator
    sim = MCPSimulator(regions, sources, proposers, distance_matrix, K=config.K, seed=config.seed)

    if verbose:
        print(f"\nRunning simulation...")

    sim.run(config.n_slots)

    # Create result object
    result = ExperimentResult(config, sim)

    if verbose:
        print_results(result, regions, sources)

    # Save if requested
    if config.save_results:
        result.save()

    return result


def print_results(result: ExperimentResult, regions: List[Region], sources: List[Source]):
    """Print experiment results."""
    stats = result.stats

    print(f"\n{'='*70}")
    print("Results")
    print(f"{'='*70}")
    print(f"Average reward per proposer per slot: {stats['avg_reward']:.4f}")

    print(f"\nProposer distribution across regions (avg over time):")
    for i, count in enumerate(stats['avg_proposer_distribution']):
        print(f"  {regions[i].name:15s}: {count:6.2f} proposers")

    print(f"\nRegion selection per slot (avg proposers per slot):")
    for i, count in enumerate(stats['avg_region_counts']):
        print(f"  {regions[i].name:15s}: {count:6.2f}")

    print(f"\nSource selection (avg selections per slot):")
    for i, count in enumerate(stats['avg_source_counts']):
        print(f"  {sources[i].name:15s} (V={sources[i].value:4.1f}): {count:6.2f}")

    print(f"\nDiversity metrics:")
    print(f"  Proposer Dist Gini:    {stats['proposer_dist_gini']:.4f} (lower = more equal)")
    print(f"  Proposer Dist Entropy: {stats['proposer_dist_entropy']:.4f} (higher = more equal)")
    print(f"  Region Gini:           {stats['region_gini']:.4f}")
    print(f"  Region Entropy:        {stats['region_entropy']:.4f}")
    print(f"  Source Gini:           {stats['source_gini']:.4f}")
    print(f"  Source Entropy:        {stats['source_entropy']:.4f}")


# ============================================================================
# COMPARISON TOOLS
# ============================================================================

def compare_experiments(results: List[ExperimentResult],
                       metrics: List[str] = None,
                       save_plots: bool = True):
    """
    Compare multiple experiments by plotting metrics over time.

    Args:
        results: List of ExperimentResult objects to compare
        metrics: List of metrics to plot. Options:
                 - 'region_gini', 'region_entropy', 'region_hhi'
                 - 'source_gini', 'source_entropy', 'source_hhi'
                 - 'proposer_dist_gini', 'proposer_dist_entropy', 'proposer_dist_hhi'
                 - 'value_share_hhi', 'value_share_entropy'
                 - 'value_share_top1', 'value_share_top3'
                 - 'region_volatility', 'proposer_dist_volatility', 'value_share_volatility'
                 - 'reward'
                 If None, plots default set of metrics
        save_plots: Whether to save plots to disk
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Cannot create comparison plots.")
        return

    if metrics is None:
        metrics = ['proposer_dist_gini', 'proposer_dist_entropy', 'proposer_dist_hhi',
                   'region_volatility', 'value_share_hhi', 'reward']

    print(f"\n[DEBUG] compare_experiments called with {len(results)} results:")
    for i, r in enumerate(results):
        print(f"  [{i}] {r.config.name} ({r.config.policy_type})")

    n_metrics = len(metrics)
    # Use 2x3 grid layout
    n_rows = 2
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing

    # Create descriptive title with experiment names
    exp_names = ", ".join([r.config.name for r in results])
    if len(exp_names) > 80:
        exp_names = ", ".join([r.config.name for r in results[:3]])
        if len(results) > 3:
            exp_names += f", +{len(results)-3} more"
    fig.suptitle(f'Experiment Comparison: {exp_names}', fontsize=14, fontweight='bold')

    # Color scheme and line styles for better distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    line_styles = ['-', '--', '-.', ':']  # Solid, dashed, dash-dot, dotted
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']  # Different markers

    for metric_idx, metric in enumerate(metrics):
        if metric_idx >= len(axes):
            print(f"Warning: More metrics ({n_metrics}) than available subplots ({len(axes)})")
            break
        ax = axes[metric_idx]

        for result_idx, result in enumerate(results):
            # Get data for this metric
            if metric == 'region_gini':
                data = result.region_gini_over_time
                ylabel = 'Region Gini (lower = more diverse)'
            elif metric == 'region_entropy':
                data = result.region_entropy_over_time
                ylabel = 'Region Entropy (higher = more diverse)'
            elif metric == 'region_hhi':
                data = result.region_hhi_over_time
                ylabel = 'Region HHI (lower = more diverse)'
            elif metric == 'source_gini':
                data = result.source_gini_over_time
                ylabel = 'Source Gini (lower = more diverse)'
            elif metric == 'source_entropy':
                data = result.source_entropy_over_time
                ylabel = 'Source Entropy (higher = more diverse)'
            elif metric == 'source_hhi':
                data = result.source_hhi_over_time
                ylabel = 'Source HHI (lower = more diverse)'
            elif metric == 'proposer_dist_gini':
                data = result.proposer_dist_gini_over_time
                ylabel = 'Proposer Distribution Gini (lower = more equal)'
            elif metric == 'proposer_dist_entropy':
                data = result.proposer_dist_entropy_over_time
                ylabel = 'Proposer Distribution Entropy (higher = more equal)'
            elif metric == 'proposer_dist_hhi':
                data = result.proposer_dist_hhi_over_time
                ylabel = 'Proposer Distribution HHI (lower = more equal)'
            elif metric == 'value_share_hhi':
                data = result.value_share_hhi_over_time
                ylabel = 'Value-Capture HHI (lower = more equal)'
            elif metric == 'value_share_entropy':
                data = result.value_share_entropy_over_time
                ylabel = 'Value-Capture Entropy (higher = more equal)'
            elif metric == 'value_share_top1':
                data = result.value_share_top1_over_time
                ylabel = 'Value-Capture Top-1 Concentration'
            elif metric == 'value_share_top3':
                data = result.value_share_top3_over_time
                ylabel = 'Value-Capture Top-3 Concentration'
            elif metric == 'region_volatility':
                data = result.region_volatility_over_time
                ylabel = 'Region Selection Volatility (L1 change)'
            elif metric == 'proposer_dist_volatility':
                data = result.proposer_dist_volatility_over_time
                ylabel = 'Population Distribution Volatility (L1 change)'
            elif metric == 'value_share_volatility':
                data = result.value_share_volatility_over_time
                ylabel = 'Value-Share Volatility (L1 change)'
            elif metric == 'reward':
                data = result.rewards
                ylabel = 'Average Reward per Proposer'
            else:
                print(f"Unknown metric: {metric}")
                continue

            # Smooth with moving average
            window = min(100, len(data) // 10)
            if window > 1:
                smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
                slots = np.arange(window-1, len(data))
            else:
                smoothed = data
                slots = np.arange(len(data))

            # Plot with distinct line style
            label = f"{result.config.name} ({result.config.policy_type})"
            linestyle = line_styles[result_idx % len(line_styles)]
            print(f"    Plotting {result.config.name}: {len(smoothed)} points, color idx {result_idx}, style {linestyle}")
            ax.plot(slots, smoothed, label=label, linewidth=2.5,
                   color=colors[result_idx], linestyle=linestyle, alpha=0.9)

        ax.set_xlabel('Slot', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()} Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='best', framealpha=0.95, fontsize=10, edgecolor='black', fancybox=True)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        print(f"  [DEBUG] Plotted {len(results)} lines for metric '{metric}'")

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave 3% space at top for suptitle

    if save_plots:
        # Create descriptive filename with experiment names
        exp_names = "_vs_".join([r.config.name[:15] for r in results[:3]])  # Limit to first 3 names
        if len(results) > 3:
            exp_names += f"_and_{len(results)-3}_more"
        filename = f'comparison_{exp_names}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved to: {filename}")

    plt.close(fig)


def plot_experiment_details(result: ExperimentResult, save_plots: bool = True):
    """
    Plot detailed time-series for a single experiment.

    Creates separate plots for:
    - Region selection over time (stacked area or line plot)
    - Source selection over time
    - Proposer distribution over time
    - Diversity metrics (Gini, Entropy)
    - Rewards

    Args:
        result: ExperimentResult object
        save_plots: Whether to save plots to disk
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Cannot create plots.")
        return

    print(f"\n[DEBUG] Plotting details for experiment: {result.config.name}")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 14))

    # Define grid: 3 rows, 3 columns
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    slots = np.arange(len(result.region_counts))

    # Get dimensions
    n_regions = result.config.n_regions
    n_sources = len(result.config.sources_config)

    # Row 1: Distributions
    # 1. Region Selection Over Time (stacked area)
    ax1 = fig.add_subplot(gs[0, 0])
    region_counts_T = result.region_counts.T  # Transpose for stacking
    colors_regions = plt.cm.tab10(np.linspace(0, 1, n_regions))

    ax1.stackplot(slots, *region_counts_T,
                  labels=[result.config.region_names[i] for i in range(n_regions)],
                  colors=colors_regions, alpha=0.7)
    ax1.set_xlabel('Slot', fontsize=10)
    ax1.set_ylabel('Number of Proposers', fontsize=10)
    ax1.set_title('Region Selection Per Slot', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # 2. Source Selection Over Time (stacked area)
    ax2 = fig.add_subplot(gs[0, 1])
    source_counts_T = result.source_counts.T

    # Get source names from config
    source_names = [f"{src[0]} (V={src[1]})" for src in result.config.sources_config]
    colors_sources = plt.cm.Set2(np.linspace(0, 1, n_sources))

    ax2.stackplot(slots, *source_counts_T,
                  labels=source_names,
                  colors=colors_sources, alpha=0.7)
    ax2.set_xlabel('Slot', fontsize=10)
    ax2.set_ylabel('Number of Proposers', fontsize=10)
    ax2.set_title('Source Selection Per Slot', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # 3. Proposer Distribution Over Time (stacked area)
    ax3 = fig.add_subplot(gs[0, 2])
    proposer_dist_T = result.proposer_distribution.T

    ax3.stackplot(slots, *proposer_dist_T,
                  labels=[result.config.region_names[i] for i in range(n_regions)],
                  colors=colors_regions, alpha=0.7)
    ax3.set_xlabel('Slot', fontsize=10)
    ax3.set_ylabel('Number of Proposers', fontsize=10)
    ax3.set_title('Proposer Distribution (All Proposers)', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    # Row 2: Traditional Metrics
    # 4. Diversity Metrics - Gini
    ax4 = fig.add_subplot(gs[1, 0])

    # Smooth data
    window = min(100, len(result.proposer_dist_gini_over_time) // 10)
    if window > 1:
        smooth_prop_gini = np.convolve(result.proposer_dist_gini_over_time,
                                       np.ones(window)/window, mode='valid')
        smooth_region_gini = np.convolve(result.region_gini_over_time,
                                         np.ones(window)/window, mode='valid')
        smooth_source_gini = np.convolve(result.source_gini_over_time,
                                         np.ones(window)/window, mode='valid')
        slots_smooth = np.arange(window-1, len(slots))
    else:
        smooth_prop_gini = result.proposer_dist_gini_over_time
        smooth_region_gini = result.region_gini_over_time
        smooth_source_gini = result.source_gini_over_time
        slots_smooth = slots

    ax4.plot(slots_smooth, smooth_prop_gini, label='Proposer Distribution', linewidth=2, alpha=0.8)
    ax4.plot(slots_smooth, smooth_region_gini, label='Region Selection', linewidth=2, alpha=0.8)
    ax4.plot(slots_smooth, smooth_source_gini, label='Source Selection', linewidth=2, alpha=0.8)
    ax4.set_xlabel('Slot', fontsize=10)
    ax4.set_ylabel('Gini Coefficient', fontsize=10)
    ax4.set_title('Gini (Inequality) Over Time', fontsize=11, fontweight='bold')
    ax4.legend(loc='best', fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3)

    # 5. Diversity Metrics - Entropy
    ax5 = fig.add_subplot(gs[1, 1])

    if window > 1:
        smooth_prop_ent = np.convolve(result.proposer_dist_entropy_over_time,
                                      np.ones(window)/window, mode='valid')
        smooth_region_ent = np.convolve(result.region_entropy_over_time,
                                        np.ones(window)/window, mode='valid')
        smooth_source_ent = np.convolve(result.source_entropy_over_time,
                                        np.ones(window)/window, mode='valid')
    else:
        smooth_prop_ent = result.proposer_dist_entropy_over_time
        smooth_region_ent = result.region_entropy_over_time
        smooth_source_ent = result.source_entropy_over_time

    ax5.plot(slots_smooth, smooth_prop_ent, label='Proposer Distribution', linewidth=2, alpha=0.8)
    ax5.plot(slots_smooth, smooth_region_ent, label='Region Selection', linewidth=2, alpha=0.8)
    ax5.plot(slots_smooth, smooth_source_ent, label='Source Selection', linewidth=2, alpha=0.8)
    ax5.set_xlabel('Slot', fontsize=10)
    ax5.set_ylabel('Normalized Entropy', fontsize=10)
    ax5.set_title('Entropy (Diversity) Over Time', fontsize=11, fontweight='bold')
    ax5.legend(loc='best', fontsize=9, framealpha=0.9)
    ax5.grid(True, alpha=0.3)

    # 6. Average Reward Over Time
    ax6 = fig.add_subplot(gs[1, 2])

    if window > 1:
        smooth_rewards = np.convolve(result.rewards, np.ones(window)/window, mode='valid')
    else:
        smooth_rewards = result.rewards

    ax6.plot(slots_smooth, smooth_rewards, linewidth=2, color='darkgreen', alpha=0.8)
    ax6.set_xlabel('Slot', fontsize=10)
    ax6.set_ylabel('Average Reward', fontsize=10)
    ax6.set_title('Average Reward Per Proposer', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Row 3: New Metrics
    # 7. HHI Metrics Over Time
    ax7 = fig.add_subplot(gs[2, 0])

    if window > 1:
        smooth_region_hhi = np.convolve(result.region_hhi_over_time,
                                        np.ones(window)/window, mode='valid')
        smooth_prop_hhi = np.convolve(result.proposer_dist_hhi_over_time,
                                      np.ones(window)/window, mode='valid')
        smooth_source_hhi = np.convolve(result.source_hhi_over_time,
                                        np.ones(window)/window, mode='valid')
    else:
        smooth_region_hhi = result.region_hhi_over_time
        smooth_prop_hhi = result.proposer_dist_hhi_over_time
        smooth_source_hhi = result.source_hhi_over_time

    ax7.plot(slots_smooth, smooth_region_hhi, label='Region Selection', linewidth=2, alpha=0.8)
    ax7.plot(slots_smooth, smooth_prop_hhi, label='Population Distribution', linewidth=2, alpha=0.8)
    ax7.plot(slots_smooth, smooth_source_hhi, label='Source Selection', linewidth=2, alpha=0.8)
    ax7.set_xlabel('Slot', fontsize=10)
    ax7.set_ylabel('HHI', fontsize=10)
    ax7.set_title('HHI (Concentration) Over Time', fontsize=11, fontweight='bold')
    ax7.legend(loc='best', fontsize=9, framealpha=0.9)
    ax7.grid(True, alpha=0.3)

    # 8. Value-Capture Concentration
    ax8 = fig.add_subplot(gs[2, 1])

    if window > 1:
        smooth_value_hhi = np.convolve(result.value_share_hhi_over_time,
                                       np.ones(window)/window, mode='valid')
        smooth_value_top1 = np.convolve(result.value_share_top1_over_time,
                                        np.ones(window)/window, mode='valid')
        smooth_value_top3 = np.convolve(result.value_share_top3_over_time,
                                        np.ones(window)/window, mode='valid')
    else:
        smooth_value_hhi = result.value_share_hhi_over_time
        smooth_value_top1 = result.value_share_top1_over_time
        smooth_value_top3 = result.value_share_top3_over_time

    ax8.plot(slots_smooth, smooth_value_hhi, label='HHI', linewidth=2, alpha=0.8, color='darkred')
    ax8.plot(slots_smooth, smooth_value_top1, label='Top-1', linewidth=2, alpha=0.8, color='orange')
    ax8.plot(slots_smooth, smooth_value_top3, label='Top-3', linewidth=2, alpha=0.8, color='gold')
    ax8.set_xlabel('Slot', fontsize=10)
    ax8.set_ylabel('Concentration', fontsize=10)
    ax8.set_title('Value-Capture Concentration', fontsize=11, fontweight='bold')
    ax8.legend(loc='best', fontsize=9, framealpha=0.9)
    ax8.grid(True, alpha=0.3)

    # 9. Volatility (L1 Change) Metrics
    ax9 = fig.add_subplot(gs[2, 2])

    if window > 1:
        smooth_region_vol = np.convolve(result.region_volatility_over_time,
                                        np.ones(window)/window, mode='valid')
        smooth_prop_vol = np.convolve(result.proposer_dist_volatility_over_time,
                                      np.ones(window)/window, mode='valid')
        smooth_value_vol = np.convolve(result.value_share_volatility_over_time,
                                       np.ones(window)/window, mode='valid')
    else:
        smooth_region_vol = result.region_volatility_over_time
        smooth_prop_vol = result.proposer_dist_volatility_over_time
        smooth_value_vol = result.value_share_volatility_over_time

    ax9.plot(slots_smooth, smooth_region_vol, label='Region Selection', linewidth=2, alpha=0.8)
    ax9.plot(slots_smooth, smooth_prop_vol, label='Population Distribution', linewidth=2, alpha=0.8)
    ax9.plot(slots_smooth, smooth_value_vol, label='Value-Capture', linewidth=2, alpha=0.8)
    ax9.set_xlabel('Slot', fontsize=10)
    ax9.set_ylabel('L1 Change', fontsize=10)
    ax9.set_title('Volatility (Distribution Churn) Over Time', fontsize=11, fontweight='bold')
    ax9.legend(loc='best', fontsize=9, framealpha=0.9)
    ax9.grid(True, alpha=0.3)

    # Main title
    policy_info = result.config.policy_type
    if result.config.policy_type == "EMA":
        policy_info += f" (η={result.config.eta}, β_reg={result.config.beta_reg}, β_src={result.config.beta_src}, c={result.config.cost_c})"
    else:
        policy_info += f" (α={result.config.alpha})"

    fig.suptitle(f'Experiment: {result.config.name} | {policy_info}',
                 fontsize=13, fontweight='bold')

    if save_plots:
        results_dir = Path(result.config.results_dir)
        results_dir.mkdir(exist_ok=True)
        filename = results_dir / f"{result.config.name}_details.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Detail plot saved to: {filename}")

    plt.close(fig)


def plot_network_setup(config: ExperimentConfig, save_plots: bool = True):
    """
    Visualize the initial network setup showing regions, sources, and topology.

    Args:
        config: ExperimentConfig object
        save_plots: Whether to save the plot to disk
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Cannot create network setup plot.")
        return

    fig, ax = plt.subplots(figsize=(16, 8))

    n_regions = config.n_regions
    region_names = config.region_names if config.region_names else [f"R{i}" for i in range(n_regions)]

    # Position regions horizontally (equally spaced)
    region_positions = np.linspace(0, 10, n_regions)
    region_y = 5  # Y position for regions

    # Draw regions as circles
    region_radius = 0.4
    for i, (x, name) in enumerate(zip(region_positions, region_names)):
        circle = plt.Circle((x, region_y), region_radius, color='lightblue',
                           ec='darkblue', linewidth=2, alpha=0.7, zorder=2)
        ax.add_patch(circle)
        ax.text(x, region_y, f"{i}", ha='center', va='center',
               fontsize=14, fontweight='bold', zorder=3)
        ax.text(x, region_y - 0.8, name, ha='center', va='top',
               fontsize=11, fontweight='bold')

    # Draw sources at their home regions
    source_y = 7.5  # Y position for sources (above regions)
    source_radius = 0.35

    if config.sources_config:
        for src_name, src_value, home_region in config.sources_config:
            src_x = region_positions[home_region]

            # Draw source as star/diamond
            circle = plt.Circle((src_x, source_y), source_radius, color='gold',
                              ec='darkorange', linewidth=2.5, alpha=0.9, zorder=2)
            ax.add_patch(circle)

            # Source label
            ax.text(src_x, source_y, "S", ha='center', va='center',
                   fontsize=12, fontweight='bold', zorder=3)
            ax.text(src_x, source_y + 0.6, f"{src_name}", ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color='darkorange')
            ax.text(src_x, source_y + 0.9, f"V={src_value}", ha='center', va='bottom',
                   fontsize=9, color='darkred')

            # Draw arrow from source to home region
            ax.annotate('', xy=(src_x, region_y + region_radius),
                       xytext=(src_x, source_y - source_radius),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.6))

    # Draw distance indicators between adjacent regions
    distance_y = 3.5  # Y position for distance labels
    for i in range(n_regions - 1):
        x1, x2 = region_positions[i], region_positions[i + 1]
        mid_x = (x1 + x2) / 2

        # Draw line showing distance
        ax.plot([x1 + region_radius, x2 - region_radius],
               [distance_y, distance_y],
               'k-', linewidth=1, alpha=0.3)

        # Distance label
        ax.text(mid_x, distance_y - 0.3, f"d={1}", ha='center', va='top',
               fontsize=9, style='italic', color='gray')

    # Add legend/info box
    info_text = (
        f"Configuration: {config.name}\n"
        f"Regions: {n_regions}  |  Sources: {len(config.sources_config) if config.sources_config else 0}  |  "
        f"Proposers: {config.n_proposers}  |  K={config.K}\n"
        f"Distance-ranked sharing: w_j = 2^(-j)  |  Migration cost: c={config.cost_c if hasattr(config, 'cost_c') else 'N/A'}\n"
        f"Initial distribution: Uniform ({config.n_proposers // n_regions} proposers per region)"
    )

    ax.text(0.5, 0.08, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Add legend at bottom left, next to config box
    legend_text = (
        "Legend:\n"
        "● Region (numbered by ID)\n"
        "● Information Source (with value V)\n"
        "d = distance units between regions"
    )
    ax.text(0.02, 0.08, legend_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    # Set axis properties
    ax.set_xlim(-1, 11)
    ax.set_ylim(2, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    policy_info = config.policy_type if hasattr(config, 'policy_type') else "N/A"
    ax.set_title(f'Network Setup: {config.name} | Policy: {policy_info}',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_plots:
        results_dir = Path(config.results_dir if hasattr(config, 'results_dir') else 'experiment_results')
        results_dir.mkdir(exist_ok=True)
        filename = results_dir / f"{config.name}_network_setup.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Network setup plot saved to: {filename}")

    plt.close(fig)


def print_comparison_table(results: List[ExperimentResult]):
    """Print a comparison table of final metrics."""
    print(f"\n{'='*120}")
    print("EXPERIMENT COMPARISON TABLE")
    print(f"{'='*120}")

    # Header
    print(f"\n{'Experiment':<20} {'Policy':<8} {'Reward':<10} {'PropDist':<10} {'PropDist':<10} "
          f"{'Region':<10} {'Region':<10} {'Source':<10} {'Source':<10}")
    print(f"{'':20} {'':8} {'':10} {'Gini':<10} {'Entropy':<10} "
          f"{'Gini':<10} {'Entropy':<10} {'Gini':<10} {'Entropy':<10}")
    print("-" * 120)

    # Data rows
    for result in results:
        stats = result.stats
        print(f"{result.config.name:<20} {result.config.policy_type:<8} "
              f"{stats['avg_reward']:<10.4f} {stats['proposer_dist_gini']:<10.4f} "
              f"{stats['proposer_dist_entropy']:<10.4f} {stats['region_gini']:<10.4f} "
              f"{stats['region_entropy']:<10.4f} {stats['source_gini']:<10.4f} "
              f"{stats['source_entropy']:<10.4f}")

    print("="*120)


# Example usage: See my_experiments.py for how to use this framework
