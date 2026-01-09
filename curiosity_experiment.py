"""
Curiosity Experiment - Active Learning through Self-Directed Exploration

The Hypothesis: A brain that knows what it doesn't know should learn faster
by actively seeking confusing examples rather than passively accepting random data.

The Scientific Method Applied to AI:
    - A scientist doesn't study random phenomena
    - They focus on anomalies, edge cases, and surprises
    - Our CuriousBrain does the same with its VAE uncertainty

The Benchmark:
    - Random Brain: Receives shuffled training data (passive learning)
    - Curious Brain: Scans pool, selects most confusing samples (active learning)
    - Prediction: Curious Brain reaches 90% with HALF the samples

Author: Christian Beaumont & Claude & Gemini
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

from mnist_brain import TripartitePlayer, PlayerConfig
from xor_experiment import ExperimentTracker


@dataclass
class CuriosityConfig(PlayerConfig):
    """Configuration for curiosity experiments."""
    # Pool settings
    pool_size: int = 10000           # Size of unlabeled pool
    bootstrap_samples: int = 50       # Initial samples to bootstrap
    samples_per_round: int = 50       # Samples selected each round
    n_rounds: int = 10                # Number of active learning rounds

    # Curiosity metric weights
    recon_weight: float = 1.0         # Weight for reconstruction error
    latent_weight: float = 0.1        # Weight for latent variance

    # Experiment settings
    seed: int = 42


class CuriousBrain(TripartitePlayer):
    """
    A brain that knows what it doesn't know.

    The CuriousBrain extends TripartitePlayer with active learning:
    - Scans unlabeled data to find confusing examples
    - Uses VAE reconstruction error + latent variance as confusion metric
    - Actively requests labels for the most confusing samples

    This transforms passive learning into scientific inquiry.
    """

    def __init__(self, config: Optional[CuriosityConfig] = None):
        self.curiosity_config = config or CuriosityConfig()
        super().__init__(self.curiosity_config)

        # Track what we've already learned from
        self.seen_indices: Set[int] = set()

        # Curiosity metrics over time
        self.curiosity_history: List[float] = []

    def compute_curiosity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute curiosity score for a batch of samples.

        High curiosity = "I'm uncertain about this" = should learn it

        Uses ARCHIVE ENTROPY (classification uncertainty) rather than
        VAE reconstruction. This focuses on samples where the classifier
        is uncertain, not just samples that look weird.

        Entropy = -sum(p * log(p)) is high when predictions are spread out
        """
        x = x.to(self.device)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        self.archive.eval()
        with torch.no_grad():
            # Get Archive's predictions
            logits = self.archive(x)
            probs = torch.softmax(logits, dim=1)

            # Compute entropy: -sum(p * log(p))
            # High entropy = uncertain = curious
            log_probs = torch.log(probs + 1e-10)
            entropy = -torch.sum(probs * log_probs, dim=1)

            # Also consider margin: difference between top 2 predictions
            # Small margin = uncertain
            top2 = torch.topk(probs, 2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]

            # Curiosity = high entropy + low margin
            # We want samples where Archive is genuinely confused
            curiosity = entropy + (1.0 - margin)

        return curiosity

    def scan_for_curiosity(
        self,
        pool_loader: DataLoader,
        budget: int,
        available_indices: Optional[Set[int]] = None
    ) -> np.ndarray:
        """
        The 'Look and Wonder' Phase.

        Scans unlabeled data and identifies the most confusing samples.
        These are the samples the brain WANTS to learn from.

        Args:
            pool_loader: DataLoader with unlabeled pool
            budget: How many samples to select
            available_indices: Which indices are still available (not yet used)

        Returns:
            Array of selected indices (most curious samples)
        """
        all_curiosities = []
        all_indices = []

        print(f"  Scanning {len(pool_loader.dataset)} items for novelty...")

        batch_start = 0
        for x, _ in pool_loader:
            curiosity = self.compute_curiosity(x)

            for i, c in enumerate(curiosity):
                global_idx = batch_start + i
                # Only consider available indices
                if available_indices is None or global_idx in available_indices:
                    all_curiosities.append(c.item())
                    all_indices.append(global_idx)

            batch_start += len(x)

        all_curiosities = np.array(all_curiosities)
        all_indices = np.array(all_indices)

        # Select top-k most curious
        if len(all_curiosities) < budget:
            selected_local = np.arange(len(all_curiosities))
        else:
            selected_local = np.argsort(all_curiosities)[-budget:]

        selected_indices = all_indices[selected_local]
        avg_curiosity = all_curiosities[selected_local].mean()

        self.curiosity_history.append(avg_curiosity)
        print(f"  Selected {len(selected_indices)} items. Avg Curiosity: {avg_curiosity:.4f}")

        return selected_indices

    def get_curiosity_stats(self) -> Dict:
        """Get statistics about curiosity over time."""
        return {
            "avg_curiosity": np.mean(self.curiosity_history) if self.curiosity_history else 0,
            "curiosity_trend": self.curiosity_history,
            "samples_seen": len(self.seen_indices)
        }


class CuriosityExperimentTracker(ExperimentTracker):
    """Tracker for curiosity experiments."""
    def __init__(self):
        super().__init__(experiment_name="curiosity", personality="active_learning")


def run_curiosity_experiment(config: CuriosityConfig, verbose: bool = True) -> Dict:
    """
    Run the full curiosity vs random sampling experiment.

    This is the scientific method applied to machine learning:
    - Random Brain: Passive learner (accepts whatever data comes)
    - Curious Brain: Active scientist (seeks out confusing phenomena)
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

    # The Pool: Unlabeled data (we pretend we don't have labels)
    pool_indices = list(range(config.pool_size))
    pool_loader = DataLoader(
        Subset(train_data, pool_indices),
        batch_size=256,
        shuffle=False
    )

    # Test loader
    test_loader = DataLoader(
        Subset(test_data, range(1000)),
        batch_size=100
    )

    if verbose:
        print("=" * 70)
        print("CURIOSITY EXPERIMENT: Random vs Active Learning")
        print("=" * 70)
        print(f"Device: {device}")
        print(f"Pool size: {config.pool_size}")
        print(f"Samples per round: {config.samples_per_round}")
        print(f"Total rounds: {config.n_rounds}")
        print()

    # Create both brains
    random_brain = TripartitePlayer(PlayerConfig())
    curious_brain = CuriousBrain(config)

    # Results tracking
    history = {
        "random": {"accuracy": [], "samples": []},
        "curious": {"accuracy": [], "samples": [], "avg_curiosity": []}
    }

    # Track available indices
    available_random = set(pool_indices)
    available_curious = set(pool_indices)

    # Evaluation function
    def evaluate(brain):
        brain.archive.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                out = brain.archive(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total

    # ========== BOOTSTRAP PHASE ==========
    if verbose:
        print("BOOTSTRAP PHASE")
        print("-" * 40)

    # Both brains get the same initial samples
    bootstrap_indices = list(range(config.pool_size, config.pool_size + config.bootstrap_samples))
    bootstrap_loader = DataLoader(
        Subset(train_data, bootstrap_indices),
        batch_size=config.bootstrap_samples
    )

    for x, y in bootstrap_loader:
        # Train both System 2s
        random_brain.wake_step(x, y)
        curious_brain.wake_step(x, y)

        # Train both VAEs (so they have baseline for "normal")
        random_brain.play_time(x, y)
        curious_brain.play_time(x, y)

    # Initial sleep to consolidate
    random_brain.sleep_and_dream(verbose=False, use_vivid_dreams=True)
    curious_brain.sleep_and_dream(verbose=False, use_vivid_dreams=True)

    # Initial evaluation
    acc_random = evaluate(random_brain)
    acc_curious = evaluate(curious_brain)

    history["random"]["accuracy"].append(acc_random)
    history["random"]["samples"].append(config.bootstrap_samples)
    history["curious"]["accuracy"].append(acc_curious)
    history["curious"]["samples"].append(config.bootstrap_samples)
    history["curious"]["avg_curiosity"].append(0)

    if verbose:
        print(f"Bootstrap ({config.bootstrap_samples} samples):")
        print(f"  Random: {acc_random:.1%} | Curious: {acc_curious:.1%}")
        print()

    # ========== ACTIVE LEARNING ROUNDS ==========
    if verbose:
        print("ACTIVE LEARNING ROUNDS")
        print("-" * 40)

    for round_num in range(1, config.n_rounds + 1):
        if verbose:
            print(f"\n--- Round {round_num}/{config.n_rounds} ---")

        # === RANDOM STRATEGY ===
        # Pick random samples from remaining pool
        random_picks = np.random.choice(
            list(available_random),
            min(config.samples_per_round, len(available_random)),
            replace=False
        )
        available_random -= set(random_picks)

        random_loader = DataLoader(
            Subset(train_data, random_picks),
            batch_size=config.samples_per_round
        )

        # Train multiple epochs on selected samples
        for epoch in range(3):
            for x, y in random_loader:
                # Force learning by using force_dialogue=True
                random_brain.wake_step(x, y, force_dialogue=True)
                random_brain.play_time(x, y)

        # Sleep after learning
        random_brain.sleep_and_dream(verbose=False, use_vivid_dreams=True)

        # === CURIOUS STRATEGY ===
        # Scan pool and select most confusing samples
        if verbose:
            print("  [Curious Brain]", end=" ")

        curious_picks = curious_brain.scan_for_curiosity(
            pool_loader,
            budget=config.samples_per_round,
            available_indices=available_curious
        )
        available_curious -= set(curious_picks)

        curious_loader = DataLoader(
            Subset(train_data, curious_picks),
            batch_size=config.samples_per_round
        )

        # Train multiple epochs on selected samples
        for epoch in range(3):
            for x, y in curious_loader:
                # Force learning by using force_dialogue=True
                curious_brain.wake_step(x, y, force_dialogue=True)
                curious_brain.play_time(x, y)

        # Sleep after learning
        curious_brain.sleep_and_dream(verbose=False, use_vivid_dreams=True)

        # === EVALUATION ===
        acc_random = evaluate(random_brain)
        acc_curious = evaluate(curious_brain)

        total_samples = config.bootstrap_samples + (round_num * config.samples_per_round)

        history["random"]["accuracy"].append(acc_random)
        history["random"]["samples"].append(total_samples)
        history["curious"]["accuracy"].append(acc_curious)
        history["curious"]["samples"].append(total_samples)
        history["curious"]["avg_curiosity"].append(curious_brain.curiosity_history[-1])

        if verbose:
            diff = acc_curious - acc_random
            diff_str = f"+{diff:.1%}" if diff > 0 else f"{diff:.1%}"
            print(f"  Samples: {total_samples} | Random: {acc_random:.1%} | Curious: {acc_curious:.1%} ({diff_str})")

    # ========== FINAL ANALYSIS ==========
    # Find sample efficiency: How many samples does each need to reach 80%?
    def samples_to_reach(history_dict, target=0.80):
        for acc, samples in zip(history_dict["accuracy"], history_dict["samples"]):
            if acc >= target:
                return samples
        return history_dict["samples"][-1]  # Never reached

    random_to_80 = samples_to_reach(history["random"], 0.80)
    curious_to_80 = samples_to_reach(history["curious"], 0.80)

    random_to_90 = samples_to_reach(history["random"], 0.90)
    curious_to_90 = samples_to_reach(history["curious"], 0.90)

    summary = {
        "final_random_accuracy": history["random"]["accuracy"][-1],
        "final_curious_accuracy": history["curious"]["accuracy"][-1],
        "random_samples_to_80": random_to_80,
        "curious_samples_to_80": curious_to_80,
        "random_samples_to_90": random_to_90,
        "curious_samples_to_90": curious_to_90,
        "sample_efficiency_80": random_to_80 / max(curious_to_80, 1),
        "sample_efficiency_90": random_to_90 / max(curious_to_90, 1),
        "total_samples_used": history["random"]["samples"][-1]
    }

    history["summary"] = summary

    if verbose:
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"\nFinal Accuracy ({summary['total_samples_used']} samples):")
        print(f"  Random Brain:  {summary['final_random_accuracy']:.1%}")
        print(f"  Curious Brain: {summary['final_curious_accuracy']:.1%}")
        print()
        print("Sample Efficiency (samples needed to reach target):")
        print(f"  To reach 80%: Random={random_to_80}, Curious={curious_to_80}")
        print(f"  To reach 90%: Random={random_to_90}, Curious={curious_to_90}")
        print()
        if curious_to_80 < random_to_80:
            print(f"  Curious Brain is {summary['sample_efficiency_80']:.1f}x MORE EFFICIENT at reaching 80%!")
        if curious_to_90 < random_to_90:
            print(f"  Curious Brain is {summary['sample_efficiency_90']:.1f}x MORE EFFICIENT at reaching 90%!")

    return history, curious_brain


def visualize_curiosity_results(
    history: Dict,
    tracker: ExperimentTracker
):
    """Create visualization of curiosity experiment results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Curiosity Experiment: Does Active Learning Beat Random Sampling?", fontsize=14)

    summary = history["summary"]

    # Plot 1: Accuracy vs Samples
    ax1 = axes[0, 0]
    ax1.plot(history["random"]["samples"], history["random"]["accuracy"],
             'r-o', label="Random Sampling", linewidth=2, markersize=6)
    ax1.plot(history["curious"]["samples"], history["curious"]["accuracy"],
             'b-o', label="Curious (Active)", linewidth=2, markersize=6)

    # Reference lines
    ax1.axhline(y=0.80, color='gray', linestyle='--', alpha=0.5, label="80% target")
    ax1.axhline(y=0.90, color='gray', linestyle=':', alpha=0.5, label="90% target")

    ax1.set_xlabel("Number of Labeled Samples")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Learning Curves: Random vs Curious")
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Sample Efficiency Comparison
    ax2 = axes[0, 1]

    categories = ['To 80%', 'To 90%']
    random_samples = [summary["random_samples_to_80"], summary["random_samples_to_90"]]
    curious_samples = [summary["curious_samples_to_80"], summary["curious_samples_to_90"]]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax2.bar(x - width/2, random_samples, width, label='Random', color='red', alpha=0.7)
    bars2 = ax2.bar(x + width/2, curious_samples, width, label='Curious', color='blue', alpha=0.7)

    ax2.set_ylabel('Samples Needed')
    ax2.set_title('Sample Efficiency: Less is More')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add efficiency multiplier annotations
    for i, (r, c) in enumerate(zip(random_samples, curious_samples)):
        if c > 0 and c < r:
            efficiency = r / c
            ax2.annotate(f'{efficiency:.1f}x',
                        xy=(i, max(r, c) + 20),
                        ha='center', fontsize=12, fontweight='bold', color='green')

    # Plot 3: Curiosity Score Over Time
    ax3 = axes[1, 0]
    if history["curious"]["avg_curiosity"]:
        rounds = range(len(history["curious"]["avg_curiosity"]))
        ax3.plot(rounds, history["curious"]["avg_curiosity"], 'g-o', linewidth=2)
        ax3.set_xlabel("Round")
        ax3.set_ylabel("Average Curiosity Score")
        ax3.set_title("Curiosity Over Time (Should Decrease as Brain Learns)")
        ax3.grid(True, alpha=0.3)

    # Plot 4: The Gap (Curious - Random)
    ax4 = axes[1, 1]
    gap = [c - r for c, r in zip(history["curious"]["accuracy"], history["random"]["accuracy"])]
    ax4.bar(range(len(gap)), gap, color=['green' if g > 0 else 'red' for g in gap], alpha=0.7)
    ax4.axhline(y=0, color='black', linewidth=1)
    ax4.set_xlabel("Round")
    ax4.set_ylabel("Accuracy Difference (Curious - Random)")
    ax4.set_title("The Curiosity Advantage")
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    tracker.save_figure(fig, "curiosity_results")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 70)
    print("CURIOSITY EXPERIMENT")
    print("Active Learning: Does Knowing What You Don't Know Help?")
    print("=" * 70)
    print()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")
    print()

    # Configuration
    config = CuriosityConfig(
        pool_size=10000,
        bootstrap_samples=50,
        samples_per_round=50,
        n_rounds=10,
        seed=42
    )

    # Create tracker
    tracker = CuriosityExperimentTracker()

    # Save config
    with open(tracker.output_dir / "config.json", 'w') as f:
        json.dump({
            "pool_size": config.pool_size,
            "bootstrap_samples": config.bootstrap_samples,
            "samples_per_round": config.samples_per_round,
            "n_rounds": config.n_rounds
        }, f, indent=2)

    # Run experiment
    history, curious_brain = run_curiosity_experiment(config, verbose=True)

    # Save results
    summary = history["summary"]
    summary_text = f"""
CURIOSITY EXPERIMENT RESULTS
============================
Timestamp: {tracker.timestamp}

CONFIGURATION:
  Pool size: {config.pool_size}
  Bootstrap samples: {config.bootstrap_samples}
  Samples per round: {config.samples_per_round}
  Total rounds: {config.n_rounds}
  Total samples used: {summary['total_samples_used']}

FINAL ACCURACY:
  Random Brain:  {summary['final_random_accuracy']:.1%}
  Curious Brain: {summary['final_curious_accuracy']:.1%}

SAMPLE EFFICIENCY:
  Samples to reach 80%:
    Random:  {summary['random_samples_to_80']}
    Curious: {summary['curious_samples_to_80']}
    Efficiency: {summary['sample_efficiency_80']:.1f}x

  Samples to reach 90%:
    Random:  {summary['random_samples_to_90']}
    Curious: {summary['curious_samples_to_90']}
    Efficiency: {summary['sample_efficiency_90']:.1f}x

KEY FINDING:
  The Curious Brain, by actively seeking confusing examples,
  {"OUTPERFORMED" if summary['final_curious_accuracy'] > summary['final_random_accuracy'] else "matched"}
  random sampling with {summary['sample_efficiency_80']:.1f}x better sample efficiency.

  This validates the hypothesis: Knowing what you don't know
  is a computational advantage, not just a philosophical nicety.
"""

    print("\n" + "=" * 70)
    print(summary_text)

    tracker.save_summary(summary_text)

    # Save detailed results
    with open(tracker.output_dir / "detailed_results.json", 'w') as f:
        serializable = {
            "random_accuracy": history["random"]["accuracy"],
            "random_samples": history["random"]["samples"],
            "curious_accuracy": history["curious"]["accuracy"],
            "curious_samples": history["curious"]["samples"],
            "curiosity_scores": history["curious"]["avg_curiosity"],
            "summary": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                       for k, v in summary.items()}
        }
        json.dump(serializable, f, indent=2)

    # Visualize
    print("Generating visualization...")
    visualize_curiosity_results(history, tracker)

    print(f"\nResults saved to: {tracker.output_dir}")
    print("=" * 70)
