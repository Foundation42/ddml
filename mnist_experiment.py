"""
Split MNIST Experiment - The Classic Catastrophic Forgetting Test

This is THE benchmark for continual learning:
    Phase 1: Learn digits 0-4
    Phase 2: Learn digits 5-9 (the "shock")
    Test: Does it still remember 0-4?

Standard networks: Drop to <20% on 0-4 (catastrophic forgetting)
Dialogue Model: Should maintain >70% through protective mechanisms

We compare:
    1. Single Network Baseline (always updates everything)
    2. MNIST Tripartite Brain (with dreaming/replay)

Author: Christian Beaumont & Claude
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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

from mnist_brain import MNISTTripartiteBrain, MNISTBrainConfig, VisualCortex
from xor_experiment import ExperimentTracker


@dataclass
class MNISTExperimentConfig:
    """Configuration for Split MNIST experiment."""
    # Data settings
    samples_per_task: int = 5000      # Training samples per task
    test_samples_per_task: int = 1000  # Test samples per task
    batch_size: int = 32

    # Training settings
    epochs_per_task: int = 3          # Epochs per task
    sleep_after_each_epoch: bool = True

    # Brain settings
    archive_hidden_dim: int = 256
    dialogue_hidden_dim: int = 256
    replay_ratio: float = 0.5         # How much to replay during sleep

    seed: int = 42


class SingleNetworkBaseline(nn.Module):
    """
    Standard single network for comparison.

    This represents how most neural networks are trained:
        - One network
        - Always updating
        - No protection against forgetting
    """
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, output_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


def get_mnist_loaders(
    config: MNISTExperimentConfig,
    task: str = "0-4"
) -> Tuple[DataLoader, DataLoader]:
    """
    Get MNIST data loaders for a specific task.

    Args:
        config: Experiment configuration
        task: "0-4" or "5-9"

    Returns:
        (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )

    # Filter by task
    if task == "0-4":
        label_filter = lambda label: label < 5
    elif task == "5-9":
        label_filter = lambda label: label >= 5
    else:
        label_filter = lambda label: True

    # Get indices
    train_indices = [i for i, (_, label) in enumerate(train_dataset) if label_filter(label)]
    test_indices = [i for i, (_, label) in enumerate(test_dataset) if label_filter(label)]

    # Limit samples
    train_indices = train_indices[:config.samples_per_task]
    test_indices = test_indices[:config.test_samples_per_task]

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=config.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_indices),
        batch_size=config.batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def train_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int = 1
) -> List[float]:
    """Train the baseline network."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)

    return epoch_losses


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict:
    """Evaluate a model on test data."""
    model.eval()
    correct = 0
    total = 0
    per_class_correct = {}
    per_class_total = {}

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            pred = out.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

            for p, label in zip(pred, y):
                label_int = label.item()
                if label_int not in per_class_correct:
                    per_class_correct[label_int] = 0
                    per_class_total[label_int] = 0
                per_class_total[label_int] += 1
                if p.item() == label_int:
                    per_class_correct[label_int] += 1

    per_class_acc = {c: per_class_correct[c] / per_class_total[c] for c in per_class_correct}

    return {
        "overall_accuracy": correct / total,
        "per_class_accuracy": per_class_acc
    }


def run_mnist_experiment(
    config: MNISTExperimentConfig,
    verbose: bool = True
) -> Dict:
    """
    Run the full Split MNIST experiment.

    Compares:
        1. Single Network Baseline
        2. MNIST Tripartite Brain with Dreaming
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data loaders
    train_0_4, test_0_4 = get_mnist_loaders(config, "0-4")
    train_5_9, test_5_9 = get_mnist_loaders(config, "5-9")

    # Create all test data loader for full evaluation
    _, test_all = get_mnist_loaders(config, "all")

    if verbose:
        print("=" * 60)
        print("SPLIT MNIST EXPERIMENT")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Samples per task: {config.samples_per_task}")
        print(f"Epochs per task: {config.epochs_per_task}")
        print(f"Replay ratio: {config.replay_ratio}")
        print()

    # Initialize models
    baseline = SingleNetworkBaseline().to(device)
    baseline_opt = optim.Adam(baseline.parameters(), lr=0.001)

    brain_config = MNISTBrainConfig(
        hidden_dim=config.dialogue_hidden_dim,
        archive_hidden_dim=config.archive_hidden_dim,
        replay_ratio=config.replay_ratio
    )
    brain = MNISTTripartiteBrain(brain_config)

    # Results storage
    results = {
        "baseline": {"accuracy_0_4": [], "accuracy_5_9": []},
        "brain": {"accuracy_0_4": [], "accuracy_5_9": [], "archive_trust": [], "archive_usage": []}
    }

    # ========== PHASE 1: Learn 0-4 ==========
    if verbose:
        print("PHASE 1: Learning digits 0-4")
        print("-" * 40)

    for epoch in range(config.epochs_per_task):
        if verbose:
            print(f"\nEpoch {epoch + 1}/{config.epochs_per_task}")

        # Train baseline
        train_baseline(baseline, train_0_4, baseline_opt, device, epochs=1)

        # Train brain
        brain.net_A.train()
        brain.net_B.train()
        step_accuracies = []

        for x, y in train_0_4:
            result = brain.wake_step(x, y)
            step_accuracies.append(result["accuracy"])

        if config.sleep_after_each_epoch:
            brain.sleep_and_dream(verbose=verbose)

        # Evaluate
        baseline_eval = evaluate_model(baseline, test_0_4, device)
        brain_eval = brain.evaluate(test_0_4, use_archive=True)

        results["baseline"]["accuracy_0_4"].append(baseline_eval["overall_accuracy"])
        results["brain"]["accuracy_0_4"].append(brain_eval["overall_accuracy"])
        results["brain"]["archive_trust"].append(brain.archive_trust)

        if verbose:
            print(f"  Baseline acc (0-4): {baseline_eval['overall_accuracy']:.1%}")
            print(f"  Brain acc (0-4):    {brain_eval['overall_accuracy']:.1%}")
            print(f"  Brain archive trust: {brain.archive_trust:.2f}")

    # Record pre-shock accuracy
    pre_shock_baseline = results["baseline"]["accuracy_0_4"][-1]
    pre_shock_brain = results["brain"]["accuracy_0_4"][-1]

    # ========== PHASE 2: The Shock (5-9) ==========
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 2: THE SHOCK - Switching to digits 5-9")
        print("=" * 60)

    for epoch in range(config.epochs_per_task):
        if verbose:
            print(f"\nEpoch {epoch + 1}/{config.epochs_per_task}")

        # Train baseline (overwrites everything!)
        train_baseline(baseline, train_5_9, baseline_opt, device, epochs=1)

        # Train brain
        step_accuracies = []
        for x, y in train_5_9:
            result = brain.wake_step(x, y)
            step_accuracies.append(result["accuracy"])

        if config.sleep_after_each_epoch:
            brain.sleep_and_dream(verbose=verbose)

        # Evaluate on BOTH tasks
        baseline_eval_0_4 = evaluate_model(baseline, test_0_4, device)
        baseline_eval_5_9 = evaluate_model(baseline, test_5_9, device)
        brain_eval_0_4 = brain.evaluate(test_0_4, use_archive=True)
        brain_eval_5_9 = brain.evaluate(test_5_9, use_archive=True)

        results["baseline"]["accuracy_0_4"].append(baseline_eval_0_4["overall_accuracy"])
        results["baseline"]["accuracy_5_9"].append(baseline_eval_5_9["overall_accuracy"])
        results["brain"]["accuracy_0_4"].append(brain_eval_0_4["overall_accuracy"])
        results["brain"]["accuracy_5_9"].append(brain_eval_5_9["overall_accuracy"])
        results["brain"]["archive_trust"].append(brain.archive_trust)

        if verbose:
            print(f"  Baseline acc (0-4): {baseline_eval_0_4['overall_accuracy']:.1%} <- FORGETTING?")
            print(f"  Baseline acc (5-9): {baseline_eval_5_9['overall_accuracy']:.1%}")
            print(f"  Brain acc (0-4):    {brain_eval_0_4['overall_accuracy']:.1%}")
            print(f"  Brain acc (5-9):    {brain_eval_5_9['overall_accuracy']:.1%}")
            print(f"  Brain archive trust: {brain.archive_trust:.2f}")

    # ========== FINAL EVALUATION ==========
    if verbose:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)

    final_baseline_0_4 = evaluate_model(baseline, test_0_4, device)
    final_baseline_5_9 = evaluate_model(baseline, test_5_9, device)
    final_brain_0_4 = brain.evaluate(test_0_4, use_archive=True)
    final_brain_5_9 = brain.evaluate(test_5_9, use_archive=True)

    # Compute forgetting metrics
    baseline_forgetting = pre_shock_baseline - final_baseline_0_4["overall_accuracy"]
    brain_forgetting = pre_shock_brain - final_brain_0_4["overall_accuracy"]

    brain_state = brain.get_state()

    summary = {
        "pre_shock_baseline": pre_shock_baseline,
        "pre_shock_brain": pre_shock_brain,
        "final_baseline_0_4": final_baseline_0_4["overall_accuracy"],
        "final_baseline_5_9": final_baseline_5_9["overall_accuracy"],
        "final_brain_0_4": final_brain_0_4["overall_accuracy"],
        "final_brain_5_9": final_brain_5_9["overall_accuracy"],
        "baseline_forgetting": baseline_forgetting,
        "brain_forgetting": brain_forgetting,
        "forgetting_reduction": baseline_forgetting - brain_forgetting,
        "brain_archive_usage": brain_state["archive_usage_rate"],
        "brain_memory_stats": brain_state["memory_stats"]
    }

    results["summary"] = summary
    results["per_class"] = {
        "baseline_0_4": final_baseline_0_4["per_class_accuracy"],
        "baseline_5_9": final_baseline_5_9["per_class_accuracy"],
        "brain_0_4": final_brain_0_4["per_class_accuracy"],
        "brain_5_9": final_brain_5_9["per_class_accuracy"]
    }

    if verbose:
        print(f"\nBaseline on digits 0-4: {final_baseline_0_4['overall_accuracy']:.1%}")
        print(f"Baseline on digits 5-9: {final_baseline_5_9['overall_accuracy']:.1%}")
        print(f"Brain on digits 0-4:    {final_brain_0_4['overall_accuracy']:.1%}")
        print(f"Brain on digits 5-9:    {final_brain_5_9['overall_accuracy']:.1%}")
        print()
        print(f"FORGETTING (drop in 0-4 accuracy):")
        print(f"  Baseline: {pre_shock_baseline:.1%} -> {final_baseline_0_4['overall_accuracy']:.1%} = -{baseline_forgetting:.1%}")
        print(f"  Brain:    {pre_shock_brain:.1%} -> {final_brain_0_4['overall_accuracy']:.1%} = -{brain_forgetting:.1%}")
        print()
        print(f"Forgetting reduction: {summary['forgetting_reduction']:.1%}")

    return results


def visualize_mnist_results(
    results: Dict,
    config: MNISTExperimentConfig,
    tracker: Optional[ExperimentTracker] = None
) -> plt.Figure:
    """Create visualization of MNIST experiment results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Split MNIST: Catastrophic Forgetting Comparison", fontsize=14)

    summary = results["summary"]
    phase1_epochs = config.epochs_per_task
    total_epochs = phase1_epochs * 2

    # ========== Plot 1: Accuracy on Task A (0-4) ==========
    ax1 = axes[0, 0]
    epochs = range(1, len(results["baseline"]["accuracy_0_4"]) + 1)

    ax1.plot(epochs, results["baseline"]["accuracy_0_4"], 'r-o', label="Baseline", linewidth=2)
    ax1.plot(epochs, results["brain"]["accuracy_0_4"], 'b-o', label="Brain (Archive)", linewidth=2)
    ax1.axvline(x=phase1_epochs + 0.5, color='black', linestyle='--', label="Task Switch")
    ax1.axhspan(0, 0.2, alpha=0.1, color='red', label="Catastrophic Forgetting Zone")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Memory of Digits 0-4 (Task A)")
    ax1.legend(loc="lower left")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # ========== Plot 2: Accuracy on Task B (5-9) ==========
    ax2 = axes[0, 1]

    # Pad Task B accuracy (wasn't measured during Phase 1)
    baseline_5_9_padded = [0] * phase1_epochs + results["baseline"]["accuracy_5_9"]
    brain_5_9_padded = [0] * phase1_epochs + results["brain"]["accuracy_5_9"]

    ax2.plot(epochs, baseline_5_9_padded, 'r-o', label="Baseline", linewidth=2)
    ax2.plot(epochs, brain_5_9_padded, 'b-o', label="Brain (Archive)", linewidth=2)
    ax2.axvline(x=phase1_epochs + 0.5, color='black', linestyle='--', label="Task Switch")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Learning Digits 5-9 (Task B)")
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # ========== Plot 3: Forgetting Comparison ==========
    ax3 = axes[1, 0]

    categories = ['Before\nTask Switch', 'After\nTask Switch']
    baseline_values = [summary["pre_shock_baseline"], summary["final_baseline_0_4"]]
    brain_values = [summary["pre_shock_brain"], summary["final_brain_0_4"]]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax3.bar(x - width/2, baseline_values, width, label='Baseline', color='red', alpha=0.7)
    bars2 = ax3.bar(x + width/2, brain_values, width, label='Brain', color='blue', alpha=0.7)

    ax3.set_ylabel('Accuracy on 0-4')
    ax3.set_title('Catastrophic Forgetting: The Key Result')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()
    ax3.set_ylim(0, 1.1)

    # Add forgetting annotations
    ax3.annotate(f'↓ {summary["baseline_forgetting"]:.0%}',
                xy=(0.2, (baseline_values[0] + baseline_values[1])/2),
                fontsize=12, color='red', fontweight='bold')
    ax3.annotate(f'↓ {summary["brain_forgetting"]:.0%}',
                xy=(1.2, (brain_values[0] + brain_values[1])/2),
                fontsize=12, color='blue', fontweight='bold')

    ax3.grid(True, alpha=0.3, axis='y')

    # ========== Plot 4: Per-Class Accuracy ==========
    ax4 = axes[1, 1]

    # Combine all classes
    all_classes = sorted(set(results["per_class"]["baseline_0_4"].keys()) |
                        set(results["per_class"]["baseline_5_9"].keys()))

    baseline_per_class = []
    brain_per_class = []

    for c in all_classes:
        if c in results["per_class"]["baseline_0_4"]:
            baseline_per_class.append(results["per_class"]["baseline_0_4"][c])
        else:
            baseline_per_class.append(results["per_class"]["baseline_5_9"].get(c, 0))

        if c in results["per_class"]["brain_0_4"]:
            brain_per_class.append(results["per_class"]["brain_0_4"][c])
        else:
            brain_per_class.append(results["per_class"]["brain_5_9"].get(c, 0))

    x = np.arange(len(all_classes))
    width = 0.35

    ax4.bar(x - width/2, baseline_per_class, width, label='Baseline', color='red', alpha=0.7)
    ax4.bar(x + width/2, brain_per_class, width, label='Brain', color='blue', alpha=0.7)
    ax4.axvline(x=4.5, color='black', linestyle='--', alpha=0.5)
    ax4.text(2, 1.02, "Task A", ha='center', fontsize=10)
    ax4.text(7, 1.02, "Task B", ha='center', fontsize=10)

    ax4.set_xlabel('Digit Class')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Per-Class Accuracy')
    ax4.set_xticks(x)
    ax4.set_xticklabels(all_classes)
    ax4.legend()
    ax4.set_ylim(0, 1.15)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if tracker:
        tracker.save_figure(fig, "mnist_results")

    return fig


class MNISTExperimentTracker(ExperimentTracker):
    """Extended tracker for MNIST experiments."""
    def __init__(self):
        super().__init__(experiment_name="mnist", personality="tripartite")


if __name__ == "__main__":
    print("=" * 60)
    print("SPLIT MNIST EXPERIMENT")
    print("Catastrophic Forgetting Benchmark")
    print("=" * 60)
    print()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU (this may be slow)")
    print()

    # Configuration
    config = MNISTExperimentConfig(
        samples_per_task=5000,
        test_samples_per_task=1000,
        epochs_per_task=3,
        batch_size=32,
        replay_ratio=0.5,
        seed=42
    )

    # Create tracker
    tracker = MNISTExperimentTracker()

    # Save config
    with open(tracker.output_dir / "config.json", 'w') as f:
        json.dump({
            "samples_per_task": config.samples_per_task,
            "epochs_per_task": config.epochs_per_task,
            "replay_ratio": config.replay_ratio
        }, f, indent=2)

    # Run experiment
    results = run_mnist_experiment(config, verbose=True)

    # Save results
    summary = results["summary"]
    summary_text = f"""
SPLIT MNIST EXPERIMENT RESULTS
==============================
Timestamp: {tracker.timestamp}

CONFIGURATION:
  Samples per task: {config.samples_per_task}
  Epochs per task: {config.epochs_per_task}
  Replay ratio: {config.replay_ratio}

RESULTS:
  Pre-shock accuracy (0-4):
    Baseline: {summary['pre_shock_baseline']:.1%}
    Brain:    {summary['pre_shock_brain']:.1%}

  Post-shock accuracy (0-4):
    Baseline: {summary['final_baseline_0_4']:.1%}
    Brain:    {summary['final_brain_0_4']:.1%}

  Post-shock accuracy (5-9):
    Baseline: {summary['final_baseline_5_9']:.1%}
    Brain:    {summary['final_brain_5_9']:.1%}

FORGETTING ANALYSIS:
  Baseline forgetting: {summary['baseline_forgetting']:.1%}
  Brain forgetting:    {summary['brain_forgetting']:.1%}
  Forgetting reduced by: {summary['forgetting_reduction']:.1%}

KEY FINDING:
  The Dialogue Model with Dreaming retained {summary['final_brain_0_4']:.0%} accuracy
  on digits 0-4 vs the Baseline's {summary['final_baseline_0_4']:.0%}.
"""

    print("\n" + "=" * 60)
    print(summary_text)

    tracker.save_summary(summary_text)

    # Save detailed results
    with open(tracker.output_dir / "detailed_results.json", 'w') as f:
        # Convert numpy types for JSON serialization
        serializable = {
            "baseline_0_4": [float(x) for x in results["baseline"]["accuracy_0_4"]],
            "baseline_5_9": [float(x) for x in results["baseline"]["accuracy_5_9"]],
            "brain_0_4": [float(x) for x in results["brain"]["accuracy_0_4"]],
            "brain_5_9": [float(x) for x in results["brain"]["accuracy_5_9"]],
            "summary": {k: float(v) if isinstance(v, (int, float)) else v
                       for k, v in summary.items()}
        }
        json.dump(serializable, f, indent=2)

    # Visualize
    print("Generating visualization...")
    visualize_mnist_results(results, config, tracker)

    print(f"\nResults saved to: {tracker.output_dir}")
    print("=" * 60)
