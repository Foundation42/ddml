"""
XOR Experiment - Testing the Dialogue System on the Classic Problem

XOR is perfect for our first test because:
    1. It's non-linearly separable (needs hidden representations)
    2. It's fast to train (seconds, not hours)
    3. We can easily test "task switching" (XOR -> XNOR)
    4. The patterns are interpretable

Experiment Protocol:
    Phase 1: Learn XOR (inputs -> XOR output)
    Phase 2: Task switch to XNOR (opposite pattern)
    Phase 3: (Optional) Switch back to XOR (test memory retention)

We'll track:
    - Surprise dynamics during learning and task switch
    - Confidence hormone behavior
    - Accuracy recovery time after task switch
    - Comparison with single-network baseline

Author: Christian Beaumont & Claude
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - no display needed
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import time

from dialogue_system import (
    DialogueSystem,
    SimpleMLP,
    HormoneConfig,
    LearningConfig,
    create_dialogue_system
)


@dataclass
class ExperimentConfig:
    """Configuration for XOR experiment."""
    phase1_steps: int = 200     # Steps for initial task
    phase2_steps: int = 200     # Steps after task switch
    phase3_steps: int = 100     # Steps for return to original (optional)
    batch_size: int = 32        # Samples per step
    noise_std: float = 0.1      # Input noise (makes it more realistic)
    seed: int = 42              # For reproducibility
    run_phase3: bool = True     # Whether to test memory retention


class ExperimentTracker:
    """
    Handles experiment output: timestamped directories, saving plots and metrics.

    Each run creates a new directory like:
        outputs/xor_2024-01-15_14-30-45_balanced/
            config.json
            metrics.json
            main_results.png
            personality_comparison.png
            summary.txt
    """

    def __init__(self, experiment_name: str = "xor", personality: str = "balanced"):
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_name = experiment_name
        self.personality = personality

        # Create output directory
        self.output_dir = Path("outputs") / f"{experiment_name}_{self.timestamp}_{personality}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Experiment output directory: {self.output_dir}")

    def save_config(self, config: ExperimentConfig, extra: Dict = None):
        """Save experiment configuration."""
        config_dict = asdict(config)
        if extra:
            config_dict.update(extra)
        config_dict['timestamp'] = self.timestamp
        config_dict['personality'] = self.personality

        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

    def save_metrics(self, dialogue_metrics: Dict, baseline_metrics: List[float], results: Dict):
        """Save all metrics for later analysis."""
        metrics = {
            'dialogue': {
                'surprise': dialogue_metrics.surprise_history,
                'confidence': dialogue_metrics.confidence_history,
                'accuracy_A': dialogue_metrics.accuracy_A_history,
                'accuracy_B': dialogue_metrics.accuracy_B_history,
                'agreement': dialogue_metrics.agreement_history,
                'update_triggered': dialogue_metrics.update_triggered,
                'effective_lr': dialogue_metrics.effective_lr_history,
            },
            'baseline': {
                'accuracy': baseline_metrics
            },
            'results': {
                'phase_boundaries': results.get('phase_boundaries', []),
            }
        }

        with open(self.output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

    def save_figure(self, fig, name: str):
        """Save a matplotlib figure."""
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    def save_summary(self, summary_text: str):
        """Save text summary of results."""
        with open(self.output_dir / "summary.txt", 'w') as f:
            f.write(summary_text)

    def get_path(self, filename: str) -> Path:
        """Get full path for a file in the output directory."""
        return self.output_dir / filename


class XORDataGenerator:
    """
    Generates streaming XOR or XNOR data.

    XOR truth table:
        0, 0 -> 0
        0, 1 -> 1
        1, 0 -> 1
        1, 1 -> 0

    XNOR is the opposite:
        0, 0 -> 1
        0, 1 -> 0
        1, 0 -> 0
        1, 1 -> 1
    """

    def __init__(self, noise_std: float = 0.1):
        self.noise_std = noise_std

    def generate_batch(
        self,
        batch_size: int,
        task: str = "xor"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of XOR or XNOR data.

        Args:
            batch_size: Number of samples
            task: "xor" or "xnor"

        Returns:
            (inputs, targets) tensors
        """
        # Generate random binary inputs
        x = torch.randint(0, 2, (batch_size, 2)).float()

        # Add noise to inputs (makes the task more realistic)
        x_noisy = x + torch.randn_like(x) * self.noise_std

        # Compute targets
        if task == "xor":
            # XOR: output is 1 if inputs differ
            y = (x[:, 0] != x[:, 1]).float().unsqueeze(1)
        elif task == "xnor":
            # XNOR: output is 1 if inputs are same
            y = (x[:, 0] == x[:, 1]).float().unsqueeze(1)
        else:
            raise ValueError(f"Unknown task: {task}")

        return x_noisy, y


class SingleNetworkBaseline:
    """
    A simple single-network baseline for comparison.

    This shows what happens WITHOUT the dialogue mechanism:
        - No selective learning (always updates)
        - No confidence tracking
        - Vulnerable to catastrophic forgetting
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, output_dim: int = 1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = SimpleMLP(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.accuracy_history: List[float] = []

    def step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Train on one batch."""
        x = x.to(self.device)
        y = y.to(self.device)

        # Always update (no gating)
        output = self.net(x)
        loss = self.criterion(output, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute accuracy
        predictions = (torch.sigmoid(output) > 0.5).float()
        accuracy = (predictions == y).float().mean().item()
        self.accuracy_history.append(accuracy)

        return {"accuracy": accuracy, "loss": loss.item()}


def run_experiment(
    config: ExperimentConfig,
    personality: str = "balanced",
    verbose: bool = True
) -> Tuple[DialogueSystem, SingleNetworkBaseline, Dict]:
    """
    Run the full XOR experiment.

    Returns:
        - Trained DialogueSystem
        - Trained baseline
        - Results dictionary with all metrics
    """
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create systems
    dialogue = create_dialogue_system(personality=personality, input_dim=2, hidden_dim=16, output_dim=1)
    baseline = SingleNetworkBaseline(input_dim=2, hidden_dim=16, output_dim=1)
    data_gen = XORDataGenerator(noise_std=config.noise_std)

    # Track phase boundaries for plotting
    phase_boundaries = []

    if verbose:
        print(f"Running XOR Experiment")
        print(f"  Personality: {personality}")
        print(f"  Device: {dialogue.device}")
        print(f"  Phase 1 (XOR): {config.phase1_steps} steps")
        print(f"  Phase 2 (XNOR): {config.phase2_steps} steps")
        if config.run_phase3:
            print(f"  Phase 3 (XOR return): {config.phase3_steps} steps")
        print()

    # ========== PHASE 1: Learn XOR ==========
    if verbose:
        print("Phase 1: Learning XOR...")

    for step in range(config.phase1_steps):
        x, y = data_gen.generate_batch(config.batch_size, task="xor")

        dialogue.step(x, y)
        baseline.step(x, y)

        if verbose and (step + 1) % 50 == 0:
            print(f"  Step {step + 1}: Dialogue acc={dialogue.metrics.accuracy_A_history[-1]:.3f}, "
                  f"conf={dialogue.metrics.confidence_history[-1]:.3f}")

    phase_boundaries.append(config.phase1_steps)

    # ========== PHASE 2: Task Switch to XNOR ==========
    if verbose:
        print("\nPhase 2: TASK SWITCH to XNOR! (The 'shock')")

    for step in range(config.phase2_steps):
        x, y = data_gen.generate_batch(config.batch_size, task="xnor")

        dialogue.step(x, y)
        baseline.step(x, y)

        if verbose and (step + 1) % 50 == 0:
            print(f"  Step {step + 1}: Dialogue acc={dialogue.metrics.accuracy_A_history[-1]:.3f}, "
                  f"conf={dialogue.metrics.confidence_history[-1]:.3f}, "
                  f"surprise={dialogue.metrics.surprise_history[-1]:.3f}")

    phase_boundaries.append(config.phase1_steps + config.phase2_steps)

    # ========== PHASE 3: Return to XOR (Memory Test) ==========
    if config.run_phase3:
        if verbose:
            print("\nPhase 3: Returning to XOR (memory retention test)")

        for step in range(config.phase3_steps):
            x, y = data_gen.generate_batch(config.batch_size, task="xor")

            dialogue.step(x, y)
            baseline.step(x, y)

            if verbose and (step + 1) % 50 == 0:
                print(f"  Step {step + 1}: Dialogue acc={dialogue.metrics.accuracy_A_history[-1]:.3f}, "
                      f"conf={dialogue.metrics.confidence_history[-1]:.3f}")

    # Compute summary statistics
    results = {
        "phase_boundaries": phase_boundaries,
        "config": config,
        "personality": personality,
    }

    if verbose:
        print("\nExperiment complete!")

    return dialogue, baseline, results


def visualize_experiment(
    dialogue: DialogueSystem,
    baseline: SingleNetworkBaseline,
    results: Dict,
    tracker: Optional[ExperimentTracker] = None
) -> plt.Figure:
    """
    Create comprehensive visualization of the experiment.

    Shows:
        1. Accuracy comparison (Dialogue vs Baseline)
        2. Internal dynamics (Surprise & Confidence)
        3. Learning efficiency (Update rate)
        4. Agreement between networks A and B

    Returns the figure for saving via tracker.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dialogue System vs Single Network on XOR/XNOR Task Switch", fontsize=14)

    phase_boundaries = results["phase_boundaries"]

    # Helper for smoothing noisy curves
    def smooth(data, window=15):
        if len(data) < window:
            return data
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode='same')

    # ========== Plot 1: Accuracy Comparison ==========
    ax1 = axes[0, 0]

    dialogue_acc = smooth(dialogue.metrics.accuracy_A_history)
    baseline_acc = smooth(baseline.accuracy_history)

    ax1.plot(dialogue_acc, label="Dialogue System (Net A)", color="blue", linewidth=2)
    ax1.plot(baseline_acc, label="Single Network Baseline", color="red", linewidth=2, alpha=0.7)

    # Mark phase boundaries
    for i, boundary in enumerate(phase_boundaries):
        label = "Task Switch" if i == 0 else "Return to XOR"
        ax1.axvline(x=boundary, color="black", linestyle="--", alpha=0.7, label=label if i == 0 else None)

    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Step")
    ax1.set_title("Learning Performance")
    ax1.legend(loc="lower right")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Add phase labels
    ax1.text(phase_boundaries[0] // 2, 0.1, "XOR", ha='center', fontsize=12, alpha=0.5)
    if len(phase_boundaries) >= 2:
        mid_phase2 = (phase_boundaries[0] + phase_boundaries[1]) // 2
        ax1.text(mid_phase2, 0.1, "XNOR", ha='center', fontsize=12, alpha=0.5)

    # ========== Plot 2: Surprise and Confidence ==========
    ax2 = axes[0, 1]

    surprise = smooth(dialogue.metrics.surprise_history, window=10)
    confidence = dialogue.metrics.confidence_history

    ax2.plot(surprise, label="Combined Surprise", color="red", alpha=0.7)
    ax2.plot(confidence, label="Confidence Hormone", color="green", linewidth=2)

    for boundary in phase_boundaries:
        ax2.axvline(x=boundary, color="black", linestyle="--", alpha=0.7)

    ax2.set_ylabel("Level")
    ax2.set_xlabel("Step")
    ax2.set_title("Internal Dynamics: The 'Shock' Response")
    ax2.legend(loc="right")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    # ========== Plot 3: Learning Efficiency (Update Rate) ==========
    ax3 = axes[1, 0]

    # Compute rolling update rate
    update_flags = [1.0 if u else 0.0 for u in dialogue.metrics.update_triggered]
    update_rate = smooth(update_flags, window=30)

    ax3.fill_between(range(len(update_rate)), update_rate, alpha=0.3, color="purple")
    ax3.plot(update_rate, color="purple", linewidth=2, label="Update Rate (% of steps learning)")

    for boundary in phase_boundaries:
        ax3.axvline(x=boundary, color="black", linestyle="--", alpha=0.7)

    ax3.set_ylabel("Update Rate")
    ax3.set_xlabel("Step")
    ax3.set_title("Learning Efficiency: Selective vs Always-On")
    ax3.legend()
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)

    # Add annotation about efficiency
    total_updates = sum(dialogue.metrics.update_triggered)
    total_steps = len(dialogue.metrics.update_triggered)
    efficiency = 1 - (total_updates / total_steps)
    ax3.text(0.02, 0.95, f"Compute saved: {efficiency:.1%}",
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ========== Plot 4: Network Agreement ==========
    ax4 = axes[1, 1]

    agreement = smooth(dialogue.metrics.agreement_history)
    ax4.plot(agreement, color="orange", linewidth=2, label="A-B Agreement Rate")

    # Also show both networks' accuracy
    acc_a = smooth(dialogue.metrics.accuracy_A_history)
    acc_b = smooth(dialogue.metrics.accuracy_B_history)
    ax4.plot(acc_a, color="blue", alpha=0.5, linewidth=1, label="Net A accuracy")
    ax4.plot(acc_b, color="cyan", alpha=0.5, linewidth=1, label="Net B accuracy")

    for boundary in phase_boundaries:
        ax4.axvline(x=boundary, color="black", linestyle="--", alpha=0.7)

    ax4.set_ylabel("Rate")
    ax4.set_xlabel("Step")
    ax4.set_title("Network Agreement & Individual Performance")
    ax4.legend(loc="lower right")
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save via tracker if provided
    if tracker:
        tracker.save_figure(fig, "main_results")

    return fig


def run_personality_comparison(
    config: ExperimentConfig,
    tracker: Optional[ExperimentTracker] = None
) -> plt.Figure:
    """
    Run the experiment with different "personalities" and compare.

    This shows how the alpha parameter affects adaptation vs stability.
    """
    personalities = ["stubborn", "balanced", "anxious"]
    colors = {"stubborn": "red", "balanced": "green", "anxious": "blue"}

    results_all = {}

    print("=" * 60)
    print("PERSONALITY COMPARISON EXPERIMENT")
    print("=" * 60)

    for personality in personalities:
        print(f"\n--- Testing {personality.upper()} personality ---")
        dialogue, baseline, results = run_experiment(config, personality=personality, verbose=False)
        results_all[personality] = {
            "dialogue": dialogue,
            "baseline": baseline,
            "results": results
        }
        print(f"  Final accuracy: {dialogue.metrics.accuracy_A_history[-1]:.3f}")
        print(f"  Final confidence: {dialogue.metrics.confidence_history[-1]:.3f}")

    # Create comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Personality Comparison: How Different Alphas Handle 'Shock'", fontsize=14)

    phase_boundaries = results_all["balanced"]["results"]["phase_boundaries"]

    # Helper for smoothing
    def smooth(data, window=15):
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode='same')

    # Plot 1: Confidence dynamics
    ax1 = axes[0]
    for personality in personalities:
        conf = results_all[personality]["dialogue"].metrics.confidence_history
        ax1.plot(conf, color=colors[personality], label=f"{personality.capitalize()}", linewidth=2)

    for boundary in phase_boundaries:
        ax1.axvline(x=boundary, color="black", linestyle="--", alpha=0.7)

    ax1.set_ylabel("Confidence")
    ax1.set_title("Confidence Hormone Response to Task Switch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy
    ax2 = axes[1]
    for personality in personalities:
        acc = smooth(results_all[personality]["dialogue"].metrics.accuracy_A_history)
        ax2.plot(acc, color=colors[personality], label=f"{personality.capitalize()}", linewidth=2)

    # Also show baseline for reference
    baseline_acc = smooth(results_all["balanced"]["baseline"].accuracy_history)
    ax2.plot(baseline_acc, color="gray", linestyle=":", label="Single Net Baseline", linewidth=2)

    for boundary in phase_boundaries:
        ax2.axvline(x=boundary, color="black", linestyle="--", alpha=0.7)

    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Step")
    ax2.set_title("Learning Performance by Personality")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if tracker:
        tracker.save_figure(fig, "personality_comparison")

    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("DIALOGUE-DRIVEN MACHINE LEARNING")
    print("XOR Experiment - Stage 1 Proof of Concept")
    print("=" * 60)
    print()

    # Check for GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU (XOR is fast enough, no worries!)")
    print()

    # Configuration
    config = ExperimentConfig(
        phase1_steps=200,
        phase2_steps=200,
        phase3_steps=100,
        batch_size=32,
        run_phase3=True
    )

    # Create experiment tracker for this run
    tracker = ExperimentTracker(experiment_name="xor", personality="balanced")
    tracker.save_config(config, extra={"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})

    # Run main experiment
    dialogue, baseline, results = run_experiment(config, personality="balanced", verbose=True)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Compute some summary stats
    phase1_end = results["phase_boundaries"][0]
    phase2_end = results["phase_boundaries"][1]

    # Phase 1 final performance
    phase1_dialogue_acc = np.mean(dialogue.metrics.accuracy_A_history[phase1_end-20:phase1_end])
    phase1_baseline_acc = np.mean(baseline.accuracy_history[phase1_end-20:phase1_end])

    # Phase 2 final performance
    phase2_dialogue_acc = np.mean(dialogue.metrics.accuracy_A_history[phase2_end-20:phase2_end])
    phase2_baseline_acc = np.mean(baseline.accuracy_history[phase2_end-20:phase2_end])

    # Recovery time (steps to reach 80% accuracy after switch)
    def find_recovery_time(acc_history, start_idx, threshold=0.8):
        for i, acc in enumerate(acc_history[start_idx:]):
            if acc > threshold:
                return i
        return -1

    dialogue_recovery = find_recovery_time(dialogue.metrics.accuracy_A_history, phase1_end)
    baseline_recovery = find_recovery_time(baseline.accuracy_history, phase1_end)

    # Compute efficiency
    total_updates = sum(dialogue.metrics.update_triggered)
    total_steps = len(dialogue.metrics.update_triggered)
    compute_saved = 1 - (total_updates / total_steps)

    # Build summary text
    summary_lines = [
        "DIALOGUE-DRIVEN MACHINE LEARNING - XOR EXPERIMENT",
        "=" * 50,
        f"Timestamp: {tracker.timestamp}",
        f"Personality: balanced",
        "",
        "CONFIGURATION:",
        f"  Phase 1 (XOR): {config.phase1_steps} steps",
        f"  Phase 2 (XNOR): {config.phase2_steps} steps",
        f"  Phase 3 (XOR return): {config.phase3_steps} steps",
        f"  Batch size: {config.batch_size}",
        f"  Seed: {config.seed}",
        "",
        "RESULTS:",
        f"  Phase 1 (XOR) Final Accuracy:",
        f"    Dialogue System: {phase1_dialogue_acc:.1%}",
        f"    Single Baseline: {phase1_baseline_acc:.1%}",
        "",
        f"  Phase 2 (XNOR) Final Accuracy:",
        f"    Dialogue System: {phase2_dialogue_acc:.1%}",
        f"    Single Baseline: {phase2_baseline_acc:.1%}",
        "",
        f"  Recovery Time (steps to 80% after switch):",
        f"    Dialogue System: {dialogue_recovery} steps",
        f"    Single Baseline: {baseline_recovery} steps",
        f"    Speedup: {baseline_recovery / max(dialogue_recovery, 1):.1f}x" if dialogue_recovery > 0 else "    Speedup: N/A",
        "",
        f"  Learning Efficiency:",
        f"    Dialogue updates: {total_updates}/{total_steps} steps ({total_updates/total_steps:.1%})",
        f"    Compute saved: {compute_saved:.1%}",
        "",
        "KEY FINDINGS:",
        f"  - Dialogue system recovered {baseline_recovery - dialogue_recovery} steps faster" if dialogue_recovery > 0 else "  - Recovery time could not be measured",
        f"  - Selective learning saved {compute_saved:.0%} of compute",
        f"  - Post-switch accuracy: Dialogue {phase2_dialogue_acc:.0%} vs Baseline {phase2_baseline_acc:.0%}",
    ]

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Save everything
    print("\n" + "=" * 60)
    print("Saving results...")
    tracker.save_metrics(dialogue.metrics, baseline.accuracy_history, results)
    tracker.save_summary(summary_text)

    # Generate visualizations
    print("Generating visualizations...")
    visualize_experiment(dialogue, baseline, results, tracker=tracker)

    # Also run personality comparison
    print("\n" + "=" * 60)
    print("Running personality comparison experiment...")
    run_personality_comparison(config, tracker=tracker)

    print("\n" + "=" * 60)
    print("All experiments complete!")
    print(f"Results saved to: {tracker.output_dir}")
    print("=" * 60)
