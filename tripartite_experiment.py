"""
Tripartite Brain Experiment - Testing System 1 + System 2 Integration

This experiment tests the "learning to be lazy" hypothesis:
    - Day 1: Everything is new, Dialogue handles all
    - Day 2+: Archive takes over known patterns
    - Task switch: Archive fails, Dialogue wakes up
    - After sleep: New patterns migrate to Archive

We measure:
    1. Archive usage rate over time (should increase)
    2. Cognitive load reduction (Dialogue calls should decrease)
    3. Response to task switches (shock and recovery)
    4. Memory consolidation effectiveness

The key prediction: By Day 3-4, the system should be handling
90%+ of routine queries via Archive, with Dialogue only waking
for genuine novelty.

Author: Christian Beaumont & Claude
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json

from tripartite_brain import (
    TripartiteBrain,
    ArchiveConfig,
    MemoryBufferConfig,
    create_tripartite_brain
)
from xor_experiment import (
    ExperimentTracker,
    XORDataGenerator,
    SingleNetworkBaseline
)


@dataclass
class TripartiteExperimentConfig:
    """Configuration for multi-day tripartite experiment."""
    num_days: int = 5              # Total days to simulate
    steps_per_day: int = 200       # Wake steps per day
    batch_size: int = 32
    noise_std: float = 0.1
    seed: int = 42

    # Task schedule: which task to run each day
    # None means same as previous day
    task_schedule: List[Optional[str]] = None  # e.g., ["xor", None, "xnor", None, "xor"]

    # Archive configuration
    archive_confidence_threshold: float = 0.50  # Effective threshold (conf * trust)
    consolidation_epochs: int = 50              # More training for better consolidation

    def __post_init__(self):
        if self.task_schedule is None:
            # Default: XOR for 2 days, switch to XNOR, then back to XOR
            self.task_schedule = ["xor", None, "xnor", None, "xor"]


class TripartiteExperimentTracker(ExperimentTracker):
    """Extended tracker for tripartite experiments."""

    def __init__(self, experiment_name: str = "tripartite"):
        super().__init__(experiment_name=experiment_name, personality="tripartite")

    def save_daily_metrics(self, daily_data: List[Dict]):
        """Save per-day metrics."""
        with open(self.output_dir / "daily_metrics.json", 'w') as f:
            json.dump(daily_data, f, indent=2)


def run_tripartite_experiment(
    config: TripartiteExperimentConfig,
    verbose: bool = True
) -> Tuple[TripartiteBrain, Dict]:
    """
    Run the multi-day tripartite experiment.

    This simulates multiple "days" of operation with sleep cycles,
    testing how the system learns to offload work to the Archive.
    """
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create brain
    archive_config = ArchiveConfig(
        confidence_threshold=config.archive_confidence_threshold,
        consolidation_epochs=config.consolidation_epochs
    )
    brain = create_tripartite_brain(
        input_dim=2,
        hidden_dim=16,
        output_dim=1,
        personality="balanced",
        archive_confidence_threshold=config.archive_confidence_threshold
    )

    # Data generator
    data_gen = XORDataGenerator(noise_std=config.noise_std)

    # Results storage
    all_results = {
        "daily_summaries": [],
        "step_metrics": [],
        "task_switches": [],
    }

    current_task = config.task_schedule[0]

    if verbose:
        print("=" * 60)
        print("TRIPARTITE BRAIN EXPERIMENT")
        print("Testing System 1 (Archive) + System 2 (Dialogue)")
        print("=" * 60)
        print(f"Device: {brain.device}")
        print(f"Days: {config.num_days}")
        print(f"Steps per day: {config.steps_per_day}")
        print(f"Task schedule: {config.task_schedule}")
        print()

    for day in range(config.num_days):
        # Check for task switch
        if day < len(config.task_schedule) and config.task_schedule[day] is not None:
            new_task = config.task_schedule[day]
            if new_task != current_task:
                if verbose:
                    print(f"\n*** TASK SWITCH: {current_task} -> {new_task} ***\n")
                all_results["task_switches"].append({
                    "day": day,
                    "from": current_task,
                    "to": new_task,
                    "step": day * config.steps_per_day
                })
            current_task = new_task

        if verbose:
            print(f"Day {day + 1}: Task = {current_task.upper()}")

        day_step_data = []

        # Wake phase: process inputs
        for step in range(config.steps_per_day):
            x, y = data_gen.generate_batch(config.batch_size, task=current_task)
            result = brain.wake_step(x, y)
            day_step_data.append(result)

            # Progress update
            if verbose and (step + 1) % 50 == 0:
                recent_acc = np.mean([d["accuracy"] for d in day_step_data[-50:]])
                recent_archive = np.mean([1 if d["system_used"] == "archive" else 0 for d in day_step_data[-50:]])
                recent_trust = day_step_data[-1].get("archive_trust", 0)
                recent_eff_conf = np.mean([d.get("effective_confidence", 0) for d in day_step_data[-50:]])
                print(f"  Step {step + 1}: acc={recent_acc:.3f}, arch_usage={recent_archive:.1%}, trust={recent_trust:.2f}, eff_conf={recent_eff_conf:.2f}")

        all_results["step_metrics"].extend(day_step_data)

        # End of day: compute summary and sleep
        archive_usage = np.mean([1 if d["system_used"] == "archive" else 0 for d in day_step_data])
        day_accuracy = np.mean([d["accuracy"] for d in day_step_data])

        # Sleep consolidation
        sleep_result = brain.end_day(verbose=verbose)

        daily_summary = {
            "day": day + 1,
            "task": current_task,
            "archive_usage": archive_usage,
            "accuracy": day_accuracy,
            "memories_consolidated": sleep_result["memories_consolidated"],
            "archive_loss": sleep_result["archive_loss"]
        }
        all_results["daily_summaries"].append(daily_summary)

        if verbose:
            print(f"  Summary: Archive usage = {archive_usage:.1%}, Accuracy = {day_accuracy:.1%}")
            print()

    # Final state
    final_state = brain.get_state()
    all_results["final_state"] = final_state

    if verbose:
        print("=" * 60)
        print("EXPERIMENT COMPLETE")
        print(f"Final Archive usage rate: {final_state['archive_usage_rate']:.1%}")
        print(f"Total Archive calls: {final_state['total_archive_calls']}")
        print(f"Total Dialogue calls: {final_state['total_dialogue_calls']}")
        print("=" * 60)

    return brain, all_results


def visualize_tripartite_results(
    results: Dict,
    config: TripartiteExperimentConfig,
    tracker: Optional[TripartiteExperimentTracker] = None
) -> plt.Figure:
    """
    Create visualization of tripartite experiment results.

    Shows:
        1. Archive usage over time (the "learning to be lazy" curve)
        2. Accuracy over time
        3. Per-day summary bars
        4. System usage breakdown
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Tripartite Brain: Learning to Be Lazy", fontsize=14)

    daily = results["daily_summaries"]
    steps = results["step_metrics"]
    switches = results["task_switches"]

    # Helper for smoothing
    def smooth(data, window=30):
        if len(data) < window:
            return data
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode='same')

    # Calculate step-by-step metrics
    archive_usage_per_step = [1 if s["system_used"] == "archive" else 0 for s in steps]
    accuracy_per_step = [s["accuracy"] for s in steps]
    trust_per_step = [s.get("archive_trust", 0) for s in steps]
    eff_conf_per_step = [s.get("effective_confidence", 0) for s in steps]

    # ========== Plot 1: Archive Usage and Trust Over Time ==========
    ax1 = axes[0, 0]

    usage_smooth = smooth(archive_usage_per_step, window=50)
    trust_smooth = smooth(trust_per_step, window=50)

    ax1.plot(usage_smooth, color="purple", linewidth=2, label="Archive Usage Rate")
    ax1.fill_between(range(len(usage_smooth)), usage_smooth, alpha=0.2, color="purple")
    ax1.plot(trust_smooth, color="orange", linewidth=2, linestyle="--", label="Archive Trust")

    # Mark task switches
    for switch in switches:
        ax1.axvline(x=switch["step"], color="red", linestyle="--", alpha=0.7)
        ax1.text(switch["step"] + 5, 0.95, f"→{switch['to'].upper()}", fontsize=9, color="red")

    # Mark day boundaries
    for day in range(1, config.num_days):
        ax1.axvline(x=day * config.steps_per_day, color="gray", linestyle=":", alpha=0.5)

    ax1.set_ylabel("Rate / Trust")
    ax1.set_xlabel("Step")
    ax1.set_title("Archive Usage & Trust (Trust crashes at task switch!)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ========== Plot 2: Accuracy Over Time ==========
    ax2 = axes[0, 1]

    acc_smooth = smooth(accuracy_per_step, window=50)
    ax2.plot(acc_smooth, color="green", linewidth=2, label="Accuracy")

    for switch in switches:
        ax2.axvline(x=switch["step"], color="red", linestyle="--", alpha=0.7)

    for day in range(1, config.num_days):
        ax2.axvline(x=day * config.steps_per_day, color="gray", linestyle=":", alpha=0.5)

    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Step")
    ax2.set_title("Learning Performance Across Days")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ========== Plot 3: Daily Summary Bars ==========
    ax3 = axes[1, 0]

    days = [d["day"] for d in daily]
    archive_usage = [d["archive_usage"] for d in daily]
    accuracies = [d["accuracy"] for d in daily]
    tasks = [d["task"] for d in daily]

    x = np.arange(len(days))
    width = 0.35

    bars1 = ax3.bar(x - width/2, archive_usage, width, label="Archive Usage", color="purple", alpha=0.7)
    bars2 = ax3.bar(x + width/2, accuracies, width, label="Accuracy", color="green", alpha=0.7)

    # Color-code by task
    task_colors = {"xor": "blue", "xnor": "orange"}
    for i, (bar, task) in enumerate(zip(bars1, tasks)):
        bar.set_edgecolor(task_colors.get(task, "gray"))
        bar.set_linewidth(2)

    ax3.set_ylabel("Rate")
    ax3.set_xlabel("Day")
    ax3.set_title("Daily Performance Summary")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Day {d}\n({t.upper()})" for d, t in zip(days, tasks)])
    ax3.legend()
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3, axis='y')

    # ========== Plot 4: Cognitive Load Reduction ==========
    ax4 = axes[1, 1]

    # Compute dialogue calls per day
    dialogue_per_day = []
    archive_per_day = []
    for day_idx in range(config.num_days):
        start = day_idx * config.steps_per_day
        end = start + config.steps_per_day
        day_steps = steps[start:end]
        dialogue_calls = sum(1 for s in day_steps if s["system_used"] == "dialogue")
        archive_calls = sum(1 for s in day_steps if s["system_used"] == "archive")
        dialogue_per_day.append(dialogue_calls)
        archive_per_day.append(archive_calls)

    ax4.bar(days, dialogue_per_day, label="Dialogue (expensive)", color="red", alpha=0.7)
    ax4.bar(days, archive_per_day, bottom=dialogue_per_day, label="Archive (cheap)", color="blue", alpha=0.7)

    ax4.set_ylabel("Number of Calls")
    ax4.set_xlabel("Day")
    ax4.set_title("Cognitive Load: Expensive vs Cheap Processing")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add efficiency annotation
    if len(dialogue_per_day) >= 2:
        initial_dialogue = dialogue_per_day[0]
        final_dialogue = dialogue_per_day[-1]
        if initial_dialogue > 0:
            reduction = (initial_dialogue - final_dialogue) / initial_dialogue
            ax4.text(0.95, 0.95, f"Dialogue reduction:\n{reduction:.0%}",
                     transform=ax4.transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if tracker:
        tracker.save_figure(fig, "tripartite_results")

    return fig


def compute_efficiency_metrics(results: Dict, config: TripartiteExperimentConfig) -> Dict:
    """Compute headline efficiency metrics."""
    daily = results["daily_summaries"]
    steps = results["step_metrics"]
    final = results["final_state"]

    # Day 1 vs Final day comparison
    day1_archive = daily[0]["archive_usage"]
    final_archive = daily[-1]["archive_usage"]

    # Compute "steady state" efficiency (last day without task switch)
    steady_state_archive = final_archive

    # Total compute saved (assuming Dialogue is 2x cost of Archive)
    # Each Archive call saves ~50% compute vs Dialogue
    total_archive_calls = final["total_archive_calls"]
    total_dialogue_calls = final["total_dialogue_calls"]
    total_calls = total_archive_calls + total_dialogue_calls

    # Compute savings: if everything was Dialogue, cost = 2 * total_calls
    # With Archive: cost = 2 * dialogue_calls + 1 * archive_calls
    if total_calls > 0:
        baseline_cost = 2 * total_calls
        actual_cost = 2 * total_dialogue_calls + 1 * total_archive_calls
        compute_savings = 1 - (actual_cost / baseline_cost)
    else:
        compute_savings = 0

    return {
        "day1_archive_usage": day1_archive,
        "final_archive_usage": final_archive,
        "archive_usage_increase": final_archive - day1_archive,
        "total_archive_calls": total_archive_calls,
        "total_dialogue_calls": total_dialogue_calls,
        "compute_savings": compute_savings,
        "total_memories_consolidated": sum(d["memories_consolidated"] for d in daily),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("TRIPARTITE BRAIN EXPERIMENT")
    print("System 1 (Archive) + System 2 (Dialogue) Integration")
    print("=" * 60)
    print()

    # Check for GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")
    print()

    # Configuration
    config = TripartiteExperimentConfig(
        num_days=5,
        steps_per_day=200,
        batch_size=32,
        task_schedule=["xor", None, "xnor", None, "xor"],  # XOR -> XNOR -> XOR
        archive_confidence_threshold=0.50,  # Effective threshold = model_conf * trust
        seed=42
    )

    # Create tracker
    tracker = TripartiteExperimentTracker(experiment_name="tripartite")
    tracker.save_config(config)

    # Run experiment
    brain, results = run_tripartite_experiment(config, verbose=True)

    # Compute efficiency metrics
    efficiency = compute_efficiency_metrics(results, config)

    # Build summary
    summary_lines = [
        "TRIPARTITE BRAIN EXPERIMENT RESULTS",
        "=" * 50,
        f"Timestamp: {tracker.timestamp}",
        "",
        "CONFIGURATION:",
        f"  Days: {config.num_days}",
        f"  Steps per day: {config.steps_per_day}",
        f"  Task schedule: {config.task_schedule}",
        f"  Archive confidence threshold: {config.archive_confidence_threshold}",
        "",
        "EFFICIENCY METRICS:",
        f"  Day 1 Archive usage: {efficiency['day1_archive_usage']:.1%}",
        f"  Final Archive usage: {efficiency['final_archive_usage']:.1%}",
        f"  Archive usage increase: +{efficiency['archive_usage_increase']:.1%}",
        "",
        f"  Total Archive calls: {efficiency['total_archive_calls']}",
        f"  Total Dialogue calls: {efficiency['total_dialogue_calls']}",
        f"  Compute savings: {efficiency['compute_savings']:.1%}",
        "",
        f"  Total memories consolidated: {efficiency['total_memories_consolidated']}",
        "",
        "KEY FINDINGS:",
        f"  - System learned to offload {efficiency['final_archive_usage']:.0%} of queries to Archive",
        f"  - Compute savings of {efficiency['compute_savings']:.0%} vs pure Dialogue",
        f"  - Archive usage increased by {efficiency['archive_usage_increase']:.0%} from Day 1 to Day {config.num_days}",
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    # Save everything
    print("\n" + "=" * 60)
    print("Saving results...")
    tracker.save_summary(summary_text)

    # Save detailed metrics
    with open(tracker.output_dir / "efficiency_metrics.json", 'w') as f:
        json.dump(efficiency, f, indent=2)

    with open(tracker.output_dir / "daily_summaries.json", 'w') as f:
        json.dump(results["daily_summaries"], f, indent=2)

    # Visualize
    print("Generating visualizations...")
    visualize_tripartite_results(results, config, tracker=tracker)

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print(f"Results saved to: {tracker.output_dir}")
    print("=" * 60)
