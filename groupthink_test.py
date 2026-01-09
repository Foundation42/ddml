"""
Groupthink Test - Can Shared Backbones Maintain Internal Conflict?

The Hypothesis:
If Network A and Network B share their eyes (backbone), they will see the world
too similarly. When they encounter something strange, they will likely hallucinate
the same wrong answer, leading to False Consensus.

Independent Brains: "I think it's a 7." / "I think it's a 1." → High Conflict (Good).
Shared Backbone: "Our shared eyes see features that look like a 7." / "Agreed." → Low Conflict (Dangerous).

Protocol:
- Train two brains on MNIST Digits 0-4 (known)
- Test disagreement on Digits 5-9 (unknown/OOD)

Success Metric: HIGH Disagreement on unseen digits (brain screams "I don't know!")
Failure Mode: LOW Disagreement = Groupthink (the soul of metacognition is lost)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


class IndependentDialogue(nn.Module):
    """ The Standard Dialogue Model (Control Group) """
    def __init__(self):
        super().__init__()
        # Two completely separate networks
        self.net_A = self._build_net()
        self.net_B = self._build_net()

    def _build_net(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net_A(x), self.net_B(x)


class SharedDialogue(nn.Module):
    """ The Efficient Model (Experimental Group) """
    def __init__(self):
        super().__init__()
        # Shared Backbone (The "Groupthink" Risk)
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU()
        )
        # Separate Heads (Can they maintain independence?)
        self.head_A = nn.Linear(64, 10)
        self.head_B = nn.Linear(64, 10)

    def forward(self, x):
        features = self.backbone(x)
        return self.head_A(features), self.head_B(features)


def train_brain(model, loader, epochs=5):
    """Train both heads on the same task"""
    opt = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for ep in range(epochs):
        total_loss = 0
        for x, y in loader:
            opt.zero_grad()
            out_A, out_B = model(x)
            # Both heads learn the task
            loss = F.cross_entropy(out_A, y) + F.cross_entropy(out_B, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  Epoch {ep+1}: Loss = {total_loss/len(loader):.4f}")


def measure_conflict(model, loader):
    """ Returns average KL Divergence (Disagreement) """
    model.eval()
    conflicts = []
    with torch.no_grad():
        for x, _ in loader:  # Ignore labels, we just want to see if they agree
            out_A, out_B = model(x)

            # Convert logits to probs
            p_A = F.softmax(out_A, dim=1)
            p_B = F.softmax(out_B, dim=1)

            # Measure KL(A || B)
            # High KL = High Conflict (Good for OOD)
            kl = F.kl_div(p_A.log(), p_B, reduction='batchmean')
            conflicts.append(kl.item())

    return np.mean(conflicts)


def measure_accuracy(model, loader):
    """Measure accuracy of both heads"""
    model.eval()
    correct_A, correct_B, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            out_A, out_B = model(x)
            pred_A = out_A.argmax(dim=1)
            pred_B = out_B.argmax(dim=1)
            correct_A += (pred_A == y).sum().item()
            correct_B += (pred_B == y).sum().item()
            total += y.size(0)
    return correct_A / total, correct_B / total


def run_groupthink_test():
    """Execute the Groupthink Test"""

    # Data Setup
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Train on 0-4 (Known)
    idx_known_train = [i for i, t in enumerate(mnist.targets) if t < 5]
    idx_known_test = [i for i, t in enumerate(mnist_test.targets) if t < 5]
    loader_known_train = DataLoader(Subset(mnist, idx_known_train[:5000]), batch_size=32, shuffle=True)
    loader_known_test = DataLoader(Subset(mnist_test, idx_known_test), batch_size=32)

    # Test on 5-9 (Unknown / OOD)
    idx_unknown = [i for i, t in enumerate(mnist_test.targets) if t >= 5]
    loader_unknown = DataLoader(Subset(mnist_test, idx_unknown), batch_size=32)

    print("=" * 60)
    print("           GROUPTHINK TEST")
    print("=" * 60)
    print("\nHypothesis: Shared backbone will reduce conflict on unknown")
    print("           data (Bad for metacognition).\n")
    print("Protocol:")
    print("  - Train on digits 0-4 (known)")
    print("  - Test disagreement on digits 5-9 (OOD)")
    print("-" * 60)

    # 1. Train Independent
    print("\n[1] Training INDEPENDENT Brain (Control)...")
    brain_indep = IndependentDialogue()
    train_brain(brain_indep, loader_known_train)

    acc_indep_A, acc_indep_B = measure_accuracy(brain_indep, loader_known_test)
    conflict_indep_known = measure_conflict(brain_indep, loader_known_test)
    conflict_indep_ood = measure_conflict(brain_indep, loader_unknown)

    print(f"\n  Accuracy on Known (0-4): Head A={acc_indep_A:.1%}, Head B={acc_indep_B:.1%}")
    print(f"  Conflict on Known (0-4):   {conflict_indep_known:.4f}")
    print(f"  Conflict on Unknown (5-9): {conflict_indep_ood:.4f}")

    # 2. Train Shared
    print("\n[2] Training SHARED BACKBONE Brain (Experimental)...")
    brain_shared = SharedDialogue()
    train_brain(brain_shared, loader_known_train)

    acc_shared_A, acc_shared_B = measure_accuracy(brain_shared, loader_known_test)
    conflict_shared_known = measure_conflict(brain_shared, loader_known_test)
    conflict_shared_ood = measure_conflict(brain_shared, loader_unknown)

    print(f"\n  Accuracy on Known (0-4): Head A={acc_shared_A:.1%}, Head B={acc_shared_B:.1%}")
    print(f"  Conflict on Known (0-4):   {conflict_shared_known:.4f}")
    print(f"  Conflict on Unknown (5-9): {conflict_shared_ood:.4f}")

    # 3. The Verdict
    print("\n" + "=" * 60)
    print("                    RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<30} {'Independent':<15} {'Shared':<15}")
    print("-" * 60)
    print(f"{'Accuracy (Known 0-4)':<30} {(acc_indep_A+acc_indep_B)/2:.1%}           {(acc_shared_A+acc_shared_B)/2:.1%}")
    print(f"{'Conflict on Known':<30} {conflict_indep_known:.4f}          {conflict_shared_known:.4f}")
    print(f"{'Conflict on OOD (5-9)':<30} {conflict_indep_ood:.4f}          {conflict_shared_ood:.4f}")

    # The critical metric
    if conflict_indep_ood > 0:
        drop = (conflict_indep_ood - conflict_shared_ood) / conflict_indep_ood * 100
    else:
        drop = 0

    print(f"\n{'OOD Conflict Drop:':<30} {drop:.1f}%")

    print("\n" + "=" * 60)
    print("                    VERDICT")
    print("=" * 60)

    if drop > 50:
        verdict = "CRITICAL FAILURE: Massive Groupthink detected!"
        verdict_detail = "Shared backbone killed metacognition. Option B is DEAD."
    elif drop > 20:
        verdict = "WARNING: Significant reduction in metacognition."
        verdict_detail = "Shared backbone partially impairs uncertainty detection."
    elif drop > 0:
        verdict = "MINOR CONCERN: Small reduction in conflict."
        verdict_detail = "Shared backbone may still be viable with regularization."
    else:
        verdict = "SUCCESS: Shared backbone maintained independence!"
        verdict_detail = "Parameter sharing does NOT cause groupthink."

    print(f"\n>> {verdict}")
    print(f"   {verdict_detail}")

    # Save results
    results = {
        'independent': {
            'accuracy_A': acc_indep_A,
            'accuracy_B': acc_indep_B,
            'conflict_known': conflict_indep_known,
            'conflict_ood': conflict_indep_ood
        },
        'shared': {
            'accuracy_A': acc_shared_A,
            'accuracy_B': acc_shared_B,
            'conflict_known': conflict_shared_known,
            'conflict_ood': conflict_shared_ood
        },
        'conflict_drop_percent': drop,
        'verdict': verdict
    }

    # Create visualization
    create_visualization(results)

    return results


def create_visualization(results):
    """Create a visualization of the groupthink test results"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Conflict comparison
    ax1 = axes[0]
    x = np.arange(2)
    width = 0.35

    indep_conflicts = [results['independent']['conflict_known'],
                       results['independent']['conflict_ood']]
    shared_conflicts = [results['shared']['conflict_known'],
                        results['shared']['conflict_ood']]

    bars1 = ax1.bar(x - width/2, indep_conflicts, width, label='Independent', color='#2ecc71')
    bars2 = ax1.bar(x + width/2, shared_conflicts, width, label='Shared', color='#e74c3c')

    ax1.set_ylabel('Conflict (KL Divergence)')
    ax1.set_title('Internal Conflict: Known vs Unknown Data')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Known (0-4)', 'Unknown (5-9)'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Annotate the critical difference
    if results['conflict_drop_percent'] > 20:
        ax1.annotate(f'{results["conflict_drop_percent"]:.0f}% drop!',
                     xy=(1 + width/2, shared_conflicts[1]),
                     xytext=(1.3, (indep_conflicts[1] + shared_conflicts[1])/2),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontsize=12, color='red')

    # Plot 2: The verdict meter
    ax2 = axes[1]
    drop = results['conflict_drop_percent']

    # Create a gauge-style visualization
    theta = np.linspace(0, np.pi, 100)
    r = 1

    # Background arc
    ax2.fill_between(theta, 0, r, alpha=0.1, color='gray')

    # Zones
    ax2.fill_between(theta[:33], 0, r, alpha=0.3, color='#2ecc71', label='Safe')
    ax2.fill_between(theta[33:66], 0, r, alpha=0.3, color='#f39c12', label='Warning')
    ax2.fill_between(theta[66:], 0, r, alpha=0.3, color='#e74c3c', label='Danger')

    # Needle position based on drop
    needle_theta = np.pi * min(drop, 100) / 100  # 0% = left, 100% = right
    ax2.arrow(0, 0, 0.8*np.cos(np.pi - needle_theta), 0.8*np.sin(np.pi - needle_theta),
              head_width=0.1, head_length=0.05, fc='black', ec='black', linewidth=2)

    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title(f'Groupthink Meter: {drop:.1f}% Conflict Drop', fontsize=12)
    ax2.legend(loc='lower center', ncol=3)

    # Add verdict text
    ax2.text(0, -0.1, results['verdict'].split(':')[0], ha='center', fontsize=14,
             fontweight='bold', color='#e74c3c' if drop > 50 else '#f39c12' if drop > 20 else '#2ecc71')

    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"outputs/groupthink_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/groupthink_results.png", dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_dir}/groupthink_results.png")

    # Save JSON results
    import json
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir}/results.json")


if __name__ == "__main__":
    run_groupthink_test()
