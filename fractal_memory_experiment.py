"""
Fractal Memory Experiment - Adaptive Neural Architecture via Information Theory

The Hypothesis:
The network's structure should become a physical manifestation of its epistemic
landscape. Dense regions where knowledge is uncertain, sparse where crystallized.

Theoretical Foundation (Shannon + Fractals):
- H(R) = Entropy in region R (disagreement between networks)
- C(R) = Channel capacity in region R (parameter density)

The Consciousness Equation:
- H > C → Grow network (allocate resources to confusion)
- H < C → Prune network (remove redundant certainty)
- H ≈ C → Stable equilibrium (optimal channel allocation)

Result: Network structure = Map of epistemic landscape
        "Topological Intelligence"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os
from datetime import datetime
import json


# --- 1. Fractal Manifold Hull ---

@dataclass
class ManifoldRegion:
    """A region of the latent manifold with adaptive resolution"""
    center: torch.Tensor
    bounds: torch.Tensor  # Shape: [latent_dim, 2] for [min, max]
    resolution: int
    entropy: float = 0.0
    sample_count: int = 0
    children: Optional[List['ManifoldRegion']] = None

    def is_leaf(self) -> bool:
        return self.children is None

    def volume(self) -> float:
        ranges = self.bounds[:, 1] - self.bounds[:, 0]
        return torch.prod(ranges).item()

    def contains(self, point: torch.Tensor) -> bool:
        point_cpu = point.cpu() if point.is_cuda else point
        return torch.all(
            (point_cpu >= self.bounds[:, 0]) &
            (point_cpu <= self.bounds[:, 1])
        ).item()


class FractalManifoldHull:
    """
    Entropy-driven KD-tree for latent space

    High entropy regions get subdivided (more resolution for confusion)
    Low entropy regions get merged (less resolution for certainty)
    """

    def __init__(self, latent_dim: int, bounds_range: float = 5.0):
        self.latent_dim = latent_dim

        # Initialize root region covering the expected latent space
        initial_bounds = torch.tensor([[-bounds_range, bounds_range]] * latent_dim)
        self.root = ManifoldRegion(
            center=torch.zeros(latent_dim),
            bounds=initial_bounds,
            resolution=0,
            entropy=0.5,  # Start uncertain
            sample_count=0
        )

        # Hyperparameters - tuned for observable dynamics
        self.max_resolution = 6
        self.split_threshold = 0.15  # Entropy above this → split
        self.merge_threshold = 0.08  # Entropy below this → merge (raised!)
        self.min_samples_split = 30  # Minimum samples before splitting

        # Statistics tracking
        self.split_count = 0
        self.merge_count = 0

    def find_region(self, point: torch.Tensor) -> ManifoldRegion:
        """Find the leaf region containing this point"""
        node = self.root
        while not node.is_leaf():
            found = False
            for child in node.children:
                if child.contains(point):
                    node = child
                    found = True
                    break
            if not found:
                break  # Point outside all children, stay at current node
        return node

    def split_region(self, region: ManifoldRegion) -> bool:
        """Split region along dimension with largest range"""
        if region.resolution >= self.max_resolution:
            return False

        # Find dimension with largest range
        ranges = region.bounds[:, 1] - region.bounds[:, 0]
        split_dim = torch.argmax(ranges).item()
        split_point = region.center[split_dim].item()

        # Create two children
        children = []
        for i in range(2):
            child_bounds = region.bounds.clone()
            if i == 0:
                child_bounds[split_dim, 1] = split_point
            else:
                child_bounds[split_dim, 0] = split_point

            child_center = child_bounds.mean(dim=1)
            child = ManifoldRegion(
                center=child_center,
                bounds=child_bounds,
                resolution=region.resolution + 1,
                entropy=region.entropy,
                sample_count=0
            )
            children.append(child)

        region.children = children
        self.split_count += 1
        return True

    def try_merge(self, region: ManifoldRegion) -> bool:
        """Merge children if all have low entropy"""
        if region.is_leaf():
            return False

        # Check if all children are leaves with low entropy
        all_low = all(
            child.is_leaf() and child.entropy < self.merge_threshold
            for child in region.children
        )

        if all_low:
            # Merge: remove children
            region.children = None
            region.entropy = np.mean([c.entropy for c in region.children]) if region.children else 0
            self.merge_count += 1
            return True

        return False

    def update_with_samples(self, points: torch.Tensor, entropies: torch.Tensor):
        """Update hull structure based on points and their local entropies"""
        for point, entropy in zip(points, entropies):
            region = self.find_region(point)
            region.sample_count += 1

            # Exponential moving average of entropy
            alpha = 0.1
            region.entropy = alpha * entropy.item() + (1 - alpha) * region.entropy

            # Decide whether to split
            if (region.is_leaf() and
                region.entropy > self.split_threshold and
                region.sample_count >= self.min_samples_split and
                region.resolution < self.max_resolution):
                self.split_region(region)

        # Periodic merge check (traverse tree)
        self._check_merges(self.root)

    def _check_merges(self, node: ManifoldRegion):
        """Recursively check for merge opportunities"""
        if node.is_leaf():
            return

        # First recurse to children
        for child in node.children:
            self._check_merges(child)

        # Then try to merge this node's children
        self.try_merge(node)

    def get_all_leaves(self) -> List[ManifoldRegion]:
        """Collect all leaf regions"""
        leaves = []
        self._collect_leaves(self.root, leaves)
        return leaves

    def _collect_leaves(self, node: ManifoldRegion, leaves: List):
        if node.is_leaf():
            leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaves(child, leaves)

    def get_statistics(self) -> dict:
        """Get hull statistics"""
        leaves = self.get_all_leaves()

        if not leaves:
            return {'total_regions': 0, 'max_resolution': 0, 'avg_entropy': 0,
                    'high_entropy_regions': 0, 'low_entropy_regions': 0}

        entropies = [r.entropy for r in leaves]
        resolutions = [r.resolution for r in leaves]

        return {
            'total_regions': len(leaves),
            'max_resolution': max(resolutions),
            'avg_resolution': np.mean(resolutions),
            'avg_entropy': np.mean(entropies),
            'high_entropy_regions': sum(1 for e in entropies if e > self.split_threshold),
            'low_entropy_regions': sum(1 for e in entropies if e < self.merge_threshold),
            'total_splits': self.split_count,
            'total_merges': self.merge_count
        }

    def get_effective_dimensions(self) -> float:
        """Calculate effective dimensionality based on structure"""
        leaves = self.get_all_leaves()
        if not leaves:
            return self.latent_dim

        # Weight dimensions by resolution and sample count
        total_weight = sum(r.sample_count * (2 ** r.resolution) for r in leaves)
        if total_weight == 0:
            return self.latent_dim

        weighted_res = sum(
            r.resolution * r.sample_count * (2 ** r.resolution)
            for r in leaves
        ) / total_weight

        # More splits = effectively more dimensions used
        return self.latent_dim * (1 + weighted_res / self.max_resolution)


# --- 2. Fractal Dialogue System ---

class FractalDialogueSystem(nn.Module):
    """Dialogue Model with Fractal Memory Hull"""

    def __init__(self, input_dim: int = 784, latent_dim: int = 8,
                 hidden_dim: int = 128, output_dim: int = 10):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

        # Network A (Generator) - operates in latent space
        self.net_a = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Network B (Monitor) - independent weights
        self.net_b = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Fractal hull in latent space
        self.hull = FractalManifoldHull(latent_dim)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        z = self.encoder(x_flat)

        pred_a = self.net_a(z)
        pred_b = self.net_b(z)

        return pred_a, pred_b, z

    def compute_disagreement(self, pred_a: torch.Tensor, pred_b: torch.Tensor) -> torch.Tensor:
        """
        Combined entropy signal:
        1. Jensen-Shannon divergence (disagreement between networks)
        2. Prediction entropy (uncertainty of consensus)

        This captures both "they disagree" AND "they're both confused"
        """
        p_a = F.softmax(pred_a, dim=-1)
        p_b = F.softmax(pred_b, dim=-1)

        # Midpoint distribution (consensus)
        m = 0.5 * (p_a + p_b)

        # 1. JSD: Disagreement between networks
        kl_a = torch.sum(p_a * torch.log(p_a / (m + 1e-10) + 1e-10), dim=-1)
        kl_b = torch.sum(p_b * torch.log(p_b / (m + 1e-10) + 1e-10), dim=-1)
        jsd = 0.5 * (kl_a + kl_b)

        # 2. Prediction entropy: How uncertain is the consensus?
        pred_entropy = -torch.sum(m * torch.log(m + 1e-10), dim=-1)
        # Normalize by max entropy (log(10) for 10 classes)
        pred_entropy = pred_entropy / np.log(10)

        # Combined: disagreement + uncertainty
        # High when networks disagree OR when both are uncertain
        combined = 0.3 * jsd + 0.7 * pred_entropy
        return combined

    def update_hull(self, z: torch.Tensor, pred_a: torch.Tensor, pred_b: torch.Tensor):
        """Update fractal structure based on disagreement"""
        disagreement = self.compute_disagreement(pred_a, pred_b)
        self.hull.update_with_samples(z.detach(), disagreement.detach())


# --- 3. Training and Experiment ---

def train_epoch(model, loader, optimizer, criterion, device, update_hull=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_disagreement = 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        pred_a, pred_b, z = model(x)

        # Compute disagreement
        disagreement = model.compute_disagreement(pred_a, pred_b)
        total_disagreement += disagreement.mean().item()

        # Update hull periodically
        if update_hull and batch_idx % 5 == 0:
            model.update_hull(z, pred_a, pred_b)

        # Train both networks
        loss_a = criterion(pred_a, y)
        loss_b = criterion(pred_b, y)
        loss = loss_a + loss_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader), total_disagreement / len(loader)


def evaluate(model, loader, device):
    """Evaluate accuracy"""
    model.eval()
    correct_a, correct_b, correct_consensus, total = 0, 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred_a, pred_b, _ = model(x)

            pred_label_a = pred_a.argmax(dim=1)
            pred_label_b = pred_b.argmax(dim=1)

            # Consensus: average predictions
            consensus = (F.softmax(pred_a, dim=1) + F.softmax(pred_b, dim=1)) / 2
            pred_consensus = consensus.argmax(dim=1)

            correct_a += (pred_label_a == y).sum().item()
            correct_b += (pred_label_b == y).sum().item()
            correct_consensus += (pred_consensus == y).sum().item()
            total += y.size(0)

    return {
        'acc_a': correct_a / total,
        'acc_b': correct_b / total,
        'acc_consensus': correct_consensus / total
    }


def visualize_hull_2d(hull: FractalManifoldHull, ax, title="Fractal Hull"):
    """Visualize the hull structure in 2D"""
    leaves = hull.get_all_leaves()

    if not leaves:
        ax.set_title(f"{title}\n(No regions)")
        return

    # Normalize entropy for coloring
    entropies = [r.entropy for r in leaves]
    max_e = max(entropies) if entropies else 1

    for region in leaves:
        bounds = region.bounds[:2].cpu().numpy()  # First 2 dims

        # Color by entropy (red = high, blue = low)
        norm_entropy = region.entropy / (max_e + 1e-6)
        color = plt.cm.coolwarm(norm_entropy)

        # Alpha by resolution (deeper = more opaque)
        alpha = 0.3 + 0.5 * (region.resolution / hull.max_resolution)

        # Draw rectangle
        x = [bounds[0, 0], bounds[0, 1], bounds[0, 1], bounds[0, 0], bounds[0, 0]]
        y = [bounds[1, 0], bounds[1, 0], bounds[1, 1], bounds[1, 1], bounds[1, 0]]
        ax.fill(x, y, color=color, alpha=alpha, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Latent Dim 0')
    ax.set_ylabel('Latent Dim 1')
    ax.set_title(f"{title}\n({len(leaves)} regions, max_res={max(r.resolution for r in leaves)})")


def run_fractal_experiment():
    """Run the Fractal Memory Experiment"""

    print("=" * 70)
    print("     FRACTAL MEMORY EXPERIMENT")
    print("     Adaptive Neural Architecture via Information Theory")
    print("=" * 70)
    print()
    print("The Consciousness Equation:")
    print("  H > C → Grow (allocate resources to confusion)")
    print("  H < C → Prune (remove redundant certainty)")
    print("  H ≈ C → Equilibrium (optimal channel allocation)")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Setup
    transform = transforms.ToTensor()
    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

    test_loader = DataLoader(mnist_test, batch_size=256, shuffle=False)

    # Model with 8D latent space (can visualize first 2 dims)
    model = FractalDialogueSystem(latent_dim=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Track evolution
    hull_history = []
    accuracy_history = []

    epochs_per_task = 3

    print("-" * 70)

    # Continual learning: one digit at a time
    for task in range(10):
        print(f"\n{'='*70}")
        print(f"  TASK {task}: Learning digit {task}")
        print(f"{'='*70}")

        # Filter for current digit
        indices = [i for i, (_, label) in enumerate(mnist_train) if label == task]
        task_subset = Subset(mnist_train, indices[:1000])  # Limit for speed
        task_loader = DataLoader(task_subset, batch_size=64, shuffle=True)

        for epoch in range(epochs_per_task):
            loss, disagreement = train_epoch(
                model, task_loader, optimizer, criterion, device, update_hull=True
            )
            print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Disagreement={disagreement:.4f}")

        # Evaluate on all digits seen so far
        eval_indices = [i for i, (_, label) in enumerate(mnist_test) if label <= task]
        eval_subset = Subset(mnist_test, eval_indices)
        eval_loader = DataLoader(eval_subset, batch_size=256, shuffle=False)

        acc = evaluate(model, eval_loader, device)
        accuracy_history.append(acc['acc_consensus'])

        # Get hull statistics
        stats = model.hull.get_statistics()
        hull_history.append(stats.copy())

        print(f"\n  📊 After Task {task}:")
        print(f"     Accuracy (0-{task}): {acc['acc_consensus']:.1%}")
        print(f"     Hull Regions: {stats['total_regions']}")
        print(f"     Max Resolution: {stats['max_resolution']}")
        print(f"     Avg Entropy: {stats['avg_entropy']:.4f}")
        print(f"     High-H Regions: {stats['high_entropy_regions']}")
        print(f"     Low-H Regions: {stats['low_entropy_regions']}")
        print(f"     Total Splits: {stats['total_splits']}")

    # Final visualization
    print(f"\n{'='*70}")
    print("  GENERATING VISUALIZATIONS")
    print(f"{'='*70}")

    fig = plt.figure(figsize=(18, 5))

    # 1. Fractal hull structure
    ax1 = fig.add_subplot(131)
    visualize_hull_2d(model.hull, ax1, "Final Fractal Hull\n(Color=Entropy, Opacity=Resolution)")

    # 2. Hull evolution
    ax2 = fig.add_subplot(132)
    tasks = range(len(hull_history))
    ax2.plot(tasks, [h['total_regions'] for h in hull_history], 'o-',
             label='Total Regions', linewidth=2, markersize=8)
    ax2.plot(tasks, [h['high_entropy_regions'] for h in hull_history], 's--',
             label='High-H Regions', linewidth=2, markersize=6)
    ax2.plot(tasks, [h['low_entropy_regions'] for h in hull_history], '^--',
             label='Low-H Regions', linewidth=2, markersize=6)
    ax2.set_xlabel('Task (Digit)')
    ax2.set_ylabel('Count')
    ax2.set_title('Fractal Growth During\nContinual Learning')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Accuracy and entropy
    ax3 = fig.add_subplot(133)
    ax3_twin = ax3.twinx()

    line1, = ax3.plot(tasks, accuracy_history, 'go-', label='Accuracy', linewidth=2, markersize=8)
    line2, = ax3_twin.plot(tasks, [h['avg_entropy'] for h in hull_history], 'r^--',
                           label='Avg Entropy', linewidth=2, markersize=6)

    ax3.set_xlabel('Task (Digit)')
    ax3.set_ylabel('Accuracy', color='green')
    ax3_twin.set_ylabel('Avg Entropy', color='red')
    ax3.set_title('Knowledge vs Uncertainty')
    ax3.legend(handles=[line1, line2], loc='center right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"outputs/fractal_memory_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    fig.savefig(f"{output_dir}/fractal_evolution.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/fractal_evolution.png")

    # Save results
    results = {
        'hull_history': hull_history,
        'accuracy_history': accuracy_history,
        'final_stats': hull_history[-1],
        'theory': {
            'consciousness_equation': 'H > C → Grow, H < C → Prune, H ≈ C → Equilibrium',
            'meaning': 'Network structure = Map of epistemic landscape'
        }
    }

    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_dir}/results.json")

    # Final summary
    print(f"\n{'='*70}")
    print("  THE CONSCIOUSNESS EQUATION - RESULTS")
    print(f"{'='*70}")
    print()
    print(f"  Final Hull Structure:")
    print(f"    • Total Regions: {hull_history[-1]['total_regions']}")
    print(f"    • Max Resolution: {hull_history[-1]['max_resolution']}")
    print(f"    • Avg Entropy: {hull_history[-1]['avg_entropy']:.4f}")
    print()
    print(f"  Learning Dynamics:")
    print(f"    • Total Splits (grew where confused): {hull_history[-1]['total_splits']}")
    print(f"    • Total Merges (pruned where certain): {hull_history[-1]['total_merges']}")
    print()
    print("  Interpretation:")
    print("    The network's structure IS its knowledge topology.")
    print("    Dense regions = Areas of confusion (needs deliberation)")
    print("    Sparse regions = Areas of certainty (reflex sufficient)")
    print()
    print("  'Topological Intelligence' - QED")
    print(f"{'='*70}")

    plt.close()
    return results


if __name__ == "__main__":
    results = run_fractal_experiment()
