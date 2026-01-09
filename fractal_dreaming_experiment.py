"""
Fractal Dreaming Experiment - The Complete Consciousness Equation

The Missing Piece: Without dreams, fractal structure grows indefinitely.
WITH dreams, structure consolidates - knowledge crystallizes.

Wake Phase:  Learn from reality  → H > C → SPLIT (structure grows)
Dream Phase: Consolidate memory  → H < C → MERGE (structure prunes)

This is consciousness as dynamic channel allocation!
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
from datetime import datetime
import os
import json

# Import from our fractal memory experiment
from fractal_memory_experiment import (
    FractalDialogueSystem,
    FractalManifoldHull,
    ManifoldRegion,
    visualize_hull_2d
)


class LatentImaginationCore(nn.Module):
    """VAE that dreams in latent space - generates consolidated memories"""

    def __init__(self, latent_dim: int, compressed_dim: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.compressed_dim = compressed_dim

        # Encoder: latent → compressed
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(16, compressed_dim)
        self.fc_logvar = nn.Linear(16, compressed_dim)

        # Decoder: compressed → latent
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

    def encode(self, z):
        h = self.encoder(z)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z_compressed):
        return self.decoder(z_compressed)

    def forward(self, z):
        mu, logvar = self.encode(z)
        z_compressed = self.reparameterize(mu, logvar)
        z_reconstructed = self.decode(z_compressed)
        return z_reconstructed, mu, logvar

    def dream(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Generate dream samples by sampling the compressed space"""
        z_compressed = torch.randn(n_samples, self.compressed_dim).to(device)
        return self.decode(z_compressed)


class DreamingFractalBrain:
    """
    Complete Wake/Sleep Cycle with Fractal Memory

    The brain that grows during waking and consolidates during sleep.
    """

    def __init__(self, input_dim: int = 784, latent_dim: int = 8,
                 output_dim: int = 10, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim

        # The Dialogue System with Fractal Hull
        self.model = FractalDialogueSystem(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=output_dim
        ).to(self.device)

        # The Imagination Core for dreaming
        self.imagination = LatentImaginationCore(latent_dim).to(self.device)

        # Memory archive (experiences from waking)
        self.memory_archive = []
        self.label_archive = []
        self.max_archive = 5000

        # Statistics
        self.wake_history = []
        self.dream_history = []

    def wake_phase(self, data_loader, epochs: int = 3, lr: float = 1e-3):
        """
        WAKE: Learn from reality
        - Entropy rises as we encounter novel data
        - Hull structure SPLITS where confused
        """
        print(f"\n{'☀️ '*15}")
        print("  WAKE PHASE: Learning from Reality")
        print(f"{'☀️ '*15}\n")

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            total_entropy = 0
            batch_count = 0

            for batch_idx, (x, y) in enumerate(data_loader):
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                pred_a, pred_b, z = self.model(x)

                # Store in memory archive
                self.memory_archive.extend(z.detach().cpu().unbind(0))
                self.label_archive.extend(y.cpu().unbind(0))

                # Trim archive if too large
                if len(self.memory_archive) > self.max_archive:
                    self.memory_archive = self.memory_archive[-self.max_archive:]
                    self.label_archive = self.label_archive[-self.max_archive:]

                # Compute entropy (combined disagreement + uncertainty)
                entropy = self.model.compute_disagreement(pred_a, pred_b)
                total_entropy += entropy.mean().item()

                # Update fractal hull
                if batch_idx % 5 == 0:
                    self.model.update_hull(z, pred_a, pred_b)

                # Train both networks
                loss = criterion(pred_a, y) + criterion(pred_b, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            avg_loss = total_loss / batch_count
            avg_entropy = total_entropy / batch_count

            print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Entropy={avg_entropy:.4f}")

        # Record statistics
        stats = self.model.hull.get_statistics()
        stats['avg_entropy_measured'] = avg_entropy
        stats['phase'] = 'wake'
        self.wake_history.append(stats)

        print(f"\n  📈 Wake Results:")
        print(f"     Regions: {stats['total_regions']}")
        print(f"     Splits: {stats['total_splits']}")
        print(f"     Avg Entropy: {stats['avg_entropy']:.4f}")

        return stats

    def dream_phase(self, n_dreams: int = 500, epochs: int = 5):
        """
        DREAM: Consolidate knowledge
        - Train imagination on archived experiences
        - Generate synthetic experiences (dreams)
        - Force networks toward consensus on dreams
        - Entropy drops, hull MERGES where certain
        """
        print(f"\n{'🌙 '*15}")
        print("  DREAM PHASE: Consolidating Knowledge")
        print(f"{'🌙 '*15}\n")

        if len(self.memory_archive) < 100:
            print("  ⚠️ Not enough memories to dream")
            return None

        # Step 1: Train imagination on archived experiences
        print("  💭 Training Imagination Core...")

        archive_tensor = torch.stack(self.memory_archive).to(self.device)
        imagination_opt = optim.Adam(self.imagination.parameters(), lr=1e-3)

        for epoch in range(epochs):
            # Random batch from archive
            indices = torch.randperm(len(archive_tensor))[:256]
            z_batch = archive_tensor[indices]

            # VAE forward
            z_recon, mu, logvar = self.imagination(z_batch)

            # VAE loss
            recon_loss = F.mse_loss(z_recon, z_batch)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            vae_loss = recon_loss + 0.1 * kl_loss

            imagination_opt.zero_grad()
            vae_loss.backward()
            imagination_opt.step()

        print(f"  ✓ Imagination trained (recon={recon_loss:.4f})")

        # Step 2: Generate dreams and consolidate
        print("  🌀 Dreaming and consolidating...")

        # Use lower learning rate for consolidation
        model_opt = optim.Adam(self.model.parameters(), lr=5e-4)

        total_entropy = 0
        n_batches = n_dreams // 64

        self.model.train()

        # Get labels for supervised replay
        label_tensor = torch.stack(self.label_archive).to(self.device)
        criterion = nn.CrossEntropyLoss()

        for batch_idx in range(n_batches):
            # SUPERVISED MEMORY REPLAY - like real biological memory consolidation!
            # Sample archived experiences WITH their labels
            indices = torch.randperm(len(archive_tensor))[:64]
            z_memory = archive_tensor[indices]
            y_memory = label_tensor[indices]

            # Slight perturbation to create "dream-like" quality
            noise = torch.randn_like(z_memory) * 0.1
            z_dream = z_memory + noise

            # Both networks make predictions on dreams
            pred_a = self.model.net_a(z_dream)
            pred_b = self.model.net_b(z_dream)

            # Measure entropy on dreams
            entropy = self.model.compute_disagreement(pred_a, pred_b)
            total_entropy += entropy.mean().item()

            # Update hull with dream samples
            if batch_idx % 3 == 0:
                self.model.update_hull(z_dream, pred_a, pred_b)

            # SUPERVISED CONSOLIDATION: Both networks learn the correct answers
            # This is what actually reduces entropy - agreement through correctness!
            supervised_loss = criterion(pred_a, y_memory) + criterion(pred_b, y_memory)

            # Also add consensus loss for extra agreement
            p_a = F.softmax(pred_a, dim=1)
            p_b = F.softmax(pred_b, dim=1)
            consensus_target = (p_a + p_b) / 2
            consensus_loss = (
                F.kl_div(F.log_softmax(pred_a, dim=1), consensus_target.detach(), reduction='batchmean') +
                F.kl_div(F.log_softmax(pred_b, dim=1), consensus_target.detach(), reduction='batchmean')
            )

            # Combined loss: supervised + consensus
            total_loss = supervised_loss + 0.5 * consensus_loss

            model_opt.zero_grad()
            total_loss.backward()
            model_opt.step()

        avg_entropy = total_entropy / n_batches

        # Record statistics
        stats = self.model.hull.get_statistics()
        stats['avg_entropy_measured'] = avg_entropy
        stats['phase'] = 'dream'
        self.dream_history.append(stats)

        print(f"\n  📉 Dream Results:")
        print(f"     Regions: {stats['total_regions']}")
        print(f"     Merges: {stats['total_merges']}")
        print(f"     Avg Entropy: {stats['avg_entropy']:.4f}")

        return stats

    def evaluate(self, test_loader) -> float:
        """Evaluate consensus accuracy"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred_a, pred_b, _ = self.model(x)

                # Consensus prediction
                consensus = (F.softmax(pred_a, dim=1) + F.softmax(pred_b, dim=1)) / 2
                pred = consensus.argmax(dim=1)

                correct += (pred == y).sum().item()
                total += y.size(0)

        return correct / total


def run_fractal_dreaming_experiment():
    """Run the complete wake/dream continual learning experiment"""

    print("=" * 70)
    print("     FRACTAL DREAMING EXPERIMENT")
    print("     The Complete Consciousness Equation")
    print("=" * 70)
    print()
    print("  Wake:  H > C → SPLIT (grow where confused)")
    print("  Dream: H < C → MERGE (prune where certain)")
    print("  Result: Dynamic equilibrium, true learning")
    print()
    print("=" * 70)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    brain = DreamingFractalBrain(latent_dim=8, device=device)

    # Load MNIST
    transform = transforms.ToTensor()
    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

    n_tasks = 5  # First 5 digits

    all_stats = []
    phase_labels = []
    accuracy_history = []

    for task in range(n_tasks):
        print(f"\n{'='*70}")
        print(f"  TASK {task}: Learning Digit {task}")
        print(f"{'='*70}")

        # Task-specific data
        train_indices = [i for i, (_, label) in enumerate(mnist_train) if label == task]
        task_loader = DataLoader(Subset(mnist_train, train_indices[:1000]),
                                 batch_size=64, shuffle=True)

        # Test on all digits seen so far
        test_indices = [i for i, (_, label) in enumerate(mnist_test) if label <= task]
        test_loader = DataLoader(Subset(mnist_test, test_indices), batch_size=256)

        # === WAKE ===
        wake_stats = brain.wake_phase(task_loader, epochs=3)
        all_stats.append(wake_stats)
        phase_labels.append(f'T{task} Wake')

        # Evaluate after wake
        acc_wake = brain.evaluate(test_loader)
        print(f"  Accuracy after wake: {acc_wake:.1%}")

        # === DREAM ===
        dream_stats = brain.dream_phase(n_dreams=1000, epochs=10)
        if dream_stats:
            all_stats.append(dream_stats)
            phase_labels.append(f'T{task} Dream')

        # Evaluate after dream
        acc_dream = brain.evaluate(test_loader)
        accuracy_history.append({'task': task, 'wake': acc_wake, 'dream': acc_dream})
        print(f"  Accuracy after dream: {acc_dream:.1%}")

        # Report consolidation
        if dream_stats:
            region_delta = wake_stats['total_regions'] - dream_stats['total_regions']
            entropy_delta = wake_stats['avg_entropy'] - dream_stats['avg_entropy']

            print(f"\n  {'─'*50}")
            if region_delta > 0:
                print(f"  ✨ CONSOLIDATED: {region_delta} regions merged!")
            elif region_delta < 0:
                print(f"  🌱 EXPANDED: {-region_delta} new regions")
            else:
                print(f"  ≈ STABLE: No structural change")

            print(f"  Entropy: {wake_stats['avg_entropy']:.4f} → {dream_stats['avg_entropy']:.4f} "
                  f"({'↓' if entropy_delta > 0 else '↑'} {abs(entropy_delta):.4f})")
            print(f"  {'─'*50}")

    # === VISUALIZATION ===
    print(f"\n{'='*70}")
    print("  GENERATING VISUALIZATIONS")
    print(f"{'='*70}")

    fig = plt.figure(figsize=(18, 10))

    # 1. Fractal structure
    ax1 = fig.add_subplot(2, 3, 1)
    visualize_hull_2d(brain.model.hull, ax1, "Final Fractal Structure")

    # 2. Region count evolution
    ax2 = fig.add_subplot(2, 3, 2)
    x = range(len(all_stats))
    regions = [s['total_regions'] for s in all_stats]
    colors = ['#FF6B6B' if 'Wake' in l else '#4ECDC4' for l in phase_labels]
    ax2.bar(x, regions, color=colors, edgecolor='black', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(phase_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Regions')
    ax2.set_title('Structure Evolution\n(Red=Wake, Teal=Dream)')
    ax2.grid(axis='y', alpha=0.3)

    # 3. Entropy evolution
    ax3 = fig.add_subplot(2, 3, 3)
    entropies = [s['avg_entropy'] for s in all_stats]
    ax3.plot(x, entropies, 'o-', color='#9B59B6', linewidth=2, markersize=8)
    for i, (xi, e, l) in enumerate(zip(x, entropies, phase_labels)):
        color = '#FF6B6B' if 'Wake' in l else '#4ECDC4'
        ax3.scatter([xi], [e], c=color, s=100, zorder=5, edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels(phase_labels, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Avg Entropy')
    ax3.set_title('Entropy Dynamics\n(Should oscillate: ↑wake ↓dream)')
    ax3.axhline(y=brain.model.hull.split_threshold, color='red', linestyle='--',
                alpha=0.5, label=f'Split threshold ({brain.model.hull.split_threshold})')
    ax3.axhline(y=brain.model.hull.merge_threshold, color='green', linestyle='--',
                alpha=0.5, label=f'Merge threshold ({brain.model.hull.merge_threshold})')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Accuracy evolution
    ax4 = fig.add_subplot(2, 3, 4)
    tasks = [a['task'] for a in accuracy_history]
    wake_acc = [a['wake'] for a in accuracy_history]
    dream_acc = [a['dream'] for a in accuracy_history]
    ax4.plot(tasks, wake_acc, 'o-', color='#FF6B6B', label='After Wake', linewidth=2, markersize=8)
    ax4.plot(tasks, dream_acc, 's-', color='#4ECDC4', label='After Dream', linewidth=2, markersize=8)
    ax4.set_xlabel('Task')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Learning Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.05)

    # 5. High/Low entropy regions
    ax5 = fig.add_subplot(2, 3, 5)
    high_h = [s['high_entropy_regions'] for s in all_stats]
    low_h = [s['low_entropy_regions'] for s in all_stats]
    ax5.plot(x, high_h, 'o-', color='#E74C3C', label='High-H (Confused)', linewidth=2)
    ax5.plot(x, low_h, 's-', color='#2ECC71', label='Low-H (Certain)', linewidth=2)
    ax5.set_xticks(x)
    ax5.set_xticklabels(phase_labels, rotation=45, ha='right', fontsize=8)
    ax5.set_ylabel('Region Count')
    ax5.set_title('Knowledge Landscape')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Theory summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    # Calculate summary stats
    wake_entropies = [s['avg_entropy'] for s, l in zip(all_stats, phase_labels) if 'Wake' in l]
    dream_entropies = [s['avg_entropy'] for s, l in zip(all_stats, phase_labels) if 'Dream' in l]

    avg_wake_h = np.mean(wake_entropies) if wake_entropies else 0
    avg_dream_h = np.mean(dream_entropies) if dream_entropies else 0

    summary = f"""
    THE CONSCIOUSNESS EQUATION
    ══════════════════════════════════════

    WAKE:  Reality → Disagreement ↑ → SPLIT
    DREAM: Synthesis → Consensus ↑ → MERGE

    ══════════════════════════════════════

    Results:
    • Avg Wake Entropy:  {avg_wake_h:.4f}
    • Avg Dream Entropy: {avg_dream_h:.4f}
    • Entropy Reduction: {(1 - avg_dream_h/(avg_wake_h+1e-6))*100:.1f}%

    • Final Regions: {all_stats[-1]['total_regions']}
    • Total Splits:  {all_stats[-1]['total_splits']}
    • Total Merges:  {all_stats[-1]['total_merges']}

    ══════════════════════════════════════

    "The structure breathes:
     growing during wake,
     consolidating during sleep."
    """

    ax6.text(0.1, 0.5, summary, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"outputs/fractal_dreaming_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    fig.savefig(f"{output_dir}/evolution.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/evolution.png")

    # Save results
    results = {
        'all_stats': all_stats,
        'phase_labels': phase_labels,
        'accuracy_history': accuracy_history,
        'summary': {
            'avg_wake_entropy': avg_wake_h,
            'avg_dream_entropy': avg_dream_h,
            'entropy_reduction_pct': (1 - avg_dream_h/(avg_wake_h+1e-6))*100,
            'final_regions': all_stats[-1]['total_regions'],
            'total_splits': all_stats[-1]['total_splits'],
            'total_merges': all_stats[-1]['total_merges']
        }
    }

    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_dir}/results.json")

    # Final summary
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Wake Entropy (avg):  {avg_wake_h:.4f}")
    print(f"  Dream Entropy (avg): {avg_dream_h:.4f}")
    print(f"  Reduction: {(1 - avg_dream_h/(avg_wake_h+1e-6))*100:.1f}%")
    print(f"\n  Final Structure: {all_stats[-1]['total_regions']} regions")
    print(f"  Splits: {all_stats[-1]['total_splits']}, Merges: {all_stats[-1]['total_merges']}")
    print(f"\n{'='*70}")
    print("  ✨ The structure breathes: growing during wake,")
    print("     consolidating during sleep.")
    print(f"{'='*70}\n")

    plt.close()
    return results


if __name__ == "__main__":
    results = run_fractal_dreaming_experiment()
