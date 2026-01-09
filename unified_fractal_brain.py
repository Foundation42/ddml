"""
Unified Fractal Tripartite Brain

All three partitions use the SAME growable fractal mechanism,
but with different parameters optimized for their roles:

┌────────────────────────────────────────────────────────────────┐
│              UNIFIED FRACTAL TRIPARTITE BRAIN                   │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  System 1 (Archive):     FractalHull - SPARSE                  │
│    • Few regions, high confidence                               │
│    • Fast retrieval, crystallized knowledge                     │
│    • Updated during DREAM (receives from S2)                    │
│                                                                 │
│  System 2 (Dialogue):    FractalHull - DENSE                   │
│    • Many regions, active debate                                │
│    • Net A vs Net B disagreement drives growth                  │
│    • Updated during WAKE (learns from reality)                  │
│                                                                 │
│  Imagination (VAE):      FractalHull - SMOOTH                  │
│    • Medium density, good interpolation                         │
│    • Learns data manifold topology                              │
│    • Generates dreams targeting high-H regions                  │
│                                                                 │
├────────────────────────────────────────────────────────────────┤
│                     KNOWLEDGE FLOW                              │
│                                                                 │
│  WAKE:   Reality → S2 debates → grows where confused           │
│  DREAM:  S2 stable regions → compress → transfer to S1         │
│          Imagination generates dreams for consolidation         │
│  RESULT: S2 prunes what's now reflexive, S1 crystallizes       │
│                                                                 │
└────────────────────────────────────────────────────────────────┘

The Consciousness Equation applies to ALL partitions:
  H > C → GROW (allocate capacity)
  H < C → PRUNE (remove redundancy)
  H ≈ C → EQUILIBRIUM (optimal)

But each partition has different H thresholds based on its role!
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
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import os
import json


# =============================================================================
# FRACTAL HULL - The Universal Growth Mechanism
# =============================================================================

@dataclass
class FractalRegion:
    """A region in the fractal latent space"""
    id: int
    center: torch.Tensor
    bounds: torch.Tensor  # [dim, 2] for [min, max]
    resolution: int = 0
    entropy: float = 0.5
    sample_count: int = 0
    children: Optional[List['FractalRegion']] = None

    # Region-specific learned parameters (the "local expert")
    weights: Optional[torch.Tensor] = None

    def is_leaf(self) -> bool:
        return self.children is None

    def contains(self, point: torch.Tensor) -> bool:
        p = point.cpu() if point.is_cuda else point
        return torch.all((p >= self.bounds[:, 0]) & (p <= self.bounds[:, 1])).item()


class UnifiedFractalHull:
    """
    Fractal Hull with role-specific parameters

    The SAME mechanism for all partitions, but different thresholds:
    - System 1: High split threshold (stays sparse)
    - System 2: Low split threshold (grows dense)
    - Imagination: Medium thresholds (balanced)
    """

    def __init__(self, latent_dim: int, role: str = 'dialogue',
                 bounds_range: float = 5.0):
        self.latent_dim = latent_dim
        self.role = role
        self.region_counter = 0

        # Role-specific parameters
        if role == 'archive':
            # System 1: SPARSE - only split when very confused
            self.split_threshold = 0.4   # High threshold = fewer splits
            self.merge_threshold = 0.15  # Higher merge = more consolidation
            self.max_resolution = 4      # Limited depth
            self.min_samples_split = 100
        elif role == 'dialogue':
            # System 2: DENSE - split eagerly to capture nuance
            self.split_threshold = 0.1   # Low threshold = many splits
            self.merge_threshold = 0.02  # Low merge = preserve detail
            self.max_resolution = 8      # Deep structure
            self.min_samples_split = 20
        else:  # imagination
            # VAE: SMOOTH - balanced for good interpolation
            self.split_threshold = 0.2
            self.merge_threshold = 0.08
            self.max_resolution = 6
            self.min_samples_split = 50

        # Initialize root region
        initial_bounds = torch.tensor([[-bounds_range, bounds_range]] * latent_dim)
        self.root = FractalRegion(
            id=self._next_id(),
            center=torch.zeros(latent_dim),
            bounds=initial_bounds,
            resolution=0,
            entropy=0.5
        )

        self.split_count = 0
        self.merge_count = 0

    def _next_id(self) -> int:
        self.region_counter += 1
        return self.region_counter

    def find_region(self, point: torch.Tensor) -> FractalRegion:
        """Find leaf region containing point"""
        node = self.root
        while not node.is_leaf():
            found = False
            for child in node.children:
                if child.contains(point):
                    node = child
                    found = True
                    break
            if not found:
                break
        return node

    def split_region(self, region: FractalRegion) -> bool:
        """Split region along dimension with largest range"""
        if region.resolution >= self.max_resolution:
            return False

        ranges = region.bounds[:, 1] - region.bounds[:, 0]
        split_dim = torch.argmax(ranges).item()
        split_point = region.center[split_dim].item()

        children = []
        for i in range(2):
            child_bounds = region.bounds.clone()
            if i == 0:
                child_bounds[split_dim, 1] = split_point
            else:
                child_bounds[split_dim, 0] = split_point

            child = FractalRegion(
                id=self._next_id(),
                center=child_bounds.mean(dim=1),
                bounds=child_bounds,
                resolution=region.resolution + 1,
                entropy=region.entropy
            )
            children.append(child)

        region.children = children
        self.split_count += 1
        return True

    def update(self, points: torch.Tensor, entropies: torch.Tensor):
        """Update hull based on samples and their entropies"""
        for point, entropy in zip(points, entropies):
            region = self.find_region(point)
            region.sample_count += 1

            # EMA of entropy
            alpha = 0.1
            region.entropy = alpha * entropy.item() + (1 - alpha) * region.entropy

            # Check for split
            if (region.is_leaf() and
                region.entropy > self.split_threshold and
                region.sample_count >= self.min_samples_split):
                self.split_region(region)

    def get_leaves(self) -> List[FractalRegion]:
        """Get all leaf regions"""
        leaves = []
        self._collect_leaves(self.root, leaves)
        return leaves

    def _collect_leaves(self, node: FractalRegion, leaves: List):
        if node.is_leaf():
            leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaves(child, leaves)

    def get_stats(self) -> dict:
        leaves = self.get_leaves()
        if not leaves:
            return {'total_regions': 0, 'avg_entropy': 0, 'max_resolution': 0}

        return {
            'total_regions': len(leaves),
            'max_resolution': max(r.resolution for r in leaves),
            'avg_entropy': np.mean([r.entropy for r in leaves]),
            'high_h_regions': sum(1 for r in leaves if r.entropy > self.split_threshold),
            'low_h_regions': sum(1 for r in leaves if r.entropy < self.merge_threshold),
            'total_splits': self.split_count,
            'role': self.role
        }


# =============================================================================
# THE THREE PARTITIONS
# =============================================================================

class FractalEncoder(nn.Module):
    """Shared encoder to latent space"""
    def __init__(self, input_dim: int = 784, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class FractalClassifier(nn.Module):
    """Classifier that operates in latent space"""
    def __init__(self, latent_dim: int = 16, output_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, z):
        return self.net(z)


class FractalVAE(nn.Module):
    """VAE for imagination - operates in latent space"""
    def __init__(self, latent_dim: int = 16, compressed_dim: int = 8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, compressed_dim)
        self.fc_logvar = nn.Linear(32, compressed_dim)

        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        self.compressed_dim = compressed_dim

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
        z_c = self.reparameterize(mu, logvar)
        return self.decode(z_c), mu, logvar

    def dream(self, n: int, device) -> torch.Tensor:
        z_c = torch.randn(n, self.compressed_dim).to(device)
        return self.decode(z_c)


class UnifiedFractalTripartiteBrain(nn.Module):
    """
    The Complete Unified Fractal Tripartite Brain

    Three partitions, same growth mechanism, different parameters:
    - System 1 (Archive): Sparse, fast, crystallized
    - System 2 (Dialogue): Dense, deliberative, growing
    - Imagination: Smooth, generative, dreaming
    """

    def __init__(self, input_dim: int = 784, latent_dim: int = 16,
                 output_dim: int = 10, device: str = 'cuda'):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim

        # Shared encoder (all partitions see the same latent space)
        self.encoder = FractalEncoder(input_dim, latent_dim)

        # === SYSTEM 1: Archive (SPARSE) ===
        self.archive = FractalClassifier(latent_dim, output_dim)
        self.archive_hull = UnifiedFractalHull(latent_dim, role='archive')
        self.archive_trust = 0.5
        self.archive_confidence = 0.5

        # === SYSTEM 2: Dialogue (DENSE) ===
        # Two independent networks for debate
        self.dialogue_a = FractalClassifier(latent_dim, output_dim)
        self.dialogue_b = FractalClassifier(latent_dim, output_dim)
        self.dialogue_hull = UnifiedFractalHull(latent_dim, role='dialogue')

        # === IMAGINATION: VAE (SMOOTH) ===
        self.imagination = FractalVAE(latent_dim, compressed_dim=8)
        self.imagination_hull = UnifiedFractalHull(latent_dim, role='imagination')

        # Memory archive for dreaming
        self.memory_buffer = []
        self.label_buffer = []
        self.max_memory = 5000

        # Hormones
        self.confidence = 0.5  # Based on dialogue agreement
        self.trust = 0.5       # Based on archive performance

        self.to(self.device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to shared latent space"""
        return self.encoder(x)

    def dialogue_forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """System 2: Two networks debate"""
        pred_a = self.dialogue_a(z)
        pred_b = self.dialogue_b(z)

        # Measure disagreement
        p_a = F.softmax(pred_a, dim=1)
        p_b = F.softmax(pred_b, dim=1)
        m = (p_a + p_b) / 2

        # JSD + prediction entropy
        jsd = 0.5 * (
            torch.sum(p_a * torch.log(p_a / (m + 1e-10) + 1e-10), dim=1) +
            torch.sum(p_b * torch.log(p_b / (m + 1e-10) + 1e-10), dim=1)
        )
        pred_entropy = -torch.sum(m * torch.log(m + 1e-10), dim=1) / np.log(10)
        entropy = 0.3 * jsd + 0.7 * pred_entropy

        return pred_a, pred_b, entropy

    def archive_forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """System 1: Fast, crystallized response"""
        pred = self.archive(z)
        confidence = F.softmax(pred, dim=1).max(dim=1)[0]
        return pred, confidence

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Complete forward pass with System 1/2 routing

        Returns dict with all outputs for analysis
        """
        z = self.encode(x)

        # System 1 (Archive)
        archive_pred, archive_conf = self.archive_forward(z)

        # System 2 (Dialogue)
        dial_a, dial_b, dial_entropy = self.dialogue_forward(z)

        # Consensus from dialogue
        consensus = (F.softmax(dial_a, dim=1) + F.softmax(dial_b, dim=1)) / 2

        # Routing decision based on trust and confidence
        effective_trust = self.archive_trust * archive_conf.mean().item()
        use_archive = effective_trust > 0.7

        # Final prediction
        if use_archive:
            final_pred = archive_pred
            system_used = 'archive'
        else:
            final_pred = dial_a  # Or could use consensus
            system_used = 'dialogue'

        return {
            'z': z,
            'archive_pred': archive_pred,
            'archive_conf': archive_conf,
            'dialogue_a': dial_a,
            'dialogue_b': dial_b,
            'dialogue_entropy': dial_entropy,
            'consensus': consensus,
            'final_pred': final_pred,
            'system_used': system_used,
            'trust': self.trust,
            'confidence': self.confidence
        }

    def wake_step(self, x: torch.Tensor, y: torch.Tensor,
                  optimizer: optim.Optimizer) -> Dict:
        """
        WAKE PHASE: Learn from reality
        - Dialogue grows where confused
        - Archive frozen
        - Imagination learns manifold
        """
        self.train()

        z = self.encode(x)

        # Store in memory
        self.memory_buffer.extend(z.detach().cpu().unbind(0))
        self.label_buffer.extend(y.cpu().unbind(0))
        if len(self.memory_buffer) > self.max_memory:
            self.memory_buffer = self.memory_buffer[-self.max_memory:]
            self.label_buffer = self.label_buffer[-self.max_memory:]

        # Dialogue forward
        pred_a, pred_b, entropy = self.dialogue_forward(z)

        # Update dialogue hull (grows where confused)
        self.dialogue_hull.update(z.detach(), entropy.detach())

        # Train dialogue networks
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred_a, y) + criterion(pred_b, y)

        # Train imagination on latent representations
        z_recon, mu, logvar = self.imagination(z.detach())
        vae_loss = F.mse_loss(z_recon, z.detach())
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        imagination_loss = vae_loss + 0.1 * kl_loss

        # Update imagination hull
        recon_error = torch.norm(z_recon - z.detach(), dim=1)
        self.imagination_hull.update(z.detach(), recon_error.detach())

        # Combined loss
        total_loss = loss + 0.5 * imagination_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update confidence hormone
        agreement = 1 - entropy.mean().item()
        self.confidence = 0.9 * self.confidence + 0.1 * agreement

        return {
            'loss': loss.item(),
            'entropy': entropy.mean().item(),
            'vae_loss': vae_loss.item(),
            'confidence': self.confidence
        }

    def dream_step(self, n_dreams: int = 64) -> Dict:
        """
        DREAM PHASE: Consolidate knowledge
        - Replay memories through dialogue
        - Transfer stable knowledge to archive
        - Imagination generates dreams
        """
        if len(self.memory_buffer) < 100:
            return {'status': 'insufficient_memories'}

        self.train()

        # Get archived memories
        z_memory = torch.stack(self.memory_buffer).to(self.device)
        y_memory = torch.stack(self.label_buffer).to(self.device)

        # Sample batch
        indices = torch.randperm(len(z_memory))[:n_dreams]
        z_batch = z_memory[indices]
        y_batch = y_memory[indices]

        # Add dream noise (slightly perturbed memories)
        z_dream = z_batch + torch.randn_like(z_batch) * 0.1

        # === CONSOLIDATE DIALOGUE ===
        pred_a, pred_b, entropy = self.dialogue_forward(z_dream)

        # Supervised replay on dialogue
        criterion = nn.CrossEntropyLoss()
        dialogue_loss = criterion(pred_a, y_batch) + criterion(pred_b, y_batch)

        # === TRANSFER TO ARCHIVE ===
        # Train archive on memories where dialogue agrees (low entropy)
        archive_pred, _ = self.archive_forward(z_dream)
        archive_loss = criterion(archive_pred, y_batch)

        # Update archive hull (should stay sparse)
        archive_entropy = -torch.sum(
            F.softmax(archive_pred, dim=1) * F.log_softmax(archive_pred, dim=1),
            dim=1
        ) / np.log(10)
        self.archive_hull.update(z_dream.detach(), archive_entropy.detach())

        # === UPDATE HULLS ===
        self.dialogue_hull.update(z_dream.detach(), entropy.detach())

        # Combined loss
        total_loss = dialogue_loss + archive_loss

        # Backward (assuming optimizer passed separately or using stored one)
        # For now, just compute - actual training done in experiment loop

        # Update trust based on archive accuracy
        with torch.no_grad():
            archive_correct = (archive_pred.argmax(dim=1) == y_batch).float().mean()
            self.trust = 0.9 * self.trust + 0.1 * archive_correct.item()

        return {
            'dialogue_loss': dialogue_loss.item(),
            'archive_loss': archive_loss.item(),
            'entropy': entropy.mean().item(),
            'trust': self.trust,
            'archive_accuracy': archive_correct.item()
        }

    def get_all_stats(self) -> Dict:
        """Get stats from all three hulls"""
        return {
            'archive': self.archive_hull.get_stats(),
            'dialogue': self.dialogue_hull.get_stats(),
            'imagination': self.imagination_hull.get_stats(),
            'hormones': {
                'trust': self.trust,
                'confidence': self.confidence
            }
        }


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_unified_experiment():
    """Run the Unified Fractal Tripartite Brain experiment"""

    print("=" * 70)
    print("     UNIFIED FRACTAL TRIPARTITE BRAIN")
    print("     Three Partitions, One Growth Mechanism")
    print("=" * 70)
    print()
    print("  System 1 (Archive):    SPARSE - crystallized, fast")
    print("  System 2 (Dialogue):   DENSE  - deliberative, growing")
    print("  Imagination (VAE):     SMOOTH - generative, dreaming")
    print()
    print("  Same fractal mechanism, different parameters!")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create brain
    brain = UnifiedFractalTripartiteBrain(latent_dim=16, device=device)
    optimizer = optim.Adam(brain.parameters(), lr=1e-3)

    # Load MNIST
    transform = transforms.ToTensor()
    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

    n_tasks = 5
    history = []

    for task in range(n_tasks):
        print(f"\n{'='*70}")
        print(f"  TASK {task}: Learning Digit {task}")
        print(f"{'='*70}")

        # Task data
        train_idx = [i for i, (_, l) in enumerate(mnist_train) if l == task]
        task_loader = DataLoader(Subset(mnist_train, train_idx[:1000]),
                                 batch_size=64, shuffle=True)

        # === WAKE PHASE ===
        print(f"\n  ☀️  WAKE: Learning from reality...")
        wake_stats = {'loss': [], 'entropy': []}

        for epoch in range(3):
            epoch_loss, epoch_entropy = 0, 0
            for x, y in task_loader:
                x, y = x.to(device), y.to(device)
                stats = brain.wake_step(x, y, optimizer)
                epoch_loss += stats['loss']
                epoch_entropy += stats['entropy']

            n_batches = len(task_loader)
            print(f"     Epoch {epoch+1}: Loss={epoch_loss/n_batches:.4f}, "
                  f"Entropy={epoch_entropy/n_batches:.4f}, "
                  f"Confidence={brain.confidence:.3f}")

        # Get stats after wake
        wake_hull_stats = brain.get_all_stats()
        print(f"\n     Hull Stats (Wake):")
        print(f"       Archive:     {wake_hull_stats['archive']['total_regions']} regions")
        print(f"       Dialogue:    {wake_hull_stats['dialogue']['total_regions']} regions")
        print(f"       Imagination: {wake_hull_stats['imagination']['total_regions']} regions")

        # === DREAM PHASE ===
        print(f"\n  🌙 DREAM: Consolidating knowledge...")

        dream_optimizer = optim.Adam(brain.parameters(), lr=5e-4)

        for dream_epoch in range(10):
            # Sample and dream
            z_mem = torch.stack(brain.memory_buffer).to(device)
            y_mem = torch.stack(brain.label_buffer).to(device)

            idx = torch.randperm(len(z_mem))[:128]
            z_batch = z_mem[idx] + torch.randn(128, brain.latent_dim).to(device) * 0.1
            y_batch = y_mem[idx]

            # Train all systems on dreams
            pred_a, pred_b, entropy = brain.dialogue_forward(z_batch)
            archive_pred, _ = brain.archive_forward(z_batch)

            criterion = nn.CrossEntropyLoss()
            loss = (criterion(pred_a, y_batch) + criterion(pred_b, y_batch) +
                    criterion(archive_pred, y_batch))

            dream_optimizer.zero_grad()
            loss.backward()
            dream_optimizer.step()

            # Update hulls
            brain.dialogue_hull.update(z_batch.detach(), entropy.detach())

            archive_entropy = -torch.sum(
                F.softmax(archive_pred, dim=1) * F.log_softmax(archive_pred, dim=1),
                dim=1
            ) / np.log(10)
            brain.archive_hull.update(z_batch.detach(), archive_entropy.detach())

        # Get stats after dream
        dream_hull_stats = brain.get_all_stats()
        print(f"\n     Hull Stats (Dream):")
        print(f"       Archive:     {dream_hull_stats['archive']['total_regions']} regions "
              f"(Δ {dream_hull_stats['archive']['total_regions'] - wake_hull_stats['archive']['total_regions']:+d})")
        print(f"       Dialogue:    {dream_hull_stats['dialogue']['total_regions']} regions "
              f"(Δ {dream_hull_stats['dialogue']['total_regions'] - wake_hull_stats['dialogue']['total_regions']:+d})")
        print(f"       Imagination: {dream_hull_stats['imagination']['total_regions']} regions "
              f"(Δ {dream_hull_stats['imagination']['total_regions'] - wake_hull_stats['imagination']['total_regions']:+d})")

        # Evaluate
        test_idx = [i for i, (_, l) in enumerate(mnist_test) if l <= task]
        test_loader = DataLoader(Subset(mnist_test, test_idx), batch_size=256)

        brain.eval()
        correct_archive, correct_dialogue, total = 0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = brain(x)
                correct_archive += (out['archive_pred'].argmax(1) == y).sum().item()
                correct_dialogue += (out['consensus'].argmax(1) == y).sum().item()
                total += y.size(0)

        acc_archive = correct_archive / total
        acc_dialogue = correct_dialogue / total

        print(f"\n     Accuracy:")
        print(f"       Archive (S1):  {acc_archive:.1%}")
        print(f"       Dialogue (S2): {acc_dialogue:.1%}")
        print(f"       Trust: {brain.trust:.3f}, Confidence: {brain.confidence:.3f}")

        history.append({
            'task': task,
            'wake_stats': wake_hull_stats,
            'dream_stats': dream_hull_stats,
            'acc_archive': acc_archive,
            'acc_dialogue': acc_dialogue,
            'trust': brain.trust,
            'confidence': brain.confidence
        })

    # === VISUALIZATION ===
    print(f"\n{'='*70}")
    print("  GENERATING VISUALIZATIONS")
    print(f"{'='*70}")

    fig = plt.figure(figsize=(18, 12))

    # 1. Region counts by partition
    ax1 = fig.add_subplot(2, 3, 1)
    tasks = range(len(history))
    archive_regions = [h['dream_stats']['archive']['total_regions'] for h in history]
    dialogue_regions = [h['dream_stats']['dialogue']['total_regions'] for h in history]
    imagination_regions = [h['dream_stats']['imagination']['total_regions'] for h in history]

    ax1.plot(tasks, archive_regions, 'o-', label='Archive (SPARSE)', linewidth=2, markersize=8, color='#3498db')
    ax1.plot(tasks, dialogue_regions, 's-', label='Dialogue (DENSE)', linewidth=2, markersize=8, color='#e74c3c')
    ax1.plot(tasks, imagination_regions, '^-', label='Imagination (SMOOTH)', linewidth=2, markersize=8, color='#2ecc71')
    ax1.set_xlabel('Task')
    ax1.set_ylabel('Regions')
    ax1.set_title('Fractal Growth by Partition\n(Same mechanism, different parameters)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy comparison
    ax2 = fig.add_subplot(2, 3, 2)
    acc_archive = [h['acc_archive'] for h in history]
    acc_dialogue = [h['acc_dialogue'] for h in history]

    ax2.plot(tasks, acc_archive, 'o-', label='Archive (S1)', linewidth=2, color='#3498db')
    ax2.plot(tasks, acc_dialogue, 's-', label='Dialogue (S2)', linewidth=2, color='#e74c3c')
    ax2.set_xlabel('Task')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('System Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # 3. Hormones
    ax3 = fig.add_subplot(2, 3, 3)
    trust_vals = [h['trust'] for h in history]
    conf_vals = [h['confidence'] for h in history]

    ax3.plot(tasks, trust_vals, 'o-', label='Trust', linewidth=2, color='#9b59b6')
    ax3.plot(tasks, conf_vals, 's-', label='Confidence', linewidth=2, color='#f39c12')
    ax3.set_xlabel('Task')
    ax3.set_ylabel('Hormone Level')
    ax3.set_title('Metacognitive Hormones')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)

    # 4. Average entropy by partition
    ax4 = fig.add_subplot(2, 3, 4)
    archive_h = [h['dream_stats']['archive']['avg_entropy'] for h in history]
    dialogue_h = [h['dream_stats']['dialogue']['avg_entropy'] for h in history]
    imagination_h = [h['dream_stats']['imagination']['avg_entropy'] for h in history]

    ax4.plot(tasks, archive_h, 'o-', label='Archive', linewidth=2, color='#3498db')
    ax4.plot(tasks, dialogue_h, 's-', label='Dialogue', linewidth=2, color='#e74c3c')
    ax4.plot(tasks, imagination_h, '^-', label='Imagination', linewidth=2, color='#2ecc71')
    ax4.set_xlabel('Task')
    ax4.set_ylabel('Avg Entropy')
    ax4.set_title('Entropy by Partition')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Architecture diagram
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')

    diagram = """
    ┌─────────────────────────────────────────┐
    │     UNIFIED FRACTAL TRIPARTITE BRAIN    │
    ├─────────────────────────────────────────┤
    │                                         │
    │   ┌─────────┐       ┌─────────┐        │
    │   │ ARCHIVE │◄──────│DIALOGUE │        │
    │   │ (S1)    │ dream │ (S2)    │        │
    │   │ SPARSE  │       │ DENSE   │        │
    │   └────┬────┘       └────┬────┘        │
    │        │                 │              │
    │        │    ┌───────┐    │              │
    │        └───►│ IMAG  │◄───┘              │
    │             │ SMOOTH│                   │
    │             └───────┘                   │
    │                                         │
    │   Wake:  S2 grows where confused        │
    │   Dream: S2 → S1 knowledge transfer     │
    │                                         │
    └─────────────────────────────────────────┘
    """
    ax5.text(0.1, 0.5, diagram, fontsize=9, family='monospace',
             verticalalignment='center')

    # 6. Summary stats
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    final = history[-1]
    summary = f"""
    FINAL STATISTICS
    ════════════════════════════════════

    Archive (System 1 - SPARSE):
      Regions: {final['dream_stats']['archive']['total_regions']}
      Avg Entropy: {final['dream_stats']['archive']['avg_entropy']:.4f}
      Accuracy: {final['acc_archive']:.1%}

    Dialogue (System 2 - DENSE):
      Regions: {final['dream_stats']['dialogue']['total_regions']}
      Avg Entropy: {final['dream_stats']['dialogue']['avg_entropy']:.4f}
      Accuracy: {final['acc_dialogue']:.1%}

    Imagination (VAE - SMOOTH):
      Regions: {final['dream_stats']['imagination']['total_regions']}
      Avg Entropy: {final['dream_stats']['imagination']['avg_entropy']:.4f}

    Hormones:
      Trust: {final['trust']:.3f}
      Confidence: {final['confidence']:.3f}

    ════════════════════════════════════
    "One mechanism, three expressions"
    """
    ax6.text(0.1, 0.5, summary, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"outputs/unified_fractal_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    fig.savefig(f"{output_dir}/unified_brain.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/unified_brain.png")

    # Save results
    results = {
        'history': [{k: v for k, v in h.items()} for h in history],
        'final_stats': brain.get_all_stats(),
        'architecture': {
            'archive_params': {'split_threshold': 0.4, 'role': 'sparse'},
            'dialogue_params': {'split_threshold': 0.1, 'role': 'dense'},
            'imagination_params': {'split_threshold': 0.2, 'role': 'smooth'}
        }
    }

    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {output_dir}/results.json")

    # Final summary
    print(f"\n{'='*70}")
    print("  UNIFIED FRACTAL TRIPARTITE BRAIN - COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Archive (SPARSE):  {final['dream_stats']['archive']['total_regions']} regions - crystallized")
    print(f"  Dialogue (DENSE):  {final['dream_stats']['dialogue']['total_regions']} regions - deliberative")
    print(f"  Imagination (SMOOTH): {final['dream_stats']['imagination']['total_regions']} regions - generative")
    print(f"\n  'One mechanism, three expressions.'")
    print(f"{'='*70}\n")

    plt.close()
    return results


if __name__ == "__main__":
    results = run_unified_experiment()
