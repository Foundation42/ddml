"""
LOD Memory Experiment - Progressive Resolution Memory Retrieval

The Hypothesis:
Memory doesn't need full resolution for every query. Like human recall,
we can store memories at multiple resolutions and only "zoom in" when needed.

Phenomenology mapping:
- "Vague memory"           → Level 0 (4-dim) answered confidently
- "Starting to take shape" → Level 1 (16-dim) being consulted
- "Tip of my tongue"       → Level 2 (64-dim) partial match
- "R-r-Robert? Richard?"   → Level 3 (256-dim) competing hypotheses
- "Richard Feynman!"       → Full resolution retrieved

Key insight: Confidence IS resolution. Most queries don't need full detail.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json


# --- 1. Hierarchical VAE Architecture ---

class LODEncoder(nn.Module):
    """Encoder that produces nested latent codes at multiple resolutions"""
    def __init__(self, input_dim=784, hidden_dim=256, latent_dims=[4, 16, 64, 256]):
        super().__init__()
        self.latent_dims = latent_dims

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Progressive latent projections (each level adds detail)
        # Level 0: 4 dims (coarsest - just the gist)
        # Level 1: 16 dims (adds some detail)
        # Level 2: 64 dims (getting clearer)
        # Level 3: 256 dims (full resolution)
        self.mu_layers = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in latent_dims
        ])
        self.logvar_layers = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in latent_dims
        ])

    def forward(self, x):
        h = self.shared(x)

        mus = []
        logvars = []
        for mu_layer, logvar_layer in zip(self.mu_layers, self.logvar_layers):
            mus.append(mu_layer(h))
            logvars.append(logvar_layer(h))

        return mus, logvars


class LODDecoder(nn.Module):
    """Decoder that can reconstruct from any resolution level"""
    def __init__(self, output_dim=784, hidden_dim=256, latent_dims=[4, 16, 64, 256]):
        super().__init__()
        self.latent_dims = latent_dims

        # Separate decoder heads for each resolution
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid()
            ) for dim in latent_dims
        ])

    def forward(self, z, level):
        """Decode from a specific resolution level"""
        return self.decoders[level](z)


class HierarchicalVAE(nn.Module):
    """VAE with Level-of-Detail latent space"""
    def __init__(self, latent_dims=[4, 16, 64, 256]):
        super().__init__()
        self.latent_dims = latent_dims
        self.num_levels = len(latent_dims)

        self.encoder = LODEncoder(latent_dims=latent_dims)
        self.decoder = LODDecoder(latent_dims=latent_dims)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """Encode to all resolution levels"""
        x_flat = x.view(x.size(0), -1)
        mus, logvars = self.encoder(x_flat)
        zs = [self.reparameterize(mu, logvar) for mu, logvar in zip(mus, logvars)]
        return zs, mus, logvars

    def decode(self, z, level):
        """Decode from a specific resolution level"""
        return self.decoder(z, level)

    def forward(self, x, level=None):
        """Forward pass - if level is None, use highest resolution"""
        zs, mus, logvars = self.encode(x)

        if level is None:
            level = self.num_levels - 1

        recon = self.decode(zs[level], level)
        return recon, mus[level], logvars[level], zs


# --- 2. Resolution-Aware Classifier ---

class LODClassifier(nn.Module):
    """Classifier that works at multiple resolution levels"""
    def __init__(self, latent_dims=[4, 16, 64, 256], num_classes=10):
        super().__init__()
        self.latent_dims = latent_dims

        # Separate classifier head for each resolution
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            ) for dim in latent_dims
        ])

    def forward(self, z, level):
        """Classify from a specific resolution level"""
        return self.classifiers[level](z)

    def get_confidence(self, z, level):
        """Get prediction and confidence at a specific level"""
        logits = self.forward(z, level)
        probs = F.softmax(logits, dim=1)
        confidence, prediction = probs.max(dim=1)
        return prediction, confidence, probs


# --- 3. LOD Memory System ---

class LODMemory:
    """
    Level-of-Detail Memory System

    Stores memories at multiple resolutions.
    Retrieves starting from lowest resolution, escalating only when uncertain.
    """
    def __init__(self, vae, classifier, confidence_thresholds=[0.9, 0.85, 0.8, 0.0]):
        self.vae = vae
        self.classifier = classifier
        self.confidence_thresholds = confidence_thresholds  # When to stop at each level
        self.num_levels = len(vae.latent_dims)

        # Memory storage at each level
        self.memories = {level: {'z': [], 'labels': []} for level in range(self.num_levels)}

        # Statistics
        self.query_stats = {'level_stops': [0] * self.num_levels, 'total_queries': 0}

    def store(self, x, label):
        """Store a memory at all resolution levels"""
        self.vae.eval()
        with torch.no_grad():
            zs, _, _ = self.vae.encode(x)
            for level, z in enumerate(zs):
                self.memories[level]['z'].append(z.cpu())
                self.memories[level]['labels'].append(label)

    def query(self, x, return_trajectory=False):
        """
        Query memory with progressive resolution escalation

        Returns the answer from the lowest resolution that is confident enough.
        This mimics human recall: "vague" → "clearer" → "I remember!"
        """
        self.vae.eval()
        self.classifier.eval()

        trajectory = []

        with torch.no_grad():
            zs, _, _ = self.vae.encode(x)

            for level in range(self.num_levels):
                z = zs[level]
                pred, conf, probs = self.classifier.get_confidence(z, level)

                level_name = ['Vague', 'Forming', 'Almost', 'Clear'][level]
                trajectory.append({
                    'level': level,
                    'level_name': level_name,
                    'prediction': pred.item(),
                    'confidence': conf.item(),
                    'latent_dim': self.vae.latent_dims[level]
                })

                # Check if confident enough to stop
                if conf.item() >= self.confidence_thresholds[level]:
                    self.query_stats['level_stops'][level] += 1
                    self.query_stats['total_queries'] += 1

                    if return_trajectory:
                        return pred, conf, trajectory
                    return pred, conf, level

            # Reached max resolution
            self.query_stats['level_stops'][self.num_levels - 1] += 1
            self.query_stats['total_queries'] += 1

            if return_trajectory:
                return pred, conf, trajectory
            return pred, conf, self.num_levels - 1

    def get_storage_stats(self):
        """Calculate storage efficiency"""
        total_floats_full = 0
        total_floats_lod = 0

        for level in range(self.num_levels):
            n_memories = len(self.memories[level]['z'])
            dim = self.vae.latent_dims[level]

            # LOD: weight by how often this level is used
            if self.query_stats['total_queries'] > 0:
                usage_ratio = self.query_stats['level_stops'][level] / self.query_stats['total_queries']
            else:
                usage_ratio = 1.0 / self.num_levels

            total_floats_lod += n_memories * dim * usage_ratio

            if level == self.num_levels - 1:
                total_floats_full = n_memories * dim

        return {
            'full_resolution_floats': total_floats_full,
            'lod_effective_floats': total_floats_lod,
            'compression_ratio': total_floats_full / max(total_floats_lod, 1)
        }


# --- 4. Training Functions ---

def train_hierarchical_vae(vae, train_loader, epochs=20):
    """Train the hierarchical VAE on all resolution levels"""
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    vae.train()

    print("Training Hierarchical VAE...")
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in train_loader:
            x_flat = x.view(x.size(0), -1)
            optimizer.zero_grad()

            # Encode to all levels
            zs, mus, logvars = vae.encode(x)

            # Reconstruction loss at each level (weighted by resolution)
            loss = 0
            for level, (z, mu, logvar) in enumerate(zip(zs, mus, logvars)):
                recon = vae.decode(z, level)

                # Reconstruction loss
                recon_loss = F.mse_loss(recon, x_flat, reduction='sum')

                # KL divergence
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                # Weight: higher resolution levels matter more for reconstruction
                weight = (level + 1) / vae.num_levels
                loss += weight * (recon_loss + kl_loss)

            loss = loss / x.size(0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.2f}")

    return vae


def train_lod_classifier(classifier, vae, train_loader, epochs=10):
    """Train classifiers at each resolution level"""
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    vae.eval()
    classifier.train()

    print("Training LOD Classifiers...")
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()

            with torch.no_grad():
                zs, _, _ = vae.encode(x)

            # Train classifier at each level
            loss = 0
            for level, z in enumerate(zs):
                logits = classifier(z, level)
                loss += F.cross_entropy(logits, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.2f}")

    return classifier


# --- 5. Evaluation Functions ---

def evaluate_lod_system(memory, test_loader, verbose=True):
    """Evaluate the LOD memory system"""
    correct = 0
    total = 0
    level_usage = [0, 0, 0, 0]

    # Reset stats
    memory.query_stats = {'level_stops': [0] * memory.num_levels, 'total_queries': 0}

    for x, y in test_loader:
        for i in range(x.size(0)):
            xi = x[i:i+1]
            yi = y[i].item()

            pred, conf, level = memory.query(xi)

            if pred.item() == yi:
                correct += 1
            total += 1
            level_usage[level] += 1

    accuracy = correct / total

    if verbose:
        print(f"\nAccuracy: {accuracy:.1%}")
        print(f"Level usage:")
        for level, count in enumerate(level_usage):
            pct = count / total * 100
            level_name = ['Vague (4-dim)', 'Forming (16-dim)', 'Almost (64-dim)', 'Clear (256-dim)'][level]
            print(f"  Level {level} ({level_name}): {count} queries ({pct:.1f}%)")

    return accuracy, level_usage


def visualize_progressive_recall(memory, test_loader, n_examples=5):
    """Visualize the progressive recall process"""
    fig, axes = plt.subplots(n_examples, 6, figsize=(15, 3*n_examples))

    examples_shown = 0
    for x, y in test_loader:
        for i in range(x.size(0)):
            if examples_shown >= n_examples:
                break

            xi = x[i:i+1]
            yi = y[i].item()

            pred, conf, trajectory = memory.query(xi, return_trajectory=True)

            # Original image
            axes[examples_shown, 0].imshow(x[i].squeeze(), cmap='gray')
            axes[examples_shown, 0].set_title(f'Original\nLabel: {yi}')
            axes[examples_shown, 0].axis('off')

            # Reconstruction at each level
            memory.vae.eval()
            with torch.no_grad():
                zs, _, _ = memory.vae.encode(xi)

                for level in range(4):
                    recon = memory.vae.decode(zs[level], level)
                    recon_img = recon.view(28, 28).cpu().numpy()

                    # Get trajectory info (query all levels for visualization)
                    z = zs[level]
                    pred_l, conf_l, _ = memory.classifier.get_confidence(z, level)
                    level_name = ['Vague', 'Forming', 'Almost', 'Clear'][level]

                    stopped = ' STOP' if conf_l.item() >= memory.confidence_thresholds[level] else ''

                    axes[examples_shown, level+1].imshow(recon_img, cmap='gray')
                    axes[examples_shown, level+1].set_title(
                        f"{level_name} ({memory.vae.latent_dims[level]}d)\n"
                        f"Pred: {pred_l.item()} ({conf_l.item():.0%}){stopped}"
                    )
                    axes[examples_shown, level+1].axis('off')

            # Final answer
            final = trajectory[-1]
            color = 'green' if pred.item() == yi else 'red'
            axes[examples_shown, 5].text(0.5, 0.5,
                f"Final: {pred.item()}\nConf: {conf.item():.0%}\nLevel: {final['level_name']}",
                ha='center', va='center', fontsize=14,
                color=color, transform=axes[examples_shown, 5].transAxes)
            axes[examples_shown, 5].axis('off')

            examples_shown += 1

        if examples_shown >= n_examples:
            break

    plt.tight_layout()
    return fig


# --- 6. Main Experiment ---

def run_lod_experiment():
    """Run the full LOD Memory experiment"""

    print("=" * 60)
    print("       LOD MEMORY EXPERIMENT")
    print("       Progressive Resolution Recall")
    print("=" * 60)

    # Setup
    transform = transforms.ToTensor()
    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

    # Resolution levels: 4 → 16 → 64 → 256 dimensions
    latent_dims = [4, 16, 64, 256]

    print(f"\nResolution levels: {latent_dims}")
    print("Confidence thresholds: [0.90, 0.85, 0.80, 0.00]")
    print("-" * 60)

    # 1. Train Hierarchical VAE
    print("\n[1] Training Hierarchical VAE...")
    vae = HierarchicalVAE(latent_dims=latent_dims)
    vae = train_hierarchical_vae(vae, train_loader, epochs=20)

    # 2. Train LOD Classifier
    print("\n[2] Training LOD Classifiers...")
    classifier = LODClassifier(latent_dims=latent_dims)
    classifier = train_lod_classifier(classifier, vae, train_loader, epochs=10)

    # 3. Create LOD Memory System
    print("\n[3] Creating LOD Memory System...")
    memory = LODMemory(vae, classifier, confidence_thresholds=[0.90, 0.85, 0.80, 0.0])

    # 4. Evaluate: LOD vs Full Resolution
    print("\n[4] Evaluating LOD System...")
    print("\n--- LOD (Progressive Resolution) ---")
    acc_lod, usage_lod = evaluate_lod_system(memory, test_loader)

    # Compare with always using full resolution
    print("\n--- Full Resolution Only (Baseline) ---")
    memory_full = LODMemory(vae, classifier, confidence_thresholds=[0.0, 0.0, 0.0, 0.0])
    acc_full, usage_full = evaluate_lod_system(memory_full, test_loader)

    # 5. Calculate efficiency
    print("\n" + "=" * 60)
    print("                    RESULTS")
    print("=" * 60)

    # Compute savings
    avg_dim_lod = sum(d * u for d, u in zip(latent_dims, usage_lod)) / sum(usage_lod)
    avg_dim_full = latent_dims[-1]  # Always 256

    compute_saved = (1 - avg_dim_lod / avg_dim_full) * 100

    print(f"\n{'Metric':<30} {'LOD':<15} {'Full Res':<15}")
    print("-" * 60)
    print(f"{'Accuracy':<30} {acc_lod:.1%}           {acc_full:.1%}")
    print(f"{'Avg Latent Dim Used':<30} {avg_dim_lod:.1f}           {avg_dim_full:.1f}")
    print(f"{'Compute/Memory Saved':<30} {compute_saved:.1f}%          0%")

    # Level breakdown
    print(f"\n{'Resolution Breakdown (LOD):'}")
    phenomenology = ['Vague memory', 'Taking shape', 'Tip of tongue', 'Full recall']
    for level, (name, count, dim) in enumerate(zip(phenomenology, usage_lod, latent_dims)):
        pct = count / sum(usage_lod) * 100
        print(f"  {name:<20} ({dim:>3}d): {pct:>5.1f}% of queries")

    # 6. Visualize
    print("\n[5] Generating visualizations...")
    fig = visualize_progressive_recall(memory, test_loader, n_examples=6)

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"outputs/lod_memory_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    fig.savefig(f"{output_dir}/progressive_recall.png", dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_dir}/progressive_recall.png")

    # Save metrics
    results = {
        'latent_dims': latent_dims,
        'confidence_thresholds': [0.90, 0.85, 0.80, 0.0],
        'lod': {
            'accuracy': acc_lod,
            'level_usage': usage_lod,
            'avg_latent_dim': avg_dim_lod
        },
        'full_resolution': {
            'accuracy': acc_full,
            'level_usage': usage_full,
            'avg_latent_dim': avg_dim_full
        },
        'compute_saved_percent': compute_saved,
        'phenomenology_mapping': {
            'level_0': 'Vague memory (gist only)',
            'level_1': 'Taking shape (some detail)',
            'level_2': 'Tip of tongue (almost there)',
            'level_3': 'Full recall (complete detail)'
        }
    }

    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_dir}/results.json")

    # 7. The Verdict
    print("\n" + "=" * 60)
    print("                    VERDICT")
    print("=" * 60)

    if compute_saved > 30 and acc_lod >= acc_full * 0.98:
        print(f"\n>> SUCCESS: {compute_saved:.0f}% compute saved with <2% accuracy loss!")
        print("   Progressive resolution recall is viable.")
    elif compute_saved > 20:
        print(f"\n>> PROMISING: {compute_saved:.0f}% savings, worth further optimization.")
    else:
        print(f"\n>> NEEDS WORK: Only {compute_saved:.0f}% savings. Thresholds need tuning.")

    print("\n   Human-like phenomenology achieved:")
    print("   'Vague memory' → 'Taking shape' → 'Tip of tongue' → 'I remember!'")

    plt.close()
    return results


if __name__ == "__main__":
    results = run_lod_experiment()
