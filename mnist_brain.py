"""
MNIST-Scale Tripartite Brain with Dreaming

Scaling up from XOR to real image classification with:
    1. VisualCortex networks (784 -> 10 classifier)
    2. Dreaming mechanism (replay old memories during sleep)
    3. Core Set retention (permanent memory of key examples)

The "Dreaming" insight from Gemini:
    Without replay, the Archive will catastrophically forget Task A
    when learning Task B. By mixing old memories with new during
    consolidation, we preserve knowledge while accommodating novelty.

This is analogous to how REM sleep replays memories to consolidate
them into long-term storage without overwriting existing knowledge.

Author: Christian Beaumont & Claude
"""

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
import random
import numpy as np


@dataclass
class MNISTBrainConfig:
    """Configuration for MNIST-scale brain."""
    # Network architecture
    input_dim: int = 784          # 28x28 flattened
    hidden_dim: int = 256         # Larger hidden layer for images
    output_dim: int = 10          # 10 digit classes

    # System 2 (Dialogue) settings
    dialogue_lr: float = 0.001
    dialogue_lr_slow: float = 0.0005  # Net B learns slower

    # System 1 (Archive) settings
    archive_lr: float = 0.0005    # Very slow learning for stability
    archive_hidden_dim: int = 256

    # Trust and confidence
    confidence_alpha: float = 0.95
    archive_trust_alpha: float = 0.90
    effective_threshold: float = 0.50

    # Memory settings
    short_term_buffer_size: int = 2000
    core_set_size: int = 500      # Permanent memory per task
    min_confidence_to_store: float = 0.0  # Store ALL training examples (was 0.70)

    # Sleep/Dreaming settings
    consolidation_epochs: int = 10
    replay_ratio: float = 0.5     # Mix 50% old memories with new


class VisualCortex(nn.Module):
    """
    A deeper network for processing 28x28 images.

    Architecture: 784 -> 256 -> 128 -> 10
    Includes dropout for regularization.
    """
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        dropout: float = 0.2
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten if needed (for raw image tensors)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class DreamingMemory:
    """
    Memory system with both short-term buffer and long-term core set.

    Short-term buffer: Today's experiences (cleared after sleep)
    Core set: Permanent examples from each task (never cleared)

    During "dreaming", we replay BOTH:
        - New memories from the buffer
        - Old memories from the core set

    This prevents catastrophic forgetting by ensuring the Archive
    sees examples from ALL tasks during consolidation.
    """

    def __init__(self, config: MNISTBrainConfig):
        self.config = config

        # Short-term: today's experiences
        self.short_term_buffer: List[Tuple[torch.Tensor, torch.Tensor, float]] = []

        # Long-term: permanent examples organized by task/class
        self.core_set: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

        # Track which classes we've seen
        self.seen_classes: set = set()

    def add_experience(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        confidence: float
    ):
        """Add a new experience to short-term buffer."""
        if confidence >= self.config.min_confidence_to_store:
            if len(self.short_term_buffer) < self.config.short_term_buffer_size:
                self.short_term_buffer.append((
                    x.detach().cpu(),
                    y.detach().cpu(),
                    confidence
                ))

            # Track seen classes
            if y.dim() == 0:
                self.seen_classes.add(y.item())
            else:
                for label in y:
                    self.seen_classes.add(label.item())

    def _update_core_set(self, x_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        Add high-quality examples to the permanent core set.

        We keep a balanced set of examples per class to ensure
        we don't forget any particular class.
        """
        max_per_class = self.config.core_set_size // max(len(self.seen_classes), 1)

        for x, y in zip(x_batch, y_batch):
            label = y.item() if y.dim() == 0 else y[0].item()

            if label not in self.core_set:
                self.core_set[label] = []

            if len(self.core_set[label]) < max_per_class:
                self.core_set[label].append((x.unsqueeze(0), y.unsqueeze(0)))

    def get_consolidation_data(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get data for sleep consolidation, mixing old and new memories.

        This is the "Dreaming" mechanism:
            - Take new memories from short-term buffer
            - Mix with old memories from core set
            - Return combined dataset for Archive training

        The mixing prevents the Archive from forgetting old tasks
        while learning new ones.
        """
        if len(self.short_term_buffer) == 0:
            return None

        # Collect new memories
        new_xs, new_ys, _ = zip(*self.short_term_buffer)
        new_x = torch.cat(new_xs, dim=0)
        new_y = torch.cat(new_ys, dim=0)

        # Update core set with some of today's high-quality examples
        self._update_core_set(new_x, new_y)

        # Collect old memories (replay/dreaming)
        old_xs, old_ys = [], []
        for class_examples in self.core_set.values():
            for x, y in class_examples:
                old_xs.append(x)
                old_ys.append(y)

        if old_xs:
            old_x = torch.cat(old_xs, dim=0)
            old_y = torch.cat(old_ys, dim=0)

            # Mix old and new based on replay_ratio
            n_old = int(len(new_x) * self.config.replay_ratio)
            if n_old > len(old_x):
                n_old = len(old_x)

            if n_old > 0:
                # Random sample from old memories
                indices = torch.randperm(len(old_x))[:n_old]
                replay_x = old_x[indices]
                replay_y = old_y[indices]

                # Combine new + replayed old
                combined_x = torch.cat([new_x, replay_x], dim=0)
                combined_y = torch.cat([new_y, replay_y], dim=0)
            else:
                combined_x, combined_y = new_x, new_y
        else:
            combined_x, combined_y = new_x, new_y

        return combined_x, combined_y

    def clear_short_term(self):
        """Clear the short-term buffer after consolidation."""
        self.short_term_buffer = []

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        core_total = sum(len(v) for v in self.core_set.values())
        return {
            "short_term_size": len(self.short_term_buffer),
            "core_set_size": core_total,
            "seen_classes": len(self.seen_classes),
            "classes": list(self.seen_classes)
        }


class MNISTTripartiteBrain:
    """
    MNIST-scale Tripartite Brain with Dreaming.

    Key differences from XOR version:
        1. VisualCortex networks (larger, deeper)
        2. DreamingMemory (core set + replay)
        3. Class-based confidence tracking
        4. Multi-class classification
    """

    def __init__(self, config: Optional[MNISTBrainConfig] = None):
        self.config = config or MNISTBrainConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # System 2: Dialogue (Conscious Processing)
        self.net_A = VisualCortex(
            self.config.input_dim,
            self.config.hidden_dim,
            self.config.output_dim
        ).to(self.device)

        self.net_B = VisualCortex(
            self.config.input_dim,
            self.config.hidden_dim,
            self.config.output_dim
        ).to(self.device)

        self.opt_A = optim.Adam(self.net_A.parameters(), lr=self.config.dialogue_lr)
        self.opt_B = optim.Adam(self.net_B.parameters(), lr=self.config.dialogue_lr_slow)

        # System 1: Archive (Reflex/Subconscious)
        self.archive = VisualCortex(
            self.config.input_dim,
            self.config.archive_hidden_dim,
            self.config.output_dim
        ).to(self.device)

        self.opt_archive = optim.Adam(
            self.archive.parameters(),
            lr=self.config.archive_lr
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Memory system with dreaming
        self.memory = DreamingMemory(self.config)

        # Metacognition state
        self.confidence = 0.5
        self.archive_trust = 0.3  # Start with some trust

        # Metrics
        self.metrics = {
            "system_used": [],
            "accuracy": [],
            "archive_trust": [],
            "confidence": []
        }

        # Counters
        self.total_steps = 0
        self.archive_calls = 0
        self.dialogue_calls = 0

    def _get_archive_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Get Archive prediction and confidence."""
        with torch.no_grad():
            logits = self.archive(x)
            probs = torch.softmax(logits, dim=1)
            # Confidence = max probability (how sure about top prediction)
            confidence = probs.max(dim=1).values.mean().item()
        return logits, confidence

    def _compute_accuracy(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        """Compute classification accuracy."""
        predictions = logits.argmax(dim=1)
        return (predictions == y).float().mean().item()

    def wake_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        force_dialogue: bool = False
    ) -> Dict:
        """
        Process one batch during wake phase.

        Uses the same trust-gated routing as XOR version,
        but adapted for multi-class classification.
        """
        x = x.to(self.device)
        y = y.to(self.device)

        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Check Archive confidence
        archive_logits, archive_confidence = self._get_archive_confidence(x)

        # Effective confidence = model confidence * trust
        effective_confidence = archive_confidence * self.archive_trust

        # Routing decision
        use_archive = (
            effective_confidence >= self.config.effective_threshold
            and not force_dialogue
        )

        if use_archive:
            # FAST PATH: Archive
            output = archive_logits
            system_used = "archive"
            self.archive_calls += 1

            # Verify and update trust
            archive_correct = self._compute_accuracy(archive_logits, y)
            self.archive_trust = (
                self.config.archive_trust_alpha * self.archive_trust +
                (1 - self.config.archive_trust_alpha) * archive_correct
            )

        else:
            # SLOW PATH: Dialogue System
            self.net_A.train()
            self.net_B.train()

            out_A = self.net_A(x)
            out_B = self.net_B(x)

            # Compute surprise (disagreement between A and B)
            surprise = torch.nn.functional.mse_loss(out_A, out_B).item()

            # External error
            acc_A = self._compute_accuracy(out_A, y)
            external_error = 1.0 - acc_A

            # Update confidence hormone
            combined_signal = 0.3 * surprise + 0.7 * external_error
            self.confidence = (
                self.config.confidence_alpha * self.confidence +
                (1 - self.config.confidence_alpha) * (1 - combined_signal)
            )

            # Learn if surprised or uncertain
            if surprise > 0.1 or self.confidence < 0.6 or external_error > 0.3:
                # Train Net A
                loss_A = self.criterion(out_A, y)
                self.opt_A.zero_grad()
                loss_A.backward(retain_graph=True)
                self.opt_A.step()

                # Train Net B (slower)
                loss_B = self.criterion(out_B, y)
                self.opt_B.zero_grad()
                loss_B.backward()
                self.opt_B.step()

                # Store in memory if confident
                self.memory.add_experience(x, y, self.confidence)

            output = out_A
            system_used = "dialogue"
            self.dialogue_calls += 1

            # Update Archive trust based on what it WOULD have done
            archive_correct = self._compute_accuracy(archive_logits, y)
            self.archive_trust = (
                self.config.archive_trust_alpha * self.archive_trust +
                (1 - self.config.archive_trust_alpha) * archive_correct
            )

        # Compute final accuracy
        accuracy = self._compute_accuracy(output, y)

        # Record metrics
        self.metrics["system_used"].append(system_used)
        self.metrics["accuracy"].append(accuracy)
        self.metrics["archive_trust"].append(self.archive_trust)
        self.metrics["confidence"].append(self.confidence)

        self.total_steps += 1

        return {
            "system_used": system_used,
            "accuracy": accuracy,
            "archive_trust": self.archive_trust,
            "confidence": self.confidence,
            "effective_confidence": effective_confidence
        }

    def sleep_and_dream(self, verbose: bool = True) -> Dict:
        """
        Sleep phase with dreaming (memory replay).

        This is where the magic happens:
            1. Get consolidated memories (new + old)
            2. Train Archive on mixed dataset
            3. Clear short-term buffer
            4. Optionally boost trust if Archive learned well
        """
        data = self.memory.get_consolidation_data()
        memory_stats = self.memory.get_stats()

        if data is None:
            if verbose:
                print("  Sleep: No memories to consolidate")
            return {"memories_consolidated": 0, "archive_accuracy": 0}

        x_all, y_all = data
        x_all = x_all.to(self.device)
        y_all = y_all.to(self.device)

        # Flatten if needed
        if x_all.dim() > 2:
            x_all = x_all.view(x_all.size(0), -1)

        n_new = len(self.memory.short_term_buffer)
        n_total = len(x_all)
        n_replay = n_total - n_new

        if verbose:
            print(f"  Sleep: Consolidating {n_new} new + {n_replay} replayed = {n_total} total memories")
            print(f"         Core set: {memory_stats['core_set_size']} examples across {memory_stats['seen_classes']} classes")

        # Create dataloader for training
        dataset = torch.utils.data.TensorDataset(x_all, y_all)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        # Train Archive
        self.archive.train()
        total_loss = 0.0
        for epoch in range(self.config.consolidation_epochs):
            epoch_loss = 0.0
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                self.opt_archive.zero_grad()
                out = self.archive(bx)
                loss = self.criterion(out, by)
                loss.backward()
                self.opt_archive.step()
                epoch_loss += loss.item()
            total_loss += epoch_loss / len(loader)

        avg_loss = total_loss / self.config.consolidation_epochs

        # Evaluate Archive after training
        self.archive.eval()
        with torch.no_grad():
            out = self.archive(x_all)
            accuracy = self._compute_accuracy(out, y_all)

        # Trust boost if Archive learned well
        if accuracy > 0.9:
            old_trust = self.archive_trust
            self.archive_trust = min(0.95, self.archive_trust + 0.2)
            if verbose:
                print(f"         Trust boost: {old_trust:.2f} -> {self.archive_trust:.2f}")

        if verbose:
            print(f"         Archive accuracy: {accuracy:.1%}, loss: {avg_loss:.4f}")

        # Clear short-term buffer (keep core set)
        self.memory.clear_short_term()

        return {
            "memories_consolidated": n_total,
            "new_memories": n_new,
            "replayed_memories": n_replay,
            "archive_accuracy": accuracy,
            "archive_loss": avg_loss
        }

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        use_archive: bool = True
    ) -> Dict:
        """
        Evaluate on a test set.

        Args:
            dataloader: Test data
            use_archive: If True, evaluate Archive; else evaluate Dialogue

        Returns:
            Dictionary with accuracy per class and overall
        """
        network = self.archive if use_archive else self.net_A
        network.eval()

        correct = 0
        total = 0
        per_class_correct = {}
        per_class_total = {}

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                if x.dim() > 2:
                    x = x.view(x.size(0), -1)

                out = network(x)
                pred = out.argmax(dim=1)

                correct += (pred == y).sum().item()
                total += y.size(0)

                # Per-class accuracy
                for p, label in zip(pred, y):
                    label_int = label.item()
                    if label_int not in per_class_correct:
                        per_class_correct[label_int] = 0
                        per_class_total[label_int] = 0

                    per_class_total[label_int] += 1
                    if p.item() == label_int:
                        per_class_correct[label_int] += 1

        overall_acc = correct / total if total > 0 else 0

        per_class_acc = {}
        for c in per_class_correct:
            per_class_acc[c] = per_class_correct[c] / per_class_total[c]

        return {
            "overall_accuracy": overall_acc,
            "per_class_accuracy": per_class_acc,
            "total_samples": total
        }

    def get_state(self) -> Dict:
        """Get current brain state."""
        return {
            "total_steps": self.total_steps,
            "archive_calls": self.archive_calls,
            "dialogue_calls": self.dialogue_calls,
            "archive_usage_rate": self.archive_calls / max(self.total_steps, 1),
            "archive_trust": self.archive_trust,
            "confidence": self.confidence,
            "memory_stats": self.memory.get_stats()
        }


# =============================================================================
# IMAGINATION CORE - The Mind's Eye (VAE for Vivid Dreaming)
# =============================================================================

class ImaginationCore(nn.Module):
    """
    The Mind's Eye: A Variational Autoencoder for generating vivid dreams.

    Instead of dreaming with random noise (like static on a TV), the
    ImaginationCore learns to DRAW what it has seen. During sleep,
    it generates realistic "paintings" of past concepts.

    This is Generative Replay - the system doesn't just remember
    decision boundaries, it can recreate the experiences themselves.

    Architecture:
        Encoder: 784 -> 400 -> latent_dim (mu + logvar)
        Decoder: latent_dim -> 400 -> 784

    The latent space allows "concept algebra":
        - Interpolate between digits
        - Generate variations of learned concepts
        - "Play" with ideas during imagination
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: Compresses images into latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mu + logvar
        )

        # Decoder: Reconstructs images from latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Output in [0, 1] for images
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample z = mu + std * epsilon
        This allows gradients to flow through the sampling operation.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode -> sample -> decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def imagine(self, n_samples: int = 64, device: torch.device = None) -> torch.Tensor:
        """
        Generate vivid dreams by sampling from the learned latent space.

        Unlike random noise, these are structured samples that resemble
        the training data - actual digit-like images.
        """
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n_samples, self.latent_dim, device=device)
        with torch.no_grad():
            dreams = self.decode(z)
        return dreams

    def imagine_class(
        self,
        class_centroids: Dict[int, torch.Tensor],
        n_per_class: int = 10,
        noise_scale: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate dreams of specific classes by sampling around class centroids.

        This allows targeted replay of specific concepts during sleep.
        """
        dreams = []
        labels = []

        for class_id, centroid in class_centroids.items():
            # Sample around the centroid with some noise
            noise = torch.randn(n_per_class, self.latent_dim, device=centroid.device) * noise_scale
            z = centroid.unsqueeze(0) + noise
            with torch.no_grad():
                class_dreams = self.decode(z)
            dreams.append(class_dreams)
            labels.extend([class_id] * n_per_class)

        return torch.cat(dreams, dim=0), torch.tensor(labels, device=dreams[0].device)


@dataclass
class PlayerConfig(MNISTBrainConfig):
    """Extended configuration for TripartitePlayer with imagination."""
    # VAE settings
    latent_dim: int = 20
    vae_hidden_dim: int = 400
    vae_lr: float = 0.001

    # Play/Dream settings
    play_epochs: int = 5          # How many epochs to train VAE during play
    dream_samples: int = 500      # How many vivid dreams to generate
    dream_ratio: float = 0.5     # Ratio of dreams to real memories


class TripartitePlayer(MNISTTripartiteBrain):
    """
    The Complete Brain: Tripartite + Imagination (VAE-based Dreaming)

    Upgrades from MNISTTripartiteBrain:
        - ImaginationCore (VAE) for generating vivid dreams
        - play_time() method for learning to draw/imagine
        - Enhanced sleep_and_dream() with generative replay

    The Daily Cycle:
        1. Wake: Learn discriminatively (classify images)
        2. Play: Learn generatively (draw images with VAE)
        3. Sleep: Generate vivid dreams + consolidate to Archive

    This moves from "Preventing Forgetting" to "Active Consolidation"
    through imagination and creative replay.
    """

    def __init__(self, config: Optional[PlayerConfig] = None):
        self.player_config = config or PlayerConfig()
        super().__init__(self.player_config)

        # The Mind's Eye - Imagination Core
        self.imagination = ImaginationCore(
            input_dim=self.player_config.input_dim,
            hidden_dim=self.player_config.vae_hidden_dim,
            latent_dim=self.player_config.latent_dim
        ).to(self.device)

        self.opt_imagination = optim.Adam(
            self.imagination.parameters(),
            lr=self.player_config.vae_lr
        )

        # Track class centroids in latent space (for targeted dreams)
        self.class_centroids: Dict[int, torch.Tensor] = {}

        # Metrics for imagination
        self.imagination_losses: List[float] = []

    def _vae_loss(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        VAE loss = Reconstruction Loss + KL Divergence

        Reconstruction: How well can we recreate the input?
        KL Divergence: How close is our latent distribution to N(0,1)?
        """
        # Flatten x for comparison
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Reconstruction loss (binary cross entropy for images in [0,1])
        # Normalize x to [0,1] if needed
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        recon_loss = nn.functional.binary_cross_entropy(recon, x_norm, reduction='sum')

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return (recon_loss + kl_loss) / x.size(0)

    def play_time(self, x: torch.Tensor, y: torch.Tensor) -> Dict:
        """
        Recess/Play: Train the VAE to reconstruct current experiences.

        This is where the system learns to DRAW, not just classify.
        By learning to recreate inputs, the imagination develops
        an internal model of what things "look like."

        Also updates class centroids in latent space for targeted dreaming.
        """
        x = x.to(self.device)
        y = y.to(self.device)

        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        self.imagination.train()

        # Forward pass
        recon, mu, logvar = self.imagination(x)

        # Compute loss
        loss = self._vae_loss(recon, x, mu, logvar)

        # Backward pass
        self.opt_imagination.zero_grad()
        loss.backward()
        self.opt_imagination.step()

        # Update class centroids (exponential moving average)
        with torch.no_grad():
            mu_detached, _ = self.imagination.encode(x)
            for i, label in enumerate(y):
                label_int = label.item()
                if label_int not in self.class_centroids:
                    self.class_centroids[label_int] = mu_detached[i].clone()
                else:
                    # EMA update
                    self.class_centroids[label_int] = (
                        0.9 * self.class_centroids[label_int] +
                        0.1 * mu_detached[i]
                    )

        self.imagination_losses.append(loss.item())

        return {
            "vae_loss": loss.item(),
            "n_classes_imagined": len(self.class_centroids)
        }

    def wake_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        force_dialogue: bool = False,
        enable_play: bool = True
    ) -> Dict:
        """
        Extended wake step that includes play time.

        After learning (if we learned), we also train the imagination
        to recreate what we saw. This prepares us for vivid dreaming.
        """
        # Call parent wake_step
        result = super().wake_step(x, y, force_dialogue)

        # Play time: Train imagination if we processed consciously
        if enable_play and result["system_used"] == "dialogue":
            play_result = self.play_time(x, y)
            result["vae_loss"] = play_result["vae_loss"]

        return result

    def sleep_and_dream(self, verbose: bool = True, use_vivid_dreams: bool = True) -> Dict:
        """
        Enhanced sleep with vivid generative dreaming.

        Instead of random noise, we generate realistic "paintings"
        of past concepts using the ImaginationCore.

        The Process:
            1. Train imagination on today's buffer (final play session)
            2. Generate vivid dreams from imagination
            3. Have Archive label its own dreams (self-supervised)
            4. Mix real memories + vivid dreams
            5. Consolidate everything to Archive
        """
        memory_stats = self.memory.get_stats()

        if len(self.memory.short_term_buffer) == 0:
            if verbose:
                print("  Sleep: No memories to consolidate")
            return {"memories_consolidated": 0, "archive_accuracy": 0, "dream_type": "none"}

        # Collect today's memories
        xs, ys, _ = zip(*self.memory.short_term_buffer)
        real_x = torch.cat(xs, dim=0).to(self.device)
        real_y = torch.cat(ys, dim=0).to(self.device)

        if real_x.dim() > 2:
            real_x = real_x.view(real_x.size(0), -1)

        n_real = len(real_x)

        # === PHASE 1: Final Play Session (Train Imagination) ===
        if use_vivid_dreams and len(self.class_centroids) > 0:
            if verbose:
                print(f"  Sleep: Final play session - refining imagination...")

            dataset = torch.utils.data.TensorDataset(real_x, real_y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

            for _ in range(self.player_config.play_epochs):
                for bx, by in loader:
                    self.play_time(bx, by)

        # === PHASE 2: Generate Vivid Dreams ===
        if use_vivid_dreams and len(self.class_centroids) > 0:
            n_dreams = self.player_config.dream_samples

            # Generate dreams around class centroids
            dreams_per_class = n_dreams // len(self.class_centroids)
            dream_images, _ = self.imagination.imagine_class(
                self.class_centroids,
                n_per_class=dreams_per_class,
                noise_scale=0.5
            )

            # Archive labels its own dreams (self-supervised)
            with torch.no_grad():
                dream_logits = self.archive(dream_images)
                dream_labels = dream_logits.argmax(dim=1)

            dream_type = "vivid"
            if verbose:
                print(f"  Sleep: Generated {len(dream_images)} vivid dreams across {len(self.class_centroids)} classes")
        else:
            # Fallback to random noise dreams (pseudo-rehearsal)
            n_dreams = int(n_real * self.player_config.dream_ratio)
            dream_images = torch.randn(n_dreams, self.player_config.input_dim, device=self.device)

            with torch.no_grad():
                dream_logits = self.archive(dream_images)
                dream_labels = dream_logits.argmax(dim=1)

            dream_type = "noise"
            if verbose:
                print(f"  Sleep: Generated {n_dreams} noise dreams (fallback)")

        # === PHASE 3: Mix Reality + Dreams ===
        # Also include core set replay
        old_xs, old_ys = [], []
        for class_examples in self.memory.core_set.values():
            for ox, oy in class_examples:
                old_xs.append(ox.to(self.device))
                old_ys.append(oy.to(self.device))

        if old_xs:
            old_x = torch.cat(old_xs, dim=0)
            old_y = torch.cat(old_ys, dim=0)
            if old_x.dim() > 2:
                old_x = old_x.view(old_x.size(0), -1)

            # Combine: real + dreams + old memories
            final_x = torch.cat([real_x, dream_images, old_x], dim=0)
            final_y = torch.cat([real_y, dream_labels, old_y], dim=0)
            n_old = len(old_x)
        else:
            final_x = torch.cat([real_x, dream_images], dim=0)
            final_y = torch.cat([real_y, dream_labels], dim=0)
            n_old = 0

        n_total = len(final_x)
        n_dreams_actual = len(dream_images)

        if verbose:
            print(f"  Sleep: Consolidating {n_real} real + {n_dreams_actual} dreams + {n_old} replayed = {n_total} total")
            print(f"         Core set: {memory_stats['core_set_size']} examples across {memory_stats['seen_classes']} classes")

        # Update core set with today's high-confidence examples
        self.memory._update_core_set(real_x.cpu(), real_y.cpu())

        # === PHASE 4: Train Archive (Consolidation) ===
        dataset = torch.utils.data.TensorDataset(final_x, final_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        self.archive.train()
        total_loss = 0.0

        for epoch in range(self.player_config.consolidation_epochs):
            epoch_loss = 0.0
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                self.opt_archive.zero_grad()
                out = self.archive(bx)
                loss = self.criterion(out, by)
                loss.backward()
                self.opt_archive.step()
                epoch_loss += loss.item()
            total_loss += epoch_loss / len(loader)

        avg_loss = total_loss / self.player_config.consolidation_epochs

        # Evaluate Archive
        self.archive.eval()
        with torch.no_grad():
            out = self.archive(final_x)
            accuracy = (out.argmax(dim=1) == final_y).float().mean().item()

        # Trust boost
        if accuracy > 0.9:
            old_trust = self.archive_trust
            self.archive_trust = min(0.95, self.archive_trust + 0.2)
            if verbose:
                print(f"         Trust boost: {old_trust:.2f} -> {self.archive_trust:.2f}")

        if verbose:
            print(f"         Archive accuracy: {accuracy:.1%}, loss: {avg_loss:.4f}")

        # Clear short-term buffer
        self.memory.clear_short_term()

        return {
            "memories_consolidated": n_total,
            "real_memories": n_real,
            "dream_memories": n_dreams_actual,
            "replayed_memories": n_old,
            "archive_accuracy": accuracy,
            "archive_loss": avg_loss,
            "dream_type": dream_type
        }

    def get_state(self) -> Dict:
        """Extended state including imagination stats."""
        state = super().get_state()
        state["imagination"] = {
            "classes_learned": list(self.class_centroids.keys()),
            "n_classes": len(self.class_centroids),
            "avg_vae_loss": np.mean(self.imagination_losses[-100:]) if self.imagination_losses else 0
        }
        return state

    def visualize_dreams(self, n_samples: int = 16) -> torch.Tensor:
        """Generate sample dreams for visualization."""
        self.imagination.eval()
        dreams = self.imagination.imagine(n_samples, self.device)
        return dreams.view(n_samples, 28, 28).cpu()


if __name__ == "__main__":
    # Quick sanity check
    print("Creating MNIST Brain...")
    brain = MNISTTripartiteBrain()
    print(f"Device: {brain.device}")

    # Test with random data
    x = torch.randn(4, 784)
    y = torch.randint(0, 10, (4,))

    print("\nWake step test:")
    result = brain.wake_step(x, y)
    print(f"  System: {result['system_used']}, Accuracy: {result['accuracy']:.2f}")

    print("\nSleep test:")
    brain.memory.add_experience(x, y, 0.9)
    sleep_result = brain.sleep_and_dream()

    print("\nMNIST Brain ready!")
