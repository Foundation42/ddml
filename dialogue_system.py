"""
Dialogue-Driven Machine Learning (DDML) - Core Implementation

The central thesis: Intelligence emerges from controlled internal conflict,
not perfect prediction. Two networks engage in continuous dialogue, with
disagreement driving selective learning.

Architecture:
    - Network A (Generator): Fast learner, makes predictions, explores
    - Network B (Monitor): Slow learner, maintains stable "reality model"
    - Diff Engine: Measures disagreement between A and B
    - Confidence Hormone: Global signal modulating plasticity

Key insight from cognitive science: The brain doesn't learn from every
experience equally. It learns most when surprised - when predictions
don't match reality. This selective learning prevents catastrophic
forgetting and enables efficient online adaptation.

Author: Christian Beaumont & Claude
"""

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class HormoneConfig:
    """
    Configuration for the confidence hormone system.

    The alpha parameter controls "personality":
        - High alpha (0.99): Stubborn, slow to lose confidence
        - Medium alpha (0.95): Balanced adaptation
        - Low alpha (0.80): Anxious, quick to doubt itself

    These could potentially be grounded in neurotransmitter dynamics:
        - Dopamine half-life in synaptic cleft: ~200ms
        - Norepinephrine clearance: varies by region

    The surprise_threshold determines when learning kicks in.
    Think of it as the "attention threshold" - how novel must
    something be before we pay attention?
    """
    alpha: float = 0.95           # Hormone decay/momentum
    initial_confidence: float = 0.5
    surprise_threshold: float = 0.1  # Lowered - trigger learning on modest disagreement
    confidence_floor: float = 0.2    # Allow confidence to go quite low
    confidence_ceiling: float = 0.95  # Cap confidence to always maintain some plasticity


@dataclass
class LearningConfig:
    """
    Learning rate configuration for the dual networks.

    Key insight: A learns fast (exploration), B learns slow (stability).
    The ratio between them matters more than absolute values.

    A ratio of 10:1 means A adapts 10x faster than B.
    This creates a natural "working memory" (A) vs "long-term memory" (B) dynamic.
    """
    lr_A: float = 0.01   # Generator learning rate (fast/plastic)
    lr_B: float = 0.001  # Monitor learning rate (slow/stable)

    # NEW INSIGHT: Modulate learning rate by confidence
    # When confident, learn slowly (don't mess with what works)
    # When uncertain, learn quickly (need to adapt!)
    modulate_by_confidence: bool = True
    min_lr_multiplier: float = 0.1  # At max confidence, LR reduced to 10%
    max_lr_multiplier: float = 2.0  # At min confidence, LR boosted to 200%


@dataclass
class DialogueMetrics:
    """Tracks the internal state for analysis and visualization."""
    surprise_history: List[float] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    accuracy_A_history: List[float] = field(default_factory=list)
    accuracy_B_history: List[float] = field(default_factory=list)
    agreement_history: List[float] = field(default_factory=list)  # How often A and B agree
    update_triggered: List[bool] = field(default_factory=list)  # Did we learn this step?
    effective_lr_history: List[float] = field(default_factory=list)

    def clear(self):
        """Reset all histories."""
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, [])


class SimpleMLP(nn.Module):
    """
    A simple feedforward network.

    For XOR, we need at least one hidden layer (it's not linearly separable).
    We use a small network to keep things interpretable.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 16,
        output_dim: int = 1,
        hidden_layers: int = 1
    ):
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer (no activation - we'll use BCEWithLogitsLoss)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffEngine:
    """
    The Corpus Callosum - measures disagreement between networks.

    INSIGHT: There are multiple ways to measure "disagreement":

    1. Output disagreement: Do A and B predict differently?
       - Simple, interpretable
       - But might miss internal representation differences

    2. Representation disagreement: Do A and B "think" differently?
       - Compare hidden layer activations
       - Richer signal, but harder to interpret

    3. Gradient disagreement: Would A and B update differently?
       - Most sophisticated
       - Captures "what they want to learn"

    For now, we use output disagreement comparing probability predictions.
    This is the simplest and most interpretable starting point.
    """

    @staticmethod
    def compute_surprise(
        output_A: torch.Tensor,
        output_B: torch.Tensor,
        normalize: bool = True
    ) -> float:
        """
        Compute the "surprise" signal from network disagreement.

        We compare the probability outputs (after sigmoid) rather than
        raw logits. This gives us a more interpretable 0-1 range.

        Args:
            output_A: Logits from Network A
            output_B: Logits from Network B
            normalize: If True, uses probability space (recommended)

        Returns:
            Scalar surprise value in [0, 1] range
        """
        if normalize:
            # Compare in probability space - much more interpretable!
            # If both predict 0.9, disagreement is low
            # If A predicts 0.9 and B predicts 0.1, disagreement is high
            prob_A = torch.sigmoid(output_A)
            prob_B = torch.sigmoid(output_B)
            # Mean absolute difference in probabilities
            # This naturally gives us a 0-1 range
            surprise = torch.abs(prob_A - prob_B).mean().item()
        else:
            # Raw MSE of logits (can be any scale)
            surprise = torch.nn.functional.mse_loss(output_A, output_B).item()

        return surprise

    @staticmethod
    def compute_agreement_rate(
        output_A: torch.Tensor,
        output_B: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """
        What fraction of predictions do A and B agree on?

        This is a more interpretable metric than raw MSE.
        """
        pred_A = (torch.sigmoid(output_A) > threshold).float()
        pred_B = (torch.sigmoid(output_B) > threshold).float()
        agreement = (pred_A == pred_B).float().mean().item()
        return agreement


class ConfidenceHormone:
    """
    The global confidence signal - a meta-learning mechanism.

    CRITICAL INSIGHT: Pure internal agreement is insufficient!
    Two networks can confidently agree on wrong answers. We need to
    track BOTH:
        1. Internal agreement (do A and B agree?)
        2. External accuracy (are we getting things right?)

    The hormone signal should drop when:
        - A and B disagree (internal conflict)
        - We're making errors (external feedback)

    This is analogous to how neuromodulators work:
        - Dopamine: Signals prediction error, modulates plasticity
        - Norepinephrine: Signals uncertainty/arousal, increases attention
        - Acetylcholine: Modulates learning rate in cortex

    The hormone affects learning in two ways:
        1. Gating: Below threshold, learning is suppressed
        2. Modulation: Learning rate scales with uncertainty
    """

    def __init__(self, config: HormoneConfig):
        self.config = config
        self.confidence = config.initial_confidence
        self._error_ema = 0.5  # Track running error rate

    def update(
        self,
        internal_surprise: float,
        external_error: float,
        error_weight: float = 0.7
    ) -> float:
        """
        Update confidence based on both internal and external signals.

        Args:
            internal_surprise: Disagreement between A and B (0-1)
            external_error: How wrong we are vs ground truth (0-1)
            error_weight: How much to weight external error vs internal surprise

        Returns the new confidence value.
        """
        # Combined surprise signal:
        # - Internal surprise: A and B disagree
        # - External error: We're making mistakes
        # External error is more important for learning (hence higher weight)
        combined_surprise = (
            (1 - error_weight) * internal_surprise +
            error_weight * external_error
        )

        # Stability is inverse of combined surprise
        stability = 1.0 - combined_surprise

        # Exponential moving average
        self.confidence = (
            self.config.alpha * self.confidence +
            (1 - self.config.alpha) * stability
        )

        # Track running error
        self._error_ema = 0.9 * self._error_ema + 0.1 * external_error

        # Clamp to bounds
        self.confidence = max(
            self.config.confidence_floor,
            min(self.config.confidence_ceiling, self.confidence)
        )

        return self.confidence

    def should_learn(self, combined_surprise: float) -> bool:
        """
        Determine if learning should occur.

        We learn when:
            1. Combined surprise exceeds threshold
            2. OR confidence is low
            3. OR running error rate is high
        """
        return (
            combined_surprise > self.config.surprise_threshold or
            self.confidence < 0.6 or
            self._error_ema > 0.3
        )

    def get_lr_multiplier(self) -> float:
        """
        Get learning rate multiplier based on current confidence.

        High confidence -> low multiplier (don't mess with success)
        Low confidence -> high multiplier (need to adapt fast)
        """
        conf_normalized = (self.confidence - self.config.confidence_floor) / (
            self.config.confidence_ceiling - self.config.confidence_floor + 1e-8
        )

        # Invert: high confidence = low multiplier
        multiplier = 0.1 + (1 - conf_normalized) * 1.9  # Range from 0.1 to 2.0

        return max(0.1, min(2.0, multiplier))

    def reset(self):
        """Reset to initial state."""
        self.confidence = self.config.initial_confidence
        self._error_ema = 0.5


class DialogueSystem:
    """
    The complete Dialogue-Driven Learning System.

    This is the main class that orchestrates:
        - Two competing/cooperating networks (A and B)
        - A diff engine measuring their disagreement
        - A confidence hormone modulating learning

    The key innovation is SELECTIVE LEARNING:
        - Not every input triggers weight updates
        - Only surprising or uncertain situations cause learning
        - This naturally implements "attention" and prevents catastrophic forgetting
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 16,
        output_dim: int = 1,
        hormone_config: Optional[HormoneConfig] = None,
        learning_config: Optional[LearningConfig] = None,
        device: str = "auto"
    ):
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Configuration
        self.hormone_config = hormone_config or HormoneConfig()
        self.learning_config = learning_config or LearningConfig()

        # Networks
        self.net_A = SimpleMLP(input_dim, hidden_dim, output_dim).to(self.device)
        self.net_B = SimpleMLP(input_dim, hidden_dim, output_dim).to(self.device)

        # Optimizers
        self.opt_A = optim.Adam(self.net_A.parameters(), lr=self.learning_config.lr_A)
        self.opt_B = optim.Adam(self.net_B.parameters(), lr=self.learning_config.lr_B)

        # Components
        self.diff_engine = DiffEngine()
        self.hormone = ConfidenceHormone(self.hormone_config)

        # Metrics tracking
        self.metrics = DialogueMetrics()

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Step counter
        self.step_count = 0

    def _compute_accuracy(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """Compute binary classification accuracy."""
        predictions = (torch.sigmoid(output) > threshold).float()
        target_binary = (target > threshold).float()
        accuracy = (predictions == target_binary).float().mean().item()
        return accuracy

    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        train: bool = True
    ) -> Dict[str, float]:
        """
        Process one batch through the dialogue system.

        The Learning Cycle:
            1. Prediction: Both networks generate outputs
            2. Comparison: Diff engine computes internal surprise (A vs B)
            3. Error measurement: Compute external error (prediction vs ground truth)
            4. Hormone update: Confidence adjusts based on BOTH internal and external signals
            5. Selective update: Only update if surprised, uncertain, or making errors

        Args:
            x: Input tensor
            y: Target tensor
            train: Whether to update weights

        Returns:
            Dictionary of metrics for this step
        """
        x = x.to(self.device)
        y = y.to(self.device)

        # 1. PREDICTION PHASE
        # Both networks make predictions
        out_A = self.net_A(x)
        out_B = self.net_B(x)

        # 2. COMPARISON PHASE (The Internal Dialogue)
        # How much do A and B disagree?
        internal_surprise = self.diff_engine.compute_surprise(out_A, out_B)
        agreement = self.diff_engine.compute_agreement_rate(out_A, out_B)

        # 3. EXTERNAL ERROR (Reality Check)
        # This is crucial - we need to know when we're wrong, not just when we disagree
        # Use Network A's prediction error as the external signal
        acc_A = self._compute_accuracy(out_A, y)
        acc_B = self._compute_accuracy(out_B, y)
        external_error = 1.0 - acc_A  # Error rate

        # 4. HORMONE UPDATE
        # Update confidence based on BOTH internal surprise and external error
        # This is the key insight: confidence should drop when we're wrong OR when we disagree
        combined_surprise = 0.3 * internal_surprise + 0.7 * external_error
        old_confidence = self.hormone.confidence
        new_confidence = self.hormone.update(internal_surprise, external_error)

        # 5. SELECTIVE LEARNING
        # Only learn if surprised, uncertain, or making errors
        should_learn = self.hormone.should_learn(combined_surprise) and train

        effective_lr = 0.0
        if should_learn:
            # Get learning rate multiplier from hormone
            if self.learning_config.modulate_by_confidence:
                lr_mult = self.hormone.get_lr_multiplier()
            else:
                lr_mult = 1.0

            effective_lr = self.learning_config.lr_A * lr_mult

            # Temporarily adjust learning rates
            for param_group in self.opt_A.param_groups:
                param_group['lr'] = self.learning_config.lr_A * lr_mult
            for param_group in self.opt_B.param_groups:
                param_group['lr'] = self.learning_config.lr_B * lr_mult

            # Network A learns from ground truth
            loss_A = self.criterion(out_A, y)
            self.opt_A.zero_grad()
            loss_A.backward(retain_graph=True)
            self.opt_A.step()

            # Network B also learns, but slower (it's the "stable branch")
            loss_B = self.criterion(out_B, y)
            self.opt_B.zero_grad()
            loss_B.backward()
            self.opt_B.step()

        # 6. METRICS
        # Record history
        self.metrics.surprise_history.append(combined_surprise)  # Track combined signal
        self.metrics.confidence_history.append(new_confidence)
        self.metrics.accuracy_A_history.append(acc_A)
        self.metrics.accuracy_B_history.append(acc_B)
        self.metrics.agreement_history.append(agreement)
        self.metrics.update_triggered.append(should_learn)
        self.metrics.effective_lr_history.append(effective_lr)

        self.step_count += 1

        return {
            "surprise": combined_surprise,
            "internal_surprise": internal_surprise,
            "external_error": external_error,
            "confidence": new_confidence,
            "accuracy_A": acc_A,
            "accuracy_B": acc_B,
            "agreement": agreement,
            "updated": should_learn,
            "effective_lr": effective_lr
        }

    def predict(self, x: torch.Tensor, use_network: str = "A") -> torch.Tensor:
        """
        Make predictions without learning.

        Args:
            x: Input tensor
            use_network: Which network to use ("A", "B", or "ensemble")

        Returns:
            Predictions (probabilities after sigmoid)
        """
        x = x.to(self.device)

        with torch.no_grad():
            if use_network == "A":
                out = self.net_A(x)
            elif use_network == "B":
                out = self.net_B(x)
            elif use_network == "ensemble":
                # Average both networks' predictions
                out = (self.net_A(x) + self.net_B(x)) / 2
            else:
                raise ValueError(f"Unknown network: {use_network}")

        return torch.sigmoid(out)

    def get_confidence(self) -> float:
        """Get current confidence level."""
        return self.hormone.confidence

    def reset_hormone(self):
        """Reset the hormone system (useful for new tasks)."""
        self.hormone.reset()

    def save(self, path: str):
        """Save model state."""
        torch.save({
            'net_A': self.net_A.state_dict(),
            'net_B': self.net_B.state_dict(),
            'opt_A': self.opt_A.state_dict(),
            'opt_B': self.opt_B.state_dict(),
            'hormone_confidence': self.hormone.confidence,
            'step_count': self.step_count,
            'hormone_config': self.hormone_config,
            'learning_config': self.learning_config,
        }, path)

    def load(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.net_A.load_state_dict(checkpoint['net_A'])
        self.net_B.load_state_dict(checkpoint['net_B'])
        self.opt_A.load_state_dict(checkpoint['opt_A'])
        self.opt_B.load_state_dict(checkpoint['opt_B'])
        self.hormone.confidence = checkpoint['hormone_confidence']
        self.step_count = checkpoint['step_count']


# Convenience function to create system with different "personalities"
def create_dialogue_system(
    personality: str = "balanced",
    **kwargs
) -> DialogueSystem:
    """
    Create a DialogueSystem with a preset personality.

    Personalities:
        - "stubborn": High alpha (0.99), slow to change beliefs
        - "balanced": Medium alpha (0.95), good all-rounder
        - "anxious": Low alpha (0.80), quick to doubt, fast to adapt
        - "curious": Low surprise threshold, learns from small novelties
    """
    personality_configs = {
        "stubborn": HormoneConfig(alpha=0.99, surprise_threshold=0.3),
        "balanced": HormoneConfig(alpha=0.95, surprise_threshold=0.2),
        "anxious": HormoneConfig(alpha=0.80, surprise_threshold=0.15),
        "curious": HormoneConfig(alpha=0.90, surprise_threshold=0.1),
    }

    if personality not in personality_configs:
        raise ValueError(f"Unknown personality: {personality}. Choose from {list(personality_configs.keys())}")

    return DialogueSystem(hormone_config=personality_configs[personality], **kwargs)


if __name__ == "__main__":
    # Quick sanity check
    print("Creating DialogueSystem...")
    system = DialogueSystem(input_dim=2, hidden_dim=16, output_dim=1)
    print(f"Device: {system.device}")

    # Test with random data
    x = torch.randn(4, 2)
    y = torch.randint(0, 2, (4, 1)).float()

    result = system.step(x, y)
    print(f"Step result: {result}")

    print("\nDialogue System ready!")
