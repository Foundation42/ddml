"""
Tripartite Brain Architecture - System 1 + System 2 Integration

This implements the full cognitive architecture:

    System 2 (Dialogue System): Conscious, expensive, plastic
        - Network A (Generator) + Network B (Monitor)
        - Handles novelty, conflict, uncertainty
        - High compute cost, high flexibility

    System 1 (Archive/Reflex): Subconscious, cheap, stable
        - Single optimized network
        - Handles "known" patterns automatically
        - Low compute cost, low flexibility

    Memory Buffer: Short-term storage for consolidation
        - Stores high-confidence Dialogue outputs
        - Transferred to Archive during "sleep"

The biological analogy:
    - Hippocampus (Dialogue): Fast learning, plastic, volatile
    - Neocortex (Archive): Slow learning, stable, permanent
    - Sleep: Consolidation from Hippocampus to Neocortex

Key Innovation: The system "learns to be lazy" - as patterns become
familiar, they migrate to the cheap Archive, freeing System 2 for
truly novel situations.

Author: Christian Beaumont & Claude
"""

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
import random

from dialogue_system import (
    DialogueSystem,
    SimpleMLP,
    HormoneConfig,
    LearningConfig,
    DialogueMetrics,
    create_dialogue_system
)


@dataclass
class ArchiveConfig:
    """
    Configuration for the Archive (System 1) network.

    The Archive is designed to be:
        - Fast: Quick inference for known patterns
        - Stable: Slow learning rate, resists forgetting
        - Confident: High threshold before trusting its output

    NOTE: effective_confidence = model_confidence * trust
    So threshold of 0.5 means we need ~70% confidence AND ~70% trust
    """
    hidden_dim: int = 32          # Can be smaller than Dialogue networks
    learning_rate: float = 0.01   # Increased for faster Archive learning
    confidence_threshold: float = 0.50  # Effective confidence threshold
    consolidation_epochs: int = 50      # More training passes during sleep


@dataclass
class MemoryBufferConfig:
    """
    Configuration for the short-term memory buffer.

    The buffer stores experiences for later consolidation.
    Think of it as "what happened today that I need to remember."
    """
    max_size: int = 1000          # Maximum experiences to store
    min_confidence: float = 0.75  # Only store high-quality memories
    retention_ratio: float = 0.1  # Keep 10% of memories after sleep (long-term replay)


@dataclass
class TripartiteMetrics:
    """Tracks metrics specific to the tripartite architecture."""
    # Per-step metrics
    system_used: List[str] = field(default_factory=list)  # "archive" or "dialogue"
    archive_confidence: List[float] = field(default_factory=list)
    dialogue_confidence: List[float] = field(default_factory=list)
    accuracy: List[float] = field(default_factory=list)

    # Per-day aggregates
    daily_archive_usage: List[float] = field(default_factory=list)
    daily_accuracy: List[float] = field(default_factory=list)
    memories_consolidated: List[int] = field(default_factory=list)

    # Cumulative
    total_archive_calls: int = 0
    total_dialogue_calls: int = 0

    def get_archive_usage_rate(self) -> float:
        """What fraction of calls used the Archive?"""
        total = self.total_archive_calls + self.total_dialogue_calls
        if total == 0:
            return 0.0
        return self.total_archive_calls / total

    def clear(self):
        """Reset all metrics."""
        self.system_used = []
        self.archive_confidence = []
        self.dialogue_confidence = []
        self.accuracy = []
        self.daily_archive_usage = []
        self.daily_accuracy = []
        self.memories_consolidated = []
        self.total_archive_calls = 0
        self.total_dialogue_calls = 0


class MemoryBuffer:
    """
    Short-term memory for experience replay and consolidation.

    Stores (input, target, confidence) tuples from successful
    Dialogue System processing. During "sleep", these memories
    are used to train the Archive network.

    INSIGHT: Not all memories are worth keeping. We filter by
    confidence - only experiences where the Dialogue System
    was highly confident get stored. This is analogous to how
    emotionally significant or repeated experiences are more
    likely to be consolidated in biological memory.
    """

    def __init__(self, config: MemoryBufferConfig):
        self.config = config
        self.buffer: Deque[Tuple[torch.Tensor, torch.Tensor, float]] = deque(maxlen=config.max_size)

    def add(self, x: torch.Tensor, y: torch.Tensor, confidence: float):
        """
        Add an experience to the buffer if confidence is high enough.

        Args:
            x: Input tensor
            y: Target tensor (ground truth)
            confidence: Dialogue system's confidence in this experience
        """
        if confidence >= self.config.min_confidence:
            # Store detached copies to avoid memory leaks
            self.buffer.append((
                x.detach().cpu(),
                y.detach().cpu(),
                confidence
            ))

    def sample_batch(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample a random batch from the buffer.

        Returns None if buffer is too small.
        """
        if len(self.buffer) < batch_size:
            return None

        samples = random.sample(list(self.buffer), batch_size)
        xs, ys, _ = zip(*samples)

        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    def get_all(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get all memories as a single batch."""
        if len(self.buffer) == 0:
            return None

        xs, ys, _ = zip(*self.buffer)
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    def clear(self, keep_ratio: float = 0.0):
        """
        Clear the buffer, optionally keeping some memories.

        Args:
            keep_ratio: Fraction of memories to retain (for long-term replay)
        """
        if keep_ratio > 0 and len(self.buffer) > 0:
            keep_count = int(len(self.buffer) * keep_ratio)
            # Keep the highest-confidence memories
            sorted_memories = sorted(self.buffer, key=lambda x: x[2], reverse=True)
            self.buffer = deque(sorted_memories[:keep_count], maxlen=self.config.max_size)
        else:
            self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class TripartiteBrain:
    """
    The complete cognitive architecture: System 1 (Archive) + System 2 (Dialogue).

    This class orchestrates:
        1. Input routing: Should we use Archive or Dialogue?
        2. Wake phase: Normal operation, learning, memory formation
        3. Sleep phase: Consolidation from memory buffer to Archive

    The key insight is COGNITIVE OFFLOADING:
        - New/uncertain situations -> expensive Dialogue processing
        - Known/confident situations -> cheap Archive lookup
        - Over time, more gets offloaded to Archive = more efficient

    CRITICAL INSIGHT (discovered during experimentation):
        Static confidence doesn't work! The Archive can be confidently WRONG
        after a task switch. We need DYNAMIC confidence that tracks actual
        performance. This is the "reality check" mechanism.

    This mirrors human expertise acquisition: novices think hard about
    everything, experts have automated most of their domain knowledge.
    But even experts must notice when their intuitions are failing!
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 16,
        output_dim: int = 1,
        personality: str = "balanced",
        archive_config: Optional[ArchiveConfig] = None,
        buffer_config: Optional[MemoryBufferConfig] = None,
        device: str = "auto"
    ):
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Configuration
        self.archive_config = archive_config or ArchiveConfig()
        self.buffer_config = buffer_config or MemoryBufferConfig()

        # System 2: The Dialogue System (Conscious Processing)
        self.dialogue = create_dialogue_system(
            personality=personality,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            device=str(self.device)
        )

        # System 1: The Archive (Subconscious/Reflex)
        self.archive = SimpleMLP(
            input_dim=input_dim,
            hidden_dim=self.archive_config.hidden_dim,
            output_dim=output_dim
        ).to(self.device)

        self.archive_optimizer = optim.Adam(
            self.archive.parameters(),
            lr=self.archive_config.learning_rate
        )
        self.archive_criterion = nn.BCEWithLogitsLoss()

        # Memory buffer for consolidation
        self.memory_buffer = MemoryBuffer(self.buffer_config)

        # Metrics
        self.metrics = TripartiteMetrics()

        # State tracking
        self.current_day = 0
        self.steps_today = 0
        self._daily_archive_uses = 0
        self._daily_steps = 0

        # CRITICAL: Dynamic trust in Archive (reality-checked confidence)
        # This tracks how well the Archive has been performing recently
        # High trust = let Archive handle things
        # Low trust = wake up Dialogue for verification
        self._archive_trust = 0.3  # Start with moderate trust (benefit of the doubt)
        self._archive_trust_alpha = 0.90  # Faster trust dynamics
        self._spot_check_rate = 0.2  # 20% spot check rate for faster feedback

    def _get_archive_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Get Archive's prediction and confidence level.

        Confidence is measured as how far from 0.5 the prediction is.
        A prediction of 0.99 or 0.01 is highly confident.
        A prediction of 0.51 is very uncertain.
        """
        with torch.no_grad():
            logits = self.archive(x)
            probs = torch.sigmoid(logits)

            # Confidence = distance from uncertainty (0.5)
            # Maps [0.5, 1.0] and [0.0, 0.5] to [0.0, 1.0]
            confidence = (torch.abs(probs - 0.5) * 2).mean().item()

        return logits, confidence

    def wake_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        force_dialogue: bool = False
    ) -> Dict[str, any]:
        """
        Process one input during the "wake" phase.

        The routing logic (with DYNAMIC TRUST):
            1. Check Archive model confidence AND accumulated trust
            2. If both are high enough -> use Archive (fast path)
            3. Otherwise -> use Dialogue (slow path)
            4. Periodically spot-check Archive to update trust
            5. If Dialogue was confident -> store in memory buffer

        The key innovation is TRUST vs CONFIDENCE:
            - Confidence: How sure the model is about THIS prediction
            - Trust: How well the model has been doing RECENTLY

        A model can be confident but wrong. Trust catches this by
        tracking actual performance over time.

        Args:
            x: Input tensor
            y: Target tensor (ground truth)
            force_dialogue: If True, always use Dialogue (for exploration)

        Returns:
            Dictionary of step metrics
        """
        x = x.to(self.device)
        y = y.to(self.device)

        # Step 1: Check Archive confidence (model's self-assessment)
        archive_logits, archive_confidence = self._get_archive_confidence(x)

        # Step 2: Compute effective confidence = model confidence * trust
        # This means even if model is confident, low trust will force Dialogue
        effective_confidence = archive_confidence * self._archive_trust

        # Step 3: Routing decision
        use_archive = (
            effective_confidence >= self.archive_config.confidence_threshold
            and not force_dialogue
        )

        # Step 4: Spot check - verify Archive accuracy to update trust
        # This is how we catch the "confidently wrong" problem
        do_spot_check = random.random() < self._spot_check_rate

        if use_archive:
            # FAST PATH: Use Archive
            output = archive_logits
            system_used = "archive"
            dialogue_confidence = 0.0

            self.metrics.total_archive_calls += 1
            self._daily_archive_uses += 1

            # Verify Archive accuracy for trust update
            archive_pred = (torch.sigmoid(archive_logits) > 0.5).float()
            target_binary = (y > 0.5).float()
            archive_correct = (archive_pred == target_binary).float().mean().item()

            # Update trust based on actual performance
            # If Archive is right -> trust increases
            # If Archive is wrong -> trust crashes
            self._archive_trust = (
                self._archive_trust_alpha * self._archive_trust +
                (1 - self._archive_trust_alpha) * archive_correct
            )

            # If trust drops too low, this will naturally wake up Dialogue
            # on subsequent steps (since effective_confidence will be low)

        else:
            # SLOW PATH: Use Dialogue System
            result = self.dialogue.step(x, y, train=True)
            output = self.dialogue.net_A(x)
            system_used = "dialogue"
            dialogue_confidence = result["confidence"]

            self.metrics.total_dialogue_calls += 1

            # Memory formation for high-confidence Dialogue outputs
            if dialogue_confidence >= self.buffer_config.min_confidence:
                self.memory_buffer.add(x, y, dialogue_confidence)

            # ALWAYS verify Archive when using Dialogue
            # This builds trust faster when Archive would have been correct
            # "I could have used Archive and it would have worked"
            archive_pred = (torch.sigmoid(archive_logits) > 0.5).float()
            target_binary = (y > 0.5).float()
            archive_correct = (archive_pred == target_binary).float().mean().item()

            # Update trust - this happens every step when using Dialogue
            self._archive_trust = (
                self._archive_trust_alpha * self._archive_trust +
                (1 - self._archive_trust_alpha) * archive_correct
            )

        # Compute final accuracy
        predictions = (torch.sigmoid(output) > 0.5).float()
        target_binary = (y > 0.5).float()
        accuracy = (predictions == target_binary).float().mean().item()

        # Record metrics
        self.metrics.system_used.append(system_used)
        self.metrics.archive_confidence.append(archive_confidence)
        self.metrics.dialogue_confidence.append(dialogue_confidence)
        self.metrics.accuracy.append(accuracy)

        self._daily_steps += 1

        return {
            "system_used": system_used,
            "archive_confidence": archive_confidence,
            "archive_trust": self._archive_trust,
            "effective_confidence": effective_confidence,
            "dialogue_confidence": dialogue_confidence,
            "accuracy": accuracy,
            "memory_buffer_size": len(self.memory_buffer)
        }

    def sleep_consolidate(self, verbose: bool = False) -> Dict[str, any]:
        """
        The "sleep" phase: consolidate memories to Archive.

        This is where short-term memories (experiences from the day)
        get transferred to long-term storage (the Archive network).

        The Archive learns from the memory buffer, essentially
        "replaying" the day's important experiences.

        INSIGHT: We only consolidate what the Dialogue System was
        confident about. This filters out noise and uncertainty,
        ensuring the Archive only learns "verified" patterns.
        """
        memories_count = len(self.memory_buffer)

        if memories_count == 0:
            if verbose:
                print("  Sleep: No memories to consolidate")
            return {"memories_consolidated": 0, "archive_loss": 0.0, "archive_accuracy": 0.0}

        # Get all memories
        data = self.memory_buffer.get_all()
        if data is None:
            return {"memories_consolidated": 0, "archive_loss": 0.0, "archive_accuracy": 0.0}

        x_all, y_all = data
        x_all = x_all.to(self.device)
        y_all = y_all.to(self.device)

        # Train Archive on memories
        total_loss = 0.0
        for epoch in range(self.archive_config.consolidation_epochs):
            self.archive_optimizer.zero_grad()
            output = self.archive(x_all)
            loss = self.archive_criterion(output, y_all)
            loss.backward()
            self.archive_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / self.archive_config.consolidation_epochs

        # Compute Archive accuracy and confidence after training
        with torch.no_grad():
            output = self.archive(x_all)
            probs = torch.sigmoid(output)
            predictions = (probs > 0.5).float()
            accuracy = (predictions == y_all).float().mean().item()
            confidence = (torch.abs(probs - 0.5) * 2).mean().item()

        # TRUST BOOST: If Archive learned well, boost trust!
        # This allows the Archive to take over faster after good sleep
        if accuracy > 0.9:
            old_trust = self._archive_trust
            self._archive_trust = min(0.95, self._archive_trust + 0.3)  # Significant boost
            trust_delta = self._archive_trust - old_trust
        else:
            trust_delta = 0.0

        if verbose:
            print(f"  Sleep: Consolidated {memories_count} memories")
            print(f"         Archive loss: {avg_loss:.4f}, accuracy: {accuracy:.1%}, confidence: {confidence:.2f}")
            if trust_delta > 0:
                print(f"         Trust boost: +{trust_delta:.2f} (Archive learned well!)")

        # Record consolidation
        self.metrics.memories_consolidated.append(memories_count)

        # Clear buffer (keeping some for long-term replay)
        self.memory_buffer.clear(keep_ratio=self.buffer_config.retention_ratio)

        return {
            "memories_consolidated": memories_count,
            "archive_loss": avg_loss,
            "archive_accuracy": accuracy,
            "archive_confidence": confidence,
            "trust_after_sleep": self._archive_trust
        }

    def end_day(self, verbose: bool = False):
        """
        End the current day: record daily metrics and run sleep consolidation.
        """
        # Record daily aggregates
        if self._daily_steps > 0:
            daily_archive_rate = self._daily_archive_uses / self._daily_steps
            daily_acc = sum(self.metrics.accuracy[-self._daily_steps:]) / self._daily_steps
        else:
            daily_archive_rate = 0.0
            daily_acc = 0.0

        self.metrics.daily_archive_usage.append(daily_archive_rate)
        self.metrics.daily_accuracy.append(daily_acc)

        if verbose:
            print(f"Day {self.current_day + 1} complete:")
            print(f"  Steps: {self._daily_steps}")
            print(f"  Archive usage: {daily_archive_rate:.1%}")
            print(f"  Accuracy: {daily_acc:.1%}")
            print(f"  Memory buffer: {len(self.memory_buffer)} experiences")

        # Sleep consolidation
        sleep_result = self.sleep_consolidate(verbose=verbose)

        # Reset daily counters
        self._daily_archive_uses = 0
        self._daily_steps = 0
        self.current_day += 1

        return sleep_result

    def get_state(self) -> Dict:
        """Get current state summary."""
        return {
            "current_day": self.current_day,
            "archive_usage_rate": self.metrics.get_archive_usage_rate(),
            "memory_buffer_size": len(self.memory_buffer),
            "dialogue_confidence": self.dialogue.get_confidence(),
            "total_archive_calls": self.metrics.total_archive_calls,
            "total_dialogue_calls": self.metrics.total_dialogue_calls,
        }

    def save(self, path: str):
        """Save the complete brain state."""
        torch.save({
            'archive': self.archive.state_dict(),
            'archive_optimizer': self.archive_optimizer.state_dict(),
            'dialogue_net_A': self.dialogue.net_A.state_dict(),
            'dialogue_net_B': self.dialogue.net_B.state_dict(),
            'current_day': self.current_day,
            'archive_config': self.archive_config,
            'buffer_config': self.buffer_config,
        }, path)

    def load(self, path: str):
        """Load brain state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.archive.load_state_dict(checkpoint['archive'])
        self.archive_optimizer.load_state_dict(checkpoint['archive_optimizer'])
        self.dialogue.net_A.load_state_dict(checkpoint['dialogue_net_A'])
        self.dialogue.net_B.load_state_dict(checkpoint['dialogue_net_B'])
        self.current_day = checkpoint['current_day']


def create_tripartite_brain(
    personality: str = "balanced",
    archive_confidence_threshold: float = 0.50,
    **kwargs
) -> TripartiteBrain:
    """
    Convenience function to create a TripartiteBrain with common configurations.

    Args:
        personality: Dialogue system personality ("stubborn", "balanced", "anxious")
        archive_confidence_threshold: How confident Archive must be to bypass Dialogue
        **kwargs: Passed to TripartiteBrain constructor
    """
    archive_config = ArchiveConfig(confidence_threshold=archive_confidence_threshold)
    return TripartiteBrain(
        personality=personality,
        archive_config=archive_config,
        **kwargs
    )


if __name__ == "__main__":
    # Quick sanity check
    print("Creating TripartiteBrain...")
    brain = create_tripartite_brain(input_dim=2, hidden_dim=16, output_dim=1)
    print(f"Device: {brain.device}")

    # Test with random data
    x = torch.randn(4, 2)
    y = torch.randint(0, 2, (4, 1)).float()

    print("\nWake step test:")
    result = brain.wake_step(x, y)
    print(f"  System used: {result['system_used']}")
    print(f"  Archive confidence: {result['archive_confidence']:.3f}")
    print(f"  Accuracy: {result['accuracy']:.3f}")

    print("\nSleep consolidation test:")
    # Add some fake memories
    for _ in range(10):
        brain.memory_buffer.add(torch.randn(1, 2), torch.randint(0, 2, (1, 1)).float(), 0.9)

    sleep_result = brain.sleep_consolidate(verbose=True)

    print("\nTripartite Brain ready!")
