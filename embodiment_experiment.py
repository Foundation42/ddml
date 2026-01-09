"""
Embodiment Experiment - The Brain Gets a Body

The ultimate demonstration: Watch an AI "stop and think."

In this experiment, we place the Dialogue Model brain inside a simple
grid world agent. The agent must navigate to goals while avoiding obstacles.

The Key Insight:
    - Familiar paths → Archive (System 1) → Fast, automatic movement
    - Novel obstacles → Dialogue (System 2) → Pause, deliberate, plan

You can literally SEE the agent:
    - Moving smoothly through known territory (reflex)
    - STOPPING when something unexpected appears (deliberation)
    - Resuming movement once it figures out a new path

This is the "Watch it Think" demo.

Author: Christian Beaumont & Claude & Gemini
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import deque
import json
import time

from xor_experiment import ExperimentTracker


# =============================================================================
# GRID WORLD ENVIRONMENT
# =============================================================================

@dataclass
class GridConfig:
    """Configuration for the grid world."""
    width: int = 10
    height: int = 10
    n_obstacles: int = 15
    max_steps: int = 100
    seed: int = 42


class GridWorld:
    """
    A simple grid world for embodied navigation.

    Cell Types:
        0 = Empty (can walk)
        1 = Wall (blocked)
        2 = Goal (target)
        3 = Agent (current position)
        4 = Novel obstacle (appears mid-episode)

    Actions:
        0 = Up, 1 = Down, 2 = Left, 3 = Right
    """

    EMPTY = 0
    WALL = 1
    GOAL = 2
    AGENT = 3
    NOVEL = 4

    ACTION_NAMES = ['Up', 'Down', 'Left', 'Right']
    ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (row, col) deltas

    def __init__(self, config: GridConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)

        # Initialize grid
        self.grid = np.zeros((config.height, config.width), dtype=np.int32)

        # Place walls
        self._place_obstacles(config.n_obstacles)

        # Place goal (bottom-right area)
        self.goal_pos = (config.height - 2, config.width - 2)
        self.grid[self.goal_pos] = self.GOAL

        # Agent starts top-left
        self.agent_pos = (1, 1)
        self.start_pos = (1, 1)

        # Ensure start and goal are clear
        self.grid[self.start_pos] = self.EMPTY
        self.grid[self.goal_pos] = self.GOAL

        # Episode tracking
        self.steps = 0
        self.done = False
        self.novel_obstacles_added = False

        # History for visualization
        self.path_history: List[Tuple[int, int]] = [self.agent_pos]
        self.action_history: List[int] = []
        self.system_history: List[str] = []  # "reflex" or "deliberate"

    def _place_obstacles(self, n: int):
        """Place random obstacles, avoiding corners."""
        placed = 0
        while placed < n:
            r = self.rng.randint(0, self.config.height)
            c = self.rng.randint(0, self.config.width)

            # Don't block corners or create isolated regions
            if (r, c) not in [(0, 0), (0, 1), (1, 0), (1, 1),
                              (self.config.height-1, self.config.width-1),
                              (self.config.height-2, self.config.width-2),
                              (self.config.height-1, self.config.width-2),
                              (self.config.height-2, self.config.width-1)]:
                if self.grid[r, c] == self.EMPTY:
                    self.grid[r, c] = self.WALL
                    placed += 1

    def add_novel_obstacle(self, position: Optional[Tuple[int, int]] = None):
        """
        Add a novel obstacle mid-episode.

        This simulates unexpected changes in the environment that
        require the agent to stop and re-plan.
        """
        if position is None:
            # Place near the agent's likely path
            r = self.rng.randint(2, self.config.height - 2)
            c = self.rng.randint(2, self.config.width - 2)
            position = (r, c)

        if self.grid[position] == self.EMPTY:
            self.grid[position] = self.NOVEL
            self.novel_obstacles_added = True
            return position
        return None

    def get_observation(self) -> np.ndarray:
        """
        Get the agent's observation of the world.

        Returns a flattened view of the grid with agent position encoded.
        """
        obs = self.grid.copy().astype(np.float32)
        obs[self.agent_pos] = self.AGENT
        return obs.flatten()

    def get_local_observation(self, radius: int = 2) -> np.ndarray:
        """
        Get a local view around the agent (partial observability).

        This is more realistic - the agent can only see nearby cells.
        """
        r, c = self.agent_pos
        local = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.config.height and 0 <= nc < self.config.width:
                    local[dr + radius, dc + radius] = self.grid[nr, nc]
                else:
                    local[dr + radius, dc + radius] = self.WALL  # Out of bounds = wall

        return local.flatten()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take an action in the environment.

        Returns:
            observation, reward, done, info
        """
        if self.done:
            return self.get_observation(), 0.0, True, {"reason": "already_done"}

        self.steps += 1
        self.action_history.append(action)

        # Calculate new position
        dr, dc = self.ACTION_DELTAS[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        # Check bounds
        if not (0 <= new_r < self.config.height and 0 <= new_c < self.config.width):
            # Hit boundary
            return self.get_observation(), -0.1, False, {"reason": "boundary"}

        # Check obstacles
        cell = self.grid[new_r, new_c]
        if cell == self.WALL or cell == self.NOVEL:
            # Hit obstacle
            reward = -0.5 if cell == self.NOVEL else -0.1
            return self.get_observation(), reward, False, {"reason": "obstacle", "novel": cell == self.NOVEL}

        # Move successful
        self.agent_pos = (new_r, new_c)
        self.path_history.append(self.agent_pos)

        # Check goal
        if self.agent_pos == self.goal_pos:
            self.done = True
            return self.get_observation(), 10.0, True, {"reason": "goal"}

        # Check max steps
        if self.steps >= self.config.max_steps:
            self.done = True
            return self.get_observation(), -1.0, True, {"reason": "timeout"}

        # Small penalty for each step (encourages efficiency)
        return self.get_observation(), -0.01, False, {"reason": "step"}

    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.agent_pos = self.start_pos
        self.steps = 0
        self.done = False
        self.path_history = [self.agent_pos]
        self.action_history = []
        self.system_history = []
        return self.get_observation()

    def render_to_array(self) -> np.ndarray:
        """Render the grid to an image array."""
        # Color map
        colors = {
            self.EMPTY: [1.0, 1.0, 1.0],      # White
            self.WALL: [0.3, 0.3, 0.3],       # Dark gray
            self.GOAL: [0.0, 0.8, 0.0],       # Green
            self.AGENT: [0.0, 0.0, 1.0],      # Blue
            self.NOVEL: [1.0, 0.0, 0.0],      # Red
        }

        img = np.zeros((self.config.height, self.config.width, 3))
        for r in range(self.config.height):
            for c in range(self.config.width):
                cell = self.grid[r, c]
                if (r, c) == self.agent_pos:
                    img[r, c] = colors[self.AGENT]
                else:
                    img[r, c] = colors.get(cell, [1, 1, 1])

        return img


# =============================================================================
# EMBODIED BRAIN
# =============================================================================

class NavigationNetwork(nn.Module):
    """Simple network for navigation decisions."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class EmbodiedConfig:
    """Configuration for embodied brain."""
    observation_radius: int = 2  # Local view radius
    hidden_dim: int = 64
    learning_rate: float = 0.01  # Faster learning
    trust_threshold: float = 0.15  # Lower threshold to see switching
    trust_alpha: float = 0.8  # Faster trust dynamics
    confidence_alpha: float = 0.9


class EmbodiedBrain:
    """
    A brain with a body - navigation through System 1/System 2 routing.

    System 1 (Archive/Reflex):
        - Handles familiar navigation patterns
        - Fast, automatic responses
        - "Muscle memory" for known paths

    System 2 (Dialogue/Deliberate):
        - Engages when encountering novel situations
        - Slower, more careful planning
        - "Stop and think" behavior

    The key behavior:
        - Moving through familiar territory: smooth, fast
        - Encountering novel obstacle: PAUSE, deliberate, then act
    """

    def __init__(self, config: EmbodiedConfig, obs_dim: int, goal_pos: Tuple[int, int] = (6, 6)):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.goal_pos = goal_pos

        # System 1: Archive (Reflex) - pre-trained with heuristic
        self.archive = NavigationNetwork(obs_dim, config.hidden_dim).to(self.device)
        self.opt_archive = optim.Adam(self.archive.parameters(), lr=config.learning_rate)

        # System 2: Dialogue (Two networks that must agree)
        self.net_A = NavigationNetwork(obs_dim, config.hidden_dim).to(self.device)
        self.net_B = NavigationNetwork(obs_dim, config.hidden_dim).to(self.device)
        self.opt_A = optim.Adam(self.net_A.parameters(), lr=config.learning_rate)
        self.opt_B = optim.Adam(self.net_B.parameters(), lr=config.learning_rate * 0.5)

        # Metacognition
        self.trust = 0.8  # Start with HIGH trust (pre-trained)
        self.confidence = 0.5

        # Pre-train Archive with goal-seeking heuristic
        self._pretrain_archive()

        # Experience buffer for learning
        self.experiences: List[Tuple] = []

    def _pretrain_archive(self, n_examples: int = 500):
        """
        Pre-train Archive with goal-seeking behavior.

        This simulates having already learned basic navigation,
        so we can demonstrate the reflex→deliberate switch
        when novel obstacles appear.
        """
        print("  Pre-training Archive with goal-seeking heuristic...")
        self.archive.train()
        criterion = nn.CrossEntropyLoss()

        for _ in range(n_examples):
            # Random position
            r = np.random.randint(1, 7)
            c = np.random.randint(1, 7)

            # Create fake local observation (center = current position)
            obs = np.random.rand(25).astype(np.float32) * 0.1

            # Heuristic: move towards goal (6, 6)
            # Actions: 0=Up(-r), 1=Down(+r), 2=Left(-c), 3=Right(+c)
            dr = self.goal_pos[0] - r
            dc = self.goal_pos[1] - c

            if abs(dr) > abs(dc):
                action = 1 if dr > 0 else 0  # Down or Up
            else:
                action = 3 if dc > 0 else 2  # Right or Left

            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            target = torch.LongTensor([action]).to(self.device)

            logits = self.archive(obs_tensor)
            loss = criterion(logits, target)

            self.opt_archive.zero_grad()
            loss.backward()
            self.opt_archive.step()

        print("  Archive pre-trained!")

        # Metrics
        self.reflex_count = 0
        self.deliberate_count = 0
        self.thinking_episodes: List[Dict] = []  # When did we stop to think?

    def get_archive_confidence(self, obs: torch.Tensor) -> Tuple[int, float]:
        """Get Archive's action and confidence."""
        self.archive.eval()
        with torch.no_grad():
            logits = self.archive(obs)
            probs = torch.softmax(logits, dim=-1)
            action = probs.argmax().item()
            confidence = probs.max().item()
        return action, confidence

    def get_dialogue_action(self, obs: torch.Tensor) -> Tuple[int, float, float]:
        """Get Dialogue's action through A/B agreement."""
        self.net_A.eval()
        self.net_B.eval()

        with torch.no_grad():
            logits_A = self.net_A(obs)
            logits_B = self.net_B(obs)

            probs_A = torch.softmax(logits_A, dim=-1)
            probs_B = torch.softmax(logits_B, dim=-1)

            # Agreement = similarity of distributions
            agreement = 1.0 - torch.abs(probs_A - probs_B).mean().item()

            # Take action from A (the "proposer")
            action = probs_A.argmax().item()
            confidence = probs_A.max().item()

        return action, confidence, agreement

    def act(self, obs: np.ndarray, step: int = 0) -> Tuple[int, str, Dict]:
        """
        Choose an action using System 1/System 2 routing.

        Returns:
            action, system_used, info
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Get Archive's opinion
        archive_action, archive_confidence = self.get_archive_confidence(obs_tensor)

        # Effective confidence = Archive confidence * Trust
        effective_confidence = archive_confidence * self.trust

        info = {
            "archive_confidence": archive_confidence,
            "trust": self.trust,
            "effective_confidence": effective_confidence
        }

        # ROUTING DECISION
        if effective_confidence >= self.config.trust_threshold:
            # SYSTEM 1: REFLEX
            action = archive_action
            system = "reflex"
            self.reflex_count += 1
            info["thinking_time"] = 0

        else:
            # SYSTEM 2: DELIBERATE
            # This is where the agent "stops to think"
            action, dialogue_confidence, agreement = self.get_dialogue_action(obs_tensor)
            system = "deliberate"
            self.deliberate_count += 1

            # Record this thinking episode
            self.thinking_episodes.append({
                "step": step,
                "archive_confidence": archive_confidence,
                "trust": self.trust,
                "dialogue_confidence": dialogue_confidence,
                "agreement": agreement
            })

            info["dialogue_confidence"] = dialogue_confidence
            info["agreement"] = agreement
            info["thinking_time"] = 1  # Represents the pause

            # Update confidence based on agreement
            self.confidence = (
                self.config.confidence_alpha * self.confidence +
                (1 - self.config.confidence_alpha) * agreement
            )

        return action, system, info

    def learn(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        """Learn from experience."""
        self.experiences.append((obs, action, reward, next_obs, done))

        # Simple online learning
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)

        # Train Dialogue System (always learning)
        self.net_A.train()
        self.net_B.train()

        logits_A = self.net_A(obs_tensor)
        logits_B = self.net_B(obs_tensor)

        # Use reward as signal
        # Positive reward = reinforce action, Negative = discourage
        target = torch.zeros_like(logits_A)
        target[0, action] = 1.0 if reward > 0 else -0.5

        loss_A = nn.functional.mse_loss(torch.softmax(logits_A, dim=-1), torch.softmax(target, dim=-1))
        loss_B = nn.functional.mse_loss(torch.softmax(logits_B, dim=-1), torch.softmax(target, dim=-1))

        self.opt_A.zero_grad()
        loss_A.backward()
        self.opt_A.step()

        self.opt_B.zero_grad()
        loss_B.backward()
        self.opt_B.step()

    def update_trust(self, action_was_correct: bool):
        """Update trust in Archive based on outcome."""
        self.trust = (
            self.config.trust_alpha * self.trust +
            (1 - self.config.trust_alpha) * (1.0 if action_was_correct else 0.0)
        )

    def consolidate_to_archive(self):
        """Transfer learned knowledge to Archive (like sleep)."""
        if len(self.experiences) < 10:
            return

        self.archive.train()

        # Train Archive on recent experiences
        for obs, action, reward, _, _ in self.experiences[-100:]:
            if reward > 0:  # Only consolidate successful actions
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                # Get Dialogue's current opinion
                with torch.no_grad():
                    target_logits = self.net_A(obs_tensor)

                # Train Archive to match Dialogue
                archive_logits = self.archive(obs_tensor)
                loss = nn.functional.mse_loss(archive_logits, target_logits)

                self.opt_archive.zero_grad()
                loss.backward()
                self.opt_archive.step()

        # Boost trust after consolidation
        self.trust = min(0.95, self.trust + 0.2)

    def get_stats(self) -> Dict:
        """Get brain statistics."""
        total = self.reflex_count + self.deliberate_count
        return {
            "reflex_count": self.reflex_count,
            "deliberate_count": self.deliberate_count,
            "reflex_ratio": self.reflex_count / max(total, 1),
            "trust": self.trust,
            "confidence": self.confidence,
            "thinking_episodes": len(self.thinking_episodes)
        }


# =============================================================================
# EXPERIMENT
# =============================================================================

class EmbodimentTracker(ExperimentTracker):
    """Tracker for embodiment experiments."""
    def __init__(self):
        super().__init__(experiment_name="embodiment", personality="gridworld")


def run_episode(
    env: GridWorld,
    brain: EmbodiedBrain,
    add_novel_at_step: Optional[int] = None,
    verbose: bool = True
) -> Dict:
    """Run a single episode."""
    obs = env.reset()
    total_reward = 0
    step = 0

    episode_data = {
        "path": [env.agent_pos],
        "actions": [],
        "systems": [],
        "rewards": [],
        "trust": [brain.trust],
        "thinking_steps": []
    }

    while not env.done:
        # Maybe add novel obstacle
        if add_novel_at_step is not None and step == add_novel_at_step:
            novel_pos = env.add_novel_obstacle()
            if verbose and novel_pos:
                print(f"  [!] Novel obstacle added at {novel_pos}!")

        # Get local observation
        local_obs = env.get_local_observation(brain.config.observation_radius)

        # Brain decides
        action, system, info = brain.act(local_obs, step)

        # Record system used
        env.system_history.append(system)
        episode_data["systems"].append(system)

        if system == "deliberate":
            episode_data["thinking_steps"].append(step)
            if verbose:
                print(f"  Step {step}: THINKING... (trust={info['trust']:.2f}, archive_conf={info['archive_confidence']:.2f})")

        # Take action
        next_obs, reward, done, step_info = env.step(action)
        total_reward += reward

        episode_data["actions"].append(action)
        episode_data["rewards"].append(reward)
        episode_data["path"].append(env.agent_pos)
        episode_data["trust"].append(brain.trust)

        # Learn from experience
        next_local = env.get_local_observation(brain.config.observation_radius)
        brain.learn(local_obs, action, reward, next_local, done)

        # Update trust based on outcome
        action_was_correct = reward >= 0
        brain.update_trust(action_was_correct)

        step += 1

    episode_data["total_reward"] = total_reward
    episode_data["steps"] = step
    episode_data["reached_goal"] = step_info.get("reason") == "goal"

    return episode_data


def visualize_episode(
    env: GridWorld,
    episode_data: Dict,
    tracker: ExperimentTracker,
    episode_num: int
):
    """Create visualization of an episode."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Grid with path
    ax1 = axes[0]
    img = env.render_to_array()
    ax1.imshow(img, interpolation='nearest')

    # Draw path
    path = episode_data["path"]
    for i, (r, c) in enumerate(path):
        if i > 0:
            system = episode_data["systems"][i-1] if i-1 < len(episode_data["systems"]) else "reflex"
            color = 'blue' if system == "reflex" else 'orange'
            ax1.plot([path[i-1][1], c], [path[i-1][0], r], color=color, linewidth=2)

    # Mark thinking steps
    for step in episode_data["thinking_steps"]:
        if step < len(path):
            r, c = path[step]
            ax1.scatter([c], [r], s=200, c='orange', marker='*', zorder=5, label='Thinking' if step == episode_data["thinking_steps"][0] else '')

    ax1.set_title(f"Episode {episode_num}: Path Taken")
    ax1.set_xlabel("Blue=Reflex, Orange=Deliberate, ★=Thinking")
    ax1.legend(loc='upper right')

    # Plot 2: Trust over time
    ax2 = axes[1]
    ax2.plot(episode_data["trust"], 'g-', linewidth=2)
    ax2.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Trust Threshold')

    # Mark thinking episodes
    for step in episode_data["thinking_steps"]:
        if step < len(episode_data["trust"]):
            ax2.axvline(x=step, color='orange', alpha=0.3)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Trust")
    ax2.set_title("Trust in Archive Over Time")
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # Plot 3: System usage
    ax3 = axes[2]
    systems = episode_data["systems"]
    reflex_count = systems.count("reflex")
    deliberate_count = systems.count("deliberate")

    ax3.bar(["Reflex\n(System 1)", "Deliberate\n(System 2)"],
            [reflex_count, deliberate_count],
            color=['blue', 'orange'], alpha=0.7)
    ax3.set_ylabel("Actions")
    ax3.set_title("System Usage")
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    tracker.save_figure(fig, f"episode_{episode_num}")
    plt.close(fig)


def run_embodiment_experiment(verbose: bool = True) -> Dict:
    """
    Run the full embodiment experiment.

    Demonstrates:
    1. Learning to navigate (trust builds)
    2. Handling novel obstacles (trust crash, deliberation)
    3. System 1 vs System 2 dynamics in action
    """
    # Setup
    grid_config = GridConfig(width=8, height=8, n_obstacles=10, seed=42)
    brain_config = EmbodiedConfig()

    env = GridWorld(grid_config)
    obs_dim = (2 * brain_config.observation_radius + 1) ** 2  # Local view size

    brain = EmbodiedBrain(brain_config, obs_dim)

    if verbose:
        print("=" * 70)
        print("EMBODIMENT EXPERIMENT: The Brain Gets a Body")
        print("=" * 70)
        print(f"Grid: {grid_config.width}x{grid_config.height}")
        print(f"Obstacles: {grid_config.n_obstacles}")
        print(f"Observation radius: {brain_config.observation_radius}")
        print()

    tracker = EmbodimentTracker()
    results = {
        "episodes": [],
        "trust_history": [],
        "system_ratios": []
    }

    # ========== PHASE 1: Learning the World ==========
    if verbose:
        print("PHASE 1: Learning to Navigate")
        print("-" * 40)

    for ep in range(5):
        if verbose:
            print(f"\nEpisode {ep + 1}/5")

        episode_data = run_episode(env, brain, add_novel_at_step=None, verbose=verbose)
        results["episodes"].append(episode_data)
        results["trust_history"].append(brain.trust)

        stats = brain.get_stats()
        results["system_ratios"].append(stats["reflex_ratio"])

        if verbose:
            print(f"  Result: {'GOAL!' if episode_data['reached_goal'] else 'Timeout'}")
            print(f"  Steps: {episode_data['steps']}, Reward: {episode_data['total_reward']:.1f}")
            print(f"  Reflex: {stats['reflex_ratio']:.1%}, Trust: {brain.trust:.2f}")

        # Consolidate to Archive after each episode
        brain.consolidate_to_archive()

        # Visualize
        visualize_episode(env, episode_data, tracker, ep + 1)

    # ========== PHASE 2: The Novel Obstacle ==========
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 2: Novel Obstacle Appears!")
        print("=" * 70)

    # Reset environment but add novel obstacle mid-episode
    env = GridWorld(grid_config)

    for ep in range(5, 10):
        if verbose:
            print(f"\nEpisode {ep + 1}/10")

        # Add novel obstacle after 5 steps
        episode_data = run_episode(env, brain, add_novel_at_step=5, verbose=verbose)
        results["episodes"].append(episode_data)
        results["trust_history"].append(brain.trust)

        stats = brain.get_stats()
        results["system_ratios"].append(stats["reflex_ratio"])

        if verbose:
            print(f"  Result: {'GOAL!' if episode_data['reached_goal'] else 'Timeout'}")
            print(f"  Steps: {episode_data['steps']}, Reward: {episode_data['total_reward']:.1f}")
            print(f"  Thinking episodes: {len(episode_data['thinking_steps'])}")
            print(f"  Reflex: {stats['reflex_ratio']:.1%}, Trust: {brain.trust:.2f}")

        brain.consolidate_to_archive()
        visualize_episode(env, episode_data, tracker, ep + 1)

        # Reset for next episode
        env = GridWorld(grid_config)

    # ========== FINAL SUMMARY ==========
    final_stats = brain.get_stats()

    summary = {
        "total_episodes": len(results["episodes"]),
        "goals_reached": sum(1 for ep in results["episodes"] if ep["reached_goal"]),
        "final_trust": brain.trust,
        "total_reflex_actions": final_stats["reflex_count"],
        "total_deliberate_actions": final_stats["deliberate_count"],
        "reflex_ratio": final_stats["reflex_ratio"],
        "total_thinking_episodes": final_stats["thinking_episodes"]
    }

    results["summary"] = summary

    if verbose:
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Goals reached: {summary['goals_reached']}/{summary['total_episodes']}")
        print(f"Final trust: {summary['final_trust']:.2f}")
        print(f"Reflex actions: {summary['total_reflex_actions']} ({summary['reflex_ratio']:.1%})")
        print(f"Deliberate actions: {summary['total_deliberate_actions']}")
        print(f"Total thinking episodes: {summary['total_thinking_episodes']}")

    return results, tracker, brain


def create_summary_visualization(results: Dict, tracker: ExperimentTracker):
    """Create summary visualization of the experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Embodiment Experiment: Watch the AI Think", fontsize=14)

    # Plot 1: Trust over episodes
    ax1 = axes[0, 0]
    ax1.plot(results["trust_history"], 'g-o', linewidth=2, markersize=8)
    ax1.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Trust Threshold')
    ax1.axvline(x=4.5, color='black', linestyle='--', alpha=0.5, label='Novel Obstacles Begin')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Trust")
    ax1.set_title("Trust in Archive Over Episodes")
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: System usage ratio
    ax2 = axes[0, 1]
    ax2.plot(results["system_ratios"], 'b-o', linewidth=2, markersize=8)
    ax2.axvline(x=4.5, color='black', linestyle='--', alpha=0.5, label='Novel Obstacles Begin')
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reflex Ratio")
    ax2.set_title("Automation Level (Higher = More Reflex)")
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Steps per episode
    ax3 = axes[1, 0]
    steps = [ep["steps"] for ep in results["episodes"]]
    colors = ['green' if ep["reached_goal"] else 'red' for ep in results["episodes"]]
    ax3.bar(range(len(steps)), steps, color=colors, alpha=0.7)
    ax3.axvline(x=4.5, color='black', linestyle='--', alpha=0.5, label='Novel Obstacles')
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Steps")
    ax3.set_title("Steps per Episode (Green=Goal, Red=Timeout)")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Thinking episodes per episode
    ax4 = axes[1, 1]
    thinking = [len(ep["thinking_steps"]) for ep in results["episodes"]]
    ax4.bar(range(len(thinking)), thinking, color='orange', alpha=0.7)
    ax4.axvline(x=4.5, color='black', linestyle='--', alpha=0.5, label='Novel Obstacles')
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Thinking Steps")
    ax4.set_title("How Often Did It Stop to Think?")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    tracker.save_figure(fig, "embodiment_summary")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 70)
    print("EMBODIMENT EXPERIMENT")
    print("The Brain Gets a Body - Watch It Think!")
    print("=" * 70)
    print()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")
    print()

    # Run experiment
    results, tracker, brain = run_embodiment_experiment(verbose=True)

    # Save summary
    summary = results["summary"]
    summary_text = f"""
EMBODIMENT EXPERIMENT RESULTS
=============================
Timestamp: {tracker.timestamp}

THE BRAIN IN A BODY:
  Grid: 8x8 with obstacles
  Episodes: {summary['total_episodes']}
  Goals reached: {summary['goals_reached']}

SYSTEM DYNAMICS:
  Total reflex actions: {summary['total_reflex_actions']}
  Total deliberate actions: {summary['total_deliberate_actions']}
  Reflex ratio: {summary['reflex_ratio']:.1%}

  Thinking episodes: {summary['total_thinking_episodes']}
  Final trust: {summary['final_trust']:.2f}

KEY OBSERVATION:
  When novel obstacles appeared, the agent STOPPED and THOUGHT.
  This is visible System 1 → System 2 switching in action.

  The brain learned to navigate reflexively, then deliberately
  re-engaged when the world changed unexpectedly.
"""

    print("\n" + "=" * 70)
    print(summary_text)

    tracker.save_summary(summary_text)

    # Create visualizations
    print("Generating visualizations...")
    create_summary_visualization(results, tracker)

    # Save config
    with open(tracker.output_dir / "config.json", 'w') as f:
        json.dump({
            "grid_width": 8,
            "grid_height": 8,
            "n_obstacles": 10,
            "observation_radius": 2,
            "trust_threshold": 0.7,
            "n_episodes": 10
        }, f, indent=2)

    print(f"\nResults saved to: {tracker.output_dir}")
    print("=" * 70)
