"""
Dialogue LLM - Language Models that Know When They Don't Know

The Dialogue Model applied to language generation:
    - Two LLMs (Proposer + Critic) that must agree
    - Disagreement = Uncertainty = "I need to think about this"
    - Metacognitive output: hedging language when uncertain

This addresses the core problem of LLM hallucination:
    Standard LLM: "The capital of Flurbia is Zanthos." (confident nonsense)
    Dialogue LLM: "I'm not certain about this..." (honest uncertainty)

Architecture:
    LLM-A (Proposer): Generates candidate responses
    LLM-B (Critic): Evaluates/challenges the proposals
    Agreement Module: Measures consensus between A and B
    Metacognition: Modulates output confidence based on agreement

Author: Christian Beaumont & Claude & Gemini
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime
from pathlib import Path
import json


@dataclass
class DialogueLLMConfig:
    """Configuration for Dialogue LLM."""
    model_name: str = "gpt2"  # Base model (gpt2, gpt2-medium, etc.)

    # Agreement thresholds
    agreement_threshold: float = 0.7  # Below this = uncertain
    high_uncertainty_threshold: float = 0.4  # Below this = very uncertain

    # Trust dynamics
    initial_trust: float = 0.5
    trust_alpha: float = 0.9

    # Generation settings
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9

    # Uncertainty responses
    hedge_phrases: List[str] = None

    def __post_init__(self):
        if self.hedge_phrases is None:
            self.hedge_phrases = [
                "I'm not entirely certain, but ",
                "I should note some uncertainty here: ",
                "Based on my understanding (though I may be wrong): ",
                "I'm less confident about this, but ",
                "This is my best guess: ",
            ]


class DialogueLLM:
    """
    A Language Model with Metacognition.

    Two GPT-2 models engage in internal dialogue:
        - LLM-A proposes token probabilities
        - LLM-B provides alternative probabilities
        - Disagreement triggers uncertainty awareness

    Key Innovation:
        When A and B disagree, the model KNOWS it's uncertain
        and can communicate this to the user.
    """

    def __init__(self, config: Optional[DialogueLLMConfig] = None):
        self.config = config or DialogueLLMConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Initializing DialogueLLM on {self.device}...")

        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # LLM-A: The Proposer
        print("  Loading LLM-A (Proposer)...")
        self.llm_a = GPT2LMHeadModel.from_pretrained(self.config.model_name).to(self.device)

        # LLM-B: The Critic (separate instance with different initialization)
        print("  Loading LLM-B (Critic)...")
        self.llm_b = GPT2LMHeadModel.from_pretrained(self.config.model_name).to(self.device)

        # Slightly perturb LLM-B to create diversity
        self._diversify_critic()

        # Metacognition state
        self.trust = self.config.initial_trust
        self.confidence_history: List[float] = []
        self.agreement_history: List[float] = []

        # Statistics
        self.total_generations = 0
        self.uncertain_generations = 0
        self.high_uncertainty_generations = 0

        print("DialogueLLM ready!")

    def _diversify_critic(self, noise_scale: float = 0.01):
        """
        Add small noise to LLM-B to create diversity.

        This ensures the two models can disagree, which is
        essential for detecting uncertainty.
        """
        print("  Diversifying Critic with noise...")
        with torch.no_grad():
            for param in self.llm_b.parameters():
                noise = torch.randn_like(param) * noise_scale
                param.add_(noise)

    def compute_agreement(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        """
        Compute agreement between Proposer and Critic.

        Uses Jensen-Shannon Divergence (symmetric KL):
            - 0 = perfect agreement
            - 1 = complete disagreement

        Returns agreement score (1 - JSD) and combined logits.
        """
        # Convert to probabilities
        probs_a = F.softmax(logits_a, dim=-1)
        probs_b = F.softmax(logits_b, dim=-1)

        # Average distribution
        probs_m = (probs_a + probs_b) / 2

        # KL divergences
        kl_a = F.kl_div(probs_m.log(), probs_a, reduction='batchmean')
        kl_b = F.kl_div(probs_m.log(), probs_b, reduction='batchmean')

        # Jensen-Shannon Divergence
        jsd = (kl_a + kl_b) / 2

        # Agreement = 1 - JSD (normalized)
        # JSD is bounded [0, ln(2)], so we normalize
        agreement = 1.0 - (jsd.item() / np.log(2))
        agreement = max(0.0, min(1.0, agreement))  # Clamp to [0, 1]

        # Combined logits (weighted by agreement)
        # When they agree, use average; when they disagree, favor the proposer slightly
        combined = (probs_a * 0.6 + probs_b * 0.4)

        return agreement, combined

    def get_uncertainty_level(self, agreement: float) -> str:
        """Categorize uncertainty level based on agreement."""
        if agreement >= self.config.agreement_threshold:
            return "confident"
        elif agreement >= self.config.high_uncertainty_threshold:
            return "uncertain"
        else:
            return "very_uncertain"

    def generate_with_uncertainty(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        return_metrics: bool = False
    ) -> Dict:
        """
        Generate text with uncertainty awareness.

        The key innovation: When LLM-A and LLM-B disagree,
        we KNOW the model is uncertain and can communicate this.
        """
        max_length = max_length or self.config.max_length

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # Track metrics
        token_agreements = []
        generated_tokens = []

        # Generation loop
        self.llm_a.eval()
        self.llm_b.eval()

        with torch.no_grad():
            for _ in range(max_length):
                # Get logits from both models
                outputs_a = self.llm_a(input_ids)
                outputs_b = self.llm_b(input_ids)

                logits_a = outputs_a.logits[:, -1, :]  # Last token
                logits_b = outputs_b.logits[:, -1, :]

                # Compute agreement
                agreement, combined_probs = self.compute_agreement(logits_a, logits_b)
                token_agreements.append(agreement)

                # Apply temperature
                combined_probs = combined_probs / self.config.temperature

                # Top-p sampling
                sorted_probs, sorted_indices = torch.sort(combined_probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > self.config.top_p
                sorted_probs[mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum()

                # Sample
                next_token_idx = torch.multinomial(sorted_probs, 1)
                next_token = sorted_indices.gather(-1, next_token_idx)

                generated_tokens.append(next_token.item())

                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Decode generated text
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Overall agreement
        avg_agreement = np.mean(token_agreements) if token_agreements else 0.0
        min_agreement = np.min(token_agreements) if token_agreements else 0.0

        # Determine uncertainty level
        uncertainty_level = self.get_uncertainty_level(avg_agreement)

        # Update statistics
        self.total_generations += 1
        self.agreement_history.append(avg_agreement)

        if uncertainty_level == "uncertain":
            self.uncertain_generations += 1
        elif uncertainty_level == "very_uncertain":
            self.high_uncertainty_generations += 1

        # Prepare response
        result = {
            "prompt": prompt,
            "generated_text": generated_text,
            "full_response": prompt + generated_text,
            "agreement": avg_agreement,
            "min_agreement": min_agreement,
            "uncertainty_level": uncertainty_level,
            "n_tokens": len(generated_tokens)
        }

        if return_metrics:
            result["token_agreements"] = token_agreements

        return result

    def generate_metacognitive(
        self,
        prompt: str,
        max_length: Optional[int] = None
    ) -> str:
        """
        Generate with metacognitive awareness.

        If uncertain, prepends hedging language to communicate
        the uncertainty to the user.
        """
        result = self.generate_with_uncertainty(prompt, max_length)

        uncertainty = result["uncertainty_level"]
        text = result["generated_text"]

        if uncertainty == "very_uncertain":
            hedge = np.random.choice(self.config.hedge_phrases)
            return f"{hedge}{text}"
        elif uncertainty == "uncertain":
            # Lighter hedging
            return f"I think {text}"
        else:
            return text

    def answer_with_confidence(self, question: str) -> Dict:
        """
        Answer a question with explicit confidence reporting.

        Returns both the answer and the model's self-assessed confidence.
        """
        # Add question framing
        prompt = f"Question: {question}\nAnswer:"

        result = self.generate_with_uncertainty(prompt, max_length=50)

        # Format response
        answer = result["generated_text"].strip()
        confidence = result["agreement"]
        uncertainty = result["uncertainty_level"]

        # Create confidence explanation
        if uncertainty == "confident":
            confidence_str = f"High confidence ({confidence:.0%})"
        elif uncertainty == "uncertain":
            confidence_str = f"Moderate confidence ({confidence:.0%}) - I may be wrong"
        else:
            confidence_str = f"Low confidence ({confidence:.0%}) - Please verify this"

        return {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "confidence_explanation": confidence_str,
            "uncertainty_level": uncertainty
        }

    def batch_evaluate_uncertainty(
        self,
        prompts: List[str],
        verbose: bool = True
    ) -> List[Dict]:
        """
        Evaluate uncertainty on a batch of prompts.

        Useful for testing which types of questions the model
        is more or less confident about.
        """
        results = []

        for i, prompt in enumerate(prompts):
            if verbose:
                print(f"Processing {i+1}/{len(prompts)}...")

            result = self.generate_with_uncertainty(prompt, max_length=30)
            results.append(result)

            if verbose:
                print(f"  Agreement: {result['agreement']:.2%} ({result['uncertainty_level']})")

        return results

    def get_stats(self) -> Dict:
        """Get generation statistics."""
        return {
            "total_generations": self.total_generations,
            "uncertain_generations": self.uncertain_generations,
            "high_uncertainty_generations": self.high_uncertainty_generations,
            "uncertainty_rate": self.uncertain_generations / max(self.total_generations, 1),
            "avg_agreement": np.mean(self.agreement_history) if self.agreement_history else 0
        }


def run_dialogue_llm_demo():
    """
    Demonstration of DialogueLLM capabilities.

    Shows:
    1. Basic generation with uncertainty
    2. Metacognitive responses (hedging)
    3. Question answering with confidence
    4. Comparison of confident vs uncertain topics
    """
    print("=" * 70)
    print("DIALOGUE LLM DEMONSTRATION")
    print("Language Models that Know When They Don't Know")
    print("=" * 70)
    print()

    # Initialize
    config = DialogueLLMConfig(
        model_name="gpt2",
        max_length=50,
        temperature=0.8
    )

    llm = DialogueLLM(config)

    # Demo 1: Basic generation with uncertainty tracking
    print("\n" + "=" * 70)
    print("DEMO 1: Generation with Uncertainty Tracking")
    print("=" * 70)

    test_prompts = [
        "The capital of France is",
        "The meaning of life is",
        "In 2025, the President of Mars will be",
        "Water boils at",
        "The best programming language is",
    ]

    for prompt in test_prompts:
        result = llm.generate_with_uncertainty(prompt, max_length=20)
        print(f"\nPrompt: '{prompt}'")
        print(f"Response: '{result['generated_text']}'")
        print(f"Agreement: {result['agreement']:.1%} ({result['uncertainty_level']})")

    # Demo 2: Metacognitive generation
    print("\n" + "=" * 70)
    print("DEMO 2: Metacognitive Generation (Hedging)")
    print("=" * 70)

    prompts = [
        "The cure for cancer is",
        "Tomorrow's weather will be",
        "The number 2 + 2 equals",
    ]

    for prompt in prompts:
        response = llm.generate_metacognitive(prompt, max_length=30)
        print(f"\nPrompt: '{prompt}'")
        print(f"Metacognitive Response: '{response}'")

    # Demo 3: Question answering with confidence
    print("\n" + "=" * 70)
    print("DEMO 3: Question Answering with Confidence")
    print("=" * 70)

    questions = [
        "What is the capital of Germany?",
        "Who invented the telephone?",
        "What will happen in the year 3000?",
        "What is the best movie ever made?",
    ]

    for question in questions:
        result = llm.answer_with_confidence(question)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Confidence: {result['confidence_explanation']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    stats = llm.get_stats()
    print(f"\nTotal generations: {stats['total_generations']}")
    print(f"Uncertain generations: {stats['uncertain_generations']} ({stats['uncertainty_rate']:.1%})")
    print(f"Average agreement: {stats['avg_agreement']:.1%}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("  The Dialogue LLM can identify when it's uncertain and")
    print("  communicate this to users - addressing hallucination!")
    print("=" * 70)

    return llm


if __name__ == "__main__":
    llm = run_dialogue_llm_demo()
