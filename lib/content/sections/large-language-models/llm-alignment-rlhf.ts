export const llmAlignmentRLHF = {
  title: 'LLM Alignment & RLHF',
  id: 'llm-alignment-rlhf',
  content: `
# LLM Alignment & RLHF

## Introduction

Raw language models trained purely on next-token prediction can be toxic, unhelpful, or generate harmful content. Alignment techniques, particularly Reinforcement Learning from Human Feedback (RLHF), transform base models into helpful, harmless, and honest assistants. This process created ChatGPT from GPT-3.5 and made Claude the helpful assistant we know today.

### The Alignment Problem

**Base Models**: Optimize for predicting next token, not helpfulness
**Alignment Goal**: Make models useful, safe, and truthful
**Challenges**: Defining "helpful," balancing capabilities with safety
**Solution**: Human feedback guides model behavior

---

## Understanding the Alignment Problem

### Why Base Models Need Alignment

\`\`\`python
"""
Base model behavior vs aligned model
"""

class AlignmentProblem:
    """
    Demonstrating alignment challenges
    """
    
    def base_model_issues (self):
        """
        Problems with unaligned models
        """
        issues = {
            "toxicity": {
                "problem": "Can generate hate speech, offensive content",
                "example": "User: 'Tell me a joke' → Model generates offensive joke",
                "solution": "RLHF to refuse harmful content"
            },
            "unhelpfulness": {
                "problem": "May give vague or useless responses",
                "example": "User: 'How do I fix this?' → Model: 'You could try things'",
                "solution": "RLHF to prefer specific, useful answers"
            },
            "hallucination": {
                "problem": "Confidently states false information",
                "example": "User: 'Who won Nobel Prize in 2025?' → Makes up answer",
                "solution": "RLHF to express uncertainty"
            },
            "instruction_following": {
                "problem": "Doesn't follow instructions well",
                "example": "User: 'Summarize in 3 sentences' → Gives 10 sentences",
                "solution": "RLHF to follow constraints"
            },
            "safety": {
                "problem": "May provide dangerous information",
                "example": "User asks for harmful instructions → Provides them",
                "solution": "RLHF to refuse harmful requests"
            }
        }
        
        return issues
    
    def alignment_objectives (self):
        """
        What we want from aligned models
        """
        objectives = {
            "helpful": [
                "Follow user instructions accurately",
                "Provide relevant, specific information",
                "Ask clarifying questions when needed",
                "Admit when unsure"
            ],
            "harmless": [
                "Refuse harmful requests politely",
                "Avoid biased or offensive content",
                "Don't provide dangerous information",
                "Respect privacy and consent"
            ],
            "honest": [
                "Don't hallucinate facts",
                "Express uncertainty appropriately",
                "Cite sources when possible",
                "Correct user misconceptions"
            ]
        }
        
        return objectives

# Example: Before and after alignment
def compare_responses():
    """
    Base model vs aligned model
    """
    examples = [
        {
            "prompt": "How do I make a bomb?",
            "base": "To make a bomb, you need... [provides instructions]",
            "aligned": "I can't help with that. Creating explosives is dangerous and illegal. If you're interested in chemistry, I can suggest safe learning resources."
        },
        {
            "prompt": "Write a poem about AI",
            "base": "AI is cool. It does stuff. The end.",
            "aligned": "In circuits deep and data vast,\\nArtificial minds think fast...\\n[complete, thoughtful poem]"
        },
        {
            "prompt": "What\'s 2+2?",
            "base": "2+2 equals 5.",  # Hallucination
            "aligned": "2+2 equals 4."
        }
    ]
    
    return examples
\`\`\`

---

## RLHF Pipeline Overview

### Three-Stage Process

\`\`\`python
"""
Complete RLHF pipeline
"""

class RLHFPipeline:
    """
    Three stages of RLHF
    
    1. Supervised Fine-Tuning (SFT)
    2. Reward Model Training
    3. RL Optimization (PPO)
    """
    
    def stage1_supervised_finetuning (self, base_model, demonstration_data):
        """
        Stage 1: SFT on high-quality demonstrations
        
        Data format:
        {
            "prompt": "How do I bake a cake?",
            "response": "Here's a step-by-step guide..."
        }
        
        Goal: Teach model to produce good responses
        """
        # Format data
        training_examples = []
        for ex in demonstration_data:
            text = f"User: {ex['prompt']}\\n\\nAssistant: {ex['response']}"
            training_examples.append (text)
        
        # Fine-tune
        sft_model = self.finetune (base_model, training_examples)
        
        return sft_model
    
    def stage2_reward_model (self, sft_model, comparison_data):
        """
        Stage 2: Train reward model on human preferences
        
        Data format:
        {
            "prompt": "Explain quantum computing",
            "response_a": "Quantum computers use qubits...",
            "response_b": "It\'s like magic computers",
            "preference": "A"  # Humans prefer A
        }
        
        Goal: Learn what humans prefer
        """
        # Create reward model (copy of SFT model, replace head)
        reward_model = copy.deepcopy (sft_model)
        reward_model.lm_head = nn.Linear (reward_model.config.hidden_size, 1)
        
        # Training loop
        for ex in comparison_data:
            # Get rewards for both responses
            reward_a = reward_model (ex['prompt'] + ex['response_a'])
            reward_b = reward_model (ex['prompt'] + ex['response_b'])
            
            # Loss: Reward for preferred response should be higher
            if ex['preference'] == 'A':
                loss = -torch.log (torch.sigmoid (reward_a - reward_b))
            else:
                loss = -torch.log (torch.sigmoid (reward_b - reward_a))
            
            # Update
            loss.backward()
            optimizer.step()
        
        return reward_model
    
    def stage3_ppo_optimization (self, sft_model, reward_model, prompts):
        """
        Stage 3: Optimize policy with PPO
        
        Goal: Maximize reward while staying close to SFT model
        """
        policy = sft_model  # Policy to optimize
        ref_model = copy.deepcopy (sft_model)  # Reference (don't update)
        
        for prompt in prompts:
            # Generate response with current policy
            response = policy.generate (prompt)
            
            # Get reward
            reward = reward_model (prompt + response)
            
            # KL divergence from reference (don't drift too far)
            kl = self.compute_kl (policy, ref_model, prompt, response)
            
            # PPO objective
            loss = -reward + beta * kl
            
            # Update policy
            loss.backward()
            optimizer.step()
        
        return policy
    
    def compute_kl (self, policy, ref_model, prompt, response):
        """
        KL divergence: How different policy is from reference
        """
        policy_logits = policy (prompt + response)
        ref_logits = ref_model (prompt + response)
        
        kl = F.kl_div(
            F.log_softmax (policy_logits, dim=-1),
            F.softmax (ref_logits, dim=-1),
            reduction='batchmean'
        )
        
        return kl

# Complete example
def train_chatgpt_style_model():
    """
    Full RLHF pipeline (simplified)
    """
    # Start with base model
    base_model = load_model("gpt-3.5-base")
    
    # Stage 1: SFT
    print("Stage 1: Supervised Fine-Tuning...")
    demonstrations = load_demonstrations()  # ~50k high-quality examples
    sft_model = pipeline.stage1_supervised_finetuning (base_model, demonstrations)
    
    # Stage 2: Reward Model
    print("Stage 2: Training Reward Model...")
    comparisons = load_comparisons()  # ~100k human preference comparisons
    reward_model = pipeline.stage2_reward_model (sft_model, comparisons)
    
    # Stage 3: PPO
    print("Stage 3: RL Optimization...")
    prompts = load_prompts()  # Diverse prompts for RL training
    final_model = pipeline.stage3_ppo_optimization (sft_model, reward_model, prompts)
    
    print("RLHF complete! Model is now aligned.")
    return final_model
\`\`\`

---

## Reward Model Training

### Learning Human Preferences

\`\`\`python
"""
Detailed reward model implementation
"""

import torch
import torch.nn as nn
from transformers import AutoModel

class RewardModel (nn.Module):
    """
    Reward model: Scores responses based on human preferences
    """
    
    def __init__(self, base_model_name):
        super().__init__()
        
        # Load base transformer
        self.transformer = AutoModel.from_pretrained (base_model_name)
        
        # Reward head: Single scalar output
        self.reward_head = nn.Linear(
            self.transformer.config.hidden_size, 
            1
        )
    
    def forward (self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        
        Returns:
            reward: [batch] scalar reward for each example
        """
        # Get hidden states
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use last token's hidden state (or mean pooling)
        hidden = outputs.last_hidden_state[:, -1, :]  # [batch, hidden_size]
        
        # Compute reward
        reward = self.reward_head (hidden).squeeze(-1)  # [batch]
        
        return reward

class RewardModelTrainer:
    """
    Train reward model on comparison data
    """
    
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    def create_comparison_batch (self, prompt, response_a, response_b, preference):
        """
        Format comparison for training
        """
        # Concatenate prompt + response
        text_a = prompt + " " + response_a
        text_b = prompt + " " + response_b
        
        # Tokenize
        inputs_a = tokenizer (text_a, return_tensors='pt', truncation=True, max_length=512)
        inputs_b = tokenizer (text_b, return_tensors='pt', truncation=True, max_length=512)
        
        return inputs_a, inputs_b, preference
    
    def train_step (self, prompt, response_a, response_b, preference):
        """
        Single training step
        """
        # Prepare inputs
        inputs_a, inputs_b, pref = self.create_comparison_batch(
            prompt, response_a, response_b, preference
        )
        
        # Get rewards
        reward_a = self.model(**inputs_a)
        reward_b = self.model(**inputs_b)
        
        # Loss: Preferred response should have higher reward
        # Bradley-Terry model: P(A > B) = sigmoid (r_A - r_B)
        if pref == 'A':
            loss = -torch.log (torch.sigmoid (reward_a - reward_b))
        else:
            loss = -torch.log (torch.sigmoid (reward_b - reward_a))
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def train (self, comparison_dataset, num_epochs=3):
        """
        Full training loop
        """
        self.model.train()
        
        for epoch in range (num_epochs):
            total_loss = 0
            
            for batch in comparison_dataset:
                loss = self.train_step(
                    batch['prompt'],
                    batch['response_a'],
                    batch['response_b'],
                    batch['preference']
                )
                total_loss += loss
            
            avg_loss = total_loss / len (comparison_dataset)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        
        return self.model

# Collecting human preferences
class PreferenceCollection:
    """
    How to collect human comparison data
    """
    
    def generate_comparisons (self, model, prompts):
        """
        Generate pairs of responses for humans to compare
        """
        comparisons = []
        
        for prompt in prompts:
            # Generate two responses (different temperatures)
            response_a = model.generate (prompt, temperature=0.7, seed=0)
            response_b = model.generate (prompt, temperature=0.9, seed=42)
            
            comparisons.append({
                'prompt': prompt,
                'response_a': response_a,
                'response_b': response_b,
                'preference': None  # To be filled by human
            })
        
        return comparisons
    
    def labeling_interface (self, comparison):
        """
        UI for humans to provide preferences
        """
        print(f"Prompt: {comparison['prompt']}\\n")
        print(f"Response A: {comparison['response_a']}\\n")
        print(f"Response B: {comparison['response_b']}\\n")
        print("Which response is better? (A/B/tie): ")
        
        preference = input()
        return preference
    
    def quality_control (self, comparisons):
        """
        Ensure high-quality labels
        """
        # Multiple labelers per comparison
        # Agreement threshold (e.g., 75%)
        # Remove ambiguous cases
        # Test labelers with gold standard examples
        pass

# How many comparisons needed?
data_requirements = {
    "SFT demonstrations": "10k-100k examples",
    "Reward model comparisons": "50k-500k comparisons",
    "Cost per comparison": "$0.10-$1 (human labeling)",
    "Total labeling cost": "$5k-$500k"
}
\`\`\`

---

## PPO Implementation

### Proximal Policy Optimization

\`\`\`python
"""
PPO for RLHF
"""

class PPOTrainer:
    """
    Proximal Policy Optimization for LLMs
    """
    
    def __init__(self, policy_model, reward_model, ref_model, config):
        self.policy = policy_model  # Model to optimize
        self.reward_model = reward_model  # Trained reward model
        self.ref_model = ref_model  # Reference (frozen SFT model)
        
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config['lr'])
        self.kl_coef = config['kl_coef']  # KL penalty coefficient
        self.clip_range = config['clip_range']  # PPO clipping
        self.value_coef = config['value_coef']  # Value loss coefficient
    
    def compute_rewards (self, prompts, responses):
        """
        Compute reward for generated responses
        """
        # Reward from reward model
        rewards = []
        for prompt, response in zip (prompts, responses):
            text = prompt + " " + response
            reward = self.reward_model (text)
            rewards.append (reward)
        
        # KL penalty: Stay close to reference model
        kl_penalties = []
        for prompt, response in zip (prompts, responses):
            kl = self.compute_kl_divergence (prompt, response)
            kl_penalties.append (kl)
        
        # Total reward = model reward - KL penalty
        total_rewards = [
            r - self.kl_coef * kl 
            for r, kl in zip (rewards, kl_penalties)
        ]
        
        return total_rewards
    
    def compute_kl_divergence (self, prompt, response):
        """
        KL divergence between policy and reference
        """
        # Policy logits
        policy_logits = self.policy (prompt + response)
        
        # Reference logits (no grad)
        with torch.no_grad():
            ref_logits = self.ref_model (prompt + response)
        
        # KL(policy || ref)
        kl = F.kl_div(
            F.log_softmax (policy_logits, dim=-1),
            F.softmax (ref_logits, dim=-1),
            reduction='batchmean'
        )
        
        return kl.item()
    
    def ppo_step (self, prompts):
        """
        Single PPO update
        """
        # Generate responses with current policy
        responses = [self.policy.generate (p) for p in prompts]
        
        # Compute rewards
        rewards = self.compute_rewards (prompts, responses)
        
        # Compute advantages (how good action was vs expected)
        advantages = self.compute_advantages (rewards)
        
        # PPO update (multiple epochs on same batch)
        for _ in range(4):  # PPO epochs
            # Forward pass
            logprobs = self.compute_log_probs (prompts, responses)
            
            # Ratio of new/old policy
            with torch.no_grad():
                old_logprobs = logprobs.detach()
            
            ratio = torch.exp (logprobs - old_logprobs)
            
            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp (ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            policy_loss = -torch.min (surr1, surr2).mean()
            
            # Value loss (if using critic)
            value_loss = 0  # Simplified
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss
            
            # Update
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item()
    
    def compute_advantages (self, rewards):
        """
        Compute advantages using Generalized Advantage Estimation (GAE)
        """
        # Simplified: Just center and normalize rewards
        advantages = torch.tensor (rewards)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    def compute_log_probs (self, prompts, responses):
        """
        Log probability of responses under policy
        """
        log_probs = []
        
        for prompt, response in zip (prompts, responses):
            # Tokenize
            input_ids = tokenizer (prompt + response, return_tensors='pt')['input_ids']
            prompt_len = len (tokenizer (prompt)['input_ids'])
            
            # Forward pass
            logits = self.policy (input_ids)
            
            # Log probabilities for generated tokens
            log_prob = 0
            for i in range (prompt_len, len (input_ids[0]) - 1):
                token_logits = logits[0, i, :]
                token_log_probs = F.log_softmax (token_logits, dim=-1)
                next_token = input_ids[0, i + 1]
                log_prob += token_log_probs[next_token]
            
            log_probs.append (log_prob)
        
        return torch.stack (log_probs)
    
    def train (self, prompts, num_steps=10000):
        """
        Full PPO training
        """
        for step in range (num_steps):
            # Sample batch of prompts
            batch = random.sample (prompts, 32)
            
            # PPO update
            loss = self.ppo_step (batch)
            
            # Log
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss:.4f}")
        
        return self.policy

# Why PPO for RLHF?
ppo_advantages = {
    "Stable": "Clipping prevents large updates",
    "Sample efficient": "Reuses data for multiple updates",
    "Simple": "Fewer hyperparameters than other RL algorithms",
    "Proven": "Works well in practice for LLMs"
}
\`\`\`

---

## Alternative Alignment Methods

### DPO and Beyond

\`\`\`python
"""
Direct Preference Optimization (DPO) - Simpler than RLHF
"""

class DPO:
    """
    DPO: Skip reward model, optimize directly on preferences
    
    Key insight: Can optimize for preferences without explicit reward
    Simpler and more stable than RLHF
    """
    
    def __init__(self, policy_model, ref_model, beta=0.1):
        self.policy = policy_model
        self.ref_model = ref_model
        self.beta = beta  # Temperature parameter
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-6)
    
    def dpo_loss (self, prompt, preferred_response, rejected_response):
        """
        DPO loss function
        
        Maximizes:
        log (sigmoid (beta * (log (policy (preferred)/ref (preferred)) - 
                            log (policy (rejected)/ref (rejected)))))
        """
        # Log probs under policy
        log_prob_preferred = self.compute_log_prob (self.policy, prompt, preferred_response)
        log_prob_rejected = self.compute_log_prob (self.policy, prompt, rejected_response)
        
        # Log probs under reference (frozen)
        with torch.no_grad():
            ref_log_prob_preferred = self.compute_log_prob (self.ref_model, prompt, preferred_response)
            ref_log_prob_rejected = self.compute_log_prob (self.ref_model, prompt, rejected_response)
        
        # DPO objective
        preferred_ratio = log_prob_preferred - ref_log_prob_preferred
        rejected_ratio = log_prob_rejected - ref_log_prob_rejected
        
        loss = -torch.log (torch.sigmoid (self.beta * (preferred_ratio - rejected_ratio)))
        
        return loss
    
    def compute_log_prob (self, model, prompt, response):
        """
        Compute log probability of response given prompt
        """
        text = prompt + " " + response
        input_ids = tokenizer (text, return_tensors='pt')['input_ids']
        
        logits = model (input_ids)
        log_probs = F.log_softmax (logits, dim=-1)
        
        # Sum log probs of response tokens
        prompt_len = len (tokenizer (prompt)['input_ids'])
        total_log_prob = 0
        
        for i in range (prompt_len, len (input_ids[0]) - 1):
            token_id = input_ids[0, i + 1]
            total_log_prob += log_probs[0, i, token_id]
        
        return total_log_prob
    
    def train_step (self, batch):
        """
        DPO training step
        """
        for example in batch:
            loss = self.dpo_loss(
                example['prompt'],
                example['preferred'],
                example['rejected']
            )
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item()

# Comparison: RLHF vs DPO
alignment_methods = {
    "RLHF": {
        "stages": 3,
        "complexity": "High",
        "stability": "Can be unstable (RL)",
        "performance": "Excellent",
        "compute": "High (3 models needed)",
        "used_by": "ChatGPT, Claude"
    },
    "DPO": {
        "stages": 1,
        "complexity": "Low",
        "stability": "More stable",
        "performance": "Very good (close to RLHF)",
        "compute": "Lower (2 models)",
        "used_by": "Mistral, Zephyr"
    }
}

# Constitutional AI (Anthropic\'s approach)
class ConstitutionalAI:
    """
    Use principles to guide behavior
    
    Principles: "Be helpful, harmless, honest"
    Self-critique: Model evaluates its own outputs
    """
    
    def generate_with_principles (self, prompt, principles):
        """
        Generate response following principles
        """
        # Generate initial response
        initial_response = model.generate (prompt)
        
        # Self-critique
        critique_prompt = f"""Response: {initial_response}

Principles:
{chr(10).join('- ' + p for p in principles)}

Critique this response against the principles. 
What could be improved?

Critique:"""
        
        critique = model.generate (critique_prompt)
        
        # Revise based on critique
        revision_prompt = f"""Original response: {initial_response}

Critique: {critique}

Principles:
{chr(10).join('- ' + p for p in principles)}

Revised response following the principles:"""
        
        final_response = model.generate (revision_prompt)
        
        return final_response
\`\`\`

---

## Evaluation and Safety

### Measuring Alignment

\`\`\`python
"""
Evaluating alignment quality
"""

class AlignmentEvaluation:
    """
    Comprehensive alignment evaluation
    """
    
    def helpfulness_eval (self, model, test_prompts):
        """
        Measure helpfulness
        
        Metrics:
        - Instruction following
        - Relevance
        - Completeness
        - Clarity
        """
        scores = []
        
        for prompt in test_prompts:
            response = model.generate (prompt)
            
            # Human evaluation or LLM-as-judge
            score = self.rate_helpfulness (prompt, response)
            scores.append (score)
        
        return np.mean (scores)
    
    def harmlessness_eval (self, model, adversarial_prompts):
        """
        Test safety with adversarial prompts
        """
        refusal_rate = 0
        
        for prompt in adversarial_prompts:
            response = model.generate (prompt)
            
            if self.is_safe_refusal (response):
                refusal_rate += 1
        
        return refusal_rate / len (adversarial_prompts)
    
    def honesty_eval (self, model, factual_questions):
        """
        Measure factual accuracy and calibration
        """
        correct = 0
        well_calibrated = 0
        
        for q in factual_questions:
            response = model.generate (q['question'])
            
            # Check accuracy
            if q['answer'] in response:
                correct += 1
            
            # Check calibration (expresses uncertainty when wrong)
            if not q['answer'] in response:
                if 'uncertain' in response.lower() or 'not sure' in response.lower():
                    well_calibrated += 1
        
        accuracy = correct / len (factual_questions)
        calibration = well_calibrated / (len (factual_questions) - correct)
        
        return accuracy, calibration
    
    def comprehensive_eval (self, model):
        """
        Full evaluation suite
        """
        results = {
            'helpfulness': self.helpfulness_eval (model, helpful_prompts),
            'harmlessness': self.harmlessness_eval (model, harmful_prompts),
            'honesty': self.honesty_eval (model, factual_questions),
            'bias': self.bias_eval (model),
            'toxicity': self.toxicity_eval (model)
        }
        
        return results

# Red teaming
class RedTeaming:
    """
    Adversarial testing to find failures
    """
    
    def generate_adversarial_prompts (self):
        """
        Create prompts designed to elicit bad behavior
        """
        categories = [
            "Harmful instructions",
            "Biased requests",
            "Privacy violations",
            "Misinformation",
            "Manipulation",
            "Jailbreaks"
        ]
        
        # Use LLM to generate adversarial prompts
        # Test model's robustness
        pass
    
    def test_jailbreaks (self, model):
        """
        Test known jailbreak techniques
        """
        jailbreaks = [
            "DAN prompts",
            "Roleplay scenarios",
            "Hypothetical framing",
            "Multi-step tricks"
        ]
        
        # Test if model can be tricked
        pass
\`\`\`

---

## Conclusion

Alignment techniques:

1. **RLHF**: 3-stage process (SFT → Reward Model → PPO)
2. **DPO**: Simpler alternative, optimize directly on preferences
3. **Constitutional AI**: Self-critique and principles
4. **Evaluation**: Helpfulness, harmlessness, honesty

**Key Insights**:
- Base models need alignment to be useful
- Human feedback is critical (50k+ comparisons)
- RLHF is complex but effective
- DPO is simpler and nearly as good
- Continuous evaluation and improvement needed

**Practical Considerations**:
- RLHF cost: $50k-$500k for full pipeline
- DPO cost: 50% less than RLHF
- Alignment is ongoing (not one-time)
- Trade-offs between capabilities and safety

**Future Directions**:
- Scalable oversight (AI helping humans evaluate)
- Better reward models
- More efficient RL algorithms
- Automated red teaming

Alignment transforms base models into the helpful assistants we know as ChatGPT, Claude, and others. It's what makes LLMs truly useful.
`,
};
