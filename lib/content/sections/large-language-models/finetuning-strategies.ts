export const finetuningStrategies = {
  title: 'Fine-tuning Strategies',
  id: 'finetuning-strategies',
  content: `
# Fine-tuning Strategies

## Introduction

While training LLMs from scratch requires millions of dollars, fine-tuning adapts existing models to specific tasks for a fraction of the cost. From full fine-tuning to parameter-efficient methods like LoRA and QLoRA, this section covers techniques to customize LLMs for your use case without breaking the bank.

### Why Fine-Tune?

**Task Specialization**: Adapt general model to specific domain
**Better Performance**: Beat zero-shot on your task
**Cost-Effective**: Use smaller, fine-tuned models vs large general models
**Control**: Customize behavior, tone, output format
**Privacy**: Keep sensitive data in-house during training

---

## Full Fine-Tuning

### Complete Model Adaptation

\`\`\`python
"""
Full fine-tuning: Update all model parameters
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch

class FullFineTuning:
    """
    Traditional fine-tuning approach
    
    Pros:
    - Best possible performance
    - Full model adaptation
    
    Cons:
    - Expensive (requires all model in GPU memory)
    - Slow training
    - Requires significant compute
    """
    
    def __init__(self, model_name="gpt2"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_dataset(self, examples):
        """
        Format dataset for fine-tuning
        """
        # For instruction following
        prompts = []
        for ex in examples:
            prompt = f"### Instruction:\\n{ex['instruction']}\\n\\n### Response:\\n{ex['response']}"
            prompts.append(prompt)
        
        # Tokenize
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Labels = input_ids (for causal LM)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def train(self, train_dataset, val_dataset, output_dir="./finetuned_model"):
        """
        Train with Hugging Face Trainer
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,  # Effective batch size = 16
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            fp16=True,  # Mixed precision
            gradient_checkpointing=True,  # Save memory
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train
        trainer.train()
        
        # Save
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer

# Example: Fine-tune GPT-2 for SQL generation
class SQLFineTuning:
    """
    Fine-tune for text-to-SQL
    """
    
    def prepare_sql_data(self, examples):
        """
        Format: Question → SQL
        """
        formatted = []
        for ex in examples:
            text = f"""### Question: {ex['question']}
### Database Schema: {ex['schema']}
### SQL:
{ex['sql']}"""
            formatted.append(text)
        
        return formatted
    
    def train_sql_model(self):
        """
        Train text-to-SQL model
        """
        # Load data
        train_data = load_dataset("wikisql", split="train[:5000]")
        val_data = load_dataset("wikisql", split="validation[:500]")
        
        # Prepare
        train_texts = self.prepare_sql_data(train_data)
        val_texts = self.prepare_sql_data(val_data)
        
        # Fine-tune
        tuner = FullFineTuning("gpt2")
        trainer = tuner.train(train_texts, val_texts)
        
        return trainer.model

# Costs of full fine-tuning
costs = {
    "GPT-2 (125M)": {
        "time": "~2 hours on 1x A100",
        "cost": "~$6",
        "feasible": "Yes"
    },
    "GPT-2-Large (774M)": {
        "time": "~12 hours on 1x A100",
        "cost": "~$36",
        "feasible": "Yes"
    },
    "LLaMA-7B": {
        "time": "~2 days on 8x A100",
        "cost": "~$1,000",
        "feasible": "With resources"
    },
    "LLaMA-70B": {
        "time": "~2 weeks on 64x A100",
        "cost": "~$50,000",
        "feasible": "Rarely practical"
    }
}
\`\`\`

---

## Low-Rank Adaptation (LoRA)

### Parameter-Efficient Fine-Tuning

\`\`\`python
"""
LoRA: Fine-tune with minimal parameters
"""

import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

class LoRAFineTuning:
    """
    LoRA: Add low-rank matrices to attention weights
    
    Key idea:
    - Freeze original model weights
    - Add small trainable matrices (A, B)
    - W' = W + BA (where B and A are low-rank)
    
    Benefits:
    - 100x fewer trainable parameters
    - 3x less GPU memory
    - Faster training
    - Can merge back to base model
    """
    
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        from transformers import AutoModelForCausalLM
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,  # Quantize base model
            device_map="auto"
        )
        
        # LoRA config
        lora_config = LoraConfig(
            r=8,  # Rank of update matrices
            lora_alpha=16,  # Scaling factor
            target_modules=["q_proj", "v_proj"],  # Which layers to adapt
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.base_model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        # Output: trainable params: 4.2M || all params: 7B || trainable: 0.06%
    
    def understand_lora(self):
        """
        How LoRA works
        """
        # Original attention: Q = X @ W_q
        # where W_q is [d_model, d_model], e.g., [4096, 4096]
        
        # LoRA: Q = X @ (W_q + B @ A)
        # where:
        # - W_q: [4096, 4096] (frozen)
        # - B: [4096, r] (trainable)
        # - A: [r, 4096] (trainable)
        # - r = 8 (rank)
        
        # Parameters:
        # Original: 4096 * 4096 = 16.7M
        # LoRA: 4096 * 8 + 8 * 4096 = 65.5K (0.4% of original!)
        
        # Example implementation
        class LoRALayer(nn.Module):
            def __init__(self, in_features, out_features, rank=8, alpha=16):
                super().__init__()
                
                # Frozen weight
                self.weight = nn.Parameter(torch.randn(out_features, in_features))
                self.weight.requires_grad = False
                
                # LoRA matrices
                self.lora_A = nn.Parameter(torch.randn(rank, in_features))
                self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
                
                self.scaling = alpha / rank
            
            def forward(self, x):
                # Original: x @ W^T
                result = x @ self.weight.T
                
                # LoRA: x @ A^T @ B^T
                lora_result = (x @ self.lora_A.T) @ self.lora_B.T
                
                return result + self.scaling * lora_result
    
    def train(self, train_dataset, output_dir="./lora_model"):
        """
        Train only LoRA parameters
        """
        from transformers import TrainingArguments, Trainer
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,  # Can use larger batch!
            gradient_accumulation_steps=2,
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            optim="adamw_torch",
            learning_rate=3e-4,  # Higher LR for LoRA
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        trainer.train()
        
        # Save only LoRA weights (few MB vs GBs for full model)
        self.model.save_pretrained(output_dir)
        
        return trainer

# Example: Fine-tune LLaMA-7B with LoRA
def finetune_llama_with_lora():
    """
    Fine-tune 7B model on single GPU
    """
    from datasets import load_dataset
    
    # Load dataset
    dataset = load_dataset("databricks/databricks-dolly-15k")
    
    # Prepare
    def format_instruction(example):
        return f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}"""
    
    dataset = dataset.map(lambda x: {"text": format_instruction(x)})
    
    # Fine-tune
    lora_tuner = LoRAFineTuning("meta-llama/Llama-2-7b-hf")
    trainer = lora_tuner.train(dataset['train'])
    
    print("LoRA weights saved! Only 5MB vs 13GB for full model")

# Inference with LoRA
def load_and_use_lora(base_model_name, lora_path):
    """
    Load LoRA adapter for inference
    """
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # Can merge LoRA weights into base model for faster inference
    model = model.merge_and_unload()
    
    return model
\`\`\`

---

## QLoRA: Quantized LoRA

### 4-bit Fine-Tuning

\`\`\`python
"""
QLoRA: Fine-tune 65B models on single GPU
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

class QLoRAFineTuning:
    """
    QLoRA: LoRA + 4-bit quantization
    
    Key innovations:
    1. 4-bit NormalFloat (NF4) quantization
    2. Double quantization
    3. Paged optimizers
    
    Result: Fine-tune 65B model on 48GB GPU!
    """
    
    def __init__(self, model_name="meta-llama/Llama-2-70b-hf"):
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,  # Double quantization
            bnb_4bit_quant_type="nf4",  # NormalFloat4
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load model in 4-bit
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Prepare for LoRA
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=64,  # Can use higher rank with QLoRA
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
    
    def memory_comparison(self):
        """
        Memory usage comparison
        """
        comparisons = {
            "Full FP16 (70B)": {
                "memory": "140GB",
                "gpus": "2x A100 (80GB)",
                "trainable_params": "70B"
            },
            "LoRA FP16 (70B)": {
                "memory": "140GB (base) + 100MB (LoRA)",
                "gpus": "2x A100 (80GB)",
                "trainable_params": "50M"
            },
            "QLoRA 4-bit (70B)": {
                "memory": "35GB (base) + 100MB (LoRA)",
                "gpus": "1x A100 (40GB) ✓",
                "trainable_params": "50M"
            }
        }
        
        return comparisons

# Complete QLoRA training example
def train_70b_model_on_single_gpu():
    """
    Fine-tune LLaMA-70B on consumer hardware
    """
    from transformers import AutoTokenizer, TrainingArguments, Trainer
    from datasets import load_dataset
    
    # Load model with QLoRA
    model_name = "meta-llama/Llama-2-70b-hf"
    tuner = QLoRAFineTuning(model_name)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset("timdettmers/openassistant-guanaco")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./qlora_70b",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # Effective batch = 16
        learning_rate=2e-4,
        fp16=False,
        bf16=True,  # Use BF16
        logging_steps=10,
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        save_strategy="epoch",
        max_grad_norm=0.3,
    )
    
    # Trainer
    trainer = Trainer(
        model=tuner.model,
        args=training_args,
        train_dataset=dataset['train'],
    )
    
    # Train
    trainer.train()
    
    # Save LoRA weights
    tuner.model.save_pretrained("./qlora_70b_final")
    
    print("Successfully fine-tuned 70B model on single GPU!")

# NormalFloat4 (NF4) quantization
class NF4Quantization:
    """
    Understanding NF4
    
    Key insight: Model weights follow normal distribution
    NF4 optimizes quantization for this distribution
    
    Result: Better quality than standard 4-bit quantization
    """
    
    def quantize_to_nf4(self, weights):
        """
        Simplified NF4 quantization
        """
        # 1. Normalize to [-1, 1]
        scale = torch.abs(weights).max()
        normalized = weights / scale
        
        # 2. Quantize to 16 levels (4 bits)
        # NF4 levels optimized for normal distribution
        nf4_levels = torch.tensor([
            -1.0, -0.6962, -0.5251, -0.3949,
            -0.2844, -0.1848, -0.0911, 0.0,
            0.0911, 0.1848, 0.2844, 0.3949,
            0.5251, 0.6962, 0.8807, 1.0
        ])
        
        # 3. Find closest level for each weight
        quantized = torch.zeros_like(normalized)
        for i, w in enumerate(normalized.flatten()):
            idx = torch.argmin(torch.abs(nf4_levels - w))
            quantized.flatten()[i] = nf4_levels[idx]
        
        # Store: 4 bits per weight + scale factor
        return quantized, scale
    
    def dequantize(self, quantized, scale):
        """
        Convert back to FP16 for computation
        """
        return quantized * scale
\`\`\`

---

## Adapter Methods

### Prefix Tuning and Prompt Tuning

\`\`\`python
"""
Alternative parameter-efficient methods
"""

# 1. Prefix Tuning
class PrefixTuning:
    """
    Add trainable prefix to each layer
    
    Key idea:
    - Prepend trainable vectors to K, V in each attention layer
    - Freeze all model weights
    - Only train prefixes
    
    Parameters: 0.1-1% of full model
    """
    
    def __init__(self, model, num_prefix_tokens=20):
        self.model = model
        self.num_prefix_tokens = num_prefix_tokens
        
        # Add prefix parameters to each layer
        for layer in model.transformer.layers:
            # Key prefix
            layer.attention.key_prefix = nn.Parameter(
                torch.randn(num_prefix_tokens, model.d_model)
            )
            
            # Value prefix
            layer.attention.value_prefix = nn.Parameter(
                torch.randn(num_prefix_tokens, model.d_model)
            )
    
    def forward_with_prefix(self, layer, x):
        """
        Attention with prefix
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute Q, K, V
        Q = layer.attention.query(x)
        K = layer.attention.key(x)
        V = layer.attention.value(x)
        
        # Prepend prefix to K, V
        prefix_K = layer.attention.key_prefix.unsqueeze(0).expand(batch_size, -1, -1)
        prefix_V = layer.attention.value_prefix.unsqueeze(0).expand(batch_size, -1, -1)
        
        K = torch.cat([prefix_K, K], dim=1)
        V = torch.cat([prefix_V, V], dim=1)
        
        # Attention
        attention_output = layer.attention.compute(Q, K, V)
        
        return attention_output

# 2. Prompt Tuning
class PromptTuning:
    """
    Even simpler: Add trainable tokens to input
    
    Key idea:
    - Prepend learnable embeddings to input
    - Freeze entire model
    - Only train input prefix embeddings
    
    Parameters: 0.01-0.1% of full model
    """
    
    def __init__(self, model, num_prompt_tokens=20):
        self.model = model
        d_model = model.config.hidden_size
        
        # Trainable prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_prompt_tokens, d_model)
        )
        
        # Freeze model
        for param in model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids):
        """
        Prepend prompt embeddings
        """
        # Get input embeddings
        inputs_embeds = self.model.transformer.wte(input_ids)
        
        # Expand prompt for batch
        batch_size = input_ids.shape[0]
        prompt_embeds = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
        
        # Forward through model
        outputs = self.model(inputs_embeds=inputs_embeds)
        
        return outputs

# 3. IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
class IA3:
    """
    Scale activations with learned vectors
    
    Key idea:
    - Multiply K, V, and FFN activations by learned scalars
    - Even fewer parameters than LoRA
    
    Parameters: 0.01% of full model
    """
    
    def __init__(self, model):
        for layer in model.transformer.layers:
            # Scaling vectors
            layer.attention.k_scale = nn.Parameter(torch.ones(model.d_model))
            layer.attention.v_scale = nn.Parameter(torch.ones(model.d_model))
            layer.ffn.scale = nn.Parameter(torch.ones(model.d_ff))
    
    def forward_attention(self, layer, x):
        """
        Attention with IA³ scaling
        """
        Q = layer.attention.query(x)
        K = layer.attention.key(x) * layer.attention.k_scale  # Scale
        V = layer.attention.value(x) * layer.attention.v_scale  # Scale
        
        return layer.attention.compute(Q, K, V)

# Method comparison
methods = {
    "Full Fine-tuning": {
        "params": "100%",
        "memory": "High",
        "performance": "Best",
        "use_case": "Small models, unlimited compute"
    },
    "LoRA": {
        "params": "0.1-1%",
        "memory": "Medium",
        "performance": "95-99% of full",
        "use_case": "Most common, good balance"
    },
    "QLoRA": {
        "params": "0.1-1%",
        "memory": "Low (4-bit)",
        "performance": "90-98% of full",
        "use_case": "Large models on limited hardware"
    },
    "Prefix Tuning": {
        "params": "0.1-0.5%",
        "memory": "Low",
        "performance": "85-95% of full",
        "use_case": "When LoRA not available"
    },
    "Prompt Tuning": {
        "params": "0.01-0.1%",
        "memory": "Minimal",
        "performance": "70-85% of full",
        "use_case": "Quick experiments, very limited compute"
    },
    "IA³": {
        "params": "0.01%",
        "memory": "Minimal",
        "performance": "85-95% of full",
        "use_case": "Extreme efficiency needed"
    }
}
\`\`\`

---

## Instruction Tuning

### Teaching Models to Follow Instructions

\`\`\`python
"""
Instruction tuning: Fine-tune on instruction-response pairs
"""

class InstructionTuning:
    """
    Format: Instruction → Response
    
    Dataset format:
    {
        "instruction": "Write a poem about AI",
        "input": "",  # Optional context
        "response": "In circuits deep..."
    }
    """
    
    def format_instruction(self, example):
        """
        Create instruction-following prompt
        """
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}
"""
        
        if example.get('input'):
            prompt += f"""
### Input:
{example['input']}
"""
        
        prompt += f"""
### Response:
{example['response']}"""
        
        return prompt
    
    def prepare_dataset(self, examples):
        """
        Format entire dataset
        """
        return [self.format_instruction(ex) for ex in examples]

# Popular instruction datasets
instruction_datasets = {
    "Alpaca": {
        "size": "52K",
        "quality": "Good",
        "source": "Generated by GPT-3.5",
        "use": "General instruction following"
    },
    "Dolly": {
        "size": "15K",
        "quality": "High",
        "source": "Human-written by Databricks",
        "use": "Commercial use allowed"
    },
    "FLAN": {
        "size": "1.8M",
        "quality": "High",
        "source": "Google",
        "use": "Diverse tasks"
    },
    "OpenAssistant": {
        "size": "161K",
        "quality": "Variable",
        "source": "Community crowdsourced",
        "use": "Dialogue and assistance"
    }
}

# Example: Instruction-tune LLaMA with QLoRA
def instruction_tune_llama():
    """
    Complete instruction tuning pipeline
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    # Load dataset
    dataset = load_dataset("tatsu-lab/alpaca")
    
    # Format instructions
    tuner = InstructionTuning()
    formatted_data = tuner.prepare_dataset(dataset['train'])
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenized = tokenizer(
        formatted_data,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    # Fine-tune with QLoRA
    qlora = QLoRAFineTuning("meta-llama/Llama-2-7b-hf")
    trainer = qlora.train(tokenized)
    
    return trainer.model

# Multi-turn conversation tuning
class ConversationTuning:
    """
    Fine-tune for multi-turn dialogue
    """
    
    def format_conversation(self, messages):
        """
        Format conversation history
        """
        formatted = ""
        for msg in messages:
            if msg['role'] == 'user':
                formatted += f"User: {msg['content']}\\n"
            else:
                formatted += f"Assistant: {msg['content']}\\n"
        
        return formatted
    
    def prepare_conversation_data(self, conversations):
        """
        Create training examples from conversations
        """
        examples = []
        
        for conv in conversations:
            # For each turn, predict assistant response given context
            for i, msg in enumerate(conv['messages']):
                if msg['role'] == 'assistant':
                    context = conv['messages'][:i+1]
                    formatted = self.format_conversation(context)
                    examples.append(formatted)
        
        return examples
\`\`\`

---

## Evaluation and Monitoring

### Measuring Fine-Tuning Success

\`\`\`python
"""
Evaluate fine-tuned models
"""

class FineTuningEvaluation:
    """
    Comprehensive evaluation framework
    """
    
    def evaluate_perplexity(self, model, test_data):
        """
        Perplexity: How surprised model is by test data
        Lower = better
        """
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in test_data:
                outputs = model(**batch)
                loss = outputs.loss
                
                total_loss += loss.item() * batch['input_ids'].numel()
                total_tokens += batch['input_ids'].numel()
        
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        return perplexity.item()
    
    def evaluate_task_accuracy(self, model, task_data):
        """
        Task-specific accuracy
        """
        correct = 0
        total = 0
        
        for example in task_data:
            # Generate prediction
            prediction = model.generate(example['input'])
            
            # Compare with ground truth
            if prediction.strip() == example['output'].strip():
                correct += 1
            total += 1
        
        accuracy = correct / total
        return accuracy
    
    def compare_before_after(self, base_model, finetuned_model, test_prompts):
        """
        Compare base vs fine-tuned
        """
        results = []
        
        for prompt in test_prompts:
            base_response = base_model.generate(prompt)
            ft_response = finetuned_model.generate(prompt)
            
            results.append({
                'prompt': prompt,
                'base': base_response,
                'finetuned': ft_response
            })
        
        return results
    
    def check_for_catastrophic_forgetting(self, model, general_tasks):
        """
        Ensure model didn't forget general knowledge
        """
        results = {}
        
        for task_name, task_data in general_tasks.items():
            accuracy = self.evaluate_task_accuracy(model, task_data)
            results[task_name] = accuracy
        
        # Flag if accuracy dropped significantly
        for task, acc in results.items():
            if acc < 0.5:  # Threshold
                print(f"Warning: Catastrophic forgetting on {task}")
        
        return results

# Monitor training
class TrainingMonitor:
    """
    Track training progress
    """
    
    def log_metrics(self, step, metrics):
        """
        Log to wandb or similar
        """
        import wandb
        
        wandb.log({
            'step': step,
            'train_loss': metrics['loss'],
            'learning_rate': metrics['lr'],
            'grad_norm': metrics['grad_norm'],
        })
    
    def check_overfitting(self, train_loss_history, val_loss_history):
        """
        Detect overfitting
        """
        if len(val_loss_history) < 10:
            return False
        
        # Check if validation loss increasing while training loss decreasing
        recent_train = train_loss_history[-10:]
        recent_val = val_loss_history[-10:]
        
        train_trend = recent_train[-1] - recent_train[0]
        val_trend = recent_val[-1] - recent_val[0]
        
        if train_trend < 0 and val_trend > 0:
            print("Warning: Possible overfitting detected")
            return True
        
        return False
\`\`\`

---

## Conclusion

Fine-tuning strategies for LLMs:

1. **Full Fine-Tuning**: Best performance but expensive
2. **LoRA**: 100x fewer parameters, 0.1% trainable
3. **QLoRA**: LoRA + 4-bit, fine-tune 70B on single GPU
4. **Prefix/Prompt Tuning**: Even fewer parameters, good for quick experiments

**Key Insights**:
- LoRA achieves 95-99% of full fine-tuning performance
- QLoRA enables fine-tuning massive models on consumer hardware
- Instruction tuning teaches models to follow commands
- Always evaluate for catastrophic forgetting

**Practical Recommendations**:
- Start with LoRA (best balance)
- Use QLoRA for models >13B parameters
- Fine-tune on 1K-10K high-quality examples
- Monitor for overfitting
- Test on held-out data

**Cost Comparison**:
- GPT-3.5 API: $0.50-$2 per 1M tokens
- Fine-tuned 7B with LoRA: $20-50 one-time + $0.01/1M tokens inference
- ROI: Breaks even after 10M-100M tokens

Fine-tuning is the most cost-effective way to get GPT-4-level performance for specific tasks.
`,
};
