/**
 * Fine-Tuning Open-Source Models Section
 * Module 16: Evaluation, Data Operations & Fine-Tuning
 */

export const fineTuningOpenSourceModels = {
  id: 'fine-tuning-open-source-models',
  title: 'Fine-Tuning Open-Source Models',
  content: `# Fine-Tuning Open-Source Models

Master fine-tuning Llama, Mistral, and other open-source LLMs with LoRA and full fine-tuning.

## Overview: Open-Source Fine-Tuning

**Why open-source?**
- ✅ Full control and customization
- ✅ No API costs (after training)
- ✅ Data privacy (train on-premise)
- ✅ Can commercialize freely
- ✅ State-of-the-art models (Llama 2, Mistral)

**Challenges:**
- ❌ Requires GPUs for training
- ❌ More complex infrastructure
- ❌ Need ML engineering expertise

## Popular Open-Source Models

\`\`\`python
class ModelSelector:
    """Choose appropriate open-source model."""
    
    MODELS = {
        'llama-2-7b': {
            'parameters': '7B',
            'vram_needed': '16GB',
            'quality': 'good',
            'speed': 'fast',
            'use_case': 'General purpose, chatbots'
        },
        'llama-2-13b': {
            'parameters': '13B',
            'vram_needed': '32GB',
            'quality': 'very good',
            'speed': 'moderate',
            'use_case': 'Higher quality responses'
        },
        'llama-2-70b': {
            'parameters': '70B',
            'vram_needed': '80GB+ (multi-GPU)',
            'quality': 'excellent',
            'speed': 'slow',
            'use_case': 'Production, highest quality'
        },
        'mistral-7b': {
            'parameters': '7B',
            'vram_needed': '16GB',
            'quality': 'excellent',
            'speed': 'fast',
            'use_case': 'Best quality for size'
        },
        'mixtral-8x7b': {
            'parameters': '47B (8x7B MoE)',
            'vram_needed': '48GB',
            'quality': 'excellent',
            'speed': 'moderate',
            'use_case': 'Production, efficient'
        }
    }
    
    @classmethod
    def recommend(cls, gpu_vram_gb: int, use_case: str) -> str:
        """Recommend model based on resources."""
        
        if gpu_vram_gb < 16:
            return "Need at least 16GB VRAM. Consider cloud GPUs."
        
        if gpu_vram_gb >= 48:
            return "mixtral-8x7b"  # Best quality
        elif gpu_vram_gb >= 32:
            return "llama-2-13b"
        else:
            return "mistral-7b"  # Best for 16GB

print(ModelSelector.recommend(gpu_vram_gb=24, use_case="chatbot"))
# "mistral-7b"
\`\`\`

## LoRA Fine-Tuning (Recommended)

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import torch

class LoRAFineTuner:
    """Fine-tune with LoRA (efficient method)."""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        use_4bit: bool = True  # Quantization for efficiency
    ):
        self.model_name = model_name
        self.use_4bit = use_4bit
        
        # Load model
        print(f"Loading {model_name}...")
        
        if use_4bit:
            # 4-bit quantization (QLoRA)
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare for LoRA
        self.model = prepare_model_for_kbit_training(self.model)
    
    def setup_lora(
        self,
        r: int = 8,  # LoRA rank (lower = fewer params)
        lora_alpha: int = 32,  # LoRA scaling
        target_modules: List[str] = None
    ):
        """Configure LoRA adapters."""
        
        if target_modules is None:
            # Default: target attention layers
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
        print(f"Total parameters: {total_params:,}")
    
    def prepare_dataset(
        self,
        examples: List[Dict[str, str]]
    ) -> Dataset:
        """Prepare dataset for training."""
        
        def format_prompt(example):
            """Format as instruction-following."""
            prompt = f"""<s>[INST] {example['input']} [/INST] {example['output']}</s>"""
            return {"text": prompt}
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(examples)
        dataset = dataset.map(format_prompt)
        
        return dataset
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        output_dir: str = "./lora-finetuned",
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4
    ):
        """Train the model with LoRA."""
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Effective batch = 4 * 4 = 16
            learning_rate=learning_rate,
            fp16=True,  # Mixed precision
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            warmup_steps=100,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit"  # Efficient optimizer
        )
        
        from transformers import Trainer, DataCollatorForLanguageModeling
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # Causal LM, not masked LM
            )
        )
        
        # Train!
        print("\\nStarting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model(output_dir)
        print(f"✅ Model saved to {output_dir}")
    
    def merge_and_save(
        self,
        lora_model_path: str,
        output_path: str = "./merged-model"
    ):
        """Merge LoRA adapters back into base model."""
        
        from peft import PeftModel
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, lora_model_path)
        
        # Merge
        print("Merging LoRA adapters...")
        model = model.merge_and_unload()
        
        # Save merged model
        model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        print(f"✅ Merged model saved to {output_path}")

# Usage
finetuner = LoRAFineTuner(
    model_name="mistralai/Mistral-7B-v0.1",
    use_4bit=True  # QLoRA for efficiency
)

# Setup LoRA
finetuner.setup_lora(r=8, lora_alpha=32)

# Prepare data
train_data = finetuner.prepare_dataset(training_examples)
val_data = finetuner.prepare_dataset(validation_examples)

# Train
finetuner.train(
    train_dataset=train_data,
    val_dataset=val_data,
    epochs=3,
    batch_size=4
)

# Merge and save (optional - for deployment)
finetuner.merge_and_save(
    lora_model_path="./lora-finetuned",
    output_path="./my-custom-model"
)
\`\`\`

## Full Fine-Tuning (Advanced)

\`\`\`python
class FullFineTuner:
    """Full parameter fine-tuning (all weights updated)."""
    
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        output_dir: str = "./full-finetuned",
        epochs: int = 1,  # Fewer epochs for full fine-tune
        batch_size: int = 1,  # Smaller batch (memory)
        learning_rate: float = 1e-5  # Lower LR
    ):
        """Full fine-tuning."""
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=16,  # Accumulate to effective batch of 16
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=500,
            save_total_limit=2,
            warmup_steps=500,
            lr_scheduler_type="linear",
            gradient_checkpointing=True  # Save memory
        )
        
        from transformers import Trainer
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        trainer.train()
        trainer.save_model(output_dir)

# Usage: Only if you have lots of data (10K+ examples) and GPUs
# Most users should use LoRA instead!
\`\`\`

## Inference with Fine-Tuned Model

\`\`\`python
class FineTunedInference:
    """Run inference with fine-tuned model."""
    
    def __init__(self, model_path: str):
        """
        Load fine-tuned model.
        Can be:
        - LoRA adapter path
        - Merged model path
        """
        
        if self._is_lora_model(model_path):
            # Load with LoRA adapters
            from peft import PeftModel
            
            base_model_name = self._get_base_model_name(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Load merged model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate response."""
        
        # Format prompt
        formatted = f"<s>[INST] {prompt} [/INST]"
        
        # Tokenize
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        response = response.split("[/INST]")[-1].strip()
        
        return response

# Usage
model = FineTunedInference("./lora-finetuned")

response = model.generate(
    prompt="Explain quantum computing",
    temperature=0.7
)

print(response)
\`\`\`

## Deployment Strategies

### 1. vLLM for Fast Inference

\`\`\`python
# vLLM: Optimized inference server
# Install: pip install vllm

from vllm import LLM, SamplingParams

class ProductionDeployment:
    """Deploy fine-tuned model with vLLM."""
    
    def __init__(self, model_path: str):
        # Load with vLLM (much faster than transformers)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,  # Multi-GPU if needed
            dtype="float16"
        )
    
    def batch_generate(
        self,
        prompts: List[str],
        temperature: float = 0.7
    ) -> List[str]:
        """Efficient batch generation."""
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=512
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        return [output.outputs[0].text for output in outputs]

# Usage
deployment = ProductionDeployment("./my-custom-model")

# Batch inference (much faster!)
responses = deployment.batch_generate([
    "Prompt 1",
    "Prompt 2",
    "Prompt 3"
])
\`\`\`

### 2. Quantization for Deployment

\`\`\`python
# Quantize for smaller size and faster inference
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./my-custom-model")
tokenizer = AutoTokenizer.from_pretrained("./my-custom-model")

# Quantize to 8-bit or 4-bit
model = model.to("cuda")  # Move to GPU first

# Save quantized
model.save_pretrained("./my-custom-model-quantized", safe_serialization=True)

# Now 2-4x smaller and faster!
\`\`\`

## Production Checklist

✅ **Environment Setup**
- [ ] GPU with sufficient VRAM
- [ ] CUDA toolkit installed
- [ ] Dependencies installed (transformers, peft, bitsandbytes)

✅ **Training**
- [ ] LoRA config optimized
- [ ] Batch size tuned for GPU
- [ ] Validation loss monitored
- [ ] Checkpoints saved

✅ **Evaluation**
- [ ] Tested on held-out data
- [ ] Compared to base model
- [ ] No catastrophic forgetting

✅ **Deployment**
- [ ] Model merged (if using LoRA)
- [ ] Optimized for inference (vLLM/quantization)
- [ ] Serving infrastructure ready
- [ ] Monitoring enabled

## Next Steps

You now understand open-source fine-tuning. Next, learn:
- RAG evaluation
- Multi-modal evaluation
- Continuous monitoring
- Building evaluation platforms
`,
};
