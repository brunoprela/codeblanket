export const quantizationModelOptimization = {
  title: 'Quantization & Model Optimization',
  content: `

# Quantization & Model Optimization

## Introduction

Running large language models is expensive—both in terms of API costs and computational requirements. Quantization is a technique that reduces model size by 70-90% by using lower-precision numbers (e.g., 8-bit or 4-bit integers instead of 32-bit floats) while maintaining most of the model's capabilities.

**Key Benefits**:
- **70-90% smaller models**: 13B model from 26GB → 3.5GB (4-bit)
- **2-4x faster inference**: Less computation required
- **Run larger models locally**: Fit models that wouldn't work otherwise
- **Lower infrastructure costs**: Smaller instances needed
- **Minimal quality loss**: <2% for 8-bit, 2-5% for 4-bit

This section covers practical quantization techniques for production LLM applications.

---

## Understanding Quantization

### Precision Levels

**32-bit Float (FP32)** - Original Training Format:
\`\`\`
Range: ±3.4 × 10^38
Precision: ~7 decimal digits
Size: 4 bytes per parameter
\`\`\`

**16-bit Float (FP16)** - Half Precision:
\`\`\`
Range: ±65,504
Precision: ~3 decimal digits  
Size: 2 bytes per parameter
Reduction: 50%
Quality loss: <1%
\`\`\`

**8-bit Integer (INT8)** - Quantized:
\`\`\`
Range: -128 to 127
Precision: Integer only
Size: 1 byte per parameter
Reduction: 75%
Quality loss: 1-2%
\`\`\`

**4-bit Integer (INT4)** - Heavily Quantized:
\`\`\`
Range: -8 to 7
Precision: Integer only
Size: 0.5 bytes per parameter
Reduction: 87.5%
Quality loss: 2-5%
\`\`\`

### Model Size Example (Llama 2 13B)

| Precision | Size | Memory Required | Speed | Quality Loss |
|-----------|------|-----------------|-------|--------------|
| FP32 | 52 GB | 60 GB | 1x | 0% |
| FP16 | 26 GB | 32 GB | 1.5x | <1% |
| 8-bit | 13 GB | 16 GB | 2-3x | 1-2% |
| 4-bit | 6.5 GB | 8 GB | 3-4x | 2-5% |

**Key Insight**: 4-bit quantization lets you run a 13B model on consumer hardware (8GB GPU) that would otherwise require expensive server GPUs.

---

## Quantization with bitsandbytes

bitsandbytes enables easy 8-bit and 4-bit quantization.

### 8-bit Quantization

\`\`\`python
# Install: pip install bitsandbytes accelerate transformers

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True,  # Nested quantization for even more savings
)

# Load model in 8-bit
model_name = "meta-llama/Llama-2-13b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",  # Automatically distribute across GPUs
)

tokenizer = AutoTokenizer.from_pretrained (model_name)

# Use model normally
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer (prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode (outputs[0], skip_special_tokens=True)

print(response)

# Memory usage: ~13GB instead of 26GB!
print(f"Model size: {model.get_memory_footprint() / 1e9:.2f} GB")
\`\`\`

### 4-bit Quantization (QLoRA)

\`\`\`python
# 4-bit quantization for maximum compression

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat 4-bit
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=quantization_config,
    device_map="auto",
)

# Memory usage: ~6.5GB instead of 26GB!
# Can run 13B model on consumer GPU (RTX 3090, 4090)
\`\`\`

**Quality Comparison**:
\`\`\`python
# Benchmark on your dataset
def evaluate_model (model, test_prompts):
    """Evaluate model quality"""
    scores = []
    
    for prompt, expected in test_prompts:
        response = generate (model, prompt)
        # Score response quality (1-5)
        score = score_response (response, expected)
        scores.append (score)
    
    return sum (scores) / len (scores)

# Results on common benchmarks:
# FP16: 4.8/5.0
# 8-bit: 4.75/5.0 (1% degradation)
# 4-bit: 4.6/5.0 (4% degradation)
\`\`\`

---

## GGUF Format (llama.cpp)

GGUF is a format optimized for inference with quantization.

### Converting Models to GGUF

\`\`\`bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Convert Hugging Face model to GGUF
python convert.py /path/to/model --outtype q4_K_M

# Quantization levels:
# q2_K - 2.5 bits per weight (smallest, lowest quality)
# q3_K_M - 3.5 bits (very small)
# q4_K_M - 4.5 bits (recommended balance)
# q5_K_M - 5.5 bits (high quality)
# q6_K - 6 bits (near FP16 quality)
# q8_0 - 8 bits (excellent quality)
\`\`\`

### Running GGUF Models

\`\`\`python
# Using llama-cpp-python
# Install: pip install llama-cpp-python

from llama_cpp import Llama

# Load quantized GGUF model
llm = Llama(
    model_path="./models/llama-2-13b-q4_K_M.gguf",
    n_ctx=4096,  # Context window
    n_gpu_layers=40,  # Offload layers to GPU (if available)
    n_threads=8,  # CPU threads
)

# Generate text
output = llm(
    "Q: What is the capital of France?\nA:",
    max_tokens=50,
    temperature=0.7,
    stop=["Q:", "\n"],
)

print(output["choices"][0]["text"])

# Memory usage: ~4GB for 13B Q4 model
# Can run on laptop with 16GB RAM!
\`\`\`

### Performance Comparison

\`\`\`python
import time

def benchmark_model (model, num_runs=10):
    """Benchmark inference speed"""
    prompt = "Explain the theory of relativity: "
    
    times = []
    for _ in range (num_runs):
        start = time.time()
        model (prompt, max_tokens=100)
        elapsed = time.time() - start
        times.append (elapsed)
    
    avg_time = sum (times) / len (times)
    tokens_per_sec = 100 / avg_time
    
    return {
        "avg_time": avg_time,
        "tokens_per_sec": tokens_per_sec
    }

# Results (13B model on RTX 4090):
# FP16: 8.2 tokens/sec
# Q8: 12.5 tokens/sec (50% faster)
# Q4: 18.3 tokens/sec (125% faster)
\`\`\`

---

## AWQ (Activation-aware Weight Quantization)

AWQ is more sophisticated, preserving quality better than naive quantization.

### Quantizing with AWQ

\`\`\`python
# Install: pip install autoawq

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-13b-hf"
quant_path = "llama-2-13b-awq"

# Load model
model = AutoAWQForCausalLM.from_pretrained (model_path)
tokenizer = AutoTokenizer.from_pretrained (model_path)

# Quantize (requires calibration data)
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# Calibrate on sample data
calibration_data = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    # ... more samples
]

model.quantize(
    tokenizer, 
    quant_config=quant_config,
    calib_data=calibration_data
)

# Save quantized model
model.save_quantized (quant_path)
tokenizer.save_pretrained (quant_path)

# Load quantized model
model = AutoAWQForCausalLM.from_quantized (quant_path)

# AWQ typically preserves 95-98% of original quality
# vs 90-94% for naive 4-bit quantization
\`\`\`

---

## GPTQ (Post-Training Quantization)

GPTQ is another high-quality quantization method.

### Using GPTQ Models

\`\`\`python
# Install: pip install auto-gptq optimum

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name = "TheBloke/Llama-2-13B-GPTQ"

# Load pre-quantized GPTQ model
model = AutoGPTQForCausalLM.from_quantized(
    model_name,
    device="cuda:0",
    use_triton=True,  # Faster inference with Triton
    use_safetensors=True,
)

tokenizer = AutoTokenizer.from_pretrained (model_name)

# Use model
prompt = "Write a Python function to calculate factorial: "
inputs = tokenizer (prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode (outputs[0]))
\`\`\`

### Quantizing Your Own Model with GPTQ

\`\`\`python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

# Configure quantization
quantize_config = BaseQuantizeConfig(
    bits=4,  # 4-bit quantization
    group_size=128,
    desc_act=False,  # Activation order
)

# Load model
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantize_config=quantize_config
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")

# Prepare calibration data
calibration_data = [
    tokenizer (text) for text in [
        "Sample text 1...",
        "Sample text 2...",
        # Need ~128 samples
    ]
]

# Quantize
model.quantize (calibration_data)

# Save
model.save_quantized("./llama-2-13b-gptq")
\`\`\`

---

## Running Models Locally vs API

### Cost Comparison

**API (OpenAI GPT-3.5)**:
\`\`\`
Input: $0.50 per 1M tokens
Output: $1.50 per 1M tokens
100K requests/day × 1000 tokens avg = 100M tokens/day
Cost: $100/day = $3,000/month
\`\`\`

**Local (13B quantized model)**:
\`\`\`
Hardware: RTX 4090 (~$1,600)
Power: ~450W × $0.12/kWh × 24h = $1.30/day = $40/month
Amortized hardware: $1,600 / 24 months = $67/month
Total: ~$107/month

Break-even: ~1 month!
\`\`\`

### When to Run Locally

✅ **Good for Local**:
- High volume (>1M requests/month)
- Privacy-sensitive data
- Consistent workload
- Need full control
- Can manage infrastructure

❌ **Stick with API**:
- Low volume (<100K requests/month)
- Variable workload
- Need latest models
- Don't want infrastructure management
- Need guaranteed uptime

---

## Production Deployment

### Serving Quantized Models

\`\`\`python
# FastAPI server with quantized model
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import asyncio

app = FastAPI()

# Load model once at startup
llm = None

@app.on_event("startup")
async def load_model():
    global llm
    print("Loading quantized model...")
    llm = Llama(
        model_path="./models/llama-2-13b-q4_K_M.gguf",
        n_ctx=4096,
        n_gpu_layers=40,
        n_threads=8,
    )
    print("Model loaded!")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    time_taken: float

@app.post("/generate", response_model=GenerateResponse)
async def generate (request: GenerateRequest):
    if llm is None:
        raise HTTPException (status_code=503, detail="Model not loaded")
    
    import time
    start = time.time()
    
    # Run inference in thread pool (CPU-bound)
    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(
        None,
        lambda: llm(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    )
    
    elapsed = time.time() - start
    
    return GenerateResponse(
        text=output["choices"][0]["text"],
        tokens_generated=len (output["choices"][0]["text"].split()),
        time_taken=elapsed
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": llm is not None}

# Run with: uvicorn server:app --host 0.0.0.0 --port 8000
\`\`\`

### Docker Deployment

\`\`\`dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Install llama-cpp-python with GPU support
RUN pip install llama-cpp-python[server] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Copy model
COPY models/llama-2-13b-q4_K_M.gguf /models/model.gguf

# Expose port
EXPOSE 8000

# Run server
CMD ["python3", "-m", "llama_cpp.server", "--model", "/models/model.gguf", "--host", "0.0.0.0", "--port", "8000"]
\`\`\`

---

## Monitoring Quantized Models

\`\`\`python
import time
from dataclasses import dataclass
from typing import List

@dataclass
class InferenceMetrics:
    tokens_per_second: float
    memory_usage_gb: float
    gpu_utilization: float
    avg_latency_ms: float

class ModelMonitor:
    def __init__(self):
        self.request_times: List[float] = []
        self.token_counts: List[int] = []
    
    def record_inference (self, time_taken: float, tokens_generated: int):
        """Record inference metrics"""
        self.request_times.append (time_taken)
        self.token_counts.append (tokens_generated)
        
        # Keep only last 100 requests
        if len (self.request_times) > 100:
            self.request_times = self.request_times[-100:]
            self.token_counts = self.token_counts[-100:]
    
    def get_metrics (self) -> InferenceMetrics:
        """Calculate current metrics"""
        if not self.request_times:
            return None
        
        avg_time = sum (self.request_times) / len (self.request_times)
        total_tokens = sum (self.token_counts)
        total_time = sum (self.request_times)
        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
        
        return InferenceMetrics(
            tokens_per_second=tokens_per_sec,
            memory_usage_gb=self._get_gpu_memory(),
            gpu_utilization=self._get_gpu_utilization(),
            avg_latency_ms=avg_time * 1000
        )
    
    def _get_gpu_memory (self) -> float:
        """Get GPU memory usage"""
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0
    
    def _get_gpu_utilization (self) -> float:
        """Get GPU utilization percentage"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates (handle)
            return util.gpu
        except:
            return 0.0

# Usage
monitor = ModelMonitor()

# After each inference
monitor.record_inference (time_taken=1.2, tokens_generated=150)

# Get current metrics
metrics = monitor.get_metrics()
print(f"Tokens/sec: {metrics.tokens_per_second:.1f}")
print(f"Memory: {metrics.memory_usage_gb:.2f} GB")
print(f"GPU util: {metrics.gpu_utilization:.1f}%")
print(f"Latency: {metrics.avg_latency_ms:.0f}ms")
\`\`\`

---

## Best Practices

### 1. Choose Quantization Level Wisely
- **8-bit**: Production use, minimal quality loss
- **4-bit**: Cost-sensitive or consumer hardware
- **3-bit/2-bit**: Research only, significant quality loss

### 2. Benchmark on Your Data
- Don't trust general benchmarks
- Test on your specific use case
- Measure quality degradation

### 3. Use Calibration Data
- For GPTQ/AWQ, provide representative samples
- 100-500 samples usually sufficient
- Better calibration = better quality

### 4. Monitor Quality in Production
- Track user feedback
- Compare against unquantized baseline
- Alert on quality degradation

### 5. Consider Hybrid Approach
- Quantized models for most requests
- Full precision for critical requests
- Route based on complexity

---

## Summary

Quantization enables running large models efficiently:

- **8-bit quantization**: 75% size reduction, 1-2% quality loss
- **4-bit quantization**: 87.5% size reduction, 2-5% quality loss
- **2-4x faster inference**: Less computation required
- **Run locally**: 13B models on consumer hardware
- **Massive cost savings**: $3,000/month API → $100/month local

**Tools**:
- bitsandbytes: Easy 8-bit/4-bit quantization
- GGUF/llama.cpp: Optimized inference
- AWQ/GPTQ: High-quality quantization
- llama-cpp-python: Production serving

For high-volume applications, quantization can reduce costs by 90%+ while maintaining acceptable quality.

`,
  exercises: [
    {
      prompt:
        'Quantize a 7B model to 4-bit using bitsandbytes. Measure memory usage, inference speed, and quality vs FP16 on your test set.',
      solution: `Use bitsandbytes 4-bit config, load model, benchmark on test prompts. Expected: 75% memory reduction, 3-4x speed improvement, 2-4% quality degradation.`,
    },
    {
      prompt:
        'Convert a Hugging Face model to GGUF Q4_K_M format and serve it with llama-cpp-python. Compare API costs vs local hosting costs.',
      solution: `Use convert.py script, quantize to Q4_K_M, deploy FastAPI server. Calculate break-even point based on request volume. Expected: break-even at 1-2 months for high volume.`,
    },
    {
      prompt:
        'Build a production deployment that routes simple queries to 4-bit quantized local model and complex queries to GPT-4 API. Measure cost savings.',
      solution: `Implement complexity classifier, route accordingly, track costs. Expected: 60-80% cost reduction with <5% quality degradation.`,
    },
  ],
};
