/**
 * Replicate & Model Hosting Section
 * Module 8: Image Generation & Computer Vision
 */

export const replicatemodelhostingSection = {
  id: 'replicate-model-hosting',
  title: 'Replicate & Model Hosting',
  content: `# Replicate & Model Hosting

Master cloud-based image generation with Replicate and other hosting platforms.

## Overview: Cloud vs. Local

Running models in the cloud offers key advantages:
- **No GPU required**: Run on any device
- **Scalability**: Handle traffic spikes
- **Variety**: Access hundreds of models instantly
- **Maintenance-free**: No setup or updates needed
- **Pay-per-use**: Cost-effective for low/variable usage

### When to Use Cloud Hosting

\`\`\`python
cloud_vs_local = {
    "use_cloud_when": [
        "No GPU available",
        "Variable/unpredictable traffic",
        "Want access to many models",
        "Getting started/prototyping",
        "Don't want maintenance",
        "Geographic distribution needed"
    ],
    
    "use_local_when": [
        "High volume (1000s of images/day)",
        "Need maximum privacy",
        "Want full control",
        "Have GPU hardware",
        "Cost sensitive at scale",
        "Custom model fine-tuning"
    ],
    
    "hybrid_approach": [
        "Local for bulk generation",
        "Cloud for variety/overflow",
        "Cloud for user-facing",
        "Local for internal"
    ]
}
\`\`\`

## Replicate Platform

### What is Replicate?

Replicate is the leading platform for running AI models:
- **1000s of models**: SD, SDXL, DALL-E, specialized models
- **Simple API**: One interface for all models
- **Fast**: Optimized infrastructure
- **Transparent pricing**: Pay per second of compute
- **Version control**: Track model versions

### Getting Started

\`\`\`python
# Installation
"""
pip install replicate
"""

import replicate
import os
from typing import Optional, Union, List
from PIL import Image
import requests
from io import BytesIO

class ReplicateClient:
    """
    Client for Replicate API.
    """
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize client.
        
        Set REPLICATE_API_TOKEN environment variable or pass token.
        Get token from: https://replicate.com/account/api-tokens
        """
        if api_token:
            os.environ["REPLICATE_API_TOKEN"] = api_token
        
        self.client = replicate
    
    def generate_sdxl(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_outputs: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate with SDXL on Replicate.
        """
        output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_outputs": num_outputs,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed if seed is not None else -1
            }
        )
        
        # Download images
        images = []
        for url in output:
            response = requests.get (url)
            img = Image.open(BytesIO(response.content))
            images.append (img)
        
        return images
    
    def img2img(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        strength: float = 0.5,
        **kwargs
    ) -> List[Image.Image]:
        """
        Image-to-image with SDXL.
        """
        # Convert image to URL (upload to Replicate or use existing URL)
        if isinstance (image, Image.Image):
            # Would need to upload - simplified here
            image_url = "https://example.com/image.png"
        else:
            image_url = image
        
        output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "image": image_url,
                "prompt": prompt,
                "strength": strength,
                **kwargs
            }
        )
        
        images = []
        for url in output:
            response = requests.get (url)
            img = Image.open(BytesIO(response.content))
            images.append (img)
        
        return images
    
    def upscale(
        self,
        image_url: str,
        scale: int = 4
    ) -> Image.Image:
        """
        Upscale image with Real-ESRGAN.
        """
        output = replicate.run(
            "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b",
            input={
                "image": image_url,
                "scale": scale
            }
        )
        
        response = requests.get (output)
        return Image.open(BytesIO(response.content))
    
    def run_custom_model(
        self,
        model_version: str,
        inputs: dict
    ) -> any:
        """
        Run any model on Replicate.
        
        Args:
            model_version: "owner/model:version_id"
            inputs: Model-specific inputs
        """
        return replicate.run (model_version, input=inputs)

# Usage
client = ReplicateClient()

# Generate with SDXL
images = client.generate_sdxl(
    prompt="a serene mountain landscape, oil painting style",
    negative_prompt="blurry, low quality",
    width=1024,
    height=768,
    num_outputs=2
)

for i, img in enumerate (images):
    img.save (f"sdxl_output_{i}.png")

# Upscale
small_image_url = "https://example.com/small.jpg"
upscaled = client.upscale (small_image_url, scale=4)
upscaled.save("upscaled.png")
\`\`\`

## Popular Models on Replicate

\`\`\`python
replicate_models = {
    "sdxl": {
        "id": "stability-ai/sdxl",
        "description": "Stable Diffusion XL - best quality",
        "use_case": "General high-quality generation",
        "cost": "~$0.01 per image"
    },
    
    "sdxl_lightning": {
        "id": "bytedance/sdxl-lightning-4step",
        "description": "Ultra-fast SDXL (4 steps)",
        "use_case": "When speed matters",
        "cost": "~$0.003 per image"
    },
    
    "real_esrgan": {
        "id": "nightmareai/real-esrgan",
        "description": "Image upscaling",
        "use_case": "Enhance resolution",
        "cost": "~$0.005 per image"
    },
    
    "controlnet": {
        "id": "jagilley/controlnet-*",
        "description": "Various ControlNet models",
        "use_case": "Precise structural control",
        "cost": "~$0.01 per image"
    },
    
    "gfpgan": {
        "id": "tencentarc/gfpgan",
        "description": "Face restoration",
        "use_case": "Fix/enhance faces",
        "cost": "~$0.005 per image"
    },
    
    "rembg": {
        "id": "cjwbw/rembg",
        "description": "Background removal",
        "use_case": "Remove backgrounds",
        "cost": "~$0.002 per image"
    }
}

# Example: Chain multiple models
def process_portrait_pipeline (image_url: str):
    """
    Complete portrait processing pipeline.
    """
    client = ReplicateClient()
    
    # 1. Remove background
    no_bg = client.run_custom_model(
        "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
        {"image": image_url}
    )
    
    # 2. Enhance face
    enhanced = client.run_custom_model(
        "tencentarc/gfpgan:9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3",
        {"img": no_bg, "version": "v1.4", "scale": 2}
    )
    
    # 3. Upscale
    final = client.upscale (enhanced, scale=2)
    
    return final
\`\`\`

## Async Generation

\`\`\`python
class AsyncReplicateClient:
    """
    Async Replicate client for better performance.
    """
    
    def __init__(self, api_token: Optional[str] = None):
        if api_token:
            os.environ["REPLICATE_API_TOKEN"] = api_token
    
    def create_prediction(
        self,
        model_version: str,
        inputs: dict
    ) -> str:
        """
        Start async prediction.
        
        Returns:
            Prediction ID
        """
        import replicate
        
        prediction = replicate.predictions.create(
            version=model_version,
            input=inputs
        )
        
        return prediction.id
    
    def get_prediction (self, prediction_id: str) -> dict:
        """
        Check prediction status.
        
        Returns:
            {
                "status": "starting" | "processing" | "succeeded" | "failed",
                "output": [...] if succeeded,
                "error": "..." if failed
            }
        """
        import replicate
        
        prediction = replicate.predictions.get (prediction_id)
        
        return {
            "status": prediction.status,
            "output": prediction.output if prediction.status == "succeeded" else None,
            "error": prediction.error if prediction.status == "failed" else None
        }
    
    def wait_for_prediction(
        self,
        prediction_id: str,
        check_interval: float = 1.0
    ) -> any:
        """
        Wait for prediction to complete.
        """
        import time
        
        while True:
            result = self.get_prediction (prediction_id)
            
            if result["status"] == "succeeded":
                return result["output"]
            elif result["status"] == "failed":
                raise Exception (f"Prediction failed: {result['error']}")
            
            time.sleep (check_interval)
    
    def batch_generate(
        self,
        model_version: str,
        inputs_list: List[dict]
    ) -> List[any]:
        """
        Generate multiple images in parallel.
        """
        # Start all predictions
        prediction_ids = []
        for inputs in inputs_list:
            pred_id = self.create_prediction (model_version, inputs)
            prediction_ids.append (pred_id)
        
        # Wait for all
        results = []
        for pred_id in prediction_ids:
            output = self.wait_for_prediction (pred_id)
            results.append (output)
        
        return results

# Usage
async_client = AsyncReplicateClient()

# Batch generate
inputs_batch = [
    {"prompt": "a cat", "width": 512, "height": 512},
    {"prompt": "a dog", "width": 512, "height": 512},
    {"prompt": "a bird", "width": 512, "height": 512},
]

results = async_client.batch_generate(
    model_version="stability-ai/sdxl:...",
    inputs_list=inputs_batch
)
\`\`\`

## Cost Management

\`\`\`python
class CostTracker:
    """
    Track and manage Replicate costs.
    """
    
    def __init__(self):
        self.costs = []
    
    def estimate_cost(
        self,
        model_type: str,
        num_images: int,
        resolution: str = "1024x1024"
    ) -> dict:
        """
        Estimate generation costs.
        """
        # Approximate costs (check Replicate for current pricing)
        cost_per_image = {
            "sdxl": {"1024x1024": 0.01, "512x512": 0.005},
            "sd21": {"512x512": 0.003},
            "sdxl_lightning": {"1024x1024": 0.003},
            "upscale": {"any": 0.005},
            "face_enhance": {"any": 0.005}
        }
        
        base_cost = cost_per_image.get (model_type, {}).get (resolution, 0.01)
        total = base_cost * num_images
        
        return {
            "per_image": base_cost,
            "num_images": num_images,
            "total": total,
            "breakdown": f"\${base_cost} × {num_images} = \${total:.3f}"
        }
    
    def log_generation(
    self,
    model: str,
    inputs: dict,
    estimated_cost: float
):
"""Log generation for tracking."""
self.costs.append({
    "model": model,
    "inputs": inputs,
    "estimated_cost": estimated_cost,
    "timestamp": datetime.now()
})
    
    def get_total_cost (self) -> float:
"""Get total estimated costs."""
return sum (c["estimated_cost"] for c in self.costs)
    
    def cost_report (self) -> dict:
"""Generate cost report."""
return {
    "total_generations": len (self.costs),
    "total_cost": self.get_total_cost(),
    "by_model": self._costs_by_model()
}
    
    def _costs_by_model (self) -> dict:
"""Break down costs by model."""
by_model = {}
for cost in self.costs:
    model = cost["model"]
if model not in by_model:
by_model[model] = { "count": 0, "cost": 0 }
by_model[model]["count"] += 1
by_model[model]["cost"] += cost["estimated_cost"]
return by_model

# Usage
tracker = CostTracker()

# Estimate before generating
estimate = tracker.estimate_cost("sdxl", num_images = 100, resolution = "1024x1024")
print(f"Estimated cost: \\$\{estimate['total']:.2f}")

# Log generation
tracker.log_generation("sdxl", { "prompt": "..." }, estimated_cost = 0.01)

# Get report
report = tracker.cost_report()
print(f"Total cost: \\$\{report['total_cost']:.2f}")
\`\`\`

## Alternative Platforms

### HuggingFace Inference API

\`\`\`python
class HuggingFaceInference:
    """
    Use HuggingFace Inference API.
    """
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.api_url = "https://api-inference.huggingface.co/models"
    
    def generate(
        self,
        model_id: str,
        prompt: str
    ) -> Image.Image:
        """
        Generate image using HF Inference API.
        
        Models:
        - stabilityai/stable-diffusion-2-1
        - stabilityai/stable-diffusion-xl-base-1.0
        - runwayml/stable-diffusion-v1-5
        """
        import requests
        
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        response = requests.post(
            f"{self.api_url}/{model_id}",
            headers=headers,
            json={"inputs": prompt}
        )
        
        return Image.open(BytesIO(response.content))

# Usage
hf = HuggingFaceInference (api_token="hf_...")

image = hf.generate(
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    prompt="a beautiful landscape"
)
\`\`\`

### Modal Labs

\`\`\`python
"""
Modal - Serverless GPU compute

# Install
pip install modal

# Example Modal function for SDXL
import modal

stub = modal.Stub("sdxl-generator")

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "diffusers",
        "torch",
        "transformers"
    ),
    gpu="A10G",
    timeout=300
)
def generate_sdxl (prompt: str):
    from diffusers import StableDiffusionXLPipeline
    import torch
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    
    image = pipe (prompt).images[0]
    return image

# Deploy
# modal deploy sdxl_app.py

# Use
with stub.run():
    image = generate_sdxl.remote("a mountain landscape")
"""
\`\`\`

## Production Best Practices

\`\`\`python
production_practices = {
    "error_handling": [
        "Implement retry logic with exponential backoff",
        "Handle rate limits gracefully",
        "Validate inputs before API calls",
        "Log all API interactions"
    ],
    
    "performance": [
        "Use async for multiple images",
        "Batch when possible",
        "Cache results to avoid duplicates",
        "Monitor response times"
    ],
    
    "cost_control": [
        "Set budget limits",
        "Monitor usage daily",
        "Use cheaper models for dev/testing",
        "Estimate costs before large batches",
        "Consider local for high volume"
    ],
    
    "reliability": [
        "Have fallback to different provider",
        "Queue failed requests for retry",
        "Monitor service status",
        "Test regularly"
    ]
}

class ProductionReplicateClient:
    """
    Production-ready Replicate client.
    """
    
    def __init__(self, api_token: str):
        self.client = ReplicateClient (api_token)
        self.tracker = CostTracker()
        self.max_retries = 3
    
    def generate_with_retry(
        self,
        model_version: str,
        inputs: dict,
        estimate_cost: float = 0.01
    ) -> any:
        """Generate with retry and cost tracking."""
        import time
        
        # Track cost
        self.tracker.log_generation (model_version, inputs, estimate_cost)
        
        # Retry logic
        for attempt in range (self.max_retries):
            try:
                return self.client.run_custom_model (model_version, inputs)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep (wait)
                else:
                    raise
    
    def check_budget (self, limit: float) -> bool:
        """Check if within budget."""
        total = self.tracker.get_total_cost()
        if total >= limit:
            raise Exception (f"Budget limit reached: \${total:.2f} >= \${ limit:.2f } ")
return True
\`\`\`

## Key Takeaways

- **Replicate**: Leading platform for cloud AI models
- **Pay-per-use**: Cost-effective for variable load
- **No GPU needed**: Run on any device
- **1000s of models**: Instant access to variety
- **Simple API**: Consistent interface
- **Async support**: Better performance
- **Cost tracking**: Essential for production
- **Alternatives**: HuggingFace, Modal Labs, Banana
- **Use cloud for**: Prototyping, variable load, no GPU
- **Use local for**: High volume, privacy, full control
`,
};
