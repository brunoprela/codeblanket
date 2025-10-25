/**
 * Image Generation Fundamentals Section
 * Module 8: Image Generation & Computer Vision
 */

export const imagegenerationfundamentalsSection = {
  id: 'image-generation-fundamentals',
  title: 'Image Generation Fundamentals',
  content: `# Image Generation Fundamentals

Master the foundations of text-to-image generation and understand how models like DALL-E, Midjourney, and Stable Diffusion work.

## Overview: The Revolution in Image Generation

Text-to-image generation has transformed from experimental research to production-ready tools in just a few years. Understanding these fundamentals is essential for building any image generation application.

### Why This Matters

- **Accessibility**: Anyone can now generate professional images
- **Speed**: Minutes instead of hours/days of manual creation
- **Iteration**: Rapidly experiment with concepts
- **Customization**: Fine-tune for specific styles and domains
- **Cost**: Often cheaper than hiring designers for certain tasks

### Real-World Applications

1. **Marketing & Advertising**: Product visualizations, ad creatives
2. **Game Development**: Concept art, textures, characters
3. **E-commerce**: Product variations, lifestyle images
4. **Content Creation**: Social media, blog illustrations
5. **Design**: Mood boards, prototyping, inspiration

## How Text-to-Image Generation Works

### The Core Process

\`\`\`
Text Prompt → Text Encoder → Diffusion Model → Generated Image
              ↓
         [Latent Space Representation]
              ↓
         [Iterative Denoising]
              ↓
         [VAE Decoder]
\`\`\`

### Key Components

#### 1. **Text Encoder** (CLIP)

Converts text into numerical representation:

\`\`\`python
# Conceptual example of how text encoding works
def encode_text_prompt (prompt: str) -> np.ndarray:
    """
    Convert text prompt to embedding vector.
    Real implementations use CLIP or similar models.
    """
    # Tokenize the text
    tokens = tokenizer.encode (prompt)  # "a cat wearing a hat"
    
    # Pass through transformer to get embeddings
    # This captures semantic meaning
    text_embedding = text_transformer (tokens)  # Shape: (77, 768)
    
    # The embedding captures concepts like:
    # - "cat" (animal features)
    # - "wearing" (spatial relationship)
    # - "hat" (object features)
    
    return text_embedding
\`\`\`

**CLIP (Contrastive Language-Image Pre-training)**:
- Trained on 400M image-text pairs
- Learns joint embedding space for images and text
- Can understand complex concepts and compositions

#### 2. **Latent Space**

Images are encoded into compressed latent representations:

\`\`\`python
# Why latent space matters
original_image_size = 512 * 512 * 3  # 786,432 pixels
latent_size = 64 * 64 * 4            # 16,384 latents

compression_ratio = original_image_size / latent_size  # ~48x smaller!

# This makes generation MUCH faster and cheaper
\`\`\`

**Benefits**:
- **Speed**: 48x smaller = much faster generation
- **Memory**: Can fit on consumer GPUs
- **Quality**: VAE learns meaningful compressed representations

#### 3. **Diffusion Process**

The magic behind image generation:

\`\`\`python
import numpy as np
from typing import Optional

def forward_diffusion(
    image: np.ndarray,
    timestep: int,
    noise_schedule: np.ndarray
) -> np.ndarray:
    """
    Forward diffusion: gradually add noise to image.
    This is used during training.
    """
    # Get noise level for this timestep
    noise_level = noise_schedule[timestep]
    
    # Add Gaussian noise
    noise = np.random.randn(*image.shape)
    noisy_image = (
        np.sqrt(1 - noise_level) * image +
        np.sqrt (noise_level) * noise
    )
    
    return noisy_image

def reverse_diffusion(
    noisy_latent: np.ndarray,
    text_embedding: np.ndarray,
    unet_model,
    timesteps: int = 50
) -> np.ndarray:
    """
    Reverse diffusion: gradually denoise to create image.
    This is the generation process.
    """
    latent = noisy_latent  # Start with pure noise
    
    for t in reversed (range (timesteps)):
        # Predict noise to remove
        predicted_noise = unet_model(
            latent, 
            timestep=t,
            text_embedding=text_embedding  # Guide with text
        )
        
        # Remove some noise
        latent = remove_noise (latent, predicted_noise, t)
        
        # Optionally show progress
        if t % 10 == 0:
            print(f"Step {timesteps - t}/{timesteps}")
    
    return latent

# Production example with real Stable Diffusion
from diffusers import StableDiffusionPipeline
import torch

def generate_image_basic(
    prompt: str,
    num_steps: int = 50,
    guidance_scale: float = 7.5
) -> np.ndarray:
    """
    Basic image generation with Stable Diffusion.
    """
    # Load model
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    
    # Generate
    with torch.autocast("cuda"):
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
        ).images[0]
    
    return np.array (image)

# Usage
image = generate_image_basic(
    "a professional photo of a cat wearing a tiny hat",
    num_steps=50,
    guidance_scale=7.5
)
\`\`\`

**Key Concepts**:
- **Forward Process**: Training - add noise progressively
- **Reverse Process**: Generation - remove noise progressively
- **Timesteps**: More steps = higher quality, but slower
- **Guidance Scale**: How strongly to follow the prompt

## Diffusion Models vs GANs

### GANs (Generative Adversarial Networks)

**How they work**:
- Generator creates images
- Discriminator judges if images are real or fake
- They compete, pushing each other to improve

\`\`\`python
# GAN training (conceptual)
def train_gan (generator, discriminator, real_images):
    for epoch in range (num_epochs):
        # Generate fake images
        noise = random_noise()
        fake_images = generator (noise)
        
        # Train discriminator
        real_scores = discriminator (real_images)  # Should be 1
        fake_scores = discriminator (fake_images)  # Should be 0
        d_loss = loss (real_scores, fake_scores)
        
        # Train generator
        fake_images = generator (noise)
        fake_scores = discriminator (fake_images)
        g_loss = loss (fake_scores, target=1)  # Want discriminator fooled
\`\`\`

**Problems with GANs**:
- **Mode collapse**: Generator makes same images
- **Training instability**: Hard to balance generator/discriminator
- **No gradual refinement**: Can't iteratively improve images
- **Limited diversity**: Struggle with variety

### Diffusion Models

**Advantages**:
- **Stable training**: No adversarial dynamics
- **High diversity**: Natural randomness in process
- **Controllable**: Can guide at each step
- **Gradual refinement**: Iterative improvement
- **Better quality**: State-of-the-art results

\`\`\`python
# Why diffusion is better for control
def generate_with_control(
    prompt: str,
    init_image: Optional[np.ndarray] = None,
    strength: float = 0.8
):
    """
    Diffusion allows fine control at each step.
    """
    if init_image is not None:
        # Start from real image, add some noise
        start_step = int((1 - strength) * num_steps)
        latent = encode_image (init_image)
        latent = add_noise (latent, level=start_step)
    else:
        # Start from pure noise
        latent = random_noise()
        start_step = 0
    
    # Denoise with guidance
    for t in range (start_step, num_steps):
        predicted_noise = model (latent, t, prompt)
        latent = remove_noise (latent, predicted_noise)
        
        # Can inject additional control at each step!
        if has_controlnet:
            latent = apply_control_signal (latent, control_image)
    
    return decode_latent (latent)
\`\`\`

## Model Architectures

### Stable Diffusion Architecture

\`\`\`
Text Prompt
    ↓
[CLIP Text Encoder]
    ↓
Text Embeddings (77 × 768)
    ↓
[UNet Denoiser] ←──┐
    ↓               │
Latent Space ──────┘ (iterative)
(64 × 64 × 4)
    ↓
[VAE Decoder]
    ↓
Final Image (512 × 512 × 3)
\`\`\`

**Component Details**:

1. **Text Encoder**: CLIP ViT-L/14
   - Vocabulary: 49,408 tokens
   - Max length: 77 tokens
   - Embedding dim: 768

2. **UNet**: 860M parameters
   - Cross-attention layers for text conditioning
   - Residual blocks
   - Self-attention for image coherence

3. **VAE**: Autoencoder
   - Encoder: 512×512 → 64×64×4
   - Decoder: 64×64×4 → 512×512
   - 8x spatial compression

### DALL-E 3 Architecture

DALL-E 3 uses a different approach:

\`\`\`python
# DALL-E 3's key improvements over SD
improvements = {
    "prompt_following": "Better understands complex prompts",
    "text_rendering": "Can write text in images (better)",
    "composition": "More accurate spatial relationships",
    "style": "More consistent artistic style",
    "details": "Finer details and textures",
}

# But it's closed source and API only
from openai import OpenAI

def generate_with_dalle3(prompt: str):
    """
    DALL-E 3 through OpenAI API.
    """
    client = OpenAI()
    
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",  # or "hd"
        n=1,
    )
    
    image_url = response.data[0].url
    return image_url
\`\`\`

### Midjourney Architecture

Midjourney is proprietary, but likely uses:
- Custom diffusion model
- Extensive fine-tuning on curated data
- Advanced prompt processing
- Multi-stage generation (upscale, refine)

## Generation Process Deep Dive

### Complete Generation Pipeline

\`\`\`python
import torch
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler
)
from typing import Optional, List

class ImageGenerator:
    """
    Production-ready image generation with full control.
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1",
        device: str = "cuda"
    ):
        self.device = device
        
        # Load pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,  # For production, keep enabled
        )
        self.pipe = self.pipe.to (device)
        
        # Enable memory optimizations
        self.pipe.enable_attention_slicing()
        
        # Optional: xformers for faster generation
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        scheduler: str = "dpm",
    ) -> List[Image.Image]:
        """
        Generate images with full control.
        
        Args:
            prompt: What to generate
            negative_prompt: What to avoid
            num_images: How many images
            steps: Quality (20-100, more = slower)
            guidance_scale: Prompt adherence (1-20, higher = stronger)
            width/height: Dimensions (multiples of 64)
            seed: For reproducibility
            scheduler: Noise schedule algorithm
        
        Returns:
            List of PIL Images
        """
        # Set scheduler
        if scheduler == "dpm":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
        elif scheduler == "euler_a":
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )
        
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator (device=self.device).manual_seed (seed)
        else:
            generator = None
        
        # Generate
        with torch.autocast("cuda"):
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            )
        
        return result.images
    
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[List[Image.Image]]:
        """
        Generate images for multiple prompts efficiently.
        """
        results = []
        for prompt in prompts:
            images = self.generate (prompt, **kwargs)
            results.append (images)
        return results

# Usage
generator = ImageGenerator()

# Single image
images = generator.generate(
    prompt="a serene mountain landscape at sunset, highly detailed",
    negative_prompt="blurry, low quality, distorted",
    steps=30,
    guidance_scale=7.5,
    seed=42,
)

images[0].save("mountain.png")

# Multiple images with variation
images = generator.generate(
    prompt="a cute robot reading a book",
    num_images=4,
    steps=25,
    guidance_scale=8.0,
)

for i, img in enumerate (images):
    img.save (f"robot_{i}.png")
\`\`\`

### Understanding Parameters

#### Number of Steps

\`\`\`python
# Steps impact quality and speed
step_configs = {
    "fast": {
        "steps": 20,
        "quality": "acceptable",
        "time": "~5 seconds",
        "use_case": "previews, iteration"
    },
    "balanced": {
        "steps": 30,
        "quality": "good",
        "time": "~7 seconds",
        "use_case": "most production use"
    },
    "high_quality": {
        "steps": 50,
        "quality": "excellent",
        "time": "~12 seconds",
        "use_case": "final outputs, detailed work"
    },
    "maximum": {
        "steps": 100,
        "quality": "marginal improvement over 50",
        "time": "~25 seconds",
        "use_case": "rarely needed"
    }
}

# Diminishing returns after ~50 steps
\`\`\`

#### Guidance Scale

\`\`\`python
# How guidance scale affects results
def demonstrate_guidance_scale():
    """
    Low scale: Creative, varied, may deviate from prompt
    High scale: Rigid, follows prompt exactly, less variation
    """
    prompt = "a photo of a red apple on a table"
    
    # Too low (1.0-3.0): Ignores prompt
    # - May generate: blue apple, apple floating, not an apple at all
    
    # Low (3.0-5.0): Loose interpretation
    # - Creative variations, some may not match prompt well
    
    # Ideal (7.0-9.0): Balanced
    # - Follows prompt while maintaining creativity
    # - Most production use cases
    
    # High (12.0-15.0): Strict adherence
    # - Exactly what you asked for
    # - Less artistic variation
    # - May look less natural
    
    # Too high (>20): Oversaturated
    # - Burnt colors, artifacts
    # - Over-emphasized features

# Guidance scale examples
scales_to_try = {
    "creative": 5.0,      # Artistic liberty
    "balanced": 7.5,      # Default, good for most
    "precise": 10.0,      # Technical accuracy
    "photorealistic": 12.0,  # Exact details
}
\`\`\`

## Quality Assessment

### Objective Metrics

\`\`\`python
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from PIL import Image
import numpy as np

class ImageQualityMetrics:
    """
    Measure generated image quality.
    """
    
    def __init__(self):
        self.fid = FrechetInceptionDistance (feature=2048)
        self.inception = InceptionScore()
    
    def calculate_fid(
        self,
        real_images: List[Image.Image],
        generated_images: List[Image.Image]
    ) -> float:
        """
        FID: Lower is better (measures similarity to real images).
        < 10: Excellent
        10-20: Good
        20-50: Acceptable
        > 50: Poor
        """
        # Convert to tensors
        real_tensors = self._images_to_tensors (real_images)
        gen_tensors = self._images_to_tensors (generated_images)
        
        # Update FID with real images
        self.fid.update (real_tensors, real=True)
        
        # Update FID with generated images
        self.fid.update (gen_tensors, real=False)
        
        # Compute score
        score = self.fid.compute()
        return float (score)
    
    def calculate_inception_score(
        self,
        images: List[Image.Image]
    ) -> tuple[float, float]:
        """
        Inception Score: Higher is better (measures quality and diversity).
        > 8: Excellent
        5-8: Good
        < 5: Poor
        
        Returns (mean, std)
        """
        tensors = self._images_to_tensors (images)
        self.inception.update (tensors)
        mean, std = self.inception.compute()
        return float (mean), float (std)
    
    def _images_to_tensors(
        self,
        images: List[Image.Image]
    ) -> torch.Tensor:
        """Convert PIL images to torch tensors."""
        arrays = [np.array (img) for img in images]
        tensors = torch.tensor (np.stack (arrays)).permute(0, 3, 1, 2)
        return tensors.float() / 255.0
\`\`\`

### Subjective Quality Factors

\`\`\`python
quality_checklist = {
    "composition": [
        "Proper framing and balance",
        "Subject correctly positioned",
        "Clear focal point",
    ],
    "details": [
        "Sharp, clear features",
        "Consistent texture quality",
        "Appropriate level of detail",
    ],
    "coherence": [
        "Physically plausible",
        "Consistent lighting",
        "Proper shadows and reflections",
    ],
    "prompt_adherence": [
        "All elements from prompt present",
        "Correct relationships between objects",
        "Proper colors and materials",
    ],
    "artifacts": [
        "No weird distortions",
        "No duplicate elements",
        "Clean edges and boundaries",
    ],
    "style": [
        "Consistent artistic style throughout",
        "Appropriate for the subject",
        "Professional appearance",
    ]
}
\`\`\`

## Common Models Comparison

### Model Selection Guide

\`\`\`python
from dataclasses import dataclass
from typing import List

@dataclass
class ModelProfile:
    name: str
    strengths: List[str]
    weaknesses: List[str]
    speed: str  # fast/medium/slow
    quality: str  # good/great/excellent
    cost: str  # free/cheap/expensive
    license: str
    best_for: List[str]

models = {
    "stable_diffusion_2_1": ModelProfile(
        name="Stable Diffusion 2.1",
        strengths=[
            "Open source and free",
            "Run locally on consumer GPU",
            "Highly customizable",
            "Large community and extensions",
            "Can fine-tune for specific use cases"
        ],
        weaknesses=[
            "Worse at text rendering",
            "Sometimes struggles with hands",
            "Requires more prompt engineering",
            "Base model not as good as DALL-E 3"
        ],
        speed="fast",
        quality="good",
        cost="free",
        license="CreativeML Open RAIL-M",
        best_for=[
            "Prototyping and experimentation",
            "High-volume generation",
            "Custom fine-tuning",
            "When you need full control"
        ]
    ),
    
    "stable_diffusion_xl": ModelProfile(
        name="Stable Diffusion XL",
        strengths=[
            "Higher resolution (1024×1024)",
            "Better composition and details",
            "Improved text rendering",
            "Open source",
            "Better at photorealism"
        ],
        weaknesses=[
            "Slower than SD 2.1",
            "Requires more VRAM",
            "Still not as good as DALL-E 3 at text"
        ],
        speed="medium",
        quality="great",
        cost="free",
        license="CreativeML Open RAIL++-M",
        best_for=[
            "High-quality outputs",
            "Photorealistic images",
            "When you have GPU resources"
        ]
    ),
    
    "dalle_3": ModelProfile(
        name="DALL-E 3",
        strengths=[
            "Best prompt following",
            "Excellent text rendering",
            "Consistent style",
            "No setup required (API)",
            "Best overall quality"
        ],
        weaknesses=[
            "Costs per image",
            "API only (no local)",
            "Less control than SD",
            "Can't fine-tune",
            "Content restrictions"
        ],
        speed="medium",
        quality="excellent",
        cost="$0.04-$0.08 per image",
        license="Proprietary",
        best_for=[
            "Production applications",
            "When quality matters most",
            "Commercial use",
            "Marketing materials"
        ]
    ),
    
    "midjourney": ModelProfile(
        name="Midjourney",
        strengths=[
            "Stunning artistic results",
            "Great for concept art",
            "Consistent style",
            "Active community",
            "Regular updates"
        ],
        weaknesses=[
            "Discord-based interface",
            "No API (unofficial ones exist)",
            "Less programmatic control",
            "Subscription required"
        ],
        speed="medium",
        quality="excellent",
        cost="$10-$60/month",
        license="Proprietary",
        best_for=[
            "Artistic and creative work",
            "Concept art",
            "Illustration",
            "When style matters most"
        ]
    )
}

def choose_model (requirements: dict) -> str:
    """
    Help choose the right model based on requirements.
    """
    if requirements.get("budget") == "free":
        if requirements.get("quality") == "high":
            return "stable_diffusion_xl"
        return "stable_diffusion_2_1"
    
    if requirements.get("text_in_images"):
        return "dalle_3"
    
    if requirements.get("artistic_style"):
        return "midjourney"
    
    if requirements.get("prompt_following") == "critical":
        return "dalle_3"
    
    if requirements.get("volume") == "high":
        return "stable_diffusion_2_1"  # Local, no per-image cost
    
    return "dalle_3"  # Default for production
\`\`\`

## Use Cases and Applications

### Real-World Examples

\`\`\`python
use_cases = {
    "marketing": {
        "description": "Generate ad creatives and product images",
        "example_prompts": [
            "product photography of a red sneaker on white background, professional lighting, high resolution",
            "lifestyle photo of someone using a laptop in a modern coffee shop, natural lighting",
            "social media banner with vibrant colors, modern design, 1200x628"
        ],
        "recommended_model": "dalle_3",
        "settings": {
            "steps": 50,
            "guidance_scale": 8.0,
            "size": "1024x1024"
        }
    },
    
    "game_development": {
        "description": "Concept art, textures, characters",
        "example_prompts": [
            "fantasy character concept art, warrior with glowing sword, detailed armor",
            "seamless stone texture, 4k, tileable, pbr ready",
            "isometric fantasy village, colorful, mobile game style"
        ],
        "recommended_model": "stable_diffusion_xl",
        "settings": {
            "steps": 40,
            "guidance_scale": 7.5,
            "size": "1024x1024"
        }
    },
    
    "ecommerce": {
        "description": "Product variations and lifestyle images",
        "example_prompts": [
            "white t-shirt on wooden table with plants in background, natural lighting",
            "coffee mug in cozy kitchen setting, morning light, steam rising",
            "same product in different colors: red, blue, green, yellow"
        ],
        "recommended_model": "stable_diffusion_xl",
        "settings": {
            "steps": 30,
            "guidance_scale": 7.0,
            "size": "1024x1024"
        }
    },
    
    "content_creation": {
        "description": "Blog images, social media, illustrations",
        "example_prompts": [
            "minimalist illustration of productivity, clean lines, pastel colors",
            "blog header image about AI technology, futuristic, professional",
            "instagram story template, modern design, mobile optimized"
        ],
        "recommended_model": "dalle_3",
        "settings": {
            "steps": 30,
            "guidance_scale": 7.5,
            "size": "1024x1024"
        }
    }
}
\`\`\`

## Getting Started Guide

### Setup for Stable Diffusion

\`\`\`python
# Installation
"""
# Create environment
conda create -n image-gen python=3.10
conda activate image-gen

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install diffusers and dependencies
pip install diffusers transformers accelerate safetensors
pip install invisible-watermark # For SDXL

# Optional but recommended
pip install xformers  # Faster generation
"""

# First generation script
from diffusers import StableDiffusionPipeline
import torch

def setup_and_generate():
    """
    Complete setup and first image generation.
    """
    # Download model (first time only, ~4GB)
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    
    # Move to GPU
    pipe = pipe.to("cuda")
    
    # Generate your first image!
    prompt = "a photograph of an astronaut riding a horse on mars"
    
    image = pipe(
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]
    
    image.save("first_generation.png")
    print("Image saved!")
    
    return image

if __name__ == "__main__":
    setup_and_generate()
\`\`\`

### Common Issues and Solutions

\`\`\`python
troubleshooting = {
    "out_of_memory": {
        "error": "CUDA out of memory",
        "solutions": [
            "Enable attention slicing: pipe.enable_attention_slicing()",
            "Use smaller image size (512x512 instead of 1024x1024)",
            "Reduce batch size (num_images_per_prompt=1)",
            "Use float16: torch_dtype=torch.float16"
        ]
    },
    
    "slow_generation": {
        "error": "Generation takes too long",
        "solutions": [
            "Install xformers: pip install xformers",
            "Enable xformers: pipe.enable_xformers_memory_efficient_attention()",
            "Reduce steps (30 instead of 50)",
            "Use DPM++ scheduler (faster, same quality)"
        ]
    },
    
    "poor_quality": {
        "error": "Images don't look good",
        "solutions": [
            "Increase steps (50 instead of 20)",
            "Adjust guidance scale (try 7-9)",
            "Improve prompt with details",
            "Add negative prompt",
            "Use SDXL instead of SD 2.1"
        ]
    },
    
    "not_following_prompt": {
        "error": "Image doesn't match prompt",
        "solutions": [
            "Increase guidance scale to 10-12",
            "Be more specific in prompt",
            "Use weight syntax: (red apple:1.5)",
            "Add negative prompt for unwanted elements",
            "Try DALL-E 3 for better prompt following"
        ]
    }
}
\`\`\`

## Next Steps

Now that you understand the fundamentals, you're ready to:

1. **Practice prompt engineering** - Learn how to write effective prompts
2. **Explore different models** - Try SD, SDXL, DALL-E 3
3. **Learn img2img** - Transform existing images
4. **Study ControlNet** - Precise control over generation
5. **Master inpainting** - Edit parts of images
6. **Build applications** - Create production systems

The next section covers DALL-E 3 API integration for production use.

## Key Takeaways

- Diffusion models are the current state-of-the-art for image generation
- They work by iteratively removing noise, guided by text embeddings
- Stable Diffusion is open source and free, DALL-E 3 is proprietary but higher quality
- Key parameters: steps (quality), guidance scale (prompt adherence), size
- Generation happens in compressed latent space for efficiency
- Different models excel at different tasks - choose based on your needs
- Quality assessment combines objective metrics (FID, IS) and subjective evaluation
`,
};
