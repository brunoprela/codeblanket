/**
 * Stable Diffusion Section
 * Module 8: Image Generation & Computer Vision
 */

export const stablediffusionSection = {
  id: 'stable-diffusion',
  title: 'Stable Diffusion',
  content: `# Stable Diffusion

Master Stable Diffusion for local, customizable, production-grade image generation.

## Overview: The Power of Open Source

Stable Diffusion revolutionized AI image generation by being **open source, free, and runnable locally**. This means complete control, no API costs, and unlimited customization.

### Why Stable Diffusion Matters

- **Free**: No per-image costs
- **Local**: Run on your own hardware
- **Open Source**: Full code access
- **Customizable**: Fine-tune for your needs
- **Private**: No data leaves your machine
- **Community**: Thousands of models and extensions

### When to Use Stable Diffusion

\`\`\`python
use_stable_diffusion_when = {
    "high_volume": "Generate thousands of images without cost",
    "customization": "Fine-tune for specific styles or subjects",
    "privacy": "Keep data local and private",
    "experimentation": "Rapid iteration without API costs",
    "offline": "No internet connection needed",
    "specific_style": "Use community models trained on specific art styles",
    "control": "Need fine-grained control over generation process",
    "integration": "Deep integration with your pipeline"
}
\`\`\`

## Getting Started

### Hardware Requirements

\`\`\`python
hardware_guide = {
    "minimum": {
        "gpu": "NVIDIA GPU with 6GB VRAM",
        "ram": "16GB system RAM",
        "storage": "20GB for models",
        "examples": ["RTX 3060", "RTX 2060 Super"],
        "quality": "512×512, slower generation",
        "suitable_for": "Personal experimentation"
    },
    
    "recommended": {
        "gpu": "NVIDIA GPU with 8-12GB VRAM",
        "ram": "32GB system RAM",
        "storage": "50GB for multiple models",
        "examples": ["RTX 3080", "RTX 4070"],
        "quality": "512×512 fast, 768×768 good speed",
        "suitable_for": "Production use, development"
    },
    
    "optimal": {
        "gpu": "NVIDIA GPU with 16-24GB VRAM",
        "ram": "64GB system RAM",
        "storage": "100GB+ for model collection",
        "examples": ["RTX 4090", "RTX A5000"],
        "quality": "1024×1024 fast, batch generation",
        "suitable_for": "Professional production, fine-tuning"
    }
}

def check_system():
    """Check if system is suitable for Stable Diffusion."""
    import torch
    
    if not torch.cuda.is_available():
        return {"status": "error", "message": "No CUDA GPU detected"}
    
    device = torch.cuda.get_device_properties(0)
    vram_gb = device.total_memory / (1024**3)
    
    if vram_gb < 6:
        return {"status": "insufficient", "vram": vram_gb}
    elif vram_gb < 10:
        return {"status": "minimum", "vram": vram_gb, "max_resolution": 512}
    elif vram_gb < 16:
        return {"status": "recommended", "vram": vram_gb, "max_resolution": 768}
    else:
        return {"status": "optimal", "vram": vram_gb, "max_resolution": 1024}
\`\`\`

### Installation & Setup

\`\`\`python
"""
Step-by-step setup for Stable Diffusion

1. Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

2. Install diffusers and dependencies
pip install diffusers transformers accelerate safetensors
pip install invisible-watermark  # For SDXL

3. Optional: Performance optimizations
pip install xformers  # Faster attention, less VRAM
pip install triton   # Additional optimizations

4. Download models (happens automatically on first use)
"""

from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler
)
import torch
from PIL import Image
from typing import Optional, List, Union

class StableDiffusionGenerator:
    """
    Complete Stable Diffusion implementation.
    """
    
    # Popular model IDs
    MODELS = {
        "sd_2_1": "stabilityai/stable-diffusion-2-1",
        "sd_2_1_base": "stabilityai/stable-diffusion-2-1-base",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "sdxl_turbo": "stabilityai/sdxl-turbo",
    }
    
    def __init__(
        self,
        model_name: str = "sd_2_1",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize Stable Diffusion pipeline.
        
        Args:
            model_name: Model to use (see MODELS dict)
            device: Device to run on
            dtype: Precision (float16 for speed, float32 for quality)
        """
        self.device = device
        self.dtype = dtype
        
        # Load model
        model_id = self.MODELS.get (model_name, model_name)
        
        print(f"Loading {model_id}...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,  # Disable for speed (be responsible!)
        )
        
        self.pipe = self.pipe.to (device)
        
        # Memory optimizations
        self.pipe.enable_attention_slicing()
        
        # Try to enable xformers (faster, less VRAM)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✓ xformers enabled")
        except:
            print("✗ xformers not available")
        
        # Set default scheduler (can change later)
        self.set_scheduler("euler_a")
        
        print("✓ Model loaded and ready")
    
    def set_scheduler (self, scheduler: str):
        """
        Change the sampling scheduler.
        
        Options:
        - 'euler_a': Fast, good quality (recommended)
        - 'dpm': Slower, higher quality
        - 'ddim': Original, stable
        - 'pndm': Faster, slightly lower quality
        """
        schedulers = {
            "euler_a": EulerAncestralDiscreteScheduler,
            "dpm": DPMSolverMultistepScheduler,
        }
        
        if scheduler in schedulers:
            self.pipe.scheduler = schedulers[scheduler].from_config(
                self.pipe.scheduler.config
            )
            print(f"✓ Scheduler set to {scheduler}")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate images.
        
        Args:
            prompt: What to generate
            negative_prompt: What to avoid
            num_images: Number of images to generate
            steps: Sampling steps (20-50 typical)
            guidance_scale: How closely to follow prompt (7-9 typical)
            width/height: Image size (must be multiples of 64)
            seed: Random seed for reproducibility
        
        Returns:
            List of PIL Images
        """
        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator (device=self.device).manual_seed (seed)
        
        # Generate
        with torch.autocast (self.device):
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
    
    def generate_grid(
        self,
        prompts: List[str],
        **kwargs
    ) -> Image.Image:
        """
        Generate multiple prompts and create a grid.
        """
        all_images = []
        
        for prompt in prompts:
            images = self.generate (prompt, num_images=1, **kwargs)
            all_images.extend (images)
        
        # Create grid
        return self._create_image_grid (all_images)
    
    def _create_image_grid(
        self,
        images: List[Image.Image],
        cols: int = 4
    ) -> Image.Image:
        """Create a grid from list of images."""
        if not images:
            return None
        
        rows = (len (images) + cols - 1) // cols
        w, h = images[0].size
        
        grid = Image.new('RGB', (cols * w, rows * h))
        
        for i, img in enumerate (images):
            x = (i % cols) * w
            y = (i // cols) * h
            grid.paste (img, (x, y))
        
        return grid

# Basic usage
generator = StableDiffusionGenerator (model_name="sd_2_1")

# Generate single image
images = generator.generate(
    prompt="a serene mountain landscape, oil painting style",
    negative_prompt="blurry, low quality, distorted",
    steps=30,
    guidance_scale=7.5,
    seed=42
)

images[0].save("mountain.png")

# Generate multiple variations
images = generator.generate(
    prompt="a cute robot assistant",
    num_images=4,
    steps=25,
    seed=123
)

for i, img in enumerate (images):
    img.save (f"robot_{i}.png")
\`\`\`

## Sampling Algorithms (Schedulers)

### Understanding Schedulers

\`\`\`python
scheduler_comparison = {
    "euler_a": {
        "name": "Euler Ancestral",
        "speed": "★★★★☆",
        "quality": "★★★★☆",
        "stability": "★★★☆☆",
        "characteristics": [
            "Good balance of speed and quality",
            "Some randomness even with same seed",
            "Popular community choice",
            "Works well at lower step counts"
        ],
        "best_for": "General use, fast iteration",
        "typical_steps": "20-30"
    },
    
    "dpm++_2m": {
        "name": "DPM++ 2M Karras",
        "speed": "★★★☆☆",
        "quality": "★★★★★",
        "stability": "★★★★★",
        "characteristics": [
            "High quality results",
            "Deterministic with same seed",
            "Smooth, coherent images",
            "Good for photorealism"
        ],
        "best_for": "High-quality final outputs",
        "typical_steps": "25-40"
    },
    
    "ddim": {
        "name": "Denoising Diffusion Implicit",
        "speed": "★★★☆☆",
        "quality": "★★★☆☆",
        "stability": "★★★★★",
        "characteristics": [
            "Original SD scheduler",
            "Very deterministic",
            "Predictable results",
            "Not the fastest"
        ],
        "best_for": "When you need consistency",
        "typical_steps": "30-50"
    },
    
    "lms": {
        "name": "Linear Multistep",
        "speed": "★★★★★",
        "quality": "★★★☆☆",
        "characteristics": [
            "Very fast",
            "Lower quality at low steps",
            "Good for prototyping"
        ],
        "best_for": "Quick previews",
        "typical_steps": "15-25"
    }
}

def demonstrate_schedulers():
    """
    Show how different schedulers affect output.
    """
    generator = StableDiffusionGenerator()
    
    prompt = "a professional photograph of a coffee cup on a desk"
    
    schedulers = ["euler_a", "dpm"]
    results = {}
    
    for scheduler in schedulers:
        generator.set_scheduler (scheduler)
        
        images = generator.generate(
            prompt=prompt,
            steps=30,
            seed=42  # Same seed for comparison
        )
        
        results[scheduler] = images[0]
        images[0].save (f"coffee_{scheduler}.png")
    
    return results
\`\`\`

## Advanced Parameters

### Guidance Scale Deep Dive

\`\`\`python
guidance_scale_guide = {
    "1.0_to_3.0": {
        "description": "Minimal guidance",
        "result": "Creative, abstract, may ignore prompt",
        "use_case": "Experimental art",
        "example": "Might generate something loosely related"
    },
    
    "4.0_to_6.0": {
        "description": "Low guidance",
        "result": "Creative interpretation of prompt",
        "use_case": "Artistic freedom",
        "example": "Follows general concept, creative details"
    },
    
    "7.0_to_9.0": {
        "description": "Balanced (recommended)",
        "result": "Good prompt following + creativity",
        "use_case": "Most generation tasks",
        "example": "Reliable, predictable results"
    },
    
    "10.0_to_13.0": {
        "description": "High guidance",
        "result": "Strong prompt adherence",
        "use_case": "Specific requirements",
        "example": "Exactly what you asked for"
    },
    
    "14.0_plus": {
        "description": "Maximum guidance",
        "result": "Over-saturated, artificial looking",
        "use_case": "Rarely useful",
        "example": "Too strong, creates artifacts"
    }
}

class AdvancedSDGenerator(StableDiffusionGenerator):
    """
    Extended SD generator with advanced features.
    """
    
    def progressive_guidance(
        self,
        prompt: str,
        guidance_scales: List[float],
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate with different guidance scales to find optimal.
        """
        results = []
        
        for scale in guidance_scales:
            images = self.generate(
                prompt=prompt,
                guidance_scale=scale,
                **kwargs
            )
            results.append (images[0])
        
        return results
    
    def multi_step_comparison(
        self,
        prompt: str,
        step_counts: List[int],
        **kwargs
    ) -> dict:
        """
        Compare different step counts.
        """
        import time
        
        results = {}
        
        for steps in step_counts:
            start = time.time()
            images = self.generate(
                prompt=prompt,
                steps=steps,
                **kwargs
            )
            elapsed = time.time() - start
            
            results[steps] = {
                "image": images[0],
                "time": elapsed
            }
        
        return results

# Example: Find optimal guidance
generator = AdvancedSDGenerator()

# Test guidance scales
scales = [5.0, 7.0, 9.0, 11.0, 13.0]
images = generator.progressive_guidance(
    prompt="a detailed portrait of a person",
    guidance_scales=scales,
    steps=30,
    seed=42
)

# Save comparison
for scale, img in zip (scales, images):
    img.save (f"guidance_{scale}.png")

# Test step counts
step_results = generator.multi_step_comparison(
    prompt="a landscape painting",
    step_counts=[20, 30, 40, 50],
    seed=42
)

for steps, data in step_results.items():
    print(f"{steps} steps: {data['time']:.2f}s")
    data['image'].save (f"steps_{steps}.png")
\`\`\`

### Dimensions and Aspect Ratios

\`\`\`python
dimension_guide = {
    "rules": [
        "Must be multiples of 64 (8 for SDXL)",
        "Total pixels affects VRAM usage",
        "Model trained size = best quality",
        "Larger = slower generation"
    ],
    
    "common_sizes": {
        "512x512": {
            "pixels": 262_144,
            "vram": "~4GB",
            "speed": "Fast",
            "quality": "Good (native for SD 2.1)",
            "use": "Quick generation, testing"
        },
        "768x768": {
            "pixels": 589_824,
            "vram": "~6GB",
            "speed": "Medium",
            "quality": "Better detail",
            "use": "Higher quality outputs"
        },
        "1024x1024": {
            "pixels": 1_048_576,
            "vram": "~10GB",
            "speed": "Slow",
            "quality": "Excellent (native for SDXL)",
            "use": "Final outputs, SDXL"
        },
        "512x768": {
            "pixels": 393_216,
            "aspect": "Portrait (2:3)",
            "use": "Vertical images, portraits"
        },
        "768x512": {
            "pixels": 393_216,
            "aspect": "Landscape (3:2)",
            "use": "Horizontal images, scenery"
        },
        "1024x576": {
            "pixels": 589_824,
            "aspect": "Widescreen (16:9)",
            "use": "Cinematic, desktop wallpapers"
        }
    }
}

def calculate_vram_usage (width: int, height: int, batch_size: int = 1) -> dict:
    """
    Estimate VRAM requirements.
    """
    pixels = width * height
    
    # Rough estimates (float16)
    base_model = 3.5  # GB
    latent_calc = (pixels / 262_144) * 2  # GB per 512x512 equivalent
    batch_multiplier = batch_size * 0.5  # GB per additional image
    
    total = base_model + latent_calc + batch_multiplier
    
    return {
        "pixels": pixels,
        "estimated_vram_gb": round (total, 1),
        "recommended_gpu": "RTX 3060 6GB" if total < 6 else "RTX 3080 10GB" if total < 10 else "RTX 4090 24GB"
    }

# Check VRAM for different sizes
for size_name, (w, h) in [
    ("512x512", (512, 512)),
    ("768x768", (768, 768)),
    ("1024x1024", (1024, 1024)),
    ("1024x1536", (1024, 1536))
]:
    vram = calculate_vram_usage (w, h)
    print(f"{size_name}: {vram['estimated_vram_gb']}GB VRAM")
\`\`\`

## Model Variants

### SD 2.1 vs SDXL

\`\`\`python
model_comparison = {
    "sd_2_1": {
        "resolution": "512×512 native",
        "parameters": "~900M",
        "vram_required": "4-6GB",
        "speed": "★★★★★ Fast",
        "quality": "★★★☆☆ Good",
        "pros": [
            "Fast generation",
            "Lower VRAM requirements",
            "Mature ecosystem",
            "Many community models"
        ],
        "cons": [
            "Lower detail than SDXL",
            "Worse at complex compositions",
            "512×512 looks dated"
        ],
        "best_for": "Fast iteration, high volume, limited hardware"
    },
    
    "sdxl": {
        "resolution": "1024×1024 native",
        "parameters": "~6.6B (base + refiner)",
        "vram_required": "8-12GB",
        "speed": "★★☆☆☆ Slower",
        "quality": "★★★★★ Excellent",
        "pros": [
            "Much better detail",
            "Better composition",
            "Better text rendering",
            "More photorealistic",
            "Handles complex prompts"
        ],
        "cons": [
            "Slower generation",
            "More VRAM needed",
            "Newer, fewer community models"
        ],
        "best_for": "High quality outputs, professional work"
    }
}

class SDXLGenerator:
    """
    Stable Diffusion XL implementation.
    """
    
    def __init__(self, device: str = "cuda"):
        from diffusers import StableDiffusionXLPipeline
        
        self.device = device
        
        # Load SDXL
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        self.pipe = self.pipe.to (device)
        self.pipe.enable_xformers_memory_efficient_attention()
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 40,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate with SDXL (higher quality, slower).
        """
        generator = None
        if seed:
            generator = torch.Generator (self.device).manual_seed (seed)
        
        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        ).images
        
        return images

# Usage comparison
print("Generating with SD 2.1 (fast)...")
sd21_gen = StableDiffusionGenerator("sd_2_1")
sd21_img = sd21_gen.generate(
    "a detailed portrait, professional photography",
    steps=30
)[0]

print("Generating with SDXL (high quality)...")
sdxl_gen = SDXLGenerator()
sdxl_img = sdxl_gen.generate(
    "a detailed portrait, professional photography",
    steps=40
)[0]

# SDXL will have much better detail and composition
\`\`\`

## Production Optimizations

### Memory Management

\`\`\`python
class OptimizedSDGenerator(StableDiffusionGenerator):
    """
    Memory-optimized Stable Diffusion.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Enable all optimizations
        self.enable_all_optimizations()
    
    def enable_all_optimizations (self):
        """Enable every optimization available."""
        
        # 1. Attention slicing (reduces VRAM)
        self.pipe.enable_attention_slicing("max")
        
        # 2. VAE slicing (helps with high res)
        try:
            self.pipe.enable_vae_slicing()
        except:
            pass
        
        # 3. CPU offload (if really low on VRAM)
        # self.pipe.enable_model_cpu_offload()  # Slower but uses less VRAM
        
        # 4. xformers (faster attention)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
        
        print("✓ All optimizations enabled")
    
    def generate_tiled(
        self,
        prompt: str,
        final_width: int,
        final_height: int,
        tile_size: int = 512,
        overlap: int = 64,
        **kwargs
    ) -> Image.Image:
        """
        Generate large images by tiling (if VRAM limited).
        """
        # This is a simplified version
        # Real implementation would need seam blending
        
        tiles = []
        
        for y in range(0, final_height, tile_size - overlap):
            row = []
            for x in range(0, final_width, tile_size - overlap):
                # Generate tile
                tile = self.generate(
                    prompt=prompt,
                    width=min (tile_size, final_width - x),
                    height=min (tile_size, final_height - y),
                    **kwargs
                )[0]
                row.append (tile)
            tiles.append (row)
        
        # Stitch tiles together (simplified)
        return self._stitch_tiles (tiles)
    
    def _stitch_tiles (self, tiles: List[List[Image.Image]]) -> Image.Image:
        """Simple tile stitching."""
        # In production, use proper seam blending
        rows = []
        for row in tiles:
            rows.append (self._concat_horizontal (row))
        return self._concat_vertical (rows)
    
    @staticmethod
    def _concat_horizontal (images: List[Image.Image]) -> Image.Image:
        widths, heights = zip(*(i.size for i in images))
        total_width = sum (widths)
        max_height = max (heights)
        
        result = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        
        for img in images:
            result.paste (img, (x_offset, 0))
            x_offset += img.width
        
        return result
    
    @staticmethod
    def _concat_vertical (images: List[Image.Image]) -> Image.Image:
        widths, heights = zip(*(i.size for i in images))
        max_width = max (widths)
        total_height = sum (heights)
        
        result = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        
        for img in images:
            result.paste (img, (0, y_offset))
            y_offset += img.height
        
        return result
\`\`\`

### Batch Processing

\`\`\`python
import queue
import threading
from dataclasses import dataclass
from typing import Callable

@dataclass
class GenerationJob:
    """Job for generation queue."""
    id: str
    prompt: str
    params: dict
    callback: Optional[Callable] = None

class BatchSDGenerator:
    """
    Process generation jobs in batches.
    """
    
    def __init__(
        self,
        model_name: str = "sd_2_1",
        batch_size: int = 4,
        num_workers: int = 1
    ):
        self.generator = OptimizedSDGenerator (model_name)
        self.batch_size = batch_size
        self.job_queue = queue.Queue()
        self.results = {}
        
        # Start workers
        self.workers = []
        for _ in range (num_workers):
            worker = threading.Thread (target=self._worker, daemon=True)
            worker.start()
            self.workers.append (worker)
    
    def submit(
        self,
        job_id: str,
        prompt: str,
        callback: Optional[Callable] = None,
        **params
    ):
        """Submit a generation job."""
        job = GenerationJob(
            id=job_id,
            prompt=prompt,
            params=params,
            callback=callback
        )
        self.job_queue.put (job)
    
    def _worker (self):
        """Worker thread that processes jobs."""
        while True:
            # Collect batch
            batch = []
            for _ in range (self.batch_size):
                try:
                    job = self.job_queue.get (timeout=1)
                    batch.append (job)
                except queue.Empty:
                    break
            
            if not batch:
                continue
            
            # Process batch
            for job in batch:
                try:
                    images = self.generator.generate(
                        prompt=job.prompt,
                        **job.params
                    )
                    
                    self.results[job.id] = images[0]
                    
                    if job.callback:
                        job.callback (job.id, images[0])
                        
                except Exception as e:
                    print(f"Error generating {job.id}: {e}")
                finally:
                    self.job_queue.task_done()
    
    def wait_all (self):
        """Wait for all jobs to complete."""
        self.job_queue.join()
    
    def get_result (self, job_id: str) -> Optional[Image.Image]:
        """Get result for a job."""
        return self.results.get (job_id)

# Usage
batch_gen = BatchSDGenerator (batch_size=4)

# Submit jobs
for i in range(10):
    batch_gen.submit(
        job_id=f"image_{i}",
        prompt=f"a photograph of subject {i}",
        steps=25,
        seed=i
    )

# Wait for completion
batch_gen.wait_all()

# Get results
for i in range(10):
    img = batch_gen.get_result (f"image_{i}")
    if img:
        img.save (f"batch_{i}.png")
\`\`\`

## Key Takeaways

- Stable Diffusion is open source, free, and runs locally
- Requires NVIDIA GPU with 6GB+ VRAM (8GB+ recommended)
- SD 2.1: Fast, 512×512 native, good for high volume
- SDXL: Slower, 1024×1024 native, much better quality
- Scheduler choice affects quality and speed
- Guidance scale 7-9 is usually optimal
- Memory optimizations critical for limited VRAM
- Perfect for: customization, privacy, high volume, experimentation
- Use DALL-E 3 when: prompt following critical, need convenience
`,
};
