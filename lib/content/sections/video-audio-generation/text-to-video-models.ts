export const textToVideoModels = {
  title: 'Text-to-Video Models',
  id: 'text-to-video-models',
  content: `
# Text-to-Video Models

## Introduction

Text-to-video generation has rapidly evolved from a research curiosity to production-ready technology. In this section, we'll dive deep into the specific models available today, how to use them in production, their strengths and weaknesses, and practical considerations for choosing and deploying them.

We'll cover:
- **Runway Gen-2**: Industry-leading quality and API access
- **Pika Labs**: Fast, accessible, web-based generation
- **Stable Video Diffusion**: Open-source alternative
- **AnimateDiff**: Community-driven animation

Each model has different trade-offs in terms of quality, speed, cost, accessibility, and control.

---

## Runway Gen-2: Production-Ready Video Generation

### Overview

**Runway Gen-2** is one of the most advanced text-to-video models publicly available (as of 2024). It\'s used by professional creators and offers:

- **Quality**: High-fidelity, cinematic outputs
- **Length**: Up to 16 seconds per generation
- **Resolution**: Up to 1280x768 (upgradable to 4K)
- **API Access**: Full programmatic control
- **Pricing**: Pay-per-generation model

### Key Features

1. **Text-to-Video**: Generate from text prompts alone
2. **Image-to-Video**: Animate static images
3. **Style Transfer**: Apply artistic styles
4. **Camera Controls**: Specify camera movements
5. **Motion Brush**: Paint areas where motion should occur

### Production Integration

\`\`\`python
"""
Complete Runway Gen-2 API Integration
Production-ready implementation with error handling, retries, and cost tracking
"""

import os
import requests
import time
import json
from typing import Optional, Dict, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from pathlib import Path

class RunwayStyle(Enum):
    """Predefined style presets"""
    REALISTIC = "realistic"
    CINEMATIC = "cinematic"
    ANIME = "anime"
    WATERCOLOR = "watercolor"
    PENCIL_SKETCH = "pencil_sketch"
    OIL_PAINTING = "oil_painting"
    CYBERPUNK = "cyberpunk"
    FANTASY = "fantasy"

class CameraMovement(Enum):
    """Camera movement types"""
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    ORBIT = "orbit"
    STATIC = "static"

@dataclass
class RunwayVideoConfig:
    """Configuration for video generation"""
    prompt: str
    duration: float = 4.0  # seconds (max 16)
    aspect_ratio: str = "16:9"  # "16:9", "9:16", "1:1", "4:3"
    style: Optional[RunwayStyle] = None
    camera_movement: Optional[CameraMovement] = None
    seed: Optional[int] = None  # For reproducibility
    interpolate: bool = True  # Smooth motion interpolation
    upscale: bool = False  # 4K upscaling (extra cost)
    image_prompt: Optional[str] = None  # URL for image-to-video
    
    # Advanced settings
    motion_amount: float = 0.5  # 0.0 to 1.0, how much motion
    prompt_weight: float = 1.0  # How strongly to follow prompt
    
    def to_api_payload (self) -> Dict:
        """Convert to API request format"""
        payload = {
            "prompt": self.prompt,
            "duration": min (self.duration, 16.0),
            "aspect_ratio": self.aspect_ratio,
            "interpolate": self.interpolate,
            "motion_amount": self.motion_amount,
            "prompt_weight": self.prompt_weight,
        }
        
        if self.style:
            payload["style"] = self.style.value
        
        if self.camera_movement:
            payload["camera_movement"] = self.camera_movement.value
        
        if self.seed is not None:
            payload["seed"] = self.seed
        
        if self.upscale:
            payload["upscale"] = True
        
        if self.image_prompt:
            payload["image_prompt"] = self.image_prompt
        
        return payload

@dataclass
class RunwayVideoResult:
    """Result from video generation"""
    id: str
    url: str
    status: str  # "pending", "processing", "completed", "failed"
    duration: float
    resolution: tuple[int, int]
    cost: float  # Estimated cost in USD
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    
    # Metadata
    prompt: str = ""
    config: Optional[Dict] = None

class RunwayGen2Client:
    """
    Production client for Runway Gen-2 API
    
    Features:
    - Automatic retry with exponential backoff
    - Cost tracking
    - Result caching
    - Progress callbacks
    - Batch generation
    """
    
    # Pricing (as of 2024, check latest)
    COST_PER_SECOND = 0.50  # $0.50 per second of video
    COST_UPSCALE = 2.00  # Additional $2 for 4K upscale
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        track_costs: bool = True,
    ):
        self.api_key = api_key or os.getenv("RUNWAY_API_KEY")
        if not self.api_key:
            raise ValueError("RUNWAY_API_KEY must be set")
        
        self.base_url = "https://api.runwayml.com/v1"
        self.cache_dir = cache_dir or Path("./runway_cache")
        self.cache_dir.mkdir (exist_ok=True)
        
        self.track_costs = track_costs
        self.total_cost = 0.0
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
    
    def generate(
        self,
        config: Union[RunwayVideoConfig, str],
        wait_for_completion: bool = True,
        progress_callback: Optional[callable] = None,
        max_retries: int = 3,
    ) -> RunwayVideoResult:
        """
        Generate video from text prompt
        
        Args:
            config: RunwayVideoConfig or simple text prompt
            wait_for_completion: If True, polls until video is ready
            progress_callback: Function called with progress updates
            max_retries: Number of retries for failed requests
        
        Returns:
            RunwayVideoResult with video URL and metadata
        """
        # Convert string to config
        if isinstance (config, str):
            config = RunwayVideoConfig (prompt=config)
        
        # Check cache
        cache_key = self._get_cache_key (config)
        cached_result = self._get_cached_result (cache_key)
        if cached_result:
            print(f"✅ Using cached result: {cached_result.url}")
            return cached_result
        
        # Submit generation request
        payload = config.to_api_payload()
        
        for attempt in range (max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/generations",
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception (f"Failed to submit generation after {max_retries} attempts: {e}")
                
                # Exponential backoff
                wait_time = 2 ** attempt
                print(f"⚠️  Request failed, retrying in {wait_time}s...")
                time.sleep (wait_time)
        
        data = response.json()
        video_id = data["id"]
        
        print(f"🎬 Generation started: {video_id}")
        
        if not wait_for_completion:
            return RunwayVideoResult(
                id=video_id,
                url="",
                status="pending",
                duration=config.duration,
                resolution=(0, 0),
                cost=self._estimate_cost (config),
                created_at=data.get("created_at", ""),
                prompt=config.prompt,
                config=asdict (config),
            )
        
        # Poll for completion
        result = self._poll_until_complete(
            video_id,
            progress_callback=progress_callback,
            max_wait_seconds=600,  # 10 minutes max
        )
        
        # Track cost
        if self.track_costs:
            self.total_cost += result.cost
            print(f"💰 Cost: \${result.cost:.2f} (Total: \${self.total_cost:.2f})")
        
        # Cache result
self._cache_result (cache_key, result)

return result
    
    def _poll_until_complete(
    self,
    video_id: str,
    progress_callback: Optional[callable] = None,
    max_wait_seconds: int = 600,
    poll_interval: int = 3,
) -> RunwayVideoResult:
"""Poll generation status until complete"""
start_time = time.time()
last_status = None

while time.time() - start_time < max_wait_seconds:
    result = self.get_status (video_id)
            
            # Call progress callback if status changed
if progress_callback and result.status != last_status:
progress_callback (result)
last_status = result.status

if result.status == "completed":
    elapsed = time.time() - start_time
print(f"✅ Generation completed in {elapsed:.1f}s")
return result
            
            elif result.status == "failed":
                raise Exception (f"Generation failed: {result.error}")
            
            elif result.status in ["pending", "processing"]:
                # Show progress
elapsed = time.time() - start_time
print(f"⏳ Still processing... ({elapsed:.0f}s elapsed)")
time.sleep (poll_interval)
            
            else:
                raise Exception (f"Unknown status: {result.status}")
        
        raise TimeoutError (f"Generation timed out after {max_wait_seconds}s")
    
    def get_status (self, video_id: str) -> RunwayVideoResult:
"""Get current status of generation"""
response = self.session.get (f"{self.base_url}/generations/{video_id}")
response.raise_for_status()

data = response.json()
        
        # Parse resolution
resolution_str = data.get("resolution", "0x0")
width, height = map (int, resolution_str.split("x"))

return RunwayVideoResult(
    id = video_id,
    url = data.get("url", ""),
    status = data["status"],
    duration = data.get("duration", 0),
    resolution = (width, height),
    cost = data.get("cost", 0),
    created_at = data.get("created_at", ""),
    completed_at = data.get("completed_at"),
    error = data.get("error"),
    prompt = data.get("prompt", ""),
    config = data.get("config"),
)
    
    def generate_batch(
    self,
    configs: List[Union[RunwayVideoConfig, str]],
    max_parallel: int = 5,
    progress_callback: Optional[callable] = None,
) -> List[RunwayVideoResult]:
"""
        Generate multiple videos in parallel

Args:
configs: List of video configurations
max_parallel: Maximum concurrent generations
progress_callback: Called with (completed, total, current_result)

Returns:
            List of results in same order as configs
"""
import concurrent.futures

results = []
completed = 0
total = len (configs)

with concurrent.futures.ThreadPoolExecutor (max_workers = max_parallel) as executor:
            # Submit all jobs
future_to_config = {
    executor.submit (self.generate, config): config 
                for config in configs
}
            
            # Collect results as they complete
for future in concurrent.futures.as_completed (future_to_config):
    try:
result = future.result()
results.append (result)
completed += 1

if progress_callback:
    progress_callback (completed, total, result)

print(f"✅ Completed {completed}/{total}")
                    
                except Exception as e:
print(f"❌ Generation failed: {e}")
results.append(None)

return results
    
    def _estimate_cost (self, config: RunwayVideoConfig) -> float:
"""Estimate cost for generation"""
cost = config.duration * self.COST_PER_SECOND

if config.upscale:
    cost += self.COST_UPSCALE

return cost
    
    def _get_cache_key (self, config: RunwayVideoConfig) -> str:
"""Generate cache key from config"""
        # Create deterministic hash of config
config_str = json.dumps (asdict (config), sort_keys = True)
return hashlib.md5(config_str.encode()).hexdigest()
    
    def _cache_result (self, cache_key: str, result: RunwayVideoResult):
"""Save result to cache"""
cache_file = self.cache_dir / f"{cache_key}.json"
with open (cache_file, "w") as f:
json.dump (asdict (result), f, indent = 2)
    
    def _get_cached_result (self, cache_key: str) -> Optional[RunwayVideoResult]:
"""Load result from cache if exists"""
cache_file = self.cache_dir / f"{cache_key}.json"

if cache_file.exists():
    with open (cache_file) as f:
    data = json.load (f)
return RunwayVideoResult(** data)

return None
    
    def get_cost_summary (self) -> Dict:
"""Get summary of costs"""
return {
    "total_cost_usd": self.total_cost,
    "average_cost_per_video": self.total_cost / max(1, len (list (self.cache_dir.glob("*.json")))),
}

# Example usage
def main():
"""Demonstrate Runway Gen-2 integration"""
    
    # Initialize client
client = RunwayGen2Client(
    api_key = os.getenv("RUNWAY_API_KEY"),
    track_costs = True,
)
    
    # Example 1: Simple text - to - video
print("\\n📹 Example 1: Simple generation")
config = RunwayVideoConfig(
    prompt = "A golden retriever puppy playing in a sunny garden, slow motion, cinematic lighting",
    duration = 4.0,
    style = RunwayStyle.CINEMATIC,
)

result = client.generate(
    config,
    progress_callback = lambda r: print(f"  Status: {r.status}"),
)

print(f"✅ Video URL: {result.url}")
print(f"   Duration: {result.duration}s")
print(f"   Resolution: {result.resolution[0]}x{result.resolution[1]}")
print(f"   Cost: \${'{'}result.cost:.2f{'}'}")
    
    # Example 2: Camera movement
print("\\n📹 Example 2: With camera movement")
config = RunwayVideoConfig(
    prompt = "A futuristic city skyline at night, neon lights reflecting off glass buildings",
    duration = 8.0,
    camera_movement = CameraMovement.DOLLY_IN,
    style = RunwayStyle.CYBERPUNK,
)

result = client.generate (config)
print(f"✅ Video URL: {result.url}")
    
    # Example 3: Image - to - video
print("\\n📹 Example 3: Animate static image")
config = RunwayVideoConfig(
    prompt = "The scene comes to life, gentle movement and atmospheric effects",
    duration = 4.0,
    image_prompt = "https://example.com/landscape.jpg",
    motion_amount = 0.3,  # Subtle motion
)

result = client.generate (config)
print(f"✅ Video URL: {result.url}")
    
    # Example 4: Batch generation
print("\\n📹 Example 4: Batch generation")
prompts = [
    "A cat chasing a laser pointer",
    "Ocean waves crashing on a beach at sunset",
    "Rain drops falling on a window pane",
    "Cherry blossoms falling in slow motion",
]

configs = [RunwayVideoConfig (prompt = p, duration = 4.0) for p in prompts]

results = client.generate_batch(
    configs,
    max_parallel = 3,
    progress_callback = lambda done, total, result: print(f"  Progress: {done}/{total}"),
)

print(f"✅ Generated {len (results)} videos")
    
    # Show cost summary
print("\\n💰 Cost Summary:")
summary = client.get_cost_summary()
print(f"   Total spent: \${summary['total_cost_usd']:.2f}")
print(f"   Average per video: \${summary['average_cost_per_video']:.2f}")

if __name__ == "__main__":
    main()
\`\`\`

---

## Pika Labs: Fast and Accessible

### Overview

**Pika Labs** focuses on speed and accessibility:

- **Speed**: Generates in 1-3 minutes
- **Quality**: Good quality for social media content
- **Length**: Up to 3 seconds currently
- **Access**: Web-based interface, Discord bot
- **Pricing**: Free tier + subscription

### Key Features

1. **Fast Generation**: Much faster than competitors
2. **Style Flexibility**: Multiple artistic styles
3. **Motion Control**: Directional motion controls
4. **Easy Iteration**: Quick edits and variations
5. **Community**: Active Discord community

### Integration Example

\`\`\`python
"""
Pika Labs API Integration (via Discord Bot)
Note: Pika primarily uses Discord. This shows conceptual API usage.
"""

import requests
import time
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class PikaConfig:
    """Pika generation configuration"""
    prompt: str
    motion: int = 3  # 1 (slow) to 4 (fast)
    guidance_scale: float = 12.0  # How closely to follow prompt
    aspect_ratio: str = "16:9"
    fps: int = 24
    style: Optional[str] = None  # "anime", "3d", "realistic"
    
@dataclass
class PikaResult:
    """Pika generation result"""
    id: str
    url: str
    status: str
    duration: float
    prompt: str

class PikaLabsClient:
    """
    Client for Pika Labs (conceptual - adapt to actual API when available)
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.pika.art/v1"  # Hypothetical
    
    def generate(
        self,
        config: PikaConfig,
        wait_for_completion: bool = True,
    ) -> PikaResult:
        """Generate video with Pika"""
        
        # Build prompt with parameters
        full_prompt = config.prompt
        
        if config.style:
            full_prompt = f"{full_prompt}, {config.style} style"
        
        # Motion parameters
        full_prompt = f"{full_prompt} -motion {config.motion}"
        full_prompt = f"{full_prompt} -gs {config.guidance_scale}"
        full_prompt = f"{full_prompt} -ar {config.aspect_ratio}"
        
        # Submit request
        payload = {
            "prompt": full_prompt,
            "parameters": {
                "fps": config.fps,
            }
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        video_id = data["id"]
        
        if not wait_for_completion:
            return PikaResult(
                id=video_id,
                url="",
                status="pending",
                duration=3.0,
                prompt=config.prompt,
            )
        
        # Wait for completion
        return self._poll_completion (video_id)
    
    def _poll_completion (self, video_id: str, max_wait: int = 180) -> PikaResult:
        """Poll until video is ready"""
        start = time.time()
        
        while time.time() - start < max_wait:
            response = requests.get(
                f"{self.base_url}/status/{video_id}",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data["status"] == "completed":
                return PikaResult(
                    id=video_id,
                    url=data["url"],
                    status="completed",
                    duration=data.get("duration", 3.0),
                    prompt=data.get("prompt", ""),
                )
            elif data["status"] == "failed":
                raise Exception (f"Generation failed: {data.get('error')}")
            
            time.sleep(5)
        
        raise TimeoutError("Generation timed out")

# Example usage
def pika_example():
    """Example Pika usage"""
    client = PikaLabsClient (api_key="your_api_key")
    
    config = PikaConfig(
        prompt="A cyberpunk street scene with neon signs, raining",
        motion=4,  # Fast motion
        style="3d",
        aspect_ratio="9:16",  # Vertical for social media
    )
    
    result = client.generate (config)
    print(f"Generated: {result.url}")

if __name__ == "__main__":
    pika_example()
\`\`\`

---

## Stable Video Diffusion: Open Source Alternative

### Overview

**Stable Video Diffusion (SVD)** is Stability AI's open-source video generation model:

- **Open Source**: Fully open weights and code
- **Image-to-Video**: Primarily animates static images
- **Quality**: Excellent for smooth, realistic motion
- **Length**: 2-4 seconds
- **Self-Hosted**: Run on your own hardware

### Architecture

SVD extends Stable Diffusion with temporal layers:

\`\`\`python
"""
Stable Video Diffusion Integration
Using the diffusers library
"""

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import numpy as np

class StableVideoDiffusion:
    """
    Wrapper for Stable Video Diffusion
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-video-diffusion-img2vid",
        device: str = "cuda",
        variant: str = "fp16",  # Use half precision for speed
    ):
        self.device = device
        
        # Load pipeline
        print(f"Loading SVD model: {model_id}")
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if variant == "fp16" else torch.float32,
            variant=variant,
        )
        self.pipe.to (device)
        
        # Enable memory optimizations
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()
    
    def generate(
        self,
        image: Union[str, Image.Image],
        num_frames: int = 14,
        decode_chunk_size: int = 8,
        motion_bucket_id: int = 127,  # 0-255, higher = more motion
        fps: int = 7,
        noise_aug_strength: float = 0.02,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Generate video from image
        
        Args:
            image: PIL Image or URL/path
            num_frames: Number of frames to generate
            decode_chunk_size: Decode frames in chunks (memory optimization)
            motion_bucket_id: Amount of motion (0=static, 255=maximum)
            fps: Frame rate for playback
            noise_aug_strength: Noise augmentation (0-1)
            seed: Random seed for reproducibility
        
        Returns:
            List of PIL Images (frames)
        """
        # Load image if path/URL
        if isinstance (image, str):
            image = load_image (image)
        
        # Resize to model input size (1024x576)
        image = image.resize((1024, 576))
        
        # Set seed
        generator = None
        if seed is not None:
            generator = torch.manual_seed (seed)
        
        # Generate video frames
        frames = self.pipe(
            image=image,
            num_frames=num_frames,
            decode_chunk_size=decode_chunk_size,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            generator=generator,
        ).frames[0]  # Extract first batch
        
        return frames
    
    def generate_from_text(
        self,
        prompt: str,
        num_frames: int = 14,
        motion_bucket_id: int = 127,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Text-to-video by first generating image, then animating
        
        Requires separate Stable Diffusion model for text-to-image
        """
        # First: Generate image from text using SD
        from diffusers import StableDiffusionPipeline
        
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        )
        sd_pipe.to (self.device)
        
        # Generate initial image
        print(f"Generating initial image: {prompt}")
        generator = torch.manual_seed (seed) if seed else None
        
        image = sd_pipe(
            prompt=prompt,
            height=576,
            width=1024,
            generator=generator,
        ).images[0]
        
        # Animate the image
        print("Animating image...")
        frames = self.generate(
            image=image,
            num_frames=num_frames,
            motion_bucket_id=motion_bucket_id,
            seed=seed,
        )
        
        return frames
    
    def save_video(
        self,
        frames: List[Image.Image],
        output_path: str,
        fps: int = 7,
    ):
        """Save frames as video file"""
        export_to_video (frames, output_path, fps=fps)
        print(f"Video saved: {output_path}")

# Example usage
def main():
    """Demonstrate Stable Video Diffusion"""
    
    # Initialize model
    svd = StableVideoDiffusion (device="cuda")
    
    # Example 1: Image-to-video
    print("\\n📹 Example 1: Animate static image")
    frames = svd.generate(
        image="path/to/image.jpg",
        num_frames=25,
        motion_bucket_id=150,  # More motion
        fps=7,
        seed=42,
    )
    
    svd.save_video (frames, "output1.mp4", fps=7)
    print(f"Generated {len (frames)} frames")
    
    # Example 2: Text-to-video (via image)
    print("\\n📹 Example 2: Text-to-video")
    frames = svd.generate_from_text(
        prompt="A serene Japanese garden with a koi pond, cherry blossoms, professional photography",
        num_frames=25,
        motion_bucket_id=127,
        seed=42,
    )
    
    svd.save_video (frames, "output2.mp4", fps=7)
    
    # Example 3: Experimenting with motion
    print("\\n📹 Example 3: Different motion levels")
    base_image = "path/to/image.jpg"
    
    for motion_level in [50, 127, 200]:
        print(f"  Generating with motion level {motion_level}")
        frames = svd.generate(
            image=base_image,
            motion_bucket_id=motion_level,
            seed=42,
        )
        
        svd.save_video(
            frames,
            f"motion_{motion_level}.mp4",
            fps=7
        )

if __name__ == "__main__":
    main()
\`\`\`

---

## AnimateDiff: Community-Driven Animation

### Overview

**AnimateDiff** is a community project that adds motion to Stable Diffusion:

- **Flexibility**: Works with any SD model
- **LoRA Support**: Use custom trained LoRAs
- **Motion Modules**: Pre-trained motion patterns
- **Free**: Completely open source
- **Active Community**: Constant improvements

### Key Concepts

1. **Motion Modules**: Pre-trained temporal layers
2. **Base Model**: Any Stable Diffusion checkpoint
3. **LoRAs**: Fine-tuned style adaptations
4. **ControlNet**: Precise motion control

\`\`\`python
"""
AnimateDiff Integration
Create animated videos with Stable Diffusion models
"""

from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from diffusers.utils import export_to_video
import torch

class AnimateDiffGenerator:
    """
    AnimateDiff video generator
    """
    
    def __init__(
        self,
        base_model: str = "runwayml/stable-diffusion-v1-5",
        motion_adapter: str = "guoyww/animatediff-motion-adapter-v1-5-2",
        device: str = "cuda",
    ):
        self.device = device
        
        # Load motion adapter
        adapter = MotionAdapter.from_pretrained(
            motion_adapter,
            torch_dtype=torch.float16
        )
        
        # Load base model with motion
        self.pipe = AnimateDiffPipeline.from_pretrained(
            base_model,
            motion_adapter=adapter,
            torch_dtype=torch.float16,
        )
        
        # Use DDIM scheduler for better quality
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config,
            beta_schedule="linear",
            steps_offset=1,
        )
        
        # Memory optimizations
        self.pipe.enable_vae_slicing()
        self.pipe.enable_model_cpu_offload()
        
        self.pipe.to (device)
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, bad quality, distorted",
        num_frames: int = 16,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Generate animated video from prompt
        
        Args:
            prompt: Description of desired video
            negative_prompt: What to avoid
            num_frames: Number of frames (16-64 typical)
            num_inference_steps: Denoising steps
            guidance_scale: How closely to follow prompt
            seed: Random seed
        
        Returns:
            List of frames
        """
        generator = None
        if seed is not None:
            generator = torch.manual_seed (seed)
        
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        return output.frames[0]
    
    def load_lora (self, lora_path: str, weight: float = 1.0):
        """
        Load LoRA for style control
        
        Args:
            lora_path: Path or HuggingFace ID
            weight: LoRA influence (0-1)
        """
        self.pipe.load_lora_weights (lora_path)
        self.pipe.fuse_lora (lora_scale=weight)

# Example usage
def animatediff_examples():
    """Demonstrate AnimateDiff"""
    
    generator = AnimateDiffGenerator()
    
    # Example 1: Basic animation
    print("\\n📹 Example 1: Basic animation")
    frames = generator.generate(
        prompt="A cat playing with a ball of yarn, cute, fluffy, 4k",
        num_frames=16,
        seed=42,
    )
    
    export_to_video (frames, "animatediff_cat.mp4", fps=8)
    
    # Example 2: With LoRA for specific style
    print("\\n📹 Example 2: With anime LoRA")
    generator.load_lora("path/to/anime_lora.safetensors", weight=0.8)
    
    frames = generator.generate(
        prompt="Anime girl waving hello, cherry blossoms falling, studio ghibli style",
        num_frames=24,
        seed=42,
    )
    
    export_to_video (frames, "animatediff_anime.mp4", fps=8)

if __name__ == "__main__":
    animatediff_examples()
\`\`\`

---

## Model Comparison and Selection Guide

### Quality vs Speed vs Cost

| Model | Quality | Speed | Cost | Ease of Use | Best For |
|-------|---------|-------|------|-------------|----------|
| **Sora** | ⭐⭐⭐⭐⭐ | ⭐⭐ | 💰💰💰💰 | ⭐⭐⭐⭐ | Cinematic, professional |
| **Runway Gen-2** | ⭐⭐⭐⭐ | ⭐⭐⭐ | 💰💰💰 | ⭐⭐⭐⭐⭐ | Production content |
| **Pika Labs** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 💰💰 | ⭐⭐⭐⭐⭐ | Social media, quick iterations |
| **Stable Video Diffusion** | ⭐⭐⭐⭐ | ⭐⭐⭐ | 💰 | ⭐⭐⭐ | Self-hosted, image animation |
| **AnimateDiff** | ⭐⭐⭐ | ⭐⭐⭐ | 💰 | ⭐⭐ | Custom styles, experimentation |

### Decision Matrix

\`\`\`python
"""
Model selection helper
"""

from enum import Enum
from dataclasses import dataclass
from typing import List

class Priority(Enum):
    QUALITY = "quality"
    SPEED = "speed"
    COST = "cost"
    LENGTH = "length"
    CONTROL = "control"
    OPENSOURCE = "opensource"

class VideoModel(Enum):
    SORA = "sora"
    RUNWAY = "runway"
    PIKA = "pika"
    SVD = "svd"
    ANIMATEDIFF = "animatediff"

@dataclass
class ModelCapabilities:
    model: VideoModel
    max_length_seconds: int
    typical_speed_seconds: int
    cost_per_second: float
    quality_score: int  # 1-10
    ease_of_use: int  # 1-10
    opensource: bool
    api_available: bool
    
MODEL_SPECS = {
    VideoModel.SORA: ModelCapabilities(
        model=VideoModel.SORA,
        max_length_seconds=60,
        typical_speed_seconds=600,  # 10 minutes
        cost_per_second=1.0,  # Estimated
        quality_score=10,
        ease_of_use=9,
        opensource=False,
        api_available=False,  # Not yet
    ),
    VideoModel.RUNWAY: ModelCapabilities(
        model=VideoModel.RUNWAY,
        max_length_seconds=16,
        typical_speed_seconds=120,  # 2 minutes
        cost_per_second=0.50,
        quality_score=8,
        ease_of_use=10,
        opensource=False,
        api_available=True,
    ),
    VideoModel.PIKA: ModelCapabilities(
        model=VideoModel.PIKA,
        max_length_seconds=3,
        typical_speed_seconds=60,
        cost_per_second=0.30,
        quality_score=7,
        ease_of_use=10,
        opensource=False,
        api_available=True,
    ),
    VideoModel.SVD: ModelCapabilities(
        model=VideoModel.SVD,
        max_length_seconds=4,
        typical_speed_seconds=30,
        cost_per_second=0.0,  # Self-hosted
        quality_score=8,
        ease_of_use=6,
        opensource=True,
        api_available=True,  # Self-hosted
    ),
    VideoModel.ANIMATEDIFF: ModelCapabilities(
        model=VideoModel.ANIMATEDIFF,
        max_length_seconds=10,
        typical_speed_seconds=60,
        cost_per_second=0.0,  # Self-hosted
        quality_score=7,
        ease_of_use=5,
        opensource=True,
        api_available=True,  # Self-hosted
    ),
}

def recommend_model(
    priorities: List[Priority],
    required_length: int = 5,
    budget_per_video: float = 10.0,
    requires_api: bool = True,
) -> VideoModel:
    """
    Recommend best model based on requirements
    
    Args:
        priorities: List of priorities in order of importance
        required_length: Minimum video length needed (seconds)
        budget_per_video: Maximum cost per video
        requires_api: Need programmatic API access
    
    Returns:
        Recommended model
    """
    scores = {model: 0 for model in VideoModel}
    
    for model, specs in MODEL_SPECS.items():
        # Filter by hard requirements
        if specs.max_length_seconds < required_length:
            continue
        
        if specs.cost_per_second * required_length > budget_per_video:
            continue
        
        if requires_api and not specs.api_available:
            continue
        
        # Score based on priorities
        for priority in priorities:
            if priority == Priority.QUALITY:
                scores[model] += specs.quality_score * 2
            elif priority == Priority.SPEED:
                # Faster is better (inverse relationship)
                scores[model] += (600 - specs.typical_speed_seconds) / 60
            elif priority == Priority.COST:
                # Free/cheaper is better
                if specs.cost_per_second == 0:
                    scores[model] += 20
                else:
                    scores[model] += max(0, 10 - specs.cost_per_second * 10)
            elif priority == Priority.LENGTH:
                scores[model] += specs.max_length_seconds / 10
            elif priority == Priority.CONTROL:
                scores[model] += specs.ease_of_use
            elif priority == Priority.OPENSOURCE:
                scores[model] += 20 if specs.opensource else 0
    
    # Return model with highest score
    best_model = max (scores.items(), key=lambda x: x[1])
    return best_model[0]

# Example usage
if __name__ == "__main__":
    # Use case 1: High quality marketing video
    print("Use case 1: Marketing video")
    model = recommend_model(
        priorities=[Priority.QUALITY, Priority.LENGTH, Priority.SPEED],
        required_length=10,
        budget_per_video=20.0,
        requires_api=True,
    )
    print(f"Recommended: {model.value}\\n")
    
    # Use case 2: Rapid prototyping on budget
    print("Use case 2: Rapid prototyping")
    model = recommend_model(
        priorities=[Priority.SPEED, Priority.COST, Priority.QUALITY],
        required_length=3,
        budget_per_video=5.0,
        requires_api=True,
    )
    print(f"Recommended: {model.value}\\n")
    
    # Use case 3: Self-hosted, customizable
    print("Use case 3: Self-hosted solution")
    model = recommend_model(
        priorities=[Priority.OPENSOURCE, Priority.CONTROL, Priority.QUALITY],
        required_length=5,
        budget_per_video=0.0,
        requires_api=False,
    )
    print(f"Recommended: {model.value}\\n")
\`\`\`

---

## Summary

**Key Takeaways:**
- **Runway Gen-2**: Best for production-quality, API-first workflows
- **Pika Labs**: Fastest generation, great for rapid iteration
- **Stable Video Diffusion**: Open-source, excellent image animation
- **AnimateDiff**: Most flexible for custom styles and experimentation
- **Choice depends on**: quality needs, budget, required length, API requirements

**Next Steps:**
- Try each model with same prompts to compare results
- Understand pricing implications for your use case
- Consider hybrid approaches (prototype with Pika, final with Runway)
- Monitor new releases - this space evolves monthly!
`,
  exercises: [
    {
      title: 'Exercise 1: Multi-Provider Video Generator',
      difficulty: 'advanced' as const,
      description:
        'Build a unified video generation interface that can use Runway, Pika, or SVD based on requirements and automatically selects the best provider.',
      hints: [
        'Create abstract base class for all providers',
        'Implement the recommendation logic from the model selection guide',
        'Add fallback logic if primary provider fails',
        'Track costs and performance metrics for each provider',
      ],
      solution: `# See RunwayGen2Client and model selection code above
# Extend with factory pattern for provider selection`,
    },
    {
      title: 'Exercise 2: Batch Video Generation Pipeline',
      difficulty: 'intermediate' as const,
      description:
        'Create a production pipeline that generates multiple videos in parallel, handles retries, tracks costs, and sends notifications when complete.',
      hints: [
        'Use ThreadPoolExecutor for parallel generation',
        'Implement exponential backoff for retries',
        'Log all generations to database',
        'Add webhook notifications',
      ],
      solution: `# See the generate_batch method in RunwayGen2Client
# Add SQLite database for tracking
# Add webhook POST on completion`,
    },
  ],
};
