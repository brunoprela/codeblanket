export const imageToVideoAnimation = {
  title: 'Image-to-Video Animation',
  id: 'image-to-video-animation',
  content: `
# Image-to-Video Animation

## Introduction

**Image-to-video** (img2vid) is one of the most practical and widely-used applications of video generation. Instead of generating a video from scratch based on text alone, you start with a static image and bring it to life with motion, creating a seamless animation.

This approach offers several advantages:
- **More control**: You specify exactly what the first frame looks like
- **Consistency**: Character/object appearance is guaranteed
- **Easier prompting**: Describe motion rather than entire scene
- **Better quality**: Models excel at adding motion to existing content

---

## Why Image-to-Video Matters

### Use Cases

**1. Product Demonstrations**
- Animate product images for e-commerce
- Create 360° views from a single angle
- Add "lifestyle" motion to static product shots

**2. Social Media Content**
- Bring photos to life for Instagram/TikTok
- Create cinemagraphs (subtle motion in still images)
- Animate user-generated content

**3. Storytelling**
- Animate comic book panels
- Bring historical photos to life
- Create motion from concept art

**4. Marketing**
- Animated advertisements from still campaigns
- Add motion to infographics
- Create dynamic presentations

**5. Memory Enhancement**
- Animate old family photos
- Create living memories
- Add subtle motion to portraits

---

## How Image-to-Video Works

### Core Concepts

**Temporal Conditioning**:
Instead of generating from noise, the model:
1. Encodes the input image into latent space
2. Uses it as strong conditioning for first frame
3. Generates subsequent frames with motion
4. Maintains visual consistency with input

\`\`\`python
"""
Image-to-Video Generation: Core Concepts
"""

import torch
import torch.nn as nn
from PIL import Image
from typing import Optional, List
import numpy as np

class ImageToVideoModel(nn.Module):
    """
    Simplified image-to-video model showing key concepts
    """
    
    def __init__(
        self,
        image_encoder: nn.Module,
        temporal_model: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.temporal_model = temporal_model
        self.decoder = decoder
    
    def forward(
        self,
        image: torch.Tensor,  # (B, C, H, W)
        motion_prompt: torch.Tensor,  # Text embedding for motion
        num_frames: int = 14,
    ) -> torch.Tensor:
        """
        Generate video from single image
        
        Key idea: Use image as strong conditioning for first frame,
        then generate subsequent frames with temporal model
        """
        batch_size = image.shape[0]
        
        # Encode input image to latent space
        image_latent = self.image_encoder(image)  # (B, latent_dim, h, w)
        
        # Create latent sequence starting with image
        # First frame is the input image
        latent_sequence = [image_latent]
        
        # Generate subsequent frames
        for t in range(1, num_frames):
            # Use previous frame(s) and motion prompt to generate next
            prev_latent = latent_sequence[-1]
            
            # Temporal model predicts next frame
            next_latent = self.temporal_model(
                prev_latent,
                motion_prompt,
                timestep=t,
            )
            
            latent_sequence.append(next_latent)
        
        # Stack into tensor
        latent_video = torch.stack(latent_sequence, dim=1)  # (B, T, C, h, w)
        
        # Decode all frames
        frames = []
        for t in range(num_frames):
            frame = self.decoder(latent_video[:, t])
            frames.append(frame)
        
        video = torch.stack(frames, dim=1)  # (B, T, C, H, W)
        
        return video

class MotionControl:
    """
    Different types of motion that can be applied to static images
    """
    
    @staticmethod
    def zoom_in(
        image: Image.Image,
        num_frames: int = 16,
        zoom_factor: float = 1.5,
    ) -> List[Image.Image]:
        """
        Simple zoom-in effect
        
        This is a basic implementation - real models use neural networks
        """
        width, height = image.size
        frames = []
        
        for t in range(num_frames):
            # Calculate zoom amount for this frame
            progress = t / (num_frames - 1)  # 0 to 1
            current_zoom = 1.0 + (zoom_factor - 1.0) * progress
            
            # Calculate crop dimensions
            new_width = int(width / current_zoom)
            new_height = int(height / current_zoom)
            
            # Center crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = left + new_width
            bottom = top + new_height
            
            # Crop and resize back to original size
            frame = image.crop((left, top, right, bottom))
            frame = frame.resize((width, height), Image.LANCZOS)
            
            frames.append(frame)
        
        return frames
    
    @staticmethod
    def pan_horizontal(
        image: Image.Image,
        num_frames: int = 16,
        direction: str = "right",
        amount: float = 0.3,
    ) -> List[Image.Image]:
        """
        Pan camera horizontally across image
        """
        width, height = image.size
        frames = []
        
        # Amount to pan (fraction of width)
        pan_distance = int(width * amount)
        
        # Extend image by padding
        extended_width = width + pan_distance
        extended = Image.new('RGB', (extended_width, height))
        extended.paste(image, (0, 0))
        # Mirror edge to fill extended area (simple approach)
        extended.paste(image.crop((width-pan_distance, 0, width, height)), (width, 0))
        
        for t in range(num_frames):
            progress = t / (num_frames - 1)
            
            if direction == "right":
                offset = int(pan_distance * progress)
            else:  # left
                offset = int(pan_distance * (1 - progress))
            
            frame = extended.crop((offset, 0, offset + width, height))
            frames.append(frame)
        
        return frames
    
    @staticmethod
    def add_subtle_motion(
        image: Image.Image,
        num_frames: int = 16,
        motion_type: str = "breathing",
    ) -> List[Image.Image]:
        """
        Add subtle, natural motion (breathing, floating, etc.)
        
        Creates cinemagraph-like effect
        """
        width, height = image.size
        frames = []
        
        for t in range(num_frames):
            # Sine wave for smooth back-and-forth motion
            progress = np.sin(2 * np.pi * t / num_frames)
            
            if motion_type == "breathing":
                # Subtle scale change (1% variation)
                scale = 1.0 + 0.01 * progress
                new_size = (int(width * scale), int(height * scale))
                frame = image.resize(new_size, Image.LANCZOS)
                
                # Center crop back to original size
                left = (frame.width - width) // 2
                top = (frame.height - height) // 2
                frame = frame.crop((left, top, left + width, top + height))
                
            elif motion_type == "floating":
                # Vertical motion
                offset = int(5 * progress)  # +/- 5 pixels
                frame = Image.new('RGB', (width, height))
                frame.paste(image, (0, offset))
            
            else:
                frame = image
            
            frames.append(frame)
        
        return frames

# Example: Motion control demonstration
def demonstrate_motion_controls():
    """Show different motion types"""
    from PIL import Image
    import imageio
    
    # Load test image
    image = Image.open("test_image.jpg")
    
    # Generate different motions
    motions = {
        "zoom_in": MotionControl.zoom_in(image, num_frames=30, zoom_factor=1.5),
        "pan_right": MotionControl.pan_horizontal(image, num_frames=30, direction="right"),
        "breathing": MotionControl.add_subtle_motion(image, num_frames=30, motion_type="breathing"),
    }
    
    # Save as GIFs
    for name, frames in motions.items():
        # Convert PIL images to numpy arrays
        frame_arrays = [np.array(frame) for frame in frames]
        imageio.mimsave(f"{name}.gif", frame_arrays, fps=10)
        print(f"Saved {name}.gif")

if __name__ == "__main__":
    demonstrate_motion_controls()
\`\`\`

---

## Stable Video Diffusion: Deep Dive

### Why SVD Excels at Image-to-Video

Stable Video Diffusion (SVD) was specifically designed for image-to-video:

**Advantages**:
1. **Strong conditioning**: Uses input image as first frame conditioning
2. **Temporal consistency**: Trained specifically for smooth motion
3. **Open source**: Can be customized and self-hosted
4. **High quality**: Rivals commercial solutions

\`\`\`python
"""
Production-Ready Stable Video Diffusion Integration
Complete implementation with all features
"""

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import numpy as np
from typing import Optional, List, Union, Dict
from pathlib import Path
import time
import json

class SVDImageToVideo:
    """
    Production wrapper for Stable Video Diffusion
    
    Features:
    - Automatic image preprocessing
    - Motion control
    - Batch processing
    - Result caching
    - Quality optimization
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        device: str = "cuda",
        enable_optimizations: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        self.device = device
        self.cache_dir = cache_dir or Path("./svd_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        print(f"Loading Stable Video Diffusion: {model_id}")
        
        # Load pipeline
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        
        if enable_optimizations:
            self._apply_optimizations()
        
        self.pipe.to(device)
        
        print("Model loaded successfully!")
    
    def _apply_optimizations(self):
        """Apply memory and speed optimizations"""
        # Enable CPU offloading for large models
        self.pipe.enable_model_cpu_offload()
        
        # Enable VAE slicing for reduced memory
        self.pipe.enable_vae_slicing()
        
        # Enable VAE tiling for very large images
        self.pipe.enable_vae_tiling()
        
        print("✅ Optimizations enabled")
    
    def preprocess_image(
        self,
        image: Union[str, Image.Image],
        target_resolution: tuple[int, int] = (1024, 576),
        maintain_aspect_ratio: bool = True,
    ) -> Image.Image:
        """
        Preprocess image for optimal results
        
        Args:
            image: PIL Image or path
            target_resolution: Target size (width, height)
            maintain_aspect_ratio: Keep original aspect ratio
        
        Returns:
            Preprocessed PIL Image
        """
        # Load if path
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if maintain_aspect_ratio:
            # Resize maintaining aspect ratio
            image.thumbnail(target_resolution, Image.LANCZOS)
            
            # Pad to exact target size
            result = Image.new("RGB", target_resolution, (0, 0, 0))
            offset = (
                (target_resolution[0] - image.width) // 2,
                (target_resolution[1] - image.height) // 2,
            )
            result.paste(image, offset)
            return result
        else:
            # Direct resize
            return image.resize(target_resolution, Image.LANCZOS)
    
    def generate(
        self,
        image: Union[str, Image.Image],
        num_frames: int = 25,
        motion_bucket_id: int = 127,
        fps: int = 7,
        decode_chunk_size: int = 8,
        noise_aug_strength: float = 0.02,
        seed: Optional[int] = None,
        preprocess: bool = True,
    ) -> Dict:
        """
        Generate video from image
        
        Args:
            image: Input image (path or PIL Image)
            num_frames: Number of frames (max 25 for XT model)
            motion_bucket_id: Motion amount (0-255, higher=more motion)
            fps: Frames per second
            decode_chunk_size: Decode frames in chunks (memory optimization)
            noise_aug_strength: Noise augmentation (0-1)
            seed: Random seed for reproducibility
            preprocess: Whether to preprocess image
        
        Returns:
            Dict with frames, metadata, and timing info
        """
        start_time = time.time()
        
        # Preprocess image
        if preprocess:
            image = self.preprocess_image(image)
        
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.manual_seed(seed)
        
        # Generate frames
        print(f"Generating {num_frames} frames (motion={motion_bucket_id})...")
        
        output = self.pipe(
            image=image,
            num_frames=num_frames,
            decode_chunk_size=decode_chunk_size,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            generator=generator,
        )
        
        frames = output.frames[0]
        
        generation_time = time.time() - start_time
        
        print(f"✅ Generated in {generation_time:.1f}s ({generation_time/num_frames:.2f}s per frame)")
        
        return {
            "frames": frames,
            "num_frames": len(frames),
            "fps": fps,
            "motion_bucket_id": motion_bucket_id,
            "generation_time": generation_time,
            "seed": seed,
        }
    
    def generate_with_motion_sweep(
        self,
        image: Union[str, Image.Image],
        motion_levels: List[int] = [50, 127, 200],
        num_frames: int = 14,
        seed: int = 42,
    ) -> Dict[int, List[Image.Image]]:
        """
        Generate videos with different motion levels to find best
        
        Args:
            image: Input image
            motion_levels: List of motion_bucket_id values to try
            num_frames: Frames per video
            seed: Fixed seed for fair comparison
        
        Returns:
            Dict mapping motion_level -> frames
        """
        results = {}
        
        # Preprocess once
        image = self.preprocess_image(image)
        
        for motion_level in motion_levels:
            print(f"\\n🎬 Generating with motion level {motion_level}...")
            
            result = self.generate(
                image=image,
                num_frames=num_frames,
                motion_bucket_id=motion_level,
                seed=seed,
                preprocess=False,  # Already preprocessed
            )
            
            results[motion_level] = result["frames"]
        
        return results
    
    def generate_loop(
        self,
        image: Union[str, Image.Image],
        num_frames: int = 16,
        motion_bucket_id: int = 100,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Generate looping video (seamless)
        
        Uses ping-pong: plays forward then backward
        """
        # Generate forward frames
        result = self.generate(
            image=image,
            num_frames=num_frames,
            motion_bucket_id=motion_bucket_id,
            seed=seed,
        )
        
        frames = result["frames"]
        
        # Create loop: forward + reverse (excluding first/last to avoid duplication)
        loop_frames = frames + frames[-2:0:-1]
        
        print(f"✅ Created loop with {len(loop_frames)} frames")
        
        return loop_frames
    
    def batch_generate(
        self,
        images: List[Union[str, Image.Image]],
        num_frames: int = 14,
        motion_bucket_id: int = 127,
        seed: Optional[int] = None,
    ) -> List[List[Image.Image]]:
        """
        Generate videos for multiple images
        
        Note: Processes sequentially to avoid OOM
        For true parallel processing, use multiple GPUs
        """
        all_results = []
        
        for i, image in enumerate(images):
            print(f"\\n[{i+1}/{len(images)}] Processing image...")
            
            result = self.generate(
                image=image,
                num_frames=num_frames,
                motion_bucket_id=motion_bucket_id,
                seed=seed,
            )
            
            all_results.append(result["frames"])
        
        return all_results
    
    def save_video(
        self,
        frames: List[Image.Image],
        output_path: str,
        fps: int = 7,
        quality: int = 9,  # 0-10, higher is better
    ):
        """
        Save frames as video file
        
        Args:
            frames: List of PIL Images
            output_path: Output file path
            fps: Frames per second
            quality: Video quality (0-10)
        """
        export_to_video(frames, output_path, fps=fps)
        
        # Get file size
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        
        print(f"💾 Saved: {output_path} ({size_mb:.2f} MB, {len(frames)} frames @ {fps}fps)")

# Production usage examples
def production_examples():
    """
    Real-world image-to-video examples
    """
    
    # Initialize generator
    svd = SVDImageToVideo(
        device="cuda",
        enable_optimizations=True,
    )
    
    # Example 1: Product animation for e-commerce
    print("\\n=== Example 1: Product Animation ===")
    
    product_image = "product_photo.jpg"
    
    # Generate with subtle motion
    result = svd.generate(
        image=product_image,
        num_frames=25,
        motion_bucket_id=80,  # Subtle motion
        fps=7,
        seed=42,
    )
    
    svd.save_video(result["frames"], "product_animation.mp4", fps=7)
    
    # Example 2: Portrait animation
    print("\\n=== Example 2: Portrait Animation ===")
    
    portrait_image = "portrait.jpg"
    
    # Very subtle motion for portraits
    result = svd.generate(
        image=portrait_image,
        num_frames=14,
        motion_bucket_id=40,  # Very subtle
        fps=7,
        seed=42,
    )
    
    svd.save_video(result["frames"], "portrait_animation.mp4", fps=7)
    
    # Example 3: Landscape with motion sweep
    print("\\n=== Example 3: Motion Sweep ===")
    
    landscape_image = "landscape.jpg"
    
    # Try different motion levels
    motion_results = svd.generate_with_motion_sweep(
        image=landscape_image,
        motion_levels=[50, 100, 150, 200],
        num_frames=14,
        seed=42,
    )
    
    # Save each version
    for motion_level, frames in motion_results.items():
        output_path = f"landscape_motion_{motion_level}.mp4"
        svd.save_video(frames, output_path)
    
    # Example 4: Looping animation
    print("\\n=== Example 4: Seamless Loop ===")
    
    abstract_image = "abstract_art.jpg"
    
    loop_frames = svd.generate_loop(
        image=abstract_image,
        num_frames=16,
        motion_bucket_id=150,
        seed=42,
    )
    
    svd.save_video(loop_frames, "abstract_loop.mp4", fps=10)
    
    # Example 5: Batch processing
    print("\\n=== Example 5: Batch Processing ===")
    
    image_paths = [
        "image1.jpg",
        "image2.jpg",
        "image3.jpg",
        "image4.jpg",
    ]
    
    batch_results = svd.batch_generate(
        images=image_paths,
        num_frames=14,
        motion_bucket_id=127,
        seed=42,
    )
    
    # Save all videos
    for i, frames in enumerate(batch_results):
        svd.save_video(frames, f"batch_video_{i+1}.mp4")
    
    print("\\n✅ All examples completed!")

if __name__ == "__main__":
    production_examples()
\`\`\`

---

## Advanced Techniques

### 1. Optical Flow-Guided Animation

Use optical flow to control motion direction:

\`\`\`python
"""
Optical flow-guided image-to-video
"""

import cv2
import numpy as np
from PIL import Image

def compute_optical_flow_map(
    image: Image.Image,
    flow_direction: str = "forward",
    flow_magnitude: float = 5.0,
) -> np.ndarray:
    """
    Create optical flow map for motion guidance
    
    Args:
        image: Input image
        flow_direction: "forward", "backward", "left", "right", "zoom_in", "zoom_out"
        flow_magnitude: Strength of flow
    
    Returns:
        Flow map (H, W, 2) with (dx, dy) for each pixel
    """
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Create flow map
    flow = np.zeros((height, width, 2), dtype=np.float32)
    
    if flow_direction == "forward":
        # Zoom in
        center_x, center_y = width // 2, height // 2
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Vector from each pixel to center
        dx = center_x - x_coords
        dy = center_y - y_coords
        
        # Normalize and scale
        magnitude = np.sqrt(dx**2 + dy**2) + 1e-5
        flow[:, :, 0] = (dx / magnitude) * flow_magnitude
        flow[:, :, 1] = (dy / magnitude) * flow_magnitude
    
    elif flow_direction == "right":
        flow[:, :, 0] = flow_magnitude  # All pixels move right
    
    elif flow_direction == "left":
        flow[:, :, 0] = -flow_magnitude
    
    elif flow_direction == "down":
        flow[:, :, 1] = flow_magnitude
    
    elif flow_direction == "up":
        flow[:, :, 1] = -flow_magnitude
    
    return flow

def apply_optical_flow(
    image: Image.Image,
    flow: np.ndarray,
    num_frames: int = 16,
) -> List[Image.Image]:
    """
    Apply optical flow to create animated frames
    
    Simple implementation - production would use neural optical flow
    """
    img_array = np.array(image)
    frames = []
    
    for t in range(num_frames):
        # Gradually apply flow
        progress = t / (num_frames - 1)
        current_flow = flow * progress
        
        # Warp image
        height, width = img_array.shape[:2]
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Apply flow
        new_x = x_coords + current_flow[:, :, 0]
        new_y = y_coords + current_flow[:, :, 1]
        
        # Remap
        warped = cv2.remap(
            img_array,
            new_x.astype(np.float32),
            new_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        
        frames.append(Image.fromarray(warped))
    
    return frames
\`\`\`

### 2. Multi-Step Animation

Chain multiple animations for complex motion:

\`\`\`python
"""
Multi-step animation pipeline
"""

class AnimationPipeline:
    """
    Chain multiple animation steps
    """
    
    def __init__(self, svd: SVDImageToVideo):
        self.svd = svd
    
    def animate_multi_step(
        self,
        image: Image.Image,
        steps: List[Dict],
    ) -> List[Image.Image]:
        """
        Apply multiple animation steps sequentially
        
        Args:
            image: Starting image
            steps: List of animation configurations
                   Each step: {"motion": int, "frames": int}
        
        Returns:
            All frames concatenated
        """
        all_frames = []
        current_image = image
        
        for i, step in enumerate(steps):
            print(f"\\nStep {i+1}/{len(steps)}: motion={step['motion']}")
            
            # Generate from current image
            result = self.svd.generate(
                image=current_image,
                num_frames=step["frames"],
                motion_bucket_id=step["motion"],
                seed=42,
                preprocess=False,
            )
            
            frames = result["frames"]
            all_frames.extend(frames)
            
            # Use last frame as next starting point
            current_image = frames[-1]
        
        return all_frames

# Example: Complex animation
def create_complex_animation():
    """
    Create multi-step animation: zoom in -> pan right -> zoom out
    """
    svd = SVDImageToVideo()
    pipeline = AnimationPipeline(svd)
    
    # Define animation steps
    steps = [
        {"motion": 100, "frames": 10},  # Zoom in
        {"motion": 150, "frames": 15},  # More motion (pan)
        {"motion": 80, "frames": 10},   # Zoom out (subtle)
    ]
    
    frames = pipeline.animate_multi_step(
        image="landscape.jpg",
        steps=steps,
    )
    
    svd.save_video(frames, "complex_animation.mp4", fps=7)
    print(f"Created {len(frames)} frame animation")
\`\`\`

---

## Motion Control Parameters

### Understanding Motion Bucket ID

The \`motion_bucket_id\` parameter (0-255) controls animation intensity:

| Range | Effect | Best For |
|-------|--------|----------|
| 0-40 | Minimal motion | Portraits, subtle breathing |
| 40-80 | Subtle motion | Products, gentle animation |
| 80-127 | Moderate motion | Landscapes, natural scenes |
| 127-180 | Active motion | Action shots, dynamic scenes |
| 180-255 | Maximum motion | Abstract, dramatic effects |

\`\`\`python
"""
Motion bucket experimentation tool
"""

def find_optimal_motion(
    image: Image.Image,
    svd: SVDImageToVideo,
    motion_range: tuple = (20, 200),
    num_samples: int = 5,
) -> int:
    """
    Automatically find optimal motion level for image
    
    Uses perceptual metrics to evaluate motion quality
    """
    from skimage.metrics import structural_similarity as ssim
    
    motion_levels = np.linspace(motion_range[0], motion_range[1], num_samples, dtype=int)
    scores = []
    
    for motion in motion_levels:
        print(f"Testing motion={motion}")
        
        result = svd.generate(
            image=image,
            num_frames=14,
            motion_bucket_id=int(motion),
            seed=42,
        )
        
        frames = result["frames"]
        
        # Calculate motion score
        # Want: noticeable motion but good consistency
        frame_arrays = [np.array(f) for f in frames]
        
        # Measure frame-to-frame differences
        differences = []
        similarities = []
        
        for i in range(len(frame_arrays) - 1):
            diff = np.mean(np.abs(frame_arrays[i+1] - frame_arrays[i]))
            differences.append(diff)
            
            # SSIM for consistency
            sim = ssim(
                frame_arrays[i],
                frame_arrays[i+1],
                multichannel=True,
                channel_axis=2,
                data_range=255,
            )
            similarities.append(sim)
        
        avg_diff = np.mean(differences)
        avg_sim = np.mean(similarities)
        
        # Score: balance between motion and consistency
        # Want high motion (high diff) but high consistency (high sim)
        score = avg_diff * avg_sim
        scores.append(score)
        
        print(f"  Motion: {avg_diff:.2f}, Consistency: {avg_sim:.3f}, Score: {score:.2f}")
    
    # Find best
    best_idx = np.argmax(scores)
    best_motion = motion_levels[best_idx]
    
    print(f"\\n✅ Optimal motion level: {best_motion}")
    
    return int(best_motion)

# Example usage
def auto_optimize_motion():
    """Automatically find best motion for image"""
    svd = SVDImageToVideo()
    
    image = Image.open("test_image.jpg")
    
    optimal_motion = find_optimal_motion(
        image=image,
        svd=svd,
        motion_range=(50, 180),
        num_samples=6,
    )
    
    # Generate final video with optimal motion
    result = svd.generate(
        image=image,
        num_frames=25,
        motion_bucket_id=optimal_motion,
        seed=42,
    )
    
    svd.save_video(result["frames"], "optimized_motion.mp4")
\`\`\`

---

## Production Best Practices

### Image Preparation

**1. Resolution**:
- Use high-resolution inputs (1024x576 minimum)
- Maintain aspect ratio to avoid distortion
- Pre-crop to desired framing

**2. Composition**:
- Clear subject with space for motion
- Good lighting and contrast
- Avoid extreme close-ups for best results

**3. Style**:
- Consistent style across batch
- Professional photography works best
- AI-generated images can work but may have artifacts

### Quality Control

\`\`\`python
"""
Quality control for image-to-video
"""

from typing import Tuple
import numpy as np

class VideoQualityChecker:
    """
    Automated quality checks for generated videos
    """
    
    @staticmethod
    def check_temporal_consistency(
        frames: List[Image.Image],
        threshold: float = 0.85,
    ) -> Tuple[bool, float]:
        """
        Check if frames are temporally consistent
        
        Returns: (is_consistent, average_similarity)
        """
        from skimage.metrics import structural_similarity as ssim
        
        similarities = []
        frame_arrays = [np.array(f) for f in frames]
        
        for i in range(len(frame_arrays) - 1):
            sim = ssim(
                frame_arrays[i],
                frame_arrays[i+1],
                multichannel=True,
                channel_axis=2,
                data_range=255,
            )
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        is_consistent = avg_similarity >= threshold
        
        return is_consistent, avg_similarity
    
    @staticmethod
    def check_motion_amount(
        frames: List[Image.Image],
        min_motion: float = 5.0,
        max_motion: float = 100.0,
    ) -> Tuple[bool, float]:
        """
        Check if motion amount is appropriate
        
        Returns: (is_good, average_motion)
        """
        frame_arrays = [np.array(f) for f in frames]
        
        motions = []
        for i in range(len(frame_arrays) - 1):
            motion = np.mean(np.abs(frame_arrays[i+1] - frame_arrays[i]))
            motions.append(motion)
        
        avg_motion = np.mean(motions)
        is_good = min_motion <= avg_motion <= max_motion
        
        return is_good, avg_motion
    
    @staticmethod
    def detect_artifacts(
        frames: List[Image.Image],
    ) -> Tuple[bool, List[int]]:
        """
        Detect visual artifacts or glitches
        
        Returns: (has_artifacts, frame_indices_with_artifacts)
        """
        artifact_frames = []
        frame_arrays = [np.array(f) for f in frames]
        
        for i in range(1, len(frame_arrays) - 1):
            # Check for sudden changes (potential artifacts)
            diff_prev = np.mean(np.abs(frame_arrays[i] - frame_arrays[i-1]))
            diff_next = np.mean(np.abs(frame_arrays[i+1] - frame_arrays[i]))
            
            # If one frame has much higher difference than others
            if diff_prev > 50 or diff_next > 50:
                artifact_frames.append(i)
        
        has_artifacts = len(artifact_frames) > 0
        
        return has_artifacts, artifact_frames
    
    @classmethod
    def full_quality_check(
        cls,
        frames: List[Image.Image],
    ) -> Dict:
        """
        Run all quality checks
        
        Returns: Dict with all check results
        """
        is_consistent, consistency_score = cls.check_temporal_consistency(frames)
        is_good_motion, motion_score = cls.check_motion_amount(frames)
        has_artifacts, artifact_frames = cls.detect_artifacts(frames)
        
        passed = is_consistent and is_good_motion and not has_artifacts
        
        return {
            "passed": passed,
            "consistency": {
                "passed": is_consistent,
                "score": consistency_score,
            },
            "motion": {
                "passed": is_good_motion,
                "score": motion_score,
            },
            "artifacts": {
                "detected": has_artifacts,
                "frames": artifact_frames,
            },
        }

# Example: Quality control in production
def production_with_qc():
    """Generate video with automatic quality control"""
    svd = SVDImageToVideo()
    qc = VideoQualityChecker()
    
    image = Image.open("input.jpg")
    
    # Generate
    result = svd.generate(
        image=image,
        num_frames=25,
        motion_bucket_id=127,
    )
    
    frames = result["frames"]
    
    # Quality check
    qc_results = qc.full_quality_check(frames)
    
    if qc_results["passed"]:
        print("✅ Quality check passed")
        svd.save_video(frames, "output.mp4")
    else:
        print("⚠️  Quality issues detected:")
        if not qc_results["consistency"]["passed"]:
            print(f"   - Low consistency: {qc_results['consistency']['score']:.3f}")
        if not qc_results["motion"]["passed"]:
            print(f"   - Motion out of range: {qc_results['motion']['score']:.2f}")
        if qc_results["artifacts"]["detected"]:
            print(f"   - Artifacts in frames: {qc_results['artifacts']['frames']}")
        
        # Retry with different parameters
        print("\\nRetrying with adjusted parameters...")
        # Implementation of retry logic
\`\`\`

---

## Summary

**Key Takeaways:**
- Image-to-video offers more control than text-to-video
- Stable Video Diffusion excels at image animation
- Motion bucket ID controls animation intensity
- Quality control is essential for production
- Batch processing enables scale

**Production Checklist:**
- ✅ Preprocess images (resolution, aspect ratio)
- ✅ Test multiple motion levels
- ✅ Implement quality checks
- ✅ Cache results to avoid regeneration
- ✅ Monitor costs and generation times

**Next Steps:**
- Experiment with different motion levels for your content type
- Build automated pipeline with quality control
- Explore optical flow for advanced control
`,
  exercises: [
    {
      title: 'Exercise 1: Motion Optimizer',
      id: 'image-to-video-animation',
      difficulty: 'intermediate' as const,
      description:
        'Build an automated tool that tests multiple motion levels and selects the best one based on perceptual quality metrics.',
      hints: [
        'Use SSIM for temporal consistency',
        'Calculate frame-to-frame differences for motion amount',
        'Balance motion vs consistency',
        'Create visualization of results',
      ],
    },
    {
      title: 'Exercise 2: Batch Image-to-Video Pipeline',
      id: 'image-to-video-animation',
      difficulty: 'advanced' as const,
      description:
        'Create a production pipeline that processes folders of images, applies optimal motion, runs quality checks, and organizes outputs.',
      hints: [
        'Watch folder for new images',
        'Auto-detect image type (portrait, landscape, product)',
        'Apply appropriate motion levels per type',
        'Reject low-quality outputs automatically',
      ],
    },
  ],
};
