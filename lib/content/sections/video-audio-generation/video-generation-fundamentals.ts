export const videoGenerationFundamentals = {
  title: 'Video Generation Fundamentals',
  id: 'video-generation-fundamentals',
  content: `
# Video Generation Fundamentals

## Introduction

Video generation represents one of the most exciting and challenging frontiers in AI. While image generation has matured with tools like DALL-E and Stable Diffusion, **video generation** introduces an entirely new dimension: **time**. This temporal dimension brings complexities around motion consistency, frame-to-frame coherence, physics understanding, and computational requirements that are orders of magnitude higher than static images.

In this section, we'll explore the fundamentals of how AI generates videos from text prompts, understand the underlying architectures, examine the current state-of-the-art models, and learn how systems like **OpenAI's Sora** likely work under the hood.

### Why Video Generation Matters

Video is the dominant medium for content consumption:
- **YouTube**: 500+ hours uploaded per minute
- **TikTok**: 1 billion+ active users
- **Netflix**: 230+ million subscribers
- **Marketing**: Video content has 1200% more shares than text and images combined

The ability to generate videos programmatically unlocks:
- **Content creation at scale**: Generate marketing videos, explainer content, educational materials
- **Personalization**: Create unique videos for each user
- **Rapid prototyping**: Test concepts before expensive production
- **Accessibility**: Enable anyone to create professional-looking videos
- **Creative exploration**: Visualize concepts that would be impossible or expensive to film

### The Challenge: Why Video is Harder Than Images

Generating a single high-quality image is difficult. Generating video is exponentially harder:

1. **Temporal Consistency**: Objects must maintain their identity and appearance across hundreds of frames
2. **Motion Physics**: Movements must obey physical laws (gravity, momentum, collision)
3. **Narrative Coherence**: Scenes must tell a coherent story over time
4. **Computational Cost**: A 5-second video at 24fps is 120 images that must be coherent
5. **Memory Requirements**: Models must "remember" what happened in previous frames
6. **Complex Prompts**: Text descriptions must specify both spatial AND temporal information

A single mistake in one frame can break immersion for the entire video.

---

## How Video Generation Works: Core Concepts

### Text-to-Video Overview

The basic pipeline for text-to-video generation:

\`\`\`
Text Prompt → Text Encoder → Video Generation Model → Frame Decoder → Video
     ↓              ↓                    ↓                    ↓            ↓
"A dog runs"   Embeddings      Latent Space Video    Individual Frames   .mp4
  (input)      (semantic)       (compressed)          (decoded)         (output)
\`\`\`

This pipeline involves several key components:

1. **Text Encoding**: Convert the prompt into a semantic representation
2. **Temporal Modeling**: Generate a sequence of latent representations that evolve over time
3. **Spatial Modeling**: Ensure each frame is visually coherent
4. **Frame Decoding**: Convert latent representations to actual pixels
5. **Post-Processing**: Upscaling, smoothing, audio addition

### Diffusion Models for Video

Most modern video generation systems (including Sora) are based on **diffusion models**:

**How Diffusion Works for Video:**

1. **Forward Process (Training)**:
   - Start with real video frames
   - Gradually add noise over many steps
   - Train model to predict and remove noise

2. **Reverse Process (Generation)**:
   - Start with pure noise
   - Iteratively denoise to create coherent video
   - Use text prompt to guide denoising direction

3. **Temporal Attention**:
   - Frames attend to adjacent frames
   - Maintains consistency across time
   - Can be causal (past only) or bidirectional

\`\`\`python
"""
Conceptual Video Diffusion Model Architecture
"""

import torch
import torch.nn as nn

class VideoAttentionBlock (nn.Module):
    """
    Attention block that handles both spatial and temporal dimensions
    """
    def __init__(self, channels: int, num_frames: int):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        
        # Spatial attention (within each frame)
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=8,
            batch_first=True
        )
        
        # Temporal attention (across frames)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=8,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm (channels)
        self.norm2 = nn.LayerNorm (channels)
        
    def forward (self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, frames, height, width, channels)
        """
        batch, frames, height, width, channels = x.shape
        
        # Spatial attention: attend within each frame
        # Reshape to (batch * frames, height * width, channels)
        x_spatial = x.reshape (batch * frames, height * width, channels)
        x_spatial_attended, _ = self.spatial_attention(
            x_spatial, x_spatial, x_spatial
        )
        x_spatial_attended = self.norm1(x_spatial_attended + x_spatial)
        
        # Reshape back to (batch, frames, height, width, channels)
        x = x_spatial_attended.reshape (batch, frames, height, width, channels)
        
        # Temporal attention: attend across frames
        # Reshape to (batch, height * width, frames, channels)
        # Then to (batch * height * width, frames, channels)
        x_temporal = x.permute(0, 2, 3, 1, 4)  # (batch, height, width, frames, channels)
        x_temporal = x_temporal.reshape (batch * height * width, frames, channels)
        
        x_temporal_attended, _ = self.temporal_attention(
            x_temporal, x_temporal, x_temporal
        )
        x_temporal_attended = self.norm2(x_temporal_attended + x_temporal)
        
        # Reshape back to original shape
        x_temporal_attended = x_temporal_attended.reshape(
            batch, height, width, frames, channels
        )
        x_out = x_temporal_attended.permute(0, 3, 1, 2, 4)  # Back to (batch, frames, h, w, c)
        
        return x_out

class SimplifiedVideoDiffusionModel (nn.Module):
    """
    Simplified video diffusion model showing key concepts
    """
    def __init__(
        self,
        num_frames: int = 16,
        frame_size: int = 64,
        channels: int = 3,
        latent_channels: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.frame_size = frame_size
        
        # Encode frames to latent space
        self.encoder = nn.Sequential(
            nn.Conv3d (channels, latent_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d (latent_channels // 4, latent_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d (latent_channels // 2, latent_channels, kernel_size=3, padding=1),
        )
        
        # Text condition embedding
        self.text_embedding = nn.Linear(768, latent_channels)  # Assuming CLIP embeddings
        
        # Timestep embedding (for diffusion steps)
        self.time_embedding = nn.Embedding(1000, latent_channels)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            VideoAttentionBlock (latent_channels, num_frames)
            for _ in range (num_layers)
        ])
        
        # Decode latent to frames
        self.decoder = nn.Sequential(
            nn.Conv3d (latent_channels, latent_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d (latent_channels // 2, latent_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d (latent_channels // 4, channels, kernel_size=3, padding=1),
        )
        
    def forward(
        self,
        noisy_video: torch.Tensor,  # (batch, channels, frames, height, width)
        text_embedding: torch.Tensor,  # (batch, 768)
        timestep: torch.Tensor,  # (batch,)
    ) -> torch.Tensor:
        """
        Predict the noise to remove from noisy_video
        """
        # Encode video to latent space
        latent = self.encoder (noisy_video)  # (batch, latent_channels, frames, h, w)
        
        # Add text condition
        text_cond = self.text_embedding (text_embedding)  # (batch, latent_channels)
        text_cond = text_cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Broadcast dims
        latent = latent + text_cond
        
        # Add timestep embedding
        time_emb = self.time_embedding (timestep)  # (batch, latent_channels)
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Broadcast dims
        latent = latent + time_emb
        
        # Reshape for attention: (batch, frames, height, width, channels)
        batch, c, f, h, w = latent.shape
        latent = latent.permute(0, 2, 3, 4, 1)  # (batch, frames, h, w, channels)
        
        # Apply attention layers
        for attention_layer in self.attention_layers:
            latent = attention_layer (latent)
        
        # Reshape back: (batch, channels, frames, height, width)
        latent = latent.permute(0, 4, 1, 2, 3)
        
        # Decode to predicted noise
        predicted_noise = self.decoder (latent)
        
        return predicted_noise

# Example usage
def generate_video_from_text(
    prompt: str,
    model: SimplifiedVideoDiffusionModel,
    num_inference_steps: int = 50,
) -> torch.Tensor:
    """
    Generate video using diffusion process
    """
    device = next (model.parameters()).device
    
    # Get text embedding (simplified - would use CLIP or similar)
    # text_encoder would be a pretrained model like CLIP
    text_embedding = torch.randn(1, 768).to (device)  # Placeholder
    
    # Start with pure noise
    video = torch.randn(
        1, 3, model.num_frames, model.frame_size, model.frame_size
    ).to (device)
    
    # Diffusion process: gradually denoise
    for t in range (num_inference_steps - 1, -1, -1):
        timestep = torch.tensor([t]).to (device)
        
        # Predict noise
        with torch.no_grad():
            predicted_noise = model (video, text_embedding, timestep)
        
        # Remove a portion of the noise
        alpha = 1.0 - (t / num_inference_steps)
        video = video - alpha * predicted_noise
        
        # Add smaller noise for next iteration (except last step)
        if t > 0:
            noise_scale = 0.1 * (t / num_inference_steps)
            video = video + noise_scale * torch.randn_like (video)
    
    return video

# Demonstration
if __name__ == "__main__":
    # Create model
    model = SimplifiedVideoDiffusionModel(
        num_frames=16,
        frame_size=64,
        channels=3,
        latent_channels=256,
        num_layers=4,
    )
    
    print(f"Model parameters: {sum (p.numel() for p in model.parameters()):,}")
    
    # Generate video
    video = generate_video_from_text(
        prompt="A dog running through a field",
        model=model,
        num_inference_steps=50,
    )
    
    print(f"Generated video shape: {video.shape}")  # (1, 3, 16, 64, 64)
    print(f"That\'s {video.shape[2]} frames of {video.shape[3]}x{video.shape[4]} video")
\`\`\`

### Key Architectural Components

**1. 3D Convolutions**
- Process spatial dimensions (height, width) AND temporal dimension (frames)
- Kernel moves through both space and time
- Captures motion patterns

**2. Temporal Attention**
- Allows frames to reference other frames
- Maintains object identity across time
- Can be causal (past frames only) or bidirectional

**3. Latent Space Compression**
- Videos are huge (1080p, 30fps = 62M pixels/sec)
- Work in compressed "latent space" (like Stable Diffusion)
- Generate in latent space, then decode to pixels

**4. Text Conditioning**
- Text embeddings guide generation
- Injected at multiple layers
- Controls both content and motion

---

## How Sora Likely Works

OpenAI's **Sora** (announced February 2024) represents a major breakthrough in video generation. While the full technical details aren't public, we can infer the architecture from the demo videos and OpenAI's description:

### Sora\'s Key Innovations

**1. Spacetime Patches**

Sora treats videos as collections of "spacetime patches" - chunks of space and time together:

\`\`\`python
"""
Spacetime Patch Tokenization (Sora-inspired)
"""

import torch
import torch.nn as nn
from typing import Tuple

class SpacetimePatchEmbedding (nn.Module):
    """
    Convert video into spacetime patches similar to how ViT tokenizes images
    """
    def __init__(
        self,
        video_size: Tuple[int, int] = (256, 256),  # (height, width)
        num_frames: int = 16,
        patch_size: Tuple[int, int] = (16, 16),  # Spatial patch size
        temporal_patch_size: int = 2,  # Frames per patch
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.video_size = video_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        
        # Calculate number of patches
        self.num_spatial_patches = (
            (video_size[0] // patch_size[0]) * (video_size[1] // patch_size[1])
        )
        self.num_temporal_patches = num_frames // temporal_patch_size
        self.total_patches = self.num_spatial_patches * self.num_temporal_patches
        
        # 3D convolution to create spacetime patches
        self.projection = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(temporal_patch_size, patch_size[0], patch_size[1]),
            stride=(temporal_patch_size, patch_size[0], patch_size[1]),
        )
        
        # Positional embeddings for each patch
        self.positional_embedding = nn.Parameter(
            torch.randn(1, self.total_patches, embed_dim)
        )
        
    def forward (self, video: torch.Tensor) -> torch.Tensor:
        """
        video: (batch, channels, frames, height, width)
        Returns: (batch, num_patches, embed_dim)
        """
        batch_size = video.shape[0]
        
        # Project to patches
        patches = self.projection (video)  # (batch, embed_dim, t_patches, h_patches, w_patches)
        
        # Flatten spatial and temporal dimensions
        patches = patches.flatten(2)  # (batch, embed_dim, total_patches)
        patches = patches.transpose(1, 2)  # (batch, total_patches, embed_dim)
        
        # Add positional embeddings
        patches = patches + self.positional_embedding
        
        return patches

class SpacetimeTransformer (nn.Module):
    """
    Transformer that processes spacetime patches
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * mlp_ratio,
                batch_first=True,
                norm_first=True,
            )
            for _ in range (num_layers)
        ])
        
    def forward (self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (batch, num_patches, embed_dim)
        """
        x = patches
        for layer in self.layers:
            x = layer (x)
        return x

class SoraLikeModel (nn.Module):
    """
    Simplified Sora-like architecture
    """
    def __init__(
        self,
        video_size: Tuple[int, int] = (256, 256),
        num_frames: int = 16,
        patch_size: Tuple[int, int] = (16, 16),
        temporal_patch_size: int = 2,
        embed_dim: int = 768,
        num_layers: int = 24,
        num_heads: int = 16,
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = SpacetimePatchEmbedding(
            video_size=video_size,
            num_frames=num_frames,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=3,
            embed_dim=embed_dim,
        )
        
        # Text conditioning
        self.text_projection = nn.Linear(768, embed_dim)  # CLIP text embeddings
        
        # Diffusion timestep embedding
        self.time_embed = nn.Sequential(
            nn.Embedding(1000, embed_dim),
            nn.SiLU(),
            nn.Linear (embed_dim, embed_dim),
        )
        
        # Spacetime transformer
        self.transformer = SpacetimeTransformer(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        
        # Decoder to reconstruct video from patches
        self.decoder = nn.Linear (embed_dim, 3 * temporal_patch_size * patch_size[0] * patch_size[1])
        
        self.video_size = video_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        
    def forward(
        self,
        noisy_video: torch.Tensor,
        text_embedding: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise to remove from noisy_video
        """
        batch_size = noisy_video.shape[0]
        
        # Convert video to spacetime patches
        patches = self.patch_embed (noisy_video)  # (batch, num_patches, embed_dim)
        
        # Add text conditioning (broadcast to all patches)
        text_cond = self.text_projection (text_embedding)  # (batch, embed_dim)
        text_cond = text_cond.unsqueeze(1)  # (batch, 1, embed_dim)
        patches = patches + text_cond
        
        # Add timestep conditioning
        time_emb = self.time_embed (timestep)  # (batch, embed_dim)
        time_emb = time_emb.unsqueeze(1)  # (batch, 1, embed_dim)
        patches = patches + time_emb
        
        # Process through transformer
        patches = self.transformer (patches)
        
        # Decode patches back to video
        pixel_values = self.decoder (patches)  # (batch, num_patches, patch_dim)
        
        # Reshape to video
        # This is simplified - actual implementation would be more complex
        predicted_noise = self.unpatchify (pixel_values)
        
        return predicted_noise
    
    def unpatchify (self, patches: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to video tensor
        patches: (batch, num_patches, patch_dim)
        """
        batch_size = patches.shape[0]
        
        # Calculate dimensions
        h_patches = self.video_size[0] // self.patch_size[0]
        w_patches = self.video_size[1] // self.patch_size[1]
        t_patches = self.num_frames // self.temporal_patch_size
        
        # Reshape to spatial-temporal grid
        patches = patches.reshape(
            batch_size, t_patches, h_patches, w_patches, -1
        )
        
        # Reshape to separate channels and patch dimensions
        patches = patches.reshape(
            batch_size, t_patches, h_patches, w_patches,
            3, self.temporal_patch_size, self.patch_size[0], self.patch_size[1]
        )
        
        # Reorganize to video format
        video = patches.permute(0, 4, 1, 5, 2, 6, 3, 7)
        video = video.reshape(
            batch_size, 3,
            self.num_frames,
            self.video_size[0],
            self.video_size[1]
        )
        
        return video

# Example usage
if __name__ == "__main__":
    model = SoraLikeModel(
        video_size=(256, 256),
        num_frames=16,
        patch_size=(16, 16),
        temporal_patch_size=2,
        embed_dim=1024,
        num_layers=24,
        num_heads=16,
    )
    
    print(f"Model parameters: {sum (p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    noisy_video = torch.randn(1, 3, 16, 256, 256)
    text_embedding = torch.randn(1, 768)
    timestep = torch.tensor([500])
    
    output = model (noisy_video, text_embedding, timestep)
    print(f"Output shape: {output.shape}")
\`\`\`

**2. Variable-Length Videos**

Sora can generate videos of different lengths and resolutions by working with variable numbers of patches. This is similar to how text transformers handle variable-length sequences.

**3. World Simulation**

Sora appears to learn a "world model" - an understanding of:
- Physics (gravity, momentum, collision)
- Object persistence (things don't disappear randomly)
- 3D space (occlusion, perspective, depth)
- Causality (actions have consequences)

**4. Massive Scale**

Estimated specifications:
- **Training data**: Millions of hours of high-quality video
- **Model size**: Likely 10B+ parameters (similar to GPT-3.5)
- **Compute**: Thousands of GPUs for months
- **Cost**: Estimated $50-100M+ in compute alone

---

## Current Video Generation Models

### Model Landscape (as of 2024)

| Model | Company | Released | Max Length | Resolution | Public Access |
|-------|---------|----------|------------|------------|---------------|
| **Sora** | OpenAI | 2024 (demo) | 60s | 1080p | Not yet |
| **Runway Gen-2** | Runway | 2023 | 16s | 720p | ✅ API |
| **Pika** | Pika Labs | 2023 | 3s | 720p | ✅ Web |
| **Stable Video Diffusion** | Stability AI | 2023 | 4s | 1024x576 | ✅ Open source |
| **AnimateDiff** | Community | 2023 | Variable | 512x512 | ✅ Open source |
| **Make-A-Video** | Meta | 2022 | 5s | 768x768 | ❌ Research only |
| **Imagen Video** | Google | 2022 | 5s | 1280x768 | ❌ Research only |

### Comparison of Approaches

\`\`\`python
"""
Production-ready interface for multiple video generation services
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union
import requests
import time

@dataclass
class VideoGenerationResult:
    """Result from video generation"""
    video_url: str
    video_id: str
    duration: float  # seconds
    resolution: tuple[int, int]  # (width, height)
    status: str  # "completed", "processing", "failed"
    error: Optional[str] = None

class VideoGenerationService(ABC):
    """Abstract base class for video generation services"""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        duration: float = 4.0,
        resolution: tuple[int, int] = (1024, 576),
        **kwargs
    ) -> VideoGenerationResult:
        """Generate video from text prompt"""
        pass
    
    @abstractmethod
    def get_status (self, video_id: str) -> VideoGenerationResult:
        """Check status of video generation"""
        pass

class RunwayGen2Service(VideoGenerationService):
    """
    Runway Gen-2 API integration
    Best for: High-quality, production-ready videos
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.runwayml.com/v1"
        
    def generate(
        self,
        prompt: str,
        duration: float = 4.0,
        resolution: tuple[int, int] = (1280, 768),
        image_prompt: Optional[str] = None,  # For image-to-video
        style: str = "realistic",
        **kwargs
    ) -> VideoGenerationResult:
        """
        Generate video using Runway Gen-2
        
        Args:
            prompt: Text description of desired video
            duration: Length in seconds (max 16s)
            resolution: Output resolution
            image_prompt: Optional starting image URL
            style: "realistic", "anime", "cinematic", etc.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": prompt,
            "duration": min (duration, 16.0),  # Max 16 seconds
            "resolution": f"{resolution[0]}x{resolution[1]}",
            "style": style,
        }
        
        if image_prompt:
            payload["image_prompt"] = image_prompt
        
        # Submit generation request
        response = requests.post(
            f"{self.base_url}/generations",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        video_id = result["id"]
        
        # Poll for completion
        return self._wait_for_completion (video_id)
    
    def get_status (self, video_id: str) -> VideoGenerationResult:
        """Check generation status"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        response = requests.get(
            f"{self.base_url}/generations/{video_id}",
            headers=headers
        )
        response.raise_for_status()
        
        data = response.json()
        
        return VideoGenerationResult(
            video_url=data.get("url", ""),
            video_id=video_id,
            duration=data.get("duration", 0),
            resolution=tuple (map (int, data.get("resolution", "0x0").split("x"))),
            status=data["status"],
            error=data.get("error")
        )
    
    def _wait_for_completion(
        self,
        video_id: str,
        max_wait: int = 300,
        poll_interval: int = 5
    ) -> VideoGenerationResult:
        """Wait for video generation to complete"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            result = self.get_status (video_id)
            
            if result.status == "completed":
                return result
            elif result.status == "failed":
                raise Exception (f"Generation failed: {result.error}")
            
            time.sleep (poll_interval)
        
        raise TimeoutError (f"Generation timed out after {max_wait} seconds")

class StableVideoDiffusionService(VideoGenerationService):
    """
    Stable Video Diffusion (open source)
    Best for: Free, self-hosted, customizable
    """
    def __init__(self, model_path: str = "stabilityai/stable-video-diffusion-img2vid"):
        self.model_path = model_path
        # Would load model here in real implementation
        
    def generate(
        self,
        prompt: str,
        duration: float = 4.0,
        resolution: tuple[int, int] = (1024, 576),
        starting_image: Optional[str] = None,
        fps: int = 7,
        **kwargs
    ) -> VideoGenerationResult:
        """
        Generate video using Stable Video Diffusion
        
        Note: SVD primarily works with image-to-video
        Text-to-video requires first generating image with Stable Diffusion
        """
        # In real implementation, would:
        # 1. Generate initial image from prompt (if not provided)
        # 2. Use SVD to animate the image
        # 3. Return result
        
        # Placeholder implementation
        return VideoGenerationResult(
            video_url="/path/to/generated/video.mp4",
            video_id="svd_" + str (time.time()),
            duration=duration,
            resolution=resolution,
            status="completed"
        )
    
    def get_status (self, video_id: str) -> VideoGenerationResult:
        """Local generation completes immediately"""
        # Would return actual status in real implementation
        pass

# Usage example
def demonstrate_video_generation():
    """
    Demonstrate using different video generation services
    """
    # Example with Runway Gen-2
    runway = RunwayGen2Service (api_key="your_api_key_here")
    
    try:
        print("Generating video with Runway Gen-2...")
        result = runway.generate(
            prompt="A golden retriever puppy playing in a sunny garden, slow motion",
            duration=4.0,
            resolution=(1280, 768),
            style="realistic"
        )
        
        print(f"✅ Video generated successfully!")
        print(f"   URL: {result.video_url}")
        print(f"   Duration: {result.duration}s")
        print(f"   Resolution: {result.resolution[0]}x{result.resolution[1]}")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
    
    # Example with image-to-video
    try:
        print("\\nGenerating video from image...")
        result = runway.generate(
            prompt="Camera slowly zooms in, dramatic lighting",
            duration=4.0,
            image_prompt="https://example.com/starting-image.jpg"
        )
        
        print(f"✅ Video generated from image!")
        print(f"   URL: {result.video_url}")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")

if __name__ == "__main__":
    demonstrate_video_generation()
\`\`\`

---

## Consistency Across Frames: The Core Challenge

Maintaining consistency is the fundamental challenge in video generation:

### Types of Consistency

1. **Identity Consistency**: Characters/objects maintain appearance
2. **Motion Consistency**: Movement follows physical laws
3. **Style Consistency**: Visual style doesn't randomly change
4. **Lighting Consistency**: Light sources remain coherent
5. **Geometric Consistency**: 3D structure remains valid

### Techniques for Ensuring Consistency

\`\`\`python
"""
Techniques for temporal consistency in video generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConsistencyLoss (nn.Module):
    """
    Loss function to enforce temporal consistency between frames
    """
    def __init__(self, lambda_flow: float = 1.0, lambda_pixel: float = 1.0):
        super().__init__()
        self.lambda_flow = lambda_flow
        self.lambda_pixel = lambda_pixel
        
    def compute_optical_flow_loss(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        predicted_flow: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensure predicted optical flow matches actual frame difference
        """
        # Warp frame1 using predicted flow
        warped_frame1 = self.warp_frame (frame1, predicted_flow)
        
        # Compute photometric loss
        photometric_loss = F.l1_loss (warped_frame1, frame2)
        
        # Compute flow smoothness loss
        flow_smoothness = self.compute_smoothness_loss (predicted_flow)
        
        return photometric_loss + 0.1 * flow_smoothness
    
    def compute_feature_consistency(
        self,
        features_t: torch.Tensor,
        features_t_plus_1: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensure feature representations are consistent between adjacent frames
        """
        # Cosine similarity between feature vectors
        similarity = F.cosine_similarity(
            features_t.flatten(2),
            features_t_plus_1.flatten(2),
            dim=2
        )
        
        # We want high similarity (close to 1)
        consistency_loss = 1.0 - similarity.mean()
        
        return consistency_loss
    
    def compute_smoothness_loss (self, flow: torch.Tensor) -> torch.Tensor:
        """
        Penalize abrupt changes in optical flow (encourages smooth motion)
        """
        # Compute gradients
        flow_dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        flow_dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        
        # L1 norm of gradients
        smoothness = flow_dx.abs().mean() + flow_dy.abs().mean()
        
        return smoothness
    
    def warp_frame(
        self,
        frame: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """
        Warp frame according to optical flow field
        """
        batch, channels, height, width = frame.shape
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange (height, device=frame.device),
            torch.arange (width, device=frame.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        grid = grid.unsqueeze(0).repeat (batch, 1, 1, 1)
        
        # Add flow to grid
        new_grid = grid + flow
        
        # Normalize grid to [-1, 1]
        new_grid[:, 0] = 2.0 * new_grid[:, 0] / (width - 1) - 1.0
        new_grid[:, 1] = 2.0 * new_grid[:, 1] / (height - 1) - 1.0
        
        # Permute for grid_sample
        new_grid = new_grid.permute(0, 2, 3, 1)
        
        # Sample using bilinear interpolation
        warped = F.grid_sample(
            frame,
            new_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return warped
    
    def forward(
        self,
        frames: torch.Tensor,  # (batch, num_frames, channels, height, width)
        predicted_flows: torch.Tensor,  # Optical flows between frames
        features: torch.Tensor,  # Feature representations
    ) -> torch.Tensor:
        """
        Compute total temporal consistency loss
        """
        num_frames = frames.shape[1]
        total_loss = 0.0
        
        # Iterate through adjacent frame pairs
        for t in range (num_frames - 1):
            frame_t = frames[:, t]
            frame_t_plus_1 = frames[:, t + 1]
            flow_t = predicted_flows[:, t]
            features_t = features[:, t]
            features_t_plus_1 = features[:, t + 1]
            
            # Optical flow loss
            flow_loss = self.compute_optical_flow_loss(
                frame_t, frame_t_plus_1, flow_t
            )
            
            # Feature consistency loss
            feature_loss = self.compute_feature_consistency(
                features_t, features_t_plus_1
            )
            
            total_loss += self.lambda_flow * flow_loss + self.lambda_pixel * feature_loss
        
        # Average over all frame pairs
        total_loss /= (num_frames - 1)
        
        return total_loss

# Example: Enforcing consistency during training
class ConsistentVideoGenerator (nn.Module):
    """
    Video generator with explicit consistency enforcement
    """
    def __init__(self):
        super().__init__()
        # Simplified architecture
        self.frame_generator = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )
        
        self.flow_predictor = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),  # 2 frames concatenated
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),  # 2 channels for x,y flow
        )
        
        self.consistency_loss = TemporalConsistencyLoss()
        
    def forward (self, initial_frame: torch.Tensor, num_frames: int = 16):
        """
        Generate video starting from initial frame
        """
        frames = [initial_frame]
        flows = []
        
        # Autoregressive generation
        for t in range (num_frames - 1):
            current_frame = frames[-1]
            
            # Predict next frame
            next_frame = self.frame_generator (current_frame)
            frames.append (next_frame)
            
            # Predict optical flow
            frame_pair = torch.cat([current_frame, next_frame], dim=1)
            flow = self.flow_predictor (frame_pair)
            flows.append (flow)
        
        return torch.stack (frames, dim=1), torch.stack (flows, dim=1)

# Training with consistency loss
def train_with_consistency():
    """
    Training loop that enforces temporal consistency
    """
    model = ConsistentVideoGenerator()
    optimizer = torch.optim.Adam (model.parameters(), lr=1e-4)
    
    # Dummy training data
    real_videos = torch.randn(4, 16, 3, 64, 64)  # (batch, frames, channels, h, w)
    
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Generate video
        initial_frames = real_videos[:, 0]
        generated_frames, predicted_flows = model (initial_frames, num_frames=16)
        
        # Reconstruction loss
        recon_loss = F.mse_loss (generated_frames, real_videos)
        
        # Consistency loss (requires features - simplified here)
        # In practice, would extract features from a pretrained network
        features = generated_frames  # Placeholder
        consistency_loss = model.consistency_loss(
            generated_frames,
            predicted_flows,
            features
        )
        
        # Total loss
        total_loss = recon_loss + 0.5 * consistency_loss
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

if __name__ == "__main__":
    train_with_consistency()
\`\`\`

---

## Temporal Coherence: Physics and Motion

Good video generation must understand and simulate physics:

### Key Physical Concepts

1. **Gravity**: Objects fall downward
2. **Momentum**: Objects in motion tend to stay in motion
3. **Collision**: Objects interact when they touch
4. **Perspective**: Objects appear smaller when farther away
5. **Occlusion**: Closer objects block farther objects

### Limitations of Current Models

Current video generation models often fail at:
- **Complex physics**: Liquid dynamics, cloth simulation, hair movement
- **Long-term coherence**: Objects disappear or morph after several seconds
- **Multi-object interactions**: Difficult to model multiple objects interacting
- **Fine details**: Small details (fingers, text) are often inconsistent

---

## Computational Requirements

Video generation is extremely compute-intensive:

### Sora-Scale Requirements (Estimated)

**Training:**
- **GPUs**: ~10,000 A100 GPUs
- **Duration**: ~3 months
- **Cost**: $50-100M+ in compute
- **Data**: 10M+ hours of video
- **Storage**: 100+ petabytes

**Inference (generating one 60s video):**
- **Time**: 10-30 minutes
- **Cost**: $10-50 per video (estimated)
- **GPU Memory**: 40-80GB
- **Model size**: 10-20 billion parameters

### Optimization Strategies

\`\`\`python
"""
Optimizations for efficient video generation
"""

import torch
from typing import Optional

class EfficientVideoGeneration:
    """
    Techniques for reducing computational cost of video generation
    """
    
    @staticmethod
    def generate_with_keyframes(
        model: nn.Module,
        prompt: str,
        num_frames: int = 120,
        keyframe_interval: int = 8,
    ) -> torch.Tensor:
        """
        Generate only keyframes, then interpolate intermediate frames
        
        This reduces computation by ~4-8x for smooth videos
        """
        # Generate keyframes only
        num_keyframes = num_frames // keyframe_interval
        
        keyframes = []
        for i in range (num_keyframes):
            # Generate keyframe (simplified)
            keyframe = model.generate_single_frame (prompt, frame_index=i * keyframe_interval)
            keyframes.append (keyframe)
        
        # Interpolate between keyframes
        all_frames = []
        for i in range (len (keyframes) - 1):
            # Add keyframe
            all_frames.append (keyframes[i])
            
            # Interpolate intermediate frames
            for j in range(1, keyframe_interval):
                alpha = j / keyframe_interval
                interpolated = EfficientVideoGeneration.interpolate_frames(
                    keyframes[i],
                    keyframes[i + 1],
                    alpha
                )
                all_frames.append (interpolated)
        
        # Add final keyframe
        all_frames.append (keyframes[-1])
        
        return torch.stack (all_frames)
    
    @staticmethod
    def interpolate_frames(
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Interpolate between two frames
        
        In practice, would use optical flow-based interpolation
        """
        # Simple linear interpolation (can be improved with optical flow)
        return (1 - alpha) * frame1 + alpha * frame2
    
    @staticmethod
    def generate_with_progressive_resolution(
        model: nn.Module,
        prompt: str,
        final_resolution: tuple = (1920, 1080),
        num_frames: int = 120,
    ) -> torch.Tensor:
        """
        Generate at low resolution, then upscale
        
        Reduces computation by 16x (for 4x downscale)
        """
        # Start at quarter resolution
        low_res = (final_resolution[0] // 4, final_resolution[1] // 4)
        
        # Generate low-resolution video
        low_res_video = model.generate(
            prompt=prompt,
            resolution=low_res,
            num_frames=num_frames
        )
        
        # Upscale using super-resolution model
        high_res_video = EfficientVideoGeneration.upscale_video(
            low_res_video,
            target_resolution=final_resolution
        )
        
        return high_res_video
    
    @staticmethod
    def upscale_video(
        video: torch.Tensor,
        target_resolution: tuple
    ) -> torch.Tensor:
        """
        Upscale video using spatial super-resolution
        """
        # In practice, would use Real-ESRGAN or similar
        # Here we use simple bilinear interpolation
        return F.interpolate(
            video,
            size=(video.shape[2], target_resolution[0], target_resolution[1]),
            mode='trilinear',
            align_corners=False
        )
    
    @staticmethod
    def generate_with_latent_caching(
        model: nn.Module,
        prompt: str,
        num_frames: int = 120,
    ) -> torch.Tensor:
        """
        Cache intermediate latent representations to avoid recomputation
        """
        latent_cache = {}
        
        frames = []
        for t in range (num_frames):
            # Check if we can reuse cached latents
            cache_key = f"frame_{t}_latent"
            
            if cache_key in latent_cache:
                latent = latent_cache[cache_key]
            else:
                # Compute latent representation
                latent = model.encode_frame (prompt, t)
                
                # Cache for future use
                latent_cache[cache_key] = latent
                
                # Limit cache size
                if len (latent_cache) > 16:
                    # Remove oldest
                    oldest_key = f"frame_{t-16}_latent"
                    latent_cache.pop (oldest_key, None)
            
            # Decode latent to frame
            frame = model.decode_latent (latent)
            frames.append (frame)
        
        return torch.stack (frames)

# Demonstration
def compare_generation_methods():
    """
    Compare efficiency of different generation methods
    """
    import time
    
    # Mock model for demonstration
    class MockVideoModel (nn.Module):
        def generate (self, prompt, resolution, num_frames):
            # Simulate computation time based on resolution and frames
            compute_cost = (resolution[0] * resolution[1] * num_frames) / 1e6
            time.sleep (compute_cost * 0.1)  # Simulate
            return torch.randn(1, 3, num_frames, resolution[1], resolution[0])
        
        def generate_single_frame (self, prompt, frame_index):
            return torch.randn(1, 3, 256, 256)
        
        def encode_frame (self, prompt, t):
            return torch.randn(1, 512)
        
        def decode_latent (self, latent):
            return torch.randn(1, 3, 256, 256)
    
    model = MockVideoModel()
    prompt = "A cat playing with a ball"
    
    print("Comparing generation methods:\\n")
    
    # Method 1: Full resolution, all frames
    start = time.time()
    video1 = model.generate(
        prompt=prompt,
        resolution=(1920, 1080),
        num_frames=120
    )
    time1 = time.time() - start
    print(f"1. Full generation: {time1:.2f}s")
    
    # Method 2: Keyframe + interpolation
    start = time.time()
    video2 = EfficientVideoGeneration.generate_with_keyframes(
        model=model,
        prompt=prompt,
        num_frames=120,
        keyframe_interval=8
    )
    time2 = time.time() - start
    print(f"2. Keyframe generation: {time2:.2f}s (speedup: {time1/time2:.1f}x)")
    
    # Method 3: Progressive resolution
    start = time.time()
    video3 = EfficientVideoGeneration.generate_with_progressive_resolution(
        model=model,
        prompt=prompt,
        final_resolution=(1920, 1080),
        num_frames=120
    )
    time3 = time.time() - start
    print(f"3. Progressive resolution: {time3:.2f}s (speedup: {time1/time3:.1f}x)")
    
    print("\\n💡 Combining methods can give 10-50x speedup!")

if __name__ == "__main__":
    compare_generation_methods()
\`\`\`

---

## Use Cases for Video Generation

### Commercial Applications

1. **Marketing & Advertising**
   - Product demos
   - Social media content
   - Personalized ads

2. **Entertainment**
   - Concept art for films
   - Music videos
   - Animation prototyping

3. **Education**
   - Explainer videos
   - Visual demonstrations
   - Language learning

4. **E-commerce**
   - Product visualizations
   - Virtual try-ons
   - 360° product views

5. **Gaming**
   - Cutscene generation
   - NPC animations
   - Trailer creation

### Quality Assessment

When evaluating generated videos, consider:

- **Temporal Consistency**: Do objects maintain identity?
- **Motion Quality**: Does movement look natural?
- **Visual Fidelity**: Is the image quality high?
- **Prompt Adherence**: Does it match the description?
- **Physics Realism**: Does it obey physical laws?
- **No Artifacts**: Are there glitches or errors?

---

## Summary

Video generation is one of the most complex and exciting areas of AI:

**Key Takeaways:**
- Video adds temporal dimension to image generation challenges
- Diffusion models extended with temporal attention are the current SOTA
- Sora represents a breakthrough with spacetime patches and world modeling
- Consistency across frames is the fundamental challenge
- Computational requirements are massive (billions of parameters, thousands of GPUs)
- Current limitations: physics, long videos, fine details
- Multiple services available: Runway, Pika, Stable Video Diffusion

**Next Steps:**
- In the following sections, we'll explore specific video generation models
- Learn how to use Runway Gen-2, Pika, and Stable Video Diffusion in production
- Understand image-to-video animation techniques
- Build complete video generation applications

The field is evolving rapidly - models that seem impossible today might be available next year!
`,
  exercises: [
    {
      title: 'Exercise 1: Implement Spacetime Patch Embedding',
      id: 'video-generation-fundamentals',
      difficulty: 'intermediate' as const,
      description:
        'Implement a spacetime patch embedding layer that converts video into patches similar to how Sora works. Test it with different patch sizes and visualize the results.',
      hints: [
        'Use 3D convolutions to create spacetime patches',
        'Think about how to handle variable video lengths',
        'Add positional embeddings for both spatial and temporal dimensions',
        'Visualize how many patches are created for different video sizes',
      ],
      solution: `# Detailed solution provided in the main content above
# See the SpacetimePatchEmbedding class implementation`,
    },
    {
      title: 'Exercise 2: Build Temporal Consistency Checker',
      id: 'video-generation-fundamentals',
      difficulty: 'intermediate' as const,
      description:
        'Create a tool that analyzes generated videos for temporal consistency issues. Compute metrics like frame-to-frame similarity, motion smoothness, and object identity persistence.',
      hints: [
        'Use optical flow to track motion between frames',
        'Compute perceptual similarity metrics (LPIPS, SSIM)',
        'Check for sudden appearance/disappearance of objects',
        'Visualize consistency scores over time',
      ],
      solution: `# See the TemporalConsistencyLoss class in the main content
# Extend it to compute metrics rather than losses`,
    },
    {
      title: 'Exercise 3: Cost Estimator for Video Generation',
      id: 'video-generation-fundamentals',
      difficulty: 'beginner' as const,
      description:
        'Build a calculator that estimates the computational cost and time required to generate videos at different resolutions, lengths, and quality settings.',
      hints: [
        'Consider: resolution, frame rate, video length, model size',
        'Include both training and inference costs',
        'Account for different hardware (A100, H100, etc.)',
        'Build a simple web interface for the calculator',
      ],
      solution: `# Create a calculator that multiplies parameters:
# Cost = base_cost * (resolution_factor * frames * quality_multiplier)
# Include real-world pricing from cloud providers`,
    },
  ],
};
