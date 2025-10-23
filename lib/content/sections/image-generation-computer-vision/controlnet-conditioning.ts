/**
 * ControlNet & Conditioning Section
 * Module 8: Image Generation & Computer Vision
 */

export const controlnetconditioningSection = {
  id: 'controlnet-conditioning',
  title: 'ControlNet & Conditioning',
  content: `# ControlNet & Conditioning

Master ControlNet for precise control over image generation using structural guidance.

## Overview: Precise Control Over Generation

ControlNet is a revolutionary technique that gives you pixel-perfect control over image generation by using structural inputs like edges, poses, depth maps, and more.

### The Problem ControlNet Solves

\`\`\`python
without_controlnet = {
    "challenge": "Hard to control exact composition",
    "example": "Want person in specific pose - text prompts are imprecise",
    "result": "Many attempts, inconsistent results",
    "control": "Low - model decides structure"
}

with_controlnet = {
    "solution": "Provide structural guidance image",
    "example": "Give pose skeleton - model follows it exactly",
    "result": "First try success, precise control",
    "control": "High - you decide structure"
}
\`\`\`

### When to Use ControlNet

- **Precise poses**: Character in exact pose
- **Consistent composition**: Multiple images with same layout
- **Architectural control**: Buildings with specific structure
- **Style transfer with structure**: Change style, keep composition
- **Line art to image**: Convert sketches to detailed images

## How ControlNet Works

### Architecture

\`\`\`
Text Prompt ──────┐
                  ↓
Input Image → [Preprocessor] → Condition Image
                                     ↓
                               [ControlNet]
                                     ↓
                               [Stable Diffusion] → Output
\`\`\`

### Available ControlNet Models

\`\`\`python
controlnet_types = {
    "canny": {
        "input": "Edge map (canny edge detection)",
        "use_case": "Preserve outlines and structure",
        "strength": "Strong control over edges",
        "best_for": [
            "Line art to image",
            "Sketch to photo",
            "Structure preservation"
        ]
    },
    
    "depth": {
        "input": "Depth map (near/far information)",
        "use_case": "3D structure control",
        "strength": "Controls spatial relationships",
        "best_for": [
            "Consistent 3D structure",
            "Architectural accuracy",
            "Scene composition"
        ]
    },
    
    "openpose": {
        "input": "Human pose skeleton",
        "use_case": "Exact human poses",
        "strength": "Precise character positioning",
        "best_for": [
            "Character in specific pose",
            "Multiple characters positioned",
            "Pose consistency across generations"
        ]
    },
    
    "scribble": {
        "input": "Rough sketches/scribbles",
        "use_case": "Rough idea to detailed image",
        "strength": "Loose guidance",
        "best_for": [
            "Quick concept sketches",
            "Rough ideas",
            "Creative exploration"
        ]
    },
    
    "mlsd": {
        "input": "Straight line detection",
        "use_case": "Architectural lines",
        "strength": "Preserves straight lines",
        "best_for": [
            "Buildings",
            "Interior design",
            "Geometric structures"
        ]
    },
    
    "normal": {
        "input": "Normal map (surface orientation)",
        "use_case": "Surface detail control",
        "strength": "Detail and texture control",
        "best_for": [
            "Detailed surfaces",
            "Texture control",
            "3D-like detail"
        ]
    },
    
    "segmentation": {
        "input": "Semantic segmentation (regions labeled)",
        "use_case": "Precise region control",
        "strength": "Perfect region placement",
        "best_for": [
            "Complex scene composition",
            "Precise object placement",
            "Architectural planning"
        ]
    }
}
\`\`\`

## Implementation

### Basic ControlNet Usage

\`\`\`python
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
import torch
from PIL import Image
import cv2
import numpy as np
from controlnet_aux import OpenposeDetector, CannyDetector
from typing import Optional

class ControlNetGenerator:
    """
    Generate images with ControlNet guidance.
    """
    
    def __init__(
        self,
        controlnet_type: str = "canny",
        device: str = "cuda"
    ):
        self.device = device
        self.controlnet_type = controlnet_type
        
        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            f"lllyasviel/sd-controlnet-{controlnet_type}",
            torch_dtype=torch.float16
        )
        
        # Load pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        self.pipe = self.pipe.to(device)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()
    
    def generate(
        self,
        prompt: str,
        control_image: Image.Image,
        negative_prompt: str = "",
        num_images: int = 1,
        steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: Optional[int] = None
    ) -> list[Image.Image]:
        """
        Generate with ControlNet conditioning.
        
        Args:
            prompt: Text description
            control_image: Preprocessed control image
            negative_prompt: What to avoid
            num_images: Number to generate
            steps: Quality
            guidance_scale: Prompt adherence
            controlnet_conditioning_scale: How strongly to follow control image (0.0-2.0)
            seed: Reproducibility
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        output = self.pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator
        )
        
        return output.images

# Basic usage
controlnet = ControlNetGenerator(controlnet_type="canny")

# Load and prepare control image
input_image = Image.open("photo.jpg")
# For canny, we need edge detection (see preprocessing section)

# Generate
results = controlnet.generate(
    prompt="a professional photograph of a person in a studio",
    control_image=control_image,  # preprocessed edges
    steps=20,
    controlnet_conditioning_scale=1.0
)

results[0].save("controlnet_output.png")
\`\`\`

### Preprocessing Control Images

\`\`\`python
class ControlImagePreprocessor:
    """
    Preprocess images for different ControlNet types.
    """
    
    def __init__(self):
        # Initialize detectors as needed
        pass
    
    def canny_edges(
        self,
        image: Image.Image,
        low_threshold: int = 100,
        high_threshold: int = 200
    ) -> Image.Image:
        """
        Extract edges using Canny edge detection.
        """
        # Convert to numpy
        image_np = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Convert back to PIL
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)
    
    def depth_map(
        self,
        image: Image.Image,
        use_midas: bool = True
    ) -> Image.Image:
        """
        Generate depth map from image.
        """
        from transformers import pipeline
        
        # Use MiDaS depth estimation
        depth_estimator = pipeline("depth-estimation")
        
        depth = depth_estimator(image)["depth"]
        
        # Normalize to 0-255
        depth_np = np.array(depth)
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        depth_np = (depth_np * 255).astype(np.uint8)
        
        return Image.fromarray(depth_np)
    
    def openpose_skeleton(
        self,
        image: Image.Image
    ) -> Image.Image:
        """
        Detect human pose and create skeleton.
        """
        from controlnet_aux import OpenposeDetector
        
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        pose = openpose(image)
        
        return pose
    
    def scribble(
        self,
        image: Image.Image,
        detect: bool = True
    ) -> Image.Image:
        """
        Create scribble-style edges.
        """
        from controlnet_aux import HEDdetector
        
        if detect:
            # Use HED boundary detection
            hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")
            scribble = hed(image, scribble=True)
        else:
            # If already a scribble, just return
            scribble = image
        
        return scribble
    
    def mlsd_lines(
        self,
        image: Image.Image
    ) -> Image.Image:
        """
        Detect straight lines (M-LSD).
        """
        from controlnet_aux import MLSDdetector
        
        mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
        lines = mlsd(image)
        
        return lines

# Usage
preprocessor = ControlImagePreprocessor()

input_photo = Image.open("person.jpg")

# Canny edges
canny = preprocessor.canny_edges(input_photo)
canny.save("control_canny.png")

# Depth map
depth = preprocessor.depth_map(input_photo)
depth.save("control_depth.png")

# OpenPose skeleton
pose = preprocessor.openpose_skeleton(input_photo)
pose.save("control_pose.png")
\`\`\`

## Practical Applications

### 1. Pose Control for Characters

\`\`\`python
class PoseControlledGeneration:
    """
    Generate characters in specific poses.
    """
    
    def __init__(self):
        self.pose_cn = ControlNetGenerator(controlnet_type="openpose")
        self.preprocessor = ControlImagePreprocessor()
    
    def generate_from_pose_reference(
        self,
        reference_image: Image.Image,
        character_description: str,
        **kwargs
    ) -> Image.Image:
        """
        Generate new character matching pose from reference.
        """
        # Extract pose from reference
        pose_skeleton = self.preprocessor.openpose_skeleton(reference_image)
        
        # Generate with pose control
        result = self.pose_cn.generate(
            prompt=character_description,
            control_image=pose_skeleton,
            **kwargs
        )[0]
        
        return result
    
    def generate_multiple_characters_same_pose(
        self,
        pose_image: Image.Image,
        character_descriptions: list[str]
    ) -> list[Image.Image]:
        """
        Generate multiple characters in same pose.
        """
        # Extract pose once
        pose_skeleton = self.preprocessor.openpose_skeleton(pose_image)
        
        results = []
        for desc in character_descriptions:
            result = self.pose_cn.generate(
                prompt=desc,
                control_image=pose_skeleton,
                steps=20,
                seed=None  # Different seed for variety
            )[0]
            results.append(result)
        
        return results

# Usage
pose_gen = PoseControlledGeneration()

# Use reference pose
reference = Image.open("person_standing.jpg")

# Generate different characters in same pose
characters = pose_gen.generate_multiple_characters_same_pose(
    pose_image=reference,
    character_descriptions=[
        "a knight in armor, medieval fantasy",
        "a superhero in costume, comic book style",
        "a business person in suit, professional photograph",
        "an astronaut in spacesuit, sci-fi"
    ]
)

for i, char in enumerate(characters):
    char.save(f"character_{i}.png")
\`\`\`

### 2. Architectural Design with Line Control

\`\`\`python
class ArchitecturalGenerator:
    """
    Generate buildings with precise line control.
    """
    
    def __init__(self):
        self.mlsd_cn = ControlNetGenerator(controlnet_type="mlsd")
        self.preprocessor = ControlImagePreprocessor()
    
    def redesign_building(
        self,
        building_photo: Image.Image,
        new_style: str
    ) -> Image.Image:
        """
        Keep structure, change style.
        """
        # Extract straight lines
        lines = self.preprocessor.mlsd_lines(building_photo)
        
        # Generate with new style
        result = self.mlsd_cn.generate(
            prompt=f"architecture photograph, {new_style} style, professional",
            control_image=lines,
            steps=25,
            controlnet_conditioning_scale=1.2  # Strong adherence to lines
        )[0]
        
        return result
    
    def sketch_to_building(
        self,
        sketch: Image.Image,
        style: str = "modern architecture"
    ) -> Image.Image:
        """
        Convert architectural sketch to realistic building.
        """
        result = self.mlsd_cn.generate(
            prompt=f"{style}, professional architecture photography, detailed",
            control_image=sketch,  # Sketch should have straight lines
            negative_prompt="blurry, distorted, low quality",
            steps=30
        )[0]
        
        return result

# Usage
arch_gen = ArchitecturalGenerator()

# Transform existing building
modern_building = Image.open("modern_building.jpg")
victorian = arch_gen.redesign_building(
    modern_building,
    "Victorian Gothic"
)

# From sketch
sketch = Image.open("building_sketch.png")
realistic = arch_gen.sketch_to_building(sketch, "modern minimalist")
\`\`\`

### 3. Depth-Controlled Scene Generation

\`\`\`python
class DepthControlledGeneration:
    """
    Generate scenes with consistent 3D structure.
    """
    
    def __init__(self):
        self.depth_cn = ControlNetGenerator(controlnet_type="depth")
        self.preprocessor = ControlImagePreprocessor()
    
    def transform_scene(
        self,
        source_image: Image.Image,
        new_description: str
    ) -> Image.Image:
        """
        Transform scene while preserving spatial structure.
        """
        # Extract depth
        depth_map = self.preprocessor.depth_map(source_image)
        
        # Generate with same depth structure
        result = self.depth_cn.generate(
            prompt=new_description,
            control_image=depth_map,
            steps=25,
            controlnet_conditioning_scale=1.0
        )[0]
        
        return result
    
    def consistent_scene_variations(
        self,
        base_image: Image.Image,
        time_of_day_prompts: list[str]
    ) -> list[Image.Image]:
        """
        Generate same scene at different times/conditions.
        Keeps spatial structure consistent.
        """
        # Extract depth once
        depth_map = self.preprocessor.depth_map(base_image)
        
        results = []
        for prompt in time_of_day_prompts:
            result = self.depth_cn.generate(
                prompt=prompt,
                control_image=depth_map,
                steps=25
            )[0]
            results.append(result)
        
        return results

# Usage
depth_gen = DepthControlledGeneration()

# Transform room
room_photo = Image.open("modern_room.jpg")
transformed = depth_gen.transform_scene(
    room_photo,
    "cozy rustic cabin interior, wooden furniture, warm lighting"
)

# Same street at different times
street = Image.open("street.jpg")
variations = depth_gen.consistent_scene_variations(
    base_image=street,
    time_of_day_prompts=[
        "same street at sunrise, golden hour, warm lighting",
        "same street at noon, bright daylight, clear sky",
        "same street at sunset, orange sky, dramatic lighting",
        "same street at night, street lights, dark sky"
    ]
)
\`\`\`

## Advanced: Multi-ControlNet

### Combining Multiple Controls

\`\`\`python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

class MultiControlNetGenerator:
    """
    Use multiple ControlNets simultaneously.
    """
    
    def __init__(
        self,
        controlnet_types: list[str],
        device: str = "cuda"
    ):
        self.device = device
        
        # Load multiple ControlNets
        controlnets = [
            ControlNetModel.from_pretrained(
                f"lllyasviel/sd-controlnet-{cn_type}",
                torch_dtype=torch.float16
            )
            for cn_type in controlnet_types
        ]
        
        # Load pipeline with multiple ControlNets
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnets,
            torch_dtype=torch.float16
        )
        
        self.pipe = self.pipe.to(device)
    
    def generate(
        self,
        prompt: str,
        control_images: list[Image.Image],
        conditioning_scales: list[float],
        **kwargs
    ) -> list[Image.Image]:
        """
        Generate with multiple control conditions.
        
        Args:
            prompt: Text description
            control_images: List of control images (one per ControlNet)
            conditioning_scales: Strength for each control (0.0-2.0)
        """
        output = self.pipe(
            prompt=prompt,
            image=control_images,
            controlnet_conditioning_scale=conditioning_scales,
            **kwargs
        )
        
        return output.images

# Usage: Canny edges + Depth
multi_cn = MultiControlNetGenerator(
    controlnet_types=["canny", "depth"]
)

preprocessor = ControlImagePreprocessor()
source = Image.open("photo.jpg")

# Prepare both controls
canny = preprocessor.canny_edges(source)
depth = preprocessor.depth_map(source)

# Generate with both
result = multi_cn.generate(
    prompt="oil painting, artistic, detailed",
    control_images=[canny, depth],
    conditioning_scales=[0.8, 0.5],  # Canny stronger than depth
    num_inference_steps=25
)[0]
\`\`\`

## Conditioning Scale Effects

\`\`\`python
conditioning_scale_guide = {
    "0.0_to_0.5": {
        "description": "Weak guidance",
        "effect": "Loose adherence to control, more creative",
        "use": "Subtle hints, inspiration only"
    },
    
    "0.5_to_0.8": {
        "description": "Moderate guidance",
        "effect": "Balanced between control and creativity",
        "use": "General structure guidance"
    },
    
    "0.8_to_1.2": {
        "description": "Strong guidance (recommended)",
        "effect": "Close adherence to control",
        "use": "Most use cases, reliable control"
    },
    
    "1.2_to_1.5": {
        "description": "Very strong guidance",
        "effect": "Strict adherence, less creative freedom",
        "use": "When precision is critical"
    },
    
    "1.5_plus": {
        "description": "Maximum guidance",
        "effect": "May create artifacts, too rigid",
        "use": "Rarely beneficial"
    }
}

def test_conditioning_scales(
    generator: ControlNetGenerator,
    prompt: str,
    control_image: Image.Image,
    scales: list[float] = [0.5, 0.8, 1.0, 1.2, 1.5]
) -> dict:
    """
    Test different conditioning scales.
    """
    results = {}
    
    for scale in scales:
        output = generator.generate(
            prompt=prompt,
            control_image=control_image,
            controlnet_conditioning_scale=scale,
            steps=20,
            seed=42  # Same seed for comparison
        )[0]
        
        results[scale] = output
    
    return results
\`\`\`

## Production Tips

\`\`\`python
production_best_practices = {
    "preprocessing": [
        "Preprocess control images offline when possible",
        "Cache preprocessed images for reuse",
        "Validate control image dimensions match output",
        "Use appropriate thresholds for edge detection"
    ],
    
    "quality": [
        "Start with conditioning_scale=1.0",
        "Use 20-30 steps for good balance",
        "Combine with negative prompts",
        "Test scale variations for optimal results"
    ],
    
    "performance": [
        "Enable xformers for speed",
        "Use CPU offload for limited VRAM",
        "Batch process when possible",
        "Consider quantization for production"
    ],
    
    "control_selection": [
        "Canny: Best for strong edges and outlines",
        "Depth: Best for 3D structure and scenes",
        "OpenPose: Best for human poses",
        "MLSD: Best for architecture and straight lines",
        "Scribble: Best for rough sketches"
    ]
}
\`\`\`

## Key Takeaways

- **ControlNet** provides precise structural control over generation
- **Different types** for different controls: canny, depth, pose, lines, etc.
- **Preprocessing required**: Convert input images to control format
- **Conditioning scale** controls adherence strength (0.8-1.2 typical)
- **OpenPose** perfect for character poses
- **Depth** great for consistent 3D structure
- **Canny edges** excellent for line art to image
- **Multi-ControlNet** allows combining multiple controls
- **Better than img2img** for structural control
- **Production use**: Style transfer, pose control, architectural design
`,
};
