/**
 * ComfyUI & Workflows Section
 * Module 8: Image Generation & Computer Vision
 */

export const comfyuiworkflowsSection = {
  id: 'comfyui-workflows',
  title: 'ComfyUI & Workflows',
  content: `# ComfyUI & Workflows

Master ComfyUI for advanced node-based image generation workflows and automation.

## Overview: Visual Programming for Image Generation

ComfyUI is a powerful node-based interface for Stable Diffusion that allows you to:
- **Build complex workflows**: Chain multiple operations visually
- **Reuse workflows**: Save and share complete pipelines
- **Precise control**: Fine-tune every parameter
- **Automation**: Generate variations programmatically
- **Custom nodes**: Extend functionality

### Why ComfyUI?

\`\`\`python
traditional_vs_comfyui = {
    "traditional_sd": {
        "interface": "Simple UI with basic options",
        "workflow": "Linear: prompt → generate → done",
        "control": "Limited to basic parameters",
        "reusability": "Start from scratch each time"
    },
    
    "comfyui": {
        "interface": "Node-based visual programming",
        "workflow": "Complex graphs with branching",
        "control": "Full control over every step",
        "reusability": "Save workflows, share with community",
        "advantages": [
            "Combine img2img, ControlNet, inpainting in one workflow",
            "Generate variations automatically",
            "A/B test parameters easily",
            "Production-ready pipelines"
        ]
    }
}
\`\`\`

## Getting Started with ComfyUI

### Installation

\`\`\`python
"""
# Clone repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install dependencies
pip install -r requirements.txt

# Download models (place in models/checkpoints/)
# SD 2.1: stabilityai/stable-diffusion-2-1
# SDXL: stabilityai/stable-diffusion-xl-base-1.0

# Run
python main.py

# Open browser to http://127.0.0.1:8188
"""

# Basic workflow structure
workflow_structure = {
    "nodes": [
        {
            "type": "CheckpointLoaderSimple",
            "outputs": ["MODEL", "CLIP", "VAE"]
        },
        {
            "type": "CLIPTextEncode",
            "inputs": ["CLIP", "text"],
            "outputs": ["CONDITIONING"]
        },
        {
            "type": "KSampler",
            "inputs": ["model", "positive", "negative", "latent"],
            "outputs": ["LATENT"]
        },
        {
            "type": "VAEDecode",
            "inputs": ["samples", "vae"],
            "outputs": ["IMAGE"]
        },
        {
            "type": "SaveImage",
            "inputs": ["images"]
        }
    ]
}
\`\`\`

## Core Workflow Patterns

### Basic Text-to-Image Workflow

\`\`\`python
basic_workflow = {
    "description": "Standard txt2img generation",
    "nodes": [
        # 1. Load checkpoint
        {
            "id": 1,
            "type": "CheckpointLoaderSimple",
            "params": {
                "ckpt_name": "sd_v2-1.safetensors"
            }
        },
        
        # 2. Encode prompt
        {
            "id": 2,
            "type": "CLIPTextEncode",
            "params": {
                "text": "a beautiful landscape, oil painting"
            },
            "inputs": {
                "clip": [1, 1]  # From node 1, output 1 (CLIP)
            }
        },
        
        # 3. Encode negative prompt
        {
            "id": 3,
            "type": "CLIPTextEncode",
            "params": {
                "text": "blurry, low quality"
            },
            "inputs": {
                "clip": [1, 1]
            }
        },
        
        # 4. Empty latent (noise)
        {
            "id": 4,
            "type": "EmptyLatentImage",
            "params": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            }
        },
        
        # 5. Sample (generate)
        {
            "id": 5,
            "type": "KSampler",
            "params": {
                "seed": 42,
                "steps": 30,
                "cfg": 7.5,
                "sampler_name": "euler_a",
                "scheduler": "normal"
            },
            "inputs": {
                "model": [1, 0],
                "positive": [2, 0],
                "negative": [3, 0],
                "latent_image": [4, 0]
            }
        },
        
        # 6. Decode
        {
            "id": 6,
            "type": "VAEDecode",
            "inputs": {
                "samples": [5, 0],
                "vae": [1, 2]
            }
        },
        
        # 7. Save
        {
            "id": 7,
            "type": "SaveImage",
            "params": {
                "filename_prefix": "ComfyUI"
            },
            "inputs": {
                "images": [6, 0]
            }
        }
    ]
}
\`\`\`

### Advanced img2img + ControlNet Workflow

\`\`\`python
advanced_workflow = {
    "description": "Combine img2img with ControlNet for precise control",
    "nodes": [
        # Load models
        {"type": "CheckpointLoaderSimple"},
        {"type": "ControlNetLoader", "params": {"control_net_name": "canny"}},
        
        # Load input image
        {"type": "LoadImage", "params": {"image": "input.png"}},
        
        # Preprocess for ControlNet
        {"type": "CannyEdgePreprocessor"},
        
        # Apply ControlNet
        {"type": "ControlNetApply", "params": {"strength": 1.0}},
        
        # Encode to latent
        {"type": "VAEEncode"},
        
        # Add noise (img2img strength)
        {"type": "LatentNoise", "params": {"strength": 0.5}},
        
        # Generate
        {"type": "KSampler"},
        
        # Decode and save
        {"type": "VAEDecode"},
        {"type": "SaveImage"}
    ]
}
\`\`\`

## Using ComfyUI API

### Python API Client

\`\`\`python
import json
import requests
import websocket
from PIL import Image
from io import BytesIO
import uuid

class ComfyUIClient:
    """
    Programmatic interface to ComfyUI.
    """
    
    def __init__(
        self,
        server_address: str = "127.0.0.1:8188"
    ):
        self.server_address = server_address
        self.client_id = str (uuid.uuid4())
    
    def queue_prompt (self, prompt: dict) -> str:
        """
        Queue a workflow for execution.
        
        Args:
            prompt: Workflow definition
        
        Returns:
            Prompt ID
        """
        p = {"prompt": prompt, "client_id": self.client_id}
        
        response = requests.post(
            f"http://{self.server_address}/prompt",
            json=p
        )
        
        return response.json()["prompt_id"]
    
    def get_image (self, filename: str, subfolder: str = "", folder_type: str = "output") -> Image.Image:
        """Download generated image."""
        response = requests.get(
            f"http://{self.server_address}/view",
            params={
                "filename": filename,
                "subfolder": subfolder,
                "type": folder_type
            }
        )
        
        return Image.open(BytesIO(response.content))
    
    def get_history (self, prompt_id: str) -> dict:
        """Get generation history."""
        response = requests.get(
            f"http://{self.server_address}/history/{prompt_id}"
        )
        return response.json()
    
    def wait_for_completion (self, prompt_id: str) -> list[Image.Image]:
        """
        Wait for workflow to complete and return images.
        """
        import time
        
        while True:
            history = self.get_history (prompt_id)
            
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                
                # Find images
                images = []
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        for img_info in node_output["images"]:
                            img = self.get_image(
                                img_info["filename"],
                                img_info.get("subfolder", ""),
                                img_info.get("type", "output")
                            )
                            images.append (img)
                
                return images
            
            time.sleep(1)
    
    def generate (self, workflow: dict) -> list[Image.Image]:
        """
        Generate images from workflow.
        
        Args:
            workflow: ComfyUI workflow JSON
        
        Returns:
            List of generated images
        """
        prompt_id = self.queue_prompt (workflow)
        return self.wait_for_completion (prompt_id)

# Usage
client = ComfyUIClient()

# Load workflow from file
with open("workflow.json", "r") as f:
    workflow = json.load (f)

# Generate
images = client.generate (workflow)

for i, img in enumerate (images):
    img.save (f"output_{i}.png")
\`\`\`

### Workflow Templates

\`\`\`python
class WorkflowBuilder:
    """
    Build ComfyUI workflows programmatically.
    """
    
    def __init__(self):
        self.nodes = {}
        self.node_id = 1
    
    def add_node(
        self,
        node_type: str,
        params: dict = {},
        inputs: dict = {}
    ) -> int:
        """Add node to workflow."""
        node_id = self.node_id
        self.node_id += 1
        
        self.nodes[str (node_id)] = {
            "class_type": node_type,
            "inputs": {**params, **inputs}
        }
        
        return node_id
    
    def build_txt2img(
        self,
        checkpoint: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        cfg: float = 7.5,
        seed: int = -1
    ) -> dict:
        """Build text-to-image workflow."""
        # Load checkpoint
        ckpt = self.add_node("CheckpointLoaderSimple", {
            "ckpt_name": checkpoint
        })
        
        # Encode prompts
        pos = self.add_node("CLIPTextEncode", {
            "text": prompt
        }, {"clip": [ckpt, 1]})
        
        neg = self.add_node("CLIPTextEncode", {
            "text": negative_prompt
        }, {"clip": [ckpt, 1]})
        
        # Empty latent
        latent = self.add_node("EmptyLatentImage", {
            "width": width,
            "height": height,
            "batch_size": 1
        })
        
        # Sample
        sample = self.add_node("KSampler", {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler_a",
            "scheduler": "normal",
            "denoise": 1.0
        }, {
            "model": [ckpt, 0],
            "positive": [pos, 0],
            "negative": [neg, 0],
            "latent_image": [latent, 0]
        })
        
        # Decode
        decode = self.add_node("VAEDecode", inputs={
            "samples": [sample, 0],
            "vae": [ckpt, 2]
        })
        
        # Save
        self.add_node("SaveImage", {
            "filename_prefix": "ComfyUI"
        }, {"images": [decode, 0]})
        
        return self.nodes
    
    def build_img2img(
        self,
        checkpoint: str,
        input_image_path: str,
        prompt: str,
        strength: float = 0.5,
        steps: int = 30,
        **kwargs
    ) -> dict:
        """Build image-to-image workflow."""
        # Load checkpoint
        ckpt = self.add_node("CheckpointLoaderSimple", {
            "ckpt_name": checkpoint
        })
        
        # Load image
        load_img = self.add_node("LoadImage", {
            "image": input_image_path
        })
        
        # Encode to latent
        vae_encode = self.add_node("VAEEncode", inputs={
            "pixels": [load_img, 0],
            "vae": [ckpt, 2]
        })
        
        # Encode prompts
        pos = self.add_node("CLIPTextEncode", {
            "text": prompt
        }, {"clip": [ckpt, 1]})
        
        neg = self.add_node("CLIPTextEncode", {
            "text": kwargs.get("negative_prompt", "")
        }, {"clip": [ckpt, 1]})
        
        # Sample with denoise (strength)
        sample = self.add_node("KSampler", {
            "seed": kwargs.get("seed", -1),
            "steps": steps,
            "cfg": kwargs.get("cfg", 7.5),
            "sampler_name": "euler_a",
            "scheduler": "normal",
            "denoise": strength  # img2img strength
        }, {
            "model": [ckpt, 0],
            "positive": [pos, 0],
            "negative": [neg, 0],
            "latent_image": [vae_encode, 0]
        })
        
        # Decode
        decode = self.add_node("VAEDecode", inputs={
            "samples": [sample, 0],
            "vae": [ckpt, 2]
        })
        
        # Save
        self.add_node("SaveImage", {
            "filename_prefix": "img2img"
        }, {"images": [decode, 0]})
        
        return self.nodes

# Usage
builder = WorkflowBuilder()

# Build txt2img workflow
workflow = builder.build_txt2img(
    checkpoint="sd_v2-1.safetensors",
    prompt="a serene mountain landscape",
    negative_prompt="blurry, low quality",
    width=768,
    height=512,
    steps=30,
    seed=42
)

# Execute
client = ComfyUIClient()
images = client.generate (workflow)
\`\`\`

## Advanced Techniques

### Batch Generation with Variations

\`\`\`python
class BatchGenerator:
    """
    Generate batches with parameter variations.
    """
    
    def __init__(self, client: ComfyUIClient):
        self.client = client
    
    def generate_grid(
        self,
        base_workflow: dict,
        variations: list[dict]
    ) -> list[Image.Image]:
        """
        Generate grid of variations.
        
        Args:
            base_workflow: Base workflow
            variations: List of parameter changes
        """
        results = []
        
        for i, var in enumerate (variations):
            print(f"Generating variation {i+1}/{len (variations)}")
            
            # Apply variation
            workflow = self._apply_variation (base_workflow, var)
            
            # Generate
            images = self.client.generate (workflow)
            results.extend (images)
        
        return results
    
    def _apply_variation (self, workflow: dict, changes: dict) -> dict:
        """Apply parameter changes to workflow."""
        import copy
        modified = copy.deepcopy (workflow)
        
        for node_id, params in changes.items():
            if node_id in modified:
                modified[node_id]["inputs"].update (params)
        
        return modified
    
    def test_parameters(
        self,
        workflow: dict,
        parameter_ranges: dict
    ) -> dict:
        """
        Test different parameter values.
        
        Example:
        parameter_ranges = {
            "5": {"cfg": [5, 7, 9, 11]},  # Node 5, CFG values
            "5": {"steps": [20, 30, 40, 50]}  # Node 5, step counts
        }
        """
        import itertools
        
        # Generate combinations
        node_ids = list (parameter_ranges.keys())
        param_names = list (parameter_ranges[node_ids[0]].keys())
        
        results = {}
        for values in itertools.product(*[parameter_ranges[nid][p] for nid in node_ids for p in param_names]):
            # Build variation
            variation = {}
            for i, nid in enumerate (node_ids):
                variation[nid] = {param_names[i]: values[i]}
            
            # Generate
            images = self.generate_grid (workflow, [variation])
            results[str (values)] = images[0]
        
        return results

# Usage
batch_gen = BatchGenerator (client)

# Test different CFG values
cfg_variations = [
    {"5": {"cfg": 5}},
    {"5": {"cfg": 7}},
    {"5": {"cfg": 9}},
    {"5": {"cfg": 11}},
]

cfg_results = batch_gen.generate_grid (base_workflow, cfg_variations)
\`\`\`

## Custom Nodes

\`\`\`python
"""
Custom nodes extend ComfyUI functionality.

Example structure:
custom_nodes/
  my_nodes/
    __init__.py
    nodes.py

# nodes.py
class MyCustomNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/processing"
    
    def process (self, image, strength):
        # Your custom processing
        processed = your_function (image, strength)
        return (processed,)

NODE_CLASS_MAPPINGS = {
    "MyCustomNode": MyCustomNode
}
"""

custom_node_template = ''
class CustomEnhancer:
    """Custom enhancement node."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enhancement_type": (["sharpen", "denoise", "color_correct"],)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance"
    CATEGORY = "image/enhancement"
    
    def enhance (self, image, enhancement_type):
        if enhancement_type == "sharpen":
            return (self.sharpen (image),)
        elif enhancement_type == "denoise":
            return (self.denoise (image),)
        elif enhancement_type == "color_correct":
            return (self.color_correct (image),)
    
    def sharpen (self, image):
        # Sharpening logic
        return image
    
    def denoise (self, image):
        # Denoising logic
        return image
    
    def color_correct (self, image):
        # Color correction logic
        return image
''
\`\`\`

## Production Workflows

\`\`\`python
production_patterns = {
    "workflow_versioning": {
        "description": "Version control for workflows",
        "practice": [
            "Save workflows with descriptive names",
            "Include version in filename: workflow_v1.2.json",
            "Document parameter changes",
            "Keep working backups"
        ]
    },
    
    "error_handling": {
        "strategies": [
            "Validate workflow before queuing",
            "Check model files exist",
            "Handle queue failures",
            "Implement retry logic",
            "Log all operations"
        ]
    },
    
    "optimization": {
        "tips": [
            "Use efficient samplers (DPM++, Euler a)",
            "Reduce steps for iteration (20-25)",
            "Batch process when possible",
            "Cache commonly used models",
            "Use VAE tiling for large images"
        ]
    }
}
\`\`\`

## Key Takeaways

- **ComfyUI**: Node-based visual workflow builder for SD
- **Flexibility**: Build complex multi-step pipelines
- **API access**: Programmatic generation and automation
- **Reusability**: Save and share workflows
- **Custom nodes**: Extend with your own functionality
- **Batch processing**: Generate variations systematically
- **Production-ready**: Suitable for automated workflows
- **Community**: Large library of shared workflows
- **Control**: Fine-tune every aspect of generation
- **Learning curve**: More complex than simple UIs, but powerful
`,
};
