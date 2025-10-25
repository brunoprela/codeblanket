/**
 * DALL-E 3 API Section
 * Module 8: Image Generation & Computer Vision
 */

export const dalle3apiSection = {
  id: 'dalle-3-api',
  title: 'DALL-E 3 API',
  content: `# DALL-E 3 API

Master OpenAI's DALL-E 3 API for production image generation with the best prompt-following capabilities available.

## Overview: Why DALL-E 3

DALL-E 3 represents the current state-of-the-art in prompt following and image quality. Unlike Stable Diffusion, it's API-only but offers unmatched capabilities.

### Key Advantages

- **Best Prompt Following**: Understands complex, detailed prompts
- **Text Rendering**: Can write legible text in images (mostly)
- **Composition**: Accurate spatial relationships
- **Consistency**: Reliable, predictable results
- **No Setup**: API call, no GPU needed
- **Commercial Use**: Clear licensing

### When to Use DALL-E 3

\`\`\`python
use_cases_comparison = {
    "use_dalle3_when": [
        "Prompt following is critical",
        "Need text in images",
        "Complex scene composition",
        "Production quality matters",
        "Want consistent results",
        "Commercial/client work",
        "Don't want to manage infrastructure"
    ],
    
    "use_stable_diffusion_when": [
        "Need fine-tuning/customization",
        "High volume (cost sensitive)",
        "Want full control",
        "Specific artistic style",
        "Offline generation needed",
        "Rapid iteration/prototyping"
    ]
}
\`\`\`

## Getting Started

### Setup

\`\`\`python
# Installation
"""
pip install openai pillow requests
"""

from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
from typing import Optional, List, Literal
import os

class DALLEGenerator:
    """
    Production-ready DALL-E 3 client.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize DALL-E client.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.client = OpenAI(api_key=api_key)
    
    def generate(
        self,
        prompt: str,
        size: Literal["1024x1024", "1024x1792", "1792x1024"] = "1024x1024",
        quality: Literal["standard", "hd"] = "standard",
        style: Literal["vivid", "natural"] = "vivid",
        n: int = 1
    ) -> dict:
        """
        Generate images with DALL-E 3.
        
        Args:
            prompt: Text description of desired image
            size: Image dimensions
            quality: 'standard' or 'hd' (costs more)
            style: 'vivid' (hyper-real) or 'natural' (realistic)
            n: Number of images (always 1 for DALL-E 3)
        
        Returns:
            Response dict with URLs and metadata
        """
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=n  # DALL-E 3 only supports n=1
        )
        
        return {
            "url": response.data[0].url,
            "revised_prompt": response.data[0].revised_prompt,
            "size": size,
            "quality": quality,
            "style": style
        }
    
    def download_image (self, url: str) -> Image.Image:
        """Download image from URL."""
        response = requests.get (url)
        return Image.open(BytesIO(response.content))
    
    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        **kwargs
    ) -> dict:
        """
        Generate and save image in one call.
        """
        result = self.generate (prompt, **kwargs)
        
        # Download image
        image = self.download_image (result["url"])
        
        # Save
        image.save (output_path)
        
        print(f"Saved to: {output_path}")
        print(f"Revised prompt: {result['revised_prompt']}")
        
        return result

# Basic usage
generator = DALLEGenerator()

result = generator.generate_and_save(
    prompt="a serene Japanese garden with a red bridge over a koi pond",
    output_path="garden.png",
    quality="hd",
    style="natural"
)
\`\`\`

## Understanding Parameters

### Size Options

\`\`\`python
size_guide = {
    "1024x1024": {
        "aspect_ratio": "1:1 (Square)",
        "best_for": [
            "Social media posts",
            "Profile images",
            "Product photos",
            "General purpose"
        ],
        "cost_standard": "$0.040",
        "cost_hd": "$0.080"
    },
    
    "1024x1792": {
        "aspect_ratio": "9:16 (Portrait)",
        "best_for": [
            "Mobile wallpapers",
            "Instagram stories",
            "Vertical posters",
            "Full-body portraits"
        ],
        "cost_standard": "$0.080",
        "cost_hd": "$0.120"
    },
    
    "1792x1024": {
        "aspect_ratio": "16:9 (Landscape)",
        "best_for": [
            "Desktop wallpapers",
            "YouTube thumbnails",
            "Presentation slides",
            "Website headers"
        ],
        "cost_standard": "$0.080",
        "cost_hd": "$0.120"
    }
}

def choose_size (use_case: str) -> str:
    """Choose appropriate size for use case."""
    mapping = {
        "instagram_post": "1024x1024",
        "instagram_story": "1024x1792",
        "youtube_thumbnail": "1792x1024",
        "website_header": "1792x1024",
        "profile_picture": "1024x1024",
        "poster": "1024x1792",
        "presentation": "1792x1024"
    }
    return mapping.get (use_case, "1024x1024")
\`\`\`

### Quality: Standard vs HD

\`\`\`python
quality_comparison = {
    "standard": {
        "resolution": "1024×1024 effective",
        "detail_level": "Good",
        "cost": "1x",
        "generation_time": "~10 seconds",
        "best_for": [
            "Prototyping",
            "High-volume generation",
            "Social media",
            "When cost matters"
        ],
        "example_cost": "$40 for 1000 square images"
    },
    
    "hd": {
        "resolution": "Better detail and fidelity",
        "detail_level": "Excellent",
        "cost": "2x",
        "generation_time": "~15 seconds",
        "best_for": [
            "Professional use",
            "Print materials",
            "Marketing campaigns",
            "Client deliverables",
            "When quality matters most"
        ],
        "example_cost": "$80 for 1000 square images"
    }
}

def should_use_hd (context: dict) -> bool:
    """
    Decide whether to use HD quality.
    """
    if context.get("use_case") in ["print", "professional", "client_facing"]:
        return True
    
    if context.get("budget") == "unlimited":
        return True
    
    if context.get("detail_critical"):
        return True
    
    return False
\`\`\`

### Style: Vivid vs Natural

\`\`\`python
style_guide = {
    "vivid": {
        "description": "Hyper-real, dramatic, cinematic",
        "characteristics": [
            "Enhanced colors and contrast",
            "Dramatic lighting",
            "Cinematic composition",
            "More 'wow' factor",
            "Slightly less realistic"
        ],
        "best_for": [
            "Marketing materials",
            "Eye-catching social media",
            "Concept art",
            "Fantasy/sci-fi",
            "When visual impact matters"
        ],
        "example": "A vivid sunset will have intense oranges and purples"
    },
    
    "natural": {
        "description": "Realistic, subdued, authentic",
        "characteristics": [
            "Natural colors",
            "Realistic lighting",
            "Believable scenes",
            "More photographic",
            "Subtle composition"
        ],
        "best_for": [
            "Photorealistic images",
            "Product photography",
            "Documentary style",
            "Professional headshots",
            "When authenticity matters"
        ],
        "example": "A natural sunset will look like actual photograph"
    }
}

# Practical examples
examples = {
    "product_photo": {
        "prompt": "professional photo of a coffee mug on a wooden table",
        "recommended_style": "natural",
        "reason": "Want realistic, believable product image"
    },
    
    "marketing_banner": {
        "prompt": "a modern tech office with people collaborating",
        "recommended_style": "vivid",
        "reason": "Want eye-catching, engaging image"
    },
    
    "concept_art": {
        "prompt": "futuristic city with flying cars",
        "recommended_style": "vivid",
        "reason": "Fantasy/sci-fi benefits from dramatic style"
    },
    
    "headshot": {
        "prompt": "professional business headshot",
        "recommended_style": "natural",
        "reason": "Need realistic, authentic appearance"
    }
}
\`\`\`

## Prompt Engineering for DALL-E 3

### The Revised Prompt System

DALL-E 3 automatically enhances your prompts:

\`\`\`python
def understand_revised_prompts():
    """
    DALL-E 3 rewrites prompts for better results.
    """
    your_prompt = "a cat"
    
    revised_prompt = """
    A close-up photograph of a domestic short-haired tabby cat
    with green eyes, sitting on a wooden floor in natural daylight,
    professional photography, sharp focus, shallow depth of field
    """
    
    # DALL-E adds:
    # - Specific details (tabby, short-haired)
    # - Context (wooden floor, daylight)
    # - Style/quality markers (professional, sharp focus)
    # - Composition guidance (close-up, shallow depth of field)
    
    return {
        "benefit": "Better, more detailed images",
        "consideration": "May add details you didn't want",
        "solution": "Be specific in your original prompt"
    }

# Example: Control the revision
generator = DALLEGenerator()

result = generator.generate(
    # Vague prompt - DALL-E will add a lot
    prompt="a dog",
    style="natural"
)
print(f"Revised: {result['revised_prompt']}")
# Might become: "A golden retriever sitting in grass..."

result = generator.generate(
    # Specific prompt - DALL-E adds less
    prompt="a small black poodle sitting on a red cushion, indoor studio lighting",
    style="natural"
)
print(f"Revised: {result['revised_prompt']}")
# Minimal changes since you were specific
\`\`\`

### Effective Prompt Patterns

\`\`\`python
class DALLEPromptBuilder:
    """
    Build effective DALL-E 3 prompts.
    """
    
    @staticmethod
    def basic_structure(
        subject: str,
        action: Optional[str] = None,
        context: Optional[str] = None,
        style: Optional[str] = None,
        details: Optional[List[str]] = None
    ) -> str:
        """
        Structure: [Subject] [Action] [Context] [Style] [Details]
        """
        parts = [subject]
        
        if action:
            parts.append (action)
        
        if context:
            parts.append (context)
        
        if style:
            parts.append (f"in {style} style")
        
        if details:
            parts.extend (details)
        
        return ", ".join (parts)
    
    @staticmethod
    def photography_style(
        subject: str,
        shot_type: str = "medium shot",
        lighting: str = "natural lighting",
        camera: Optional[str] = None,
        additional: Optional[str] = None
    ) -> str:
        """
        Photography-style prompt.
        """
        parts = [
            f"{shot_type} photograph of {subject}",
            lighting
        ]
        
        if camera:
            parts.append (f"shot on {camera}")
        
        if additional:
            parts.append (additional)
        
        return ", ".join (parts)
    
    @staticmethod
    def illustration_style(
        subject: str,
        art_style: str,
        color_palette: Optional[str] = None,
        mood: Optional[str] = None
    ) -> str:
        """
        Illustration-style prompt.
        """
        parts = [
            f"{art_style} illustration of {subject}"
        ]
        
        if color_palette:
            parts.append (f"{color_palette} color palette")
        
        if mood:
            parts.append (f"{mood} mood")
        
        return ", ".join (parts)

# Usage examples
builder = DALLEPromptBuilder()

# Photography
photo_prompt = builder.photography_style(
    subject="a coffee cup on a desk with laptop",
    shot_type="overhead shot",
    lighting="soft morning light",
    camera="Canon EOS R5",
    additional="shallow depth of field, warm tones"
)
# Result: "overhead shot photograph of a coffee cup on a desk with laptop, 
#          soft morning light, shot on Canon EOS R5, shallow depth of field, warm tones"

# Illustration
illustration_prompt = builder.illustration_style(
    subject="a friendly robot helper",
    art_style="modern flat design",
    color_palette="vibrant blue and orange",
    mood="cheerful and welcoming"
)
# Result: "modern flat design illustration of a friendly robot helper,
#          vibrant blue and orange color palette, cheerful and welcoming mood"
\`\`\`

### Advanced Prompting Techniques

\`\`\`python
advanced_techniques = {
    "text_in_images": {
        "description": "DALL-E 3 can render text (mostly)",
        "example": ''
        A vintage poster with the text "COFFEE SHOP" in bold serif font,
        below it "EST. 1995" in smaller text, coffee bean illustrations around the text,
        cream background, retro aesthetic
        '',
        "tips": [
            "Put text in quotes",
            "Specify font style (serif, sans-serif, handwritten)",
            "Keep text short (1-5 words works best)",
            "Specify size relationship (large, small)",
            "Describe text placement"
        ]
    },
    
    "precise_composition": {
        "description": "Control exact layout",
        "example": ''
        A split composition image: left half shows a busy city street in daylight,
        right half shows the same street at night with neon signs,
        sharp vertical division in the middle
        '',
        "tips": [
            "Use spatial terms (left/right, top/bottom, center)",
            "Describe relationships (in front of, behind, next to)",
            "Specify splits and divisions",
            "Mention symmetry or balance"
        ]
    },
    
    "multiple_objects": {
        "description": "Include several specific items",
        "example": ''
        A flat lay photo on white marble: a laptop (top left), a coffee mug (top right),
        a notebook with pen (center), wireless headphones (bottom left),
        a succulent plant (bottom right), professional product photography
        '',
        "tips": [
            "List each object with position",
            "Use consistent view (flat lay, scene, etc.)",
            "Specify count if needed (three apples)",
            "Describe arrangement pattern"
        ]
    },
    
    "style_mixing": {
        "description": "Combine multiple style references",
        "example": ''
        A portrait combining the geometric style of Picasso with the color palette
        of Matisse, showing a woman in contemplation, modern digital art
        '',
        "tips": [
            "Reference specific artists or styles",
            "Explain what aspect from each (color, geometry, texture)",
            "Add medium (oil painting, digital art, etc.)",
            "Specify dominant style"
        ]
    }
}

# Practical implementation
class AdvancedDALLEPrompts:
    """
    Generate advanced DALL-E 3 prompts.
    """
    
    @staticmethod
    def text_in_image(
        text_content: str,
        text_style: str,
        background: str,
        additional_elements: Optional[str] = None
    ) -> str:
        """Generate prompt for text-containing images."""
        prompt = f'An image with the text "{text_content}" in {text_style}, {background}'
        
        if additional_elements:
            prompt += f", {additional_elements}"
        
        return prompt
    
    @staticmethod
    def multi_panel_layout(
        panels: List[dict]
    ) -> str:
        """Create multi-panel composition."""
        panel_descriptions = []
        
        for i, panel in enumerate (panels, 1):
            position = panel.get("position", f"panel {i}")
            content = panel["content"]
            panel_descriptions.append (f"{position}: {content}")
        
        return f"A multi-panel composition with {', '.join (panel_descriptions)}"

# Examples
advanced = AdvancedDALLEPrompts()

# Text logo
logo_prompt = advanced.text_in_image(
    text_content="MOUNTAIN PEAK",
    text_style="bold geometric font with mountain silhouette integrated into letters",
    background="minimalist design on dark blue background",
    additional_elements="modern logo design, professional, clean"
)

# Before/after comparison
comparison_prompt = advanced.multi_panel_layout([
    {"position": "left panel", "content": "cluttered messy desk"},
    {"position": "right panel", "content": "organized clean desk with labeled storage"},
    {"additional": "split screen comparison, same desk, dramatic transformation"}
])
\`\`\`

## Production Integration

### Complete Generation System

\`\`\`python
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import json
import logging

@dataclass
class GenerationResult:
    """Store generation results and metadata."""
    url: str
    local_path: Optional[str]
    prompt: str
    revised_prompt: str
    timestamp: datetime
    size: str
    quality: str
    style: str
    cost: float

class ProductionDALLE:
    """
    Production-ready DALL-E system with error handling,
    retries, cost tracking, and logging.
    """
    
    # Costs in USD
    COSTS = {
        "1024x1024": {"standard": 0.040, "hd": 0.080},
        "1024x1792": {"standard": 0.080, "hd": 0.120},
        "1792x1024": {"standard": 0.080, "hd": 0.120},
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        save_directory: str = "./generated_images",
        max_retries: int = 3
    ):
        self.client = OpenAI(api_key=api_key)
        self.save_dir = save_directory
        self.max_retries = max_retries
        self.total_cost = 0.0
        self.generation_count = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create save directory
        os.makedirs (save_directory, exist_ok=True)
    
    def generate_with_retry(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
        save: bool = True,
        filename: Optional[str] = None
    ) -> GenerationResult:
        """
        Generate with automatic retry on failure.
        """
        last_error = None
        
        for attempt in range (self.max_retries):
            try:
                # Generate
                response = self.client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    style=style,
                    n=1
                )
                
                # Calculate cost
                cost = self.COSTS[size][quality]
                self.total_cost += cost
                self.generation_count += 1
                
                # Get data
                url = response.data[0].url
                revised_prompt = response.data[0].revised_prompt
                
                # Save if requested
                local_path = None
                if save:
                    if filename is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"dalle3_{timestamp}.png"
                    
                    local_path = os.path.join (self.save_dir, filename)
                    
                    # Download and save
                    img_data = requests.get (url).content
                    with open (local_path, 'wb') as f:
                        f.write (img_data)
                    
                    self.logger.info (f"Saved image to {local_path}")
                
                # Create result
                result = GenerationResult(
                    url=url,
                    local_path=local_path,
                    prompt=prompt,
                    revised_prompt=revised_prompt,
                    timestamp=datetime.now(),
                    size=size,
                    quality=quality,
                    style=style,
                    cost=cost
                )
                
                self.logger.info(
                    f"Generated image (attempt {attempt + 1}): "
                    f"cost=\${cost:.3f}, total=\${self.total_cost:.2f}"
                )

return result
                
            except Exception as e:
last_error = e
self.logger.warning(
    f"Attempt {attempt + 1} failed: {str (e)}"
)

if attempt < self.max_retries - 1:
                    # Exponential backoff
wait_time = 2 ** attempt
time.sleep (wait_time)
        
        # All retries failed
        raise Exception (f"Failed after {self.max_retries} attempts: {last_error}")
    
    def batch_generate(
    self,
    prompts: List[str],
    delay_between: float = 1.0,
        ** kwargs
) -> List[GenerationResult]:
"""
        Generate multiple images with rate limiting.
        """
results = []

for i, prompt in enumerate (prompts):
    self.logger.info (f"Generating {i+1}/{len (prompts)}")

result = self.generate_with_retry (prompt, ** kwargs)
results.append (result)
            
            # Rate limiting
if i < len (prompts) - 1:
    time.sleep (delay_between)

return results
    
    def get_statistics (self) -> dict:
"""Get usage statistics."""
return {
    "total_generations": self.generation_count,
    "total_cost": self.total_cost,
    "average_cost": self.total_cost / max (self.generation_count, 1)
}
    
    def export_metadata (self, filepath: str):
"""Export generation metadata."""
stats = self.get_statistics()

with open (filepath, 'w') as f:
json.dump (stats, f, indent = 2)

# Usage
dalle = ProductionDALLE(save_directory = "./my_images")

# Single generation
result = dalle.generate_with_retry(
    prompt = "a modern minimalist logo for a tech startup",
    quality = "hd",
    style = "natural"
)

print(f"Generated: {result.local_path}")
print(f"Cost: \${result.cost}")
print(f"Revised: {result.revised_prompt}")

# Batch generation
prompts = [
    "a serene mountain landscape",
    "a bustling city street at night",
    "a peaceful beach at sunrise"
]

results = dalle.batch_generate(
    prompts,
    quality = "standard",
    delay_between = 2.0
)

# Check total cost
stats = dalle.get_statistics()
print(f"Total cost: \${stats['total_cost']:.2f}")
print(f"Generated {stats['total_generations']} images")
\`\`\`

### Error Handling

\`\`\`python
common_errors = {
    "content_policy_violation": {
        "error": "Your request was rejected as a result of our safety system",
        "cause": "Prompt triggered content filter",
        "solutions": [
            "Rephrase to be more neutral",
            "Remove potentially sensitive terms",
            "Be more abstract/less specific",
            "Don't include violence, adult content, hate speech"
        ]
    },
    
    "rate_limit": {
        "error": "Rate limit exceeded",
        "cause": "Too many requests",
        "solutions": [
            "Add delays between requests",
            "Implement exponential backoff",
            "Track rate limits",
            "Use queuing system for high volume"
        ]
    },
    
    "invalid_prompt": {
        "error": "Invalid prompt",
        "cause": "Empty or malformed prompt",
        "solutions": [
            "Ensure prompt is not empty",
            "Check for special characters",
            "Validate before sending"
        ]
    }
}
\`\`\`

## Cost Optimization

### Strategies for Cost-Effective Generation

\`\`\`python
class CostOptimizedDALLE:
    """
    Minimize costs while maintaining quality.
    """
    
    @staticmethod
    def estimate_cost(
        num_images: int,
        size: str = "1024x1024",
        quality: str = "standard"
    ) -> dict:
        """
        Estimate generation costs.
        """
        costs = {
            "1024x1024": {"standard": 0.040, "hd": 0.080},
            "1024x1792": {"standard": 0.080, "hd": 0.120},
            "1792x1024": {"standard": 0.080, "hd": 0.120},
        }
        
        per_image = costs[size][quality]
        total = per_image * num_images
        
        return {
            "per_image": per_image,
            "total": total,
            "num_images": num_images
        }
    
    @staticmethod
    def optimize_settings (requirements: dict) -> dict:
        """
        Choose optimal settings for requirements.
        """
        # Default to most cost-effective
        settings = {
            "size": "1024x1024",
            "quality": "standard",
            "style": "vivid"
        }
        
        # Adjust based on requirements
        if requirements.get("print_quality"):
            settings["quality"] = "hd"
        
        if requirements.get("aspect_ratio") == "portrait":
            settings["size"] = "1024x1792"
        elif requirements.get("aspect_ratio") == "landscape":
            settings["size"] = "1792x1024"
        
        if requirements.get("photorealistic"):
            settings["style"] = "natural"
        
        return settings
    
    @staticmethod
    def cache_strategy (prompt: str, cache_dict: dict) -> Optional[str]:
        """
        Check cache before generating.
        """
        # Simple hash-based caching
        import hashlib
        
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        
        if prompt_hash in cache_dict:
            return cache_dict[prompt_hash]
        
        return None

# Example: Cost-conscious workflow
optimizer = CostOptimizedDALLE()

# 1. Estimate costs
estimate = optimizer.estimate_cost(
    num_images=100,
    size="1024x1024",
    quality="standard"
)
print(f"Estimated cost: \${estimate['total']:.2f}")

# 2. Optimize settings
settings = optimizer.optimize_settings({
    "print_quality": False,
    "aspect_ratio": "square",
    "photorealistic": False
})
print(f"Recommended settings: {settings}")

# 3. Use caching
image_cache = {}

def generate_cached (prompt: str) -> str:
    # Check cache
cached = optimizer.cache_strategy (prompt, image_cache)
if cached:
    print("Using cached image")
return cached
    
    # Generate new
    result = dalle.generate_with_retry (prompt, ** settings)
    
    # Cache result
prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
image_cache[prompt_hash] = result.local_path

return result.local_path
\`\`\`

## Best Practices

\`\`\`python
best_practices = {
    "prompting": [
        "Be specific and detailed",
        "Specify style, lighting, composition",
        "Use quality markers (professional, high-resolution)",
        "Mention camera/art style for consistency",
        "Put important elements first in prompt"
    ],
    
    "production": [
        "Always handle errors gracefully",
        "Implement retry logic with backoff",
        "Track costs per request",
        "Cache results when possible",
        "Rate limit your requests",
        "Log all generations for debugging"
    ],
    
    "quality": [
        "Use HD for client/print work",
        "Use standard for prototyping",
        "Choose natural for photorealism",
        "Choose vivid for marketing",
        "Generate multiple variations",
        "Review revised prompts"
    ],
    
    "cost": [
        "Start with standard quality",
        "Use 1024x1024 when possible",
        "Batch similar requests",
        "Cache duplicate requests",
        "Estimate costs before large batches"
    ]
}
\`\`\`

## Key Takeaways

- DALL-E 3 offers best-in-class prompt following and composition
- Access through OpenAI API with simple Python SDK
- Three sizes available: 1024x1024 (cheapest), portrait, landscape
- Quality: standard vs HD (2x cost, better detail)
- Style: vivid (dramatic) vs natural (realistic)
- DALL-E automatically enhances prompts (revised_prompt)
- Costs $0.04-$0.12 per image depending on settings
- Production systems need: error handling, retries, cost tracking, caching
- Best for: professional work, complex prompts, text rendering
- Use Stable Diffusion for: high volume, fine-tuning, offline use
`,
};
