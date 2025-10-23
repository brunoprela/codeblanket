/**
 * Image-to-Image Generation Section
 * Module 8: Image Generation & Computer Vision
 */

export const imagetoimagegenerationSection = {
  id: 'image-to-image-generation',
  title: 'Image-to-Image Generation',
  content: `# Image-to-Image Generation

Master img2img techniques for transforming existing images, creating variations, and style transfers.

## Overview: Building on Existing Images

Image-to-image (img2img) starts with an existing image and transforms it according to a text prompt. This is incredibly powerful for:

- **Style transfer**: Turn photo into painting
- **Variations**: Create similar but different images
- **Refinement**: Improve AI-generated images
- **Concept exploration**: Quick iterations on designs
- **Photo enhancement**: Artistic transformations

### img2img vs txt2img

\`\`\`python
comparison = {
    "text_to_image": {
        "input": "Text prompt only",
        "output": "New image from scratch",
        "control": "Low - model decides everything",
        "use_cases": ["Creating new concepts", "Original art"],
        "challenges": "Hard to get exact composition"
    },
    
    "image_to_image": {
        "input": "Image + text prompt",
        "output": "Transformed version of input",
        "control": "High - input provides structure",
        "use_cases": [
            "Style transfer",
            "Refinement",
            "Variations",
            "Photo to art conversion"
        ],
        "advantages": "Precise composition control"
    }
}
\`\`\`

## How img2img Works

### The Process

\`\`\`
Input Image → Encode to Latent → Add Noise → Denoise with Prompt → Output
                ↓
         [Noise Level = Strength]
                ↓
         [More noise = more changes]
\`\`\`

### The Strength Parameter

The most important img2img parameter:

\`\`\`python
strength_guide = {
    "0.1_to_0.3": {
        "description": "Minimal changes",
        "effect": "Subtle refinement, color adjustment",
        "preserves": "Almost everything",
        "use_case": "Fix small issues, color grading",
        "example": "Input: portrait with harsh shadows → Output: same portrait, softer shadows"
    },
    
    "0.3_to_0.5": {
        "description": "Moderate changes",
        "effect": "Style adjustment, detail enhancement",
        "preserves": "Composition and main elements",
        "use_case": "Style transfer keeping structure",
        "example": "Input: photo of building → Output: watercolor painting of same building"
    },
    
    "0.5_to_0.7": {
        "description": "Significant changes",
        "effect": "Major style shift, structural modifications",
        "preserves": "General layout",
        "use_case": "Creative transformations",
        "example": "Input: day photo → Output: night scene with same layout"
    },
    
    "0.7_to_0.9": {
        "description": "Heavy changes",
        "effect": "Almost new image, loose reference",
        "preserves": "Basic shapes only",
        "use_case": "Using image as inspiration",
        "example": "Input: rough sketch → Output: detailed artwork loosely based on sketch"
    },
    
    "0.9_to_1.0": {
        "description": "Almost txt2img",
        "effect": "Mostly new, very loose reference",
        "preserves": "Very little",
        "use_case": "Image as weak guidance",
        "example": "Nearly ignores input image"
    }
}
\`\`\`

## Implementation

### Basic img2img

\`\`\`python
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
from typing import Optional, List

class Img2ImgGenerator:
    """
    Image-to-image generation with Stable Diffusion.
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1",
        device: str = "cuda"
    ):
        self.device = device
        
        # Load img2img pipeline
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.pipe = self.pipe.to(device)
        self.pipe.enable_attention_slicing()
        
        # Try xformers
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.5,
        steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Transform an image based on prompt.
        
        Args:
            image: Input image (PIL Image)
            prompt: What to transform into
            negative_prompt: What to avoid
            strength: How much to change (0.0-1.0)
            steps: Quality (more steps = better)
            guidance_scale: Prompt adherence
            seed: For reproducibility
        
        Returns:
            List of transformed images
        """
        # Resize to appropriate size
        image = self._resize_image(image)
        
        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate
        with torch.autocast(self.device):
            result = self.pipe(
                prompt=prompt,
                image=image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                generator=generator
            )
        
        return result.images
    
    def _resize_image(
        self,
        image: Image.Image,
        max_size: int = 768
    ) -> Image.Image:
        """
        Resize image to appropriate dimensions.
        Must be multiple of 64.
        """
        w, h = image.size
        
        # Scale down if too large
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            w = int(w * scale)
            h = int(h * scale)
        
        # Round to nearest 64
        w = (w // 64) * 64
        h = (h // 64) * 64
        
        # Ensure minimum size
        w = max(w, 512)
        h = max(h, 512)
        
        return image.resize((w, h), Image.LANCZOS)
    
    def generate_variations(
        self,
        image: Image.Image,
        prompt: str,
        num_variations: int = 4,
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate multiple variations of an image.
        """
        results = []
        
        for i in range(num_variations):
            # Use different seed for each
            images = self.generate(
                image=image,
                prompt=prompt,
                seed=42 + i if kwargs.get('seed') else None,
                **{k: v for k, v in kwargs.items() if k != 'seed'}
            )
            results.extend(images)
        
        return results

# Usage
img2img = Img2ImgGenerator()

# Load input image
input_image = Image.open("photo.jpg")

# Transform to oil painting
oil_painting = img2img.generate(
    image=input_image,
    prompt="oil painting, impressionist style, brush strokes, canvas texture",
    strength=0.5,
    steps=50
)[0]

oil_painting.save("oil_painting.png")

# Generate variations
variations = img2img.generate_variations(
    image=input_image,
    prompt="same scene, different lighting and colors",
    num_variations=4,
    strength=0.4
)

for i, var in enumerate(variations):
    var.save(f"variation_{i}.png")
\`\`\`

## Common Use Cases

### 1. Style Transfer

\`\`\`python
class StyleTransfer:
    """
    Apply artistic styles to photos.
    """
    
    def __init__(self, img2img_generator):
        self.gen = img2img_generator
    
    def to_oil_painting(self, image: Image.Image) -> Image.Image:
        """Convert photo to oil painting."""
        return self.gen.generate(
            image=image,
            prompt="""
            oil painting, impressionist style,
            visible brush strokes, canvas texture,
            artistic, vibrant colors, museum quality
            """,
            negative_prompt="photo, photograph, realistic",
            strength=0.55,
            steps=50
        )[0]
    
    def to_watercolor(self, image: Image.Image) -> Image.Image:
        """Convert to watercolor painting."""
        return self.gen.generate(
            image=image,
            prompt="""
            watercolor painting, soft colors,
            flowing paint, paper texture,
            artistic, delicate, transparent layers
            """,
            negative_prompt="photo, digital, harsh lines",
            strength=0.5,
            steps=50
        )[0]
    
    def to_anime(self, image: Image.Image) -> Image.Image:
        """Convert to anime style."""
        return self.gen.generate(
            image=image,
            prompt="""
            anime style, manga, cel shading,
            vibrant colors, clean lines,
            studio quality, detailed
            """,
            negative_prompt="realistic, photo, 3d",
            strength=0.65,
            steps=50
        )[0]
    
    def to_sketch(self, image: Image.Image) -> Image.Image:
        """Convert to pencil sketch."""
        return self.gen.generate(
            image=image,
            prompt="""
            pencil sketch, graphite drawing,
            hand-drawn, detailed shading,
            black and white, artistic
            """,
            negative_prompt="color, photo, digital",
            strength=0.6,
            steps=50
        )[0]
    
    def to_vintage_photo(self, image: Image.Image) -> Image.Image:
        """Apply vintage photo effect."""
        return self.gen.generate(
            image=image,
            prompt="""
            vintage photograph, 1970s aesthetic,
            faded colors, film grain, nostalgic,
            retro, slightly desaturated
            """,
            negative_prompt="modern, digital, sharp, vibrant",
            strength=0.35,
            steps=40
        )[0]

# Usage
style_transfer = StyleTransfer(img2img)

photo = Image.open("portrait.jpg")

# Try different styles
oil = style_transfer.to_oil_painting(photo)
watercolor = style_transfer.to_watercolor(photo)
anime = style_transfer.to_anime(photo)
sketch = style_transfer.to_sketch(photo)

oil.save("oil.png")
watercolor.save("watercolor.png")
\`\`\`

### 2. Photo Enhancement and Refinement

\`\`\`python
class PhotoEnhancer:
    """
    Enhance and refine photographs.
    """
    
    def __init__(self, img2img_generator):
        self.gen = img2img_generator
    
    def enhance_details(self, image: Image.Image) -> Image.Image:
        """Add more detail and sharpness."""
        return self.gen.generate(
            image=image,
            prompt="""
            highly detailed, sharp focus, enhanced clarity,
            professional photography, high resolution,
            improved details, crisp, clear
            """,
            negative_prompt="blurry, soft, low quality",
            strength=0.25,  # Subtle changes
            steps=40,
            guidance_scale=7.0
        )[0]
    
    def improve_lighting(self, image: Image.Image) -> Image.Image:
        """Improve lighting and exposure."""
        return self.gen.generate(
            image=image,
            prompt="""
            perfect lighting, well-exposed, balanced exposure,
            professional photography, natural light,
            no harsh shadows, even illumination
            """,
            negative_prompt="dark, underexposed, overexposed, harsh shadows",
            strength=0.3,
            steps=40
        )[0]
    
    def enhance_colors(self, image: Image.Image) -> Image.Image:
        """Enhance color vibrancy."""
        return self.gen.generate(
            image=image,
            prompt="""
            vibrant colors, enhanced saturation,
            color grading, cinematic colors,
            beautiful color palette, rich tones
            """,
            negative_prompt="dull, washed out, desaturated, gray",
            strength=0.3,
            steps=40
        )[0]
    
    def make_professional(self, image: Image.Image) -> Image.Image:
        """Give photo professional look."""
        return self.gen.generate(
            image=image,
            prompt="""
            professional photography, studio quality,
            perfect composition, professional lighting,
            high-end photography, magazine quality,
            award-winning, masterfully composed
            """,
            negative_prompt="amateur, low quality, poor composition",
            strength=0.35,
            steps=50
        )[0]

# Usage
enhancer = PhotoEnhancer(img2img)

photo = Image.open("photo.jpg")

# Various enhancements
detailed = enhancer.enhance_details(photo)
better_lit = enhancer.improve_lighting(photo)
vibrant = enhancer.enhance_colors(photo)
professional = enhancer.make_professional(photo)
\`\`\`

### 3. Time and Weather Transformations

\`\`\`python
class SceneTransformer:
    """
    Transform scene conditions (time, weather, season).
    """
    
    def __init__(self, img2img_generator):
        self.gen = img2img_generator
    
    def to_night(self, image: Image.Image) -> Image.Image:
        """Convert day scene to night."""
        return self.gen.generate(
            image=image,
            prompt="""
            night time, dark sky, artificial lighting,
            street lights, night photography,
            moonlight, stars, evening atmosphere
            """,
            negative_prompt="daylight, bright, sunny",
            strength=0.6,
            steps=50
        )[0]
    
    def to_sunset(self, image: Image.Image) -> Image.Image:
        """Add sunset lighting."""
        return self.gen.generate(
            image=image,
            prompt="""
            golden hour, sunset lighting, warm colors,
            orange and pink sky, dramatic sunset,
            beautiful evening light, magic hour
            """,
            negative_prompt="midday, harsh light",
            strength=0.45,
            steps=50
        )[0]
    
    def add_rain(self, image: Image.Image) -> Image.Image:
        """Make scene rainy."""
        return self.gen.generate(
            image=image,
            prompt="""
            rainy day, wet surfaces, rain drops,
            overcast sky, puddles, rainfall,
            moody atmosphere, gray sky
            """,
            negative_prompt="sunny, dry, clear sky",
            strength=0.5,
            steps=50
        )[0]
    
    def add_snow(self, image: Image.Image) -> Image.Image:
        """Add snow to scene."""
        return self.gen.generate(
            image=image,
            prompt="""
            winter scene, snow covered, snowing,
            white snow, cold atmosphere,
            winter wonderland, icy
            """,
            negative_prompt="summer, warm, green",
            strength=0.55,
            steps=50
        )[0]
    
    def to_autumn(self, image: Image.Image) -> Image.Image:
        """Transform to autumn colors."""
        return self.gen.generate(
            image=image,
            prompt="""
            autumn season, fall colors, orange and red leaves,
            fall foliage, autumn atmosphere,
            warm autumn tones, seasonal colors
            """,
            negative_prompt="spring, summer, green",
            strength=0.5,
            steps=50
        )[0]

# Usage
transformer = SceneTransformer(img2img)

day_photo = Image.open("street_day.jpg")

# Transform conditions
night = transformer.to_night(day_photo)
sunset = transformer.to_sunset(day_photo)
rainy = transformer.add_rain(day_photo)
snowy = transformer.add_snow(day_photo)
\`\`\`

### 4. Creative Reimagining

\`\`\`python
class CreativeReimagineer:
    """
    Creatively reimagine images in different contexts.
    """
    
    def __init__(self, img2img_generator):
        self.gen = img2img_generator
    
    def make_futuristic(self, image: Image.Image) -> Image.Image:
        """Add futuristic/sci-fi elements."""
        return self.gen.generate(
            image=image,
            prompt="""
            futuristic, sci-fi, cyberpunk aesthetic,
            neon lights, holographic elements,
            advanced technology, futuristic city,
            high-tech, dystopian future
            """,
            negative_prompt="old, vintage, historical",
            strength=0.6,
            steps=50
        )[0]
    
    def make_fantasy(self, image: Image.Image) -> Image.Image:
        """Add fantasy/magical elements."""
        return self.gen.generate(
            image=image,
            prompt="""
            fantasy world, magical atmosphere,
            ethereal lighting, mystical elements,
            enchanted, fairy tale, magical realm,
            fantasy art style, epic fantasy
            """,
            negative_prompt="realistic, modern, mundane",
            strength=0.65,
            steps=50
        )[0]
    
    def make_post_apocalyptic(self, image: Image.Image) -> Image.Image:
        """Transform to post-apocalyptic."""
        return self.gen.generate(
            image=image,
            prompt="""
            post-apocalyptic, abandoned, overgrown with plants,
            ruined, decayed, dystopian, wasteland,
            nature reclaiming, desolate, survival
            """,
            negative_prompt="pristine, new, maintained, populated",
            strength=0.7,
            steps=50
        )[0]
    
    def miniaturize(self, image: Image.Image) -> Image.Image:
        """Make scene look like miniature/tilt-shift."""
        return self.gen.generate(
            image=image,
            prompt="""
            tilt-shift photography, miniature effect,
            toy-like, shallow depth of field,
            small scale model, diorama style,
            selective focus, tiny world
            """,
            negative_prompt="normal scale, full focus",
            strength=0.4,
            steps=50
        )[0]

# Usage
reimaginer = CreativeReimagineer(img2img)

city_photo = Image.open("city.jpg")

futuristic = reimaginer.make_futuristic(city_photo)
fantasy = reimaginer.make_fantasy(city_photo)
apocalyptic = reimaginer.make_post_apocalyptic(city_photo)
\`\`\`

## Advanced Techniques

### Progressive Refinement

\`\`\`python
class ProgressiveRefiner:
    """
    Refine images through multiple passes.
    """
    
    def __init__(self, img2img_generator):
        self.gen = img2img_generator
    
    def refine_iteratively(
        self,
        image: Image.Image,
        prompt: str,
        iterations: int = 3,
        strength: float = 0.3
    ) -> List[Image.Image]:
        """
        Apply img2img multiple times for gradual refinement.
        """
        results = [image]
        current = image
        
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}")
            
            # Gradually reduce strength
            current_strength = strength * (1.0 - (i * 0.1))
            
            refined = self.gen.generate(
                image=current,
                prompt=prompt,
                strength=current_strength,
                steps=40
            )[0]
            
            results.append(refined)
            current = refined
        
        return results
    
    def upscale_and_refine(
        self,
        image: Image.Image,
        scale: int = 2
    ) -> Image.Image:
        """
        Upscale image and add details.
        
        Note: For production, use specialized upscaling models.
        This is a simplified example.
        """
        # Simple upscale
        w, h = image.size
        upscaled = image.resize((w * scale, h * scale), Image.LANCZOS)
        
        # Add details with img2img
        detailed = self.gen.generate(
            image=upscaled,
            prompt="""
            highly detailed, sharp focus, enhanced details,
            high resolution, crisp, clear, professional quality
            """,
            strength=0.2,  # Subtle detail enhancement
            steps=50
        )[0]
        
        return detailed

# Usage
refiner = ProgressiveRefiner(img2img)

# Progressive refinement
rough_image = Image.open("rough.png")
refinement_steps = refiner.refine_iteratively(
    image=rough_image,
    prompt="professional photography, highly detailed, perfect",
    iterations=3,
    strength=0.3
)

# Save each step
for i, img in enumerate(refinement_steps):
    img.save(f"refinement_step_{i}.png")
\`\`\`

### Strength Testing

\`\`\`python
class StrengthExplorer:
    """
    Test different strength values to find optimal.
    """
    
    def __init__(self, img2img_generator):
        self.gen = img2img_generator
    
    def test_strengths(
        self,
        image: Image.Image,
        prompt: str,
        strengths: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
        **kwargs
    ) -> dict:
        """
        Generate with different strength values.
        """
        results = {}
        
        for strength in strengths:
            print(f"Testing strength: {strength}")
            
            output = self.gen.generate(
                image=image,
                prompt=prompt,
                strength=strength,
                **kwargs
            )[0]
            
            results[strength] = output
        
        return results
    
    def create_strength_comparison(
        self,
        results: dict,
        output_path: str = "strength_comparison.png"
    ):
        """
        Create side-by-side comparison of different strengths.
        """
        from PIL import Image, ImageDraw, ImageFont
        
        images = list(results.values())
        strengths = list(results.keys())
        
        # Create grid
        w, h = images[0].size
        cols = min(len(images), 3)
        rows = (len(images) + cols - 1) // cols
        
        grid = Image.new('RGB', (w * cols, h * rows + 30))
        draw = ImageDraw.Draw(grid)
        
        for i, (strength, img) in enumerate(results.items()):
            x = (i % cols) * w
            y = (i // cols) * h + 30
            
            grid.paste(img, (x, y))
            
            # Add label
            draw.text(
                (x + 10, (i // cols) * h + 5),
                f"Strength: {strength}",
                fill='white'
            )
        
        grid.save(output_path)
        print(f"Comparison saved to {output_path}")

# Usage
explorer = StrengthExplorer(img2img)

input_img = Image.open("photo.jpg")

# Test different strengths
results = explorer.test_strengths(
    image=input_img,
    prompt="oil painting style, impressionist",
    strengths=[0.3, 0.4, 0.5, 0.6, 0.7],
    steps=40,
    seed=42
)

# Create comparison image
explorer.create_strength_comparison(results)
\`\`\`

## Production Workflows

### Batch img2img Processing

\`\`\`python
import os
from pathlib import Path
from typing import Callable

class BatchImg2ImgProcessor:
    """
    Process multiple images with img2img.
    """
    
    def __init__(self, img2img_generator):
        self.gen = img2img_generator
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        transform_fn: Callable,
        file_pattern: str = "*.jpg"
    ):
        """
        Process all images in directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Where to save outputs
            transform_fn: Function that takes Image, returns Image
            file_pattern: Glob pattern for files
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find all images
        images = list(input_path.glob(file_pattern))
        total = len(images)
        
        print(f"Processing {total} images...")
        
        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{total}] Processing {img_path.name}")
            
            # Load and transform
            image = Image.open(img_path)
            result = transform_fn(image)
            
            # Save with same name
            output_file = output_path / img_path.name
            result.save(output_file)
        
        print(f"Done! Saved to {output_dir}")
    
    def apply_style_to_batch(
        self,
        input_dir: str,
        output_dir: str,
        style_prompt: str,
        strength: float = 0.5,
        **kwargs
    ):
        """Apply same style to all images."""
        
        def transform(image):
            return self.gen.generate(
                image=image,
                prompt=style_prompt,
                strength=strength,
                **kwargs
            )[0]
        
        self.process_directory(input_dir, output_dir, transform)

# Usage
batch = BatchImg2ImgProcessor(img2img)

# Apply oil painting style to folder of photos
batch.apply_style_to_batch(
    input_dir="./photos",
    output_dir="./oil_paintings",
    style_prompt="oil painting, impressionist style, artistic",
    strength=0.5,
    steps=40
)
\`\`\`

## Key Takeaways

- **img2img** transforms existing images based on text prompts
- **Strength parameter** controls transformation amount (0.0-1.0)
- Low strength (0.2-0.4): subtle changes, detail enhancement
- Medium strength (0.4-0.6): style transfer, moderate changes
- High strength (0.6-0.8): major transformations, creative reimagining
- **Use cases**: style transfer, photo enhancement, variations, scene transformation
- **Input image provides structure**: easier to control composition than txt2img
- **Multiple passes**: can refine iteratively for better results
- **Batch processing**: apply same transformation to many images
- **Testing strengths**: find optimal value for your use case
`,
};
