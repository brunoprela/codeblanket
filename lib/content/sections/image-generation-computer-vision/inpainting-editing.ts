/**
 * Inpainting & Editing Section
 * Module 8: Image Generation & Computer Vision
 */

export const inpaintingeditingSection = {
  id: 'inpainting-editing',
  title: 'Inpainting & Editing',
  content: `# Inpainting & Editing

Master inpainting for precise image editing - removing, replacing, and adding elements to images.

## Overview: Surgical Image Editing

Inpainting allows you to edit specific parts of an image while keeping the rest unchanged. It\'s like Photoshop's content-aware fill but powered by AI and text prompts.

### What Inpainting Does

\`\`\`python
inpainting_capabilities = {
    "remove_objects": "Erase unwanted elements",
    "replace_elements": "Change specific parts",
    "add_objects": "Insert new elements naturally",
    "fix_issues": "Repair defects or problems",
    "extend_images": "Outpainting - expand beyond borders",
    "background_changes": "Replace backgrounds completely"
}

# Example workflow
"Photo with person in background" 
→ Mask the person
→ Prompt: "empty park bench"
→ Result: Person removed, natural background
\`\`\`

## How Inpainting Works

### The Process

\`\`\`
Original Image + Mask + Prompt → Inpainted Result
     ↓             ↓
  [Keep this] [Change this]
                  ↓
          [Generate based on prompt]
                  ↓
          [Blend with original]
\`\`\`

### Key Concepts

\`\`\`python
inpainting_concepts = {
    "mask": {
        "description": "Black/white image showing what to change",
        "white": "Areas to inpaint (change)",
        "black": "Areas to keep (preserve)",
        "importance": "Mask quality = result quality"
    },
    
    "mask_blur": {
        "description": "Feather edges of mask",
        "purpose": "Smooth blending at boundaries",
        "typical_value": "4-8 pixels",
        "effect": "Higher = softer transitions"
    },
    
    "inpaint_vs_original": {
        "inpaint_model": "Trained specifically for inpainting",
        "regular_model": "Can inpaint but less seamlessly",
        "recommendation": "Use inpaint models when available"
    }
}
\`\`\`

## Implementation

### Basic Inpainting

\`\`\`python
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image, ImageDraw
import numpy as np
from typing import Optional, Tuple

class Inpainter:
    """
    Image inpainting with Stable Diffusion.
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-inpainting",
        device: str = "cuda"
    ):
        self.device = device
        
        # Load inpainting pipeline
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        
        self.pipe = self.pipe.to (device)
        self.pipe.enable_attention_slicing()
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Inpaint masked region.
        
        Args:
            image: Original image
            mask: Mask (white = inpaint, black = keep)
            prompt: What to generate in masked area
            negative_prompt: What to avoid
            steps: Quality
            guidance_scale: Prompt adherence
            seed: Reproducibility
        
        Returns:
            Inpainted image
        """
        # Ensure sizes match
        if image.size != mask.size:
            mask = mask.resize (image.size, Image.LANCZOS)
        
        # Resize to appropriate size
        image, mask = self._prepare_inputs (image, mask)
        
        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator (device=self.device).manual_seed (seed)
        
        # Inpaint
        result = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        return result
    
    def _prepare_inputs(
        self,
        image: Image.Image,
        mask: Image.Image,
        max_size: int = 768
    ) -> Tuple[Image.Image, Image.Image]:
        """Resize and prepare inputs."""
        w, h = image.size
        
        # Scale down if too large
        if max (w, h) > max_size:
            scale = max_size / max (w, h)
            w = int (w * scale)
            h = int (h * scale)
        
        # Round to multiples of 64
        w = (w // 64) * 64
        h = (h // 64) * 64
        
        # Resize
        image = image.resize((w, h), Image.LANCZOS)
        mask = mask.resize((w, h), Image.LANCZOS)
        
        return image, mask
    
    def create_mask_from_bbox(
        self,
        image_size: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
        feather: int = 10
    ) -> Image.Image:
        """
        Create mask from bounding box.
        
        Args:
            image_size: (width, height)
            bbox: (x1, y1, x2, y2)
            feather: Blur radius for soft edges
        """
        # Create black image
        mask = Image.new('RGB', image_size, 'black')
        draw = ImageDraw.Draw (mask)
        
        # Draw white rectangle
        draw.rectangle (bbox, fill='white')
        
        # Apply Gaussian blur for feathering
        from PIL import ImageFilter
        mask = mask.filter(ImageFilter.GaussianBlur (radius=feather))
        
        return mask

# Usage
inpainter = Inpainter()

# Load image
image = Image.open("photo.jpg")

# Create mask (white = areas to inpaint)
mask = Image.open("mask.png")
# Or create from bbox:
# mask = inpainter.create_mask_from_bbox(
#     image_size=image.size,
#     bbox=(100, 100, 300, 400),
#     feather=10
# )

# Inpaint
result = inpainter.inpaint(
    image=image,
    mask=mask,
    prompt="empty space, natural background",
    negative_prompt="person, object, distorted",
    steps=50
)

result.save("inpainted.png")
\`\`\`

## Common Use Cases

### 1. Object Removal

\`\`\`python
class ObjectRemover:
    """
    Remove unwanted objects from images.
    """
    
    def __init__(self, inpainter: Inpainter):
        self.inpainter = inpainter
    
    def remove_object(
        self,
        image: Image.Image,
        mask: Image.Image,
        background_description: str = "natural background, same as surroundings"
    ) -> Image.Image:
        """
        Remove object and fill with background.
        """
        result = self.inpainter.inpaint(
            image=image,
            mask=mask,
            prompt=f"{background_description}, seamless, photorealistic",
            negative_prompt="object, person, artifact, seam, distortion",
            steps=50,
            guidance_scale=7.0
        )
        
        return result
    
    def remove_person(
        self,
        image: Image.Image,
        person_mask: Image.Image
    ) -> Image.Image:
        """Remove person from photo."""
        return self.remove_object(
            image=image,
            mask=person_mask,
            background_description="empty scene, no people, natural continuation of background"
        )
    
    def remove_watermark(
        self,
        image: Image.Image,
        watermark_mask: Image.Image
    ) -> Image.Image:
        """Remove watermark or text."""
        return self.remove_object(
            image=image,
            mask=watermark_mask,
            background_description="clean, no text, no watermark, seamless texture"
        )

# Usage
remover = ObjectRemover (inpainter)

photo = Image.open("photo_with_person.jpg")
person_mask = Image.open("person_mask.png")

# Remove person
clean_photo = remover.remove_person (photo, person_mask)
clean_photo.save("no_person.png")
\`\`\`

### 2. Object Replacement

\`\`\`python
class ObjectReplacer:
    """
    Replace objects in images.
    """
    
    def __init__(self, inpainter: Inpainter):
        self.inpainter = inpainter
    
    def replace_object(
        self,
        image: Image.Image,
        mask: Image.Image,
        new_object: str,
        style_match: bool = True
    ) -> Image.Image:
        """
        Replace masked object with something new.
        """
        # Build prompt to match style
        if style_match:
            prompt = f"{new_object}, matching the style and lighting of the image, photorealistic, seamless"
        else:
            prompt = new_object
        
        result = self.inpainter.inpaint(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt="distorted, unnatural, inconsistent lighting",
            steps=50,
            guidance_scale=8.0
        )
        
        return result
    
    def change_outfit(
        self,
        image: Image.Image,
        clothing_mask: Image.Image,
        new_outfit: str
    ) -> Image.Image:
        """Change person's clothing."""
        return self.replace_object(
            image=image,
            mask=clothing_mask,
            new_object=f"{new_outfit}, realistic clothing, natural fit",
            style_match=True
        )
    
    def replace_background(
        self,
        image: Image.Image,
        background_mask: Image.Image,
        new_background: str
    ) -> Image.Image:
        """Replace entire background."""
        return self.replace_object(
            image=image,
            mask=background_mask,
            new_object=new_background,
            style_match=True
        )

# Usage
replacer = ObjectReplacer (inpainter)

# Change furniture
room = Image.open("room.jpg")
couch_mask = Image.open("couch_mask.png")

new_room = replacer.replace_object(
    image=room,
    mask=couch_mask,
    new_object="modern gray sectional sofa"
)

# Change background
portrait = Image.open("portrait.jpg")
bg_mask = Image.open("background_mask.png")

studio = replacer.replace_background(
    image=portrait,
    mask=bg_mask,
    new_background="professional photography studio, gray backdrop"
)
\`\`\`

### 3. Adding Objects

\`\`\`python
class ObjectAdder:
    """
    Add new objects to images.
    """
    
    def __init__(self, inpainter: Inpainter):
        self.inpainter = inpainter
    
    def add_object(
        self,
        image: Image.Image,
        insertion_mask: Image.Image,
        object_description: str,
        blend_naturally: bool = True
    ) -> Image.Image:
        """
        Add object in specified location.
        """
        prompt_parts = [object_description]
        
        if blend_naturally:
            prompt_parts.extend([
                "naturally integrated",
                "matching lighting and perspective",
                "photorealistic",
                "seamless composition"
            ])
        
        prompt = ", ".join (prompt_parts)
        
        result = self.inpainter.inpaint(
            image=image,
            mask=insertion_mask,
            prompt=prompt,
            negative_prompt="floating, unrealistic, wrong perspective, poor lighting match",
            steps=50,
            guidance_scale=8.5
        )
        
        return result
    
    def add_furniture(
        self,
        room_image: Image.Image,
        location_mask: Image.Image,
        furniture: str
    ) -> Image.Image:
        """Add furniture to room."""
        return self.add_object(
            image=room_image,
            insertion_mask=location_mask,
            object_description=f"{furniture}, interior design, matching room style"
        )
    
    def add_person(
        self,
        scene: Image.Image,
        location_mask: Image.Image,
        person_description: str
    ) -> Image.Image:
        """Add person to scene."""
        return self.add_object(
            image=scene,
            insertion_mask=location_mask,
            object_description=f"{person_description}, natural pose, realistic lighting"
        )

# Usage
adder = ObjectAdder (inpainter)

# Add lamp to room
room = Image.open("room.jpg")
corner_mask = inpainter.create_mask_from_bbox(
    image_size=room.size,
    bbox=(50, 50, 200, 400),
    feather=15
)

with_lamp = adder.add_furniture(
    room_image=room,
    location_mask=corner_mask,
    furniture="modern floor lamp, minimalist design"
)
\`\`\`

## Outpainting: Extending Images

### Image Extension

\`\`\`python
class Outpainter:
    """
    Extend images beyond their borders (outpainting).
    """
    
    def __init__(self, inpainter: Inpainter):
        self.inpainter = inpainter
    
    def extend_image(
        self,
        image: Image.Image,
        extend_left: int = 0,
        extend_right: int = 0,
        extend_top: int = 0,
        extend_bottom: int = 0,
        prompt: str = "natural continuation of the scene"
    ) -> Image.Image:
        """
        Extend image in specified directions.
        
        Args:
            image: Original image
            extend_*: Pixels to extend in each direction
            prompt: Description for extended areas
        """
        orig_w, orig_h = image.size
        
        # Calculate new size
        new_w = orig_w + extend_left + extend_right
        new_h = orig_h + extend_top + extend_bottom
        
        # Create canvas
        canvas = Image.new('RGB', (new_w, new_h), 'black')
        
        # Paste original image
        canvas.paste (image, (extend_left, extend_top))
        
        # Create mask (white = extend)
        mask = Image.new('RGB', (new_w, new_h), 'white')
        
        # Black out original image area in mask
        black_box = Image.new('RGB', (orig_w, orig_h), 'black')
        mask.paste (black_box, (extend_left, extend_top))
        
        # Inpaint extensions
        result = self.inpainter.inpaint(
            image=canvas,
            mask=mask,
            prompt=f"{prompt}, coherent, seamless extension",
            negative_prompt="seam, border, edge, discontinuity",
            steps=50,
            guidance_scale=7.5
        )
        
        return result
    
    def make_square(
        self,
        image: Image.Image,
        prompt: str = "natural continuation"
    ) -> Image.Image:
        """
        Extend image to make it square.
        """
        w, h = image.size
        
        if w > h:
            # Extend vertically
            diff = w - h
            return self.extend_image(
                image=image,
                extend_top=diff // 2,
                extend_bottom=diff - (diff // 2),
                prompt=prompt
            )
        elif h > w:
            # Extend horizontally
            diff = h - w
            return self.extend_image(
                image=image,
                extend_left=diff // 2,
                extend_right=diff - (diff // 2),
                prompt=prompt
            )
        else:
            return image
    
    def make_panorama(
        self,
        images: list[Image.Image],
        overlap: int = 50
    ) -> Image.Image:
        """
        Stitch images into panorama with AI blending.
        (Simplified version)
        """
        # Concatenate images
        total_width = sum (img.width for img in images) - (len (images) - 1) * overlap
        max_height = max (img.height for img in images)
        
        canvas = Image.new('RGB', (total_width, max_height))
        
        x_offset = 0
        for img in images:
            canvas.paste (img, (x_offset, 0))
            x_offset += img.width - overlap
        
        # Create mask for seams
        mask = self._create_seam_mask (images, overlap, canvas.size)
        
        # Blend seams
        result = self.inpainter.inpaint(
            image=canvas,
            mask=mask,
            prompt="seamless panorama, natural continuation",
            steps=40
        )
        
        return result
    
    def _create_seam_mask (self, images, overlap, canvas_size):
        """Create mask for seam areas."""
        mask = Image.new('RGB', canvas_size, 'black')
        draw = ImageDraw.Draw (mask)
        
        x_offset = 0
        for i, img in enumerate (images[:-1]):
            seam_x = x_offset + img.width - overlap
            draw.rectangle(
                [seam_x - 20, 0, seam_x + 20, canvas_size[1]],
                fill='white'
            )
            x_offset += img.width - overlap
        
        return mask

# Usage
outpainter = Outpainter (inpainter)

# Extend image
portrait = Image.open("portrait.jpg")
wider = outpainter.extend_image(
    image=portrait,
    extend_left=200,
    extend_right=200,
    prompt="professional photography studio background"
)

# Make square for social media
landscape = Image.open("landscape.jpg")
square = outpainter.make_square(
    image=landscape,
    prompt="sky and clouds continuation"
)
\`\`\`

## Interactive Mask Creation

\`\`\`python
class InteractiveMaskCreator:
    """
    Tools for creating masks programmatically.
    """
    
    @staticmethod
    def create_brush_mask(
        image_size: Tuple[int, int],
        brush_strokes: list[list[Tuple[int, int]]],
        brush_size: int = 20
    ) -> Image.Image:
        """
        Create mask from brush strokes.
        
        Args:
            image_size: (width, height)
            brush_strokes: List of stroke points [[points], [points], ...]
            brush_size: Brush diameter
        """
        mask = Image.new('RGB', image_size, 'black')
        draw = ImageDraw.Draw (mask)
        
        for stroke in brush_strokes:
            if len (stroke) < 2:
                continue
            
            # Draw lines between points
            for i in range (len (stroke) - 1):
                draw.line(
                    [stroke[i], stroke[i + 1]],
                    fill='white',
                    width=brush_size
                )
            
            # Draw circles at points for smooth strokes
            for point in stroke:
                draw.ellipse(
                    [
                        point[0] - brush_size // 2,
                        point[1] - brush_size // 2,
                        point[0] + brush_size // 2,
                        point[1] + brush_size // 2
                    ],
                    fill='white'
                )
        
        return mask
    
    @staticmethod
    def create_polygon_mask(
        image_size: Tuple[int, int],
        polygon_points: list[Tuple[int, int]],
        feather: int = 10
    ) -> Image.Image:
        """
        Create mask from polygon selection.
        """
        mask = Image.new('RGB', image_size, 'black')
        draw = ImageDraw.Draw (mask)
        
        draw.polygon (polygon_points, fill='white')
        
        # Feather edges
        from PIL import ImageFilter
        mask = mask.filter(ImageFilter.GaussianBlur (radius=feather))
        
        return mask
    
    @staticmethod
    def invert_mask (mask: Image.Image) -> Image.Image:
        """Invert mask (swap black and white)."""
        from PIL import ImageOps
        return ImageOps.invert (mask.convert('L')).convert('RGB')
    
    @staticmethod
    def dilate_mask (mask: Image.Image, iterations: int = 5) -> Image.Image:
        """Expand mask area."""
        from PIL import ImageFilter
        
        result = mask
        for _ in range (iterations):
            result = result.filter(ImageFilter.MaxFilter(3))
        
        return result
    
    @staticmethod
    def erode_mask (mask: Image.Image, iterations: int = 5) -> Image.Image:
        """Shrink mask area."""
        from PIL import ImageFilter
        
        result = mask
        for _ in range (iterations):
            result = result.filter(ImageFilter.MinFilter(3))
        
        return result

# Usage
mask_creator = InteractiveMaskCreator()

# Create mask from brush strokes
strokes = [
    [(100, 100), (150, 120), (200, 140)],  # Stroke 1
    [(180, 200), (220, 240), (260, 280)],  # Stroke 2
]

brush_mask = mask_creator.create_brush_mask(
    image_size=(800, 600),
    brush_strokes=strokes,
    brush_size=30
)

# Create polygon mask
polygon = [(100, 100), (300, 100), (300, 400), (100, 400)]
poly_mask = mask_creator.create_polygon_mask(
    image_size=(800, 600),
    polygon_points=polygon,
    feather=15
)
\`\`\`

## Production Tips

\`\`\`python
production_inpainting_tips = {
    "mask_quality": [
        "Feather mask edges (5-15 pixels)",
        "Slightly expand mask beyond object",
        "Clean, precise masks = better results",
        "Test different feather amounts"
    ],
    
    "prompts": [
        "Be specific about context",
        "Mention 'matching lighting' or 'same style'",
        "Use 'seamless' and 'natural'",
        "Negative prompt 'seam', 'border', 'edge'"
    ],
    
    "quality": [
        "Use 40-60 steps for best quality",
        "Guidance scale 7-9 typical",
        "Higher steps for complex inpainting",
        "Generate multiple, pick best"
    ],
    
    "challenges": [
        "Large areas harder than small",
        "Complex textures difficult",
        "Precise object boundaries hard",
        "May need multiple attempts"
    ]
}
\`\`\`

## Key Takeaways

- **Inpainting** edits specific image regions while preserving rest
- **Mask** defines what to change (white) vs keep (black)
- **Feathering** mask edges critical for seamless blending
- **Use cases**: object removal, replacement, addition, outpainting
- **Object removal**: Prompt for natural background continuation
- **Object addition**: Match lighting and perspective in prompt
- **Outpainting**: Extend images beyond original borders
- **Quality masks** = quality results
- **Use inpainting models** for best results
- **Multiple attempts** often needed for perfect results
`,
};
