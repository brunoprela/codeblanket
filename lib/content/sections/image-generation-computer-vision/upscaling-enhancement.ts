/**
 * Upscaling & Enhancement Section
 * Module 8: Image Generation & Computer Vision
 */

export const upscalingenhancementSection = {
  id: 'upscaling-enhancement',
  title: 'Upscaling & Enhancement',
  content: `# Upscaling & Enhancement

Master AI-powered image upscaling and enhancement for increasing resolution and improving quality.

## Overview: Beyond Simple Resizing

Traditional upscaling (bilinear, bicubic) just interpolates pixels. AI upscaling **generates** realistic details that should be there.

### Why AI Upscaling Matters

\`\`\`python
traditional_vs_ai = {
    "traditional_upscaling": {
        "method": "Interpolate between existing pixels",
        "result": "Blurry, soft, no new details",
        "quality": "Poor for large scale factors",
        "use_case": "When blurriness is acceptable"
    },
    
    "ai_upscaling": {
        "method": "Generate realistic details using AI",
        "result": "Sharp, detailed, natural looking",
        "quality": "Excellent even at 4x",
        "use_case": "Professional work, restoration, enhancement"
    }
}

# Example impact
original_size = (512, 512)
upscaled_4x = (2048, 2048)  # 4x resolution
pixels_added = (2048 * 2048) - (512 * 512)  # 3,932,160 new pixels!
# AI must intelligently generate all these pixels
\`\`\`

## Real-ESRGAN: Industry Standard

### What is Real-ESRGAN?

Real-ESRGAN (Enhanced Super-Resolution GAN) is the go-to tool for AI upscaling:
- **Fast**: Processes images quickly
- **Quality**: Excellent results
- **Versatile**: Works on photos, art, anime
- **Open source**: Free to use

### Implementation

\`\`\`python
from PIL import Image
import numpy as np
import torch
from typing import Optional

class RealESRGANUpscaler:
    """
    Upscale images using Real-ESRGAN.
    """
    
    def __init__(
        self,
        model_name: str = "RealESRGAN_x4plus",
        device: str = "cuda"
    ):
        """
        Initialize Real-ESRGAN.
        
        Models:
        - RealESRGAN_x4plus: General photos (4x)
        - RealESRGAN_x4plus_anime_6B: Anime/art (4x)
        - RealESRGAN_x2plus: Faster 2x upscale
        """
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            raise ImportError(
                "Install Real-ESRGAN: pip install realesrgan basicsr"
            )
        
        self.device = device
        self.model_name = model_name
        
        # Model architecture
        if 'anime' in model_name:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=6,
                num_grow_ch=32, scale=4
            )
        else:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23,
                num_grow_ch=32, scale=4
            )
        
        # Load upscaler
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=f"{model_name}.pth",
            model=model,
            tile=400,  # Tile size for large images
            tile_pad=10,
            pre_pad=0,
            half=True if device == 'cuda' else False,
            device=device
        )
    
    def upscale(
        self,
        image: Image.Image,
        outscale: float = 4.0,
        face_enhance: bool = False
    ) -> Image.Image:
        """
        Upscale image.
        
        Args:
            image: Input image
            outscale: Scale factor (2, 4, etc.)
            face_enhance: Use additional face enhancement
        
        Returns:
            Upscaled image
        """
        import cv2
        
        # Convert to numpy
        img_np = np.array (image)
        img_np = cv2.cvtColor (img_np, cv2.COLOR_RGB2BGR)
        
        # Upscale
        output, _ = self.upsampler.enhance(
            img_np,
            outscale=outscale,
            face_enhance=face_enhance
        )
        
        # Convert back
        output = cv2.cvtColor (output, cv2.COLOR_BGR2RGB)
        return Image.fromarray (output)
    
    def upscale_batch(
        self,
        images: list[Image.Image],
        **kwargs
    ) -> list[Image.Image]:
        """Upscale multiple images."""
        return [self.upscale (img, **kwargs) for img in images]
    
    def upscale_with_fallback(
        self,
        image: Image.Image,
        target_size: tuple[int, int]
    ) -> Image.Image:
        """
        Upscale to target size, using tiling for large images.
        """
        current_w, current_h = image.size
        target_w, target_h = target_size
        
        # Calculate required scale
        scale_w = target_w / current_w
        scale_h = target_h / current_h
        scale = max (scale_w, scale_h)
        
        # Upscale
        if scale <= 4:
            upscaled = self.upscale (image, outscale=scale)
        else:
            # Multiple passes for >4x
            temp = image
            while scale > 1:
                step_scale = min (scale, 4)
                temp = self.upscale (temp, outscale=step_scale)
                scale /= step_scale
            upscaled = temp
        
        # Final resize to exact target
        return upscaled.resize (target_size, Image.LANCZOS)

# Usage
upscaler = RealESRGANUpscaler (model_name="RealESRGAN_x4plus")

# Upscale small image
small_image = Image.open("small_photo.jpg")  # e.g., 512x512
print(f"Original size: {small_image.size}")

upscaled = upscaler.upscale (small_image, outscale=4.0)
print(f"Upscaled size: {upscaled.size}")  # 2048x2048

upscaled.save("upscaled_4x.png")

# Upscale with face enhancement
portrait = Image.open("portrait.jpg")
enhanced = upscaler.upscale(
    portrait,
    outscale=4.0,
    face_enhance=True
)
\`\`\`

## Stable Diffusion Upscaling

### SD Upscale Pipeline

\`\`\`python
from diffusers import StableDiffusionUpscalePipeline
import torch

class SDUpscaler:
    """
    Upscale using Stable Diffusion.
    Adds creative details during upscaling.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # Load SD upscale model
        self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to (device)
        self.pipe.enable_attention_slicing()
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    
    def upscale(
        self,
        image: Image.Image,
        prompt: str = "",
        negative_prompt: str = "blurry, low quality",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        noise_level: int = 20
    ) -> Image.Image:
        """
        Upscale 4x with Stable Diffusion.
        
        Args:
            image: Input image (max 512x512 recommended)
            prompt: Optional guidance for upscaling
            negative_prompt: What to avoid
            num_inference_steps: Quality
            guidance_scale: Prompt adherence
            noise_level: Amount of noise added (0-100)
                        Higher = more creative, Lower = more faithful
        
        Returns:
            4x upscaled image
        """
        upscaled = self.pipe(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_level=noise_level
        ).images[0]
        
        return upscaled
    
    def controlled_upscale(
        self,
        image: Image.Image,
        style_description: str
    ) -> Image.Image:
        """
        Upscale with style control.
        """
        prompt = f"{style_description}, highly detailed, sharp, high resolution"
        
        return self.upscale(
            image=image,
            prompt=prompt,
            noise_level=15,  # Some creativity
            num_inference_steps=50,
            guidance_scale=8.0
        )

# Usage
sd_upscaler = SDUpscaler()

# Basic upscale
small = Image.open("image_512.jpg")
upscaled = sd_upscaler.upscale (small)

# Upscale with style
art = Image.open("artwork_512.jpg")
enhanced = sd_upscaler.controlled_upscale(
    art,
    style_description="oil painting, detailed brushstrokes, artistic"
)
\`\`\`

## Comparison: Different Methods

\`\`\`python
class UpscalingComparison:
    """
    Compare different upscaling methods.
    """
    
    def __init__(self):
        self.realesrgan = RealESRGANUpscaler()
        self.sd_upscaler = SDUpscaler()
    
    def compare_methods(
        self,
        image: Image.Image,
        methods: list[str] = ["lanczos", "realesrgan", "sd"]
    ) -> dict[str, Image.Image]:
        """
        Compare different upscaling methods.
        
        Methods:
        - lanczos: Traditional high-quality interpolation
        - realesrgan: Fast AI upscaling
        - sd: Stable Diffusion upscaling (creative)
        """
        results = {}
        original_size = image.size
        target_size = (original_size[0] * 4, original_size[1] * 4)
        
        if "lanczos" in methods:
            print("Testing Lanczos...")
            results["lanczos"] = image.resize (target_size, Image.LANCZOS)
        
        if "realesrgan" in methods:
            print("Testing Real-ESRGAN...")
            results["realesrgan"] = self.realesrgan.upscale (image, outscale=4)
        
        if "sd" in methods:
            print("Testing SD Upscale...")
            results["sd"] = self.sd_upscaler.upscale (image)
        
        return results
    
    def create_comparison_grid(
        self,
        results: dict[str, Image.Image],
        output_path: str = "upscale_comparison.png"
    ):
        """Create side-by-side comparison."""
        from PIL import ImageDraw, ImageFont
        
        images = list (results.values())
        labels = list (results.keys())
        
        # Create grid
        w, h = images[0].size
        grid_w = w * len (images)
        grid_h = h + 50  # Extra space for labels
        
        grid = Image.new('RGB', (grid_w, grid_h), 'white')
        draw = ImageDraw.Draw (grid)
        
        for i, (label, img) in enumerate (results.items()):
            x = i * w
            grid.paste (img, (x, 0))
            
            # Add label
            draw.text(
                (x + 10, h + 10),
                label.upper(),
                fill='black'
            )
        
        grid.save (output_path)
        print(f"Comparison saved to {output_path}")

# Usage
comparator = UpscalingComparison()

test_image = Image.open("test_512.jpg")

# Compare all methods
results = comparator.compare_methods (test_image)

# Create comparison
comparator.create_comparison_grid (results)

# Quality assessment
for method, img in results.items():
    file_size = len (img.tobytes())
    print(f"{method}: {img.size}, ~{file_size / 1024 / 1024:.2f} MB")
\`\`\`

## Specialized Upscaling

### Anime/Art Upscaling

\`\`\`python
class AnimeUpscaler:
    """
    Specialized upscaling for anime and illustrations.
    """
    
    def __init__(self):
        self.upscaler = RealESRGANUpscaler(
            model_name="RealESRGAN_x4plus_anime_6B"
        )
    
    def upscale_anime(
        self,
        image: Image.Image,
        preserve_lines: bool = True
    ) -> Image.Image:
        """
        Upscale anime/manga artwork.
        """
        result = self.upscaler.upscale (image, outscale=4)
        
        if preserve_lines:
            # Post-process to sharpen lines
            result = self._sharpen_lines (result)
        
        return result
    
    def _sharpen_lines (self, image: Image.Image) -> Image.Image:
        """Sharpen line art."""
        from PIL import ImageFilter, ImageEnhance
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness (image)
        sharpened = enhancer.enhance(1.5)
        
        # Slight unsharp mask
        unsharp = sharpened.filter(ImageFilter.UnsharpMask (radius=1, percent=150))
        
        return unsharp

# Usage
anime_upscaler = AnimeUpscaler()

anime_image = Image.open("anime_art.jpg")
upscaled = anime_upscaler.upscale_anime (anime_image, preserve_lines=True)
\`\`\`

### Progressive Upscaling

\`\`\`python
class ProgressiveUpscaler:
    """
    Upscale in multiple steps for extreme scaling.
    """
    
    def __init__(self):
        self.upscaler = RealESRGANUpscaler()
    
    def progressive_upscale(
        self,
        image: Image.Image,
        target_scale: int,
        step_size: int = 2
    ) -> Image.Image:
        """
        Upscale progressively to maintain quality.
        
        Example: 8x upscale = 2x → 2x → 2x
        Better than direct 8x.
        """
        current = image
        remaining_scale = target_scale
        
        steps = []
        while remaining_scale > 1:
            step = min (step_size, remaining_scale)
            steps.append (step)
            remaining_scale /= step
        
        print(f"Upscaling strategy: {' → '.join([f'{s}x' for s in steps])}")
        
        for i, step in enumerate (steps):
            print(f"Step {i+1}/{len (steps)}: {step}x upscale")
            current = self.upscaler.upscale (current, outscale=step)
        
        return current
    
    def upscale_to_size(
        self,
        image: Image.Image,
        target_width: int,
        target_height: int
    ) -> Image.Image:
        """
        Upscale to specific dimensions.
        """
        current_w, current_h = image.size
        
        # Calculate required scale
        scale_w = target_width / current_w
        scale_h = target_height / current_h
        scale = max (scale_w, scale_h)
        
        # Progressive upscale
        upscaled = self.progressive_upscale(
            image,
            target_scale=int (np.ceil (scale))
        )
        
        # Crop to exact size
        return upscaled.resize((target_width, target_height), Image.LANCZOS)

# Usage
progressive = ProgressiveUpscaler()

# Extreme upscaling: 256x256 → 4096x4096 (16x)
tiny = Image.open("tiny_256.jpg")
huge = progressive.progressive_upscale (tiny, target_scale=16)

print(f"Upscaled from {tiny.size} to {huge.size}")
\`\`\`

## Enhancement After Upscaling

\`\`\`python
class PostUpscaleEnhancer:
    """
    Enhance images after upscaling.
    """
    
    @staticmethod
    def enhance_details (image: Image.Image) -> Image.Image:
        """Enhance fine details."""
        from PIL import ImageFilter, ImageEnhance
        
        # Sharpen
        sharpness = ImageEnhance.Sharpness (image)
        sharpened = sharpness.enhance(1.3)
        
        # Contrast
        contrast = ImageEnhance.Contrast (sharpened)
        enhanced = contrast.enhance(1.1)
        
        return enhanced
    
    @staticmethod
    def reduce_noise (image: Image.Image) -> Image.Image:
        """Reduce upscaling artifacts."""
        import cv2
        import numpy as np
        
        img_np = np.array (image)
        
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            img_np,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        return Image.fromarray (denoised)
    
    @staticmethod
    def color_correction (image: Image.Image) -> Image.Image:
        """Correct colors after upscaling."""
        from PIL import ImageEnhance
        
        # Adjust saturation
        color = ImageEnhance.Color (image)
        saturated = color.enhance(1.1)
        
        # Adjust brightness
        brightness = ImageEnhance.Brightness (saturated)
        corrected = brightness.enhance(1.05)
        
        return corrected
    
    def full_enhancement (self, image: Image.Image) -> Image.Image:
        """Apply all enhancements."""
        enhanced = self.enhance_details (image)
        enhanced = self.reduce_noise (enhanced)
        enhanced = self.color_correction (enhanced)
        
        return enhanced

# Usage
enhancer = PostUpscaleEnhancer()

upscaled = Image.open("upscaled_rough.png")
final = enhancer.full_enhancement (upscaled)
final.save("upscaled_enhanced.png")
\`\`\`

## Production Pipeline

\`\`\`python
class ProductionUpscalePipeline:
    """
    Complete production upscaling pipeline.
    """
    
    def __init__(self):
        self.upscaler = RealESRGANUpscaler()
        self.enhancer = PostUpscaleEnhancer()
    
    def upscale_for_print(
        self,
        image: Image.Image,
        target_dpi: int = 300,
        print_size_inches: tuple[float, float] = (8, 10)
    ) -> Image.Image:
        """
        Upscale image for printing.
        
        Args:
            image: Input image
            target_dpi: Dots per inch (300 for print, 72 for screen)
            print_size_inches: (width, height) in inches
        """
        # Calculate required pixel dimensions
        target_w = int (print_size_inches[0] * target_dpi)
        target_h = int (print_size_inches[1] * target_dpi)
        
        print(f"Target size for {print_size_inches}" at {target_dpi} DPI:")
        print(f"{target_w}x{target_h} pixels")
        
        # Calculate scale factor
        current_w, current_h = image.size
        scale = max (target_w / current_w, target_h / current_h)
        
        print(f"Required scale: {scale:.2f}x")
        
        # Upscale
        if scale <= 4:
            upscaled = self.upscaler.upscale (image, outscale=scale)
        else:
            # Progressive for large scales
            progressive = ProgressiveUpscaler()
            upscaled = progressive.progressive_upscale (image, int (np.ceil (scale)))
        
        # Resize to exact dimensions
        final = upscaled.resize((target_w, target_h), Image.LANCZOS)
        
        # Enhance
        final = self.enhancer.full_enhancement (final)
        
        return final
    
    def upscale_batch(
        self,
        input_dir: str,
        output_dir: str,
        scale: float = 4.0,
        enhance: bool = True
    ):
        """
        Batch upscale directory of images.
        """
        from pathlib import Path
        
        input_path = Path (input_dir)
        output_path = Path (output_dir)
        output_path.mkdir (exist_ok=True)
        
        images = list (input_path.glob("*.jpg")) + list (input_path.glob("*.png"))
        total = len (images)
        
        print(f"Upscaling {total} images...")
        
        for i, img_file in enumerate (images, 1):
            print(f"[{i}/{total}] {img_file.name}")
            
            # Load
            img = Image.open (img_file)
            
            # Upscale
            upscaled = self.upscaler.upscale (img, outscale=scale)
            
            # Enhance
            if enhance:
                upscaled = self.enhancer.full_enhancement (upscaled)
            
            # Save
            output_file = output_path / img_file.name
            upscaled.save (output_file, quality=95)
        
        print(f"Done! Saved to {output_dir}")

# Usage
pipeline = ProductionUpscalePipeline()

# Upscale for 8x10" print at 300 DPI
photo = Image.open("photo.jpg")
print_ready = pipeline.upscale_for_print(
    photo,
    target_dpi=300,
    print_size_inches=(8, 10)
)
print_ready.save("photo_print_ready.tiff", dpi=(300, 300))

# Batch upscale
pipeline.upscale_batch(
    input_dir="./low_res",
    output_dir="./high_res",
    scale=4.0,
    enhance=True
)
\`\`\`

## Key Takeaways

- **AI upscaling** generates realistic details vs. traditional blurring
- **Real-ESRGAN**: Fast, excellent quality, industry standard
- **SD Upscale**: Creative upscaling with prompt control
- **Anime models**: Specialized for illustrations and art
- **Progressive upscaling**: Multiple steps for extreme scales (>4x)
- **Post-enhancement**: Sharpening, noise reduction, color correction
- **Print preparation**: Calculate DPI requirements accurately
- **Batch processing**: Efficient for multiple images
- **Trade-offs**: Speed vs quality, faithful vs creative
- **Use cases**: Print preparation, restoration, detail enhancement
`,
};
