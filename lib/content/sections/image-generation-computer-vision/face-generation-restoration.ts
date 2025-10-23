/**
 * Face Generation & Restoration Section
 * Module 8: Image Generation & Computer Vision
 */

export const facegenerationrestorationSection = {
  id: 'face-generation-restoration',
  title: 'Face Generation & Restoration',
  content: `# Face Generation & Restoration

Master face-specific generation and restoration techniques for high-quality portrait results.

## Overview: The Challenge of Faces

Faces are uniquely challenging for AI:
- **High expectations**: We're extremely sensitive to facial errors
- **Complex features**: Eyes, teeth, skin must be perfect
- **Uncanny valley**: Small errors = deeply unsettling
- **Ethical concerns**: Deepfakes, identity, consent

### Why Face-Specific Tools Matter

\`\`\`python
general_vs_specialized = {
    "general_models": {
        "strengths": "Good at everything",
        "face_quality": "Acceptable but not perfect",
        "common_issues": [
            "Asymmetric faces",
            "Weird eyes",
            "Bad teeth",
            "Unnatural skin",
            "Multiple faces blended"
        ]
    },
    
    "face_specialized": {
        "strengths": "Exceptional faces",
        "face_quality": "Near-perfect",
        "fixes": [
            "Symmetry",
            "Realistic eyes",
            "Natural teeth",
            "Proper skin texture",
            "Clean features"
        ]
    }
}
\`\`\`

## Face Generation with Stable Diffusion

### Optimized Face Prompts

\`\`\`python
class FacePromptBuilder:
    """
    Build prompts optimized for face generation.
    """
    
    @staticmethod
    def professional_headshot(
        subject: str = "person",
        age: str = "adult",
        gender: Optional[str] = None,
        ethnicity: Optional[str] = None,
        expression: str = "neutral, slight smile",
        additional: List[str] = []
    ) -> dict:
        """
        Build professional headshot prompt.
        """
        # Build subject description
        descriptors = [age]
        if gender:
            descriptors.append(gender)
        if ethnicity:
            descriptors.append(ethnicity)
        descriptors.append(subject)
        
        subject_desc = " ".join(descriptors)
        
        prompt_parts = [
            f"professional headshot photograph of {subject_desc}",
            expression,
            "looking at camera",
            "professional photography",
            "studio lighting",
            "sharp focus on face",
            "shallow depth of field",
            "clean background",
            "high resolution",
            "perfect symmetry",
            "clear skin",
            "natural look",
        ]
        
        prompt_parts.extend(additional)
        
        negative_parts = [
            "blurry", "low quality", "distorted",
            "bad anatomy", "bad face", "asymmetric face",
            "bad eyes", "crossed eyes", "looking away",
            "bad teeth", "open mouth",
            "multiple heads", "deformed",
            "extra fingers", "bad hands"
        ]
        
        return {
            "prompt": ", ".join(prompt_parts),
            "negative_prompt": ", ".join(negative_parts)
        }
    
    @staticmethod
    def artistic_portrait(
        subject: str,
        art_style: str = "oil painting",
        mood: str = "contemplative",
        lighting: str = "rembrandt lighting"
    ) -> dict:
        """
        Artistic portrait prompt.
        """
        prompt = f"""
        {art_style} portrait of {subject},
        {mood} expression,
        {lighting},
        masterpiece,
        highly detailed face,
        perfect features,
        professional art,
        trending on artstation
        """.strip()
        
        negative = """
        bad anatomy, deformed face, asymmetric,
        distorted features, ugly, low quality
        """.strip()
        
        return {"prompt": prompt, "negative_prompt": negative}
    
    @staticmethod
    def cinematic_portrait(
        subject: str,
        scene: str = "dramatic lighting",
        camera: str = "shot on Arri Alexa"
    ) -> dict:
        """
        Cinematic movie-style portrait.
        """
        prompt = f"""
        cinematic portrait of {subject},
        {scene},
        film grain,
        {camera},
        85mm f/1.4,
        professional color grading,
        dramatic mood,
        perfect face,
        movie still,
        high production value
        """.strip()
        
        negative = "amateur, bad face, distorted, low quality"
        
        return {"prompt": prompt, "negative_prompt": negative}

# Usage
builder = FacePromptBuilder()

# Professional headshot
headshot_prompts = builder.professional_headshot(
    subject="businesswoman",
    age="middle-aged",
    expression="confident, professional smile",
    additional=["wearing business suit", "elegant"]
)

# Artistic portrait
art_prompts = builder.artistic_portrait(
    subject="elderly man with beard",
    art_style="renaissance oil painting",
    mood="wise and thoughtful"
)

# Generate with optimized prompts
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")

headshot = pipe(**headshot_prompts, num_inference_steps=50).images[0]
headshot.save("professional_headshot.png")
\`\`\`

## Face Restoration

### GFPGAN: Face Enhancement

\`\`\`python
class FaceRestorer:
    """
    Restore and enhance faces using GFPGAN.
    """
    
    def __init__(self, version: str = "1.4"):
        """
        Initialize GFPGAN.
        
        Args:
            version: Model version (1.3 or 1.4)
        """
        try:
            import gfpgan
            from gfpgan import GFPGANer
        except ImportError:
            raise ImportError("Install gfpgan: pip install gfpgan")
        
        model_path = f"GFPGANv{version}.pth"
        
        self.restorer = GFPGANer(
            model_path=model_path,
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None  # Can add background upsampler
        )
    
    def restore_face(
        self,
        image: Image.Image,
        has_aligned: bool = False,
        only_center_face: bool = False,
        paste_back: bool = True,
        weight: float = 0.5
    ) -> Image.Image:
        """
        Restore faces in image.
        
        Args:
            image: Input image
            has_aligned: If faces are already aligned
            only_center_face: Only restore center face
            paste_back: Paste restored face back to original
            weight: Blend weight (0=original, 1=fully restored)
        
        Returns:
            Restored image
        """
        import cv2
        import numpy as np
        
        # Convert PIL to numpy
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Restore
        _, _, output = self.restorer.enhance(
            img_np,
            has_aligned=has_aligned,
            only_center_face=only_center_face,
            paste_back=paste_back,
            weight=weight
        )
        
        # Convert back to PIL
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return Image.fromarray(output)
    
    def restore_multiple(
        self,
        images: List[Image.Image],
        **kwargs
    ) -> List[Image.Image]:
        """Restore faces in multiple images."""
        return [self.restore_face(img, **kwargs) for img in images]

# Usage
restorer = FaceRestorer(version="1.4")

# Restore low-quality face photo
old_photo = Image.open("blurry_face.jpg")
restored = restorer.restore_face(
    image=old_photo,
    weight=0.7  # 70% restored, 30% original
)

restored.save("face_restored.png")

# Restore AI-generated faces
ai_faces = [Image.open(f"gen_{i}.png") for i in range(4)]
enhanced = restorer.restore_multiple(ai_faces, weight=0.5)
\`\`\`

### CodeFormer: Advanced Face Restoration

\`\`\`python
class CodeFormerRestorer:
    """
    CodeFormer for high-quality face restoration.
    Often better than GFPGAN for severe damage.
    """
    
    def __init__(self):
        try:
            from basicsr.archs.codeformer_arch import CodeFormer
        except ImportError:
            raise ImportError("Install CodeFormer dependencies")
        
        self.model = CodeFormer(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=['32', '64', '128', '256']
        )
        
        # Load weights
        checkpoint = torch.load('CodeFormer.pth')
        self.model.load_state_dict(checkpoint['params_ema'])
        self.model.eval()
        self.model = self.model.to('cuda')
    
    def restore(
        self,
        image: Image.Image,
        fidelity_weight: float = 0.5
    ) -> Image.Image:
        """
        Restore face with CodeFormer.
        
        Args:
            image: Input image
            fidelity_weight: Balance between quality and fidelity
                            (0=more quality, 1=more faithful to input)
        """
        import cv2
        import numpy as np
        
        # Preprocess
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_np).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to('cuda')
        
        # Restore
        with torch.no_grad():
            output = self.model(
                img_tensor,
                w=fidelity_weight,
                adain=True
            )[0]
        
        # Post-process
        output = output.squeeze().permute(1, 2, 0).cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(output)

# Usage
codeformer = CodeFormerRestorer()

damaged_face = Image.open("damaged_face.jpg")

# Try different fidelity weights
restored_quality = codeformer.restore(damaged_face, fidelity_weight=0.2)
restored_faithful = codeformer.restore(damaged_face, fidelity_weight=0.8)

restored_quality.save("restored_quality_focused.png")
restored_faithful.save("restored_faithful.png")
\`\`\`

## Face Detection and Alignment

\`\`\`python
class FaceDetector:
    """
    Detect and align faces for processing.
    """
    
    def __init__(self):
        try:
            import dlib
        except ImportError:
            raise ImportError("Install dlib: pip install dlib")
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    def detect_faces(
        self,
        image: Image.Image
    ) -> List[dict]:
        """
        Detect faces and landmarks.
        
        Returns:
            List of face dictionaries with bbox and landmarks
        """
        import numpy as np
        import cv2
        
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = self.detector(gray)
        
        results = []
        for face in faces:
            # Get landmarks
            landmarks = self.predictor(gray, face)
            
            # Extract coordinates
            points = []
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                points.append((x, y))
            
            results.append({
                'bbox': (face.left(), face.top(), face.right(), face.bottom()),
                'landmarks': points
            })
        
        return results
    
    def align_face(
        self,
        image: Image.Image,
        landmarks: List[Tuple[int, int]],
        output_size: Tuple[int, int] = (256, 256)
    ) -> Image.Image:
        """
        Align face based on eye positions.
        """
        import numpy as np
        import cv2
        
        # Get eye centers (landmarks 36-41: left eye, 42-47: right eye)
        left_eye = np.mean([landmarks[i] for i in range(36, 42)], axis=0)
        right_eye = np.mean([landmarks[i] for i in range(42, 48)], axis=0)
        
        # Calculate angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate center
        center = ((left_eye[0] + right_eye[0]) / 2,
                 (left_eye[1] + right_eye[1]) / 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate
        img_np = np.array(image)
        rotated = cv2.warpAffine(
            img_np, M, (img_np.shape[1], img_np.shape[0]),
            flags=cv2.INTER_CUBIC
        )
        
        # Crop and resize
        x, y, w, h = self._get_face_box(landmarks)
        cropped = rotated[y:y+h, x:x+w]
        aligned = cv2.resize(cropped, output_size)
        
        return Image.fromarray(aligned)
    
    def _get_face_box(self, landmarks):
        """Calculate bounding box from landmarks."""
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        
        # Add padding
        pad = int(0.2 * min(w, h))
        return x - pad, y - pad, w + 2*pad, h + 2*pad

# Usage
detector = FaceDetector()

photo = Image.open("group_photo.jpg")

# Detect all faces
faces = detector.detect_faces(photo)
print(f"Found {len(faces)} faces")

# Align first face
if faces:
    aligned = detector.align_face(
        image=photo,
        landmarks=faces[0]['landmarks']
    )
    aligned.save("aligned_face.png")
\`\`\`

## Post-Processing and Enhancement

\`\`\`python
class FaceEnhancer:
    """
    Additional face enhancement techniques.
    """
    
    @staticmethod
    def enhance_eyes(image: Image.Image, factor: float = 1.3) -> Image.Image:
        """
        Enhance eye region for more vivid eyes.
        """
        from PIL import ImageEnhance
        import cv2
        import numpy as np
        
        # Detect eyes (simplified - use face detector in production)
        img_np = np.array(image)
        
        # Create mask for eye region (simplified)
        h, w = img_np.shape[:2]
        eye_region = (
            int(w * 0.3), int(h * 0.3),
            int(w * 0.7), int(h * 0.5)
        )
        
        # Enhance eye region
        eye_img = image.crop(eye_region)
        enhancer = ImageEnhance.Sharpness(eye_img)
        eye_enhanced = enhancer.enhance(factor)
        
        result = image.copy()
        result.paste(eye_enhanced, eye_region[:2])
        
        return result
    
    @staticmethod
    def smooth_skin(
        image: Image.Image,
        strength: float = 0.5
    ) -> Image.Image:
        """
        Apply skin smoothing.
        """
        import cv2
        import numpy as np
        
        img_np = np.array(image)
        
        # Bilateral filter for skin smoothing
        smoothed = cv2.bilateralFilter(
            img_np,
            d=9,
            sigmaColor=75,
            sigmaSpace=75
        )
        
        # Blend with original
        result = cv2.addWeighted(
            img_np,
            1 - strength,
            smoothed,
            strength,
            0
        )
        
        return Image.fromarray(result)
    
    @staticmethod
    def enhance_contrast(
        image: Image.Image,
        clip_limit: float = 2.0
    ) -> Image.Image:
        """
        Enhance face contrast with CLAHE.
        """
        import cv2
        import numpy as np
        
        img_np = np.array(image)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(enhanced)

# Usage
enhancer = FaceEnhancer()

face_photo = Image.open("portrait.jpg")

# Apply enhancements
vivid_eyes = enhancer.enhance_eyes(face_photo, factor=1.4)
smooth_skin = enhancer.smooth_skin(face_photo, strength=0.6)
enhanced = enhancer.enhance_contrast(smooth_skin, clip_limit=2.5)

enhanced.save("enhanced_portrait.png")
\`\`\`

## Production Workflow

\`\`\`python
class FaceGenerationPipeline:
    """
    Complete face generation and restoration pipeline.
    """
    
    def __init__(self):
        # Load models
        self.generator = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16
        ).to("cuda")
        
        self.face_restorer = FaceRestorer()
        self.face_enhancer = FaceEnhancer()
    
    def generate_professional_headshot(
        self,
        description: str,
        num_candidates: int = 4,
        restore: bool = True,
        enhance: bool = True
    ) -> List[Image.Image]:
        """
        Generate professional headshots with restoration.
        
        Workflow:
        1. Generate multiple candidates
        2. Restore faces
        3. Enhance
        4. Return best results
        """
        # Build prompt
        prompt_builder = FacePromptBuilder()
        prompts = prompt_builder.professional_headshot(description)
        
        # Generate candidates
        print(f"Generating {num_candidates} candidates...")
        candidates = self.generator(
            **prompts,
            num_images_per_prompt=num_candidates,
            num_inference_steps=50
        ).images
        
        results = []
        for i, img in enumerate(candidates):
            print(f"Processing candidate {i+1}/{num_candidates}")
            
            # Restore face
            if restore:
                img = self.face_restorer.restore_face(img, weight=0.5)
            
            # Enhance
            if enhance:
                img = self.face_enhancer.smooth_skin(img, strength=0.3)
                img = self.face_enhancer.enhance_contrast(img, clip_limit=2.0)
            
            results.append(img)
        
        return results
    
    def restore_old_photo(
        self,
        photo: Image.Image,
        aggressive: bool = True
    ) -> Image.Image:
        """
        Restore old or damaged photo.
        """
        print("Restoring photo...")
        
        # Face restoration
        weight = 0.8 if aggressive else 0.5
        restored = self.face_restorer.restore_face(photo, weight=weight)
        
        # Enhancement
        enhanced = self.face_enhancer.enhance_contrast(restored, clip_limit=2.5)
        enhanced = self.face_enhancer.smooth_skin(enhanced, strength=0.2)
        
        return enhanced

# Usage
pipeline = FaceGenerationPipeline()

# Generate headshots
headshots = pipeline.generate_professional_headshot(
    description="confident businesswoman in her 40s",
    num_candidates=4,
    restore=True,
    enhance=True
)

for i, img in enumerate(headshots):
    img.save(f"headshot_candidate_{i}.png")

# Restore old photo
old_photo = Image.open("old_family_photo.jpg")
restored = pipeline.restore_old_photo(old_photo, aggressive=True)
restored.save("restored_family_photo.png")
\`\`\`

## Ethical Considerations

\`\`\`python
ethical_guidelines = {
    "consent": [
        "Never generate faces of real people without consent",
        "Don't generate minors",
        "Respect privacy and identity",
        "Clear labeling of AI-generated images"
    ],
    
    "misuse_prevention": [
        "Watermark AI-generated faces",
        "Implement detection systems",
        "Don't facilitate deepfakes",
        "Document intended use cases"
    ],
    
    "representation": [
        "Generate diverse faces (age, ethnicity, gender)",
        "Avoid stereotypes",
        "Ensure inclusive training data",
        "Test for bias"
    ],
    
    "transparency": [
        "Disclose AI generation",
        "Explain limitations",
        "Provide attribution",
        "Enable verification"
    ]
}

class EthicalFaceGenerator:
    """
    Face generator with ethical safeguards.
    """
    
    def __init__(self):
        self.generation_log = []
    
    def generate_with_watermark(
        self,
        prompt: str,
        **kwargs
    ) -> Image.Image:
        """
        Generate face with AI watermark.
        """
        # Generate
        image = self.generate(prompt, **kwargs)
        
        # Add watermark
        from PIL import ImageDraw, ImageFont
        
        draw = ImageDraw.Draw(image)
        watermark_text = "AI Generated"
        
        # Add to corner
        draw.text((10, 10), watermark_text, fill='white')
        
        # Log generation
        self.generation_log.append({
            'timestamp': datetime.now(),
            'prompt': prompt,
            'watermarked': True
        })
        
        return image
\`\`\`

## Key Takeaways

- **Faces are challenging**: Require special attention and tools
- **GFPGAN**: Standard tool for face restoration
- **CodeFormer**: Often better for severe damage
- **Face alignment**: Critical for consistent results
- **Post-processing**: Eyes, skin, contrast enhancement
- **Prompt optimization**: Specific prompts for better faces
- **Multiple candidates**: Generate several, pick best
- **Workflow**: Generate → Restore → Enhance
- **Ethical concerns**: Consent, misuse prevention, transparency
- **Production use**: Combine multiple tools for best results
`,
};
