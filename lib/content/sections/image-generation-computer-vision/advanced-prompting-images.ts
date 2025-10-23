/**
 * Advanced Prompting for Images Section
 * Module 8: Image Generation & Computer Vision
 */

export const advancedpromptingimagesSection = {
  id: 'advanced-prompting-images',
  title: 'Advanced Prompting for Images',
  content: `# Advanced Prompting for Images

Master the art of crafting effective prompts for high-quality, consistent image generation.

## Overview: Prompting is an Art AND a Science

Unlike text generation, image prompts require understanding:
- **Visual concepts**: How to describe what you see
- **Technical photography terms**: Lighting, composition, angles
- **Art styles and movements**: References that guide aesthetics
- **Weight and emphasis**: Making certain elements stronger

### Why Advanced Prompting Matters

\`\`\`python
basic_vs_advanced = {
    "basic_prompt": {
        "text": "a cat",
        "result": "Generic cat photo",
        "problems": [
            "Unpredictable style",
            "Random composition",
            "Inconsistent quality",
            "Wasted generations"
        ]
    },
    
    "advanced_prompt": {
        "text": """
        a professional studio photograph of a gray tabby cat,
        soft diffused lighting, shallow depth of field, sitting pose,
        clean white background, shot on Canon EOS R5, 85mm f/1.8,
        sharp focus on eyes, high resolution, award-winning pet photography
        """,
        "result": "Exactly what you wanted",
        "benefits": [
            "Predictable style",
            "Controlled composition",
            "Consistent quality",
            "First-try success"
        ]
    }
}
\`\`\`

## Prompt Structure

### The Anatomy of a Great Prompt

\`\`\`python
class PromptStructure:
    """
    Effective prompt structure for images.
    """
    
    @staticmethod
    def build_prompt(
        subject: str,
        action: str = "",
        context: str = "",
        style: str = "",
        technical: List[str] = [],
        quality: List[str] = []
    ) -> str:
        """
        Build structured prompt.
        
        Order matters:
        1. Subject (most important)
        2. Action/pose
        3. Context/environment
        4. Style/aesthetic
        5. Technical details
        6. Quality modifiers
        """
        parts = []
        
        # 1. Subject (what)
        parts.append(subject)
        
        # 2. Action (doing what)
        if action:
            parts.append(action)
        
        # 3. Context (where, when)
        if context:
            parts.append(context)
        
        # 4. Style
        if style:
            parts.append(style)
        
        # 5. Technical details
        if technical:
            parts.extend(technical)
        
        # 6. Quality modifiers
        if quality:
            parts.extend(quality)
        
        return ", ".join(parts)
    
    @staticmethod
    def build_photography_prompt(
        subject: str,
        shot_type: str = "medium shot",
        lighting: str = "natural lighting",
        camera: str = "",
        lens: str = "",
        style: str = "professional photography"
    ) -> str:
        """Specialized for photography style."""
        parts = [
            f"{shot_type} of {subject}",
            lighting,
        ]
        
        if camera:
            parts.append(f"shot on {camera}")
        if lens:
            parts.append(f"{lens} lens")
        
        parts.extend([
            style,
            "high resolution",
            "sharp focus"
        ])
        
        return ", ".join(parts)
    
    @staticmethod
    def build_art_prompt(
        subject: str,
        art_style: str,
        artist_reference: str = "",
        medium: str = "",
        mood: str = ""
    ) -> str:
        """Specialized for artistic style."""
        parts = [
            f"{art_style} of {subject}"
        ]
        
        if artist_reference:
            parts.append(f"in the style of {artist_reference}")
        if medium:
            parts.append(f"{medium} medium")
        if mood:
            parts.append(f"{mood} mood")
        
        parts.extend([
            "highly detailed",
            "masterpiece",
            "trending on artstation"
        ])
        
        return ", ".join(parts)

# Examples
builder = PromptStructure()

# General structured prompt
general = builder.build_prompt(
    subject="a steaming cup of coffee",
    action="on a wooden table",
    context="cozy café interior, morning light through window",
    style="warm color grading",
    technical=["shallow depth of field", "bokeh background"],
    quality=["professional photography", "high resolution"]
)
# Result: "a steaming cup of coffee, on a wooden table, cozy café interior..."

# Photography prompt
photo = builder.build_photography_prompt(
    subject="a vintage car",
    shot_type="low angle shot",
    lighting="golden hour lighting",
    camera="Canon EOS R5",
    lens="24mm f/1.4",
    style="automotive photography"
)
# Result: "low angle shot of a vintage car, golden hour lighting..."

# Art prompt
art = builder.build_art_prompt(
    subject="a mystical forest",
    art_style="fantasy digital painting",
    artist_reference="Greg Rutkowski",
    medium="digital art",
    mood="ethereal and magical"
)
# Result: "fantasy digital painting of a mystical forest, in the style of..."
\`\`\`

## Subject Description

### Being Specific

\`\`\`python
specificity_examples = {
    "vague": {
        "prompt": "a dog",
        "issues": [
            "Unknown breed",
            "Unknown color",
            "Unknown size",
            "Unknown pose",
            "Generic result"
        ]
    },
    
    "better": {
        "prompt": "a golden retriever",
        "improvement": "Specified breed",
        "still_missing": ["Age", "Action", "Context"]
    },
    
    "good": {
        "prompt": "a young golden retriever puppy sitting on grass",
        "improvement": "Age, pose, and basic context",
        "still_missing": ["Lighting", "Quality", "Style"]
    },
    
    "excellent": {
        "prompt": """
        a young golden retriever puppy sitting on green grass,
        natural daylight, shallow depth of field, professional pet photography,
        sharp focus on face, bokeh background, joyful expression
        """,
        "why_excellent": [
            "Specific subject (breed, age)",
            "Clear pose and context",
            "Defined lighting",
            "Style specification",
            "Quality markers",
            "Emotional direction"
        ]
    }
}

def improve_subject_description(basic_subject: str) -> dict:
    """
    Suggest improvements for subject description.
    """
    improvements = {
        "person": {
            "add": ["age", "gender", "ethnicity", "clothing", "expression", "pose"],
            "example": "young woman in casual clothing, smiling, sitting pose"
        },
        "animal": {
            "add": ["species/breed", "age", "color", "action", "environment"],
            "example": "adult gray tabby cat, sitting upright, indoor setting"
        },
        "object": {
            "add": ["material", "color", "size", "condition", "placement"],
            "example": "red ceramic coffee mug, on wooden table, clean and modern"
        },
        "scene": {
            "add": ["location", "time of day", "weather", "atmosphere"],
            "example": "mountain landscape, sunrise, misty atmosphere, serene"
        }
    }
    
    return improvements
\`\`\`

## Negative Prompts

### What to Avoid

Negative prompts are crucial for quality:

\`\`\`python
class NegativePromptBuilder:
    """
    Build effective negative prompts.
    """
    
    # Common negative prompt elements
    QUALITY_NEGATIVES = [
        "blurry", "low quality", "low resolution",
        "pixelated", "jpeg artifacts", "compression artifacts",
        "grainy", "noisy", "out of focus"
    ]
    
    ANATOMICAL_NEGATIVES = [
        "bad anatomy", "deformed", "disfigured",
        "extra limbs", "missing limbs", "extra fingers",
        "bad hands", "bad proportions", "mutated"
    ]
    
    STYLE_NEGATIVES = [
        "cartoon", "anime", "drawing",  # If you want realistic
        "photorealistic",  # If you want artistic
        "oversaturated", "undersaturated",
        "overexposed", "underexposed"
    ]
    
    COMPOSITION_NEGATIVES = [
        "cropped", "cut off", "bad framing",
        "tilted", "distorted", "warped"
    ]
    
    @classmethod
    def build_for_photography(cls) -> str:
        """Negative prompt for realistic photography."""
        negatives = (
            cls.QUALITY_NEGATIVES +
            cls.ANATOMICAL_NEGATIVES +
            [
                "cartoon", "anime", "drawing", "painting",
                "illustration", "3d render", "cgi"
            ]
        )
        return ", ".join(negatives)
    
    @classmethod
    def build_for_portrait(cls) -> str:
        """Negative prompt for portraits."""
        negatives = (
            cls.QUALITY_NEGATIVES +
            cls.ANATOMICAL_NEGATIVES +
            [
                "bad face", "ugly face", "asymmetric face",
                "crossed eyes", "bad eyes", "closed eyes",
                "bad skin", "bad teeth", "multiple heads"
            ]
        )
        return ", ".join(negatives)
    
    @classmethod
    def build_for_art(cls) -> str:
        """Negative prompt for artistic work."""
        negatives = cls.QUALITY_NEGATIVES + [
            "photorealistic", "photo", "photograph",
            "ugly", "messy", "cluttered"
        ]
        return ", ".join(negatives)
    
    @classmethod
    def custom(cls, avoid: List[str]) -> str:
        """Build custom negative prompt."""
        base = cls.QUALITY_NEGATIVES.copy()
        base.extend(avoid)
        return ", ".join(base)

# Usage
neg_builder = NegativePromptBuilder()

# For realistic photos
photo_neg = neg_builder.build_for_photography()
# "blurry, low quality, ..., cartoon, anime, drawing, ..."

# For portraits
portrait_neg = neg_builder.build_for_portrait()
# "blurry, low quality, ..., bad face, ugly face, ..."

# Custom
custom_neg = neg_builder.custom([
    "watermark", "text", "signature",
    "cluttered background"
])
\`\`\`

## Weighting and Emphasis

### Stable Diffusion Weight Syntax

\`\`\`python
class PromptWeighting:
    """
    Apply weights to emphasize or de-emphasize elements.
    
    Syntax:
    - (keyword:1.5) = 1.5x emphasis
    - (keyword:0.8) = 0.8x emphasis (less important)
    - (keyword:1.0) = normal (same as no parentheses)
    - ((keyword)) = 1.1x emphasis (shorthand)
    - [keyword] = 0.9x emphasis (shorthand)
    """
    
    @staticmethod
    def emphasize(text: str, weight: float = 1.3) -> str:
        """Emphasize a term."""
        return f"({text}:{weight})"
    
    @staticmethod
    def de_emphasize(text: str, weight: float = 0.7) -> str:
        """De-emphasize a term."""
        return f"({text}:{weight})"
    
    @staticmethod
    def build_weighted_prompt(elements: dict) -> str:
        """
        Build prompt with different weights.
        
        Args:
            elements: {text: weight} dictionary
        """
        weighted = []
        
        for text, weight in elements.items():
            if weight != 1.0:
                weighted.append(f"({text}:{weight})")
            else:
                weighted.append(text)
        
        return ", ".join(weighted)

# Examples
weighter = PromptWeighting()

# Emphasize specific features
prompt_weighted = weighter.build_weighted_prompt({
    "a portrait of a woman": 1.0,
    "blue eyes": 1.4,  # Really want blue eyes
    "long red hair": 1.3,  # Strong emphasis on red hair
    "wearing a hat": 0.8,  # Hat is optional
    "professional photography": 1.0
})
# Result: "a portrait of a woman, (blue eyes:1.4), (long red hair:1.3), ..."

# Control color saturation
color_control = weighter.build_weighted_prompt({
    "a sunset landscape": 1.0,
    "vibrant colors": 1.5,  # Really want vivid colors
    "orange and purple sky": 1.3,
    "calm atmosphere": 1.0
})

# Balance conflicting elements
balanced = weighter.build_weighted_prompt({
    "a futuristic city": 1.0,
    "cyberpunk aesthetic": 1.2,
    "but not too dark": 1.0,  # Clarification
    "neon lights": 1.3,
    "rainy": 0.7,  # Just a bit of rain
    "highly detailed": 1.0
})
\`\`\`

## Style and Aesthetic Control

### Photography Styles

\`\`\`python
photography_styles = {
    "portrait": {
        "style_terms": [
            "professional portrait photography",
            "studio lighting",
            "shallow depth of field",
            "bokeh background",
            "sharp focus on eyes"
        ],
        "cameras": ["Canon EOS R5", "Sony A7R IV", "Nikon Z9"],
        "lenses": ["85mm f/1.4", "50mm f/1.2", "135mm f/1.8"],
        "lighting": ["soft lighting", "rembrandt lighting", "butterfly lighting"]
    },
    
    "landscape": {
        "style_terms": [
            "landscape photography",
            "golden hour lighting",
            "dramatic sky",
            "wide angle view",
            "deep depth of field"
        ],
        "cameras": ["Canon EOS R5", "Sony A7R V"],
        "lenses": ["16-35mm f/2.8", "24-70mm f/2.8"],
        "lighting": ["golden hour", "blue hour", "dramatic clouds"]
    },
    
    "product": {
        "style_terms": [
            "product photography",
            "clean white background",
            "studio lighting",
            "professional", 
            "commercial"
        ],
        "cameras": ["Phase One", "Canon EOS R5"],
        "lenses": ["100mm macro f/2.8"],
        "lighting": ["soft box lighting", "even lighting", "no harsh shadows"]
    },
    
    "street": {
        "style_terms": [
            "street photography",
            "candid",
            "natural lighting",
            "documentary style",
            "urban environment"
        ],
        "cameras": ["Leica M11", "Fujifilm X-T5"],
        "lenses": ["35mm f/1.4", "28mm f/2"],
        "lighting": ["natural light", "available light", "harsh shadows"]
    },
    
    "macro": {
        "style_terms": [
            "macro photography",
            "extreme close-up",
            "shallow depth of field",
            "detailed texture",
            "sharp focus"
        ],
        "cameras": ["Canon EOS R5", "Sony A7R IV"],
        "lenses": ["100mm macro f/2.8", "180mm macro f/3.5"],
        "lighting": ["diffused lighting", "ring flash", "soft lighting"]
    }
}

def build_photography_style_prompt(
    subject: str,
    style: str = "portrait"
) -> str:
    """
    Build prompt with photography style.
    """
    style_config = photography_styles.get(style, photography_styles["portrait"])
    
    prompt_parts = [
        f"{style_config['style_terms'][0]} of {subject}",
        *style_config['style_terms'][1:],
        f"shot on {style_config['cameras'][0]}",
        f"{style_config['lenses'][0]} lens",
        "high resolution",
        "award-winning photography"
    ]
    
    return ", ".join(prompt_parts)

# Examples
portrait_prompt = build_photography_style_prompt(
    "a businessman in a suit",
    style="portrait"
)
# "professional portrait photography of a businessman in a suit, studio lighting..."

landscape_prompt = build_photography_style_prompt(
    "mountain peaks with lake",
    style="landscape"
)
# "landscape photography of mountain peaks with lake, golden hour lighting..."
\`\`\`

### Artistic Styles

\`\`\`python
artistic_styles = {
    "digital_art": {
        "descriptors": [
            "digital art", "digital painting",
            "highly detailed", "concept art",
            "trending on artstation"
        ],
        "artist_references": [
            "Greg Rutkowski", "Artgerm", "Alphonse Mucha"
        ]
    },
    
    "oil_painting": {
        "descriptors": [
            "oil painting", "traditional art",
            "brush strokes", "canvas texture",
            "masterpiece", "museum quality"
        ],
        "artist_references": [
            "Rembrandt", "Vermeer", "John Singer Sargent"
        ]
    },
    
    "watercolor": {
        "descriptors": [
            "watercolor painting", "soft colors",
            "flowing", "transparent layers",
            "artistic", "delicate"
        ],
        "artist_references": [
            "Winslow Homer", "John James Audubon"
        ]
    },
    
    "anime": {
        "descriptors": [
            "anime style", "manga",
            "cel shading", "vibrant colors",
            "detailed", "studio quality"
        ],
        "artist_references": [
            "Makoto Shinkai", "Studio Ghibli", "Kyoto Animation"
        ]
    },
    
    "3d_render": {
        "descriptors": [
            "3d render", "octane render",
            "unreal engine", "ray tracing",
            "highly detailed", "4k", "8k"
        ],
        "artist_references": [
            "Beeple", "Peter Guthrie"
        ]
    },
    
    "pixel_art": {
        "descriptors": [
            "pixel art", "16-bit style",
            "retro game art", "vibrant colors",
            "crisp pixels", "isometric"
        ],
        "artist_references": []
    }
}

def build_artistic_prompt(
    subject: str,
    style: str = "digital_art",
    include_artist: bool = True
) -> str:
    """Build artistic style prompt."""
    style_config = artistic_styles.get(style, artistic_styles["digital_art"])
    
    parts = [
        f"{style_config['descriptors'][0]} of {subject}",
        *style_config['descriptors'][1:]
    ]
    
    if include_artist and style_config.get('artist_references'):
        artist = style_config['artist_references'][0]
        parts.insert(1, f"in the style of {artist}")
    
    return ", ".join(parts)

# Examples
digital_art = build_artistic_prompt(
    "a dragon flying over mountains",
    style="digital_art"
)
# "digital art of a dragon flying over mountains, in the style of Greg Rutkowski..."

oil_painting = build_artistic_prompt(
    "a still life with fruits",
    style="oil_painting"
)
# "oil painting of a still life with fruits, in the style of Rembrandt..."
\`\`\`

## Lighting and Mood

### Lighting Types

\`\`\`python
lighting_guide = {
    "natural": {
        "golden_hour": "warm, soft, horizontal light, long shadows, magical",
        "blue_hour": "cool tones, twilight, serene atmosphere",
        "midday": "harsh shadows, bright, high contrast",
        "overcast": "soft, even lighting, no harsh shadows, subdued",
        "sunrise_sunset": "warm colors, dramatic, long shadows"
    },
    
    "studio": {
        "rembrandt": "dramatic, triangle of light on cheek, moody",
        "butterfly": "centered light above, symmetrical, glamorous",
        "split": "half face lit, half shadow, dramatic",
        "rim": "backlit, glowing edges, dramatic silhouette",
        "soft_box": "even, diffused, flattering, commercial"
    },
    
    "dramatic": {
        "chiaroscuro": "strong contrast, deep shadows, dramatic",
        "low_key": "mostly dark, focused highlights, moody",
        "high_key": "bright, minimal shadows, airy",
        "side_lighting": "textured, dimensional, dramatic",
        "backlighting": "silhouette, glowing edges, atmospheric"
    },
    
    "mood": {
        "cinematic": "moody, color grading, film-like, atmospheric",
        "ethereal": "soft, glowing, dreamy, magical",
        "noir": "high contrast, shadows, mysterious, dramatic",
        "warm": "golden tones, cozy, inviting, comfortable",
        "cool": "blue tones, calm, professional, clean"
    }
}

def add_lighting(base_prompt: str, lighting_type: str, category: str = "natural") -> str:
    """Add lighting description to prompt."""
    lighting = lighting_guide.get(category, {}).get(lighting_type, "")
    
    if lighting:
        return f"{base_prompt}, {lighting} lighting"
    return base_prompt

# Examples
portrait_with_lighting = add_lighting(
    "a portrait of a woman",
    "rembrandt",
    "studio"
)
# "a portrait of a woman, dramatic, triangle of light on cheek, moody lighting"

landscape_with_lighting = add_lighting(
    "mountain landscape",
    "golden_hour",
    "natural"
)
# "mountain landscape, warm, soft, horizontal light, long shadows, magical lighting"
\`\`\`

## Advanced Composition

### Camera Angles and Framing

\`\`\`python
composition_guide = {
    "camera_angles": {
        "eye_level": "neutral, natural, straight-on view",
        "low_angle": "looking up, dramatic, empowering, heroic",
        "high_angle": "looking down, vulnerable, overview",
        "dutch_angle": "tilted, dynamic, unsettling, action",
        "birds_eye": "directly overhead, flat lay, graphic",
        "worms_eye": "extremely low, dramatic perspective"
    },
    
    "shot_types": {
        "extreme_close_up": "very tight, details, intimate",
        "close_up": "face or object fills frame, detailed",
        "medium_shot": "waist up, standard portrait",
        "full_shot": "entire subject, head to toe",
        "wide_shot": "subject in environment, context",
        "extreme_wide": "vast scene, epic scale"
    },
    
    "framing": {
        "rule_of_thirds": "subject off-center, balanced, professional",
        "centered": "symmetrical, formal, balanced",
        "leading_lines": "lines guide eye to subject, depth",
        "frame_within_frame": "natural framing, focused attention",
        "negative_space": "minimalist, emphasis on subject",
        "fill_frame": "subject fills entire frame, impactful"
    }
}

class CompositionBuilder:
    """Build prompts with specific composition."""
    
    @staticmethod
    def with_angle(subject: str, angle: str) -> str:
        """Add camera angle."""
        angle_desc = composition_guide["camera_angles"].get(angle, "")
        return f"{angle} view of {subject}, {angle_desc}"
    
    @staticmethod
    def with_shot_type(subject: str, shot: str) -> str:
        """Add shot type."""
        shot_desc = composition_guide["shot_types"].get(shot, "")
        return f"{shot} of {subject}, {shot_desc}"
    
    @staticmethod
    def with_framing(subject: str, framing: str) -> str:
        """Add framing technique."""
        frame_desc = composition_guide["framing"].get(framing, "")
        return f"{subject}, {frame_desc} composition"

comp = CompositionBuilder()

# Low angle hero shot
hero = comp.with_angle(
    "a superhero in costume",
    "low_angle"
)
# "low angle view of a superhero in costume, looking up, dramatic, empowering, heroic"

# Close-up portrait
portrait = comp.with_shot_type(
    "a woman's face",
    "close_up"
)
# "close-up of a woman's face, face or object fills frame, detailed"

# Rule of thirds landscape
landscape = comp.with_framing(
    "a lone tree in a field",
    "rule_of_thirds"
)
# "a lone tree in a field, subject off-center, balanced, professional composition"
\`\`\`

## Quality Boosters

### Universal Quality Terms

\`\`\`python
quality_modifiers = {
    "resolution": [
        "4k", "8k", "high resolution",
        "ultra detailed", "highly detailed",
        "sharp focus", "crisp", "clear"
    ],
    
    "professional": [
        "professional photography",
        "award-winning",
        "masterpiece",
        "best quality",
        "professional lighting"
    ],
    
    "artistic": [
        "trending on artstation",
        "behance HD",
        "featured on artstation",
        "instagram worthy",
        "gallery quality"
    ],
    
    "technical": [
        "sharp focus",
        "shallow depth of field",
        "bokeh",
        "perfect composition",
        "rule of thirds"
    ]
}

def add_quality_boosters(
    prompt: str,
    boost_type: str = "professional",
    count: int = 3
) -> str:
    """
    Add quality-boosting terms to prompt.
    """
    boosters = quality_modifiers.get(boost_type, quality_modifiers["professional"])
    selected = boosters[:count]
    
    return f"{prompt}, {', '.join(selected)}"

# Examples
photo = "a portrait of a businessman"
boosted = add_quality_boosters(photo, "professional", 3)
# "a portrait of a businessman, professional photography, award-winning, masterpiece"

art = "a fantasy landscape with castle"
boosted_art = add_quality_boosters(art, "artistic", 2)
# "a fantasy landscape with castle, trending on artstation, behance HD"
\`\`\`

## Production Prompt Templates

### Reusable Prompt Systems

\`\`\`python
class PromptTemplateSystem:
    """
    Production-ready prompt templates.
    """
    
    @staticmethod
    def product_photography(
        product: str,
        background: str = "white",
        angle: str = "straight-on",
        lighting: str = "soft studio lighting"
    ) -> dict:
        """E-commerce product photography."""
        return {
            "prompt": f"""
            professional product photography of {product},
            clean {background} background,
            {angle} angle,
            {lighting},
            commercial photography,
            high resolution,
            sharp focus,
            no shadows,
            studio quality,
            8k
            """.strip(),
            "negative": "cluttered, messy, low quality, blurry, shadows, distracting elements"
        }
    
    @staticmethod
    def social_media_portrait(
        subject: str,
        mood: str = "friendly",
        style: str = "professional"
    ) -> dict:
        """Social media profile/post image."""
        return {
            "prompt": f"""
            {style} portrait photography of {subject},
            {mood} expression,
            shallow depth of field,
            natural lighting,
            subtle bokeh background,
            instagram aesthetic,
            high quality,
            sharp focus,
            modern style
            """.strip(),
            "negative": "blurry, low quality, bad anatomy, distorted face, ugly"
        }
    
    @staticmethod
    def concept_art(
        subject: str,
        style: str = "fantasy",
        mood: str = "epic"
    ) -> dict:
        """Game/film concept art."""
        return {
            "prompt": f"""
            {style} concept art of {subject},
            {mood} atmosphere,
            highly detailed,
            digital painting,
            trending on artstation,
            dramatic lighting,
            vibrant colors,
            by Greg Rutkowski and Alphonse Mucha,
            masterpiece,
            8k
            """.strip(),
            "negative": "blurry, low quality, amateur, simple, plain"
        }
    
    @staticmethod
    def marketing_banner(
        subject: str,
        style: str = "modern",
        colors: str = "vibrant"
    ) -> dict:
        """Marketing banner/hero image."""
        return {
            "prompt": f"""
            {style} marketing photography of {subject},
            {colors} colors,
            professional composition,
            eye-catching,
            clean design,
            commercial photography,
            high quality,
            sharp focus,
            perfect lighting,
            4k resolution
            """.strip(),
            "negative": "cluttered, messy, low quality, amateur, dull colors"
        }

# Usage
templates = PromptTemplateSystem()

# Product photo
product = templates.product_photography(
    product="red sneaker",
    background="white",
    angle="3/4 view"
)
print(f"Prompt: {product['prompt']}")
print(f"Negative: {product['negative']}")

# Social media portrait
portrait = templates.social_media_portrait(
    subject="young professional woman",
    mood="confident",
    style="modern casual"
)

# Concept art
concept = templates.concept_art(
    subject="futuristic city with flying cars",
    style="cyberpunk",
    mood="atmospheric"
)
\`\`\`

## Testing and Iteration

### A/B Testing Prompts

\`\`\`python
class PromptTester:
    """
    Test and compare prompt variations.
    """
    
    def __init__(self, generator):
        self.generator = generator
    
    def test_variations(
        self,
        base_prompt: str,
        variations: dict,
        **kwargs
    ) -> dict:
        """
        Test multiple prompt variations.
        
        Args:
            base_prompt: Base description
            variations: {name: additional_terms} dict
        """
        results = {}
        
        for name, terms in variations.items():
            full_prompt = f"{base_prompt}, {terms}"
            
            images = self.generator.generate(
                prompt=full_prompt,
                **kwargs
            )
            
            results[name] = {
                "prompt": full_prompt,
                "image": images[0]
            }
        
        return results
    
    def test_negative_prompts(
        self,
        prompt: str,
        negative_variations: List[str],
        **kwargs
    ) -> dict:
        """Test different negative prompts."""
        results = {}
        
        for i, negative in enumerate(negative_variations):
            images = self.generator.generate(
                prompt=prompt,
                negative_prompt=negative,
                **kwargs
            )
            
            results[f"variation_{i}"] = {
                "negative": negative,
                "image": images[0]
            }
        
        return results

# Example: Test style variations
# tester = PromptTester(generator)

# style_tests = tester.test_variations(
#     base_prompt="a portrait of a woman",
#     variations={
#         "professional": "professional photography, studio lighting",
#         "artistic": "digital painting, artstation, detailed",
#         "cinematic": "cinematic lighting, film grain, dramatic"
#     },
#     steps=30,
#     seed=42
# )
\`\`\`

## Key Takeaways

- **Structure matters**: Subject → Action → Context → Style → Technical → Quality
- **Be specific**: Detailed descriptions produce better results
- **Use negative prompts**: Essential for quality control
- **Weights control emphasis**: (keyword:1.5) for more, (keyword:0.7) for less
- **Style references work**: "shot on Canon R5" or "by Greg Rutkowski"
- **Quality boosters help**: "award-winning", "trending on artstation"
- **Lighting is critical**: Defines mood and atmosphere
- **Composition guides**: Camera angles, framing, shot types
- **Templates save time**: Reusable patterns for common use cases
- **Test and iterate**: A/B test variations to find what works
`,
};
