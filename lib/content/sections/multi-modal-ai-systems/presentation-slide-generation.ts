export const presentationSlideGeneration = {
  title: 'Presentation & Slide Generation',
  id: 'presentation-slide-generation',
  description:
    'Master auto-generating presentations and slides from content using AI - from outlines to complete slide decks with visuals.',
  content: `
# Presentation & Slide Generation

## Introduction

Generating presentations automatically from content is a powerful application of multi-modal AI. This involves not just creating text content, but designing layouts, selecting appropriate visuals, maintaining consistent styling, and producing professional slide decks.

In this section, we'll explore how to build systems that can generate complete presentations from prompts, outlines, or existing content, including slide layouts, speaker notes, and visual elements.

## Presentation Generation Pipeline

### 1. Content → Outline → Slides → Visuals

\`\`\`
Input Content → [Structure] → Outline → [Design] → Slide Layout → [Generate] → Visuals → Final Presentation
\`\`\`

### 2. Components

**Content Structure:**
- Title slide
- Agenda/outline
- Content slides
- Conclusion
- Q&A slide

**Design Elements:**
- Layout templates
- Color schemes
- Typography
- Images and graphics
- Charts and diagrams

## Building a Presentation Generator

### Basic Implementation

\`\`\`python
from typing import List, Dict, Any, Optional
from openai import OpenAI
import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
import requests
import base64

client = OpenAI()

def generate_presentation_outline(
    topic: str,
    target_audience: str = "general",
    duration_minutes: int = 15,
    num_slides: int = 10
) -> Dict[str, Any]:
    """
    Generate structured outline for presentation.
    
    Args:
        topic: Presentation topic
        target_audience: Target audience (general, technical, executive, etc.)
        duration_minutes: Presentation duration
        num_slides: Desired number of slides
    
    Returns:
        Structured outline
    """
    prompt = f"""Create a presentation outline for: {topic}

Requirements:
- Target audience: {target_audience}
- Duration: {duration_minutes} minutes
- Number of slides: {num_slides}

Generate a JSON outline with this structure:
{{
  "title": "Presentation title",
  "subtitle": "Subtitle or tagline",
  "slides": [
    {{
      "slide_number": 1,
      "title": "Slide title",
      "content_type": "title|bullet_points|image|chart|quote|conclusion",
      "key_points": ["point 1", "point 2", ...],
      "speaker_notes": "Notes for presenter",
      "visual_suggestion": "Description of image or chart"
    }}
  ]
}}

Make it engaging and well-structured for the audience."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    outline = json.loads (response.choices[0].message.content)
    return outline

# Generate outline
outline = generate_presentation_outline(
    "The Future of Artificial Intelligence",
    target_audience="business leaders",
    duration_minutes=20,
    num_slides=15
)

print(f"Title: {outline['title']}")
print(f"Slides: {len (outline['slides'])}")
\`\`\`

### Creating PowerPoint Slides

\`\`\`python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

def create_title_slide(
    prs: Presentation,
    title: str,
    subtitle: str
):
    """Create title slide."""
    slide_layout = prs.slide_layouts[0]  # Title slide layout
    slide = prs.slides.add_slide (slide_layout)
    
    # Set title
    title_shape = slide.shapes.title
    title_shape.text = title
    
    # Set subtitle
    subtitle_shape = slide.placeholders[1]
    subtitle_shape.text = subtitle
    
    return slide

def create_content_slide(
    prs: Presentation,
    title: str,
    bullet_points: List[str],
    image_path: Optional[str] = None
):
    """Create content slide with bullet points and optional image."""
    # Use title and content layout
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide (slide_layout)
    
    # Set title
    title_shape = slide.shapes.title
    title_shape.text = title
    
    # Add bullet points
    content_shape = slide.placeholders[1]
    text_frame = content_shape.text_frame
    text_frame.clear()  # Clear default text
    
    for point in bullet_points:
        p = text_frame.add_paragraph()
        p.text = point
        p.level = 0
        p.font.size = Pt(18)
    
    # Add image if provided
    if image_path:
        left = Inches(7)
        top = Inches(2)
        width = Inches(3)
        slide.shapes.add_picture (image_path, left, top, width=width)
    
    return slide

def create_image_slide(
    prs: Presentation,
    title: str,
    image_path: str,
    caption: Optional[str] = None
):
    """Create slide with large image."""
    slide_layout = prs.slide_layouts[6]  # Blank layout
    slide = prs.slides.add_slide (slide_layout)
    
    # Add title
    title_shape = slide.shapes.add_textbox(
        Inches(0.5),
        Inches(0.5),
        Inches(9),
        Inches(0.8)
    )
    title_frame = title_shape.text_frame
    title_paragraph = title_frame.paragraphs[0]
    title_paragraph.text = title
    title_paragraph.font.size = Pt(32)
    title_paragraph.font.bold = True
    
    # Add image
    left = Inches(1)
    top = Inches(1.5)
    width = Inches(8)
    slide.shapes.add_picture (image_path, left, top, width=width)
    
    # Add caption if provided
    if caption:
        caption_shape = slide.shapes.add_textbox(
            Inches(1),
            Inches(6.5),
            Inches(8),
            Inches(0.5)
        )
        caption_frame = caption_shape.text_frame
        caption_para = caption_frame.paragraphs[0]
        caption_para.text = caption
        caption_para.font.size = Pt(14)
        caption_para.font.italic = True
        caption_para.alignment = PP_ALIGN.CENTER
    
    return slide

def generate_slide_image(
    description: str,
    style: str = "professional"
) -> str:
    """Generate image for slide using DALL-E."""
    prompt = f"""Create a {style} presentation slide image: {description}

Style: Clean, modern, suitable for business presentation.
High quality, 16:9 aspect ratio."""

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1792x1024",  # 16:9 aspect ratio
        quality="hd"
    )
    
    # Download image
    img_url = response.data[0].url
    img_data = requests.get (img_url).content
    
    # Save image
    import hashlib
    img_hash = hashlib.md5(description.encode()).hexdigest()[:8]
    img_path = f"slide_image_{img_hash}.png"
    
    with open (img_path, "wb") as f:
        f.write (img_data)
    
    return img_path

def create_presentation_from_outline(
    outline: Dict[str, Any],
    generate_images: bool = True,
    output_filename: str = "presentation.pptx"
) -> str:
    """
    Create complete PowerPoint presentation from outline.
    
    Args:
        outline: Presentation outline
        generate_images: Whether to generate images with DALL-E
        output_filename: Output filename
    
    Returns:
        Path to generated presentation
    """
    # Create presentation object
    prs = Presentation()
    
    # Set slide size to 16:9
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625)
    
    # Create title slide
    create_title_slide(
        prs,
        outline["title"],
        outline.get("subtitle", "")
    )
    
    # Create content slides
    for slide_data in outline["slides"]:
        content_type = slide_data.get("content_type", "bullet_points")
        
        if content_type == "title":
            # Skip, already created
            continue
        
        elif content_type == "bullet_points":
            # Generate image if requested
            image_path = None
            if generate_images and slide_data.get("visual_suggestion"):
                try:
                    image_path = generate_slide_image(
                        slide_data["visual_suggestion"]
                    )
                except Exception as e:
                    print(f"Failed to generate image: {e}")
            
            create_content_slide(
                prs,
                slide_data["title"],
                slide_data.get("key_points", []),
                image_path
            )
        
        elif content_type == "image":
            if generate_images:
                image_path = generate_slide_image(
                    slide_data.get("visual_suggestion", slide_data["title"])
                )
                
                create_image_slide(
                    prs,
                    slide_data["title"],
                    image_path,
                    caption=slide_data.get("speaker_notes", "")
                )
            else:
                # Create bullet point slide instead
                create_content_slide(
                    prs,
                    slide_data["title"],
                    slide_data.get("key_points", [])
                )
        
        elif content_type == "conclusion":
            create_content_slide(
                prs,
                slide_data["title"],
                slide_data.get("key_points", [])
            )
    
    # Save presentation
    prs.save (output_filename)
    
    return output_filename

# Complete workflow
outline = generate_presentation_outline(
    "The Future of AI",
    target_audience="technical",
    num_slides=12
)

pptx_path = create_presentation_from_outline(
    outline,
    generate_images=True,
    output_filename="ai_future_presentation.pptx"
)

print(f"Presentation created: {pptx_path}")
\`\`\`

### Production Presentation Generator

\`\`\`python
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig (level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PresentationConfig:
    """Configuration for presentation generation."""
    theme: str = "professional"  # professional, creative, minimal, bold
    color_scheme: str = "blue"  # blue, green, red, purple, orange
    font_size: str = "medium"  # small, medium, large
    include_images: bool = True
    image_style: str = "professional"  # professional, creative, illustration
    include_charts: bool = False
    include_speaker_notes: bool = True

class ProductionPresentationGenerator:
    """Production-ready presentation generator."""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        
        # Theme configurations
        self.themes = {
            "professional": {
                "colors": {"blue": (44, 62, 80), "accent": (52, 152, 219)},
                "font": "Calibri"
            },
            "creative": {
                "colors": {"primary": (155, 89, 182), "accent": (241, 196, 15)},
                "font": "Arial"
            }
        }
    
    def generate(
        self,
        topic: str,
        target_audience: str = "general",
        duration_minutes: int = 15,
        config: Optional[PresentationConfig] = None
    ) -> str:
        """
        Generate complete presentation.
        
        Args:
            topic: Presentation topic
            target_audience: Target audience
            duration_minutes: Duration in minutes
            config: Presentation configuration
        
        Returns:
            Path to generated presentation file
        """
        if config is None:
            config = PresentationConfig()
        
        logger.info (f"Generating presentation on '{topic}' for {target_audience}")
        
        # Step 1: Generate outline
        num_slides = max(5, duration_minutes // 2)  # ~2 minutes per slide
        outline = self._generate_outline (topic, target_audience, num_slides)
        
        # Step 2: Enhance slide content
        enhanced_outline = self._enhance_slides (outline, config)
        
        # Step 3: Generate images if requested
        if config.include_images:
            enhanced_outline = self._add_images (enhanced_outline, config.image_style)
        
        # Step 4: Create PowerPoint
        output_filename = f"{topic.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx"
        self._create_pptx (enhanced_outline, config, output_filename)
        
        logger.info (f"Presentation created: {output_filename}")
        
        return output_filename
    
    def _generate_outline(
        self,
        topic: str,
        audience: str,
        num_slides: int
    ) -> Dict[str, Any]:
        """Generate presentation outline."""
        # Use generate_presentation_outline function from above
        return generate_presentation_outline (topic, audience, num_slides * 2, num_slides)
    
    def _enhance_slides(
        self,
        outline: Dict[str, Any],
        config: PresentationConfig
    ) -> Dict[str, Any]:
        """Enhance slide content with better phrasing."""
        for slide in outline["slides"]:
            # Enhance bullet points
            if "key_points" in slide and slide["key_points"]:
                enhancement_prompt = f"""Improve these bullet points for a presentation:

{chr(10).join (slide["key_points"])}

Make them:
- Concise and punchy
- Parallel structure
- Action-oriented where appropriate
- Easy to read on a slide

Return as JSON array of strings."""

                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": enhancement_prompt}],
                        temperature=0.7
                    )
                    
                    enhanced_points = json.loads (response.choices[0].message.content)
                    slide["key_points"] = enhanced_points
                except Exception as e:
                    logger.warning (f"Failed to enhance slide content: {e}")
        
        return outline
    
    def _add_images(
        self,
        outline: Dict[str, Any],
        style: str
    ) -> Dict[str, Any]:
        """Generate and add images to slides."""
        for slide in outline["slides"]:
            if slide.get("visual_suggestion"):
                try:
                    image_path = generate_slide_image(
                        slide["visual_suggestion"],
                        style=style
                    )
                    slide["image_path"] = image_path
                except Exception as e:
                    logger.warning (f"Failed to generate image for slide: {e}")
                    slide["image_path"] = None
        
        return outline
    
    def _create_pptx(
        self,
        outline: Dict[str, Any],
        config: PresentationConfig,
        output_filename: str
    ):
        """Create PowerPoint file."""
        # Use create_presentation_from_outline with modifications
        prs = Presentation()
        
        # Configure theme
        # (Implementation would apply colors, fonts, etc.)
        
        # Create slides
        create_presentation_from_outline(
            outline,
            generate_images=False,  # Already generated
            output_filename=output_filename
        )
        
        return output_filename

# Usage
generator = ProductionPresentationGenerator(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

config = PresentationConfig(
    theme="professional",
    color_scheme="blue",
    include_images=True,
    image_style="professional"
)

pptx_path = generator.generate(
    topic="Machine Learning in Healthcare",
    target_audience="healthcare executives",
    duration_minutes=20,
    config=config
)

print(f"Generated: {pptx_path}")
\`\`\`

## Advanced Features

### Adding Charts

\`\`\`python
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches

def create_chart_slide(
    prs: Presentation,
    title: str,
    chart_data: Dict[str, List[float]],
    chart_type: str = "bar"
) -> None:
    """Create slide with chart."""
    slide_layout = prs.slide_layouts[5]  # Blank layout
    slide = prs.slides.add_slide (slide_layout)
    
    # Add title
    title_shape = slide.shapes.add_textbox(
        Inches(0.5), Inches(0.5), Inches(9), Inches(0.8)
    )
    title_frame = title_shape.text_frame
    title_frame.text = title
    
    # Prepare chart data
    chart_data_obj = CategoryChartData()
    chart_data_obj.categories = list (chart_data.keys())
    chart_data_obj.add_series('Series 1', list (chart_data.values()))
    
    # Add chart
    x, y, cx, cy = Inches(1), Inches(2), Inches(8), Inches(4.5)
    
    chart_type_map = {
        "bar": XL_CHART_TYPE.BAR_CLUSTERED,
        "column": XL_CHART_TYPE.COLUMN_CLUSTERED,
        "line": XL_CHART_TYPE.LINE,
        "pie": XL_CHART_TYPE.PIE
    }
    
    chart = slide.shapes.add_chart(
        chart_type_map.get (chart_type, XL_CHART_TYPE.BAR_CLUSTERED),
        x, y, cx, cy,
        chart_data_obj
    ).chart
    
    return slide

# Generate chart data from text
def generate_chart_from_description(
    description: str
) -> Dict[str, Any]:
    """Generate chart data from natural language description."""
    prompt = f"""Convert this description into chart data:

{description}

Return as JSON:
{{
  "title": "Chart title",
  "chart_type": "bar|column|line|pie",
  "data": {{
    "category1": value1,
    "category2": value2,
    ...
  }}
}}"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads (response.choices[0].message.content)

# Example
chart_info = generate_chart_from_description(
    "Show quarterly revenue: Q1 $1.2M, Q2 $1.5M, Q3 $1.8M, Q4 $2.1M"
)

prs = Presentation()
create_chart_slide(
    prs,
    chart_info["title"],
    chart_info["data"],
    chart_info["chart_type"]
)
prs.save("chart_presentation.pptx")
\`\`\`

### Design Templates

\`\`\`python
class PresentationTheme:
    """Presentation theme with colors and styling."""
    
    def __init__(
        self,
        name: str,
        primary_color: tuple,
        accent_color: tuple,
        background_color: tuple,
        text_color: tuple
    ):
        self.name = name
        self.primary_color = RGBColor(*primary_color)
        self.accent_color = RGBColor(*accent_color)
        self.background_color = RGBColor(*background_color)
        self.text_color = RGBColor(*text_color)
    
    def apply_to_slide (self, slide):
        """Apply theme to slide."""
        # Set background
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = self.background_color

# Predefined themes
THEMES = {
    "professional": PresentationTheme(
        "Professional",
        primary_color=(44, 62, 80),
        accent_color=(52, 152, 219),
        background_color=(255, 255, 255),
        text_color=(44, 62, 80)
    ),
    "creative": PresentationTheme(
        "Creative",
        primary_color=(155, 89, 182),
        accent_color=(241, 196, 15),
        background_color=(44, 62, 80),
        text_color=(255, 255, 255)
    ),
    "minimal": PresentationTheme(
        "Minimal",
        primary_color=(0, 0, 0),
        accent_color=(100, 100, 100),
        background_color=(255, 255, 255),
        text_color=(0, 0, 0)
    )
}
\`\`\`

### Speaker Notes

\`\`\`python
def add_speaker_notes(
    slide,
    notes_text: str
):
    """Add speaker notes to slide."""
    notes_slide = slide.notes_slide
    text_frame = notes_slide.notes_text_frame
    text_frame.text = notes_text

def generate_speaker_notes(
    slide_title: str,
    slide_content: List[str]
) -> str:
    """Generate speaker notes for slide."""
    prompt = f"""Generate speaker notes for this presentation slide:

Title: {slide_title}

Content:
{chr(10).join (f'- {point}' for point in slide_content)}

Provide notes that:
- Expand on the bullet points
- Include examples or anecdotes
- Suggest transitions
- Are 2-3 sentences per bullet point

Return as plain text."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content

# Add notes to slides
for slide_data in outline["slides"]:
    notes = generate_speaker_notes(
        slide_data["title"],
        slide_data.get("key_points", [])
    )
    # Add to slide (when creating)
    slide_data["speaker_notes"] = notes
\`\`\`

## Best Practices

### 1. Slide Design Principles

- **One idea per slide**: Don't overcrowd
- **Limit bullets**: 3-5 points max
- **Use visuals**: Images > text
- **Consistent styling**: Same fonts, colors throughout
- **High contrast**: Readable from distance
- **White space**: Don't fill every pixel

### 2. Content Guidelines

- **Clear headlines**: Descriptive titles
- **Concise bullets**: 1-2 lines each
- **Parallel structure**: Consistent grammar
- **Action-oriented**: Start with verbs
- **Data visualization**: Use charts for numbers

### 3. Image Selection

- **High quality**: Professional photos
- **Relevant**: Directly related to content
- **Consistent style**: Same visual language
- **Proper sizing**: Fill space appropriately
- **Attribution**: Credit sources if needed

## Real-World Applications

### 1. Sales Presentations

\`\`\`python
def generate_sales_presentation(
    product_name: str,
    features: List[str],
    target_market: str
) -> str:
    """Generate sales presentation."""
    # Generate compelling narrative
    outline_prompt = f"""Create a sales presentation outline for {product_name}

Target market: {target_market}
Key features: {', '.join (features)}

Include:
- Problem statement
- Solution (product)
- Key benefits
- Social proof
- Call to action

15 slides total."""

    outline = generate_presentation_outline(
        product_name,
        target_audience=f"{target_market} buyers",
        num_slides=15
    )
    
    return create_presentation_from_outline (outline)
\`\`\`

### 2. Educational Content

\`\`\`python
def generate_educational_presentation(
    topic: str,
    learning_level: str = "beginner"
) -> str:
    """Generate educational presentation with explanations."""
    outline = generate_presentation_outline(
        f"{topic} - {learning_level} level",
        target_audience=f"{learning_level} students",
        num_slides=20
    )
    
    # Add detailed speaker notes for teacher
    for slide in outline["slides"]:
        slide["speaker_notes"] = generate_speaker_notes(
            slide["title"],
            slide.get("key_points", [])
        )
    
    return create_presentation_from_outline(
        outline,
        generate_images=True,
        output_filename=f"{topic}_lesson.pptx"
    )
\`\`\`

## Summary

Presentation and slide generation automates creating professional slide decks:

**Key Capabilities:**
- Auto-generate outlines from topics
- Create structured slides with layouts
- Generate relevant images for slides
- Add charts and data visualizations
- Apply consistent themes and styling
- Generate speaker notes

**Production Patterns:**
- Start with outline generation
- Enhance content for readability
- Generate visuals separately
- Apply consistent theming
- Add speaker notes
- Validate final output

**Best Practices:**
- Keep slides simple (3-5 bullets max)
- Use high-quality images
- Maintain consistent styling
- Generate speaker notes
- Follow design principles
- Test on actual projector/screen

**Applications:**
- Sales and marketing presentations
- Educational content
- Business reports
- Conference talks
- Training materials
- Product launches

Next, we'll explore multi-modal agents that can reason and act across modalities.
`,
  codeExamples: [
    {
      title: 'Production Presentation Generator',
      description:
        'Complete system for generating professional presentations with images and charts',
      language: 'python',
      code: `# See ProductionPresentationGenerator class in content above`,
    },
  ],
  practicalTips: [
    'Generate outline first, then enhance content - two-step process produces better results',
    "Limit slides to 3-5 bullet points maximum - presentations aren't documents",
    'Generate images at 16:9 aspect ratio (1792x1024) for modern slides',
    'Use temperature 0.7-0.8 for creative content generation',
    "Add speaker notes - they provide context that shouldn't be on slides",
    'Apply consistent themes - same colors, fonts, layouts throughout',
    'Test presentations on actual display before delivery',
    'Cache generated images to avoid regenerating identical visuals',
  ],
  quiz: '/quizzes/multi-modal-ai-systems/presentation-slide-generation',
};
