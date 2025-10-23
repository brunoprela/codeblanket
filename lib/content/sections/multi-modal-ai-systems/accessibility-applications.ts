export const accessibilityApplications = {
  title: 'Accessibility Applications',
  id: 'accessibility-applications',
  description:
    'Master building accessibility tools using multi-modal AI - alt-text generation, audio descriptions, sign language translation, and more.',
  content: `
# Accessibility Applications

## Introduction

Multi-modal AI has transformative potential for accessibility, enabling people with disabilities to access content and interact with technology in new ways. From automatically generating alt-text for images to providing real-time audio descriptions, these applications can significantly improve digital accessibility.

In this section, we'll explore how to build production-ready accessibility tools using multi-modal AI.

## Core Accessibility Use Cases

### 1. Image Alt-Text Generation

Generate descriptive alt-text for images to make them accessible to screen reader users.

\`\`\`python
from typing import Optional
from openai import OpenAI
import base64

client = OpenAI()

def generate_alt_text(
    image_path: str,
    context: Optional[str] = None,
    max_length: int = 125  # WCAG recommendation
) -> str:
    """
    Generate accessible alt-text for image.
    
    Args:
        image_path: Path to image
        context: Optional context (e.g., "product page", "blog post")
        max_length: Maximum character length
    
    Returns:
        Alt-text description
    """
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    prompt = f"""Generate concise, descriptive alt-text for this image.

Guidelines:
- Be specific and descriptive
- Focus on relevant content
- Keep under {max_length} characters
- Don't start with "Image of" or "Picture of"
- Describe important details that convey meaning
- Avoid redundant information

{f'Context: This image appears in a {context}' if context else ''}

Provide only the alt-text, nothing else."""

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }],
        max_tokens=100,
        temperature=0.3  # More deterministic for consistency
    )
    
    alt_text = response.choices[0].message.content.strip()
    
    # Ensure length constraint
    if len(alt_text) > max_length:
        alt_text = alt_text[:max_length-3] + "..."
    
    return alt_text

# Example usage
alt_text = generate_alt_text(
    "product.jpg",
    context="e-commerce product page"
)
print(f'<img src="product.jpg" alt="{alt_text}">')
\`\`\`

### 2. Long Description Generation

For complex images that need more detailed descriptions.

\`\`\`python
def generate_long_description(
    image_path: str,
    image_type: str = "chart"  # chart, diagram, infographic, complex
) -> str:
    """
    Generate detailed long description for complex images.
    
    Used when alt-text alone isn't sufficient.
    """
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    prompts = {
        "chart": """Provide a detailed description of this chart/graph.

Include:
- Type of chart (bar, line, pie, etc.)
- What data is being shown
- Key trends or patterns
- Important data points and values
- Axis labels and units

Format as structured text suitable for screen readers.""",
        
        "diagram": """Describe this diagram in detail.

Include:
- Overall purpose or concept
- Main components and their relationships
- Flow or process (if applicable)
- Labels and annotations
- Spatial relationships

Use clear, logical structure.""",
        
        "infographic": """Provide a comprehensive description of this infographic.

Include:
- Main topic or message
- Key statistics and data
- Visual sections and their content
- Important quotes or highlights

Organize in reading order.""",
        
        "complex": """Provide a detailed description of this complex image.

Describe:
- Overall scene or purpose
- Main elements and their relationships
- Important details
- Context and meaning

Structure logically for audio presentation."""
    }
    
    prompt = prompts.get(image_type, prompts["complex"])
    
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }],
        max_tokens=800,
        temperature=0.3
    )
    
    return response.choices[0].message.content

# Generate long description
long_desc = generate_long_description("complex_chart.png", "chart")
print(f"Long description: {long_desc}")
\`\`\`

### 3. Audio Descriptions for Video

Generate audio descriptions for video content.

\`\`\`python
import cv2
import numpy as np
from typing import List, Dict, Any

def generate_video_audio_descriptions(
    video_path: str,
    description_density: str = "standard"  # minimal, standard, extended
) -> List[Dict[str, Any]]:
    """
    Generate audio descriptions for video.
    
    Audio descriptions fill in visual information during natural pauses.
    """
    # Extract key frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames_to_describe = []
    frame_count = 0
    
    # Sample every 2 seconds for standard density
    sample_interval = int(fps * 2) if description_density == "standard" else int(fps * 5)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_interval == 0:
            timestamp = frame_count / fps
            
            # Save frame
            frame_path = f"frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)
            
            frames_to_describe.append({
                "timestamp": timestamp,
                "frame_path": frame_path
            })
        
        frame_count += 1
    
    cap.release()
    
    # Generate descriptions for each frame
    descriptions = []
    
    for frame_info in frames_to_describe:
        with open(frame_info["frame_path"], "rb") as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        
        prompt = """Describe what's happening in this video frame for audio description.

Focus on:
- Key visual actions
- Important scene changes
- Facial expressions and body language
- Text on screen
- Scene setting if it changed

Keep it concise (1-2 sentences) as it will be read during video pauses."""

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data}",
                            "detail": "low"
                        }
                    }
                ]
            }],
            max_tokens=100
        )
        
        description = response.choices[0].message.content
        
        descriptions.append({
            "timestamp": frame_info["timestamp"],
            "description": description,
            "duration": len(description.split()) / 2.5  # Approximate speaking time
        })
    
    return descriptions

# Generate audio descriptions
descriptions = generate_video_audio_descriptions("video.mp4", "standard")

for desc in descriptions:
    print(f"[{desc['timestamp']:.1f}s]: {desc['description']}")
\`\`\`

### 4. Document Accessibility

Make documents accessible.

\`\`\`python
def make_document_accessible(
    pdf_path: str
) -> Dict[str, Any]:
    """
    Analyze document and generate accessibility enhancements.
    
    Returns:
        Accessibility metadata and enhancements
    """
    import fitz  # PyMuPDF
    
    doc = fitz.open(pdf_path)
    
    accessibility_data = {
        "title": "",
        "language": "en",
        "pages": [],
        "images_without_alt": [],
        "tables": [],
        "headings": []
    }
    
    for page_num, page in enumerate(doc):
        page_data = {
            "page": page_num + 1,
            "text": page.get_text(),
            "images": []
        }
        
        # Extract and describe images
        for img_index, img in enumerate(page.get_images()):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Save image temporarily
            img_path = f"page_{page_num+1}_img_{img_index}.png"
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            
            # Generate alt-text
            alt_text = generate_alt_text(img_path)
            
            page_data["images"].append({
                "index": img_index,
                "alt_text": alt_text,
                "original_path": img_path
            })
        
        accessibility_data["pages"].append(page_data)
    
    doc.close()
    
    return accessibility_data

# Make document accessible
accessible_doc = make_document_accessible("report.pdf")

print(f"Pages processed: {len(accessible_doc['pages'])}")
print(f"Images described: {sum(len(p['images']) for p in accessible_doc['pages'])}")
\`\`\`

## Production Accessibility System

\`\`\`python
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AccessibilityEnhancement:
    """Represents an accessibility enhancement."""
    content_type: str  # image, video, document, audio
    original_path: str
    enhancements: Dict[str, Any]
    wcag_level: str  # A, AA, AAA
    timestamp: float

class ProductionAccessibilitySystem:
    """Production-ready accessibility enhancement system."""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.enhancements: List[AccessibilityEnhancement] = []
    
    def enhance_image(
        self,
        image_path: str,
        context: Optional[str] = None,
        wcag_level: str = "AA"
    ) -> AccessibilityEnhancement:
        """
        Enhance image accessibility.
        
        Args:
            image_path: Path to image
            context: Optional context
            wcag_level: Target WCAG level (A, AA, AAA)
        
        Returns:
            AccessibilityEnhancement
        """
        enhancements = {}
        
        # Generate alt-text (required for WCAG A)
        enhancements["alt_text"] = generate_alt_text(
            image_path,
            context=context
        )
        
        # Generate long description for complex images (WCAG AA/AAA)
        if wcag_level in ["AA", "AAA"]:
            # Determine if image is complex
            complexity = self._assess_image_complexity(image_path)
            
            if complexity == "complex":
                enhancements["long_description"] = generate_long_description(
                    image_path,
                    "complex"
                )
        
        # Create enhancement record
        enhancement = AccessibilityEnhancement(
            content_type="image",
            original_path=image_path,
            enhancements=enhancements,
            wcag_level=wcag_level,
            timestamp=time.time()
        )
        
        self.enhancements.append(enhancement)
        logger.info(f"Enhanced image: {image_path}")
        
        return enhancement
    
    def _assess_image_complexity(self, image_path: str) -> str:
        """Determine if image is simple or complex."""
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        
        prompt = """Is this image simple or complex?

Simple: Basic photos, icons, simple illustrations
Complex: Charts, diagrams, infographics, detailed illustrations

Answer with just "simple" or "complex"."""

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data}",
                            "detail": "low"
                        }
                    }
                ]
            }],
            max_tokens=10
        )
        
        return response.choices[0].message.content.strip().lower()
    
    def enhance_video(
        self,
        video_path: str,
        include_captions: bool = True,
        include_audio_descriptions: bool = True
    ) -> AccessibilityEnhancement:
        """Enhance video accessibility."""
        enhancements = {}
        
        if include_captions:
            # Extract audio and generate captions
            captions = self._generate_captions(video_path)
            enhancements["captions"] = captions
        
        if include_audio_descriptions:
            # Generate audio descriptions
            descriptions = generate_video_audio_descriptions(video_path)
            enhancements["audio_descriptions"] = descriptions
        
        enhancement = AccessibilityEnhancement(
            content_type="video",
            original_path=video_path,
            enhancements=enhancements,
            wcag_level="AA",
            timestamp=time.time()
        )
        
        self.enhancements.append(enhancement)
        logger.info(f"Enhanced video: {video_path}")
        
        return enhancement
    
    def _generate_captions(self, video_path: str) -> List[Dict[str, Any]]:
        """Generate captions for video."""
        # Extract audio
        import subprocess
        audio_path = "extracted_audio.mp3"
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "mp3", "-y", audio_path
        ], check=True)
        
        # Transcribe with timestamps
        with open(audio_path, "rb") as f:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        
        captions = []
        for segment in transcription.segments:
            captions.append({
                "start": segment['start'],
                "end": segment['end'],
                "text": segment['text']
            })
        
        return captions
    
    def batch_enhance(
        self,
        items: List[Dict[str, str]],
        max_concurrent: int = 5
    ) -> List[AccessibilityEnhancement]:
        """Batch enhance multiple items."""
        from concurrent.futures import ThreadPoolExecutor
        
        def enhance_one(item: Dict[str, str]) -> AccessibilityEnhancement:
            content_type = item['type']
            path = item['path']
            
            if content_type == "image":
                return self.enhance_image(path, context=item.get('context'))
            elif content_type == "video":
                return self.enhance_video(path)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            enhancements = list(executor.map(enhance_one, items))
        
        return enhancements
    
    def export_enhancements(
        self,
        output_path: str = "accessibility_enhancements.json"
    ):
        """Export all enhancements to JSON."""
        import json
        
        data = []
        for enhancement in self.enhancements:
            data.append({
                "content_type": enhancement.content_type,
                "original_path": enhancement.original_path,
                "enhancements": enhancement.enhancements,
                "wcag_level": enhancement.wcag_level,
                "timestamp": enhancement.timestamp
            })
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(data)} enhancements to {output_path}")

# Usage
accessibility_system = ProductionAccessibilitySystem(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Enhance single image
enhancement = accessibility_system.enhance_image(
    "hero_image.jpg",
    context="homepage hero section",
    wcag_level="AA"
)

print(f"Alt-text: {enhancement.enhancements['alt_text']}")

# Batch enhance website images
items = [
    {"type": "image", "path": "img1.jpg", "context": "product page"},
    {"type": "image", "path": "img2.jpg", "context": "blog post"},
    {"type": "video", "path": "promo.mp4"}
]

enhancements = accessibility_system.batch_enhance(items)

# Export all enhancements
accessibility_system.export_enhancements()
\`\`\`

## WCAG Compliance

### WCAG Levels

**Level A (Minimum):**
- All images have alt-text
- Videos have captions
- Color is not the only way to convey information

**Level AA (Mid-Range):**
- Alt-text is descriptive and meaningful
- Videos have audio descriptions
- Contrast ratios meet standards
- Text can be resized to 200%

**Level AAA (Enhanced):**
- Extended audio descriptions
- Sign language interpretation
- Enhanced contrast
- No timing requirements

### Compliance Checker

\`\`\`python
def check_accessibility_compliance(
    html_content: str,
    images: List[str],
    videos: List[str]
) -> Dict[str, Any]:
    """Check accessibility compliance."""
    issues = []
    
    # Check for images without alt-text
    # (Would parse HTML in real implementation)
    
    # Check for complex images without long descriptions
    
    # Check for videos without captions
    
    # Calculate compliance level
    if not issues:
        level = "AAA"
    elif len(issues) <= 2:
        level = "AA"
    else:
        level = "A" if len(issues) <= 5 else "Non-compliant"
    
    return {
        "compliance_level": level,
        "issues": issues,
        "recommendations": []
    }
\`\`\`

## Best Practices

### 1. Alt-Text Guidelines

- **Be concise**: Under 125 characters when possible
- **Be specific**: "Golden retriever puppy" not "dog"
- **Skip redundancy**: Don't say "image of"
- **Context matters**: Alt-text should fit surrounding content
- **Avoid subjective**: Stick to objective descriptions
- **Empty alt for decorative**: Use alt="" for purely decorative images

### 2. Audio Description Guidelines

- **Natural pauses**: Fit descriptions in dialogue pauses
- **Objective**: Describe what's visible, not interpretation
- **Priority**: Focus on plot-relevant visual information
- **Brevity**: Keep it concise
- **Clarity**: Use clear, simple language

### 3. Caption Guidelines

- **Accuracy**: Transcribe exactly
- **Speaker ID**: Indicate who is speaking
- **Sound effects**: Include [important sound effects]
- **Music**: Describe [upbeat music] when relevant
- **Timing**: Sync accurately with audio

## Real-World Applications

### 1. Website Accessibility Audit

\`\`\`python
def audit_website_accessibility(
    website_url: str
) -> Dict[str, Any]:
    """Audit website for accessibility issues."""
    # Would crawl website and check:
    # - Images without alt-text
    # - Videos without captions
    # - Low contrast text
    # - Missing ARIA labels
    # - Keyboard navigation issues
    
    # Generate report with recommendations
    pass
\`\`\`

### 2. E-Learning Platform

\`\`\`python
def make_course_accessible(
    course_materials: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Make educational course materials accessible."""
    # Add alt-text to all images
    # Add captions to all videos
    # Generate transcripts for audio lectures
    # Create accessible PDF versions
    pass
\`\`\`

## Summary

Accessibility applications make content available to everyone:

**Key Capabilities:**
- Auto-generate image alt-text
- Create detailed long descriptions
- Generate video audio descriptions
- Produce accurate captions
- Make documents accessible
- WCAG compliance checking

**Production Patterns:**
- Generate alt-text for all images
- Assess image complexity
- Batch process for efficiency
- Export in standard formats
- Validate compliance levels
- Monitor and update

**Best Practices:**
- Follow WCAG guidelines
- Be concise yet descriptive
- Consider context
- Test with actual users
- Regular audits
- Keep content updated

**Applications:**
- Website accessibility
- E-learning platforms
- Video platforms
- Document publishing
- Mobile applications
- Government services

Next, we'll tie everything together with building complete multi-modal products.
`,
  codeExamples: [
    {
      title: 'Production Accessibility System',
      description:
        'Complete system for enhancing content accessibility with WCAG compliance',
      language: 'python',
      code: `# See ProductionAccessibilitySystem class in content above`,
    },
  ],
  practicalTips: [
    'Keep alt-text under 125 characters - screen readers handle this best',
    'Use temperature=0.3 for alt-text generation - consistency is important',
    'Assess image complexity before generating long descriptions - not all images need them',
    "Generate captions with timestamps using Whisper's verbose_json format",
    'Batch process website images for efficiency - use ThreadPoolExecutor',
    'Test alt-text with actual screen readers (NVDA, JAWS, VoiceOver)',
    'Export enhancements in standard formats (WebVTT for captions, JSON for metadata)',
    'Regular accessibility audits - content changes require updated descriptions',
  ],
  quiz: '/quizzes/multi-modal-ai-systems/accessibility-applications',
};
