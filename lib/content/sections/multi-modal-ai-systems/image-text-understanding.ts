export const imageTextUnderstanding = {
  title: 'Image + Text Understanding',
  id: 'image-text-understanding',
  description:
    'Master visual question answering, image captioning, and building systems that can understand and describe images with natural language.',
  content: `
# Image + Text Understanding

## Introduction

The ability to bridge the gap between visual and linguistic information is one of the most powerful capabilities in modern AI. Image + text understanding enables machines to answer questions about images, generate descriptive captions, perform visual reasoning, and extract structured information from visual content.

In this section, we'll explore the various tasks that combine image and text understanding, learn how to build production systems, and understand how to optimize for accuracy, cost, and latency.

## Core Tasks

### 1. Visual Question Answering (VQA)

VQA involves answering natural language questions about images. This is one of the most versatile multi-modal tasks.

**Types of Questions:**

**Existence:**
- "Is there a person in this image?"
- "Does this image contain a dog?"
- "Are there any cars visible?"

**Counting:**
- "How many people are in this photo?"
- "Count the number of chairs in the room"
- "How many items are on the table?"

**Attribute Recognition:**
- "What color is the car?"
- "What is the person wearing?"
- "What\'s the weather like in this image?"

**Spatial Reasoning:**
- "What's to the left of the building?"
- "Where is the cat sitting?"
- "What's in the background?"

**Activity Recognition:**
- "What is the person doing?"
- "What sport is being played?"
- "What\'s happening in this scene?"

**Complex Reasoning:**
- "Why might the person be smiling?"
- "What time of day is it likely to be?"
- "What's the purpose of this setup?"

### 2. Image Captioning

Generating natural language descriptions of image content.

**Caption Types:**

**Simple Captions:**
- "A dog sitting on grass"
- "A woman riding a bicycle"
- "A sunset over mountains"

**Detailed Captions:**
- "A golden retriever dog with shiny fur sitting on freshly cut green grass in a park, with trees visible in the background on a sunny day"

**Dense Captions:**
- Describing multiple regions and objects
- Relationships between objects
- Detailed attributes of each element

**Structured Captions:**
- JSON format with objects, attributes, locations
- Hierarchical descriptions
- Tagged elements

### 3. Visual Reasoning

Understanding relationships, context, and making inferences from images.

**Types:**
- Spatial relationships (above, below, left, right)
- Temporal reasoning (before/after in image sequences)
- Causal reasoning (what might happen next)
- Contextual understanding (scene type, purpose)
- Common sense reasoning (typical vs unusual)

## Building a VQA System

### Basic Implementation

\`\`\`python
import os
import base64
from typing import Optional
from openai import OpenAI
from PIL import Image
import io

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image_file (image_path: str) -> str:
    """Encode image file to base64."""
    with open (image_path, "rb") as f:
        return base64.b64encode (f.read()).decode('utf-8')

def answer_image_question(
    image_path: str,
    question: str,
    model: str = "gpt-4-vision-preview",
    max_tokens: int = 300
) -> str:
    """
    Answer a question about an image.
    
    Args:
        image_path: Path to image file
        question: Natural language question
        model: Model to use (gpt-4-vision-preview, etc.)
        max_tokens: Maximum tokens in response
    
    Returns:
        Answer to the question
    """
    # Encode image
    base64_image = encode_image_file (image_path)
    
    # Create message with image and question
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=max_tokens,
        temperature=0.0  # Lower temperature for factual questions
    )
    
    return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    answer = answer_image_question(
        "photo.jpg",
        "What color is the car in this image?"
    )
    print(f"Answer: {answer}")
\`\`\`

### Production VQA System

\`\`\`python
import os
import base64
import hashlib
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import json
import redis
from openai import OpenAI
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig (level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VQAResponse:
    """Structured VQA response."""
    answer: str
    confidence: float
    reasoning: Optional[str] = None
    objects_detected: Optional[List[str]] = None
    cached: bool = False
    latency_ms: float = 0

class ProductionVQASystem:
    """Production-ready Visual Question Answering system."""
    
    def __init__(
        self,
        openai_api_key: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        cache_ttl: int = 86400,  # 24 hours
        model: str = "gpt-4-vision-preview"
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        self.cache_ttl = cache_ttl
        self.model = model
    
    def _get_cache_key (self, image_hash: str, question: str) -> str:
        """Generate cache key for image + question pair."""
        question_hash = hashlib.sha256(question.encode()).hexdigest()
        return f"vqa:{image_hash}:{question_hash}"
    
    def _get_image_hash (self, image_bytes: bytes) -> str:
        """Generate hash of image bytes."""
        return hashlib.sha256(image_bytes).hexdigest()
    
    def _optimize_image(
        self,
        image_bytes: bytes,
        max_size: int = 2048,
        quality: int = 85
    ) -> bytes:
        """Optimize image to reduce cost and latency."""
        img = Image.open (io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode in ('RGBA', 'LA'):
                background.paste (img, mask=img.split()[-1])
            img = background
        
        # Resize if too large
        if max (img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to JPEG
        buffer = io.BytesIO()
        img.save (buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue()
    
    def _encode_image (self, image_bytes: bytes) -> str:
        """Encode image bytes to base64."""
        return base64.b64encode (image_bytes).decode('utf-8')
    
    def _build_prompt(
        self,
        question: str,
        require_reasoning: bool = False,
        detect_objects: bool = False
    ) -> str:
        """Build enhanced prompt for better responses."""
        prompt_parts = [question]
        
        if require_reasoning:
            prompt_parts.append(
                "\\n\\nExplain your reasoning step by step."
            )
        
        if detect_objects:
            prompt_parts.append(
                "\\n\\nAlso list any relevant objects you detect in the image."
            )
        
        return " ".join (prompt_parts)
    
    def answer_question(
        self,
        image_bytes: bytes,
        question: str,
        require_reasoning: bool = False,
        detect_objects: bool = False,
        use_cache: bool = True,
        detail_level: str = "auto"
    ) -> VQAResponse:
        """
        Answer a question about an image.
        
        Args:
            image_bytes: Raw image bytes
            question: Natural language question
            require_reasoning: Whether to include reasoning in response
            detect_objects: Whether to detect objects in image
            use_cache: Whether to use cache for responses
            detail_level: 'low', 'high', or 'auto' for cost/quality tradeoff
        
        Returns:
            VQAResponse with answer and metadata
        """
        start_time = datetime.now()
        
        # Get image hash for caching
        image_hash = self._get_image_hash (image_bytes)
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key (image_hash, question)
            cached_response = self.redis_client.get (cache_key)
            if cached_response:
                logger.info (f"Cache hit for question: {question[:50]}...")
                response_dict = json.loads (cached_response)
                return VQAResponse(**response_dict, cached=True)
        
        # Optimize image
        optimized_image = self._optimize_image (image_bytes)
        base64_image = self._encode_image (optimized_image)
        
        # Build prompt
        prompt = self._build_prompt (question, require_reasoning, detect_objects)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": detail_level
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.0
            )
            
            answer = response.choices[0].message.content
            
            # Calculate latency
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Parse response (basic parsing, can be enhanced)
            vqa_response = VQAResponse(
                answer=answer,
                confidence=0.9,  # Could be enhanced with confidence estimation
                reasoning=answer if require_reasoning else None,
                objects_detected=None,  # Could parse from response
                cached=False,
                latency_ms=latency_ms
            )
            
            # Cache response
            if use_cache:
                cache_data = {
                    "answer": answer,
                    "confidence": 0.9,
                    "reasoning": vqa_response.reasoning,
                    "objects_detected": vqa_response.objects_detected,
                    "latency_ms": latency_ms
                }
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps (cache_data)
                )
            
            logger.info(
                f"VQA completed in {latency_ms:.2f}ms - Question: {question[:50]}..."
            )
            
            return vqa_response
        
        except Exception as e:
            logger.error (f"VQA failed: {str (e)}")
            raise

# Example usage
if __name__ == "__main__":
    vqa = ProductionVQASystem(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Read image
    with open("test_image.jpg", "rb") as f:
        image_bytes = f.read()
    
    # Ask multiple questions
    questions = [
        "What is the main subject of this image?",
        "What colors are prominent in this image?",
        "Is this indoors or outdoors?",
        "How many people are visible?",
        "What time of day does this appear to be?"
    ]
    
    for question in questions:
        response = vqa.answer_question(
            image_bytes,
            question,
            require_reasoning=True
        )
        print(f"\\nQ: {question}")
        print(f"A: {response.answer}")
        print(f"Latency: {response.latency_ms:.2f}ms, Cached: {response.cached}")
\`\`\`

## Image Captioning System

### Basic Caption Generator

\`\`\`python
def generate_caption(
    image_path: str,
    caption_style: str = "detailed"
) -> str:
    """
    Generate caption for an image.
    
    Args:
        image_path: Path to image
        caption_style: 'simple', 'detailed', or 'dense'
    
    Returns:
        Generated caption
    """
    prompts = {
        "simple": "Describe this image in one simple sentence.",
        "detailed": "Provide a detailed description of this image, including colors, objects, activities, and setting.",
        "dense": "Provide a comprehensive description of this image, describing all visible objects, their attributes, relationships, and the overall scene in detail."
    }
    
    base64_image = encode_image_file (image_path)
    
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts.get (caption_style, prompts["detailed"])},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    
    return response.choices[0].message.content
\`\`\`

### Advanced Structured Captioning

\`\`\`python
from typing import List, Dict
from pydantic import BaseModel, Field

class ImageObject(BaseModel):
    """Represents an object in the image."""
    name: str = Field (description="Name of the object")
    attributes: List[str] = Field (description="Attributes like color, size, etc.")
    location: str = Field (description="Location in image (e.g., 'left side', 'center', 'background')")
    confidence: str = Field (description="Confidence level: high, medium, or low")

class StructuredCaption(BaseModel):
    """Structured image caption with detailed information."""
    brief_caption: str = Field (description="One sentence summary")
    detailed_caption: str = Field (description="Detailed paragraph description")
    scene_type: str = Field (description="Type of scene: indoor, outdoor, portrait, landscape, etc.")
    dominant_colors: List[str] = Field (description="Main colors in the image")
    objects: List[ImageObject] = Field (description="Main objects in the image")
    mood: str = Field (description="Overall mood or atmosphere")
    activities: List[str] = Field (description="Activities or actions visible")

def generate_structured_caption (image_path: str) -> StructuredCaption:
    """Generate structured caption with detailed breakdown."""
    prompt = """Analyze this image and provide a structured description in the following JSON format:

{
  "brief_caption": "One sentence summary of the image",
  "detailed_caption": "Detailed paragraph describing the image",
  "scene_type": "indoor/outdoor/portrait/landscape/etc",
  "dominant_colors": ["color1", "color2", "color3"],
  "objects": [
    {
      "name": "object name",
      "attributes": ["attribute1", "attribute2"],
      "location": "position in image",
      "confidence": "high/medium/low"
    }
  ],
  "mood": "overall mood or atmosphere",
  "activities": ["activity1", "activity2"]
}

Be thorough and specific in your analysis."""

    base64_image = encode_image_file (image_path)
    
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=800,
        temperature=0.3
    )
    
    # Parse JSON response
    import json
    caption_data = json.loads (response.choices[0].message.content)
    return StructuredCaption(**caption_data)

# Example usage
caption = generate_structured_caption("photo.jpg")
print(f"Brief: {caption.brief_caption}")
print(f"\\nDetailed: {caption.detailed_caption}")
print(f"\\nScene: {caption.scene_type}")
print(f"Colors: {', '.join (caption.dominant_colors)}")
print(f"\\nObjects detected: {len (caption.objects)}")
for obj in caption.objects:
    print(f"  - {obj.name} ({obj.location}): {', '.join (obj.attributes)}")
\`\`\`

## Document Understanding

Visual question answering excels at understanding documents, charts, and diagrams.

### Chart Analysis

\`\`\`python
def analyze_chart (image_path: str) -> Dict[str, Any]:
    """
    Analyze a chart or graph image.
    
    Returns structured information about the chart.
    """
    prompt = """Analyze this chart/graph and extract:

1. Chart type (bar, line, pie, scatter, etc.)
2. Title and axis labels
3. Main data points and values
4. Key trends or patterns
5. Any insights or conclusions

Provide the information in JSON format:
{
  "chart_type": "type",
  "title": "chart title",
  "x_axis": "x-axis label",
  "y_axis": "y-axis label",
  "data_points": [
    {"label": "label1", "value": value1},
    {"label": "label2", "value": value2}
  ],
  "trends": ["trend1", "trend2"],
  "insights": ["insight1", "insight2"]
}"""

    base64_image = encode_image_file (image_path)
    
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"  # Use high detail for charts
                        }
                    }
                ]
            }
        ],
        max_tokens=1000,
        temperature=0.0  # Factual extraction
    )
    
    import json
    return json.loads (response.choices[0].message.content)

# Example
chart_data = analyze_chart("revenue_chart.png")
print(f"Chart Type: {chart_data['chart_type']}")
print(f"Title: {chart_data['title']}")
print(f"\\nData Points:")
for point in chart_data['data_points']:
    print(f"  {point['label']}: {point['value']}")
print(f"\\nKey Insights:")
for insight in chart_data['insights']:
    print(f"  - {insight}")
\`\`\`

### OCR and Text Extraction

\`\`\`python
def extract_text_from_image(
    image_path: str,
    preserve_layout: bool = True
) -> str:
    """
    Extract text from image with OCR.
    
    Args:
        image_path: Path to image with text
        preserve_layout: Whether to preserve spatial layout
    
    Returns:
        Extracted text
    """
    if preserve_layout:
        prompt = """Extract all text from this image, preserving the spatial layout and formatting as much as possible. 
        
Use spaces and line breaks to maintain the relative positions of text elements."""
    else:
        prompt = "Extract all text from this image in reading order."
    
    base64_image = encode_image_file (image_path)
    
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
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
            }
        ],
        max_tokens=1500,
        temperature=0.0
    )
    
    return response.choices[0].message.content

# Extract text from screenshot, receipt, document, etc.
text = extract_text_from_image("receipt.jpg")
print(text)
\`\`\`

### Table Extraction

\`\`\`python
def extract_table_from_image (image_path: str) -> List[Dict[str, Any]]:
    """
    Extract table data from image.
    
    Returns list of dictionaries representing rows.
    """
    prompt = """Extract the table data from this image. 
    
Return the data as a JSON array where each element represents a row:
[
  {"column1": "value1", "column2": "value2", ...},
  {"column1": "value3", "column2": "value4", ...}
]

Use the first row as column headers if present."""

    base64_image = encode_image_file (image_path)
    
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
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
            }
        ],
        max_tokens=2000,
        temperature=0.0
    )
    
    import json
    return json.loads (response.choices[0].message.content)

# Example
table_data = extract_table_from_image("table_screenshot.png")
print(f"Extracted {len (table_data)} rows")
for i, row in enumerate (table_data[:3]):  # Show first 3 rows
    print(f"Row {i+1}: {row}")
\`\`\`

## Multi-Image Analysis

### Comparing Images

\`\`\`python
def compare_images(
    image1_path: str,
    image2_path: str,
    comparison_type: str = "differences"
) -> str:
    """
    Compare two images and describe differences or similarities.
    
    Args:
        image1_path: First image
        image2_path: Second image
        comparison_type: 'differences', 'similarities', or 'both'
    
    Returns:
        Comparison description
    """
    prompts = {
        "differences": "Compare these two images and describe the main differences between them.",
        "similarities": "Compare these two images and describe what they have in common.",
        "both": "Compare these two images, describing both their similarities and differences."
    }
    
    base64_image1 = encode_image_file (image1_path)
    base64_image2 = encode_image_file (image2_path)
    
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts[comparison_type]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image1}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image2}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )
    
    return response.choices[0].message.content

# Compare two product images
comparison = compare_images("product_v1.jpg", "product_v2.jpg", "both")
print(comparison)
\`\`\`

### Image Sequence Analysis

\`\`\`python
def analyze_image_sequence(
    image_paths: List[str],
    analysis_type: str = "story"
) -> str:
    """
    Analyze a sequence of images.
    
    Args:
        image_paths: List of image paths in order
        analysis_type: 'story', 'changes', or 'process'
    
    Returns:
        Analysis of the sequence
    """
    prompts = {
        "story": "Look at this sequence of images and describe the story they tell, in chronological order.",
        "changes": "Analyze this sequence of images and describe how things change from one image to the next.",
        "process": "These images show steps in a process. Describe each step and the overall process."
    }
    
    # Build message content with all images
    content = [{"type": "text", "text": prompts[analysis_type]}]
    
    for image_path in image_paths:
        base64_image = encode_image_file (image_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{"role": "user", "content": content}],
        max_tokens=800
    )
    
    return response.choices[0].message.content

# Analyze a sequence
sequence = ["step1.jpg", "step2.jpg", "step3.jpg", "step4.jpg"]
process_description = analyze_image_sequence (sequence, "process")
print(process_description)
\`\`\`

## Optimizing for Production

### Batch Processing

\`\`\`python
import asyncio
from typing import List, Tuple
import aiohttp

async def batch_vqa(
    image_question_pairs: List[Tuple[str, str]],
    max_concurrent: int = 5
) -> List[str]:
    """
    Process multiple VQA requests concurrently.
    
    Args:
        image_question_pairs: List of (image_path, question) tuples
        max_concurrent: Maximum concurrent requests
    
    Returns:
        List of answers in same order as input
    """
    semaphore = asyncio.Semaphore (max_concurrent)
    
    async def process_one (image_path: str, question: str) -> str:
        async with semaphore:
            # In production, use async OpenAI client
            return answer_image_question (image_path, question)
    
    tasks = [
        process_one (img, q)
        for img, q in image_question_pairs
    ]
    
    return await asyncio.gather(*tasks)

# Process 100 images concurrently
pairs = [
    ("image1.jpg", "What is this?"),
    ("image2.jpg", "What color is dominant?"),
    # ... 98 more
]

answers = asyncio.run (batch_vqa (pairs))
\`\`\`

### Cost Optimization

\`\`\`python
def estimate_vqa_cost(
    image_size_bytes: int,
    detail_level: str = "auto",
    num_questions: int = 1
) -> float:
    """
    Estimate cost of VQA operations.
    
    GPT-4 Vision pricing (example, check current pricing):
    - Low detail: ~85 tokens per image
    - High detail: ~170 tokens per image (1024x1024)
    - Plus regular text token costs
    """
    # Convert image size to dimensions (approximate)
    from PIL import Image
    import io
    
    # Base costs per 1K tokens (example)
    INPUT_COST_PER_1K = 0.01
    OUTPUT_COST_PER_1K = 0.03
    
    # Image tokens
    if detail_level == "low":
        image_tokens = 85
    else:
        # High detail: varies by size, approximately
        image_tokens = 170
    
    # Question tokens (estimate)
    question_tokens = 50 * num_questions
    
    # Answer tokens (estimate)
    answer_tokens = 100 * num_questions
    
    # Calculate cost
    input_cost = (image_tokens + question_tokens) / 1000 * INPUT_COST_PER_1K
    output_cost = answer_tokens / 1000 * OUTPUT_COST_PER_1K
    
    return input_cost + output_cost

# Estimate cost for 1000 images
total_cost = estimate_vqa_cost(500_000, "auto", 1) * 1000
print(f"Estimated cost for 1000 images: \\$\{total_cost:.2f}")
\`\`\`

## Best Practices

### 1. Prompt Engineering for Vision

**Be Specific:**
\`\`\`python
# Bad
"What\'s in this image?"

# Good
"List all visible products in this retail shelf image, including brand names and approximate quantities."
\`\`\`

**Request Structured Output:**
\`\`\`python
# Returns consistent JSON
"Identify the main subject, background elements, and colors in this image. Return as JSON: {subject: str, background: list[str], colors: list[str]}"
    \`\`\`

**Provide Context:**
\`\`\`python
# Context helps accuracy
"This is a medical prescription image. Extract the medication name, dosage, and frequency. Be precise with numbers."
    \`\`\`

### 2. Error Handling

\`\`\`python
def safe_vqa (image_path: str, question: str, max_retries: int = 3) -> Optional[str]:
    """VQA with comprehensive error handling."""
    for attempt in range (max_retries):
        try:
            return answer_image_question (image_path, question)
        except openai.BadRequestError as e:
            if "image" in str (e).lower():
                logger.error (f"Invalid image: {image_path}")
                return None
            raise
        except openai.RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning (f"Rate limited, waiting {wait_time}s")
                time.sleep (wait_time)
                continue
            raise
        except Exception as e:
            logger.error (f"VQA failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise
    
    return None
\`\`\`

### 3. Quality Validation

\`\`\`python
def validate_vqa_response(
    question: str,
    answer: str,
    min_length: int = 10,
    max_length: int = 500
) -> bool:
    """
    Validate VQA response quality.
    
    Returns True if response seems valid.
    """
    # Check length
    if len (answer) < min_length or len (answer) > max_length:
        return False
    
    # Check for common failure patterns
    failure_patterns = [
        "i cannot",
        "i'm unable to",
        "i don't see",
        "no image provided",
        "error"
    ]
    
    answer_lower = answer.lower()
    for pattern in failure_patterns:
        if pattern in answer_lower:
            return False
    
    # For counting questions, check if answer contains a number
    if "how many" in question.lower():
        import re
        if not re.search (r'\\d+', answer):
            return False
    
    return True
\`\`\`

## Real-World Applications

### 1. E-Commerce Product Analysis

\`\`\`python
def analyze_product_image (image_path: str) -> Dict[str, Any]:
    """Comprehensive product analysis for e-commerce."""
    questions = {
        "category": "What category of product is this? (e.g., clothing, electronics, furniture)",
        "color": "What are the main colors of this product?",
        "condition": "Does this product appear new, used, or damaged? Explain.",
        "brand": "Is there a visible brand name or logo? If yes, what is it?",
        "features": "List the key features or characteristics visible in this image.",
        "accessories": "Are there any accessories, parts, or additional items visible?"
    }
    
    results = {}
    for key, question in questions.items():
        results[key] = answer_image_question (image_path, question)
    
    return results
\`\`\`

### 2. Content Moderation

\`\`\`python
def moderate_image_content (image_path: str) -> Dict[str, Any]:
    """Check image for inappropriate content."""
    checks = {
        "inappropriate": "Does this image contain any inappropriate, offensive, or unsafe content? Answer with Yes or No, and explain if Yes.",
        "violence": "Is there any violence or disturbing content in this image?",
        "minors": "Are there any people who appear to be minors in this image?",
        "text": "Is there any visible text that could be offensive or inappropriate?"
    }
    
    results = {}
    for check_name, question in checks.items():
        answer = answer_image_question (image_path, question)
        results[check_name] = {
            "answer": answer,
            "flagged": answer.lower().startswith("yes")
        }
    
    return results
\`\`\`

### 3. Accessibility Alt-Text Generation

\`\`\`python
def generate_alt_text(
    image_path: str,
    context: Optional[str] = None
) -> str:
    """
    Generate accessibility alt-text for images.
    
    Args:
        image_path: Path to image
        context: Optional context (e.g., "product page", "blog post")
    
    Returns:
        Alt-text description
    """
    if context:
        prompt = f"""Generate concise alt-text for this image to be used in a {context}. 
        
The alt-text should:
- Be descriptive but concise (1-2 sentences)
- Focus on the most important elements
- Be useful for screen reader users
- Not start with "Image of" or "Picture of"

Provide only the alt-text, nothing else."""
    else:
        prompt = "Generate concise, descriptive alt-text for this image suitable for screen readers."
    
    return answer_image_question (image_path, prompt)

# Usage
alt_text = generate_alt_text(
    "hero_image.jpg",
    context="homepage hero section"
)
print(f'<img src="hero_image.jpg" alt="{alt_text}">')
\`\`\`

## Summary

Image + text understanding is a cornerstone of multi-modal AI, enabling:

**Core Capabilities:**
- Visual question answering for any question about images
- Automatic image captioning at various detail levels
- Document, chart, and diagram understanding
- OCR and text extraction
- Multi-image analysis and comparison

**Production Considerations:**
- Optimize images to reduce costs (resize, compress)
- Implement caching for repeated queries
- Use structured prompts for consistent output
- Handle errors gracefully with retries
- Validate responses for quality
- Monitor costs and usage

**Best Practices:**
- Be specific in prompts
- Request structured output (JSON)
- Use appropriate detail levels
- Implement batch processing for scale
- Cache aggressively
- Validate outputs

**Applications:**
- E-commerce (product analysis, search)
- Accessibility (alt-text generation)
- Content moderation
- Document processing
- Customer support
- Healthcare (with appropriate disclaimers)

In the next sections, we'll explore video understanding, audio processing, and building complete multi-modal systems.
`,
  codeExamples: [
    {
      title: 'Production VQA System',
      description:
        'Complete visual question answering system with caching, optimization, and error handling',
      language: 'python',
      code: `# See ProductionVQASystem class in content above`,
    },
  ],
  practicalTips: [
    'Always optimize images before sending - resize to 1024-2048px and compress to 85% quality',
    'Use temperature=0.0 for factual questions, 0.3-0.7 for creative descriptions',
    'Cache responses aggressively - most images get asked the same questions repeatedly',
    'Request structured JSON output for consistency and easier parsing',
    "Use 'low' detail for simple tasks (object detection), 'high' for detailed analysis (charts, documents)",
    'Batch similar requests to reduce overhead and latency',
    'Implement exponential backoff for rate limits',
    'Validate responses for common failure patterns before using',
  ],
  quiz: '/quizzes/multi-modal-ai-systems/image-text-understanding',
};
