/**
 * Image Processing for LLMs Section
 * Module 3: File Processing & Document Understanding
 */

export const imageprocessingllmsSection = {
  id: 'image-processing-llms',
  title: 'Image Processing for LLMs',
  content: `# Image Processing for LLMs

Master image processing for building AI applications that can understand and extract information from images using vision-enabled LLMs.

## Overview: Images in LLM Applications

Vision-enabled LLMs like GPT-4V, Claude 3, and Gemini can understand images, making image processing crucial for modern AI applications. From OCR to diagram understanding to screenshot analysis, images unlock new capabilities.

**Use Cases:**
- OCR and text extraction from images
- Screenshot analysis for UI understanding
- Diagram and chart interpretation
- Document scanning and processing
- Visual search and similarity
- Image-based Q&A systems

## Image File Handling Basics

\`\`\`python
# pip install Pillow
from PIL import Image
import os
from pathlib import Path

def read_image_basic(filepath: str):
    """Basic image reading with Pillow."""
    img = Image.open(filepath)
    
    print(f"Format: {img.format}")
    print(f"Size: {img.size}")  # (width, height)
    print(f"Mode: {img.mode}")  # RGB, RGBA, L, etc.
    print(f"Info: {img.info}")
    
    return img

# Usage
img = read_image_basic("document.png")
\`\`\`

## Image Preprocessing for OCR

\`\`\`python
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

def preprocess_for_ocr(image_path: str, output_path: str = None):
    """
    Preprocess image for better OCR results.
    
    Steps: convert to grayscale, increase contrast, denoise.
    """
    img = Image.open(image_path)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    
    # Denoise
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    # Threshold to binary (black and white)
    threshold = 128
    img = img.point(lambda p: 255 if p > threshold else 0)
    
    if output_path:
        img.save(output_path)
    
    return img

# Usage
preprocessed = preprocess_for_ocr("scanned_doc.jpg", "preprocessed.png")
\`\`\`

## OCR with Tesseract

\`\`\`python
# pip install pytesseract
import pytesseract
from PIL import Image

def extract_text_from_image(image_path: str) -> str:
    """Extract text from image using OCR."""
    img = Image.open(image_path)
    
    # Run OCR
    text = pytesseract.image_to_string(img)
    
    return text.strip()

def extract_text_with_confidence(image_path: str):
    """Extract text with confidence scores."""
    img = Image.open(image_path)
    
    # Get detailed data
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    # Extract words with confidence > 60
    results = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60:
            results.append({
                'text': data['text'][i],
                'confidence': data['conf'][i],
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i]
            })
    
    return results

# Usage
text = extract_text_from_image("receipt.jpg")
detailed = extract_text_with_confidence("receipt.jpg")
\`\`\`

## Using Vision LLMs

\`\`\`python
from openai import OpenAI
from anthropic import Anthropic
import base64
from pathlib import Path

def image_to_base64(image_path: str) -> str:
    """Convert image to base64 for API."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def analyze_image_gpt4v(image_path: str, prompt: str) -> str:
    """
    Analyze image using GPT-4V.
    
    Can understand screenshots, diagrams, charts, OCR, etc.
    """
    client = OpenAI()
    
    # Read image as base64
    base64_image = image_to_base64(image_path)
    
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
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def analyze_image_claude(image_path: str, prompt: str) -> str:
    """
    Analyze image using Claude 3.
    
    Excellent for document understanding and analysis.
    """
    client = Anthropic()
    
    # Read image
    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Determine media type
    ext = Path(image_path).suffix.lower()
    media_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    media_type = media_types.get(ext, 'image/jpeg')
    
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )
    
    return message.content[0].text

# Usage Examples
# OCR from image
text = analyze_image_gpt4v("receipt.jpg", "Extract all text from this receipt.")

# Diagram understanding
explanation = analyze_image_claude(
    "architecture_diagram.png",
    "Explain this system architecture diagram in detail."
)

# Chart analysis
analysis = analyze_image_gpt4v(
    "sales_chart.png",
    "Analyze this sales chart and extract the key insights."
)
\`\`\`

## Image Manipulation

\`\`\`python
from PIL import Image, ImageDraw, ImageFont

def resize_image(image_path: str, max_size: tuple = (1024, 1024)) -> Image:
    """Resize image while maintaining aspect ratio."""
    img = Image.open(image_path)
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img

def crop_image(image_path: str, box: tuple) -> Image:
    """
    Crop image to specified box.
    
    box: (left, top, right, bottom)
    """
    img = Image.open(image_path)
    return img.crop(box)

def add_text_to_image(
    image_path: str,
    text: str,
    position: tuple,
    output_path: str
):
    """Add text annotation to image."""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Use default font or load custom
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Add text
    draw.text(position, text, fill="red", font=font)
    
    img.save(output_path)

# Usage
resized = resize_image("large_image.jpg")
cropped = crop_image("screenshot.png", (100, 100, 500, 400))
add_text_to_image("diagram.png", "Important Section", (50, 50), "annotated.png")
\`\`\`

## Screenshot Analysis

\`\`\`python
from PIL import Image
import pytesseract
from typing import Dict, List

def analyze_screenshot(screenshot_path: str) -> Dict:
    """
    Analyze screenshot to extract UI elements and text.
    
    Useful for UI testing, automation, accessibility.
    """
    img = Image.open(screenshot_path)
    
    # Extract text with positions
    ocr_data = pytesseract.image_to_data(
        img,
        output_type=pytesseract.Output.DICT
    )
    
    # Group text by regions
    text_regions = []
    for i in range(len(ocr_data['text'])):
        if ocr_data['text'][i].strip():
            text_regions.append({
                'text': ocr_data['text'][i],
                'confidence': ocr_data['conf'][i],
                'bbox': (
                    ocr_data['left'][i],
                    ocr_data['top'][i],
                    ocr_data['left'][i] + ocr_data['width'][i],
                    ocr_data['top'][i] + ocr_data['height'][i]
                )
            })
    
    return {
        'image_size': img.size,
        'text_regions': text_regions,
        'full_text': pytesseract.image_to_string(img)
    }

# Usage with vision LLM for deeper understanding
def understand_ui_with_llm(screenshot_path: str) -> Dict:
    """Use vision LLM to understand UI structure."""
    
    # First get basic OCR
    basic_analysis = analyze_screenshot(screenshot_path)
    
    # Then use vision LLM for semantic understanding
    prompt = """Analyze this UI screenshot and provide:
    1. Main components (buttons, forms, navigation)
    2. User flow and interactions
    3. Accessibility issues
    4. Improvement suggestions"""
    
    llm_analysis = analyze_image_claude(screenshot_path, prompt)
    
    return {
        'ocr_data': basic_analysis,
        'llm_analysis': llm_analysis
    }
\`\`\`

## Production Image Processor

\`\`\`python
from PIL import Image
import pytesseract
from pathlib import Path
from typing import Dict, Optional, List
import logging
from openai import OpenAI

class ImageProcessor:
    """
    Production-grade image processor for LLM applications.
    
    Handles OCR, preprocessing, and vision LLM integration.
    """
    
    def __init__(self, use_vision_llm: bool = True):
        self.logger = logging.getLogger(__name__)
        self.use_vision_llm = use_vision_llm
        if use_vision_llm:
            self.client = OpenAI()
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process image with OCR and optional vision LLM.
        
        Returns comprehensive analysis.
        """
        path = Path(image_path)
        
        if not path.exists():
            self.logger.error(f"Image not found: {image_path}")
            return {}
        
        try:
            result = {
                'filepath': str(path),
                'filename': path.name,
                'metadata': {},
                'ocr_text': '',
                'ocr_data': [],
                'vision_analysis': ''
            }
            
            # Load image and get metadata
            img = Image.open(image_path)
            result['metadata'] = {
                'format': img.format,
                'size': img.size,
                'mode': img.mode,
                'file_size': path.stat().st_size
            }
            
            # Run OCR
            result['ocr_text'] = self._extract_text_ocr(img)
            result['ocr_data'] = self._extract_detailed_ocr(img)
            
            # Run vision LLM if enabled
            if self.use_vision_llm:
                result['vision_analysis'] = self._analyze_with_vision_llm(image_path)
            
            self.logger.info(f"Processed image: {image_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process image: {e}")
            return {}
    
    def _extract_text_ocr(self, img: Image) -> str:
        """Extract text using OCR."""
        try:
            return pytesseract.image_to_string(img).strip()
        except Exception as e:
            self.logger.warning(f"OCR failed: {e}")
            return ""
    
    def _extract_detailed_ocr(self, img: Image) -> List[Dict]:
        """Extract text with positions and confidence."""
        try:
            data = pytesseract.image_to_data(
                img,
                output_type=pytesseract.Output.DICT
            )
            
            results = []
            for i in range(len(data['text'])):
                if data['text'][i].strip() and int(data['conf'][i]) > 50:
                    results.append({
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]),
                        'bbox': (
                            data['left'][i],
                            data['top'][i],
                            data['width'][i],
                            data['height'][i]
                        )
                    })
            
            return results
        except Exception as e:
            self.logger.warning(f"Detailed OCR failed: {e}")
            return []
    
    def _analyze_with_vision_llm(self, image_path: str) -> str:
        """Analyze image using vision LLM."""
        try:
            base64_image = self._image_to_base64(image_path)
            
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in detail, including any text, objects, layout, and key information."
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
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.warning(f"Vision LLM analysis failed: {e}")
            return ""
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64."""
        import base64
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    def extract_structured_data(
        self,
        image_path: str,
        schema_description: str
    ) -> str:
        """
        Extract structured data from image using vision LLM.
        
        Example: Extract invoice data, form fields, etc.
        """
        if not self.use_vision_llm:
            return ""
        
        try:
            base64_image = self._image_to_base64(image_path)
            
            prompt = f"""Extract structured information from this image.

Schema: {schema_description}

Return the data in JSON format."""
            
            response = self.client.chat.completions.create(
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
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Structured extraction failed: {e}")
            return ""

# Usage Examples
processor = ImageProcessor(use_vision_llm=True)

# Process image with OCR and vision analysis
result = processor.process_image("document.jpg")
print(f"OCR Text: {result['ocr_text']}")
print(f"Vision Analysis: {result['vision_analysis']}")

# Extract structured data (e.g., invoice)
invoice_schema = '''
{
  "invoice_number": "string",
  "date": "string",
  "vendor": "string",
  "total": "number",
  "items": [{"description": "string", "amount": "number"}]
}
'''

invoice_data = processor.extract_structured_data("invoice.jpg", invoice_schema)
print(f"Extracted Invoice Data: {invoice_data}")
\`\`\`

## Key Takeaways

1. **Use Pillow (PIL)** for basic image operations
2. **Preprocess images** for better OCR results
3. **pytesseract** for traditional OCR
4. **Vision LLMs** for semantic understanding
5. **GPT-4V and Claude 3** excel at different tasks
6. **Combine OCR + Vision LLM** for best results
7. **Handle image formats** appropriately (JPEG, PNG, etc.)
8. **Resize images** before sending to APIs (cost/speed)
9. **Extract structured data** using vision LLMs
10. **Screenshot analysis** enables UI automation

These patterns enable building sophisticated image processing pipelines for LLM applications, from document scanning to visual search.`,
  videoUrl: undefined,
};
