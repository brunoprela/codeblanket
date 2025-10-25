/**
 * Computer Vision with LLMs Section
 * Module 8: Image Generation & Computer Vision
 */

export const computervisionllmsSection = {
  id: 'computer-vision-llms',
  title: 'Computer Vision with LLMs',
  content: `# Computer Vision with LLMs

Master vision-capable LLMs (GPT-4V, Claude 3, Gemini) for image understanding and analysis.

## Overview: LLMs Can See

Vision-capable LLMs can:
- **Describe images**: Generate detailed captions
- **Answer questions**: About image content
- **OCR**: Read text in images
- **Analyze**: Charts, diagrams, documents
- **Detect objects**: Identify and count elements
- **Compare**: Multiple images
- **Extract data**: Structured information from visual content

### Why Vision LLMs Matter

\`\`\`python
capabilities = {
    "traditional_cv": {
        "strengths": "Fast, specialized tasks",
        "examples": ["Object detection", "Face recognition"],
        "limitation": "Fixed task, no reasoning"
    },
    
    "vision_llms": {
        "strengths": "Understanding + reasoning",
        "examples": [
            "Explain what's happening in image",
            "Why is this funny?",
            "What\'s wrong with this diagram?",
            "Extract all text and structure it"
        ],
        "advantage": "General intelligence applied to vision"
    }
}
\`\`\`

## GPT-4 Vision (GPT-4V)

### Basic Usage

\`\`\`python
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO
from typing import Optional, Union
import requests

class GPT4Vision:
    """
    Use GPT-4 Vision for image understanding.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key)
    
    def analyze_image(
        self,
        image: Union[str, Image.Image],
        prompt: str = "What's in this image?",
        max_tokens: int = 300,
        detail: str = "auto"
    ) -> str:
        """
        Analyze an image with GPT-4V.
        
        Args:
            image: Image URL, file path, or PIL Image
            prompt: Question or instruction
            max_tokens: Response length
            detail: 'low', 'high', or 'auto' (affects cost and quality)
        
        Returns:
            Model\'s response
        """
        # Prepare image
        if isinstance (image, str):
            if image.startswith(('http://', 'https://')):
                # URL
                image_input = {"type": "image_url", "image_url": {"url": image, "detail": detail}}
            else:
                # Local file
                image_input = self._encode_local_image (image, detail)
        else:
            # PIL Image
            image_input = self._encode_pil_image (image, detail)
        
        # Call API
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        image_input
                    ]
                }
            ],
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def _encode_local_image (self, filepath: str, detail: str) -> dict:
        """Encode local image to base64."""
        with open (filepath, "rb") as f:
            base64_image = base64.b64encode (f.read()).decode('utf-8')
        
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": detail
            }
        }
    
    def _encode_pil_image (self, image: Image.Image, detail: str) -> dict:
        """Encode PIL Image to base64."""
        buffered = BytesIO()
        image.save (buffered, format="JPEG")
        base64_image = base64.b64encode (buffered.getvalue()).decode('utf-8')
        
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": detail
            }
        }
    
    def compare_images(
        self,
        images: list[Union[str, Image.Image]],
        prompt: str = "Compare these images. What are the differences?"
    ) -> str:
        """Compare multiple images."""
        # Prepare all images
        image_inputs = []
        for img in images:
            if isinstance (img, str) and img.startswith(('http://', 'https://')):
                image_inputs.append({"type": "image_url", "image_url": {"url": img}})
            elif isinstance (img, Image.Image):
                image_inputs.append (self._encode_pil_image (img, "auto"))
            else:
                image_inputs.append (self._encode_local_image (img, "auto"))
        
        # Create message
        content = [{"type": "text", "text": prompt}] + image_inputs
        
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "user", "content": content}],
            max_tokens=500
        )
        
        return response.choices[0].message.content

# Usage
gpt4v = GPT4Vision()

# Analyze image
description = gpt4v.analyze_image(
    image="photo.jpg",
    prompt="Describe this image in detail.",
    detail="high"
)
print(description)

# OCR
text = gpt4v.analyze_image(
    image="document.jpg",
    prompt="Extract all text from this image.",
    max_tokens=1000
)

# Answer questions
answer = gpt4v.analyze_image(
    image="chart.png",
    prompt="What does this chart show? What are the key trends?"
)

# Compare images
comparison = gpt4v.compare_images(
    images=["before.jpg", "after.jpg"],
    prompt="What changed between these two images?"
)
\`\`\`

## Claude 3 Vision

\`\`\`python
from anthropic import Anthropic
import base64

class Claude3Vision:
    """
    Use Claude 3 for vision tasks.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = Anthropic (api_key=api_key)
    
    def analyze_image(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        model: str = "claude-3-opus-20240229",
        max_tokens: int = 1024
    ) -> str:
        """
        Analyze image with Claude 3.
        
        Models:
        - claude-3-opus: Best quality
        - claude-3-sonnet: Balanced
        - claude-3-haiku: Fastest
        """
        # Encode image
        if isinstance (image, str):
            with open (image, "rb") as f:
                image_data = base64.standard_b64encode (f.read()).decode("utf-8")
        else:
            buffered = BytesIO()
            image.save (buffered, format="JPEG")
            image_data = base64.standard_b64encode (buffered.getvalue()).decode("utf-8")
        
        # Call API
        message = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
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

# Usage
claude3 = Claude3Vision()

result = claude3.analyze_image(
    image="diagram.jpg",
    prompt="Explain this diagram step by step.",
    model="claude-3-opus-20240229"
)
\`\`\`

## Practical Applications

### 1. Document Intelligence

\`\`\`python
class DocumentAnalyzer:
    """
    Analyze documents, invoices, receipts, forms.
    """
    
    def __init__(self, vision_model: GPT4Vision):
        self.model = vision_model
    
    def extract_invoice_data (self, image: Union[str, Image.Image]) -> dict:
        """Extract structured data from invoice."""
        prompt = """
        Extract the following information from this invoice:
        - Invoice number
        - Date
        - Vendor name
        - Vendor address
        - Customer name
        - Customer address
        - Line items (description, quantity, price)
        - Subtotal
        - Tax
        - Total
        
        Return as JSON.
        """
        
        response = self.model.analyze_image (image, prompt, max_tokens=1000)
        
        # Parse JSON
        import json
        try:
            data = json.loads (response)
        except:
            data = {"raw_response": response}
        
        return data
    
    def analyze_form (self, image: Union[str, Image.Image]) -> dict:
        """Analyze filled form."""
        prompt = """
        Analyze this form:
        1. List all fields and their values
        2. Identify any missing required fields
        3. Check for any illegible or ambiguous entries
        
        Return as structured JSON.
        """
        
        response = self.model.analyze_image (image, prompt, max_tokens=800)
        return {"analysis": response}
    
    def extract_table_data (self, image: Union[str, Image.Image]) -> str:
        """Extract table to markdown."""
        prompt = """
        Convert the table in this image to markdown format.
        Preserve all rows and columns accurately.
        """
        
        return self.model.analyze_image (image, prompt, max_tokens=1000)

# Usage
doc_analyzer = DocumentAnalyzer (gpt4v)

# Extract invoice
invoice = Image.open("invoice.jpg")
invoice_data = doc_analyzer.extract_invoice_data (invoice)
print(json.dumps (invoice_data, indent=2))

# Extract table
table_markdown = doc_analyzer.extract_table_data("table.png")
print(table_markdown)
\`\`\`

### 2. Chart and Graph Analysis

\`\`\`python
class ChartAnalyzer:
    """
    Analyze charts, graphs, and data visualizations.
    """
    
    def __init__(self, vision_model: GPT4Vision):
        self.model = vision_model
    
    def analyze_chart (self, image: Union[str, Image.Image]) -> dict:
        """Comprehensive chart analysis."""
        prompt = """
        Analyze this chart:
        1. What type of chart is this?
        2. What is it showing? (title, axes, legend)
        3. What are the key data points and trends?
        4. What insights can you draw?
        5. Are there any outliers or notable patterns?
        """
        
        analysis = self.model.analyze_image (image, prompt, max_tokens=600)
        return {"analysis": analysis}
    
    def extract_chart_data (self, image: Union[str, Image.Image]) -> str:
        """Extract numeric data from chart."""
        prompt = """
        Extract all numeric data from this chart.
        List each data series with its values.
        Format as CSV or structured text.
        """
        
        return self.model.analyze_image (image, prompt, max_tokens=800)
    
    def compare_charts(
        self,
        images: list[Union[str, Image.Image]]
    ) -> str:
        """Compare multiple charts."""
        prompt = """
        Compare these charts:
        1. What do they have in common?
        2. What are the key differences?
        3. What trends can you see across all charts?
        4. What conclusions can you draw?
        """
        
        return self.model.compare_images (images, prompt)

# Usage
chart_analyzer = ChartAnalyzer (gpt4v)

# Analyze single chart
chart = Image.open("sales_chart.png")
analysis = chart_analyzer.analyze_chart (chart)
print(analysis["analysis"])

# Extract data
data = chart_analyzer.extract_chart_data (chart)

# Compare quarters
q1_chart = Image.open("q1.png")
q2_chart = Image.open("q2.png")
comparison = chart_analyzer.compare_charts([q1_chart, q2_chart])
\`\`\`

### 3. Visual QA System

\`\`\`python
class VisualQA:
    """
    Question-answering system for images.
    """
    
    def __init__(self, vision_model: GPT4Vision):
        self.model = vision_model
    
    def ask_question(
        self,
        image: Union[str, Image.Image],
        question: str
    ) -> str:
        """Ask any question about image."""
        return self.model.analyze_image (image, question)
    
    def multi_turn_conversation(
        self,
        image: Union[str, Image.Image],
        questions: list[str]
    ) -> list[dict]:
        """Have multi-turn conversation about image."""
        conversation = []
        
        for question in questions:
            answer = self.ask_question (image, question)
            conversation.append({
                "question": question,
                "answer": answer
            })
        
        return conversation
    
    def batch_questions(
        self,
        image: Union[str, Image.Image],
        questions: list[str]
    ) -> dict:
        """Answer multiple questions at once."""
        combined_prompt = "Answer these questions about the image:\n"
        for i, q in enumerate (questions, 1):
            combined_prompt += f"{i}. {q}\n"
        
        response = self.model.analyze_image(
            image,
            combined_prompt,
            max_tokens=1000
        )
        
        return {"questions": questions, "response": response}

# Usage
qa = VisualQA(gpt4v)

image = Image.open("scene.jpg")

# Single question
answer = qa.ask_question (image, "How many people are in this image?")

# Multiple questions
conversation = qa.multi_turn_conversation(
    image,
    questions=[
        "What is the main subject of this image?",
        "What time of day is it?",
        "What\'s the mood or atmosphere?",
        "What details stand out?"
    ]
)

for qa_pair in conversation:
    print(f"Q: {qa_pair['question']}")
    print(f"A: {qa_pair['answer']}\n")
\`\`\`

### 4. Image Validation and Quality Control

\`\`\`python
class ImageValidator:
    """
    Validate images for quality, compliance, appropriateness.
    """
    
    def __init__(self, vision_model: GPT4Vision):
        self.model = vision_model
    
    def check_quality (self, image: Union[str, Image.Image]) -> dict:
        """Check technical quality."""
        prompt = """
        Assess this image's technical quality:
        1. Is it in focus or blurry?
        2. Is the exposure appropriate (not too dark/bright)?
        3. Is the composition good?
        4. Are there any obvious defects or artifacts?
        5. Overall quality rating (1-10)
        
        Return assessment as JSON.
        """
        
        response = self.model.analyze_image (image, prompt)
        return {"quality_check": response}
    
    def check_content (self, image: Union[str, Image.Image]) -> dict:
        """Check content appropriateness."""
        prompt = """
        Analyze this image's content:
        1. What is shown in the image?
        2. Is there any inappropriate content?
        3. Are there any safety or compliance concerns?
        4. Is it suitable for general audiences?
        
        Return analysis as JSON.
        """
        
        response = self.model.analyze_image (image, prompt)
        return {"content_check": response}
    
    def verify_compliance(
        self,
        image: Union[str, Image.Image],
        requirements: list[str]
    ) -> dict:
        """Verify image meets requirements."""
        req_text = "\n".join([f"- {req}" for req in requirements])
        
        prompt = f"""
        Check if this image meets these requirements:
        {req_text}
        
        For each requirement, indicate YES or NO and explain.
        Return as structured format.
        """
        
        response = self.model.analyze_image (image, prompt)
        return {"compliance": response}

# Usage
validator = ImageValidator (gpt4v)

# Quality check
product_photo = Image.open("product.jpg")
quality = validator.check_quality (product_photo)

# Compliance check
requirements = [
    "Product is centered in frame",
    "White background with no shadows",
    "Product takes up at least 80% of image",
    "No watermarks or text overlays",
    "High resolution and sharp focus"
]

compliance = validator.verify_compliance (product_photo, requirements)
print(compliance["compliance"])
\`\`\`

## Production Patterns

\`\`\`python
class ProductionVisionPipeline:
    """
    Production-ready vision LLM pipeline.
    """
    
    def __init__(self):
        self.gpt4v = GPT4Vision()
        self.cache = {}
    
    def analyze_with_retry(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        max_retries: int = 3
    ) -> str:
        """Analyze with automatic retry."""
        import time
        
        for attempt in range (max_retries):
            try:
                return self.gpt4v.analyze_image (image, prompt)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep (wait)
                else:
                    raise
    
    def analyze_with_cache(
        self,
        image_path: str,
        prompt: str
    ) -> str:
        """Use cache to avoid duplicate API calls."""
        import hashlib
        
        # Create cache key
        with open (image_path, 'rb') as f:
            image_hash = hashlib.md5(f.read()).hexdigest()
        
        cache_key = f"{image_hash}:{prompt}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Analyze
        result = self.analyze_with_retry (image_path, prompt)
        
        # Cache
        self.cache[cache_key] = result
        
        return result
    
    def batch_analyze(
        self,
        images: list[str],
        prompt: str,
        parallel: bool = False
    ) -> list[dict]:
        """Analyze multiple images."""
        results = []
        
        if parallel:
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor (max_workers=5) as executor:
                futures = [
                    executor.submit (self.analyze_with_cache, img, prompt)
                    for img in images
                ]
                
                for img, future in zip (images, futures):
                    results.append({
                        "image": img,
                        "analysis": future.result()
                    })
        else:
            for img in images:
                result = self.analyze_with_cache (img, prompt)
                results.append({
                    "image": img,
                    "analysis": result
                })
        
        return results

# Usage
pipeline = ProductionVisionPipeline()

# Batch processing
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = pipeline.batch_analyze(
    images,
    prompt="Describe this image briefly.",
    parallel=True
)
\`\`\`

## Cost Optimization

\`\`\`python
cost_optimization = {
    "image_detail": {
        "low": "$0.01 per image",
        "high": "$0.03 per image",
        "when_to_use_low": [
            "Simple classification",
            "General description",
            "When speed matters"
        ],
        "when_to_use_high": [
            "OCR and text extraction",
            "Detailed analysis",
            "Small text or fine details"
        ]
    },
    
    "strategies": [
        "Use 'low' detail for initial filtering",
        "Upgrade to 'high' only when needed",
        "Cache results to avoid duplicates",
        "Batch similar questions together",
        "Resize images to reduce token usage",
        "Use cheaper models (Claude Haiku) for simple tasks"
    ]
}
\`\`\`

## Key Takeaways

- **Vision LLMs** understand and reason about images
- **GPT-4V** excellent for general vision tasks
- **Claude 3** strong alternative with competitive pricing
- **Use cases**: OCR, charts, documents, QA, validation
- **Detail level** affects cost and quality
- **Multiple images** can be compared in single request
- **Production patterns**: Retry logic, caching, batching
- **Cost optimization**: Use appropriate detail level
- **Structured extraction**: Request JSON for parsing
- **Multi-modal**: Combine vision and text reasoning
`,
};
