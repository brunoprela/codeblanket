export const crossModalGeneration = {
  title: 'Cross-Modal Generation',
  id: 'cross-modal-generation',
  description:
    'Master generating content in one modality from another - text to image, image to music, video to text, and more complex cross-modal pipelines.',
  content: `
# Cross-Modal Generation

## Introduction

Cross-modal generation is the ability to create content in one modality from input in another. This enables powerful applications like generating images from text descriptions, creating music from images, producing videos from scripts, and building complex multi-step generation pipelines.

In this section, we'll explore various cross-modal generation tasks, learn how to chain multiple modalities together, and build production systems that can transform content across modalities.

## Cross-Modal Generation Tasks

### Text → Image

Generate images from textual descriptions.

**Models:**
- DALL-E 3 (OpenAI)
- Stable Diffusion
- Midjourney
- Imagen (Google)

**Use Cases:**
- Marketing content creation
- Product visualization
- Concept art
- Educational illustrations
- Social media graphics

### Image → Text

Generate textual descriptions from images.

**Tasks:**
- Image captioning
- Visual question answering
- OCR and text extraction
- Scene description
- Alt-text generation

### Text → Audio

Generate audio from text.

**Types:**
- Text-to-speech (TTS)
- Text-to-music
- Sound effect generation from descriptions

**Models:**
- OpenAI TTS
- ElevenLabs
- MusicGen
- AudioCraft

### Audio → Text

Generate text from audio.

**Types:**
- Speech-to-text transcription
- Music transcription
- Audio description
- Sentiment analysis from voice

**Models:**
- Whisper (OpenAI)
- Google Speech-to-Text
- AssemblyAI

### Image → Image

Transform images to different styles or variations.

**Types:**
- Style transfer
- Image-to-image translation
- Super-resolution
- Inpainting
- Colorization

### Text → Video

Generate videos from text descriptions.

**Models:**
- Runway Gen-2
- Pika
- Stable Video Diffusion
- AnimateDiff

### Video → Text

Extract information from videos.

**Types:**
- Video summarization
- Action recognition
- Scene description
- Transcription with visual context

## Building Cross-Modal Pipelines

### Text → Image → Text Pipeline

\`\`\`python
from openai import OpenAI
import base64
from PIL import Image
import io
from typing import Dict, Any

client = OpenAI()

def text_to_image_to_text(
    initial_prompt: str,
    analysis_question: str
) -> Dict[str, Any]:
    """
    Generate image from text, then analyze the generated image.
    
    Args:
        initial_prompt: Text prompt for image generation
        analysis_question: Question to ask about generated image
    
    Returns:
        Dictionary with image and analysis
    """
    # Step 1: Generate image from text
    image_response = client.images.generate(
        model="dall-e-3",
        prompt=initial_prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )
    
    image_url = image_response.data[0].url
    
    # Download image
    import requests
    img_data = requests.get (image_url).content
    
    # Save image
    with open("generated_image.png", "wb") as f:
        f.write (img_data)
    
    # Step 2: Analyze the generated image
    base64_image = base64.b64encode (img_data).decode('utf-8')
    
    analysis_response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Question about this image: {analysis_question}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }],
        max_tokens=300
    )
    
    analysis = analysis_response.choices[0].message.content
    
    return {
        "original_prompt": initial_prompt,
        "image_url": "generated_image.png",
        "analysis_question": analysis_question,
        "analysis": analysis,
        "revised_prompt": image_response.data[0].revised_prompt
    }

# Example usage
result = text_to_image_to_text(
    "A futuristic city with flying cars at sunset",
    "Describe the architectural style and mood of this city"
)

print(f"Generated image from: {result['original_prompt']}")
print(f"Analysis: {result['analysis']}")
\`\`\`

### Image → Text → Image Pipeline

Modify images through natural language.

\`\`\`python
def image_edit_pipeline(
    original_image_path: str,
    edit_instruction: str
) -> Dict[str, Any]:
    """
    Analyze image, generate edit instructions, create new image.
    
    Args:
        original_image_path: Path to original image
        edit_instruction: Natural language edit instruction
    
    Returns:
        Results with analysis and new image
    """
    # Step 1: Analyze original image
    with open (original_image_path, "rb") as f:
        image_data = f.read()
    
    base64_image = base64.b64encode (image_data).decode('utf-8')
    
    analysis_response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Analyze this image and then generate a detailed image generation prompt that applies this edit: {edit_instruction}

Provide:
1. Brief description of original image
2. Detailed prompt for generating edited version

Format as JSON:
{{
  "original_description": "...",
  "generation_prompt": "..."
}}"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }],
        max_tokens=400
    )
    
    import json
    response_data = json.loads (analysis_response.choices[0].message.content)
    
    # Step 2: Generate new image based on prompt
    new_image_response = client.images.generate(
        model="dall-e-3",
        prompt=response_data["generation_prompt"],
        size="1024x1024",
        quality="standard"
    )
    
    new_image_url = new_image_response.data[0].url
    
    # Download new image
    new_img_data = requests.get (new_image_url).content
    with open("edited_image.png", "wb") as f:
        f.write (new_img_data)
    
    return {
        "original_description": response_data["original_description"],
        "edit_instruction": edit_instruction,
        "generation_prompt": response_data["generation_prompt"],
        "new_image_path": "edited_image.png"
    }

# Example
result = image_edit_pipeline(
    "room.jpg",
    "Add plants and make it look more cozy"
)

print(f"Edit: {result['edit_instruction']}")
print(f"Generated with prompt: {result['generation_prompt']}")
\`\`\`

### Text → Audio → Text Pipeline

Generate audio, then analyze it.

\`\`\`python
def text_to_speech_to_analysis(
    text: str,
    voice: str = "alloy"
) -> Dict[str, Any]:
    """
    Convert text to speech, then transcribe and analyze.
    
    Args:
        text: Text to convert to speech
        voice: TTS voice to use
    
    Returns:
        Speech file and analysis
    """
    # Step 1: Text to speech
    speech_response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )
    
    speech_response.stream_to_file("speech.mp3")
    
    # Step 2: Transcribe the speech
    with open("speech.mp3", "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    
    # Step 3: Analyze differences
    analysis_prompt = f"""Compare the original text with the transcription:

Original: {text}

Transcription: {transcription.text}

Analysis:
1. Are they identical or different?
2. If different, what changed?
3. Quality assessment of the speech-to-text pipeline

Return as JSON."""

    analysis_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": analysis_prompt}],
        temperature=0.3
    )
    
    return {
        "original_text": text,
        "speech_file": "speech.mp3",
        "transcription": transcription.text,
        "analysis": analysis_response.choices[0].message.content
    }

# Example
result = text_to_speech_to_analysis(
    "The quick brown fox jumps over the lazy dog.",
    voice="nova"
)

print(f"Transcription: {result['transcription']}")
print(f"Analysis: {result['analysis']}")
\`\`\`

### Video → Audio → Text Pipeline

Extract and process audio from video.

\`\`\`python
import subprocess
from pathlib import Path

def video_to_audio_to_text(
    video_path: str
) -> Dict[str, Any]:
    """
    Extract audio from video and transcribe.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Transcription and analysis
    """
    # Step 1: Extract audio from video
    audio_path = "extracted_audio.mp3"
    
    # Use ffmpeg to extract audio
    subprocess.run([
        "ffmpeg",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "mp3",
        "-y",  # Overwrite
        audio_path
    ], check=True)
    
    # Step 2: Transcribe audio
    with open (audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )
    
    # Step 3: Analyze transcript
    analysis_prompt = f"""Analyze this video transcript:

{transcription.text}

Provide:
1. Summary (2-3 sentences)
2. Main topics discussed
3. Key points
4. Tone and style

Return as JSON."""

    analysis_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": analysis_prompt}]
    )
    
    import json
    analysis = json.loads (analysis_response.choices[0].message.content)
    
    return {
        "video_path": video_path,
        "audio_path": audio_path,
        "duration": transcription.duration,
        "language": transcription.language,
        "transcript": transcription.text,
        "analysis": analysis
    }

# Example
result = video_to_audio_to_text("presentation.mp4")
print(f"Duration: {result['duration']}s")
print(f"Summary: {result['analysis']['summary']}")
\`\`\`

## Advanced Cross-Modal Pipelines

### Image → Caption → Image Variations

\`\`\`python
def generate_image_variations_via_text(
    image_path: str,
    num_variations: int = 3
) -> List[str]:
    """
    Generate variations of an image by going through text description.
    
    1. Describe the image
    2. Generate variations of the description
    3. Generate new images from varied descriptions
    
    Args:
        image_path: Path to original image
        num_variations: Number of variations to generate
    
    Returns:
        List of paths to generated variation images
    """
    # Step 1: Describe the original image
    with open (image_path, "rb") as f:
        image_data = f.read()
    
    base64_image = base64.b64encode (image_data).decode('utf-8')
    
    description_response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Provide a detailed description of this image suitable for image generation."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }],
        max_tokens=200
    )
    
    original_description = description_response.choices[0].message.content
    
    # Step 2: Generate variations of the description
    variations_prompt = f"""Given this image description, generate {num_variations} creative variations:

Original: {original_description}

Create variations that maintain the core elements but change:
- Style (e.g., photorealistic, artistic, cartoon)
- Mood or atmosphere
- Time of day or lighting
- Small details

Return as JSON array of strings: ["variation 1", "variation 2", ...]"""

    variations_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": variations_prompt}],
        temperature=0.8
    )
    
    import json
    variations = json.loads (variations_response.choices[0].message.content)
    
    # Step 3: Generate images from variations
    generated_images = []
    
    for i, variation in enumerate (variations):
        image_response = client.images.generate(
            model="dall-e-3",
            prompt=variation,
            size="1024x1024",
            quality="standard"
        )
        
        # Download image
        img_url = image_response.data[0].url
        img_data = requests.get (img_url).content
        
        output_path = f"variation_{i+1}.png"
        with open (output_path, "wb") as f:
            f.write (img_data)
        
        generated_images.append (output_path)
    
    return generated_images

# Generate variations
variations = generate_image_variations_via_text("photo.jpg", num_variations=3)
print(f"Generated {len (variations)} variations")
\`\`\`

### Multi-Step Content Generation

\`\`\`python
from typing import List, Dict, Any

def generate_complete_blog_post(
    topic: str,
    include_images: bool = True
) -> Dict[str, Any]:
    """
    Generate complete blog post with text and images.
    
    Pipeline:
    1. Generate outline
    2. Write each section
    3. Generate relevant images for sections
    4. Compile into complete post
    
    Args:
        topic: Blog post topic
        include_images: Whether to generate images
    
    Returns:
        Complete blog post data
    """
    # Step 1: Generate outline
    outline_prompt = f"""Create a blog post outline for: {topic}

Include:
- Title
- Introduction paragraph
- 3-5 main sections with titles
- Conclusion paragraph

Return as JSON:
{{
  "title": "...",
  "introduction": "...",
  "sections": [
    {{"title": "...", "key_points": ["...", "..."]}}
  ],
  "conclusion": "..."
}}"""

    outline_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": outline_prompt}],
        temperature=0.7
    )
    
    import json
    outline = json.loads (outline_response.choices[0].message.content)
    
    # Step 2: Write each section
    sections_content = []
    
    for section in outline["sections"]:
        section_prompt = f"""Write a detailed blog post section:

Title: {section['title']}
Key points to cover: {', '.join (section['key_points'])}

Write 2-3 paragraphs of engaging, informative content."""

        section_response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": section_prompt}],
            temperature=0.7
        )
        
        section_text = section_response.choices[0].message.content
        
        # Step 3: Generate image for section if requested
        image_path = None
        if include_images:
            image_prompt = f"""Create an illustration for a blog post section titled "{section['title']}" about {topic}. 
            
Style: Clean, professional, informative. Suitable for a blog post."""

            image_response = client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                quality="standard"
            )
            
            img_url = image_response.data[0].url
            img_data = requests.get (img_url).content
            
            image_path = f"section_{len (sections_content) + 1}.png"
            with open (image_path, "wb") as f:
                f.write (img_data)
        
        sections_content.append({
            "title": section['title'],
            "content": section_text,
            "image": image_path
        })
    
    # Step 4: Compile complete post
    complete_post = {
        "title": outline["title"],
        "introduction": outline["introduction"],
        "sections": sections_content,
        "conclusion": outline["conclusion"],
        "metadata": {
            "topic": topic,
            "word_count": sum (len (s["content"].split()) for s in sections_content),
            "num_images": len([s for s in sections_content if s["image"]])
        }
    }
    
    return complete_post

# Generate complete blog post
post = generate_complete_blog_post("The Future of Artificial Intelligence", include_images=True)

print(f"Title: {post['title']}")
print(f"Sections: {len (post['sections'])}")
print(f"Word count: {post['metadata']['word_count']}")
print(f"Images: {post['metadata']['num_images']}")
\`\`\`

### Video Generation from Script

\`\`\`python
def script_to_storyboard_to_images(
    script: str
) -> Dict[str, Any]:
    """
    Convert script to visual storyboard.
    
    Pipeline:
    1. Break script into scenes
    2. Generate scene descriptions
    3. Create images for each scene
    
    Args:
        script: Video script text
    
    Returns:
        Storyboard with images
    """
    # Step 1: Break into scenes
    scenes_prompt = f"""Break this script into visual scenes:

Script:
{script}

For each scene, provide:
- Scene number
- Duration (approximate seconds)
- Visual description
- Key action or focus

Return as JSON array."""

    scenes_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": scenes_prompt}],
        temperature=0.7
    )
    
    import json
    scenes = json.loads (scenes_response.choices[0].message.content)
    
    # Step 2: Generate image for each scene
    storyboard = []
    
    for scene in scenes:
        # Generate detailed visual prompt
        prompt_enhancement = f"""Create a detailed image generation prompt for this scene:

{scene['visual_description']}

Style: Cinematic, professional video production style."""

        prompt_response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt_enhancement}],
            temperature=0.7
        )
        
        enhanced_prompt = prompt_response.choices[0].message.content
        
        # Generate image
        image_response = client.images.generate(
            model="dall-e-3",
            prompt=enhanced_prompt,
            size="1024x1024",
            quality="standard"
        )
        
        img_url = image_response.data[0].url
        img_data = requests.get (img_url).content
        
        image_path = f"scene_{scene['scene_number']}.png"
        with open (image_path, "wb") as f:
            f.write (img_data)
        
        storyboard.append({
            "scene_number": scene['scene_number'],
            "duration": scene['duration'],
            "description": scene['visual_description'],
            "image": image_path
        })
    
    return {
        "original_script": script,
        "num_scenes": len (storyboard),
        "total_duration": sum (s['duration'] for s in storyboard),
        "storyboard": storyboard
    }

# Example
script = """
Scene 1: A busy city street in the morning. People rushing to work.
Scene 2: Close-up of a coffee shop. Steam rising from cups.
Scene 3: Inside the coffee shop. A barista making coffee.
"""

storyboard = script_to_storyboard_to_images (script)
print(f"Generated {storyboard['num_scenes']} scenes")
print(f"Total duration: {storyboard['total_duration']} seconds")
\`\`\`

## Production Cross-Modal System

\`\`\`python
from typing import Optional, Union, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig (level=logging.INFO)
logger = logging.getLogger(__name__)

class Modality(Enum):
    """Supported modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

@dataclass
class GenerationStep:
    """Represents a step in cross-modal pipeline."""
    input_modality: Modality
    output_modality: Modality
    model: str
    parameters: Dict[str, Any]

class CrossModalPipeline:
    """Production cross-modal generation pipeline."""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.steps: List[GenerationStep] = []
        self.results: List[Dict[str, Any]] = []
    
    def add_step(
        self,
        input_modality: Modality,
        output_modality: Modality,
        model: str,
        **parameters
    ):
        """Add a step to the pipeline."""
        step = GenerationStep(
            input_modality=input_modality,
            output_modality=output_modality,
            model=model,
            parameters=parameters
        )
        self.steps.append (step)
    
    def execute (self, initial_input: Union[str, bytes]) -> List[Dict[str, Any]]:
        """
        Execute the pipeline.
        
        Args:
            initial_input: Starting input (text or bytes)
        
        Returns:
            List of results from each step
        """
        self.results = []
        current_input = initial_input
        
        for i, step in enumerate (self.steps):
            logger.info(
                f"Executing step {i+1}/{len (self.steps)}: "
                f"{step.input_modality.value} → {step.output_modality.value}"
            )
            
            try:
                result = self._execute_step (step, current_input)
                self.results.append (result)
                current_input = result["output"]
            
            except Exception as e:
                logger.error (f"Step {i+1} failed: {e}")
                raise
        
        return self.results
    
    def _execute_step(
        self,
        step: GenerationStep,
        input_data: Union[str, bytes]
    ) -> Dict[str, Any]:
        """Execute a single pipeline step."""
        # Determine transformation
        transform = f"{step.input_modality.value}_to_{step.output_modality.value}"
        
        if transform == "text_to_image":
            return self._text_to_image (input_data, step.parameters)
        
        elif transform == "image_to_text":
            return self._image_to_text (input_data, step.parameters)
        
        elif transform == "text_to_audio":
            return self._text_to_audio (input_data, step.parameters)
        
        elif transform == "audio_to_text":
            return self._audio_to_text (input_data, step.parameters)
        
        else:
            raise NotImplementedError (f"Transform {transform} not implemented")
    
    def _text_to_image (self, text: str, params: Dict) -> Dict[str, Any]:
        """Generate image from text."""
        response = self.client.images.generate(
            model=params.get("model", "dall-e-3"),
            prompt=text,
            size=params.get("size", "1024x1024"),
            quality=params.get("quality", "standard")
        )
        
        # Download image
        img_url = response.data[0].url
        import requests
        img_data = requests.get (img_url).content
        
        output_path = params.get("output_path", "generated.png")
        with open (output_path, "wb") as f:
            f.write (img_data)
        
        return {
            "input": text,
            "output": output_path,
            "output_type": "image",
            "metadata": {
                "revised_prompt": response.data[0].revised_prompt
            }
        }
    
    def _image_to_text (self, image_path: str, params: Dict) -> Dict[str, Any]:
        """Analyze image and generate text."""
        with open (image_path, "rb") as f:
            image_data = f.read()
        
        base64_image = base64.b64encode (image_data).decode('utf-8')
        
        prompt = params.get("prompt", "Describe this image in detail.")
        
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
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
            }],
            max_tokens=params.get("max_tokens", 300)
        )
        
        text_output = response.choices[0].message.content
        
        return {
            "input": image_path,
            "output": text_output,
            "output_type": "text"
        }
    
    def _text_to_audio (self, text: str, params: Dict) -> Dict[str, Any]:
        """Generate audio from text."""
        response = self.client.audio.speech.create(
            model=params.get("model", "tts-1"),
            voice=params.get("voice", "alloy"),
            input=text
        )
        
        output_path = params.get("output_path", "speech.mp3")
        response.stream_to_file (output_path)
        
        return {
            "input": text,
            "output": output_path,
            "output_type": "audio"
        }
    
    def _audio_to_text (self, audio_path: str, params: Dict) -> Dict[str, Any]:
        """Transcribe audio to text."""
        with open (audio_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        return {
            "input": audio_path,
            "output": transcription.text,
            "output_type": "text"
        }

# Example: Complex pipeline
pipeline = CrossModalPipeline (openai_api_key=os.getenv("OPENAI_API_KEY"))

# Text → Image → Text → Audio pipeline
pipeline.add_step(
    Modality.TEXT,
    Modality.IMAGE,
    "dall-e-3",
    output_path="step1.png"
)

pipeline.add_step(
    Modality.IMAGE,
    Modality.TEXT,
    "gpt-4-vision",
    prompt="Describe this image poetically"
)

pipeline.add_step(
    Modality.TEXT,
    Modality.AUDIO,
    "tts-1",
    voice="nova",
    output_path="step3.mp3"
)

# Execute pipeline
results = pipeline.execute("A serene mountain landscape at sunrise")

print("Pipeline completed!")
for i, result in enumerate (results):
    print(f"Step {i+1}: {result['output_type']} - {result['output']}")
\`\`\`

## Best Practices

### 1. Quality Control

\`\`\`python
def validate_generation_quality(
    output_path: str,
    output_type: str,
    quality_threshold: float = 0.7
) -> bool:
    """Validate quality of generated content."""
    if output_type == "image":
        # Check image properties
        from PIL import Image
        img = Image.open (output_path)
        
        # Check size
        if img.size[0] < 512 or img.size[1] < 512:
            return False
        
        # Check if image is mostly one color (likely error)
        colors = img.getcolors (img.size[0] * img.size[1])
        if colors and len (colors) < 10:
            return False
    
    elif output_type == "audio":
        # Check audio duration
        from pydub import AudioSegment
        audio = AudioSegment.from_file (output_path)
        
        if len (audio) < 100:  # Less than 0.1 seconds
            return False
    
    return True
\`\`\`

### 2. Cost Monitoring

\`\`\`python
def estimate_pipeline_cost (steps: List[GenerationStep]) -> float:
    """Estimate cost of cross-modal pipeline."""
    cost = 0.0
    
    for step in steps:
        if step.output_modality == Modality.IMAGE:
            # DALL-E 3 cost (example)
            cost += 0.040  # $0.040 per image (standard quality)
        
        elif step.output_modality == Modality.AUDIO:
            # TTS cost (example)
            # Assuming average 100 characters
            cost += 0.0015  # $0.015 per 1K characters
        
        elif step.input_modality == Modality.AUDIO:
            # Whisper cost (example)
            # Assuming 1 minute audio
            cost += 0.006  # $0.006 per minute
        
        elif step.input_modality == Modality.IMAGE:
            # Vision API cost (example)
            cost += 0.01  # Approximate cost
    
    return cost
\`\`\`

### 3. Error Handling

\`\`\`python
def safe_cross_modal_generation(
    pipeline: CrossModalPipeline,
    initial_input: Union[str, bytes],
    max_retries: int = 3
) -> Optional[List[Dict[str, Any]]]:
    """Execute pipeline with error handling."""
    for attempt in range (max_retries):
        try:
            results = pipeline.execute (initial_input)
            
            # Validate each result
            for i, result in enumerate (results):
                if not validate_generation_quality(
                    result["output"],
                    result["output_type"]
                ):
                    raise ValueError (f"Step {i+1} produced low quality output")
            
            return results
        
        except Exception as e:
            logger.warning (f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    
    return None
\`\`\`

## Real-World Applications

### 1. Marketing Content Generation

\`\`\`python
def generate_marketing_content(
    product_description: str,
    target_audience: str
) -> Dict[str, Any]:
    """Generate complete marketing materials."""
    # Generate ad copy
    copy_prompt = f"""Create compelling ad copy for:

Product: {product_description}
Target audience: {target_audience}

Include:
- Headline (catchy, under 10 words)
- Body copy (2-3 sentences)
- Call to action

Return as JSON."""

    copy_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": copy_prompt}]
    )
    
    import json
    copy = json.loads (copy_response.choices[0].message.content)
    
    # Generate product image
    image_prompt = f"""Professional product photograph: {product_description}. 
    Style: Clean, modern, marketing-ready. Studio lighting."""

    image_response = client.images.generate(
        model="dall-e-3",
        prompt=image_prompt,
        size="1024x1024",
        quality="hd"
    )
    
    return {
        "copy": copy,
        "image_url": image_response.data[0].url,
        "product": product_description,
        "audience": target_audience
    }
\`\`\`

### 2. Educational Content Creation

\`\`\`python
def create_educational_content(
    topic: str,
    difficulty: str = "beginner"
) -> Dict[str, Any]:
    """Generate educational content with explanations and visuals."""
    # Generate explanation
    explanation = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Explain {topic} for a {difficulty} level student. Be clear and concise."
        }]
    ).choices[0].message.content
    
    # Generate diagram
    diagram_prompt = f"""Educational diagram illustrating {topic}. 
    Style: Clear, simple, informative, suitable for {difficulty} level."""

    diagram = client.images.generate(
        model="dall-e-3",
        prompt=diagram_prompt
    )
    
    # Generate audio explanation
    audio = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=explanation
    )
    audio.stream_to_file("explanation.mp3")
    
    return {
        "topic": topic,
        "difficulty": difficulty,
        "text_explanation": explanation,
        "diagram_url": diagram.data[0].url,
        "audio_explanation": "explanation.mp3"
    }
\`\`\`

## Summary

Cross-modal generation enables transforming content across modalities:

**Key Capabilities:**
- Text → Image: Generate visuals from descriptions
- Image → Text: Describe and analyze images
- Text → Audio: Convert text to speech or music
- Audio → Text: Transcribe and analyze audio
- Complex pipelines: Chain multiple transformations

**Production Patterns:**
- Build flexible pipelines for multi-step generation
- Validate quality at each step
- Monitor costs across transformations
- Handle errors and retries
- Cache intermediate results

**Best Practices:**
- Validate output quality programmatically
- Estimate costs before execution
- Implement robust error handling
- Use appropriate models for each modality
- Monitor and log pipeline execution
- Cache results where possible

**Applications:**
- Marketing content generation
- Educational materials
- Content repurposing
- Storyboard creation
- Accessibility (alt-text, audio descriptions)
- Creative workflows

Next, we'll explore document intelligence for processing complex documents with mixed content.
`,
  codeExamples: [
    {
      title: 'Production Cross-Modal Pipeline',
      description:
        'Flexible pipeline system for chaining cross-modal transformations',
      language: 'python',
      code: `# See CrossModalPipeline class in content above`,
    },
  ],
  practicalTips: [
    'Always validate quality of generated content programmatically before using',
    'Estimate costs for complete pipelines before execution - multi-step can be expensive',
    'Cache intermediate results to avoid re-generation if pipeline fails',
    "Use appropriate quality settings: 'standard' for prototypes, 'hd' for final output",
    'Implement retry logic for each step independently',
    'Log each transformation for debugging and monitoring',
    "Consider parallel execution when steps don't depend on each other",
    'Use lower quality/cheaper models for testing, upgrade for production',
  ],
  quiz: '/quizzes/multi-modal-ai-systems/cross-modal-generation',
};
