export const videoTextUnderstanding = {
  title: 'Video + Text Understanding',
  id: 'video-text-understanding',
  description:
    'Master video analysis, summarization, and temporal reasoning to build systems that can understand and process video content with AI.',
  content: `
# Video + Text Understanding

## Introduction

Video represents the most information-rich modality, combining visual information, temporal dynamics, motion, and often audio. Understanding video

 with AI enables applications ranging from content moderation and automated summarization to action recognition and video search.

In this section, we'll explore how to process videos with modern multi-modal models, extract meaningful insights, and build production systems that can understand video content at scale.

## Why Video Understanding is Hard

###

 1. Temporal Dimension

Videos are sequences of frames, typically 24-60 frames per second:

**Challenges:**
- A 1-minute video at 30fps = 1,800 frames
- Processing every frame is expensive
- Need to understand temporal relationships
- Motion and change over time matter

**Solutions:**
- Frame sampling strategies
- Temporal pooling
- Key frame extraction
- Motion analysis

### 2. Computational Cost

Video processing is expensive:

**Example Costs:**
- 1 frame at 1024x1024 = 1M pixels
- 1 minute video = 1,800 frames
- Total pixels = 1.8 billion pixels

**Cost Implications:**
- Processing time: seconds to minutes per video
- API costs: $0.01-$0.50 per minute of video
- Storage: videos are large files
- Bandwidth: transferring video data

### 3. Modality Complexity

Videos contain multiple types of information:

**Visual:**
- Objects and scenes
- People and faces
- Text overlays
- Visual effects

**Temporal:**
- Actions and activities
- Movement and motion
- Scene transitions
- Event sequences

**Audio (if present):**
- Speech and dialogue
- Music and sound effects
- Ambient sounds
- Audio cues

## Current Model Capabilities

### GPT-4 Vision (GPT-4V)

**Video Processing:**
- Process frames individually
- No native video understanding
- No audio processing
- Manual frame extraction required

**Strategy:**
- Extract key frames
- Analyze frames independently or in sequence
- Aggregate insights
- Infer temporal relationships

**Best For:**
- Scene understanding
- Object tracking across frames
- Visual content analysis
- Frame-by-frame tasks

### Gemini 1.5 Pro

**Video Processing:**
- Native video understanding
- Can process video files directly
- Long context (handles long videos)
- Audio and visual together
- Temporal reasoning built-in

**Capabilities:**
- Video summarization
- Action recognition
- Scene detection
- Audio transcription
- Multi-modal understanding

**Best For:**
- Long-form video analysis
- Videos with audio
- Complex temporal reasoning
- Production video processing

### Claude 3

**Video Processing:**
- No direct video support
- Frame-by-frame processing
- Strong at image understanding
- Good for extracted frames

**Use Cases:**
- Frame analysis
- Scene description
- Visual content extraction
- When combined with frame extraction

## Frame Extraction Strategies

### 1. Uniform Sampling

Extract frames at regular intervals.

\`\`\`python
import cv2
from typing import List
import numpy as np
from pathlib import Path

def extract_uniform_frames(
    video_path: str,
    num_frames: int = 10,
    output_dir: Optional[str] = None
) -> List[np.ndarray]:
    """
    Extract uniformly spaced frames from video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        output_dir: Optional directory to save frames
    
    Returns:
        List of frame arrays
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame indices to extract
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            # Optionally save frame
            if output_dir:
                output_path = Path(output_dir) / f"frame_{idx:05d}.jpg"
                cv2.imwrite(str(output_path), frame)
    
    cap.release()
    
    return frames

# Example usage
frames = extract_uniform_frames("video.mp4", num_frames=16)
print(f"Extracted {len(frames)} frames")
\`\`\`

### 2. Key Frame Detection

Extract frames where significant changes occur.

\`\`\`python
def extract_key_frames(
    video_path: str,
    threshold: float = 30.0,
    max_frames: int = 50
) -> List[np.ndarray]:
    """
    Extract key frames based on scene changes.
    
    Args:
        video_path: Path to video
        threshold: Threshold for scene change detection
        max_frames: Maximum frames to extract
    
    Returns:
        List of key frames
    """
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    prev_frame = None
    frame_count = 0
    
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference from previous frame
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            mean_diff = np.mean(diff)
            
            # If significant change, save frame
            if mean_diff > threshold:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        else:
            # Always include first frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        prev_frame = gray
        frame_count += 1
    
    cap.release()
    
    return frames

# Extract key frames
key_frames = extract_key_frames("video.mp4", threshold=30.0)
print(f"Detected {len(key_frames)} key frames")
\`\`\`

### 3. Smart Sampling

Extract frames at different densities based on content.

\`\`\`python
def smart_frame_extraction(
    video_path: str,
    min_frames: int = 8,
    max_frames: int = 32,
    scene_threshold: float = 30.0
) -> List[tuple[np.ndarray, float]]:
    """
    Extract frames intelligently based on scene changes.
    
    Returns frames with their timestamps.
    """
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # First pass: detect scene changes
    scene_changes = []
    prev_frame = None
    
    for frame_idx in range(0, total_frames, int(fps)):  # Sample every second
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            if np.mean(diff) > scene_threshold:
                timestamp = frame_idx / fps
                scene_changes.append((frame_idx, timestamp))
        
        prev_frame = gray
    
    # Second pass: extract frames around scene changes
    frames_to_extract = set()
    
    # Always include first and last frame
    frames_to_extract.add(0)
    frames_to_extract.add(total_frames - 1)
    
    # Add frames around scene changes
    for frame_idx, _ in scene_changes:
        frames_to_extract.add(frame_idx)
    
    # If not enough frames, add uniform samples
    if len(frames_to_extract) < min_frames:
        uniform_indices = np.linspace(0, total_frames - 1, min_frames, dtype=int)
        frames_to_extract.update(uniform_indices)
    
    # If too many frames, keep most important
    if len(frames_to_extract) > max_frames:
        # Keep frames around scene changes
        sorted_frames = sorted(frames_to_extract)
        frames_to_extract = set(sorted_frames[::len(sorted_frames)//max_frames])
    
    # Extract selected frames
    results = []
    for frame_idx in sorted(frames_to_extract):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = frame_idx / fps
            results.append((frame_rgb, timestamp))
    
    cap.release()
    
    return results

# Smart extraction
frames_with_timestamps = smart_frame_extraction("video.mp4")
for frame, timestamp in frames_with_timestamps:
    print(f"Frame at {timestamp:.2f}s, shape: {frame.shape}")
\`\`\`

## Video Analysis with GPT-4 Vision

### Frame-by-Frame Analysis

\`\`\`python
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
from typing import List, Dict, Any

client = OpenAI()

def encode_frame(frame: np.ndarray) -> str:
    """Encode numpy frame to base64 JPEG."""
    # Convert to PIL Image
    pil_image = Image.fromarray(frame)
    
    # Compress to JPEG
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    
    # Encode to base64
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def analyze_video_frames(
    frames: List[np.ndarray],
    prompt: str,
    model: str = "gpt-4-vision-preview"
) -> str:
    """
    Analyze multiple video frames with a single prompt.
    
    Args:
        frames: List of frame arrays
        prompt: Analysis prompt
        model: Model to use
    
    Returns:
        Analysis text
    """
    # Build message content with all frames
    content = [{"type": "text", "text": prompt}]
    
    for frame in frames:
        base64_frame = encode_frame(frame)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_frame}",
                "detail": "low"  # Use low detail for video frames
            }
        })
    
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": content
        }],
        max_tokens=1000
    )
    
    return response.choices[0].message.content

# Example: Analyze video frames
frames = extract_uniform_frames("video.mp4", num_frames=8)

summary = analyze_video_frames(
    frames,
    """These frames are from a video in chronological order. 
    
Provide:
1. A summary of what happens in the video
2. The main actions or events
3. Any notable changes or transitions
4. The overall context or setting

Be concise but thorough."""
)

print(summary)
\`\`\`

### Video Summarization

\`\`\`python
from typing import Optional
from dataclasses import dataclass

@dataclass
class VideoSummary:
    """Structured video summary."""
    brief_summary: str
    detailed_summary: str
    key_events: List[str]
    scenes: List[Dict[str, Any]]
    people: List[str]
    objects: List[str]
    setting: str
    duration: float

def summarize_video(
    video_path: str,
    num_frames: int = 16
) -> VideoSummary:
    """
    Generate comprehensive video summary.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to analyze
    
    Returns:
        Structured video summary
    """
    # Extract frames
    frames = extract_uniform_frames(video_path, num_frames)
    
    # Get video metadata
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    # Analyze frames
    prompt = f"""These {len(frames)} frames are sampled uniformly from a {duration:.1f}-second video.

Analyze the video and provide a JSON response with:

{{
  "brief_summary": "One sentence summary",
  "detailed_summary": "Detailed paragraph describing the video",
  "key_events": ["event1", "event2", ...],
  "scenes": [
    {{"description": "scene description", "approximate_time": "beginning/middle/end"}}
  ],
  "people": ["description of people present"],
  "objects": ["notable objects"],
  "setting": "description of location/environment"
}}

Be specific and detailed."""

    response_text = analyze_video_frames(frames, prompt)
    
    # Parse JSON response
    import json
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback to basic summary
        data = {
            "brief_summary": response_text[:200],
            "detailed_summary": response_text,
            "key_events": [],
            "scenes": [],
            "people": [],
            "objects": [],
            "setting": "Unknown"
        }
    
    return VideoSummary(
        brief_summary=data.get("brief_summary", ""),
        detailed_summary=data.get("detailed_summary", ""),
        key_events=data.get("key_events", []),
        scenes=data.get("scenes", []),
        people=data.get("people", []),
        objects=data.get("objects", []),
        setting=data.get("setting", ""),
        duration=duration
    )

# Example usage
summary = summarize_video("meeting_recording.mp4", num_frames=16)
print(f"Video Duration: {summary.duration:.1f}s")
print(f"\\nBrief Summary: {summary.brief_summary}")
print(f"\\nDetailed Summary: {summary.detailed_summary}")
print(f"\\nKey Events:")
for event in summary.key_events:
    print(f"  - {event}")
\`\`\`

### Action Recognition

\`\`\`python
def recognize_actions(
    video_path: str,
    num_frames: int = 12
) -> List[Dict[str, Any]]:
    """
    Recognize actions and activities in video.
    
    Returns list of recognized actions with timestamps.
    """
    frames_with_times = smart_frame_extraction(video_path, max_frames=num_frames)
    
    frames = [f[0] for f in frames_with_times]
    timestamps = [f[1] for f in frames_with_times]
    
    prompt = f"""These frames show different moments from a video at these timestamps:
{', '.join([f'{t:.1f}s' for t in timestamps])}

For each frame/timestamp, identify the main action or activity happening. Return as JSON:

[
  {{
    "timestamp": 0.0,
    "action": "description of action",
    "confidence": "high/medium/low",
    "people_involved": 1
  }}
]

Focus on identifying clear actions like walking, talking, sitting, working, etc."""

    response_text = analyze_video_frames(frames, prompt)
    
    import json
    try:
        actions = json.loads(response_text)
        return actions
    except json.JSONDecodeError:
        return []

# Recognize actions
actions = recognize_actions("video.mp4")
for action in actions:
    print(f"{action['timestamp']}s: {action['action']} ({action['confidence']} confidence)")
\`\`\`

## Video Question Answering

\`\`\`python
def answer_video_question(
    video_path: str,
    question: str,
    num_frames: int = 12,
    use_key_frames: bool = True
) -> str:
    """
    Answer a question about a video.
    
    Args:
        video_path: Path to video
        question: Natural language question
        num_frames: Number of frames to analyze
        use_key_frames: Whether to use key frame detection
    
    Returns:
        Answer to question
    """
    # Extract frames
    if use_key_frames:
        frames_data = smart_frame_extraction(video_path, max_frames=num_frames)
        frames = [f[0] for f in frames_data]
        timestamps = [f[1] for f in frames_data]
        
        timestamp_info = f"Frames are from these timestamps: {', '.join([f'{t:.1f}s' for t in timestamps])}"
    else:
        frames = extract_uniform_frames(video_path, num_frames)
        timestamp_info = f"Frames are uniformly sampled from the video"
    
    prompt = f"""These frames are from a video in chronological order. {timestamp_info}

Question: {question}

Analyze the frames and answer the question. Be specific and reference what you see in the frames."""

    return analyze_video_frames(frames, prompt)

# Example questions
questions = [
    "How many people appear in this video?",
    "What is the main activity in this video?",
    "What objects are visible on the table?",
    "Does the person pick up anything? If yes, what?",
    "What room or location is this video recorded in?"
]

for q in questions:
    answer = answer_video_question("video.mp4", q)
    print(f"Q: {q}")
    print(f"A: {answer}\\n")
\`\`\`

## Using Gemini for Native Video Understanding

\`\`\`python
import google.generativeai as genai
import os

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def analyze_video_with_gemini(
    video_path: str,
    prompt: str
) -> str:
    """
    Analyze video using Gemini's native video understanding.
    
    Args:
        video_path: Path to video file
        prompt: Analysis prompt
    
    Returns:
        Analysis text
    """
    # Upload video file
    video_file = genai.upload_file(path=video_path)
    
    # Wait for processing
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)
    
    if video_file.state.name == "FAILED":
        raise Exception("Video processing failed")
    
    # Use Gemini 1.5 Pro for video
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    
    # Generate content
    response = model.generate_content(
        [video_file, prompt],
        request_options={"timeout": 600}
    )
    
    return response.text

# Example usage
summary = analyze_video_with_gemini(
    "meeting.mp4",
    """Analyze this meeting recording and provide:
    
1. Summary of discussion topics
2. Key decisions made
3. Action items mentioned
4. Participants and their roles

Be detailed and specific."""
)

print(summary)
\`\`\`

### Video Search with Gemini

\`\`\`python
def search_video_content(
    video_path: str,
    search_query: str
) -> List[Dict[str, Any]]:
    """
    Search for specific content in video.
    
    Returns timestamps where the query content appears.
    """
    prompt = f"""Search this video for: "{search_query}"

Provide timestamps and descriptions of when this appears. Return as JSON:

[
  {{
    "timestamp": "MM:SS",
    "description": "what's happening at this time",
    "relevance": "high/medium/low"
  }}
]

If the content doesn't appear in the video, return an empty array."""

    result_text = analyze_video_with_gemini(video_path, prompt)
    
    import json
    try:
        results = json.loads(result_text)
        return results
    except json.JSONDecodeError:
        return []

# Search video
results = search_video_content("lecture.mp4", "mentions of machine learning")
print(f"Found {len(results)} mentions:")
for result in results:
    print(f"  {result['timestamp']}: {result['description']}")
\`\`\`

## Production Video Processing System

\`\`\`python
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import redis
import hashlib
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VideoAnalysisResult:
    """Result of video analysis."""
    video_id: str
    summary: str
    key_points: List[str]
    timestamps: List[Dict[str, Any]]
    duration: float
    frame_count: int
    processing_time: float
    cached: bool = False

class ProductionVideoAnalyzer:
    """Production-ready video analysis system."""
    
    def __init__(
        self,
        openai_api_key: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        cache_ttl: int = 86400
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.cache_ttl = cache_ttl
    
    def _get_video_hash(self, video_path: str) -> str:
        """Generate hash of video file."""
        with open(video_path, 'rb') as f:
            # Hash first and last 1MB to identify video
            first_chunk = f.read(1024 * 1024)
            f.seek(-1024 * 1024, 2)
            last_chunk = f.read(1024 * 1024)
            
            combined = first_chunk + last_chunk
            return hashlib.sha256(combined).hexdigest()
    
    def _get_cache_key(self, video_hash: str, task: str) -> str:
        """Generate cache key for video analysis."""
        return f"video_analysis:{video_hash}:{task}"
    
    def analyze(
        self,
        video_path: str,
        task: str = "summary",
        num_frames: int = 16,
        use_cache: bool = True
    ) -> VideoAnalysisResult:
        """
        Analyze video with caching and optimization.
        
        Args:
            video_path: Path to video file
            task: Analysis task (summary, actions, qa, etc.)
            num_frames: Number of frames to analyze
            use_cache: Whether to use cache
        
        Returns:
            VideoAnalysisResult
        """
        start_time = datetime.now()
        
        # Get video hash
        video_hash = self._get_video_hash(video_path)
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(video_hash, f"{task}_{num_frames}")
            cached_result = self.redis_client.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit for video {video_hash[:8]}")
                result_dict = json.loads(cached_result)
                return VideoAnalysisResult(**result_dict, cached=True)
        
        # Extract frames
        frames = extract_uniform_frames(video_path, num_frames)
        
        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        # Analyze based on task
        if task == "summary":
            prompt = """Analyze these video frames and provide:

1. Brief summary (2-3 sentences)
2. Key points or events (list)
3. Important timestamps and what happens

Return as JSON."""
            
            analysis = analyze_video_frames(frames, prompt)
        
        # Parse results (simplified)
        try:
            data = json.loads(analysis)
            summary = data.get("summary", analysis[:200])
            key_points = data.get("key_points", [])
            timestamps = data.get("timestamps", [])
        except:
            summary = analysis
            key_points = []
            timestamps = []
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = VideoAnalysisResult(
            video_id=video_hash,
            summary=summary,
            key_points=key_points,
            timestamps=timestamps,
            duration=duration,
            frame_count=num_frames,
            processing_time=processing_time,
            cached=False
        )
        
        # Cache result
        if use_cache:
            result_dict = {
                "video_id": result.video_id,
                "summary": result.summary,
                "key_points": result.key_points,
                "timestamps": result.timestamps,
                "duration": result.duration,
                "frame_count": result.frame_count,
                "processing_time": result.processing_time
            }
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(result_dict)
            )
        
        logger.info(
            f"Video analysis completed in {processing_time:.2f}s "
            f"({num_frames} frames from {duration:.1f}s video)"
        )
        
        return result

# Usage
analyzer = ProductionVideoAnalyzer(openai_api_key=os.getenv("OPENAI_API_KEY"))

result = analyzer.analyze(
    "video.mp4",
    task="summary",
    num_frames=12
)

print(f"Video Duration: {result.duration:.1f}s")
print(f"Processing Time: {result.processing_time:.2f}s")
print(f"Cached: {result.cached}")
print(f"\\nSummary: {result.summary}")
\`\`\`

## Best Practices

### 1. Frame Selection

**For Short Videos (< 1 min):**
- Use 8-16 uniformly sampled frames
- Low detail level to reduce cost

**For Medium Videos (1-5 min):**
- Use key frame detection
- 16-32 frames
- Focus on scene changes

**For Long Videos (> 5 min):**
- Segment into chunks
- Process each chunk separately
- Aggregate results

### 2. Cost Optimization

\`\`\`python
def estimate_video_analysis_cost(
    duration_seconds: float,
    num_frames: int,
    detail_level: str = "low"
) -> float:
    """Estimate cost of video analysis."""
    # Token costs (example)
    tokens_per_frame = 85 if detail_level == "low" else 170
    input_cost_per_1k = 0.01
    output_cost_per_1k = 0.03
    
    # Calculate
    image_tokens = tokens_per_frame * num_frames
    text_tokens = 500  # Prompt + response
    
    input_cost = image_tokens / 1000 * input_cost_per_1k
    output_cost = text_tokens / 1000 * output_cost_per_1k
    
    return input_cost + output_cost

# Example: 2-minute video
cost = estimate_video_analysis_cost(120, num_frames=16, detail_level="low")
print(f"Estimated cost: \${cost:.4f}")
\`\`\`

### 3. Error Handling

\`\`\`python
def safe_video_analysis(video_path: str, max_retries: int = 3):
    """Video analysis with error handling."""
    for attempt in range(max_retries):
        try:
            # Check file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")
            
            # Check file size (max 100MB for processing)
            file_size = os.path.getsize(video_path)
            if file_size > 100 * 1024 * 1024:
                logger.warning(f"Large video file: {file_size / 1024 / 1024:.1f}MB")
            
            # Attempt analysis
            frames = extract_uniform_frames(video_path, num_frames=12)
            
            if len(frames) == 0:
                raise ValueError("No frames extracted from video")
            
            return analyze_video_frames(frames, "Summarize this video")
        
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    
    return None
\`\`\`

## Real-World Applications

### 1. Content Moderation

\`\`\`python
def moderate_video_content(video_path: str) -> Dict[str, Any]:
    """Check video for inappropriate content."""
    frames = extract_uniform_frames(video_path, num_frames=20)
    
    prompt = """Review these frames for content moderation. Check for:

1. Inappropriate or explicit content
2. Violence or disturbing imagery
3. Hate symbols or offensive gestures
4. Illegal activities

Return JSON:
{
  "is_safe": true/false,
  "issues_found": [],
  "severity": "none/low/medium/high",
  "flagged_timestamps": []
}"""

    result = analyze_video_frames(frames, prompt)
    return json.loads(result)
\`\`\`

### 2. Video Search

\`\`\`python
def create_video_search_index(video_path: str) -> List[Dict[str, Any]]:
    """Create searchable index of video content."""
    frames_with_times = smart_frame_extraction(video_path, max_frames=30)
    
    index = []
    for frame, timestamp in frames_with_times:
        # Describe frame
        description = analyze_video_frames(
            [frame],
            "Describe what's visible in this frame in detail."
        )
        
        index.append({
            "timestamp": timestamp,
            "description": description,
            "frame_index": len(index)
        })
    
    return index
\`\`\`

### 3. Automated Transcription Summarization

\`\`\`python
def summarize_video_with_transcript(
    video_path: str,
    transcript: str
) -> str:
    """Summarize video using both visual and transcript."""
    frames = extract_uniform_frames(video_path, num_frames=12)
    
    prompt = f"""Analyze this video along with its transcript.

Transcript:
{transcript}

Provide a comprehensive summary that incorporates both what is seen in the frames and what is said in the transcript."""

    return analyze_video_frames(frames, prompt)
\`\`\`

## Summary

Video + text understanding enables powerful applications:

**Key Techniques:**
- Frame extraction (uniform, key frames, smart sampling)
- Multi-frame analysis with GPT-4 Vision
- Native video understanding with Gemini
- Temporal reasoning and action recognition

**Production Considerations:**
- Extract 8-32 frames depending on video length
- Use low detail level for cost optimization
- Implement aggressive caching
- Handle errors and retries
- Monitor processing costs

**Applications:**
- Video summarization
- Content moderation
- Video search and indexing
- Action recognition
- Quality control
- Training and education

**Best Practices:**
- Choose frame count based on video length
- Use key frame detection for long videos
- Combine with audio transcription for complete understanding
- Cache results extensively
- Process videos asynchronously
- Provide progress indicators for users

Next, we'll explore audio + text processing for speech recognition, transcription, and audio understanding.
`,
  codeExamples: [
    {
      title: 'Smart Video Frame Extraction',
      description:
        'Intelligently extract frames from videos based on scene changes',
      language: 'python',
      code: `# See smart_frame_extraction function in content above`,
    },
  ],
  practicalTips: [
    'Extract 8-16 frames for videos under 1 minute, 16-32 for longer videos',
    "Use 'low' detail level for video frames - high detail is rarely needed and much more expensive",
    'Implement key frame detection for long videos to focus on scene changes',
    'Cache analysis results aggressively - video hashing is fast and saves significant costs',
    "Process videos asynchronously - analysis takes time, don't block user interactions",
    'Combine frame analysis with audio transcription for complete understanding',
    'Use Gemini 1.5 Pro for native video understanding when available',
    'Consider segmenting very long videos (>5 minutes) and processing chunks separately',
  ],
  quiz: '/quizzes/multi-modal-ai-systems/video-text-understanding',
};
