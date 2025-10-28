export const textToSpeechElevenlabs = {
  title: 'Text-to-Speech (ElevenLabs)',
  id: 'text-to-speech-elevenlabs',
  content: `
# Text-to-Speech (ElevenLabs)

## Introduction

**Text-to-Speech (TTS)** technology has advanced dramatically, with AI models now producing voices indistinguishable from human speech. **ElevenLabs** leads the industry with:

- **Ultra-realistic voices**: Natural intonation, emotion, and prosody
- **Voice cloning**: Create custom voices from short audio samples
- **Multilingual**: 29+ languages with accurate accents
- **Emotion control**: Adjust tone, stability, and clarity
- **Real-time streaming**: Low-latency speech generation
- **Voice library**: Pre-made professional voices

**Use Cases:**
- Audiobook narration
- Video voiceovers and narration
- Virtual assistants and chatbots
- Accessibility (text-to-speech for visually impaired)
- Content localization
- Gaming and animation
- Podcast generation

---

## How Modern TTS Works

### Architecture

Modern neural TTS systems use deep learning:

1. **Text Processing**: Normalize and phonetize text
2. **Prosody Prediction**: Determine rhythm, stress, intonation
3. **Acoustic Model**: Generate mel spectrograms
4. **Vocoder**: Convert spectrograms to audio waveforms

\`\`\`python
"""
Production ElevenLabs TTS Integration
"""

import requests
import os
from typing import Optional, List, Dict, BinaryIO
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import time
import json

class Model(Enum):
    """ElevenLabs TTS models"""
    MULTILINGUAL_V2 = "eleven_multilingual_v2"
    MULTILINGUAL_V1 = "eleven_multilingual_v1"
    MONOLINGUAL_V1 = "eleven_monolingual_v1"
    TURBO_V2 = "eleven_turbo_v2"  # Fastest, lowest latency

@dataclass
class VoiceSettings:
    """Voice generation settings"""
    stability: float = 0.5  # 0-1, higher = more consistent
    similarity_boost: float = 0.75  # 0-1, higher = closer to original
    style: float = 0.0  # 0-1, how much style to apply
    use_speaker_boost: bool = True  # Enhance clarity

@dataclass
class Voice:
    """Voice information"""
    voice_id: str
    name: str
    category: str  # "premade", "cloned", "generated"
    description: Optional[str] = None
    labels: Optional[Dict] = None
    samples: Optional[List[str]] = None

class ElevenLabsTTS:
    """
    Production-ready ElevenLabs Text-to-Speech client
    
    Features:
    - Voice generation with customization
    - Voice cloning from samples
    - Streaming for low latency
    - Multi-language support
    - Cost tracking
    - Result caching
    """
    
    BASE_URL = "https://api.elevenlabs.io/v1"
    
    # Pricing (check latest)
    COST_PER_CHARACTER = {
        "free": 0,  # 10,000 chars/month
        "starter": 0.00018,  # $5/month base
        "creator": 0.00015,  # Better rate
        "pro": 0.00012,  # Best rate
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        plan: str = "starter",
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize ElevenLabs client
        
        Args:
            api_key: ElevenLabs API key
            plan: Your subscription plan (for cost tracking)
            cache_dir: Where to cache generated audio
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY must be set")
        
        self.plan = plan
        self.cost_per_char = self.COST_PER_CHARACTER[plan]
        self.total_cost = 0.0
        self.total_characters = 0
        
        self.cache_dir = cache_dir or Path("./elevenlabs_cache")
        self.cache_dir.mkdir (exist_ok=True)
        
        self.headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
    
    def get_voices (self) -> List[Voice]:
        """
        Get available voices
        
        Returns:
            List of Voice objects
        """
        response = requests.get(
            f"{self.BASE_URL}/voices",
            headers=self.headers
        )
        response.raise_for_status()
        
        data = response.json()
        
        voices = []
        for v in data["voices"]:
            voices.append(Voice(
                voice_id=v["voice_id"],
                name=v["name"],
                category=v["category"],
                description=v.get("description"),
                labels=v.get("labels"),
                samples=v.get("samples"),
            ))
        
        return voices
    
    def generate(
        self,
        text: str,
        voice_id: str,
        model: Model = Model.MULTILINGUAL_V2,
        voice_settings: Optional[VoiceSettings] = None,
        output_path: Optional[Path] = None,
    ) -> bytes:
        """
        Generate speech from text
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use
            model: TTS model
            voice_settings: Voice customization
            output_path: Where to save audio (optional)
        
        Returns:
            Audio bytes (MP3)
        """
        # Default settings
        if voice_settings is None:
            voice_settings = VoiceSettings()
        
        # Build request
        payload = {
            "text": text,
            "model_id": model.value,
            "voice_settings": {
                "stability": voice_settings.stability,
                "similarity_boost": voice_settings.similarity_boost,
                "style": voice_settings.style,
                "use_speaker_boost": voice_settings.use_speaker_boost,
            }
        }
        
        # Make request
        print(f"Generating speech ({len (text)} characters)...")
        start_time = time.time()
        
        response = requests.post(
            f"{self.BASE_URL}/text-to-speech/{voice_id}",
            headers=self.headers,
            json=payload,
        )
        response.raise_for_status()
        
        audio_bytes = response.content
        generation_time = time.time() - start_time
        
        # Track costs
        cost = len (text) * self.cost_per_char
        self.total_characters += len (text)
        self.total_cost += cost
        
        print(f"✅ Generated in {generation_time:.2f}s")
        print(f"   Cost: \${cost:.4f} (Total: \\$\{self.total_cost:.4f})")
        
        # Save if path provided
if output_path:
    output_path = Path (output_path)
output_path.parent.mkdir (parents = True, exist_ok = True)

with open (output_path, "wb") as f:
f.write (audio_bytes)

print(f"💾 Saved: {output_path}")

return audio_bytes
    
    def stream_generate(
    self,
    text: str,
    voice_id: str,
    model: Model = Model.TURBO_V2,  # Use fastest for streaming
        voice_settings: Optional[VoiceSettings] = None,
    ):
"""
        Generate speech with streaming (lower latency)
        
        Yields audio chunks as they're generated

Args:
text: Text to convert
voice_id: Voice to use
model: TTS model (use TURBO for best streaming)
    voice_settings: Voice settings

Yields:
            Audio chunks (bytes)
"""
if voice_settings is None:
voice_settings = VoiceSettings()

payload = {
    "text": text,
    "model_id": model.value,
    "voice_settings": {
        "stability": voice_settings.stability,
        "similarity_boost": voice_settings.similarity_boost,
        "style": voice_settings.style,
        "use_speaker_boost": voice_settings.use_speaker_boost,
    }
}

response = requests.post(
    f"{self.BASE_URL}/text-to-speech/{voice_id}/stream",
    headers = self.headers,
    json = payload,
    stream = True,
)
response.raise_for_status()
        
        # Stream chunks
for chunk in response.iter_content (chunk_size = 4096):
    if chunk:
        yield chunk
    
    def clone_voice(
            self,
            name: str,
            audio_files: List[Path],
            description: Optional[str] = None,
        ) -> str:
"""
        Clone voice from audio samples

Args:
name: Name for cloned voice
            audio_files: List of audio file paths(25MB max total)
description: Optional description

Returns:
            Voice ID of cloned voice
"""
        # Prepare files
files = []
for i, audio_file in enumerate (audio_files):
    files.append((
        'files',
        (audio_file.name, open (audio_file, 'rb'), 'audio/mpeg')
    ))
        
        # Build request
data = { 'name': name }
if description:
    data['description'] = description
        
        # Upload and clone
print(f"Cloning voice from {len (audio_files)} samples...")

response = requests.post(
    f"{self.BASE_URL}/voices/add",
    headers = { "xi-api-key": self.api_key },
    data = data,
    files = files,
)
response.raise_for_status()
        
        # Close files
for _, (_, file_obj, _) in files:
    file_obj.close()

voice_id = response.json()["voice_id"]

print(f"✅ Voice cloned: {voice_id}")

return voice_id
    
    def get_voice_settings (self, voice_id: str) -> VoiceSettings:
"""Get default settings for voice"""
response = requests.get(
    f"{self.BASE_URL}/voices/{voice_id}/settings",
    headers = self.headers,
)
response.raise_for_status()

data = response.json()

return VoiceSettings(
    stability = data["stability"],
    similarity_boost = data["similarity_boost"],
    style = data.get("style", 0.0),
    use_speaker_boost = data.get("use_speaker_boost", True),
)
    
    def batch_generate(
    self,
    texts: List[str],
    voice_id: str,
    output_dir: Path,
    model: Model = Model.MULTILINGUAL_V2,
    voice_settings: Optional[VoiceSettings] = None,
) -> List[Path]:
"""
        Generate speech for multiple texts
        
        Args:
    texts: List of texts
voice_id: Voice to use
output_dir: Output directory
model: TTS model
voice_settings: Voice settings

Returns:
            List of output file paths
"""
output_dir = Path (output_dir)
output_dir.mkdir (parents = True, exist_ok = True)

output_paths = []

for i, text in enumerate (texts):
    print(f"\\n[{i+1}/{len (texts)}] Generating...")

output_path = output_dir / f"speech_{i+1:03d}.mp3"

try:
self.generate(
    text = text,
    voice_id = voice_id,
    model = model,
    voice_settings = voice_settings,
    output_path = output_path,
)
output_paths.append (output_path)
                
            except Exception as e:
print(f"❌ Failed: {e}")
output_paths.append(None)

return output_paths
    
    def get_usage (self) -> Dict:
"""
        Get API usage statistics

Returns:
            Dict with usage info
"""
response = requests.get(
    f"{self.BASE_URL}/user",
    headers = self.headers,
)
response.raise_for_status()

return response.json()

# Advanced features
class TTSPipeline:
"""
    Complete TTS pipeline with preprocessing
    """
    
    def __init__(self, tts_client: ElevenLabsTTS):
self.tts = tts_client
    
    def generate_from_long_text(
    self,
    text: str,
    voice_id: str,
    output_path: Path,
    chunk_size: int = 5000,  # Characters per chunk
):
"""
        Generate from long text by chunking
        
        ElevenLabs has character limits per request
"""
        # Split into sentences
import re
        sentences = re.split (r'(?<=[.!?])\\s+', text)
        
        # Group into chunks
chunks = []
current_chunk = []
current_length = 0

for sentence in sentences:
    if current_length + len (sentence) > chunk_size and current_chunk:
chunks.append(" ".join (current_chunk))
current_chunk = [sentence]
current_length = len (sentence)
            else:
current_chunk.append (sentence)
current_length += len (sentence)

if current_chunk:
    chunks.append(" ".join (current_chunk))

print(f"Split into {len (chunks)} chunks")
        
        # Generate each chunk
audio_chunks = []
for i, chunk in enumerate (chunks):
    print(f"\\nChunk {i+1}/{len (chunks)}")

audio_bytes = self.tts.generate(
    text = chunk,
    voice_id = voice_id,
)
audio_chunks.append (audio_bytes)
        
        # Concatenate audio
        from pydub import AudioSegment
        
        combined = AudioSegment.empty()
for audio_bytes in audio_chunks:
    segment = AudioSegment.from_mp3(BytesIO(audio_bytes))
combined += segment
        
        # Export
combined.export (output_path, format = "mp3")

print(f"✅ Combined audio saved: {output_path}")
    
    def generate_with_emotion(
    self,
    text: str,
    voice_id: str,
    emotion: str,  # "neutral", "happy", "sad", "angry", etc.
    output_path: Path,
):
"""
        Generate with specific emotion
        
        Adjust voice settings to convey emotion
"""
        # Emotion presets
emotion_settings = {
    "neutral": VoiceSettings(
        stability = 0.5,
        similarity_boost = 0.75,
        style = 0.0,
    ),
    "happy": VoiceSettings(
        stability = 0.3,  # Less stable = more variation
                similarity_boost = 0.7,
        style = 0.4,  # More style
    ),
    "sad": VoiceSettings(
        stability = 0.7,  # More stable = less energy
                similarity_boost = 0.8,
        style = 0.2,
    ),
    "angry": VoiceSettings(
        stability = 0.2,  # Highly variable
                similarity_boost = 0.6,
        style = 0.6,  # Strong style
    ),
    "excited": VoiceSettings(
        stability = 0.2,
        similarity_boost = 0.7,
        style = 0.5,
    ),
}

settings = emotion_settings.get (emotion, VoiceSettings())

self.tts.generate(
    text = text,
    voice_id = voice_id,
    voice_settings = settings,
    output_path = output_path,
)

# Production examples
def production_examples():
"""Real-world ElevenLabs usage"""
    
    # Initialize client
tts = ElevenLabsTTS(
    api_key = os.getenv("ELEVENLABS_API_KEY"),
    plan = "starter",
)
    
    # Example 1: List available voices
print("\\n=== Example 1: Available Voices ===")
voices = tts.get_voices()

print(f"Found {len (voices)} voices:\\n")
for voice in voices[: 5]:  # Show first 5
print(f"  {voice.name} ({voice.voice_id})")
if voice.labels:
    print(f"    Labels: {voice.labels}")
    
    # Pick a voice
narrator_voice = voices[0].voice_id
    
    # Example 2: Simple generation
print("\\n=== Example 2: Simple Generation ===")

text = """
    Welcome to this AI - generated narration. 
    ElevenLabs produces incredibly realistic speech 
    that's nearly indistinguishable from human voices.
"""

tts.generate(
    text = text,
    voice_id = narrator_voice,
    output_path = "simple_narration.mp3",
)
    
    # Example 3: Custom voice settings
print("\\n=== Example 3: Custom Settings ===")

custom_settings = VoiceSettings(
    stability = 0.7,  # More consistent
        similarity_boost = 0.85,  # Very close to original
        style = 0.2,
    use_speaker_boost = True,
)

tts.generate(
    text = "This narration uses custom voice settings for optimal quality.",
    voice_id = narrator_voice,
    voice_settings = custom_settings,
    output_path = "custom_voice.mp3",
)
    
    # Example 4: Streaming (low latency)
print("\\n=== Example 4: Streaming ===")

with open("streamed_audio.mp3", "wb") as f:
for chunk in tts.stream_generate(
    text = "This is streamed audio with lower latency.",
    voice_id = narrator_voice,
    model = Model.TURBO_V2,
):
    f.write (chunk)

print("✅ Streamed audio saved")
    
    # Example 5: Voice cloning
print("\\n=== Example 5: Voice Cloning ===")
    
    # Prepare audio samples (need 1 - 3 minutes of clean audio)
sample_files = [
    Path("voice_sample_1.mp3"),
    Path("voice_sample_2.mp3"),
    Path("voice_sample_3.mp3"),
]

if all (f.exists() for f in sample_files):
    cloned_voice_id = tts.clone_voice(
        name = "My Custom Voice",
        audio_files = sample_files,
        description = "Cloned from training samples",
    )
        
        # Use cloned voice
tts.generate(
    text = "This is spoken in the cloned voice.",
    voice_id = cloned_voice_id,
    output_path = "cloned_voice_speech.mp3",
)
    
    # Example 6: Batch generation
print("\\n=== Example 6: Batch Generation ===")

scripts = [
    "Chapter one: The Beginning.",
    "Chapter two: The Journey.",
    "Chapter three: The Climax.",
    "Chapter four: The Resolution.",
    "Chapter five: The End.",
]

output_paths = tts.batch_generate(
    texts = scripts,
    voice_id = narrator_voice,
    output_dir = Path("audiobook_chapters"),
)

print(f"\\n✅ Generated {len (output_paths)} audio files")
    
    # Example 7: Long text (automatic chunking)
print("\\n=== Example 7: Long Text ===")

pipeline = TTSPipeline (tts)

long_text = """
[Your very long text here - could be an entire article, chapter, or script]
""" * 100  # Simulate long text

pipeline.generate_from_long_text(
    text = long_text,
    voice_id = narrator_voice,
    output_path = "long_narration.mp3",
    chunk_size = 5000,
)
    
    # Example 8: Emotion control
print("\\n=== Example 8: Emotion Control ===")

emotions = ["neutral", "happy", "sad", "angry", "excited"]

for emotion in emotions:
    pipeline.generate_with_emotion(
        text = f"This sentence is spoken with {emotion} emotion.",
        voice_id = narrator_voice,
        emotion = emotion,
        output_path = f"emotion_{emotion}.mp3",
    )
    
    # Show usage stats
print("\\n=== Usage Statistics ===")
usage = tts.get_usage()
print(json.dumps (usage, indent = 2))

print(f"\\n💰 Session cost: \\$\{tts.total_cost:.4f}")
print(f"   Characters processed: {tts.total_characters:,}")

if __name__ == "__main__":
    production_examples()
\`\`\`

---

## Alternative TTS Services

### Comparison

| Service | Quality | Languages | Cloning | Price | Best For |
|---------|---------|-----------|---------|-------|----------|
| **ElevenLabs** | ⭐⭐⭐⭐⭐ | 29+ | ✅ | $$$ | Professional content |
| **Play.ht** | ⭐⭐⭐⭐ | 140+ | ✅ | $$ | Multilingual |
| **Murf.ai** | ⭐⭐⭐⭐ | 20+ | ❌ | $$ | Business videos |
| **Amazon Polly** | ⭐⭐⭐ | 30+ | ❌ | $ | High volume |
| **Google Cloud TTS** | ⭐⭐⭐ | 40+ | ❌ | $ | Integration |
| **Azure TTS** | ⭐⭐⭐ | 100+ | ❌ | $ | Enterprise |

---

## Best Practices

### 1. Voice Selection

- **Narration**: Choose clear, authoritative voices
- **Conversational**: Natural, friendly tones
- **Characters**: Distinct voices for each character
- **Audiobooks**: Engaging but not distracting

### 2. Text Preparation

\`\`\`python
"""
Text preprocessing for better TTS
"""

import re

def preprocess_for_tts (text: str) -> str:
    """
    Prepare text for optimal TTS output
    """
    # Remove unwanted characters
    text = re.sub (r'[\\x00-\\x1f\\x7f-\\x9f]', ', text)
    
    # Fix common issues
    text = text.replace('&', 'and')
    text = text.replace('@', 'at')
    text = text.replace('#', 'number')
    
    # Expand contractions
    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace (contraction, expansion)
    
    # Add pauses with punctuation
    text = re.sub (r'([.!?])\\s+', r'\\1  ', text)  # Double space for longer pause
    
    # Remove extra whitespace
    text = re.sub (r'\\s+', ' ', text)
    
    return text.strip()

# Usage
raw_text = "Your text here..."
processed = preprocess_for_tts (raw_text)
tts.generate (processed, voice_id="...")
\`\`\`

### 3. Quality Control

\`\`\`python
"""
Quality checks for generated audio
"""

def check_audio_quality (audio_path: Path) -> Dict:
    """
    Analyze generated audio quality
    """
    from pydub import AudioSegment
    
    audio = AudioSegment.from_file (audio_path)
    
    return {
        "duration_seconds": len (audio) / 1000,
        "channels": audio.channels,
        "frame_rate": audio.frame_rate,
        "sample_width": audio.sample_width,
        "loudness": audio.dBFS,  # Average loudness
        "max_loudness": audio.max_dBFS,
    }
\`\`\`

---

## Summary

**Key Takeaways:**
- ElevenLabs provides industry-leading voice quality
- Voice cloning enables custom voices from samples
- Streaming reduces latency for real-time applications
- Cost scales with character count
- Proper text preprocessing improves output quality

**Production Checklist:**
- ✅ Choose appropriate voice for content type
- ✅ Preprocess text for optimal results
- ✅ Use streaming for real-time applications
- ✅ Implement error handling and retries
- ✅ Track costs and usage
- ✅ Cache generated audio when possible

**Next Steps:**
- Integrate with video generation pipeline
- Build automated voiceover system
- Experiment with emotion control
- Create custom cloned voices for brand consistency
`,
  exercises: [
    {
      title: 'Exercise 1: Automated Audiobook Generator',
      id: 'text-to-speech-elevenlabs',
      difficulty: 'advanced' as const,
      description:
        'Build a system that converts ebooks to audiobooks with chapter detection, multiple narrators for dialogue, and proper pacing.',
      hints: [
        'Parse ebook formats (EPUB, PDF)',
        'Detect chapter boundaries',
        'Identify dialogue vs narration',
        'Use different voices for characters',
        'Add appropriate pauses',
      ],
    },
    {
      title: 'Exercise 2: Real-time Voice Assistant',
      id: 'text-to-speech-elevenlabs',
      difficulty: 'intermediate' as const,
      description:
        'Create a voice assistant that uses Whisper for speech-to-text and ElevenLabs for text-to-speech with streaming for low latency.',
      hints: [
        'Implement streaming for both directions',
        'Add wake word detection',
        'Minimize latency with TURBO model',
        'Handle overlapping speech',
        'Add conversation context',
      ],
    },
  ],
};
