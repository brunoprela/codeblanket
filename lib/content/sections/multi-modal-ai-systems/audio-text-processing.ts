export const audioTextProcessing = {
  title: 'Audio + Text Processing',
  id: 'audio-text-processing',
  description:
    'Master speech recognition, audio transcription, and building systems that can understand and process audio with AI.',
  content: `
# Audio + Text Processing

## Introduction

Audio represents a critical modality for many applications, from meeting transcription and podcast processing to voice assistants and accessibility tools. Modern AI systems can transcribe speech with high accuracy, understand speaker intent, analyze sentiment, and even generate natural-sounding speech.

In this section, we'll explore audio-to-text transcription with Whisper, text-to-speech synthesis, audio analysis, and building production audio processing systems.

## Audio Processing Fundamentals

### Audio File Formats

Common audio formats and their characteristics:

**WAV (Waveform Audio File Format):**
- Uncompressed
- Large file sizes
- High quality
- Good for processing

**MP3 (MPEG Audio Layer 3):**
- Compressed (lossy)
- Smaller file sizes
- Good quality
- Universal support

**M4A/AAC:**
- Compressed (lossy)
- Better than MP3 at same bitrate
- Apple ecosystem

**FLAC:**
- Compressed (lossless)
- Medium file sizes
- Perfect quality
- Archival use

**OGG/Opus:**
- Compressed (lossy)
- Excellent for speech
- Low latency
- Open source

### Audio Properties

**Sample Rate:**
- 16kHz: Telephone quality, sufficient for speech
- 44.1kHz: CD quality
- 48kHz: Professional audio
- Higher rates for specialized applications

**Bit Depth:**
- 8-bit: Low quality
- 16-bit: CD quality, sufficient for most uses
- 24-bit: Professional audio
- 32-bit: Maximum quality

**Channels:**
- Mono (1 channel): Speech, podcasts
- Stereo (2 channels): Music, ambient recordings
- Multi-channel: Surround sound

## Speech-to-Text with Whisper

### Whisper Overview

OpenAI's Whisper is a robust speech recognition model:

**Capabilities:**
- Multi-language (99+ languages)
- High accuracy
- Handles accents and noise
- Automatic punctuation
- Timestamp generation
- Speaker diarization (with additional tools)

**Models:**
- **tiny**: Fastest, least accurate
- **base**: Good balance
- **small**: Better accuracy
- **medium**: High accuracy
- **large**: Best accuracy, slower

### Basic Transcription

\`\`\`python
import openai
from pathlib import Path

client = openai.OpenAI()

def transcribe_audio(
    audio_path: str,
    model: str = "whisper-1",
    language: Optional[str] = None
) -> str:
    """
    Transcribe audio file to text.
    
    Args:
        audio_path: Path to audio file
        model: Whisper model to use
        language: Optional language code (e.g., 'en', 'es')
    
    Returns:
        Transcribed text
    """
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            language=language
        )
    
    return transcript.text

# Example usage
text = transcribe_audio("meeting.mp3")
print(text)
\`\`\`

### Transcription with Timestamps

\`\`\`python
from typing import List, Dict, Any

def transcribe_with_timestamps(
    audio_path: str,
    response_format: str = "verbose_json"
) -> Dict[str, Any]:
    """
    Transcribe audio with word-level timestamps.
    
    Args:
        audio_path: Path to audio file
        response_format: 'json', 'verbose_json', 'srt', 'vtt'
    
    Returns:
        Dictionary with transcript and timing information
    """
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format=response_format,
            timestamp_granularities=["word", "segment"]
        )
    
    return transcript

# Get detailed transcription
result = transcribe_with_timestamps("podcast.mp3")

print(f"Duration: {result.duration}s")
print(f"Language: {result.language}")
print(f"\\nTranscript: {result.text}")
print(f"\\nSegments with timestamps:")

for segment in result.segments:
    start = segment['start']
    end = segment['end']
    text = segment['text']
    print(f"[{start:.2f}s - {end:.2f}s]: {text}")
\`\`\`

### Translation

Whisper can also translate audio to English:

\`\`\`python
def translate_audio_to_english(audio_path: str) -> str:
    """
    Translate audio in any language to English text.
    
    Args:
        audio_path: Path to audio file in any language
    
    Returns:
        English translation
    """
    with open(audio_path, "rb") as audio_file:
        translation = client.audio.translations.create(
            model="whisper-1",
            file=audio_file
        )
    
    return translation.text

# Translate Spanish audio to English
english_text = translate_audio_to_english("spanish_podcast.mp3")
print(english_text)
\`\`\`

## Production Transcription System

\`\`\`python
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import redis
import hashlib
import json
from pydub import AudioSegment
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TranscriptionSegment:
    """Represents a transcription segment with timing."""
    start: float
    end: float
    text: str
    confidence: Optional[float] = None

@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    text: str
    language: str
    duration: float
    segments: List[TranscriptionSegment]
    word_count: int
    processing_time: float
    cached: bool = False

class ProductionTranscriptionSystem:
    """Production-ready audio transcription system."""
    
    def __init__(
        self,
        openai_api_key: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        cache_ttl: int = 86400 * 7,  # 7 days
        max_file_size_mb: int = 25
    ):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.cache_ttl = cache_ttl
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
    
    def _get_audio_hash(self, audio_path: str) -> str:
        """Generate hash of audio file."""
        with open(audio_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def _get_cache_key(self, audio_hash: str, language: Optional[str]) -> str:
        """Generate cache key."""
        lang_suffix = language or "auto"
        return f"transcription:{audio_hash}:{lang_suffix}"
    
    def _convert_to_mp3(self, audio_path: str, output_path: str) -> str:
        """Convert audio to MP3 if needed."""
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Resample if needed (16kHz is sufficient for speech)
        if audio.frame_rate > 16000:
            audio = audio.set_frame_rate(16000)
        
        # Export as MP3
        audio.export(output_path, format="mp3", bitrate="64k")
        
        return output_path
    
    def _split_audio(
        self,
        audio_path: str,
        chunk_length_ms: int = 10 * 60 * 1000  # 10 minutes
    ) -> List[str]:
        """Split large audio files into chunks."""
        audio = AudioSegment.from_file(audio_path)
        
        chunks = []
        for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
            chunk = audio[start:start + chunk_length_ms]
            chunk_path = f"chunk_{i}.mp3"
            chunk.export(chunk_path, format="mp3")
            chunks.append(chunk_path)
        
        return chunks
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        use_cache: bool = True,
        include_timestamps: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe audio file with caching and optimization.
        
        Args:
            audio_path: Path to audio file
            language: Optional language code
            use_cache: Whether to use cached results
            include_timestamps: Whether to include timing information
        
        Returns:
            TranscriptionResult
        """
        start_time = datetime.now()
        
        # Validate file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Get audio hash for caching
        audio_hash = self._get_audio_hash(audio_path)
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(audio_hash, language)
            cached_result = self.redis_client.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit for audio {audio_hash[:8]}")
                result_dict = json.loads(cached_result)
                
                # Reconstruct segments
                segments = [
                    TranscriptionSegment(**seg)
                    for seg in result_dict['segments']
                ]
                
                return TranscriptionResult(
                    text=result_dict['text'],
                    language=result_dict['language'],
                    duration=result_dict['duration'],
                    segments=segments,
                    word_count=result_dict['word_count'],
                    processing_time=result_dict['processing_time'],
                    cached=True
                )
        
        # Check file size
        file_size = os.path.getsize(audio_path)
        
        if file_size > self.max_file_size_bytes:
            logger.warning(
                f"Large audio file ({file_size / 1024 / 1024:.1f}MB), "
                "splitting into chunks..."
            )
            chunks = self._split_audio(audio_path)
            
            # Transcribe each chunk
            all_segments = []
            time_offset = 0
            
            for chunk_path in chunks:
                chunk_result = self._transcribe_file(
                    chunk_path,
                    language,
                    include_timestamps
                )
                
                # Adjust timestamps
                for seg in chunk_result.segments:
                    seg.start += time_offset
                    seg.end += time_offset
                    all_segments.append(seg)
                
                time_offset += chunk_result.duration
                
                # Clean up chunk file
                os.remove(chunk_path)
            
            # Combine results
            full_text = " ".join([seg.text for seg in all_segments])
            total_duration = time_offset
        
        else:
            # Transcribe single file
            result = self._transcribe_file(audio_path, language, include_timestamps)
            full_text = result.text
            total_duration = result.duration
            all_segments = result.segments
        
        # Calculate metrics
        word_count = len(full_text.split())
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Detect language if not specified
        detected_language = language or all_segments[0].language if all_segments else "en"
        
        final_result = TranscriptionResult(
            text=full_text,
            language=detected_language,
            duration=total_duration,
            segments=all_segments,
            word_count=word_count,
            processing_time=processing_time,
            cached=False
        )
        
        # Cache result
        if use_cache:
            result_dict = {
                'text': final_result.text,
                'language': final_result.language,
                'duration': final_result.duration,
                'segments': [
                    {
                        'start': seg.start,
                        'end': seg.end,
                        'text': seg.text,
                        'confidence': seg.confidence
                    }
                    for seg in final_result.segments
                ],
                'word_count': final_result.word_count,
                'processing_time': final_result.processing_time
            }
            
            cache_key = self._get_cache_key(audio_hash, language)
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(result_dict)
            )
        
        logger.info(
            f"Transcription completed in {processing_time:.2f}s "
            f"({word_count} words from {total_duration:.1f}s audio)"
        )
        
        return final_result
    
    def _transcribe_file(
        self,
        audio_path: str,
        language: Optional[str],
        include_timestamps: bool
    ) -> TranscriptionResult:
        """Internal method to transcribe a single file."""
        with open(audio_path, "rb") as audio_file:
            if include_timestamps:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
                
                segments = [
                    TranscriptionSegment(
                        start=seg['start'],
                        end=seg['end'],
                        text=seg['text']
                    )
                    for seg in response.segments
                ]
            else:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language
                )
                
                segments = []
            
            return TranscriptionResult(
                text=response.text,
                language=response.language if hasattr(response, 'language') else 'en',
                duration=response.duration if hasattr(response, 'duration') else 0,
                segments=segments,
                word_count=len(response.text.split()),
                processing_time=0,
                cached=False
            )

# Usage
transcriber = ProductionTranscriptionSystem(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

result = transcriber.transcribe(
    "meeting.mp3",
    language="en",
    include_timestamps=True
)

print(f"Transcribed {result.word_count} words in {result.processing_time:.2f}s")
print(f"Duration: {result.duration:.1f}s")
print(f"Language: {result.language}")
print(f"\\nTranscript: {result.text}")
\`\`\`

## Text-to-Speech (TTS)

### OpenAI TTS

\`\`\`python
def text_to_speech(
    text: str,
    voice: str = "alloy",
    model: str = "tts-1",
    output_path: str = "speech.mp3"
) -> str:
    """
    Convert text to speech.
    
    Args:
        text: Text to convert
        voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        model: Model to use (tts-1 or tts-1-hd)
        output_path: Where to save audio file
    
    Returns:
        Path to generated audio file
    """
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text
    )
    
    response.stream_to_file(output_path)
    
    return output_path

# Generate speech
audio_file = text_to_speech(
    "Hello! This is a test of the text-to-speech system.",
    voice="nova",
    model="tts-1-hd"
)

print(f"Audio saved to: {audio_file}")
\`\`\`

### Streaming TTS

\`\`\`python
def stream_text_to_speech(
    text: str,
    voice: str = "alloy"
):
    """Stream TTS output as it's generated."""
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )
    
    # Stream to output
    for chunk in response.iter_bytes(chunk_size=1024):
        # Process chunk (e.g., send to client, play audio)
        yield chunk

# Stream audio
for audio_chunk in stream_text_to_speech("Hello world!"):
    pass  # Handle audio chunk
\`\`\`

## Audio Analysis

### Speaker Diarization

\`\`\`python
from pyannote.audio import Pipeline

def identify_speakers(
    audio_path: str,
    auth_token: str
) -> List[Dict[str, Any]]:
    """
    Identify different speakers in audio.
    
    Requires pyannote.audio and HuggingFace token.
    
    Args:
        audio_path: Path to audio file
        auth_token: HuggingFace auth token
    
    Returns:
        List of speaker segments
    """
    # Load diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=auth_token
    )
    
    # Run diarization
    diarization = pipeline(audio_path)
    
    # Extract segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    
    return segments

# Combine with transcription
def transcribe_with_speakers(
    audio_path: str,
    hf_token: str
) -> List[Dict[str, Any]]:
    """Transcribe audio with speaker labels."""
    # Get transcription with timestamps
    result = transcriber.transcribe(audio_path, include_timestamps=True)
    
    # Get speaker segments
    speakers = identify_speakers(audio_path, hf_token)
    
    # Match transcription segments with speakers
    labeled_segments = []
    
    for trans_seg in result.segments:
        # Find overlapping speaker
        for spk_seg in speakers:
            if (trans_seg.start >= spk_seg['start'] and 
                trans_seg.start < spk_seg['end']):
                labeled_segments.append({
                    "start": trans_seg.start,
                    "end": trans_seg.end,
                    "text": trans_seg.text,
                    "speaker": spk_seg['speaker']
                })
                break
    
    return labeled_segments

# Usage
segments = transcribe_with_speakers("meeting.mp3", hf_token=HF_TOKEN)

for seg in segments:
    print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker']}: {seg['text']}")
\`\`\`

### Audio Classification

\`\`\`python
def classify_audio_content(audio_path: str) -> Dict[str, Any]:
    """Classify audio content type."""
    # First transcribe
    result = transcriber.transcribe(audio_path)
    
    # Use LLM to classify
    classification_prompt = f"""Analyze this audio transcript and classify it:

Transcript: {result.text}

Classify:
1. Content type (meeting, podcast, lecture, conversation, etc.)
2. Tone (formal, casual, professional, etc.)
3. Topics discussed (list main topics)
4. Number of speakers (estimate)
5. Language proficiency (native, intermediate, beginner)

Return as JSON."""

    from openai import OpenAI
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": classification_prompt}],
        temperature=0.3
    )
    
    import json
    return json.loads(response.choices[0].message.content)

# Classify audio
classification = classify_audio_content("audio.mp3")
print(json.dumps(classification, indent=2))
\`\`\`

### Sentiment Analysis

\`\`\`python
def analyze_audio_sentiment(audio_path: str) -> Dict[str, Any]:
    """Analyze sentiment in audio transcript."""
    # Transcribe
    result = transcriber.transcribe(audio_path, include_timestamps=True)
    
    # Analyze sentiment for each segment
    sentiments = []
    
    for segment in result.segments:
        # Use LLM for sentiment
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"""Analyze sentiment of this text: "{segment.text}"
                
                Return JSON: {{"sentiment": "positive/negative/neutral", "confidence": 0.0-1.0}}"""
            }],
            temperature=0.0
        )
        
        import json
        sentiment_data = json.loads(response.choices[0].message.content)
        
        sentiments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "sentiment": sentiment_data['sentiment'],
            "confidence": sentiment_data['confidence']
        })
    
    # Calculate overall sentiment
    positive = sum(1 for s in sentiments if s['sentiment'] == 'positive')
    negative = sum(1 for s in sentiments if s['sentiment'] == 'negative')
    neutral = sum(1 for s in sentiments if s['sentiment'] == 'neutral')
    
    total = len(sentiments)
    overall = "positive" if positive > negative else "negative" if negative > positive else "neutral"
    
    return {
        "overall_sentiment": overall,
        "sentiment_distribution": {
            "positive": positive / total,
            "negative": negative / total,
            "neutral": neutral / total
        },
        "segments": sentiments
    }

# Analyze sentiment
sentiment_analysis = analyze_audio_sentiment("call.mp3")
print(f"Overall: {sentiment_analysis['overall_sentiment']}")
print(f"Distribution: {sentiment_analysis['sentiment_distribution']}")
\`\`\`

## Podcast Processing

\`\`\`python
from typing import Optional

def process_podcast(
    audio_path: str,
    generate_summary: bool = True,
    generate_chapters: bool = True,
    extract_quotes: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive podcast processing.
    
    Args:
        audio_path: Path to podcast audio
        generate_summary: Whether to generate summary
        generate_chapters: Whether to detect chapters
        extract_quotes: Whether to extract notable quotes
    
    Returns:
        Complete podcast analysis
    """
    # Transcribe
    result = transcriber.transcribe(audio_path, include_timestamps=True)
    
    podcast_data = {
        "transcript": result.text,
        "duration": result.duration,
        "word_count": result.word_count
    }
    
    # Generate summary
    if generate_summary:
        summary_prompt = f"""Summarize this podcast transcript in 2-3 paragraphs:

{result.text}

Focus on main points, key insights, and conclusions."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        podcast_data['summary'] = response.choices[0].message.content
    
    # Generate chapters
    if generate_chapters:
        chapters_prompt = f"""Analyze this podcast and identify main chapters/sections.

{result.text}

Return JSON array:
[
  {{"title": "Chapter title", "timestamp": "MM:SS", "description": "Brief description"}}
]"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": chapters_prompt}]
        )
        
        import json
        podcast_data['chapters'] = json.loads(response.choices[0].message.content)
    
    # Extract quotes
    if extract_quotes:
        quotes_prompt = f"""Extract 5-10 notable or insightful quotes from this transcript:

{result.text}

Return as JSON array of strings."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": quotes_prompt}]
        )
        
        podcast_data['quotes'] = json.loads(response.choices[0].message.content)
    
    return podcast_data

# Process podcast
podcast_analysis = process_podcast("podcast.mp3")

print(f"Duration: {podcast_analysis['duration']:.1f}s")
print(f"\\nSummary: {podcast_analysis['summary']}")
print(f"\\nChapters:")
for chapter in podcast_analysis['chapters']:
    print(f"  {chapter['timestamp']}: {chapter['title']}")
print(f"\\nNotable Quotes:")
for quote in podcast_analysis['quotes']:
    print(f'  - "{quote}"')
\`\`\`

## Real-World Applications

### 1. Meeting Transcription and Summarization

\`\`\`python
def process_meeting(audio_path: str) -> Dict[str, Any]:
    """Process meeting recording."""
    result = transcriber.transcribe(audio_path, include_timestamps=True)
    
    analysis_prompt = f"""Analyze this meeting transcript and extract:

1. Meeting summary (2-3 sentences)
2. Key discussion points (list)
3. Decisions made (list)
4. Action items (list with assigned person if mentioned)
5. Follow-up needed (list)

Transcript:
{result.text}

Return as JSON."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": analysis_prompt}]
    )
    
    import json
    analysis = json.loads(response.choices[0].message.content)
    
    return {
        "duration": result.duration,
        "transcript": result.text,
        "analysis": analysis
    }
\`\`\`

### 2. Customer Service Call Analysis

\`\`\`python
def analyze_service_call(audio_path: str) -> Dict[str, Any]:
    """Analyze customer service call."""
    result = transcriber.transcribe(audio_path, include_timestamps=True)
    
    analysis_prompt = f"""Analyze this customer service call:

{result.text}

Extract:
1. Customer issue/complaint
2. Resolution provided
3. Customer satisfaction (satisfied/neutral/unsatisfied)
4. Agent performance (excellent/good/needs improvement)
5. Follow-up required (yes/no)
6. Key concerns raised

Return as JSON."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": analysis_prompt}]
    )
    
    import json
    return json.loads(response.choices[0].message.content)
\`\`\`

### 3. Lecture/Educational Content Processing

\`\`\`python
def process_lecture(audio_path: str) -> Dict[str, Any]:
    """Process educational lecture."""
    result = transcriber.transcribe(audio_path, include_timestamps=True)
    
    # Generate study materials
    materials_prompt = f"""From this lecture transcript, generate:

1. Key concepts (list with brief explanations)
2. Important definitions
3. Examples provided
4. Summary of main points
5. Potential exam questions

Transcript:
{result.text}

Return as JSON."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": materials_prompt}]
    )
    
    import json
    return json.loads(response.choices[0].message.content)
\`\`\`

## Best Practices

### 1. Audio Preprocessing

\`\`\`python
from pydub import AudioSegment
from pydub.effects import normalize

def preprocess_audio(
    input_path: str,
    output_path: str
) -> str:
    """Preprocess audio for better transcription."""
    # Load audio
    audio = AudioSegment.from_file(input_path)
    
    # Convert to mono
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Normalize volume
    audio = normalize(audio)
    
    # Resample to 16kHz (sufficient for speech)
    audio = audio.set_frame_rate(16000)
    
    # Export
    audio.export(output_path, format="mp3", bitrate="64k")
    
    return output_path
\`\`\`

### 2. Cost Optimization

\`\`\`python
def estimate_transcription_cost(
    audio_duration_seconds: float,
    price_per_minute: float = 0.006
) -> float:
    """Estimate Whisper API transcription cost."""
    minutes = audio_duration_seconds / 60
    return minutes * price_per_minute

# Example: 1 hour of audio
cost = estimate_transcription_cost(3600)
print(f"Cost for 1 hour: \${cost:.2f}")
\`\`\`

### 3. Quality Validation

\`\`\`python
def validate_transcription(
    transcription: str,
    expected_min_words: int = 50
) -> bool:
    """Validate transcription quality."""
    # Check length
    word_count = len(transcription.split())
    if word_count < expected_min_words:
        return False
    
    # Check for error patterns
    if "[" in transcription or "]" in transcription:
        # Whisper uses brackets for uncertain sections
        return False
    
    # Check for repetition (sign of poor audio quality)
    words = transcription.lower().split()
    if len(set(words)) / len(words) < 0.3:
        # Less than 30% unique words suggests issues
        return False
    
    return True
\`\`\`

## Summary

Audio + text processing enables powerful applications:

**Key Capabilities:**
- High-accuracy speech transcription with Whisper
- Multi-language support (99+ languages)
- Timestamp generation for precise timing
- Text-to-speech synthesis with natural voices
- Speaker diarization and identification
- Sentiment and emotion analysis
- Content classification

**Production Considerations:**
- Preprocess audio (mono, normalize, resample to 16kHz)
- Cache transcriptions aggressively
- Split large files into chunks
- Handle errors and poor audio quality
- Validate transcription results
- Monitor costs (~ $0.006/minute)

**Applications:**
- Meeting transcription and summarization
- Podcast processing and indexing
- Customer service call analysis
- Educational content processing
- Accessibility (audio descriptions, captions)
- Voice interfaces and assistants

**Best Practices:**
- Convert to mono and resample for cost savings
- Use language parameter when known for better accuracy
- Implement caching to avoid re-transcribing
- Validate output quality
- Combine with LLMs for analysis and summarization
- Handle multiple speakers with diarization

Next, we'll explore multi-modal RAG systems that combine text, images, audio, and video for powerful retrieval-augmented generation.
`,
  codeExamples: [
    {
      title: 'Production Transcription System',
      description:
        'Complete audio transcription system with caching, chunking, and optimization',
      language: 'python',
      code: `# See ProductionTranscriptionSystem class in content above`,
    },
  ],
  practicalTips: [
    'Always preprocess audio: convert to mono, resample to 16kHz, normalize volume',
    'Cache transcriptions aggressively - use audio file hash as cache key',
    'Split audio files larger than 25MB into chunks before processing',
    'Specify language parameter if known - improves accuracy and reduces latency',
    'Use verbose_json format to get timestamps for precise timing',
    'Combine transcription with speaker diarization for multi-speaker audio',
    'Validate transcription output for minimum word count and error patterns',
    'Use TTS with streaming for real-time audio generation',
  ],
  quiz: '/quizzes/multi-modal-ai-systems/audio-text-processing',
};
