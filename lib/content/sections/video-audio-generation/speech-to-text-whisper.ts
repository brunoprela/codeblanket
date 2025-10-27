export const speechToTextWhisper = {
  title: 'Speech-to-Text (Whisper)',
  id: 'speech-to-text-whisper',
  content: `
# Speech-to-Text (Whisper)

## Introduction

**OpenAI Whisper** is a state-of-the-art automatic speech recognition (ASR) system that achieves human-level accuracy across multiple languages. Released as open-source in September 2022, Whisper has become the de facto standard for production speech-to-text applications.

**Why Whisper Matters:**
- **Multilingual**: Supports 90+ languages
- **Robust**: Works with accents, background noise, technical terminology
- **Open Source**: Free to use, can be self-hosted
- **Timestamps**: Word-level and sentence-level timestamps
- **Translation**: Can translate non-English speech to English text
- **No training needed**: Works out-of-the-box

**Use Cases:**
- Video transcription and subtitles
- Meeting and lecture notes
- Voice commands and assistants
- Accessibility features
- Content search and indexing
- Podcast transcription

---

## How Whisper Works

### Architecture

Whisper is an encoder-decoder transformer trained on 680,000 hours of multilingual data:

**Pipeline:**1. **Audio Preprocessing**: Convert audio to mel spectrogram
2. **Encoder**: Process audio features with transformer
3. **Decoder**: Generate text tokens autoregressively
4. **Post-processing**: Format, punctuate, add timestamps

\`\`\`python
"""
Complete Whisper Integration for Production
"""

import whisper
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
from dataclang import dataclass
import time
import json

@dataclass
class TranscriptionSegment:
    """Single segment of transcription"""
    id: int
    start: float  # Start time in seconds
    end: float    # End time in seconds
    text: str
    words: Optional[List[Dict]] = None  # Word-level timestamps
    confidence: Optional[float] = None

@dataclass
class TranscriptionResult:
    """Complete transcription result"""
    text: str  # Full transcription
    language: str
    segments: List[TranscriptionSegment]
    duration: float  # Audio duration
    processing_time: float  # Time taken to transcribe

class WhisperTranscriber:
    """
    Production-ready Whisper transcription
    
    Features:
    - Multiple model sizes
    - Batch processing
    - Language detection
    - Word-level timestamps
    - Custom vocabulary
    - VAD (Voice Activity Detection)
    """
    
    # Model sizes and their characteristics
    MODEL_SPECS = {
        "tiny": {"params": "39M", "vram": "~1GB", "speed": "~32x"},
        "base": {"params": "74M", "vram": "~1GB", "speed": "~16x"},
        "small": {"params": "244M", "vram": "~2GB", "speed": "~6x"},
        "medium": {"params": "769M", "vram": "~5GB", "speed": "~2x"},
        "large": {"params": "1550M", "vram": "~10GB", "speed": "~1x"},
    }
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        download_root: Optional[Path] = None,
    ):
        """
        Initialize Whisper transcriber
        
        Args:
            model_size: "tiny", "base", "small", "medium", "large"
            device: "cuda" or "cpu"
            download_root: Where to cache models
        """
        self.model_size = model_size
        self.device = device
        
        print(f"Loading Whisper {model_size} model...")
        specs = self.MODEL_SPECS[model_size]
        print(f"  Parameters: {specs['params']}")
        print(f"  VRAM: {specs['vram']}")
        print(f"  Speed: {specs['speed']} realtime")
        
        # Load model
        self.model = whisper.load_model(
            model_size,
            device=device,
            download_root=download_root,
        )
        
        print("✅ Model loaded successfully")
    
    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",  # or "translate"
        word_timestamps: bool = False,
        initial_prompt: Optional[str] = None,
        temperature: float = 0.0,
        best_of: int = 5,
        beam_size: int = 5,
        patience: float = 1.0,
        condition_on_previous_text: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe audio file
        
        Args:
            audio: Audio file path or numpy array
            language: Language code (e.g., "en", "es", "fr")
            task: "transcribe" or "translate" (to English)
            word_timestamps: Include word-level timestamps
            initial_prompt: Prompt to guide transcription style
            temperature: Sampling temperature (0 = deterministic)
            best_of: Number of candidates when sampling
            beam_size: Beam size for beam search
            patience: Patience for beam search
            condition_on_previous_text: Use previous text as context
        
        Returns:
            TranscriptionResult with text and timestamps
        """
        start_time = time.time()
        
        # Load audio if path
        if isinstance (audio, (str, Path)):
            audio = whisper.load_audio (str (audio))
        
        # Transcribe
        result = self.model.transcribe(
            audio,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            temperature=temperature,
            best_of=best_of,
            beam_size=beam_size,
            patience=patience,
            condition_on_previous_text=condition_on_previous_text,
        )
        
        processing_time = time.time() - start_time
        
        # Parse segments
        segments = []
        for i, seg in enumerate (result["segments"]):
            segments.append(TranscriptionSegment(
                id=i,
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                words=seg.get("words"),
            ))
        
        # Calculate audio duration
        duration = len (audio) / whisper.audio.SAMPLE_RATE
        
        return TranscriptionResult(
            text=result["text"].strip(),
            language=result["language"],
            segments=segments,
            duration=duration,
            processing_time=processing_time,
        )
    
    def detect_language (self, audio: Union[str, Path, np.ndarray]) -> str:
        """
        Detect language of audio
        
        Returns:
            Language code (e.g., "en", "es", "fr")
        """
        # Load audio if needed
        if isinstance (audio, (str, Path)):
            audio = whisper.load_audio (str (audio))
        
        # Pad/trim to 30 seconds
        audio = whisper.pad_or_trim (audio)
        
        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram (audio).to (self.device)
        
        # Detect language
        _, probs = self.model.detect_language (mel)
        
        # Get most likely language
        detected_language = max (probs, key=probs.get)
        
        print(f"Detected language: {detected_language} ({probs[detected_language]:.2%} confidence)")
        
        return detected_language
    
    def transcribe_with_diarization(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
    ) -> List[Dict]:
        """
        Transcribe with speaker diarization
        
        Note: Requires pyannote.audio for diarization
        
        Returns:
            List of segments with speaker labels
        """
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise ImportError("Install pyannote.audio: pip install pyannote.audio")
        
        # Load diarization pipeline
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization"
        )
        
        # Run diarization
        diarization = diarization_pipeline (str (audio_path))
        
        # Run transcription
        transcription = self.transcribe (audio_path, word_timestamps=True)
        
        # Align transcription with speakers
        aligned = []
        for segment in transcription.segments:
            # Find speaker for this timestamp
            speaker = self._find_speaker_at_time(
                diarization,
                (segment.start + segment.end) / 2
            )
            
            aligned.append({
                "speaker": speaker,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            })
        
        return aligned
    
    def _find_speaker_at_time (self, diarization, time: float) -> str:
        """Find which speaker is talking at given time"""
        for turn, _, speaker in diarization.itertracks (yield_label=True):
            if turn.start <= time <= turn.end:
                return speaker
        return "UNKNOWN"
    
    def batch_transcribe(
        self,
        audio_files: List[Union[str, Path]],
        **transcribe_kwargs
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple files
        
        Args:
            audio_files: List of audio file paths
            **transcribe_kwargs: Arguments passed to transcribe()
        
        Returns:
            List of transcription results
        """
        results = []
        
        for i, audio_file in enumerate (audio_files):
            print(f"\\n[{i+1}/{len (audio_files)}] Transcribing {audio_file}")
            
            try:
                result = self.transcribe (audio_file, **transcribe_kwargs)
                results.append (result)
                
                print(f"✅ Completed in {result.processing_time:.1f}s")
                print(f"   Duration: {result.duration:.1f}s")
                print(f"   Speed: {result.duration / result.processing_time:.1f}x realtime")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                results.append(None)
        
        return results
    
    def export_srt(
        self,
        result: TranscriptionResult,
        output_path: Union[str, Path],
    ):
        """
        Export transcription as SRT subtitles
        
        Args:
            result: TranscriptionResult
            output_path: Output .srt file path
        """
        with open (output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate (result.segments, start=1):
                # Format timestamps
                start_time = self._format_timestamp_srt (segment.start)
                end_time = self._format_timestamp_srt (segment.end)
                
                # Write segment
                f.write (f"{i}\\n")
                f.write (f"{start_time} --> {end_time}\\n")
                f.write (f"{segment.text}\\n")
                f.write("\\n")
        
        print(f"💾 Exported SRT: {output_path}")
    
    def export_vtt(
        self,
        result: TranscriptionResult,
        output_path: Union[str, Path],
    ):
        """Export as WebVTT subtitles"""
        with open (output_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\\n\\n")
            
            for segment in result.segments:
                start_time = self._format_timestamp_vtt (segment.start)
                end_time = self._format_timestamp_vtt (segment.end)
                
                f.write (f"{start_time} --> {end_time}\\n")
                f.write (f"{segment.text}\\n")
                f.write("\\n")
        
        print(f"💾 Exported VTT: {output_path}")
    
    def export_json(
        self,
        result: TranscriptionResult,
        output_path: Union[str, Path],
    ):
        """Export as JSON"""
        data = {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
            "segments": [
                {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                }
                for seg in result.segments
            ]
        }
        
        with open (output_path, "w", encoding="utf-8") as f:
            json.dump (data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Exported JSON: {output_path}")
    
    @staticmethod
    def _format_timestamp_srt (seconds: float) -> str:
        """Format seconds as SRT timestamp: 00:00:00,000"""
        hours = int (seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int (seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    @staticmethod
    def _format_timestamp_vtt (seconds: float) -> str:
        """Format seconds as VTT timestamp: 00:00:00.000"""
        hours = int (seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int (seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

# Advanced features
class WhisperWithVAD:
    """
    Whisper with Voice Activity Detection
    
    VAD segments audio before transcription for:
    - Faster processing (skip silence)
    - Better accuracy (natural breaks)
    - Lower costs (process less audio)
    """
    
    def __init__(self, transcriber: WhisperTranscriber):
        self.transcriber = transcriber
        
        # Load VAD model
        try:
            import torch
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            self.get_speech_timestamps = utils[0]
            print("✅ VAD model loaded")
        except Exception as e:
            print(f"⚠️  VAD not available: {e}")
            self.vad_model = None
    
    def transcribe_with_vad(
        self,
        audio_path: Union[str, Path],
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ) -> TranscriptionResult:
        """
        Transcribe with VAD pre-processing
        
        Args:
            audio_path: Audio file path
            min_speech_duration_ms: Minimum speech segment length
            min_silence_duration_ms: Minimum silence between segments
        
        Returns:
            TranscriptionResult
        """
        # Load audio
        audio = whisper.load_audio (str (audio_path))
        
        # Detect speech segments
        print("Running VAD...")
        speech_timestamps = self.get_speech_timestamps(
            torch.from_numpy (audio),
            self.vad_model,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            return_seconds=True,
        )
        
        print(f"Found {len (speech_timestamps)} speech segments")
        
        # Transcribe each segment
        all_segments = []
        full_text = []
        
        for i, timestamp in enumerate (speech_timestamps):
            start_sample = int (timestamp['start'] * whisper.audio.SAMPLE_RATE)
            end_sample = int (timestamp['end'] * whisper.audio.SAMPLE_RATE)
            
            # Extract segment
            segment_audio = audio[start_sample:end_sample]
            
            # Transcribe
            result = self.transcriber.transcribe (segment_audio)
            
            # Adjust timestamps
            for seg in result.segments:
                seg.start += timestamp['start']
                seg.end += timestamp['start']
                all_segments.append (seg)
            
            full_text.append (result.text)
        
        # Combine results
        return TranscriptionResult(
            text=" ".join (full_text),
            language=result.language if result else "unknown",
            segments=all_segments,
            duration=len (audio) / whisper.audio.SAMPLE_RATE,
            processing_time=0,  # Not tracked here
        )

# Production examples
def production_examples():
    """Real-world Whisper usage examples"""
    
    # Initialize transcriber
    transcriber = WhisperTranscriber(
        model_size="base",  # Good balance of speed and accuracy
        device="cuda",
    )
    
    # Example 1: Basic transcription
    print("\\n=== Example 1: Basic Transcription ===")
    result = transcriber.transcribe(
        "podcast_episode.mp3",
        language="en",  # Specify if known (faster)
    )
    
    print(f"Transcription: {result.text[:200]}...")
    print(f"Language: {result.language}")
    print(f"Duration: {result.duration:.1f}s")
    print(f"Processing time: {result.processing_time:.1f}s")
    print(f"Realtime factor: {result.duration / result.processing_time:.1f}x")
    
    # Export formats
    transcriber.export_srt (result, "podcast.srt")
    transcriber.export_vtt (result, "podcast.vtt")
    transcriber.export_json (result, "podcast.json")
    
    # Example 2: Language detection
    print("\\n=== Example 2: Language Detection ===")
    detected_lang = transcriber.detect_language("unknown_language.mp3")
    
    result = transcriber.transcribe(
        "unknown_language.mp3",
        language=detected_lang,
    )
    
    # Example 3: Translation to English
    print("\\n=== Example 3: Translation ===")
    result = transcriber.transcribe(
        "spanish_audio.mp3",
        language="es",
        task="translate",  # Translate to English
    )
    
    print(f"Translated: {result.text}")
    
    # Example 4: Word-level timestamps
    print("\\n=== Example 4: Word-Level Timestamps ===")
    result = transcriber.transcribe(
        "speech.mp3",
        word_timestamps=True,
    )
    
    for segment in result.segments[:2]:  # First 2 segments
        print(f"\\n[{segment.start:.2f}s - {segment.end:.2f}s]")
        if segment.words:
            for word in segment.words:
                print(f"  {word['start']:.2f}s: {word['word']}")
    
    # Example 5: Custom vocabulary/style
    print("\\n=== Example 5: Custom Vocabulary ===")
    result = transcriber.transcribe(
        "technical_talk.mp3",
        initial_prompt="This is a technical presentation about machine learning, neural networks, and transformers.",
        temperature=0.0,  # Deterministic
    )
    
    # Example 6: Batch processing
    print("\\n=== Example 6: Batch Processing ===")
    audio_files = [
        "video1.mp4",
        "video2.mp4",
        "video3.mp4",
    ]
    
    results = transcriber.batch_transcribe(
        audio_files,
        language="en",
    )
    
    # Save all transcriptions
    for audio_file, result in zip (audio_files, results):
        if result:
            output_path = Path (audio_file).with_suffix('.srt')
            transcriber.export_srt (result, output_path)
    
    # Example 7: With VAD (faster, skips silence)
    print("\\n=== Example 7: With VAD ===")
    vad_transcriber = WhisperWithVAD(transcriber)
    
    result = vad_transcriber.transcribe_with_vad(
        "long_podcast.mp3",
        min_speech_duration_ms=500,
        min_silence_duration_ms=300,
    )
    
    print(f"Transcribed with VAD: {len (result.segments)} segments")

if __name__ == "__main__":
    production_examples()
\`\`\`

---

## OpenAI Whisper API

For cloud-based transcription without managing infrastructure:

\`\`\`python
"""
OpenAI Whisper API Integration
"""

import openai
from pathlib import Path
from typing import Optional

class WhisperAPI:
    """
    OpenAI Whisper API client
    
    Advantages:
    - No infrastructure management
    - Latest model improvements
    - Automatic scaling
    
    Limitations:
    - Costs per minute
    - File size limit (25MB)
    - Internet required
    """
    
    def __init__(self, api_key: Optional[str] = None):
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",  # or "srt", "vtt", "text"
        temperature: float = 0.0,
    ) -> dict:
        """
        Transcribe using OpenAI API
        
        Args:
            audio_path: Audio file path
            language: Language code
            prompt: Custom vocabulary/context
            response_format: Output format
            temperature: Sampling temperature
        
        Returns:
            Transcription result
        """
        with open (audio_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
            )
        
        return transcript
    
    def translate(
        self,
        audio_path: Union[str, Path],
        prompt: Optional[str] = None,
    ) -> dict:
        """Translate audio to English"""
        with open (audio_path, "rb") as audio_file:
            translation = openai.Audio.translate(
                model="whisper-1",
                file=audio_file,
                prompt=prompt,
            )
        
        return translation

# Example
def api_example():
    """Use OpenAI Whisper API"""
    api = WhisperAPI()
    
    # Transcribe
    result = api.transcribe(
        "audio.mp3",
        language="en",
        response_format="srt",  # Get SRT directly
    )
    
    # Save
    with open("subtitles.srt", "w") as f:
        f.write (result)
\`\`\`

---

## Best Practices

### 1. Model Selection

| Model | When to Use |
|-------|-------------|
| **tiny** | Real-time transcription, very fast prototyping |
| **base** | Most use cases, good balance |
| **small** | Better accuracy needed, still fast |
| **medium** | High accuracy required, have GPU |
| **large** | Best possible accuracy, specialized domains |

### 2. Performance Optimization

\`\`\`python
"""
Optimization strategies for Whisper
"""

# 1. Use FP16 for 2x speedup
model = whisper.load_model("base", device="cuda").half()

# 2. Use VAD to skip silence
vad_transcriber = WhisperWithVAD(transcriber)

# 3. Batch process with threading
from concurrent.futures import ThreadPoolExecutor

def transcribe_parallel (audio_files, num_workers=4):
    transcriber = WhisperTranscriber("base")
    
    with ThreadPoolExecutor (max_workers=num_workers) as executor:
        results = list (executor.map (transcriber.transcribe, audio_files))
    
    return results

# 4. Use faster-whisper (CTranslate2 backend)
# pip install faster-whisper
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.mp3")
\`\`\`

### 3. Handling Different Audio Formats

\`\`\`python
"""
Audio preprocessing for Whisper
"""

from pydub import AudioSegment

def preprocess_audio(
    input_path: str,
    output_path: str = "temp_audio.wav",
) -> str:
    """
    Convert audio to Whisper-friendly format
    
    Whisper expects:
    - 16kHz sample rate
    - Mono channel
    - WAV or common formats
    """
    # Load audio (supports many formats)
    audio = AudioSegment.from_file (input_path)
    
    # Convert to mono
    audio = audio.set_channels(1)
    
    # Resample to 16kHz
    audio = audio.set_frame_rate(16000)
    
    # Export
    audio.export (output_path, format="wav")
    
    return output_path

# Usage
processed_path = preprocess_audio("input.mp3")
result = transcriber.transcribe (processed_path)
\`\`\`

---

## Summary

**Key Takeaways:**
- Whisper provides state-of-the-art speech recognition
- Multiple model sizes for different speed/accuracy tradeoffs
- Supports 90+ languages with translation
- Word-level timestamps for precise alignment
- VAD significantly improves speed by skipping silence
- Can self-host or use OpenAI API

**Production Checklist:**
- ✅ Choose appropriate model size
- ✅ Add VAD for longer audio
- ✅ Implement error handling and retries
- ✅ Export in required format (SRT, VTT, JSON)
- ✅ Monitor costs if using API
- ✅ Test with domain-specific audio

**Next Steps:**
- Integrate with video processing pipeline
- Build automated subtitle generation
- Combine with translation for multilingual content
- Add speaker diarization for interviews/meetings
`,
  exercises: [
    {
      title: 'Exercise 1: Automated Subtitle Generator',
      difficulty: 'intermediate' as const,
      description:
        'Build a tool that automatically generates multi-format subtitles (SRT, VTT, JSON) from video files with options for language detection and translation.',
      hints: [
        'Extract audio from video with ffmpeg',
        'Use Whisper for transcription',
        'Export in multiple formats',
        'Add CLI interface with progress bars',
      ],
    },
    {
      title: 'Exercise 2: Meeting Transcription Service',
      difficulty: 'advanced' as const,
      description:
        'Create a service that transcribes meeting recordings with speaker diarization, generates summaries, and identifies action items.',
      hints: [
        'Combine Whisper with pyannote for diarization',
        'Use LLM to generate summaries from transcript',
        'Extract action items with prompt engineering',
        'Store results in structured database',
      ],
    },
  ],
};
