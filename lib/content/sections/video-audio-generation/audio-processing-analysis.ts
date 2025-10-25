export const audioProcessingAnalysis = {
  title: 'Audio Processing & Analysis',
  id: 'audio-processing-analysis',
  content: `
# Audio Processing & Analysis

## Introduction

Audio processing is essential for preparing, analyzing, and enhancing audio for AI applications. This section covers production-ready techniques for:

- **Format conversion** between audio formats
- **Noise reduction** and audio enhancement  
- **Audio separation** (vocals, instruments, etc.)
- **Analysis and feature extraction**
- **FFmpeg integration** for professional workflows

---

## FFmpeg: The Swiss Army Knife

FFmpeg is the industry standard for audio/video processing:

\`\`\`python
"""
Production FFmpeg Integration
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict
import json

class FFmpegProcessor:
    """
    Production-ready FFmpeg wrapper for audio processing
    """
    
    @staticmethod
    def convert_format(
        input_path: Path,
        output_path: Path,
        format: str = "wav",
        sample_rate: int = 44100,
        channels: int = 2,
        bitrate: str = "192k",
    ):
        """Convert audio format"""
        cmd = [
            "ffmpeg", "-i", str (input_path),
            "-ar", str (sample_rate),
            "-ac", str (channels),
            "-b:a", bitrate,
            "-y",  # Overwrite
            str (output_path)
        ]
        
        subprocess.run (cmd, check=True, capture_output=True)
        print(f"✅ Converted: {output_path}")
    
    @staticmethod
    def extract_audio_from_video(
        video_path: Path,
        output_path: Path,
        format: str = "mp3",
    ):
        """Extract audio track from video"""
        cmd = [
            "ffmpeg", "-i", str (video_path),
            "-vn",  # No video
            "-acodec", "libmp3lame" if format == "mp3" else "copy",
            "-y",
            str (output_path)
        ]
        
        subprocess.run (cmd, check=True, capture_output=True)
        print(f"✅ Extracted audio: {output_path}")
    
    @staticmethod
    def normalize_audio(
        input_path: Path,
        output_path: Path,
        target_level: float = -16.0,  # LUFS
    ):
        """Normalize audio loudness"""
        cmd = [
            "ffmpeg", "-i", str (input_path),
            "-af", f"loudnorm=I={target_level}:TP=-1.5:LRA=11",
            "-y",
            str (output_path)
        ]
        
        subprocess.run (cmd, check=True, capture_output=True)
        print(f"✅ Normalized: {output_path}")
    
    @staticmethod
    def get_audio_info (audio_path: Path) -> Dict:
        """Get audio file information"""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str (audio_path)
        ]
        
        result = subprocess.run (cmd, capture_output=True, text=True, check=True)
        data = json.loads (result.stdout)
        
        audio_stream = next(
            s for s in data["streams"] if s["codec_type"] == "audio"
        )
        
        return {
            "duration": float (data["format"]["duration"]),
            "sample_rate": int (audio_stream["sample_rate"]),
            "channels": int (audio_stream["channels"]),
            "codec": audio_stream["codec_name"],
            "bitrate": int (data["format"].get("bit_rate", 0)),
        }

# Example usage
def ffmpeg_examples():
    """FFmpeg processing examples"""
    
    ff = FFmpegProcessor()
    
    # Convert MP3 to WAV
    ff.convert_format("input.mp3", "output.wav", format="wav")
    
    # Extract audio from video
    ff.extract_audio_from_video("video.mp4", "audio.mp3")
    
    # Normalize
    ff.normalize_audio("input.wav", "normalized.wav")
    
    # Get info
    info = ff.get_audio_info("audio.mp3")
    print(json.dumps (info, indent=2))

if __name__ == "__main__":
    ffmpeg_examples()
\`\`\`

---

## Noise Reduction

\`\`\`python
"""
AI-powered noise reduction
"""

import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np

class NoiseReducer:
    """Remove background noise from audio"""
    
    @staticmethod
    def reduce_noise(
        audio_path: Path,
        output_path: Path,
        noise_profile_duration: float = 1.0,
    ):
        """
        Reduce noise using spectral gating
        
        Args:
            audio_path: Input audio
            output_path: Output audio
            noise_profile_duration: Duration of noise sample (seconds)
        """
        # Load audio
        audio, sr = librosa.load (audio_path, sr=None)
        
        # Use first N seconds as noise profile
        noise_sample_frames = int (noise_profile_duration * sr)
        noise_sample = audio[:noise_sample_frames]
        
        # Reduce noise
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            y_noise=noise_sample,
            stationary=True,
        )
        
        # Save
        sf.write (output_path, reduced, sr)
        print(f"✅ Noise reduced: {output_path}")
    
    @staticmethod
    def enhance_speech(
        audio_path: Path,
        output_path: Path,
    ):
        """Enhance speech by reducing non-speech frequencies"""
        audio, sr = librosa.load (audio_path, sr=None)
        
        # Apply bandpass filter for speech (300Hz - 3400Hz)
        from scipy.signal import butter, filtfilt
        
        nyquist = sr / 2
        low = 300 / nyquist
        high = 3400 / nyquist
        
        b, a = butter(5, [low, high], btype='band')
        filtered = filtfilt (b, a, audio)
        
        # Normalize
        filtered = filtered / np.max (np.abs (filtered))
        
        sf.write (output_path, filtered, sr)
        print(f"✅ Speech enhanced: {output_path}")
\`\`\`

---

## Audio Separation

\`\`\`python
"""
Separate audio into components (vocals, drums, bass, other)
"""

from spleeter.separator import Separator

class AudioSeparator:
    """Separate audio into stems"""
    
    def __init__(self, stems: int = 4):
        """
        Initialize separator
        
        Args:
            stems: 2 (vocals/accompaniment), 4 (vocals/drums/bass/other), or 5
        """
        self.separator = Separator (f'spleeter:{stems}stems')
        print(f"✅ Loaded {stems}-stem separator")
    
    def separate(
        self,
        audio_path: Path,
        output_dir: Path,
    ):
        """
        Separate audio into stems
        
        Args:
            audio_path: Input audio
            output_dir: Output directory for stems
        """
        output_dir = Path (output_dir)
        output_dir.mkdir (parents=True, exist_ok=True)
        
        print(f"Separating {audio_path}...")
        
        # Separate
        self.separator.separate_to_file(
            str (audio_path),
            str (output_dir),
        )
        
        print(f"✅ Stems saved to {output_dir}")

# Example
def separation_example():
    """Separate vocals from music"""
    
    separator = AudioSeparator (stems=2)  # vocals + accompaniment
    
    separator.separate(
        audio_path="song.mp3",
        output_dir="separated/",
    )
    
    # Result:
    # separated/song/vocals.wav
    # separated/song/accompaniment.wav
    
    print("✅ Vocals and music separated")
\`\`\`

---

## Audio Analysis

\`\`\`python
"""
Extract features and analyze audio
"""

import librosa
import numpy as np
from typing import Dict

class AudioAnalyzer:
    """Analyze audio characteristics"""
    
    @staticmethod
    def extract_features (audio_path: Path) -> Dict:
        """Extract comprehensive audio features"""
        
        # Load audio
        y, sr = librosa.load (audio_path)
        
        # Tempo and beats
        tempo, beats = librosa.beat.beat_track (y=y, sr=sr)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid (y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff (y=y, sr=sr)[0]
        
        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc (y=y, sr=sr, n_mfcc=13)
        
        # Chroma (pitch classes)
        chroma = librosa.feature.chroma_stft (y=y, sr=sr)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate (y)[0]
        
        # RMS energy
        rms = librosa.feature.rms (y=y)[0]
        
        return {
            "duration": len (y) / sr,
            "sample_rate": sr,
            "tempo": float (tempo),
            "beats": len (beats),
            "spectral_centroid_mean": float (np.mean (spectral_centroids)),
            "spectral_centroid_std": float (np.std (spectral_centroids)),
            "spectral_rolloff_mean": float (np.mean (spectral_rolloff)),
            "mfcc_mean": mfccs.mean (axis=1).tolist(),
            "chroma_mean": chroma.mean (axis=1).tolist(),
            "zcr_mean": float (np.mean (zcr)),
            "rms_mean": float (np.mean (rms)),
        }
    
    @staticmethod
    def detect_silence(
        audio_path: Path,
        threshold_db: float = -40.0,
    ) -> list:
        """Detect silent segments"""
        y, sr = librosa.load (audio_path)
        
        # Convert to dB
        db = librosa.amplitude_to_db (np.abs (y), ref=np.max)
        
        # Find silent frames
        silent_frames = db < threshold_db
        
        # Convert frames to time segments
        frame_length = 2048
        hop_length = 512
        
        times = librosa.frames_to_time(
            np.arange (len (silent_frames)),
            sr=sr,
            hop_length=hop_length
        )
        
        # Group consecutive silent frames
        segments = []
        start = None
        
        for i, (time, is_silent) in enumerate (zip (times, silent_frames)):
            if is_silent and start is None:
                start = time
            elif not is_silent and start is not None:
                segments.append({"start": start, "end": time})
                start = None
        
        return segments
    
    @staticmethod
    def transcribe_music (audio_path: Path) -> Dict:
        """
        Transcribe music to notes/chords
        
        Basic implementation - production would use specialized models
        """
        y, sr = librosa.load (audio_path)
        
        # Detect pitch
        pitches, magnitudes = librosa.piptrack (y=y, sr=sr)
        
        # Get dominant pitch per frame
        pitch_contour = []
        for t in range (pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_contour.append (pitch)
        
        return {
            "pitch_contour": pitch_contour,
            "mean_pitch": np.mean([p for p in pitch_contour if p > 0]),
        }

# Production example
def analysis_pipeline():
    """Complete audio analysis pipeline"""
    
    analyzer = AudioAnalyzer()
    
    # Extract features
    features = analyzer.extract_features("song.mp3")
    
    print("Audio Features:")
    print(f"  Duration: {features['duration']:.2f}s")
    print(f"  Tempo: {features['tempo']:.1f} BPM")
    print(f"  Beats: {features['beats']}")
    print(f"  Spectral Centroid: {features['spectral_centroid_mean']:.1f} Hz")
    
    # Detect silence
    silent_segments = analyzer.detect_silence("song.mp3")
    print(f"\\nSilent segments: {len (silent_segments)}")
    for seg in silent_segments[:3]:
        print(f"  {seg['start']:.2f}s - {seg['end']:.2f}s")
    
    # Music transcription
    transcription = analyzer.transcribe_music("song.mp3")
    print(f"\\nMean pitch: {transcription['mean_pitch']:.1f} Hz")
\`\`\`

---

## Production Pipeline

\`\`\`python
"""
Complete audio processing pipeline
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class ProcessingStep(Enum):
    CONVERT = "convert"
    NORMALIZE = "normalize"
    DENOISE = "denoise"
    ENHANCE_SPEECH = "enhance_speech"
    SEPARATE = "separate"
    ANALYZE = "analyze"

@dataclass
class AudioProcessingConfig:
    steps: List[ProcessingStep]
    output_format: str = "wav"
    sample_rate: int = 44100
    normalize_level: float = -16.0

class AudioProcessingPipeline:
    """Complete audio processing pipeline"""
    
    def __init__(self):
        self.ffmpeg = FFmpegProcessor()
        self.noise_reducer = NoiseReducer()
        self.separator = AudioSeparator()
        self.analyzer = AudioAnalyzer()
    
    def process(
        self,
        input_path: Path,
        config: AudioProcessingConfig,
        output_dir: Path,
    ) -> Dict:
        """
        Process audio through pipeline
        
        Args:
            input_path: Input audio file
            config: Processing configuration
            output_dir: Output directory
        
        Returns:
            Results dictionary
        """
        output_dir = Path (output_dir)
        output_dir.mkdir (parents=True, exist_ok=True)
        
        current_file = input_path
        results = {"steps": []}
        
        for step in config.steps:
            print(f"\\nStep: {step.value}")
            
            if step == ProcessingStep.CONVERT:
                output = output_dir / f"converted.{config.output_format}"
                self.ffmpeg.convert_format(
                    current_file, output,
                    format=config.output_format,
                    sample_rate=config.sample_rate,
                )
                current_file = output
                results["steps"].append({"convert": str (output)})
            
            elif step == ProcessingStep.NORMALIZE:
                output = output_dir / "normalized.wav"
                self.ffmpeg.normalize_audio(
                    current_file, output,
                    target_level=config.normalize_level,
                )
                current_file = output
                results["steps"].append({"normalize": str (output)})
            
            elif step == ProcessingStep.DENOISE:
                output = output_dir / "denoised.wav"
                self.noise_reducer.reduce_noise (current_file, output)
                current_file = output
                results["steps"].append({"denoise": str (output)})
            
            elif step == ProcessingStep.ENHANCE_SPEECH:
                output = output_dir / "enhanced.wav"
                self.noise_reducer.enhance_speech (current_file, output)
                current_file = output
                results["steps"].append({"enhance": str (output)})
            
            elif step == ProcessingStep.SEPARATE:
                self.separator.separate (current_file, output_dir / "stems")
                results["steps"].append({"separate": str (output_dir / "stems")})
            
            elif step == ProcessingStep.ANALYZE:
                features = self.analyzer.extract_features (current_file)
                results["features"] = features
                results["steps"].append({"analyze": features})
        
        results["final_output"] = str (current_file)
        
        return results

# Example: Process podcast audio
def podcast_processing_example():
    """Process podcast audio for optimal quality"""
    
    pipeline = AudioProcessingPipeline()
    
    config = AudioProcessingConfig(
        steps=[
            ProcessingStep.CONVERT,
            ProcessingStep.DENOISE,
            ProcessingStep.ENHANCE_SPEECH,
            ProcessingStep.NORMALIZE,
            ProcessingStep.ANALYZE,
        ],
        output_format="wav",
        sample_rate=44100,
        normalize_level=-16.0,
    )
    
    results = pipeline.process(
        input_path="raw_podcast.mp3",
        config=config,
        output_dir="processed_podcast/",
    )
    
    print("\\n✅ Processing complete!")
    print(f"Final output: {results['final_output']}")
    print(f"Duration: {results['features']['duration']:.1f}s")
    print(f"Steps completed: {len (results['steps'])}")

if __name__ == "__main__":
    podcast_processing_example()
\`\`\`

---

## Summary

**Key Takeaways:**
- FFmpeg is essential for format conversion and basic processing
- Noise reduction improves audio quality significantly
- Audio separation enables remixing and karaoke
- Feature extraction enables analysis and search
- Pipelines automate complex workflows

**Production Checklist:**
- ✅ Install FFmpeg and dependencies
- ✅ Normalize audio levels
- ✅ Remove background noise
- ✅ Extract relevant features
- ✅ Handle multiple formats
- ✅ Implement error handling

**Next Steps:**
- Build automated audio processing API
- Integrate with transcription pipeline
- Create audio quality checker
- Implement batch processing
`,
  exercises: [
    {
      title: 'Exercise 1: Audio Quality Enhancer API',
      id: 'audio-processing-analysis',
      difficulty: 'intermediate' as const,
      description:
        'Build a REST API that accepts audio uploads and automatically enhances quality with noise reduction, normalization, and format optimization.',
      hints: [
        'Use FastAPI for the API',
        'Implement async processing',
        'Add quality metrics',
        'Support multiple formats',
      ],
    },
    {
      title: 'Exercise 2: Music Stem Separator',
      id: 'audio-processing-analysis',
      difficulty: 'advanced' as const,
      description:
        'Create a tool that separates music into stems (vocals, drums, bass, other) and allows remixing with volume controls for each stem.',
      hints: [
        'Use Spleeter for separation',
        'Build mixer for stems',
        'Add effects per stem',
        'Export mixed result',
      ],
    },
  ],
};
