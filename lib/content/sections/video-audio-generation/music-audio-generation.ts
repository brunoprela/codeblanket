export const musicAudioGeneration = {
  title: 'Music & Audio Generation',
  id: 'music-audio-generation',
  content: `
# Music & Audio Generation

## Introduction

AI-powered music and audio generation has evolved from simple MIDI synthesis to creating professional-quality music, sound effects, and ambient audio. Modern AI can:

- **Generate original music** in any style or genre
- **Create sound effects** for games, films, and apps
- **Remix and mashup** existing tracks
- **Generate ambient soundscapes**
- **Compose adaptive music** that responds to context

**Key Technologies:**
- **MusicGen** (Meta): Text-to-music generation
- **AudioCraft**: Complete audio generation suite
- **Riffusion**: Stable Diffusion for music
- **MusicLM** (Google): High-quality music generation
- **AIVA, Amper**: Commercial music generation platforms

---

## MusicGen and AudioCraft

### Overview

**AudioCraft** is Meta\'s suite of generative AI models for audio:
- **MusicGen**: Music generation from text
- **AudioGen**: Sound effects and ambient audio
- **EnCodec**: High-fidelity audio compression

\`\`\`python
"""
Production MusicGen Integration
"""

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch
from typing import List, Optional
from pathlib import Path
import numpy as np

class MusicGenerator:
    """
    Production-ready music generation with MusicGen
    
    Features:
    - Text-to-music generation
    - Melody conditioning
    - Multi-track generation
    - Style control
    - Duration control
    """
    
    # Available models
    MODELS = {
        "small": "facebook/musicgen-small",  # Fastest, 300M params
        "medium": "facebook/musicgen-medium",  # Balanced, 1.5B params
        "large": "facebook/musicgen-large",  # Best quality, 3.3B params
        "melody": "facebook/musicgen-melody",  # Melody conditioning
    }
    
    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize MusicGen
        
        Args:
            model_size: "small", "medium", "large", or "melody"
            device: "cuda" or "cpu"
        """
        self.model_size = model_size
        self.device = device
        
        print(f"Loading MusicGen {model_size}...")
        self.model = MusicGen.get_pretrained (self.MODELS[model_size])
        self.model.to (device)
        
        # Set default generation params
        self.model.set_generation_params(
            duration=10,  # seconds
            temperature=1.0,
            top_k=250,
            top_p=0.0,
            cfg_coef=3.0,  # Classifier-free guidance
        )
        
        print("✅ Model loaded")
    
    def generate(
        self,
        descriptions: List[str],
        duration: float = 10.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        cfg_coef: float = 3.0,
    ) -> List[np.ndarray]:
        """
        Generate music from text descriptions
        
        Args:
            descriptions: List of text descriptions
            duration: Duration in seconds (max 30s for small/medium)
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            cfg_coef: Classifier-free guidance coefficient
        
        Returns:
            List of audio arrays (sample_rate = 32kHz)
        """
        # Set generation parameters
        self.model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_coef=cfg_coef,
        )
        
        print(f"Generating {len (descriptions)} tracks ({duration}s each)...")
        
        # Generate
        with torch.no_grad():
            wav = self.model.generate (descriptions)
        
        # Convert to numpy
        audio_arrays = wav.cpu().numpy()
        
        print("✅ Generation complete")
        
        return list (audio_arrays)
    
    def generate_with_melody(
        self,
        descriptions: List[str],
        melody_audio: np.ndarray,
        melody_sample_rate: int,
        duration: float = 10.0,
    ) -> List[np.ndarray]:
        """
        Generate music conditioned on a melody
        
        Only works with "melody" model
        
        Args:
            descriptions: Text descriptions
            melody_audio: Melody audio array
            melody_sample_rate: Sample rate of melody
            duration: Output duration
        
        Returns:
            Generated audio conditioned on melody
        """
        if self.model_size != "melody":
            raise ValueError("Melody conditioning requires 'melody' model")
        
        # Convert melody to tensor
        melody_tensor = torch.from_numpy (melody_audio).unsqueeze(0)
        
        # Resample if needed (MusicGen uses 32kHz)
        if melody_sample_rate != self.model.sample_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                melody_sample_rate,
                self.model.sample_rate
            )
            melody_tensor = resampler (melody_tensor)
        
        # Set duration
        self.model.set_generation_params (duration=duration)
        
        print(f"Generating with melody conditioning...")
        
        # Generate
        with torch.no_grad():
            wav = self.model.generate_with_chroma(
                descriptions=descriptions,
                melody_wavs=melody_tensor,
                melody_sample_rate=self.model.sample_rate,
            )
        
        audio_arrays = wav.cpu().numpy()
        
        return list (audio_arrays)
    
    def generate_continuation(
        self,
        prompt_audio: np.ndarray,
        description: str,
        duration: float = 10.0,
    ) -> np.ndarray:
        """
        Continue existing audio
        
        Args:
            prompt_audio: Initial audio to continue
            description: Description of continuation
            duration: Total duration (including prompt)
        
        Returns:
            Continued audio
        """
        # Convert to tensor
        prompt_tensor = torch.from_numpy (prompt_audio).unsqueeze(0)
        
        self.model.set_generation_params (duration=duration)
        
        print("Generating continuation...")
        
        with torch.no_grad():
            wav = self.model.generate_continuation(
                prompt=prompt_tensor,
                prompt_sample_rate=self.model.sample_rate,
                descriptions=[description],
            )
        
        return wav.cpu().numpy()[0]
    
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: Path,
        sample_rate: int = 32000,
    ):
        """Save generated audio"""
        output_path = Path (output_path)
        output_path.parent.mkdir (parents=True, exist_ok=True)
        
        # audio_write handles normalization and format
        audio_write(
            str (output_path.with_suffix(')),
            torch.from_numpy (audio),
            sample_rate,
            strategy="loudness",
            loudness_compressor=True,
        )
        
        print(f"💾 Saved: {output_path}")

# Sound effects generation
class SoundEffectGenerator:
    """
    Generate sound effects with AudioGen
    """
    
    def __init__(self, device: str = "cuda"):
        from audiocraft.models import AudioGen
        
        print("Loading AudioGen...")
        self.model = AudioGen.get_pretrained("facebook/audiogen-medium")
        self.model.to (device)
        
        self.model.set_generation_params (duration=5.0)
        
        print("✅ AudioGen loaded")
    
    def generate(
        self,
        descriptions: List[str],
        duration: float = 5.0,
    ) -> List[np.ndarray]:
        """
        Generate sound effects
        
        Args:
            descriptions: Descriptions of sound effects
            duration: Duration in seconds
        
        Returns:
            Generated audio arrays
        """
        self.model.set_generation_params (duration=duration)
        
        print(f"Generating {len (descriptions)} sound effects...")
        
        with torch.no_grad():
            wav = self.model.generate (descriptions)
        
        return list (wav.cpu().numpy())

# Production examples
def production_examples():
    """Real-world music generation examples"""
    
    # Initialize generator
    music_gen = MusicGenerator (model_size="medium", device="cuda")
    
    # Example 1: Simple music generation
    print("\\n=== Example 1: Basic Music Generation ===")
    
    prompts = [
        "upbeat electronic dance music with heavy bass",
        "calm acoustic guitar with soft piano",
        "epic orchestral soundtrack with strings and brass",
        "jazz piano trio with walking bass",
    ]
    
    audio_arrays = music_gen.generate(
        descriptions=prompts,
        duration=15.0,
        temperature=1.0,
    )
    
    # Save tracks
    for i, (prompt, audio) in enumerate (zip (prompts, audio_arrays)):
        output_path = f"track_{i+1}.wav"
        music_gen.save_audio (audio, output_path)
        print(f"Generated: {prompt[:50]}...")
    
    # Example 2: Genre-specific generation
    print("\\n=== Example 2: Genre-Specific ===")
    
    genres = {
        "lofi_hiphop": "lofi hip hop beat with vinyl crackle and mellow piano",
        "synthwave": "80s synthwave with retro synthesizers and drum machines",
        "ambient": "ambient atmospheric soundscape with pads and textures",
        "rock": "energetic rock music with electric guitar and drums",
    }
    
    for name, description in genres.items():
        audio = music_gen.generate([description], duration=20.0)[0]
        music_gen.save_audio (audio, f"genre_{name}.wav")
    
    # Example 3: Melody conditioning
    print("\\n=== Example 3: Melody Conditioning ===")
    
    # Load existing melody
    import torchaudio
    melody, sr = torchaudio.load("input_melody.wav")
    
    melody_gen = MusicGenerator (model_size="melody")
    
    audio = melody_gen.generate_with_melody(
        descriptions=["orchestral arrangement"],
        melody_audio=melody.numpy(),
        melody_sample_rate=sr,
        duration=15.0,
    )[0]
    
    melody_gen.save_audio (audio, "melody_conditioned.wav")
    
    # Example 4: Music continuation
    print("\\n=== Example 4: Continuation ===")
    
    # Generate initial segment
    initial = music_gen.generate(
        ["upbeat pop song with vocals"],
        duration=10.0
    )[0]
    
    # Continue it
    continued = music_gen.generate_continuation(
        prompt_audio=initial,
        description="continuing the upbeat pop song with a bridge section",
        duration=20.0,  # Total duration
    )
    
    music_gen.save_audio (continued, "continued_song.wav")
    
    # Example 5: Sound effects
    print("\\n=== Example 5: Sound Effects ===")
    
    sfx_gen = SoundEffectGenerator()
    
    sound_effects = [
        "door creaking open slowly",
        "footsteps on wooden floor",
        "wind blowing through trees",
        "car engine starting",
        "crowd cheering loudly",
    ]
    
    sfx_audio = sfx_gen.generate (sound_effects, duration=3.0)
    
    for i, (desc, audio) in enumerate (zip (sound_effects, sfx_audio)):
        output = Path (f"sfx_{i+1}_{desc.replace(' ', '_')[:20]}.wav")
        music_gen.save_audio (audio, output)
    
    print("\\n✅ All examples completed")

if __name__ == "__main__":
    production_examples()
\`\`\`

---

## Advanced Music Generation Techniques

### 1. Style Transfer for Music

\`\`\`python
"""
Music style transfer
"""

class MusicStyleTransfer:
    """Transfer style from one track to another"""
    
    def __init__(self):
        # Would use models like Jukebox or custom training
        pass
    
    def transfer_style(
        self,
        content_audio: np.ndarray,
        style_audio: np.ndarray,
        output_path: Path,
    ):
        """
        Transfer style from style_audio to content_audio
        
        Conceptual implementation
        """
        # Extract style features
        style_features = self._extract_style (style_audio)
        
        # Apply to content
        stylized = self._apply_style (content_audio, style_features)
        
        return stylized
\`\`\`

### 2. Adaptive Music Generation

\`\`\`python
"""
Generate music that adapts to context
"""

class AdaptiveMusicGenerator:
    """
    Generate music that changes based on context
    
    Use cases:
    - Game soundtracks that adapt to gameplay
    - Fitness music that matches exercise intensity
    - Meditation music that responds to heart rate
    """
    
    def __init__(self, music_gen: MusicGenerator):
        self.music_gen = music_gen
        self.current_intensity = 0.5
    
    def generate_adaptive_track(
        self,
        base_description: str,
        intensity_levels: List[float],  # 0.0 to 1.0
        segment_duration: float = 5.0,
    ) -> np.ndarray:
        """
        Generate music with varying intensity
        
        Args:
            base_description: Base music description
            intensity_levels: Intensity for each segment
            segment_duration: Duration of each segment
        
        Returns:
            Complete adaptive track
        """
        segments = []
        
        for i, intensity in enumerate (intensity_levels):
            # Modify description based on intensity
            description = self._adjust_for_intensity(
                base_description,
                intensity
            )
            
            print(f"Segment {i+1}: Intensity {intensity:.2f}")
            
            # Generate segment
            audio = self.music_gen.generate(
                [description],
                duration=segment_duration,
                temperature=0.8 + intensity * 0.4,  # Higher intensity = more variation
            )[0]
            
            segments.append (audio)
        
        # Concatenate with crossfades
        return self._crossfade_segments (segments)
    
    def _adjust_for_intensity (self, base: str, intensity: float) -> str:
        """Modify description based on intensity level"""
        if intensity < 0.3:
            return f"calm and gentle {base}"
        elif intensity < 0.7:
            return f"moderate paced {base}"
        else:
            return f"intense and energetic {base}"
    
    def _crossfade_segments(
        self,
        segments: List[np.ndarray],
        fade_duration: float = 1.0,
    ) -> np.ndarray:
        """Crossfade segments together"""
        import numpy as np
        
        sample_rate = 32000
        fade_samples = int (fade_duration * sample_rate)
        
        result = segments[0]
        
        for segment in segments[1:]:
            # Create fade curves
            fade_out = np.linspace(1, 0, fade_samples)
            fade_in = np.linspace(0, 1, fade_samples)
            
            # Apply crossfade
            overlap = len (result) - fade_samples
            result[overlap:] *= fade_out
            segment[:fade_samples] *= fade_in
            result[overlap:] += segment[:fade_samples]
            
            # Append rest of segment
            result = np.concatenate([result, segment[fade_samples:]])
        
        return result

# Example: Game music
def game_music_example():
    """Generate adaptive music for a game"""
    
    music_gen = MusicGenerator()
    adaptive = AdaptiveMusicGenerator (music_gen)
    
    # Simulate game intensity over time
    # 0.0 = calm exploration, 1.0 = intense combat
    intensity_timeline = [
        0.2,  # Start calm
        0.2,
        0.4,  # Slight tension
        0.6,
        0.8,  # Building up
        1.0,  # Combat!
        1.0,
        0.7,  # Winding down
        0.4,
        0.2,  # Back to calm
    ]
    
    adaptive_track = adaptive.generate_adaptive_track(
        base_description="fantasy adventure music with orchestral instruments",
        intensity_levels=intensity_timeline,
        segment_duration=6.0,
    )
    
    music_gen.save_audio (adaptive_track, "game_adaptive_music.wav")
    
    print(f"✅ Generated {len (adaptive_track)/32000:.1f}s adaptive track")
\`\`\`

---

## Commercial Music Generation Platforms

### Comparison

| Platform | Quality | Licensing | Price | Best For |
|----------|---------|-----------|-------|----------|
| **AIVA** | ⭐⭐⭐⭐ | Royalty-free | €15-€200/mo | Soundtracks, background |
| **Amper** | ⭐⭐⭐ | Royalty-free | $15-$50/mo | Content creators |
| **Soundraw** | ⭐⭐⭐ | Royalty-free | $17-$30/mo | Videos, podcasts |
| **Mubert** | ⭐⭐⭐ | Licensing varies | Free-$50/mo | Streaming, ambience |
| **Boomy** | ⭐⭐ | Revenue share | Free | Quick generation |

---

## Audio Effects and Processing

\`\`\`python
"""
Apply effects to generated music
"""

from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range

class AudioProcessor:
    """Post-process generated music"""
    
    @staticmethod
    def normalize_loudness (audio_path: Path, output_path: Path):
        """Normalize audio loudness"""
        audio = AudioSegment.from_file (audio_path)
        normalized = normalize (audio, headroom=0.1)
        normalized.export (output_path, format="wav")
    
    @staticmethod
    def add_reverb (audio_path: Path, output_path: Path, room_size: float = 0.5):
        """Add reverb effect"""
        # Would use pedalboard or similar
        pass
    
    @staticmethod
    def apply_eq (audio_path: Path, output_path: Path):
        """Apply equalization"""
        # Boost or cut specific frequencies
        pass
    
    @staticmethod
    def master_track(
        audio_path: Path,
        output_path: Path,
        target_loudness: float = -14.0,  # LUFS
    ):
        """
        Master audio track to professional standards
        
        Args:
            audio_path: Input audio
            output_path: Output audio
            target_loudness: Target integrated loudness in LUFS
        """
        from pyloudnorm import Meter
        import soundfile as sf
        
        # Load audio
        audio, sr = sf.read (audio_path)
        
        # Measure loudness
        meter = Meter (sr)
        loudness = meter.integrated_loudness (audio)
        
        # Normalize to target
        loudness_normalized = pyloudnorm.normalize.loudness(
            audio, loudness, target_loudness
        )
        
        # Apply limiter to prevent clipping
        limited = np.clip (loudness_normalized, -1.0, 1.0)
        
        # Save
        sf.write (output_path, limited, sr)
        
        print(f"✅ Mastered: {loudness:.1f} LUFS → {target_loudness:.1f} LUFS")
\`\`\`

---

## Production Best Practices

### 1. Prompt Engineering for Music

**Effective prompts include:**
- **Genre**: "jazz", "electronic", "classical"
- **Mood**: "upbeat", "melancholic", "energetic"
- **Instruments**: "piano", "electric guitar", "strings"
- **Tempo**: "fast", "slow", "moderate"
- **Production style**: "lo-fi", "polished", "raw"

**Examples:**
- ✅ "upbeat 80s synthwave with retro drum machines and catchy melody"
- ✅ "calm acoustic guitar instrumental with nature sounds"
- ❌ "music" (too vague)
- ❌ "something cool" (not specific)

### 2. Quality Control

\`\`\`python
"""
Quality checks for generated music
"""

def analyze_audio_quality (audio_path: Path) -> dict:
    """Analyze quality metrics"""
    import librosa
    
    # Load audio
    y, sr = librosa.load (audio_path)
    
    # Compute metrics
    spectral_centroid = librosa.feature.spectral_centroid (y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth (y=y, sr=sr).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate (y).mean()
    rms_energy = librosa.feature.rms (y=y).mean()
    
    return {
        "duration": len (y) / sr,
        "sample_rate": sr,
        "spectral_centroid": float (spectral_centroid),
        "spectral_bandwidth": float (spectral_bandwidth),
        "zero_crossing_rate": float (zero_crossing_rate),
        "rms_energy": float (rms_energy),
    }
\`\`\`

---

## Summary

**Key Takeaways:**
- MusicGen enables text-to-music generation
- AudioGen creates sound effects
- Melody conditioning provides control
- Adaptive music responds to context
- Commercial platforms offer royalty-free music
- Post-processing enhances quality

**Production Checklist:**
- ✅ Use specific, detailed prompts
- ✅ Choose appropriate model size
- ✅ Apply post-processing and mastering
- ✅ Check licensing for commercial use
- ✅ Test with target audience
- ✅ Optimize file sizes for delivery

**Next Steps:**
- Experiment with different prompt styles
- Build adaptive music systems
- Integrate with video generation
- Create sound effect libraries
`,
  exercises: [
    {
      title: 'Exercise 1: Adaptive Soundtrack Generator',
      id: 'music-audio-generation',
      difficulty: 'advanced' as const,
      description:
        'Build a system that generates game soundtracks that adapt to player actions and game state in real-time.',
      hints: [
        'Track game intensity metrics',
        'Generate music segments for different states',
        'Implement smooth crossfading',
        'Cache generated segments for performance',
      ],
    },
    {
      title: 'Exercise 2: Sound Effect Library Builder',
      id: 'music-audio-generation',
      difficulty: 'intermediate' as const,
      description:
        'Create a tool that generates comprehensive sound effect libraries from text descriptions with automatic categorization and tagging.',
      hints: [
        'Generate variations of each sound',
        'Auto-categorize by type',
        'Create preview audio',
        'Export in multiple formats',
      ],
    },
  ],
};
