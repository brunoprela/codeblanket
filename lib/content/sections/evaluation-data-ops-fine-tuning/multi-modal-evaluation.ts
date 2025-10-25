/**
 * Multi-Modal Evaluation Section
 * Module 16: Evaluation, Data Operations & Fine-Tuning
 */

export const multiModalEvaluation = {
  id: 'multi-modal-evaluation',
  title: 'Multi-Modal Evaluation',
  content: `# Multi-Modal Evaluation

Master evaluating AI systems that generate or understand images, video, and audio.

## Overview: Multi-Modal Evaluation Challenges

Text is easy to evaluate programmatically. **Images, video, audio are hard.**

**Challenges:**
- No single "correct" image/video/audio
- Subjective quality dimensions
- Expensive to evaluate (compute + human time)
- Multiple aspects to assess (visual quality, relevance, safety)

## Image Generation Evaluation

\`\`\`python
from typing import List, Dict, Any
import torch
from PIL import Image

class ImageEvaluator:
    """Evaluate generated images."""
    
    def __init__(self):
        # Load CLIP for text-image alignment
        from transformers import CLIPProcessor, CLIPModel
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    def clip_score(
        self,
        image: Image.Image,
        text_prompt: str
    ) -> float:
        """
        CLIP Score: How well does image match text prompt?
        
        Higher = better text-image alignment
        Range: typically 0-1
        """
        
        inputs = self.clip_processor(
            text=[text_prompt],
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        outputs = self.clip_model(**inputs)
        
        # Cosine similarity between image and text embeddings
        logits_per_image = outputs.logits_per_image
        score = logits_per_image[0][0].item() / 100.0  # Normalize
        
        return score
    
    def aesthetic_score (self, image: Image.Image) -> float:
        """
        Aesthetic quality score.
        Uses pre-trained aesthetic predictor.
        """
        # Placeholder - use aesthetic predictor model
        # https://github.com/christophschuhmann/improved-aesthetic-predictor
        return 0.75  # Placeholder
    
    def safety_check (self, image: Image.Image) -> Dict[str, Any]:
        """
        Check for unsafe/NSFW content.
        """
        # Use safety classifier
        # Placeholder
        return {
            'is_safe': True,
            'nsfw_probability': 0.01,
            'violence_probability': 0.00
        }
    
    def evaluate_batch(
        self,
        generated_images: List[Image.Image],
        prompts: List[str]
    ) -> Dict[str, float]:
        """Evaluate batch of images."""
        
        clip_scores = []
        aesthetic_scores = []
        unsafe_count = 0
        
        for img, prompt in zip (generated_images, prompts):
            # CLIP score
            clip = self.clip_score (img, prompt)
            clip_scores.append (clip)
            
            # Aesthetic score
            aesthetic = self.aesthetic_score (img)
            aesthetic_scores.append (aesthetic)
            
            # Safety
            safety = self.safety_check (img)
            if not safety['is_safe']:
                unsafe_count += 1
        
        return {
            'avg_clip_score': sum (clip_scores) / len (clip_scores),
            'avg_aesthetic_score': sum (aesthetic_scores) / len (aesthetic_scores),
            'safety_pass_rate': 1 - (unsafe_count / len (generated_images))
        }

# Usage
evaluator = ImageEvaluator()

# Evaluate single image
image = Image.open("generated_image.png")
prompt = "A beautiful sunset over the ocean"

score = evaluator.clip_score (image, prompt)
print(f"CLIP Score: {score:.2f}")  # 0.85 = good match

# Batch evaluation
results = evaluator.evaluate_batch (generated_images, prompts)
print(f"Avg CLIP: {results['avg_clip_score']:.2f}")
print(f"Avg Aesthetic: {results['avg_aesthetic_score']:.2f}")
print(f"Safety: {results['safety_pass_rate']:.1%}")
\`\`\`

## Video Evaluation

\`\`\`python
class VideoEvaluator:
    """Evaluate generated videos."""
    
    def temporal_consistency(
        self,
        video_frames: List[Image.Image]
    ) -> float:
        """
        Measure consistency between frames.
        
        High consistency = smooth video
        Low consistency = jittery/glitchy
        """
        from sentence_transformers import SentenceTransformer
        from scipy.spatial.distance import cosine
        
        # Use CLIP to embed each frame
        model = SentenceTransformer('clip-ViT-B-32')
        
        frame_embeddings = []
        for frame in video_frames:
            emb = model.encode (frame)
            frame_embeddings.append (emb)
        
        # Calculate consecutive frame similarity
        similarities = []
        for i in range (len (frame_embeddings) - 1):
            sim = 1 - cosine (frame_embeddings[i], frame_embeddings[i+1])
            similarities.append (sim)
        
        # Average similarity
        consistency = sum (similarities) / len (similarities)
        
        return consistency
    
    def motion_quality(
        self,
        video_frames: List[Image.Image]
    ) -> float:
        """
        Assess motion quality.
        
        Detects:
        - Unnatural movements
        - Static frames (no motion when expected)
        - Too much motion (chaos)
        """
        import cv2
        import numpy as np
        
        # Calculate optical flow between frames
        motion_scores = []
        
        for i in range (len (video_frames) - 1):
            frame1 = np.array (video_frames[i].convert('L'))
            frame2 = np.array (video_frames[i+1].convert('L'))
            
            # Optical flow
            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Motion magnitude
            magnitude = np.sqrt (flow[..., 0]**2 + flow[..., 1]**2)
            avg_motion = magnitude.mean()
            
            motion_scores.append (avg_motion)
        
        # Ideal: moderate motion (not too static, not too chaotic)
        avg_motion = np.mean (motion_scores)
        
        # Score based on deviation from ideal
        ideal_motion = 5.0  # Tuned empirically
        quality = 1 - min (abs (avg_motion - ideal_motion) / ideal_motion, 1.0)
        
        return quality
    
    def evaluate_video(
        self,
        video_frames: List[Image.Image],
        prompt: str
    ) -> Dict[str, float]:
        """Complete video evaluation."""
        
        # Temporal consistency
        consistency = self.temporal_consistency (video_frames)
        
        # Motion quality
        motion = self.motion_quality (video_frames)
        
        # Text-video alignment (use first/middle/last frames)
        key_frames = [video_frames[0], video_frames[len (video_frames)//2], video_frames[-1]]
        img_eval = ImageEvaluator()
        alignment_scores = [img_eval.clip_score (frame, prompt) for frame in key_frames]
        alignment = sum (alignment_scores) / len (alignment_scores)
        
        return {
            'temporal_consistency': consistency,
            'motion_quality': motion,
            'prompt_alignment': alignment,
            'overall_score': (consistency + motion + alignment) / 3
        }

# Usage
video_eval = VideoEvaluator()

# Extract frames from video
video_frames = extract_frames("generated_video.mp4")

scores = video_eval.evaluate_video(
    video_frames,
    prompt="A cat playing with a ball"
)

print(f"Temporal Consistency: {scores['temporal_consistency']:.2f}")
print(f"Motion Quality: {scores['motion_quality']:.2f}")
print(f"Prompt Alignment: {scores['prompt_alignment']:.2f}")
print(f"Overall: {scores['overall_score']:.2f}")
\`\`\`

## Audio Evaluation

\`\`\`python
class AudioEvaluator:
    """Evaluate generated audio."""
    
    def audio_quality_metrics(
        self,
        audio_path: str
    ) -> Dict[str, float]:
        """
        Assess technical audio quality.
        """
        import librosa
        
        # Load audio
        y, sr = librosa.load (audio_path)
        
        # Signal-to-noise ratio
        snr = self._calculate_snr (y)
        
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid (y=y, sr=sr)[0].mean()
        
        # Zero crossing rate (texture)
        zcr = librosa.feature.zero_crossing_rate (y)[0].mean()
        
        return {
            'snr_db': snr,
            'spectral_centroid': spectral_centroid,
            'zero_crossing_rate': zcr
        }
    
    def speech_quality(
        self,
        audio_path: str,
        transcript: str
    ) -> Dict[str, Any]:
        """
        Evaluate speech synthesis quality.
        """
        # Transcribe audio
        from transformers import pipeline
        
        asr = pipeline("automatic-speech-recognition")
        transcription = asr (audio_path)['text']
        
        # Compare to expected transcript
        wer = self._word_error_rate (transcription, transcript)
        
        # Naturalness (use MOS predictor)
        naturalness = self._predict_mos (audio_path)
        
        return {
            'word_error_rate': wer,
            'naturalness_mos': naturalness,  # 1-5 scale
            'intelligible': wer < 0.15  # <15% WER = intelligible
        }
    
    def _calculate_snr (self, signal):
        """Signal-to-noise ratio."""
        # Placeholder
        return 25.0  # dB
    
    def _word_error_rate (self, hypothesis: str, reference: str) -> float:
        """Calculate WER."""
        # Simplified WER calculation
        hyp_words = hypothesis.lower().split()
        ref_words = reference.lower().split()
        
        # Levenshtein distance
        from Levenshtein import distance
        
        wer = distance (hyp_words, ref_words) / len (ref_words)
        return wer
    
    def _predict_mos (self, audio_path: str) -> float:
        """Predict Mean Opinion Score (1-5)."""
        # Use MOS predictor model
        # Placeholder
        return 4.2

# Usage
audio_eval = AudioEvaluator()

# Technical quality
quality = audio_eval.audio_quality_metrics("generated_audio.wav")
print(f"SNR: {quality['snr_db']:.1f} dB")

# Speech quality
speech = audio_eval.speech_quality(
    "generated_speech.wav",
    transcript="Hello, how are you today?"
)
print(f"WER: {speech['word_error_rate']:.2%}")
print(f"Naturalness: {speech['naturalness_mos']:.1f}/5.0")
\`\`\`

## Human Evaluation for Multi-Modal

\`\`\`python
class MultiModalHumanEval:
    """Collect human judgments for multi-modal outputs."""
    
    def create_comparison_task(
        self,
        prompt: str,
        outputs: List[Dict]
    ) -> Dict[str, Any]:
        """
        Create pairwise comparison task.
        
        outputs: [
            {'id': 'model_a', 'image': Image, 'video': ...},
            {'id': 'model_b', 'image': Image, 'video': ...}
        ]
        """
        
        return {
            'prompt': prompt,
            'outputs': outputs,
            'questions': [
                {
                    'dimension': 'overall_quality',
                    'question': 'Which output is better overall?',
                    'type': 'choice',
                    'options': ['A', 'B', 'Tie']
                },
                {
                    'dimension': 'prompt_alignment',
                    'question': 'Which better matches the prompt?',
                    'type': 'choice',
                    'options': ['A', 'B', 'Tie']
                },
                {
                    'dimension': 'visual_quality',
                    'question': 'Which has better visual quality?',
                    'type': 'choice',
                    'options': ['A', 'B', 'Tie']
                }
            ]
        }
    
    def calculate_elo_ratings(
        self,
        comparisons: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate ELO ratings from pairwise comparisons.
        
        comparisons: [
            {'model_a': 'gpt', 'model_b': 'claude', 'winner': 'model_a'},
            ...
        ]
        """
        # Initialize ELO ratings
        ratings = {}
        K = 32  # ELO K-factor
        
        for comp in comparisons:
            model_a = comp['model_a']
            model_b = comp['model_b']
            winner = comp['winner']
            
            # Initialize if new
            if model_a not in ratings:
                ratings[model_a] = 1500
            if model_b not in ratings:
                ratings[model_b] = 1500
            
            # Expected scores
            expected_a = 1 / (1 + 10 ** ((ratings[model_b] - ratings[model_a]) / 400))
            expected_b = 1 - expected_a
            
            # Actual scores
            if winner == 'model_a':
                actual_a, actual_b = 1, 0
            elif winner == 'model_b':
                actual_a, actual_b = 0, 1
            else:  # Tie
                actual_a, actual_b = 0.5, 0.5
            
            # Update ratings
            ratings[model_a] += K * (actual_a - expected_a)
            ratings[model_b] += K * (actual_b - expected_b)
        
        return ratings

# Usage
human_eval = MultiModalHumanEval()

# Create comparison task
task = human_eval.create_comparison_task(
    prompt="A sunset over mountains",
    outputs=[
        {'id': 'stable_diffusion', 'image': sd_image},
        {'id': 'dall_e_3', 'image': dalle_image}
    ]
)

# Collect human judgments (via annotation platform)
# ...

# Calculate ELO rankings
comparisons = [
    {'model_a': 'stable_diffusion', 'model_b': 'dall_e_3', 'winner': 'model_b'},
    # ... more comparisons
]

rankings = human_eval.calculate_elo_ratings (comparisons)
print("ELO Rankings:")
for model, rating in sorted (rankings.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model}: {rating:.0f}")
\`\`\`

## Production Checklist

✅ **Automated Metrics**
- [ ] CLIP scores for image-text alignment
- [ ] Aesthetic quality scores
- [ ] Safety/NSFW detection
- [ ] Technical quality metrics (SNR, WER, etc.)

✅ **Human Evaluation**
- [ ] Pairwise comparison setup
- [ ] Rating dimensions defined
- [ ] Sufficient sample size
- [ ] Inter-annotator agreement checked

✅ **Multi-Modal Specific**
- [ ] Temporal consistency for video
- [ ] Motion quality assessment
- [ ] Audio intelligibility verified
- [ ] Cross-modal coherence checked

✅ **Continuous Monitoring**
- [ ] Quality metrics dashboard
- [ ] Regression detection
- [ ] User feedback integration
- [ ] Model comparison tracking

## Next Steps

You now understand multi-modal evaluation. Next, learn:
- Continuous evaluation & monitoring
- Building complete evaluation platforms
- Production deployment strategies
`,
};
