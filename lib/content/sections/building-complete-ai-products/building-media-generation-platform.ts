export const buildingMediaGenerationPlatform = {
  title: 'Building a Media Generation Platform',
  id: 'building-media-generation-platform',
  content: `
# Building a Media Generation Platform

## Introduction

A media generation platform combines image, video, and audio generation into a unified system. Think Midjourney meets Runway meets ElevenLabs—users can create any type of media through a single interface.

This section covers building a production platform that:
- Generates images from text (DALL-E, Stable Diffusion)
- Creates videos from text or images (Runway, Stable Video)
- Synthesizes speech and music (ElevenLabs, MusicGen)
- Manages GPU resources efficiently
- Handles async job queues at scale
- Provides gallery and asset management

### Architecture Overview

\`\`\`
┌──────────────────────────────────────────────────────────┐
│         Media Generation Platform                         │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────┐      ┌─────────────┐      ┌────────────┐ │
│  │  Web UI  │─────▶│  FastAPI    │─────▶│   Redis    │ │
│  │ (Next.js)│      │   Server    │      │   Queue    │ │
│  └──────────┘      └─────────────┘      └─────┬──────┘ │
│                                                 │         │
│                           ┌─────────────────────┼────────┐│
│                           │                     │        ││
│                    ┌──────▼──────┐      ┌──────▼──────┐││
│                    │   Image     │      │   Video     │││
│                    │   Worker    │      │   Worker    │││
│                    │   (GPU)     │      │   (GPU)     │││
│                    └──────┬──────┘      └──────┬──────┘││
│                           │                     │        ││
│                    ┌──────▼──────┐      ┌──────▼──────┐││
│                    │   Audio     │      │  Storage    │││
│                    │   Worker    │      │    (S3)     │││
│                    │   (CPU)     │      └─────────────┘││
│                    └─────────────┘                      ││
└──────────────────────────────────────────────────────────┘
\`\`\`

---

## Queue System

### Job Queue with Priority

\`\`\`python
"""
Job queue system for media generation
"""

from enum import Enum
from celery import Celery
import redis

class JobType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"

class JobPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3

class MediaGenerationJob(BaseModel):
    job_id: str
    user_id: str
    job_type: JobType
    priority: JobPriority
    params: Dict[str, any]
    created_at: datetime
    status: str = "queued"

# Configure Celery with Redis
celery_app = Celery(
    'media_generation',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

celery_app.conf.update(
    task_routes={
        'generate_image': {'queue': 'gpu_high'},
        'generate_video': {'queue': 'gpu_low'},
        'generate_audio': {'queue': 'cpu'},
    },
    task_priority_steps=[0, 1, 2, 3],  # Map to JobPriority
)

@celery_app.task (bind=True, priority=1)
def generate_image (self, job: MediaGenerationJob):
    """Generate image (GPU task)"""
    try:
        # Update status
        update_job_status (job.job_id, "processing")
        
        # Generate image
        result = run_image_generation (job.params)
        
        # Upload to S3
        url = upload_to_s3(result, job.job_id)
        
        # Update status
        update_job_status (job.job_id, "completed", {"url": url})
        
        return {"url": url}
    
    except Exception as e:
        update_job_status (job.job_id, "failed", {"error": str (e)})
        raise

@celery_app.task (bind=True, priority=0)
def generate_video (self, job: MediaGenerationJob):
    """Generate video (GPU task, lower priority)"""
    # Similar to image generation
    pass

@celery_app.task (bind=True, priority=1)
def generate_audio (self, job: MediaGenerationJob):
    """Generate audio (CPU task)"""
    # Audio generation
    pass
\`\`\`

---

## Image Generation

### Multi-Model Image Generator

\`\`\`python
"""
Image generation with multiple models
"""

from openai import AsyncOpenAI
from stability_sdk import client as stability_client
import replicate

class ImageGenerator:
    """
    Unified interface for multiple image generation models
    """
    
    def __init__(self):
        self.openai_client = AsyncOpenAI()
        self.stability = stability_client.StabilityInference(
            key=os.getenv("STABILITY_KEY")
        )
    
    async def generate(
        self,
        prompt: str,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "natural"
    ) -> Dict[str, any]:
        """
        Generate image with specified model
        """
        if model == "dall-e-3":
            return await self._generate_dalle (prompt, size, quality, style)
        elif model == "stable-diffusion":
            return await self._generate_sd (prompt, size)
        elif model == "midjourney":
            return await self._generate_midjourney (prompt)
        else:
            raise ValueError (f"Unknown model: {model}")
    
    async def _generate_dalle(
        self,
        prompt: str,
        size: str,
        quality: str,
        style: str
    ) -> Dict:
        """Generate with DALL-E 3"""
        response = await self.openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=1
        )
        
        return {
            "url": response.data[0].url,
            "revised_prompt": response.data[0].revised_prompt,
            "model": "dall-e-3",
            "cost": 0.04 if quality == "standard" else 0.08
        }
    
    async def _generate_sd (self, prompt: str, size: str) -> Dict:
        """Generate with Stable Diffusion via Replicate"""
        output = replicate.run(
            "stability-ai/sdxl:latest",
            input={
                "prompt": prompt,
                "width": int (size.split("x")[0]),
                "height": int (size.split("x")[1])
            }
        )
        
        return {
            "url": output[0],
            "model": "stable-diffusion-xl",
            "cost": 0.005
        }
    
    async def _generate_midjourney (self, prompt: str) -> Dict:
        """Generate with Midjourney-style model"""
        # Use Midjourney API proxy or similar model
        pass

# Usage
generator = ImageGenerator()
result = await generator.generate(
    prompt="A futuristic city at sunset, cyberpunk style",
    model="dall-e-3",
    quality="hd"
)
\`\`\`

---

## Video Generation

### Text-to-Video Pipeline

\`\`\`python
"""
Video generation with Runway and Stable Video Diffusion
"""

import requests
from diffusers import StableVideoDiffusionPipeline
import torch

class VideoGenerator:
    """
    Generate videos from text or images
    """
    
    def __init__(self):
        self.runway_key = os.getenv("RUNWAY_API_KEY")
        
        # Load Stable Video Diffusion locally
        self.svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        self.svd_pipeline.to("cuda")
    
    async def generate_from_text(
        self,
        prompt: str,
        duration: int = 4,
        fps: int = 24
    ) -> str:
        """
        Generate video from text using Runway Gen-2
        """
        # Call Runway API
        response = requests.post(
            "https://api.runwayml.com/v1/generate",
            headers={"Authorization": f"Bearer {self.runway_key}"},
            json={
                "prompt": prompt,
                "duration": duration,
                "resolution": "720p"
            }
        )
        
        data = response.json()
        
        # Poll for completion
        job_id = data["id"]
        video_url = await self._poll_runway_job (job_id)
        
        return video_url
    
    async def generate_from_image(
        self,
        image_path: str,
        motion_strength: float = 0.5
    ) -> str:
        """
        Animate static image with Stable Video Diffusion
        """
        from PIL import Image
        
        # Load image
        image = Image.open (image_path).convert("RGB")
        image = image.resize((1024, 576))
        
        # Generate video frames
        frames = self.svd_pipeline(
            image,
            decode_chunk_size=8,
            motion_bucket_id=127 * motion_strength,
            noise_aug_strength=0.02
        ).frames[0]
        
        # Save as video
        output_path = f"/tmp/video_{uuid.uuid4()}.mp4"
        export_to_video (frames, output_path, fps=7)
        
        return output_path
    
    async def _poll_runway_job (self, job_id: str) -> str:
        """Poll Runway until video is ready"""
        import asyncio
        
        while True:
            response = requests.get(
                f"https://api.runwayml.com/v1/jobs/{job_id}",
                headers={"Authorization": f"Bearer {self.runway_key}"}
            )
            
            data = response.json()
            
            if data["status"] == "completed":
                return data["output"]["url"]
            elif data["status"] == "failed":
                raise Exception (f"Video generation failed: {data['error']}")
            
            await asyncio.sleep(5)

# Usage
video_gen = VideoGenerator()
video_url = await video_gen.generate_from_text(
    prompt="A drone flying over a futuristic city",
    duration=4
)
\`\`\`

---

## Audio Generation

### Speech and Music Synthesis

\`\`\`python
"""
Audio generation with ElevenLabs and MusicGen
"""

from elevenlabs import generate, set_api_key, voices
from audiocraft.models import MusicGen
import torch

class AudioGenerator:
    """
    Generate speech and music
    """
    
    def __init__(self):
        set_api_key (os.getenv("ELEVENLABS_API_KEY"))
        
        # Load MusicGen
        self.music_model = MusicGen.get_pretrained('facebook/musicgen-medium')
    
    async def generate_speech(
        self,
        text: str,
        voice: str = "Adam",
        model: str = "eleven_monolingual_v1"
    ) -> bytes:
        """
        Generate speech with ElevenLabs
        """
        audio = generate(
            text=text,
            voice=voice,
            model=model
        )
        
        return audio
    
    async def generate_music(
        self,
        description: str,
        duration: int = 10
    ) -> torch.Tensor:
        """
        Generate music with MusicGen
        """
        self.music_model.set_generation_params(
            duration=duration,
            temperature=1.0
        )
        
        wav = self.music_model.generate([description])
        
        return wav[0]
    
    def list_voices (self) -> List[str]:
        """List available voices"""
        return [v.name for v in voices()]

# Usage
audio_gen = AudioGenerator()

# Speech
speech = await audio_gen.generate_speech(
    text="Hello, this is AI-generated speech.",
    voice="Bella"
)

# Music
music = await audio_gen.generate_music(
    description="Upbeat electronic dance music with synth leads",
    duration=30
)
\`\`\`

---

## GPU Resource Management

### Efficient GPU Allocation

\`\`\`python
"""
GPU resource manager for media generation
"""

import nvidia_smi
from contextlib import contextmanager
import queue
import threading

class GPUManager:
    """
    Manage GPU resources across workers
    """
    
    def __init__(self, num_gpus: int = None):
        nvidia_smi.nvmlInit()
        self.num_gpus = num_gpus or nvidia_smi.nvmlDeviceGetCount()
        self.gpu_queue = queue.Queue()
        
        # Initialize queue with GPU IDs
        for i in range (self.num_gpus):
            self.gpu_queue.put (i)
    
    @contextmanager
    def allocate_gpu (self):
        """
        Context manager to allocate GPU
        """
        gpu_id = self.gpu_queue.get()  # Wait for available GPU
        
        try:
            # Set CUDA device
            os.environ['CUDA_VISIBLE_DEVICES'] = str (gpu_id)
            yield gpu_id
        finally:
            # Release GPU back to pool
            self.gpu_queue.put (gpu_id)
    
    def get_gpu_memory (self, gpu_id: int) -> Dict:
        """Get GPU memory usage"""
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex (gpu_id)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo (handle)
        
        return {
            "total": info.total,
            "used": info.used,
            "free": info.free,
            "percent": (info.used / info.total) * 100
        }
    
    def get_all_gpu_stats (self) -> List[Dict]:
        """Get stats for all GPUs"""
        stats = []
        for i in range (self.num_gpus):
            stats.append (self.get_gpu_memory (i))
        return stats

# Usage in worker
gpu_manager = GPUManager()

@celery_app.task
def generate_image_task (prompt: str):
    with gpu_manager.allocate_gpu() as gpu_id:
        print(f"Using GPU {gpu_id}")
        # Run model on allocated GPU
        result = model.generate (prompt)
    return result
\`\`\`

---

## Storage & CDN

### Asset Management

\`\`\`python
"""
Storage and CDN management for generated media
"""

import boto3
from botocore.exceptions import ClientError

class MediaStorage:
    """
    Store and serve generated media
    """
    
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('AWS_SECRET_KEY')
        )
        self.bucket = os.getenv('S3_BUCKET')
        self.cdn_url = os.getenv('CDN_URL')
    
    def upload(
        self,
        file_data: bytes,
        file_name: str,
        content_type: str,
        metadata: Dict = None
    ) -> str:
        """
        Upload file to S3 with CDN delivery
        """
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=file_name,
                Body=file_data,
                ContentType=content_type,
                Metadata=metadata or {},
                ACL='public-read'
            )
            
            # Return CDN URL
            return f"{self.cdn_url}/{file_name}"
        
        except ClientError as e:
            raise Exception (f"Upload failed: {e}")
    
    def generate_presigned_url(
        self,
        file_name: str,
        expiration: int = 3600
    ) -> str:
        """Generate temporary download URL"""
        return self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket, 'Key': file_name},
            ExpiresIn=expiration
        )
    
    def delete (self, file_name: str):
        """Delete file from S3"""
        self.s3.delete_object(Bucket=self.bucket, Key=file_name)

# Usage
storage = MediaStorage()

# Upload generated image
with open('generated.png', 'rb') as f:
    url = storage.upload(
        f.read(),
        'images/generated_123.png',
        'image/png',
        metadata={'user_id': 'user_123', 'prompt': 'sunset'}
    )

print(f"Image available at: {url}")
\`\`\`

---

## API Endpoints

### FastAPI Media Generation API

\`\`\`python
"""
FastAPI endpoints for media generation
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI(title="Media Generation Platform")

class GenerateImageRequest(BaseModel):
    prompt: str
    model: str = "dall-e-3"
    size: str = "1024x1024"
    quality: str = "standard"

class GenerateVideoRequest(BaseModel):
    prompt: str
    duration: int = 4
    fps: int = 24

class JobResponse(BaseModel):
    job_id: str
    status: str
    estimated_time_seconds: int

@app.post("/api/images/generate", response_model=JobResponse)
async def generate_image (request: GenerateImageRequest):
    """
    Queue image generation job
    """
    job_id = str (uuid.uuid4())
    
    # Create job
    job = MediaGenerationJob(
        job_id=job_id,
        user_id="user_123",  # Get from auth
        job_type=JobType.IMAGE,
        priority=JobPriority.NORMAL,
        params=request.dict()
    )
    
    # Queue job
    generate_image.apply_async(
        args=[job],
        task_id=job_id,
        priority=job.priority.value
    )
    
    return JobResponse(
        job_id=job_id,
        status="queued",
        estimated_time_seconds=30
    )

@app.get("/api/jobs/{job_id}")
async def get_job_status (job_id: str):
    """
    Get job status
    """
    result = celery_app.AsyncResult (job_id)
    
    if result.ready():
        if result.successful():
            return {
                "status": "completed",
                "result": result.result
            }
        else:
            return {
                "status": "failed",
                "error": str (result.info)
            }
    else:
        return {
            "status": "processing",
            "progress": result.info.get('progress', 0) if result.info else 0
        }

@app.post("/api/videos/generate", response_model=JobResponse)
async def generate_video (request: GenerateVideoRequest):
    """Queue video generation"""
    job_id = str (uuid.uuid4())
    
    job = MediaGenerationJob(
        job_id=job_id,
        user_id="user_123",
        job_type=JobType.VIDEO,
        priority=JobPriority.NORMAL,
        params=request.dict()
    )
    
    generate_video.apply_async (args=[job], task_id=job_id)
    
    return JobResponse(
        job_id=job_id,
        status="queued",
        estimated_time_seconds=120
    )

@app.get("/api/gallery")
async def get_user_gallery(
    user_id: str,
    page: int = 1,
    per_page: int = 20
):
    """
    Get user's generated media gallery
    """
    # Query database for user's generations
    items = db.query(
        "SELECT * FROM generations WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (user_id, per_page, (page - 1) * per_page)
    )
    
    return {
        "items": items,
        "page": page,
        "per_page": per_page,
        "total": db.count("generations", user_id=user_id)
    }
\`\`\`

---

## Frontend Integration

### React Media Generation UI

\`\`\`typescript
/**
 * React component for media generation
 */

import React, { useState } from 'react';
import axios from 'axios';

interface Generation {
  id: string;
  type: 'image' | 'video' | 'audio';
  url: string;
  prompt: string;
  created_at: string;
  status: 'completed' | 'processing' | 'failed';
}

export const MediaGenerator: React.FC = () => {
  const [prompt, setPrompt] = useState(');
  const [type, setType] = useState<'image' | 'video'>('image');
  const [loading, setLoading] = useState (false);
  const [generations, setGenerations] = useState<Generation[]>([]);

  const generate = async () => {
    setLoading (true);
    
    try {
      const response = await axios.post(\`/api/\${type}s/generate\`, {
        prompt,
        model: 'dall-e-3'
      });
      
      const { job_id } = response.data;
      
      // Poll for completion
      await pollJobStatus (job_id);
      
    } catch (error) {
      console.error('Generation failed:', error);
    } finally {
      setLoading (false);
    }
  };

  const pollJobStatus = async (jobId: string) => {
    const interval = setInterval (async () => {
      const response = await axios.get(\`/api/jobs/\${jobId}\`);
      
      if (response.data.status === 'completed') {
        clearInterval (interval);
        // Add to gallery
        const newGen: Generation = {
          id: jobId,
          type,
          url: response.data.result.url,
          prompt,
          created_at: new Date().toISOString(),
          status: 'completed'
        };
        setGenerations([newGen, ...generations]);
      } else if (response.data.status === 'failed') {
        clearInterval (interval);
        alert('Generation failed');
      }
    }, 2000);
  };

  return (
    <div className="media-generator">
      <div className="controls">
        <select value={type} onChange={e => setType (e.target.value as any)}>
          <option value="image">Image</option>
          <option value="video">Video</option>
        </select>
        
        <textarea
          value={prompt}
          onChange={e => setPrompt (e.target.value)}
          placeholder="Describe what you want to create..."
          rows={3}
        />
        
        <button onClick={generate} disabled={loading || !prompt}>
          {loading ? 'Generating...' : 'Generate'}
        </button>
      </div>

      <div className="gallery">
        {generations.map (gen => (
          <div key={gen.id} className="generation-item">
            {gen.type === 'image' ? (
              <img src={gen.url} alt={gen.prompt} />
            ) : (
              <video src={gen.url} controls />
            )}
            <p>{gen.prompt}</p>
          </div>
        ))}
      </div>
    </div>
  );
};
\`\`\`

---

## Conclusion

Building a production media generation platform requires:

1. **Queue System**: Handle async jobs with priorities
2. **Multi-Model Support**: DALL-E, Stable Diffusion, Runway, etc.
3. **GPU Management**: Efficient resource allocation
4. **Storage & CDN**: Fast global delivery
5. **Cost Tracking**: Monitor per-generation costs
6. **Gallery System**: User asset management
7. **Monitoring**: Track queue depth, GPU utilization

**Key Technologies**:
- **Celery**: Job queue
- **Redis**: Message broker
- **FastAPI**: API server
- **S3 + CloudFront**: Storage and CDN
- **Multiple AI APIs**: OpenAI, Stability AI, Replicate, ElevenLabs

This powers platforms like Midjourney, Runway, and Leonardo.ai.
`,
};
