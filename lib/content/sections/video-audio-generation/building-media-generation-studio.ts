export const buildingMediaGenerationStudio = {
  title: 'Building a Media Generation Studio',
  id: 'building-media-generation-studio',
  content: `
# Building a Media Generation Studio

## Introduction

This section brings everything together: building a complete, production-ready media generation studio that handles video, audio, images, and avatars at scale.

**What We'll Build:**
- Complete system architecture
- GPU resource management
- Job queue system
- Storage and CDN
- Cost tracking and optimization
- Monitoring and alerting
- User interface

This is your capstone project for Module 9.

---

## System Architecture

\`\`\`python
"""
Complete Media Generation Studio Architecture
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List
import asyncio
from pathlib import Path
import json

class MediaType(Enum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    AVATAR = "avatar"

class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class GenerationJob:
    """Media generation job"""
    job_id: str
    user_id: str
    media_type: MediaType
    status: JobStatus
    config: Dict
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_url: Optional[str] = None
    error: Optional[str] = None
    cost: float = 0.0

class MediaGenerationStudio:
    """
    Complete media generation platform
    
    Architecture:
    - FastAPI backend
    - Celery for job queue
    - Redis for caching
    - PostgreSQL for metadata
    - S3 for storage
    - Multiple GPU workers
    """
    
    def __init__(self):
        self.job_queue = JobQueue()
        self.gpu_manager = GPUManager()
        self.storage = StorageManager()
        self.cost_tracker = CostTracker()
    
    async def submit_job(
        self,
        user_id: str,
        media_type: MediaType,
        config: Dict,
    ) -> str:
        """
        Submit generation job
        
        Args:
            user_id: User identifier
            media_type: Type of media to generate
            config: Generation configuration
        
        Returns:
            Job ID
        """
        # Create job
        job = GenerationJob(
            job_id=self._generate_job_id(),
            user_id=user_id,
            media_type=media_type,
            status=JobStatus.PENDING,
            config=config,
            created_at=self._get_timestamp(),
        )
        
        # Validate and estimate cost
        estimated_cost = self.cost_tracker.estimate_cost (media_type, config)
        job.cost = estimated_cost
        
        # Add to queue
        await self.job_queue.enqueue (job)
        
        print(f"✅ Job submitted: {job.job_id}")
        print(f"   Estimated cost: \${estimated_cost:.4f}")

return job.job_id
    
    async def process_job (self, job: GenerationJob):
"""Process a generation job"""

job.status = JobStatus.PROCESSING
job.started_at = self._get_timestamp()

try:
            # Allocate GPU
gpu = await self.gpu_manager.allocate()

print(f"Processing job {job.job_id} on GPU {gpu.id}")
            
            # Route to appropriate generator
if job.media_type == MediaType.VIDEO:
    result = await self._generate_video (job, gpu)
            elif job.media_type == MediaType.AUDIO:
result = await self._generate_audio (job, gpu)
            elif job.media_type == MediaType.IMAGE:
result = await self._generate_image (job, gpu)
            elif job.media_type == MediaType.AVATAR:
result = await self._generate_avatar (job, gpu)
            
            # Upload result
result_url = await self.storage.upload (result, job.job_id)
            
            # Update job
job.status = JobStatus.COMPLETED
job.result_url = result_url
job.completed_at = self._get_timestamp()
            
            # Track actual cost
self.cost_tracker.record_cost (job)

print(f"✅ Job completed: {job.job_id}")
            
        except Exception as e:
job.status = JobStatus.FAILED
job.error = str (e)
print(f"❌ Job failed: {job.job_id} - {e}")
        
        finally:
            # Release GPU
await self.gpu_manager.release (gpu)
    
    async def _generate_video (self, job: GenerationJob, gpu):
"""Generate video"""
        # Route to RunwayGen2, SVD, etc.
    pass
    
    async def _generate_audio (self, job: GenerationJob, gpu):
"""Generate audio"""
        # Route to Whisper, ElevenLabs, MusicGen
pass
    
    async def _generate_image (self, job: GenerationJob, gpu):
"""Generate image"""
        # Route to Stable Diffusion, DALL - E
pass
    
    async def _generate_avatar (self, job: GenerationJob, gpu):
"""Generate avatar video"""
        # Route to Wav2Lip, D - ID
pass
    
    def _generate_job_id (self) -> str:
"""Generate unique job ID"""
import uuid
        return str (uuid.uuid4())
    
    def _get_timestamp (self) -> str:
"""Get current timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat()

# GPU Management
class GPUManager:
"""Manage GPU resources"""
    
    def __init__(self, num_gpus: int = 4):
self.num_gpus = num_gpus
self.available = list (range (num_gpus))
self.in_use = {}
    
    async def allocate (self) -> "GPU":
"""Allocate GPU for job"""
while not self.available:
await asyncio.sleep(1)  # Wait for GPU
        
        gpu_id = self.available.pop(0)
        gpu = GPU(gpu_id)
        self.in_use[gpu_id] = gpu

return gpu
    
    async def release (self, gpu: "GPU"):
"""Release GPU"""
if gpu.id in self.in_use:
            del self.in_use[gpu.id]
self.available.append (gpu.id)

@dataclass
class GPU:
"""GPU resource"""
id: int

# Job Queue
class JobQueue:
"""Job queue with priority"""
    
    def __init__(self):
self.queue = asyncio.Queue()
    
    async def enqueue (self, job: GenerationJob):
"""Add job to queue"""
await self.queue.put (job)
    
    async def dequeue (self) -> GenerationJob:
"""Get next job"""
return await self.queue.get()

# Storage
class StorageManager:
"""Handle file storage"""
    
    def __init__(self, bucket: str = "media-studio"):
self.bucket = bucket
    
    async def upload (self, file_path: Path, job_id: str) -> str:
"""Upload file to S3/storage"""
        # Upload to cloud storage
url = f"https://cdn.example.com/{job_id}/{file_path.name}"
return url

# Cost Tracking
class CostTracker:
"""Track generation costs"""
    
    # Cost per second of generation
COSTS = {
    MediaType.VIDEO: 0.50,  # per second
        MediaType.AUDIO: 0.10,  # per second
        MediaType.IMAGE: 0.02,  # per image
        MediaType.AVATAR: 1.00,  # per minute
}
    
    def estimate_cost (self, media_type: MediaType, config: Dict) -> float:
"""Estimate job cost"""
base_cost = self.COSTS[media_type]

if media_type in [MediaType.VIDEO, MediaType.AUDIO, MediaType.AVATAR]:
    duration = config.get("duration", 10)
return base_cost * duration
        else:
return base_cost
    
    def record_cost (self, job: GenerationJob):
"""Record actual cost"""
        # Log to database for analytics
        pass

# Example usage
async def studio_example():
"""Run media generation studio"""

studio = MediaGenerationStudio()
    
    # Submit video generation job
job_id = await studio.submit_job(
    user_id = "user123",
    media_type = MediaType.VIDEO,
    config = {
        "prompt": "A dog running in a field",
        "duration": 5.0,
        "resolution": "1080p",
    }
)

print(f"Job submitted: {job_id}")

if __name__ == "__main__":
    asyncio.run (studio_example())
\`\`\`

---

## FastAPI Backend

\`\`\`python
"""
FastAPI backend for media studio
"""

from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Media Generation Studio")

class GenerationRequest(BaseModel):
    media_type: str
    config: dict

class JobResponse(BaseModel):
    job_id: str
    status: str
    estimated_cost: float

studio = MediaGenerationStudio()

@app.post("/generate", response_model=JobResponse)
async def generate (request: GenerationRequest, background_tasks: BackgroundTasks):
    """Submit generation job"""
    
    job_id = await studio.submit_job(
        user_id="anonymous",  # Would get from auth
        media_type=MediaType (request.media_type),
        config=request.config,
    )
    
    # Process in background
    background_tasks.add_task (process_job_background, job_id)
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        estimated_cost=0.0,
    )

@app.get("/jobs/{job_id}")
async def get_job_status (job_id: str):
    """Get job status"""
    # Query from database
    return {"job_id": job_id, "status": "processing"}

@app.get("/jobs/{job_id}/result")
async def get_job_result (job_id: str):
    """Get job result"""
    # Return video URL
    return {"url": "https://..."}

async def process_job_background (job_id: str):
    """Process job in background"""
    # Get job from queue and process
    pass

if __name__ == "__main__":
    uvicorn.run (app, host="0.0.0.0", port=8000)
\`\`\`

---

## Monitoring and Alerting

\`\`\`python
"""
Monitoring system for production
"""

from dataclasses import dataclass
from typing import Dict
import time

@dataclass
class Metrics:
    """System metrics"""
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    avg_processing_time: float
    gpu_utilization: float
    total_cost: float

class MonitoringSystem:
    """Monitor studio health"""
    
    def __init__(self):
        self.metrics = Metrics(
            total_jobs=0,
            completed_jobs=0,
            failed_jobs=0,
            avg_processing_time=0.0,
            gpu_utilization=0.0,
            total_cost=0.0,
        )
    
    def record_job_completion (self, job: GenerationJob, duration: float):
        """Record job completion"""
        self.metrics.total_jobs += 1
        
        if job.status == JobStatus.COMPLETED:
            self.metrics.completed_jobs += 1
        else:
            self.metrics.failed_jobs += 1
        
        # Update average processing time
        self.metrics.avg_processing_time = (
            (self.metrics.avg_processing_time * (self.metrics.total_jobs - 1) + duration)
            / self.metrics.total_jobs
        )
        
        # Update cost
        self.metrics.total_cost += job.cost
    
    def get_metrics (self) -> Dict:
        """Get current metrics"""
        return {
            "total_jobs": self.metrics.total_jobs,
            "completed_jobs": self.metrics.completed_jobs,
            "failed_jobs": self.metrics.failed_jobs,
            "success_rate": self.metrics.completed_jobs / max(1, self.metrics.total_jobs),
            "avg_processing_time": self.metrics.avg_processing_time,
            "total_cost": self.metrics.total_cost,
            "avg_cost_per_job": self.metrics.total_cost / max(1, self.metrics.total_jobs),
        }
    
    def check_alerts (self):
        """Check for alert conditions"""
        metrics = self.get_metrics()
        
        # Alert if success rate < 90%
        if metrics["success_rate"] < 0.9:
            self.send_alert("Low success rate", metrics)
        
        # Alert if avg processing time > 60s
        if metrics["avg_processing_time"] > 60:
            self.send_alert("Slow processing", metrics)
    
    def send_alert (self, message: str, metrics: Dict):
        """Send alert (email, Slack, PagerDuty)"""
        print(f"🚨 ALERT: {message}")
        print(f"   Metrics: {json.dumps (metrics, indent=2)}")
\`\`\`

---

## Complete Production Setup

\`\`\`python
"""
Complete production configuration
"""

import os
from pathlib import Path

class ProductionConfig:
    """Production configuration"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")
    DID_API_KEY = os.getenv("DID_API_KEY")
    
    # Infrastructure
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/media_studio")
    S3_BUCKET = os.getenv("S3_BUCKET", "media-studio-prod")
    
    # GPU Config
    NUM_GPUS = int (os.getenv("NUM_GPUS", "4"))
    GPU_MEMORY_PER_JOB = int (os.getenv("GPU_MEMORY_PER_JOB", "8"))  # GB
    
    # Queue Config
    MAX_QUEUE_SIZE = int (os.getenv("MAX_QUEUE_SIZE", "1000"))
    MAX_CONCURRENT_JOBS = int (os.getenv("MAX_CONCURRENT_JOBS", "10"))
    
    # Costs
    MAX_COST_PER_JOB = float (os.getenv("MAX_COST_PER_JOB", "10.0"))
    DAILY_BUDGET = float (os.getenv("DAILY_BUDGET", "1000.0"))
    
    # Monitoring
    METRICS_INTERVAL = int (os.getenv("METRICS_INTERVAL", "60"))  # seconds
    ALERT_EMAIL = os.getenv("ALERT_EMAIL")
    SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK")

# Docker Compose setup
docker_compose = """
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/media_studio
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    deploy:
      replicas: 3
  
  worker:
    build: .
    command: celery -A tasks worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/media_studio
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
      replicas: 4
  
  db:
    image: postgres:14
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=media_studio
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7
    volumes:
      - redis_data:/data
  
  monitoring:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

volumes:
  postgres_data:
  redis_data:
"""
\`\`\`

---

## Deployment Checklist

**Infrastructure:**
- ✅ GPU servers (AWS p3/p4, GCP A100)
- ✅ Load balancer
- ✅ Redis cluster
- ✅ PostgreSQL with replicas
- ✅ S3/CloudFlare R2 storage
- ✅ CDN for delivery

**Security:**
- ✅ API authentication (JWT)
- ✅ Rate limiting per user
- ✅ Input validation
- ✅ Output content moderation
- ✅ HTTPS/TLS
- ✅ Secret management (AWS Secrets Manager)

**Monitoring:**
- ✅ Prometheus + Grafana
- ✅ Error tracking (Sentry)
- ✅ Log aggregation (CloudWatch)
- ✅ Cost tracking dashboard
- ✅ Alert system

**Optimization:**
- ✅ Result caching
- ✅ Model caching on GPUs
- ✅ Batch similar requests
- ✅ Progressive enhancement
- ✅ Automatic scaling

---

## Summary

**Congratulations!** You've completed Module 9 and learned how to build a complete media generation studio.

**What You've Learned:**
- ✅ Video generation (Sora, Runway, SVD)
- ✅ Audio generation (Whisper, ElevenLabs, MusicGen)
- ✅ Image-to-video animation
- ✅ Video editing with AI
- ✅ Avatar generation
- ✅ Complete system architecture
- ✅ Production deployment

**Next Steps:**
- Deploy your own media studio
- Integrate with existing applications
- Build specialized media tools
- Explore Module 10: Multi-Modal AI Systems

**Production Launch Checklist:**
- ✅ Test all generation types
- ✅ Implement monitoring
- ✅ Set up billing
- ✅ Create user documentation
- ✅ Perform load testing
- ✅ Launch! 🚀
`,
  exercises: [
    {
      title: 'Exercise 1: Deploy Production Studio',
      difficulty: 'expert' as const,
      description:
        'Deploy a complete media generation studio to AWS/GCP with GPU workers, job queue, storage, and monitoring. Handle 100+ concurrent users.',
      hints: [
        'Use Docker + Kubernetes',
        'Set up GPU autoscaling',
        'Implement cost limits',
        'Add comprehensive monitoring',
        'Load test before launch',
      ],
    },
    {
      title: 'Exercise 2: Multi-Tenant Studio',
      difficulty: 'expert' as const,
      description:
        'Build a SaaS media studio with multiple tenants, usage-based billing, custom branding, and white-label capabilities.',
      hints: [
        'Implement tenant isolation',
        'Add usage metering',
        'Create billing integration',
        'Build admin dashboard',
        'Support custom domains',
      ],
    },
  ],
};
