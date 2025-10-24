/**
 * Building an Image Generation Platform Section
 * Module 8: Image Generation & Computer Vision
 */

export const buildingimmagegenerationplatformSection = {
  id: 'building-image-generation-platform',
  title: 'Building an Image Generation Platform',
  content: `# Building an Image Generation Platform

Build a complete, production-ready image generation platform from scratch.

## Overview: Architecture Design

A production image generation platform needs:
- **Queue system**: Handle async generation
- **Multiple models**: SD, SDXL, DALL-E support
- **Storage**: Efficient image storage and CDN
- **User management**: Authentication, quotas
- **API**: RESTful + WebSocket for real-time
- **UI**: Intuitive generation interface
- **Monitoring**: Usage, costs, performance
- **Scaling**: Handle thousands of users

### System Architecture

\`\`\`python
platform_architecture = {
    "frontend": {
        "tech": "React/Next.js",
        "features": [
            "Image generation UI",
            "Real-time generation status",
            "Gallery management",
            "Parameter controls",
            "User dashboard"
        ]
    },
    
    "backend_api": {
        "tech": "FastAPI (Python)",
        "responsibilities": [
            "Request validation",
            "Queue management",
            "User authentication",
            "Rate limiting",
            "Result retrieval"
        ]
    },
    
    "worker_service": {
        "tech": "Celery + GPU workers",
        "responsibilities": [
            "Image generation",
            "Model loading",
            "Image processing",
            "Storage upload"
        ]
    },
    
    "storage": {
        "tech": "S3 + CloudFront CDN",
        "purpose": "Store and serve generated images"
    },
    
    "database": {
        "tech": "PostgreSQL",
        "stores": [
            "Users and auth",
            "Generation history",
            "Model parameters",
            "Usage metrics"
        ]
    },
    
    "queue": {
        "tech": "Redis/RabbitMQ",
        "purpose": "Async job queue"
    }
}
\`\`\`

## Core Components

### 1. Generation Worker

\`\`\`python
from celery import Celery
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
from PIL import Image
import boto3
from typing import Optional
import logging

# Initialize Celery
celery_app = Celery(
    'image_generation',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

class GenerationWorker:
    """
    Worker for image generation.
    """
    
    def __init__(self):
        self.models = {}
        self.s3_client = boto3.client('s3')
        self.bucket_name = 'generated-images'
        
        # Preload models
        self.load_models()
    
    def load_models(self):
        """Preload frequently used models."""
        logging.info("Loading models...")
        
        # SD 2.1
        self.models['sd21'] = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16
        ).to("cuda")
        
        # SDXL
        self.models['sdxl'] = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
        
        logging.info("Models loaded")
    
    def generate_image(
        self,
        model_name: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Generate single image."""
        pipe = self.models.get(model_name)
        if not pipe:
            raise ValueError(f"Unknown model: {model_name}")
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        return image
    
    def upload_to_s3(
        self,
        image: Image.Image,
        key: str
    ) -> str:
        """Upload image to S3 and return URL."""
        from io import BytesIO
        
        # Convert to bytes
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Upload
        self.s3_client.upload_fileobj(
            buffer,
            self.bucket_name,
            key,
            ExtraArgs={'ContentType': 'image/png'}
        )
        
        # Return CDN URL
        return f"https://cdn.example.com/{key}"

# Celery tasks
worker = GenerationWorker()

@celery_app.task(bind=True)
def generate_task(
    self,
    job_id: str,
    user_id: str,
    params: dict
) -> dict:
    """
    Celery task for image generation.
    """
    try:
        # Update status
        self.update_state(state='PROCESSING')
        
        # Generate
        image = worker.generate_image(**params)
        
        # Upload
        key = f"generations/{user_id}/{job_id}.png"
        url = worker.upload_to_s3(image, key)
        
        return {
            'status': 'completed',
            'url': url,
            'job_id': job_id
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e),
            'job_id': job_id
        }
\`\`\`

### 2. API Server

\`\`\`python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime

app = FastAPI(title="Image Generation API")
security = HTTPBearer()

# Models
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    model: str = "sd21"
    width: int = 512
    height: int = 512
    steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None

class GenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    url: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

# Database (simplified - use proper ORM in production)
class Database:
    """Simple in-memory database (use PostgreSQL in production)."""
    
    def __init__(self):
        self.jobs = {}
        self.users = {}
    
    def create_job(self, user_id: str, params: dict) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            'user_id': user_id,
            'status': 'queued',
            'params': params,
            'created_at': datetime.now(),
            'url': None,
            'error': None
        }
        return job_id
    
    def update_job(self, job_id: str, updates: dict):
        if job_id in self.jobs:
            self.jobs[job_id].update(updates)
    
    def get_job(self, job_id: str) -> dict:
        return self.jobs.get(job_id)
    
    def get_user_quota(self, user_id: str) -> dict:
        user = self.users.get(user_id, {
            'daily_limit': 100,
            'used_today': 0,
            'tier': 'free'
        })
        return user

db = Database()

# Auth dependency
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """Verify auth token and return user ID."""
    token = credentials.credentials
    # TODO: Verify JWT token
    # For now, return mock user ID
    return "user_123"

# Endpoints
@app.post("/generate", response_model=GenerationResponse)
async def generate_image(
    request: GenerationRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Queue image generation.
    """
    # Check quota
    quota = db.get_user_quota(user_id)
    if quota['used_today'] >= quota['daily_limit']:
        raise HTTPException(
            status_code=429,
            detail="Daily quota exceeded"
        )
    
    # Create job
    job_id = db.create_job(user_id, request.dict())
    
    # Queue generation
    task = generate_task.delay(
        job_id=job_id,
        user_id=user_id,
        params=request.dict()
    )
    
    # Update quota
    quota['used_today'] += 1
    
    return GenerationResponse(
        job_id=job_id,
        status="queued",
        message="Generation queued successfully"
    )

@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    user_id: str = Depends(get_current_user)
):
    """
    Get generation job status.
    """
    job = db.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job['user_id'] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Check Celery task status
    task_result = generate_task.AsyncResult(job_id)
    
    if task_result.ready():
        result = task_result.result
        db.update_job(job_id, {
            'status': result['status'],
            'url': result.get('url'),
            'error': result.get('error'),
            'completed_at': datetime.now()
        })
        job = db.get_job(job_id)
    
    return JobStatus(**job)

@app.get("/history")
async def get_generation_history(
    limit: int = 20,
    user_id: str = Depends(get_current_user)
):
    """
    Get user's generation history.
    """
    user_jobs = [
        job for job in db.jobs.values()
        if job['user_id'] == user_id
    ]
    
    # Sort by created_at, newest first
    user_jobs.sort(key=lambda x: x['created_at'], reverse=True)
    
    return user_jobs[:limit]

@app.get("/quota")
async def get_quota(user_id: str = Depends(get_current_user)):
    """Get user's quota information."""
    return db.get_user_quota(user_id)

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
\`\`\`

### 3. WebSocket for Real-Time Updates

\`\`\`python
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        self.active_connections[user_id].discard(websocket)
    
    async def send_update(self, user_id: str, message: dict):
        """Send update to all user's connections."""
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except:
                    pass

manager = ConnectionManager()

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time updates.
    """
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            
            # Handle ping/pong
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)

# Modify generate_task to send updates
@celery_app.task(bind=True)
def generate_task_with_updates(
    self,
    job_id: str,
    user_id: str,
    params: dict
) -> dict:
    """Generate with real-time updates."""
    import asyncio
    
    async def send_update(status: str, progress: float = 0):
        await manager.send_update(user_id, {
            'job_id': job_id,
            'status': status,
            'progress': progress
        })
    
    try:
        # Send updates throughout process
        asyncio.run(send_update('starting', 0))
        
        image = worker.generate_image(**params)
        
        asyncio.run(send_update('uploading', 90))
        
        key = f"generations/{user_id}/{job_id}.png"
        url = worker.upload_to_s3(image, key)
        
        asyncio.run(send_update('completed', 100))
        
        return {'status': 'completed', 'url': url}
    
    except Exception as e:
        asyncio.run(send_update('failed', 0))
        return {'status': 'failed', 'error': str(e)}
\`\`\`

### 4. Frontend React Component

\`\`\`typescript
// ImageGenerator.tsx
import React, { useState, useEffect } from 'react';

interface GenerationParams {
  prompt: string;
  negativePrompt?: string;
  model: string;
  width: number;
  height: number;
  steps: number;
  guidanceScale: number;
}

const ImageGenerator: React.FC = () => {
  const [params, setParams] = useState<GenerationParams>({
    prompt: ',
    negativePrompt: ',
    model: 'sd21',
    width: 512,
    height: 512,
    steps: 30,
    guidanceScale: 7.5,
  });
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    // Connect to WebSocket
    const websocket = new WebSocket('ws://localhost:8000/ws/user_123');
    
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.status === 'completed') {
        setResult(data.url);
        setLoading(false);
      } else if (data.status === 'failed') {
        alert('Generation failed: ' + data.error);
        setLoading(false);
      }
    };
    
    setWs(websocket);
    
    return () => websocket.close();
  }, []);

  const generate = async () => {
    setLoading(true);
    setResult(null);
    
    try {
      const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer YOUR_TOKEN',
        },
        body: JSON.stringify(params),
      });
      
      const data = await response.json();
      console.log('Job queued:', data.job_id);
      
    } catch (error) {
      console.error('Error:', error);
      setLoading(false);
    }
  };

  return (
    <div className="image-generator">
      <h1>AI Image Generator</h1>
      
      <div className="controls">
        <textarea
          placeholder="Describe the image..."
          value={params.prompt}
          onChange={(e) => setParams({ ...params, prompt: e.target.value })}
          rows={4}
        />
        
        <button onClick={generate} disabled={loading || !params.prompt}>
          {loading ? 'Generating...' : 'Generate'}
        </button>
      </div>
      
      {result && (
        <div className="result">
          <img src={result} alt="Generated" />
          <a href={result} download>Download</a>
        </div>
      )}
    </div>
  );
};

export default ImageGenerator;
\`\`\`

## Deployment Architecture

\`\`\`python
deployment_architecture = {
    "infrastructure": {
        "api_servers": {
            "service": "ECS/Kubernetes",
            "scaling": "Auto-scale based on CPU",
            "count": "2-10 instances"
        },
        
        "worker_nodes": {
            "service": "EC2 with GPU (g4dn.xlarge)",
            "gpu": "NVIDIA T4",
            "scaling": "Manual or spot instances",
            "count": "2-20 based on demand"
        },
        
        "redis": {
            "service": "ElastiCache",
            "purpose": "Queue and cache"
        },
        
        "postgres": {
            "service": "RDS",
            "replication": "Multi-AZ"
        },
        
        "storage": {
            "service": "S3 + CloudFront",
            "lifecycle": "Move to Glacier after 90 days"
        }
    },
    
    "monitoring": {
        "tools": ["Prometheus", "Grafana", "Sentry"],
        "metrics": [
            "Generation queue length",
            "Average generation time",
            "Success/failure rates",
            "GPU utilization",
            "Cost per image"
        ]
    },
    
    "costs": {
        "monthly_estimate": {
            "api_servers": "$200 (2 t3.medium)",
            "workers": "$1,000 (2 g4dn.xlarge)",
            "database": "$100 (RDS)",
            "storage": "$50 (S3 + CloudFront)",
            "total": "~$1,350/month base"
        }
    }
}
\`\`\`

## Production Checklist

\`\`\`python
production_checklist = {
    "functionality": [
        "✓ Multiple model support",
        "✓ Queue system working",
        "✓ Image storage and CDN",
        "✓ User authentication",
        "✓ Quota management",
        "✓ Generation history"
    ],
    
    "performance": [
        "✓ Preload models on workers",
        "✓ Connection pooling",
        "✓ CDN caching",
        "✓ Database indexes",
        "✓ Redis caching",
        "✓ Async processing"
    ],
    
    "reliability": [
        "✓ Error handling",
        "✓ Retry logic",
        "✓ Health checks",
        "✓ Graceful degradation",
        "✓ Backup system",
        "✓ Monitoring alerts"
    ],
    
    "security": [
        "✓ JWT authentication",
        "✓ Rate limiting",
        "✓ Input validation",
        "✓ CORS configuration",
        "✓ HTTPS only",
        "✓ Content moderation"
    ],
    
    "scalability": [
        "✓ Horizontal scaling",
        "✓ Load balancing",
        "✓ Auto-scaling policies",
        "✓ Database replication",
        "✓ CDN distribution"
    ]
}
\`\`\`

## Key Takeaways

- **Architecture**: API + Workers + Queue + Storage + DB
- **Async processing**: Essential for user experience
- **Queue system**: Celery/Redis for job management
- **WebSockets**: Real-time status updates
- **Multiple models**: Support SD, SDXL, specialized models
- **Storage**: S3 + CDN for images
- **Scaling**: Horizontal API, GPU workers as needed
- **Monitoring**: Track costs, performance, usage
- **Security**: Auth, rate limiting, validation
- **Production-ready**: ~$1,500/month for moderate traffic
`,
};
