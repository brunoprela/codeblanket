export const buildingMultiModalProducts = {
  title: 'Building Multi-Modal Products',
  id: 'building-multi-modal-products',
  description:
    'Master building complete end-to-end multi-modal AI products from architecture design through production deployment.',
  content: `
# Building Multi-Modal Products

## Introduction

Building production multi-modal AI products requires combining all the techniques we've learned: vision, audio, text processing, cross-modal generation, document intelligence, and more. This section ties everything together with patterns, architectures, and best practices for shipping complete products.

## Product Architecture

### Layered Architecture

\`\`\`
┌─────────────────────────────────────────────┐
│          User Interface Layer               │
│  (Web, Mobile, API)                        │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│        Application Logic Layer              │
│  (Business rules, workflows)               │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│      Multi-Modal Processing Layer           │
│  ├─ Vision Processing                      │
│  ├─ Audio Processing                       │
│  ├─ Text Processing                        │
│  └─ Cross-Modal Operations                 │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│          Data & Storage Layer               │
│  (Databases, Vector stores, File storage)  │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│          Infrastructure Layer               │
│  (Compute, Networking, Monitoring)         │
└─────────────────────────────────────────────┘
\`\`\`

### Microservices Architecture

\`\`\`
[API Gateway] → [Load Balancer]
                      ↓
        ┌─────────────┴─────────────┐
        ↓             ↓              ↓
  [Vision Service] [Audio Service] [Text Service]
        ↓             ↓              ↓
    [GPU Pool]    [GPU Pool]    [CPU Pool]
        ↓             ↓              ↓
              [Message Queue]
                      ↓
         [Orchestration Service]
                      ↓
         [Results Database] + [Cache]
\`\`\`

## Complete Product Example

### Multi-Modal Content Intelligence Platform

\`\`\`python
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

logging.basicConfig (level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Supported content types."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    TEXT = "text"

@dataclass
class ProcessingJob:
    """Represents a content processing job."""
    job_id: str
    content_type: ContentType
    input_path: str
    operations: List[str]  # ["caption", "transcribe", "extract_text", etc.]
    status: str  # pending, processing, completed, failed
    result: Optional[Dict[str, Any]] = None
    created_at: float = 0
    completed_at: Optional[float] = None

class MultiModalContentPlatform:
    """
    Complete multi-modal content intelligence platform.
    
    Features:
    - Process images, videos, audio, documents
    - Extract text, generate captions, transcribe audio
    - Semantic search across all content
    - Generate insights and summaries
    - API for integration
    """
    
    def __init__(
        self,
        openai_api_key: str,
        redis_host: str = "localhost",
        postgres_connection: Optional[str] = None
    ):
        self.openai_key = openai_api_key
        
        # Initialize processing modules (from previous sections)
        self.vision_processor = None  # VisionAgent, etc.
        self.audio_processor = None   # Audio transcription
        self.document_processor = None  # Document intelligence
        
        # Storage
        self.redis_client = redis.Redis (host=redis_host)
        self.db_connection = postgres_connection
        
        # Job queue
        self.job_queue = []
        
    def submit_job(
        self,
        content_path: str,
        content_type: ContentType,
        operations: List[str]
    ) -> str:
        """
        Submit a content processing job.
        
        Args:
            content_path: Path to content file
            content_type: Type of content
            operations: List of operations to perform
        
        Returns:
            Job ID
        """
        import uuid
        
        job_id = str (uuid.uuid4())
        
        job = ProcessingJob(
            job_id=job_id,
            content_type=content_type,
            input_path=content_path,
            operations=operations,
            status="pending",
            created_at=time.time()
        )
        
        self.job_queue.append (job)
        
        logger.info (f"Job {job_id} submitted: {content_type.value} with {len (operations)} operations")
        
        # Process asynchronously (in real implementation)
        self._process_job (job)
        
        return job_id
    
    def _process_job (self, job: ProcessingJob):
        """Process a job."""
        logger.info (f"Processing job {job.job_id}")
        
        job.status = "processing"
        results = {}
        
        try:
            for operation in job.operations:
                if job.content_type == ContentType.IMAGE:
                    if operation == "caption":
                        results["caption"] = self._caption_image (job.input_path)
                    elif operation == "ocr":
                        results["text"] = self._extract_text_from_image (job.input_path)
                    elif operation == "analyze":
                        results["analysis"] = self._analyze_image (job.input_path)
                
                elif job.content_type == ContentType.VIDEO:
                    if operation == "transcribe":
                        results["transcript"] = self._transcribe_video (job.input_path)
                    elif operation == "summarize":
                        results["summary"] = self._summarize_video (job.input_path)
                    elif operation == "scenes":
                        results["scenes"] = self._detect_scenes (job.input_path)
                
                elif job.content_type == ContentType.AUDIO:
                    if operation == "transcribe":
                        results["transcript"] = self._transcribe_audio (job.input_path)
                    elif operation == "classify":
                        results["classification"] = self._classify_audio (job.input_path)
                
                elif job.content_type == ContentType.DOCUMENT:
                    if operation == "extract":
                        results["data"] = self._extract_document_data (job.input_path)
                    elif operation == "summarize":
                        results["summary"] = self._summarize_document (job.input_path)
            
            job.status = "completed"
            job.result = results
            job.completed_at = time.time()
            
            logger.info (f"Job {job.job_id} completed successfully")
        
        except Exception as e:
            logger.error (f"Job {job.job_id} failed: {e}")
            job.status = "failed"
            job.result = {"error": str (e)}
    
    def get_job_status (self, job_id: str) -> Optional[ProcessingJob]:
        """Get status of a job."""
        for job in self.job_queue:
            if job.job_id == job_id:
                return job
        return None
    
    def search_content(
        self,
        query: str,
        content_types: Optional[List[ContentType]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search across all processed content.
        
        Uses semantic search across text, images, videos, etc.
        """
        # Implement multi-modal RAG search
        # (Using techniques from multi-modal RAG section)
        pass
    
    def generate_insights(
        self,
        content_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Generate insights across multiple pieces of content.
        
        Examples:
        - Common themes
        - Sentiment analysis
        - Key entities
        - Trends
        """
        pass
    
    # Helper methods for processing operations
    def _caption_image (self, image_path: str) -> str:
        """Generate image caption."""
        # Use techniques from image-text understanding section
        pass
    
    def _extract_text_from_image (self, image_path: str) -> str:
        """Extract text from image."""
        pass
    
    def _analyze_image (self, image_path: str) -> Dict[str, Any]:
        """Comprehensive image analysis."""
        pass
    
    def _transcribe_video (self, video_path: str) -> str:
        """Transcribe video audio."""
        # Use techniques from video-text understanding section
        pass
    
    def _summarize_video (self, video_path: str) -> str:
        """Summarize video content."""
        pass
    
    def _detect_scenes (self, video_path: str) -> List[Dict[str, Any]]:
        """Detect scenes in video."""
        pass
    
    def _transcribe_audio (self, audio_path: str) -> str:
        """Transcribe audio."""
        # Use techniques from audio-text processing section
        pass
    
    def _classify_audio (self, audio_path: str) -> Dict[str, Any]:
        """Classify audio content."""
        pass
    
    def _extract_document_data (self, doc_path: str) -> Dict[str, Any]:
        """Extract structured data from document."""
        # Use techniques from document intelligence section
        pass
    
    def _summarize_document (self, doc_path: str) -> str:
        """Summarize document."""
        pass

# Usage
platform = MultiModalContentPlatform(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Submit jobs
image_job = platform.submit_job(
    "product_image.jpg",
    ContentType.IMAGE,
    operations=["caption", "ocr", "analyze"]
)

video_job = platform.submit_job(
    "presentation.mp4",
    ContentType.VIDEO,
    operations=["transcribe", "summarize"]
)

# Check status
job_status = platform.get_job_status (image_job)
print(f"Job status: {job_status.status}")

if job_status.status == "completed":
    print(f"Results: {job_status.result}")
\`\`\`

## Production Considerations

### 1. Scalability

\`\`\`python
class ScalableProcessingQueue:
    """Scalable job queue with worker pool."""
    
    def __init__(self, max_workers: int = 10):
        self.queue = []
        self.workers = []
        self.max_workers = max_workers
    
    def add_job (self, job):
        """Add job to queue."""
        self.queue.append (job)
        self._ensure_workers()
    
    def _ensure_workers (self):
        """Ensure we have enough workers."""
        active_workers = len([w for w in self.workers if w.is_alive()])
        
        if active_workers < self.max_workers and self.queue:
            # Start new worker
            import threading
            worker = threading.Thread (target=self._process_queue)
            worker.start()
            self.workers.append (worker)
    
    def _process_queue (self):
        """Process jobs from queue."""
        while self.queue:
            job = self.queue.pop(0)
            # Process job
            self._process_job (job)
\`\`\`

### 2. Cost Optimization

\`\`\`python
class CostTracker:
    """Track and optimize costs."""
    
    def __init__(self):
        self.costs = []
    
    def track_operation(
        self,
        operation: str,
        input_size: int,
        model: str
    ) -> float:
        """Track cost of operation."""
        # Calculate cost based on pricing
        cost = self._calculate_cost (operation, input_size, model)
        
        self.costs.append({
            "operation": operation,
            "cost": cost,
            "timestamp": time.time()
        })
        
        return cost
    
    def _calculate_cost (self, operation: str, input_size: int, model: str) -> float:
        """Calculate operation cost."""
        # Pricing table
        pricing = {
            "gpt-4-vision": {"input": 0.01, "output": 0.03},
            "whisper": 0.006,
            "dall-e-3": 0.040,
            "tts-1": 0.015
        }
        
        # Calculate based on operation and model
        return 0.0  # Simplified
    
    def get_total_cost (self) -> float:
        """Get total cost."""
        return sum (c["cost"] for c in self.costs)
    
    def get_cost_breakdown (self) -> Dict[str, float]:
        """Get cost breakdown by operation."""
        breakdown = {}
        for cost_entry in self.costs:
            op = cost_entry["operation"]
            breakdown[op] = breakdown.get (op, 0) + cost_entry["cost"]
        return breakdown
\`\`\`

### 3. Monitoring & Observability

\`\`\`python
class PlatformMonitor:
    """Monitor platform health and performance."""
    
    def __init__(self):
        self.metrics = []
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a metric."""
        self.metrics.append({
            "name": metric_name,
            "value": value,
            "tags": tags or {},
            "timestamp": time.time()
        })
    
    def get_dashboard_data (self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        # Aggregate metrics
        return {
            "total_jobs": len([m for m in self.metrics if m["name"] == "job_completed"]),
            "average_latency": self._calculate_average_latency(),
            "error_rate": self._calculate_error_rate(),
            "cost_per_hour": self._calculate_cost_per_hour()
        }
    
    def _calculate_average_latency (self) -> float:
        """Calculate average processing latency."""
        latencies = [
            m["value"]
            for m in self.metrics
            if m["name"] == "processing_latency"
        ]
        return sum (latencies) / len (latencies) if latencies else 0.0
    
    def _calculate_error_rate (self) -> float:
        """Calculate error rate."""
        total = len([m for m in self.metrics if m["name"] in ["job_completed", "job_failed"]])
        errors = len([m for m in self.metrics if m["name"] == "job_failed"])
        return errors / total if total > 0 else 0.0
    
    def _calculate_cost_per_hour (self) -> float:
        """Calculate cost per hour."""
        # Get costs from last hour
        one_hour_ago = time.time() - 3600
        recent_costs = [
            m["value"]
            for m in self.metrics
            if m["name"] == "cost" and m["timestamp"] > one_hour_ago
        ]
        return sum (recent_costs)
\`\`\`

### 4. Error Handling & Recovery

\`\`\`python
class RobustProcessor:
    """Processor with comprehensive error handling."""
    
    def __init__(self):
        self.retry_config = {
            "max_retries": 3,
            "backoff_factor": 2
        }
    
    def process_with_retry(
        self,
        job: ProcessingJob,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Process job with automatic retry."""
        for attempt in range (max_retries):
            try:
                return self._process (job)
            
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * self.retry_config["backoff_factor"]
                    logger.warning (f"Rate limited, waiting {wait_time}s")
                    time.sleep (wait_time)
                    continue
                raise
            
            except InvalidInputError as e:
                # Don't retry invalid input
                logger.error (f"Invalid input: {e}")
                raise
            
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning (f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(1)
                    continue
                raise
        
        raise Exception("Max retries exceeded")
    
    def _process (self, job: ProcessingJob) -> Dict[str, Any]:
        """Process job."""
        # Implementation
        pass
\`\`\`

## Best Practices

### 1. API Design

\`\`\`python
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Multi-Modal Content Platform API")

class JobSubmission(BaseModel):
    content_type: str
    operations: List[str]

class JobStatus(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]]

@app.post("/jobs/submit")
async def submit_job(
    file: UploadFile = File(...),
    content_type: str = "image",
    operations: List[str] = ["caption"]
) -> Dict[str, str]:
    """Submit a processing job."""
    # Save uploaded file
    file_path = f"uploads/{file.filename}"
    with open (file_path, "wb") as f:
        f.write (await file.read())
    
    # Submit to platform
    job_id = platform.submit_job(
        file_path,
        ContentType (content_type),
        operations
    )
    
    return {"job_id": job_id, "status": "submitted"}

@app.get("/jobs/{job_id}")
async def get_job_status (job_id: str) -> JobStatus:
    """Get job status."""
    job = platform.get_job_status (job_id)
    
    if not job:
        raise HTTPException (status_code=404, detail="Job not found")
    
    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        result=job.result
    )

@app.post("/search")
async def search_content(
    query: str,
    content_types: Optional[List[str]] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search across content."""
    types = [ContentType (t) for t in content_types] if content_types else None
    
    results = platform.search_content (query, types, limit)
    
    return results
\`\`\`

### 2. Testing

\`\`\`python
import pytest

def test_image_processing():
    """Test image processing pipeline."""
    platform = MultiModalContentPlatform (openai_api_key="test")
    
    job_id = platform.submit_job(
        "test_image.jpg",
        ContentType.IMAGE,
        ["caption"]
    )
    
    assert job_id is not None
    
    # Wait for completion (in real test, would use async)
    time.sleep(5)
    
    job = platform.get_job_status (job_id)
    
    assert job.status == "completed"
    assert "caption" in job.result

def test_cost_tracking():
    """Test cost tracking."""
    tracker = CostTracker()
    
    cost = tracker.track_operation("image_caption", 1024, "gpt-4-vision")
    
    assert cost > 0
    assert tracker.get_total_cost() == cost
\`\`\`

### 3. Documentation

\`\`\`markdown
# Multi-Modal Content Platform

## Overview
Platform for processing images, videos, audio, and documents using AI.

## API Endpoints

### POST /jobs/submit
Submit a content processing job.

**Parameters:**
- file: Content file to process
- content_type: Type of content (image, video, audio, document)
- operations: List of operations to perform

**Response:**
\`\`\`json
{
  "job_id": "uuid",
  "status": "submitted"
}
\`\`\`

### GET /jobs/{job_id}
Get status of a job.

**Response:**
\`\`\`json
{
  "job_id": "uuid",
  "status": "completed|processing|failed",
  "result": {...}
}
\`\`\`

## Supported Operations

### Images
- caption: Generate image caption
- ocr: Extract text
- analyze: Comprehensive analysis

### Videos
- transcribe: Transcribe audio
- summarize: Generate summary
- scenes: Detect scene changes

### Audio
- transcribe: Speech-to-text
- classify: Classify content type

### Documents
- extract: Extract structured data
- summarize: Generate summary
\`\`\`

## Summary

Building multi-modal products requires:

**Architecture:**
- Layered architecture for separation of concerns
- Microservices for scalability
- Message queues for async processing
- Caching for performance

**Production Features:**
- Scalable job queue
- Cost tracking and optimization
- Comprehensive monitoring
- Robust error handling
- Well-designed API
- Complete testing

**Best Practices:**
- Start simple, add complexity gradually
- Monitor everything
- Optimize costs continuously
- Test thoroughly
- Document comprehensively
- Plan for scale from day one

**Key Considerations:**
- Multi-modal requires significant compute
- Costs can scale quickly
- Quality varies by model and task
- User experience matters
- Accessibility is important
- Security and privacy are critical

This completes our journey through multi-modal AI systems. You now have the knowledge to build sophisticated multi-modal products!
`,
  codeExamples: [
    {
      title: 'Multi-Modal Content Platform',
      description:
        'Complete platform architecture for processing multi-modal content at scale',
      language: 'python',
      code: `# See MultiModalContentPlatform class in content above`,
    },
  ],
  practicalTips: [
    'Start with MVP using single modality, then expand to multi-modal gradually',
    'Implement comprehensive cost tracking from day one - costs scale quickly',
    "Use job queues for async processing - users don't want to wait",
    'Cache aggressively at every layer - processing is expensive',
    'Monitor everything: latency, costs, errors, quality metrics',
    "Design API with versioning in mind - you'll iterate frequently",
    'Test with real-world data, not just perfect examples',
    'Plan for 10x scale even if starting small',
  ],
  quiz: '/quizzes/multi-modal-ai-systems/building-multi-modal-products',
};
