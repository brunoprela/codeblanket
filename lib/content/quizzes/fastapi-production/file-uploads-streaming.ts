export const fileUploadsStreamingQuiz = [
  {
    id: 1,
    question:
      "Design a production file upload system for a video platform that handles uploads up to 5GB. The system must: (1) validate file type and size, (2) scan for viruses, (3) transcode to multiple formats (1080p, 720p, 480p), (4) generate thumbnails, and (5) store in S3. How would you implement chunked uploads with resume capability? Design the complete flow including progress tracking, error handling, and background processing. What happens if a user's connection drops mid-upload?",
    answer: `**Production Video Upload System**:

**Architecture**: Client → API (chunks) → S3 (presigned) → Background Worker (transcode) → CDN

**Implementation**:

\`\`\`python
# 1. Initiate Upload
@app.post("/uploads/initiate")
async def initiate_upload(
    filename: str,
    file_size: int,
    content_type: str,
    current_user: User = Depends (get_current_user),
    db: Session = Depends (get_db)
):
    """Initiate chunked upload"""
    # Validate
    if file_size > 5 * 1024**3:  # 5GB
        raise HTTPException(400, "File too large")
    
    if content_type not in ["video/mp4", "video/quicktime", "video/x-msvideo"]:
        raise HTTPException(415, "Invalid video format")
    
    # Create upload record
    upload_id = str (uuid.uuid4())
    chunk_size = 5 * 1024**2  # 5MB chunks
    total_chunks = math.ceil (file_size / chunk_size)
    
    upload = Upload(
        id=upload_id,
        user_id=current_user.id,
        filename=filename,
        file_size=file_size,
        content_type=content_type,
        total_chunks=total_chunks,
        uploaded_chunks=[],  # Track completed chunks
        status="initiated"
    )
    db.add (upload)
    db.commit()
    
    return {
        "upload_id": upload_id,
        "chunk_size": chunk_size,
        "total_chunks": total_chunks
    }

# 2. Upload Chunks with Resume
@app.post("/uploads/{upload_id}/chunk/{chunk_number}")
async def upload_chunk(
    upload_id: str,
    chunk_number: int,
    chunk: UploadFile = File(...),
    db: Session = Depends (get_db)
):
    """Upload single chunk, supports resume"""
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        raise HTTPException(404)
    
    # Check if chunk already uploaded (idempotent)
    if chunk_number in upload.uploaded_chunks:
        return {"status": "already_uploaded", "chunk": chunk_number}
    
    # Save chunk to S3
    s3_key = f"uploads/{upload_id}/chunk_{chunk_number}"
    
    # Stream chunk to S3
    s3_client.upload_fileobj(
        chunk.file,
        S3_BUCKET,
        s3_key
    )
    
    # Mark chunk as uploaded
    upload.uploaded_chunks.append (chunk_number)
    db.commit()
    
    # Check if all chunks uploaded
    if len (upload.uploaded_chunks) == upload.total_chunks:
        # Trigger assembly
        assemble_video_task.delay (upload_id)
    
    return {
        "chunk": chunk_number,
        "total": upload.total_chunks,
        "uploaded": len (upload.uploaded_chunks),
        "percent": (len (upload.uploaded_chunks) / upload.total_chunks) * 100
    }

# 3. Resume Upload (get missing chunks)
@app.get("/uploads/{upload_id}/status")
async def get_upload_status (upload_id: str, db: Session = Depends (get_db)):
    """Get upload status for resume"""
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        raise HTTPException(404)
    
    # Calculate missing chunks
    all_chunks = set (range (upload.total_chunks))
    uploaded = set (upload.uploaded_chunks)
    missing = sorted (all_chunks - uploaded)
    
    return {
        "upload_id": upload_id,
        "status": upload.status,
        "uploaded_chunks": len (uploaded),
        "total_chunks": upload.total_chunks,
        "missing_chunks": missing
    }

# 4. Assemble & Process (Background Task)
@celery_app.task
def assemble_video_task (upload_id: str):
    """Assemble chunks and process video"""
    upload = db.query(Upload).get (upload_id)
    
    # 1. Assemble chunks
    output_key = f"videos/{upload_id}/original.mp4"
    
    # Multipart upload to S3
    mpu = s3_client.create_multipart_upload(
        Bucket=S3_BUCKET,
        Key=output_key
    )
    
    parts = []
    for i in range (upload.total_chunks):
        chunk_key = f"uploads/{upload_id}/chunk_{i}"
        
        # Copy chunk to multipart upload
        part = s3_client.upload_part_copy(
            Bucket=S3_BUCKET,
            Key=output_key,
            CopySource={'Bucket': S3_BUCKET, 'Key': chunk_key},
            PartNumber=i+1,
            UploadId=mpu['UploadId']
        )
        
        parts.append({
            'PartNumber': i+1,
            'ETag': part['CopyPartResult']['ETag']
        })
    
    # Complete multipart upload
    s3_client.complete_multipart_upload(
        Bucket=S3_BUCKET,
        Key=output_key,
        UploadId=mpu['UploadId'],
        MultipartUpload={'Parts': parts}
    )
    
    # 2. Virus scan
    scan_video (output_key)
    
    # 3. Trigger transcoding
    transcode_video.delay (upload_id, output_key)
    
    # 4. Delete chunks
    for i in range (upload.total_chunks):
        s3_client.delete_object(Bucket=S3_BUCKET, Key=f"uploads/{upload_id}/chunk_{i}")
    
    upload.status = "assembled"
    db.commit()

@celery_app.task
def transcode_video (upload_id: str, source_key: str):
    """Transcode to multiple resolutions"""
    # Use AWS MediaConvert or FFmpeg
    
    resolutions = [
        {"name": "1080p", "width": 1920, "height": 1080, "bitrate": "5M"},
        {"name": "720p", "width": 1280, "height": 720, "bitrate": "2.5M"},
        {"name": "480p", "width": 854, "height": 480, "bitrate": "1M"}
    ]
    
    for res in resolutions:
        output_key = f"videos/{upload_id}/{res['name']}.mp4"
        
        # Transcode using FFmpeg
        ffmpeg_command = f"""
        ffmpeg -i s3://{S3_BUCKET}/{source_key} 
        -vf scale={res['width']}:{res['height']} 
        -b:v {res['bitrate']} 
        -c:v libx264 
        -c:a aac 
        s3://{S3_BUCKET}/{output_key}
        """
        
        subprocess.run (ffmpeg_command, shell=True, check=True)
    
    # Generate thumbnail
    generate_thumbnail.delay (upload_id, source_key)
    
    # Mark complete
    upload = db.query(Upload).get (upload_id)
    upload.status = "ready"
    db.commit()
\`\`\`

**Connection Drop Handling**: Client detects disconnect, calls /status endpoint to get missing chunks, resumes from where it left off. All chunks are idempotent (re-uploading same chunk is safe).`,
  },
  {
    id: 2,
    question:
      'Compare three strategies for file uploads: (1) uploading through API server to S3, (2) presigned URLs for direct S3 upload, and (3) multipart upload directly to S3. For each approach, analyze: performance, security, complexity, cost, and use cases. When would you use each? Implement a presigned URL approach with proper validation and callback confirmation.',
    answer: `**Upload Strategy Comparison**:

**1. Upload through API Server**:
\`\`\`
Client → API Server → S3
\`\`\`

Pros:
- Full control: validation, virus scanning before S3
- Simpler for small files (< 10MB)
- Can transform files (resize images) before storage

Cons:
- API server bottleneck (CPU, bandwidth)
- Higher costs (EC2 bandwidth charges)
- Slower for large files
- Doesn't scale well

Use case: Small files with validation/processing

\`\`\`python
@app.post("/upload")
async def upload_file (file: UploadFile = File(...)):
    # Validate
    await validate_file (file)
    
    # Upload to S3
    s3_client.upload_fileobj (file.file, S3_BUCKET, file.filename)
\`\`\`

**2. Presigned URLs (Direct Upload)**:
\`\`\`
Client ← presigned URL ← API Server
Client → S3 (direct)
Client → API Server (confirm)
\`\`\`

Pros:
- Fast: direct to S3, no API bottleneck
- Lower costs: no API bandwidth
- Scales infinitely (S3 handles load)
- Simple implementation

Cons:
- Limited validation (size, type only)
- No virus scanning before upload
- Must scan after upload
- 2-step process (get URL, then upload)

Use case: **BEST for large files (> 10MB)**, production default

\`\`\`python
# Step 1: Generate presigned URL
@app.post("/upload/presigned-url")
async def get_presigned_url(
    filename: str,
    filesize: int,
    content_type: str,
    current_user: User = Depends (get_current_user)
):
    # Validate request
    if filesize > 5 * 1024**3:
        raise HTTPException(400, "File too large")
    
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Invalid file type")
    
    # Generate S3 key
    file_key = f"uploads/{current_user.id}/{uuid.uuid4()}/{filename}"
    
    # Presigned POST (more flexible than PUT)
    presigned_post = s3_client.generate_presigned_post(
        Bucket=S3_BUCKET,
        Key=file_key,
        Fields={
            "Content-Type": content_type,
            "x-amz-meta-user-id": str (current_user.id)
        },
        Conditions=[
            ["content-length-range", 1, filesize * 1.1],  # Allow 10% tolerance
            {"Content-Type": content_type},
            ["starts-with", "$x-amz-meta-user-id", ""]
        ],
        ExpiresIn=900  # 15 minutes
    )
    
    # Store pending upload
    pending_upload = PendingUpload(
        file_key=file_key,
        user_id=current_user.id,
        filename=filename,
        filesize=filesize,
        expires_at=datetime.utcnow() + timedelta (minutes=15)
    )
    db.add (pending_upload)
    db.commit()
    
    return {
        "upload_url": presigned_post["url"],
        "fields": presigned_post["fields"],
        "file_key": file_key
    }

# Step 2: Confirm upload
@app.post("/upload/confirm")
async def confirm_upload(
    file_key: str,
    current_user: User = Depends (get_current_user),
    db: Session = Depends (get_db)
):
    # Verify file in S3
    try:
        response = s3_client.head_object(Bucket=S3_BUCKET, Key=file_key)
        
        # Verify user_id metadata matches
        if response["Metadata"].get("user-id") != str (current_user.id):
            raise HTTPException(403, "File user mismatch")
        
        filesize = response["ContentLength"]
        
    except ClientError:
        raise HTTPException(404, "File not found in S3")
    
    # Create upload record
    upload = Upload(
        user_id=current_user.id,
        file_key=file_key,
        filename=Path (file_key).name,
        filesize=filesize,
        status="uploaded"
    )
    db.add (upload)
    db.commit()
    
    # Trigger virus scan
    scan_file_task.delay (file_key)
    
    return {"upload_id": upload.id, "status": "confirmed"}
\`\`\`

**3. S3 Multipart Upload**:
\`\`\`
Client ← initiate MPU ← API
Client → S3 (part 1, 2, 3... direct)
Client → API (complete MPU)
\`\`\`

Pros:
- Best for very large files (> 100MB, up to 5TB)
- Parallel part uploads (faster)
- Resume capability (re-upload failed parts)
- Network resilience

Cons:
- Most complex implementation
- Must manage part tracking
- S3 charges for incomplete uploads if not cleaned up

Use case: Very large files (> 1GB), need resume capability

**Comparison Table**:

| Aspect | Via API | Presigned URL | Multipart |
|--------|---------|---------------|-----------|
| **Speed** | Slow | Fast | Fastest |
| **File size** | < 10MB | < 1GB | > 1GB |
| **Validation** | Full | Limited | Limited |
| **Complexity** | Simple | Medium | Complex |
| **Cost** | High | Low | Low |
| **Resume** | No | No | Yes |
| **Virus scan** | Before | After | After |

**Production Recommendation**: 
- Small files (< 10MB): Via API
- Large files (10MB - 1GB): Presigned URL
- Very large (> 1GB): Multipart
`,
  },
  {
    id: 3,
    question:
      'Design a streaming response system for generating and downloading large CSV exports (millions of rows) from a database. The system must: (1) stream data without loading everything into memory, (2) support filtering and pagination at the database level, (3) compress the output stream, and (4) provide progress updates. Implement both the streaming endpoint and client-side progress tracking. How would you handle cancellation if the user stops the download?',
    answer: `**Streaming CSV Export System**:

\`\`\`python
"""
Memory-efficient CSV streaming with progress
"""

from fastapi.responses import StreamingResponse
import gzip
import io
from sqlalchemy import select

@app.get("/export/users.csv")
async def export_users_csv(
    filters: UserFilters = Depends(),
    compress: bool = True,
    current_user: User = Depends (get_current_user),
    db: Session = Depends (get_db)
):
    """
    Stream CSV export with compression and progress
    """
    # Build query with filters
    query = select(User).where(User.tenant_id == current_user.tenant_id)
    
    if filters.created_after:
        query = query.where(User.created_at >= filters.created_after)
    
    if filters.status:
        query = query.where(User.status == filters.status)
    
    # Count total (for progress)
    total_count = db.execute (select (func.count()).select_from (query.subquery())).scalar()
    
    # Generate export ID for progress tracking
    export_id = str (uuid.uuid4())
    export_progress[export_id] = {
        "total": total_count,
        "processed": 0,
        "status": "started"
    }
    
    async def generate_csv_stream():
        """
        Generator that yields CSV data in chunks
        Memory-efficient: processes batches, not entire dataset
        """
        # CSV header
        yield "id,email,username,status,created_at\\n"
        
        # Stream data in batches
        batch_size = 1000
        offset = 0
        processed = 0
        
        while True:
            # Fetch batch
            users = db.execute(
                query.offset (offset).limit (batch_size)
            ).scalars().all()
            
            if not users:
                break
            
            # Generate CSV rows for batch
            for user in users:
                # Escape CSV fields
                email = user.email.replace('"', '""')
                username = user.username.replace('"', '""')
                
                row = f'{user.id},"{email}","{username}",{user.status},{user.created_at}\\n'
                yield row
                
                processed += 1
            
            # Update progress
            export_progress[export_id] = {
                "total": total_count,
                "processed": processed,
                "status": "in_progress",
                "percent": (processed / total_count * 100) if total_count > 0 else 0
            }
            
            offset += batch_size
            
            # Check for cancellation
            if export_id in cancelled_exports:
                export_progress[export_id]["status"] = "cancelled"
                break
        
        # Mark complete
        if export_id not in cancelled_exports:
            export_progress[export_id]["status"] = "completed"
    
    # Wrap in compression if requested
    if compress:
        async def compressed_stream():
            """
            Compress CSV stream with gzip
            """
            buffer = io.BytesIO()
            
            with gzip.GzipFile (fileobj=buffer, mode='wb') as gz_file:
                async for chunk in generate_csv_stream():
                    gz_file.write (chunk.encode('utf-8'))
                    
                    # Yield compressed data when buffer reaches threshold
                    if buffer.tell() > 1024 * 1024:  # 1MB
                        buffer.seek(0)
                        yield buffer.read()
                        buffer.seek(0)
                        buffer.truncate()
            
            # Yield remaining data
            buffer.seek(0)
            yield buffer.read()
        
        return StreamingResponse(
            compressed_stream(),
            media_type="application/gzip",
            headers={
                "Content-Disposition": f"attachment; filename=users_{export_id}.csv.gz",
                "X-Export-ID": export_id  # For progress tracking
            }
        )
    
    else:
        return StreamingResponse(
            generate_csv_stream(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=users_{export_id}.csv",
                "X-Export-ID": export_id
            }
        )

# Progress tracking endpoint
@app.get("/export/progress/{export_id}")
async def get_export_progress (export_id: str):
    """Get export progress"""
    if export_id not in export_progress:
        raise HTTPException(404, "Export not found")
    
    return export_progress[export_id]

# Cancellation endpoint
@app.post("/export/cancel/{export_id}")
async def cancel_export (export_id: str):
    """Cancel ongoing export"""
    if export_id not in export_progress:
        raise HTTPException(404)
    
    cancelled_exports.add (export_id)
    
    return {"status": "cancelled"}

# Cleanup old export progress (background task)
@celery_app.task
def cleanup_old_exports():
    """Remove export progress older than 1 hour"""
    cutoff = datetime.utcnow() - timedelta (hours=1)
    
    to_remove = []
    for export_id, progress in export_progress.items():
        if progress.get("started_at", datetime.utcnow()) < cutoff:
            to_remove.append (export_id)
    
    for export_id in to_remove:
        del export_progress[export_id]
        cancelled_exports.discard (export_id)
\`\`\`

**Client-Side Implementation**:

\`\`\`javascript
// Client-side streaming download with progress

async function downloadCsvWithProgress() {
    // Start download
    const response = await fetch('/export/users.csv?compress=true');
    const exportId = response.headers.get('X-Export-ID');
    
    // Track progress in separate request
    const progressInterval = setInterval (async () => {
        const progressResponse = await fetch(\`/export/progress/\${exportId}\`);
        const progress = await progressResponse.json();
        
        updateProgressBar (progress.percent);
        
        if (progress.status === 'completed' || progress.status === 'cancelled') {
            clearInterval (progressInterval);
        }
    }, 1000);  // Poll every second
    
    // Stream response to file
    const reader = response.body.getReader();
    const chunks = [];
    
    while (true) {
        const {done, value} = await reader.read();
        
        if (done) break;
        
        chunks.push (value);
    }
    
    // Create blob and download
    const blob = new Blob (chunks, {type: 'application/gzip'});
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'users.csv.gz';
    a.click();
    
    clearInterval (progressInterval);
}

// Cancel download
async function cancelDownload (exportId) {
    await fetch(\`/export/cancel/\${exportId}\`, {method: 'POST'});
}
\`\`\`

**Key Techniques**:
1. **Batch processing**: Fetch 1000 rows at a time, never load entire dataset
2. **Generator functions**: Yield data incrementally
3. **Compression**: gzip stream reduces bandwidth 70-90%
4. **Progress tracking**: Separate endpoint with export_id
5. **Cancellation**: Set flag, check in generator loop
6. **Cleanup**: Background task removes old progress data`,
  },
].map(({ id, ...q }, idx) => ({
  id: `fastapi-file-uploads-streaming-q-${idx + 1}`,
  question: q.question,
  sampleAnswer: String(q.answer),
  keyPoints: [],
}));
