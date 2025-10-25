export const fileUploadsStreaming = {
  title: 'File Uploads & Streaming Responses',
  id: 'file-uploads-streaming',
  content: `
# File Uploads & Streaming Responses

## Introduction

Modern APIs handle files: user avatars, document uploads, video processing, data exports. Naive implementations fail at scale: loading 1GB files into memory crashes servers, blocking uploads frustrate users. Production APIs need streaming, validation, progress tracking, and cloud storage integration.

**Why file handling matters:**
- **Memory efficiency**: Stream files without loading entire content
- **User experience**: Progress tracking for large uploads
- **Security**: Validate file types, scan for malware
- **Scalability**: Direct uploads to cloud storage (S3)
- **Performance**: Async processing, chunked transfers

**Common scenarios:**
- User profile pictures (< 10MB)
- Document uploads (PDFs, Word docs)
- Video/audio files (100MB - 10GB)
- CSV imports (millions of rows)
- Generated reports (streaming downloads)
- Image processing pipelines

In this section, you'll master:
- File upload handling with validation
- Streaming large file uploads
- Direct S3 uploads with presigned URLs
- Streaming responses for downloads
- Progress tracking
- Production patterns

---

## Basic File Uploads

### Simple File Upload

\`\`\`python
"""
Basic file upload endpoint
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import shutil
from pathlib import Path

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a single file
    
    UploadFile advantages:
    - Spooled to disk after threshold (1MB default)
    - Async file operations
    - Automatic cleanup
    """
    # Save file
    file_path = UPLOAD_DIR / file.filename
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file_path.stat().st_size
    }

# Multiple files
@app.post("/upload/multiple/")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """
    Upload multiple files at once
    """
    uploaded = []
    
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        uploaded.append({
            "filename": file.filename,
            "size": file_path.stat().st_size
        })
    
    return {"files": uploaded, "count": len(uploaded)}
\`\`\`

---

## File Validation

### Validation Layer

\`\`\`python
"""
File upload validation: size, type, content
"""

from fastapi import HTTPException, status
import magic  # python-magic for file type detection
from PIL import Image
import io

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_DOCUMENT_TYPES = {"application/pdf", "application/msword", 
                         "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

async def validate_file_size(file: UploadFile, max_size: int = MAX_FILE_SIZE):
    """
    Validate file size without loading entire file
    """
    # Read in chunks to avoid memory issues
    size = 0
    
    while chunk := await file.read(8192):  # 8KB chunks
        size += len(chunk)
        
        if size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max size: {max_size / (1024*1024)}MB"
            )
    
    # Reset file pointer
    await file.seek(0)
    
    return size

async def validate_file_type(
    file: UploadFile,
    allowed_types: set
) -> str:
    """
    Validate file type using magic number (not just extension)
    """
    # Read first 2048 bytes for magic number detection
    header = await file.read(2048)
    await file.seek(0)
    
    # Detect actual file type
    mime_type = magic.from_buffer(header, mime=True)
    
    if mime_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type {mime_type} not allowed. Allowed: {allowed_types}"
        )
    
    return mime_type

async def validate_image(file: UploadFile) -> dict:
    """
    Validate image file: dimensions, format
    """
    # Read file into memory (small images only)
    contents = await file.read()
    await file.seek(0)
    
    try:
        image = Image.open(io.BytesIO(contents))
        
        # Check dimensions
        width, height = image.size
        if width > 10000 or height > 10000:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large: {width}x{height}. Max: 10000x10000"
            )
        
        # Check format
        if image.format.lower() not in ['jpeg', 'png', 'gif', 'webp']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image format: {image.format}"
            )
        
        return {
            "width": width,
            "height": height,
            "format": image.format,
            "mode": image.mode
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

# Validated upload endpoint
@app.post("/upload/image/")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload image with validation
    """
    # Validate size
    file_size = await validate_file_size(file, max_size=10*1024*1024)
    
    # Validate type
    mime_type = await validate_file_type(file, ALLOWED_IMAGE_TYPES)
    
    # Validate image properties
    image_info = await validate_image(file)
    
    # Generate safe filename
    safe_filename = generate_safe_filename(file.filename)
    file_path = UPLOAD_DIR / safe_filename
    
    # Save file
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "filename": safe_filename,
        "size": file_size,
        "mime_type": mime_type,
        **image_info
    }

def generate_safe_filename(filename: str) -> str:
    """
    Generate safe filename: sanitize, add UUID
    """
    import uuid
    from pathlib import Path
    
    # Get extension
    ext = Path(filename).suffix
    
    # Generate UUID
    unique_id = uuid.uuid4().hex[:8]
    
    # Sanitize original name (remove special chars)
    safe_name = "".join(c for c in Path(filename).stem if c.isalnum() or c in (' ', '-', '_'))
    safe_name = safe_name[:50]  # Limit length
    
    return f"{safe_name}_{unique_id}{ext}"
\`\`\`

---

## Streaming Large Files

### Chunked Upload

\`\`\`python
"""
Stream large files in chunks
"""

@app.post("/upload/chunked/")
async def upload_large_file(file: UploadFile = File(...)):
    """
    Stream large file upload (e.g., 1GB video)
    Memory-efficient: processes chunks without loading entire file
    """
    file_path = UPLOAD_DIR / generate_safe_filename(file.filename)
    
    # Stream file in chunks
    chunk_size = 1024 * 1024  # 1MB chunks
    total_size = 0
    
    with file_path.open("wb") as buffer:
        while chunk := await file.read(chunk_size):
            buffer.write(chunk)
            total_size += len(chunk)
            
            # Could add progress tracking here
            print(f"Uploaded: {total_size / (1024*1024):.2f}MB")
    
    return {
        "filename": file_path.name,
        "size": total_size,
        "size_mb": f"{total_size / (1024*1024):.2f}MB"
    }
\`\`\`

### Progress Tracking with WebSocket

\`\`\`python
"""
Track upload progress with WebSocket
"""

from fastapi import WebSocket

# Store upload progress
upload_progress: Dict[str, dict] = {}

@app.websocket("/ws/upload/{upload_id}")
async def upload_progress_ws(websocket: WebSocket, upload_id: str):
    """
    WebSocket for real-time upload progress
    """
    await websocket.accept()
    
    try:
        while True:
            # Send progress updates
            if upload_id in upload_progress:
                await websocket.send_json(upload_progress[upload_id])
            
            await asyncio.sleep(0.5)  # Update every 500ms
            
    except WebSocketDisconnect:
        pass

@app.post("/upload/tracked/{upload_id}")
async def upload_with_progress(
    upload_id: str,
    file: UploadFile = File(...)
):
    """
    Upload file with progress tracking
    """
    file_path = UPLOAD_DIR / generate_safe_filename(file.filename)
    
    chunk_size = 1024 * 1024  # 1MB
    total_size = 0
    
    # Initialize progress
    upload_progress[upload_id] = {
        "status": "uploading",
        "uploaded": 0,
        "total": None,
        "percent": 0
    }
    
    with file_path.open("wb") as buffer:
        while chunk := await file.read(chunk_size):
            buffer.write(chunk)
            total_size += len(chunk)
            
            # Update progress
            upload_progress[upload_id] = {
                "status": "uploading",
                "uploaded": total_size,
                "total": total_size,  # Estimate
                "percent": 0  # Can't know total until complete
            }
    
    # Mark complete
    upload_progress[upload_id] = {
        "status": "complete",
        "uploaded": total_size,
        "total": total_size,
        "percent": 100
    }
    
    return {"upload_id": upload_id, "size": total_size}
\`\`\`

---

## Cloud Storage Integration

### S3 Direct Upload with Presigned URLs

\`\`\`python
"""
Direct S3 upload using presigned URLs
Avoids routing files through API server
"""

import boto3
from botocore.exceptions import ClientError

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

@app.post("/upload/presigned-url/")
async def get_presigned_upload_url(
    filename: str,
    content_type: str,
    current_user: User = Depends(get_current_user)
):
    """
    Generate presigned URL for direct S3 upload
    
    Flow:
    1. Client requests presigned URL
    2. Server generates URL (valid for 15 minutes)
    3. Client uploads directly to S3 using URL
    4. Client notifies server of completion
    """
    # Generate unique S3 key
    file_key = f"uploads/{current_user.id}/{generate_safe_filename(filename)}"
    
    try:
        # Generate presigned POST URL
        presigned_post = s3_client.generate_presigned_post(
            Bucket=S3_BUCKET,
            Key=file_key,
            Fields={
                "Content-Type": content_type,
                "x-amz-meta-user-id": str(current_user.id)
            },
            Conditions=[
                ["content-length-range", 1, 100*1024*1024],  # 1 byte - 100MB
                {"Content-Type": content_type}
            ],
            ExpiresIn=900  # 15 minutes
        )
        
        return {
            "upload_url": presigned_post["url"],
            "fields": presigned_post["fields"],
            "file_key": file_key,
            "expires_in": 900
        }
        
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/confirm/")
async def confirm_s3_upload(
    file_key: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Confirm S3 upload and create database record
    """
    # Verify file exists in S3
    try:
        response = s3_client.head_object(Bucket=S3_BUCKET, Key=file_key)
        file_size = response["ContentLength"]
        content_type = response["ContentType"]
        
    except ClientError:
        raise HTTPException(status_code=404, detail="File not found in S3")
    
    # Create database record
    upload = Upload(
        user_id=current_user.id,
        file_key=file_key,
        filename=Path(file_key).name,
        size=file_size,
        content_type=content_type,
        storage="s3"
    )
    
    db.add(upload)
    db.commit()
    
    return {
        "id": upload.id,
        "file_key": file_key,
        "size": file_size
    }
\`\`\`

---

## Streaming Responses

### Stream File Downloads

\`\`\`python
"""
Stream file downloads
"""

from fastapi.responses import StreamingResponse
import aiofiles

@app.get("/download/{file_id}")
async def download_file(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Stream file download
    Memory-efficient for large files
    """
    # Get file record
    upload = db.query(Upload).filter(
        Upload.id == file_id,
        Upload.user_id == current_user.id
    ).first()
    
    if not upload:
        raise HTTPException(status_code=404)
    
    # Stream file
    file_path = UPLOAD_DIR / upload.file_key
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    async def file_stream():
        """Generator that yields file chunks"""
        async with aiofiles.open(file_path, mode='rb') as f:
            while chunk := await f.read(1024*1024):  # 1MB chunks
                yield chunk
    
    return StreamingResponse(
        file_stream(),
        media_type=upload.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{upload.filename}"'
        }
    )

# Stream from S3
@app.get("/download/s3/{file_id}")
async def download_from_s3(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Stream file from S3
    """
    upload = db.query(Upload).filter(
        Upload.id == file_id,
        Upload.user_id == current_user.id
    ).first()
    
    if not upload:
        raise HTTPException(status_code=404)
    
    # Get S3 object
    try:
        s3_object = s3_client.get_object(Bucket=S3_BUCKET, Key=upload.file_key)
        
        async def s3_stream():
            """Stream from S3"""
            for chunk in s3_object['Body'].iter_chunks(chunk_size=1024*1024):
                yield chunk
        
        return StreamingResponse(
            s3_stream(),
            media_type=upload.content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{upload.filename}"'
            }
        )
        
    except ClientError:
        raise HTTPException(status_code=404, detail="File not found in S3")
\`\`\`

### Stream Generated Content

\`\`\`python
"""
Stream generated content (CSV, reports)
"""

@app.get("/export/users.csv")
async def export_users_csv(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Stream CSV export
    Memory-efficient: doesn't load all data at once
    """
    async def generate_csv():
        """Generator that yields CSV rows"""
        # Header
        yield "id,email,username,created_at\\n"
        
        # Stream users from database
        offset = 0
        batch_size = 1000
        
        while True:
            users = db.query(User).offset(offset).limit(batch_size).all()
            
            if not users:
                break
            
            for user in users:
                yield f"{user.id},{user.email},{user.username},{user.created_at}\\n"
            
            offset += batch_size
    
    return StreamingResponse(
        generate_csv(),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=users.csv"
        }
    )
\`\`\`

---

## Production Patterns

### Virus Scanning

\`\`\`python
"""
Scan uploaded files for viruses
"""

import clamd  # ClamAV integration

clam = clamd.ClamAV()

async def scan_file_for_viruses(file_path: Path) -> bool:
    """
    Scan file using ClamAV
    Returns True if clean, raises exception if infected
    """
    try:
        scan_result = clam.scan(str(file_path))
        
        if scan_result:
            # Virus found
            file_path.unlink()  # Delete infected file
            raise HTTPException(
                status_code=400,
                detail="Virus detected in uploaded file"
            )
        
        return True
        
    except Exception as e:
        logger.error(f"Virus scan failed: {e}")
        raise HTTPException(status_code=500, detail="Virus scan failed")
\`\`\`

---

## Summary

✅ **File uploads**: UploadFile with validation  
✅ **Validation**: Size, type, content (magic numbers)  
✅ **Streaming**: Chunked uploads for large files  
✅ **S3 integration**: Presigned URLs for direct upload  
✅ **Streaming downloads**: Memory-efficient file serving  
✅ **Progress tracking**: WebSocket for real-time updates  
✅ **Security**: Virus scanning, safe filenames  

### Best Practices

**1. Never load entire file into memory**:
- Use streaming for files > 10MB
- Process in chunks (1MB chunks)

**2. Validate everything**:
- File size (before processing)
- File type (magic numbers, not extension)
- Content (for images: dimensions, for docs: validity)

**3. Use cloud storage**:
- Direct S3 uploads (presigned URLs)
- Avoid routing files through API
- CDN for serving files

**4. Security**:
- Virus scanning (ClamAV)
- Safe filenames (UUID, sanitize)
- Access control (user owns file)

### Next Steps

In the next section, we'll explore **Error Handling & Validation**: comprehensive error handling strategies, custom exception handlers, and production error tracking.
`,
};
