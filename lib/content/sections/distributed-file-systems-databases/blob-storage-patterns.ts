/**
 * Blob Storage Patterns Section
 */

export const blobStorageSection = {
  id: 'blob-storage-patterns',
  title: 'Blob Storage Patterns',
  content: `Blob (Binary Large Object) storage is a fundamental pattern for storing unstructured data like images, videos, documents, and backups at massive scale.

## What is Blob Storage?

**Blob storage** = Object storage for unstructured data

**Characteristics**:
- Store any type of binary data
- Flat namespace (no directory hierarchy)
- HTTP/REST API access
- Massively scalable
- Highly durable
- Cost-effective

**Examples**:
- AWS S3
- Azure Blob Storage  
- Google Cloud Storage
- MinIO (self-hosted)

---

## Blob Storage vs Other Storage Types

| Feature         | Blob Storage | File Storage | Block Storage |
|-----------------|--------------|--------------|---------------|
| Interface       | HTTP/REST    | POSIX/NFS    | Block I/O     |
| Hierarchy       | Flat (keys)  | Hierarchical | None          |
| Typical size    | MB to TB     | KB to GB     | Fixed blocks  |
| Mutability      | Immutable    | Mutable      | Mutable       |
| Use case        | Backups, media| Shared files | Databases, VMs|
| Access          | Over network | Network/local| Direct attach |
| Consistency     | Eventual/strong| Strong    | Strong        |

**When to use Blob**:
- ✅ Static content (images, videos, documents)
- ✅ Backups and archives
- ✅ Data lakes
- ✅ Unstructured data
- ✅ Content distribution
- ✅ Cost-sensitive workloads

**When NOT to use Blob**:
- ❌ Frequently modified files
- ❌ Random access within file
- ❌ POSIX operations required
- ❌ Low latency requirements (< 10ms)
- ❌ Transactional workloads

---

## Common Blob Storage Patterns

### 1. Static Content Hosting

**Pattern**: Serve static assets (images, CSS, JS) from blob storage

\`\`\`
User Request → CDN (CloudFront) → Blob Storage (S3)
              ↑ Cache hit (fast)
              ↓ Cache miss (fetch from S3)
\`\`\`

**Implementation**:
\`\`\`
# S3 bucket
my-website-assets/
  css/
    styles.css
  js/
    app.js
  images/
    logo.png
    hero.jpg
\`\`\`

**Benefits**:
- ✅ Offload from web servers
- ✅ Unlimited scalability
- ✅ Low cost
- ✅ CDN integration for fast delivery

**Configuration**:
\`\`\`json
{
  "CacheControl": "max-age=31536000",  // 1 year
  "ContentType": "image/jpeg"
}
\`\`\`

**Best practices**:
- Set correct Content-Type headers
- Use cache-control headers
- Implement cache busting (versioned filenames)
- Integrate with CDN
- Compress assets (gzip, brotli)

### 2. User-Generated Content (UGC)

**Pattern**: Users upload photos, videos, documents

\`\`\`
Client → Upload directly to Blob Storage (presigned URL)
         ↓
      Processing (Lambda/Function)
         ↓
      Thumbnail, validation, virus scan
         ↓
      Store metadata in database
\`\`\`

**Direct upload flow**:
\`\`\`
1. Client requests upload URL from backend
2. Backend generates presigned URL (valid for 15 minutes)
3. Client uploads directly to S3 using presigned URL
4. S3 triggers Lambda on upload
5. Lambda processes, creates thumbnail
6. Lambda updates database with metadata
\`\`\`

**Code example (presigned URL)**:
\`\`\`python
# Backend generates presigned URL
presigned_url = s3.generate_presigned_url(
    'put_object',
    Params={
        'Bucket': 'ugc-uploads',
        'Key': f'uploads/{user_id}/{uuid}.jpg',
        'ContentType': 'image/jpeg'
    },
    ExpiresIn=900  # 15 minutes
)
return {'upload_url': presigned_url}
\`\`\`

\`\`\`javascript
// Client uploads directly
fetch (presignedUrl, {
  method: 'PUT',
  body: imageFile,
  headers: {'Content-Type': 'image/jpeg'}
})
\`\`\`

**Benefits**:
- ✅ Offload upload bandwidth from servers
- ✅ Scales automatically
- ✅ Secure (time-limited presigned URLs)
- ✅ Process asynchronously

### 3. Backup and Archive

**Pattern**: Store backups with lifecycle policies

\`\`\`
Database → Nightly backup → Blob Storage
                                ↓
                         Lifecycle Policy
                                ↓
          Day 0-30: Standard Storage
          Day 31-90: Infrequent Access
          Day 91+: Glacier
          Day 365+: Delete
\`\`\`

**Lifecycle configuration**:
\`\`\`xml
<LifecycleConfiguration>
  <Rule>
    <Prefix>backups/mysql/</Prefix>
    <Transition>
      <Days>30</Days>
      <StorageClass>STANDARD_IA</StorageClass>
    </Transition>
    <Transition>
      <Days>90</Days>
      <StorageClass>GLACIER</StorageClass>
    </Transition>
    <Expiration>
      <Days>365</Days>
    </Expiration>
  </Rule>
</LifecycleConfiguration>
\`\`\`

**3-2-1 backup strategy**:
- **3** copies of data
- **2** different media types
- **1** offsite copy (blob storage!)

**Example implementation**:
\`\`\`bash
# Daily MySQL backup
mysqldump --all-databases | gzip > backup-$(date +%Y%m%d).sql.gz

# Upload to S3
aws s3 cp backup-*.sql.gz s3://backups/mysql/

# S3 lifecycle handles aging
\`\`\`

### 4. Data Lake Storage

**Pattern**: Central repository for all data

\`\`\`
Data Sources → Blob Storage (Data Lake)
  - App logs        ↓
  - DB exports   Processing
  - IoT data        ↓
  - Web analytics   Analytics (Athena, Spark)
                    ↓
                  Insights
\`\`\`

**Data lake organization**:
\`\`\`
data-lake/
  raw/                  (immutable raw data)
    logs/2024/01/15/
    events/2024/01/15/
  processed/            (cleaned, transformed)
    logs-parsed/
    events-aggregated/
  curated/              (business-ready datasets)
    user-behavior/
    sales-metrics/
\`\`\`

**Best practices**:
- Partition by date (year/month/day)
- Use columnar formats (Parquet, ORC)
- Compress data
- Tag data for governance
- Implement data catalog (AWS Glue, Azure Data Catalog)

### 5. Media Processing Pipeline

**Pattern**: Upload → Process → Distribute

\`\`\`
User uploads video → S3 bucket (raw/)
        ↓ S3 event trigger
      Lambda starts transcoding job (MediaConvert, FFmpeg)
        ↓
    Multiple formats generated
        ↓
  S3 bucket (processed/)
        ↓
    CloudFront CDN
        ↓
    End users
\`\`\`

**Directory structure**:
\`\`\`
media/
  raw/
    videos/
      abc123.mp4
  processed/
    videos/
      abc123-1080p.mp4
      abc123-720p.mp4
      abc123-480p.mp4
    thumbnails/
      abc123-thumb.jpg
\`\`\`

**Adaptive bitrate streaming**:
\`\`\`
Original → Transcode → HLS/DASH segments
                    ↓
              1080p, 720p, 480p, 360p
                    ↓
            Player chooses based on bandwidth
\`\`\`

### 6. Multipart Upload Pattern

**For large files (> 100 MB)**:

\`\`\`
Large file (5 GB)
    ↓
Split into parts (100 MB each, 50 parts)
    ↓
Upload parts in parallel (10 concurrent)
    ↓
Complete multipart upload
\`\`\`

**Implementation**:
\`\`\`python
import boto3
from boto3.s3.transfer import TransferConfig

# Configure multipart
config = TransferConfig(
    multipart_threshold=100 * 1024 * 1024,  # 100 MB
    multipart_chunksize=100 * 1024 * 1024,  # 100 MB
    max_concurrency=10
)

# Upload with automatic multipart
s3.upload_file(
    'large-file.zip',
    'my-bucket',
    'large-file.zip',
    Config=config
)
\`\`\`

**Benefits**:
- ✅ Parallel uploads (faster)
- ✅ Resume on failure
- ✅ Upload while creating file (streaming)

**Best practices**:
- Use for files > 100 MB
- 100 MB part size recommended
- Clean up incomplete uploads via lifecycle policy

### 7. Content Deduplication

**Pattern**: Store files only once, reference multiple times

\`\`\`
File uploaded → Calculate hash (SHA-256)
                    ↓
            Check if hash exists in DB
                    ↓
    Yes: Reference existing file
    No:  Upload to blob storage, store hash in DB
\`\`\`

**Implementation**:
\`\`\`python
import hashlib

# Calculate file hash
def calculate_hash (file_path):
    sha256 = hashlib.sha256()
    with open (file_path, 'rb') as f:
        for chunk in iter (lambda: f.read(4096), b''):
            sha256.update (chunk)
    return sha256.hexdigest()

# Check for duplicate
file_hash = calculate_hash('document.pdf')
existing = db.query (f"SELECT blob_key FROM files WHERE hash = '{file_hash}'")

if existing:
    # Reference existing blob
    db.insert({'user_id': user_id, 'blob_key': existing.blob_key, 'hash': file_hash})
else:
    # Upload new blob
    blob_key = f'files/{file_hash}.pdf'
    s3.upload_file('document.pdf', 'my-bucket', blob_key)
    db.insert({'user_id': user_id, 'blob_key': blob_key, 'hash': file_hash})
\`\`\`

**Benefits**:
- ✅ Save storage space
- ✅ Faster uploads (if duplicate)
- ✅ Bandwidth savings

**Use cases**:
- File sharing platforms (Dropbox)
- Email attachments
- Document management

### 8. Versioning and Rollback

**Pattern**: Keep multiple versions of files

\`\`\`
# S3 versioning enabled
PUT /bucket/config.json (v1) → Version ID: abc123
PUT /bucket/config.json (v2) → Version ID: def456
PUT /bucket/config.json (v3) → Version ID: ghi789

GET /bucket/config.json → Returns v3 (latest)
GET /bucket/config.json?versionId=abc123 → Returns v1
\`\`\`

**Rollback implementation**:
\`\`\`python
# Rollback to previous version
def rollback (bucket, key, target_version_id):
    # Copy old version as new current version
    s3.copy_object(
        CopySource={'Bucket': bucket, 'Key': key, 'VersionId': target_version_id},
        Bucket=bucket,
        Key=key
    )
\`\`\`

**Lifecycle for old versions**:
\`\`\`xml
<NoncurrentVersionExpiration>
  <NoncurrentDays>30</NoncurrentDays>
</NoncurrentVersionExpiration>
\`\`\`

---

## Advanced Patterns

### 9. Presigned POST for Browser Uploads

**Allow browsers to upload directly** without exposing credentials:

\`\`\`python
# Backend generates presigned POST
presigned_post = s3.generate_presigned_post(
    Bucket='uploads',
    Key='uploads/\${filename}',
    Fields={'acl': 'private', 'Content-Type': 'image/jpeg'},
    Conditions=[
        {'acl': 'private'},
        ['content-length-range', 1024, 10485760],  # 1KB to 10MB
        ['starts-with', '$key', 'uploads/']
    ],
    ExpiresIn=900
)
\`\`\`

**Browser HTML form**:
\`\`\`html
<form action="{presigned_post['url']}" method="post" enctype="multipart/form-data">
  <input type="hidden" name="key" value="uploads/\${filename}">
  <input type="hidden" name="acl" value="private">
  <input type="hidden" name="policy" value="{presigned_post['fields']['policy']}">
  <input type="hidden" name="x-amz-signature" value="{presigned_post['fields']['x-amz-signature']}">
  <input type="file" name="file">
  <button type="submit">Upload</button>
</form>
\`\`\`

### 10. Signed URLs for Time-Limited Access

**Share private files temporarily**:

\`\`\`python
# Generate URL valid for 1 hour
url = s3.generate_presigned_url(
    'get_object',
    Params={'Bucket': 'private-files', 'Key': 'report.pdf'},
    ExpiresIn=3600
)
# Share URL with user
\`\`\`

**Use cases**:
- Sharing confidential documents
- Download links in emails
- Temporary access to paid content

### 11. Event-Driven Processing

**Trigger actions on blob events**:

\`\`\`
S3 Event → SQS Queue → Worker processes
        ↓
   Lambda function
\`\`\`

**Example: Image processing**:
\`\`\`python
# Lambda triggered on S3 upload
def handler (event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        
        # Download image
        s3.download_file (bucket, key, '/tmp/image.jpg')
        
        # Create thumbnail
        create_thumbnail('/tmp/image.jpg', '/tmp/thumb.jpg')
        
        # Upload thumbnail
        s3.upload_file('/tmp/thumb.jpg', bucket, f'thumbnails/{key}')
\`\`\`

### 12. Cross-Region Replication

**Replicate for disaster recovery or low latency**:

\`\`\`
Primary Region (us-east-1)
    Bucket A
        ↓ Replication
Secondary Region (eu-west-1)
    Bucket B (replica)
\`\`\`

**Use cases**:
- Disaster recovery
- Compliance (data residency)
- Low-latency access for global users

---

## Performance Optimization

### 1. Parallel Uploads/Downloads

\`\`\`python
from concurrent.futures import ThreadPoolExecutor

def upload_part (part):
    # Upload one part
    pass

with ThreadPoolExecutor (max_workers=10) as executor:
    executor.map (upload_part, parts)
\`\`\`

### 2. Connection Pooling

\`\`\`python
from botocore.config import Config

config = Config(
    max_pool_connections=50  # Increase from default 10
)
s3 = boto3.client('s3', config=config)
\`\`\`

### 3. Request Routing

**Use appropriate endpoint**:
- S3 transfer acceleration for global uploads
- Regional endpoints for same-region access
- S3 access points for complex access patterns

### 4. Caching Strategy

**CDN caching**:
\`\`\`
Cache-Control: max-age=31536000, immutable  (static assets)
Cache-Control: max-age=3600                  (frequently updated)
Cache-Control: no-cache                      (dynamic content)
\`\`\`

---

## Security Best Practices

### 1. Principle of Least Privilege

\`\`\`json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject"],
  "Resource": "arn:aws:s3:::my-bucket/public/*"
}
\`\`\`

### 2. Encryption

**At rest**: SSE-S3, SSE-KMS, or client-side
**In transit**: HTTPS/TLS

### 3. Access Logging

\`\`\`
Enable S3 access logging → Log bucket
Analyze logs for suspicious activity
\`\`\`

### 4. Bucket Policies

\`\`\`json
{
  "Statement": [{
    "Effect": "Deny",
    "Principal": "*",
    "Action": "s3:*",
    "Resource": "arn:aws:s3:::my-bucket/*",
    "Condition": {
      "Bool": {"aws:SecureTransport": "false"}
    }
  }]
}
\`\`\`
Enforce HTTPS!

### 5. Versioning + MFA Delete

Enable MFA for deleting versions (extra protection)

---

## Cost Optimization

### 1. Storage Class Optimization

\`\`\`
Hot data → Standard
Warm data → Standard-IA
Cold data → Glacier
Unknown → Intelligent-Tiering
\`\`\`

### 2. Lifecycle Policies

Automate transitions to cheaper classes

### 3. Compression

\`\`\`bash
# Compress before upload
gzip large-file.log
aws s3 cp large-file.log.gz s3://bucket/
\`\`\`

**Savings**: 70-90% for text files

### 4. Request Optimization

- Batch operations
- Use S3 Select (query data without downloading)
- Use byte-range fetches (download only needed parts)

### 5. Clean Up Incomplete Uploads

\`\`\`xml
<AbortIncompleteMultipartUpload>
  <DaysAfterInitiation>7</DaysAfterInitiation>
</AbortIncompleteMultipartUpload>
\`\`\`

---

## Monitoring and Debugging

### Key Metrics

**S3 metrics**:
- BucketSizeBytes
- NumberOfObjects
- AllRequests (request rate)
- 4xxErrors, 5xxErrors
- FirstByteLatency
- BytesDownloaded, BytesUploaded

**CloudWatch alarms**:
\`\`\`
Alert if 4xxErrors > 100/minute
Alert if FirstByteLatency > 1000ms (p99)
\`\`\`

### Access Patterns

**S3 Storage Class Analysis**:
- Shows access patterns
- Recommends lifecycle policies
- Visualizes cost savings

---

## Interview Tips

**Explain blob storage patterns in 2 minutes**:
"Blob storage is for unstructured data like images, videos, backups. Key patterns include: direct upload via presigned URLs to offload servers, lifecycle policies for automatic archiving, multipart upload for large files, and event-driven processing via S3 events triggering Lambda. For static content, integrate with CDN. For backups, use lifecycle transitions to Glacier. For data lakes, organize by date partitions. Deduplication via hash checking saves space. Versioning enables rollback. Always encrypt at rest and in transit."

**Common patterns to mention**:
- Static content hosting + CDN
- UGC with presigned URLs
- Media processing pipeline
- Data lake with partitioning
- Backup with lifecycle policies

**Trade-offs to discuss**:
- Direct upload: Offload servers but complex client logic
- Versioning: Safety vs storage cost
- Multipart: Speed vs complexity
- Replication: Durability vs cost

---

## Key Takeaways

🔑 Blob storage = HTTP-accessible, immutable object storage
🔑 Presigned URLs enable direct client uploads (offload servers)
🔑 Multipart upload for large files (> 100 MB)
🔑 Lifecycle policies automate archiving and cost optimization
🔑 Event-driven processing via S3 events + Lambda
🔑 Deduplication via hash checking saves storage
🔑 Versioning enables rollback and compliance
🔑 CDN integration for fast global content delivery
🔑 Cross-region replication for DR and low latency
🔑 Always encrypt, use HTTPS, follow least privilege
`,
};
