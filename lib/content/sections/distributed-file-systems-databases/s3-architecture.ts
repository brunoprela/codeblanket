/**
 * Amazon S3 Architecture Section
 */

export const s3ArchitectureSection = {
  id: 's3-architecture',
  title: 'Amazon S3 Architecture',
  content: `Amazon S3 (Simple Storage Service) is the most widely used object storage service in the world, storing trillions of objects and serving millions of requests per second.

## What is Amazon S3?

**Object storage service** launched in 2006:
- Store and retrieve any amount of data
- Pay only for what you use
- 99.999999999% (11 nines) durability
- 99.99% availability SLA
- Virtually unlimited scalability

**Scale**: 
- 100+ trillion objects stored (as of 2021)
- Millions of requests per second
- Exabytes of data

---

## Object Storage vs File Storage

### File Storage (Traditional)

\`\`\`
/home/
  /user/
    /documents/
      report.pdf
      data.csv
\`\`\`

**Characteristics**:
- Hierarchical directory structure
- Files organized in folders
- POSIX operations (open, read, seek, write, close)
- Good for: OS, shared drives

### Object Storage (S3)

\`\`\`
Bucket: my-bucket
  Object: user/documents/report.pdf
  Object: user/documents/data.csv
  Object: images/photo.jpg
\`\`\`

**Characteristics**:
- Flat namespace with simulated hierarchy
- Objects identified by keys (no true folders!)
- HTTP operations (PUT, GET, DELETE)
- Immutable objects (no append, no seek)
- Rich metadata (custom key-value pairs)
- Good for: Cloud storage, backups, data lakes

**Key difference**: S3 has no real folders! \`user/documents/report.pdf\` is just a key, not a path.

---

## S3 Core Concepts

### 1. Buckets

**Container for objects**:
- Globally unique name (across all AWS accounts)
- Region-specific (data stays in region)
- Unlimited objects per bucket
- 100 buckets per account (soft limit, can increase)

**Naming rules**:
\`\`\`
✅ my-bucket-123
✅ company.data.2024
❌ MyBucket (no uppercase)
❌ my_bucket (no underscores)
❌ 192.168.1.1 (no IP format)
\`\`\`

### 2. Objects

**Data + Metadata**:
- **Key**: Unique identifier (up to 1024 bytes)
- **Value**: Object data (0 bytes to 5 TB)
- **Metadata**: System metadata + user-defined
- **Version ID**: If versioning enabled
- **Access control**: Object-level permissions

**Example**:
\`\`\`
Key: 2024/logs/application.log
Value: (log file data)
Metadata:
  Content-Type: text/plain
  x-amz-server-side-encryption: AES256
  Custom-Tag: production
Size: 500 MB
Last Modified: 2024-01-15T10:30:00Z
ETag: "abc123def456..."
\`\`\`

### 3. Keys (Object Identifiers)

**Key design matters**:

**Bad (hot partition)**:
\`\`\`
logs/2024-01-15-10:00.log
logs/2024-01-15-10:01.log
logs/2024-01-15-10:02.log
\`\`\`
All start with "logs/" → same partition!

**Good (distributed)**:
\`\`\`
a3f9/logs/2024-01-15-10:00.log  (hash prefix)
b7d2/logs/2024-01-15-10:01.log
c1e8/logs/2024-01-15-10:02.log
\`\`\`
Random prefixes → distributed across partitions!

**Note**: As of 2018, S3 automatically handles this, but good key design still helps.

---

## S3 Architecture

### High-Level Design

\`\`\`
                    ┌──────────────────┐
                    │   API Gateway    │
                    │  (Load Balancer) │
                    └──────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Storage    │    │   Storage    │    │   Storage    │
│   Node 1     │    │   Node 2     │    │   Node 3     │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌──────────────────┐
                    │  Metadata Store  │
                    │  (Distributed DB)│
                    └──────────────────┘
\`\`\`

### Internal Architecture (Simplified)

**1. Request Routing**:
- Client → AWS edge location (CloudFront or Route 53)
- Routed to appropriate S3 endpoint
- Load balanced across many servers

**2. Metadata Layer**:
- Maps bucket + key → physical storage locations
- Distributed database (rumored to use Paxos/Raft)
- Replicated across multiple AZs

**3. Storage Layer**:
- Objects stored on disk across multiple devices
- Replicated across multiple Availability Zones (AZs)
- Uses erasure coding + replication for durability

---

## S3 Durability and Availability

### 11 Nines Durability (99.999999999%)

**What does 11 nines mean?**
- Store 10 million objects
- Expect to lose 1 object every 10,000 years!

**How S3 achieves this**:

**1. Redundancy across AZs**:
\`\`\`
Availability Zone 1    Availability Zone 2    Availability Zone 3
     Replica 1              Replica 2              Replica 3
\`\`\`

**2. Erasure coding** (for Standard storage class):
- Object split into data + parity chunks
- Can reconstruct object from subset of chunks
- More space-efficient than full replication

**Example (simplified)**:
\`\`\`
Original: [A, B, C, D]
Encoded:  [A, B, C, D, P1, P2]
Can lose any 2 chunks and still reconstruct!
\`\`\`

**3. Continuous verification**:
- Background integrity checks
- Automatic healing of corrupt data
- Proactive replacement before failures

### Availability (99.99%)

**Expected downtime**: 52 minutes/year

**How S3 achieves availability**:
- Multiple replicas in multiple AZs
- Automatic failover
- No single point of failure
- Partitioning for load distribution

---

## S3 Consistency Model

### Evolution of Consistency

**Pre-2020 (Eventual Consistency)**:
\`\`\`
PUT object → Success
GET object → Might return old version or 404 (for ~1 second)
\`\`\`

**Post-December 2020 (Strong Consistency)** 🎉:
\`\`\`
PUT object → Success
GET object → Always returns latest version!
\`\`\`

### Strong Read-After-Write Consistency

**Guarantees**:
- PUT new object → immediately readable
- Overwrite (PUT to existing key) → immediately see new version
- DELETE object → immediately get 404
- List operations → immediately reflect changes

**Examples**:

\`\`\`
// Scenario 1: New object
PUT /bucket/key1 (data: "v1") → 200 OK
GET /bucket/key1 → Returns "v1" (immediately!)

// Scenario 2: Overwrite
PUT /bucket/key1 (data: "v2") → 200 OK
GET /bucket/key1 → Returns "v2" (not "v1"!)

// Scenario 3: Delete
DELETE /bucket/key1 → 204 No Content
GET /bucket/key1 → 404 Not Found (immediately!)

// Scenario 4: List
PUT /bucket/key2
List bucket → Includes key2 (immediately!)
\`\`\`

**How S3 achieves strong consistency**:
- Distributed consensus protocol
- Write not acknowledged until replicated to quorum
- Metadata updates are atomic

**Impact**:
- ✅ Simpler application logic (no retry loops)
- ✅ Safe to read immediately after write
- ✅ Database-like semantics
- Slight write latency increase (negligible)

---

## S3 Storage Classes

Different classes for different access patterns:

### 1. S3 Standard

**Use case**: Frequently accessed data

**Characteristics**:
- 11 nines durability
- 99.99% availability
- Low latency, high throughput
- Replicated across ≥3 AZs

**Cost**: $0.023/GB-month (us-east-1)

**Example**: Active website assets, mobile app content

### 2. S3 Intelligent-Tiering

**Use case**: Unknown or changing access patterns

**How it works**:
- Monitors access patterns
- Automatically moves to cheaper tier if not accessed for 30 days
- No retrieval fees!

**Tiers**:
- Frequent Access (< 30 days)
- Infrequent Access (30-90 days)
- Archive Instant Access (> 90 days)
- Archive Access (optional, > 90 days)
- Deep Archive Access (optional, > 180 days)

**Cost**: Small monitoring fee + storage cost

### 3. S3 Standard-IA (Infrequent Access)

**Use case**: Data accessed less than once a month

**Characteristics**:
- Same durability and latency as Standard
- Lower storage cost
- Retrieval fee per GB

**Cost**: $0.0125/GB-month + $0.01/GB retrieval

**Example**: Backups, disaster recovery files

### 4. S3 One Zone-IA

**Use case**: Infrequently accessed, reproducible data

**Characteristics**:
- Stored in single AZ (not 3 AZs)
- 20% cheaper than Standard-IA
- Less durable (AZ failure = data loss)

**Cost**: $0.01/GB-month

**Example**: Secondary backups, thumbnails (can regenerate)

### 5. S3 Glacier Instant Retrieval

**Use case**: Archive with instant access (milliseconds)

**Cost**: $0.004/GB-month + higher retrieval cost

**Example**: Medical images, news archives

### 6. S3 Glacier Flexible Retrieval

**Use case**: Archive, retrieval in minutes to hours

**Retrieval options**:
- Expedited: 1-5 minutes ($0.03/GB)
- Standard: 3-5 hours ($0.01/GB)
- Bulk: 5-12 hours ($0.0025/GB)

**Cost**: $0.0036/GB-month

**Example**: Annual records, compliance archives

### 7. S3 Glacier Deep Archive

**Use case**: Long-term archive, rarely accessed

**Retrieval**: 12-48 hours

**Cost**: $0.00099/GB-month (cheapest!)

**Example**: 7-10 year regulatory archives

---

## S3 Versioning

**Enable versioning** to keep multiple versions of objects:

\`\`\`
# Versioning OFF
PUT /bucket/doc.txt (v1) → Key: doc.txt
PUT /bucket/doc.txt (v2) → Key: doc.txt (v1 lost!)

# Versioning ON
PUT /bucket/doc.txt (v1) → Version ID: 111
PUT /bucket/doc.txt (v2) → Version ID: 222
PUT /bucket/doc.txt (v3) → Version ID: 333

GET /bucket/doc.txt → Returns v3 (latest)
GET /bucket/doc.txt?versionId=111 → Returns v1
\`\`\`

**Benefits**:
- ✅ Protect against accidental deletion
- ✅ Recover previous versions
- ✅ Compliance requirements

**Cost**:
- All versions stored and charged
- Delete creates "delete marker" (doesn't free space)
- Must permanently delete versions to free space

**Lifecycle integration**:
\`\`\`xml
<LifecycleConfiguration>
  <Rule>
    <NoncurrentVersionExpiration>
      <NoncurrentDays>30</NoncurrentDays>
    </NoncurrentVersionExpiration>
  </Rule>
</LifecycleConfiguration>
\`\`\`

---

## S3 Performance

### Request Rates

**Per prefix**: 
- 3,500 PUT/COPY/POST/DELETE requests/second
- 5,500 GET/HEAD requests/second

**Scaling**:
- Use multiple prefixes for higher throughput
- Example: 10 prefixes = 55,000 GET requests/second

### Transfer Acceleration

**Problem**: Uploading to distant region is slow

**Solution**: Transfer Acceleration via CloudFront edge locations

\`\`\`
Without acceleration:
Client (Tokyo) → Internet → S3 (us-east-1) = 250ms RTT

With acceleration:
Client (Tokyo) → CloudFront (Tokyo) → AWS backbone → S3 (us-east-1) = 50ms RTT
\`\`\`

**Use case**: Users globally uploading large files

**Cost**: $0.04-$0.08/GB

### Multipart Upload

**For files > 100 MB**:

\`\`\`
Large file (5 GB)
    ↓
Split into parts (100 MB each)
    ↓
Upload parts in parallel
    ↓
Complete multipart upload
\`\`\`

**Benefits**:
- ✅ Parallel uploads (faster)
- ✅ Resume on failure (only re-upload failed parts)
- ✅ Upload while creating file (streaming)

**Code example**:
\`\`\`python
# Initiate multipart upload
response = s3.create_multipart_upload(Bucket='my-bucket', Key='large-file.zip')
upload_id = response['UploadId']

# Upload parts
parts = []
for i, part_data in enumerate(file_parts):
    response = s3.upload_part(
        Bucket='my-bucket',
        Key='large-file.zip',
        PartNumber=i+1,
        UploadId=upload_id,
        Body=part_data
    )
    parts.append({'PartNumber': i+1, 'ETag': response['ETag']})

# Complete upload
s3.complete_multipart_upload(
    Bucket='my-bucket',
    Key='large-file.zip',
    UploadId=upload_id,
    MultipartUpload={'Parts': parts}
)
\`\`\`

**Best practices**:
- Use 100 MB part size (5 MB to 5 GB allowed)
- Upload up to 10,000 parts
- Maximum object size: 5 TB

### Byte-Range Fetches

**Download specific byte ranges**:

\`\`\`python
# Download first 1MB of 1GB file
response = s3.get_object(
    Bucket='my-bucket',
    Key='large-file.zip',
    Range='bytes=0-1048576'
)
\`\`\`

**Benefits**:
- ✅ Parallel downloads (faster)
- ✅ Download only needed portion
- ✅ Resume downloads

---

## S3 Security

### 1. Encryption

**Server-Side Encryption (SSE)**:

**SSE-S3** (S3-managed keys):
\`\`\`
PUT /bucket/object
x-amz-server-side-encryption: AES256
\`\`\`
- S3 manages encryption keys
- AES-256 encryption
- Free

**SSE-KMS** (AWS KMS keys):
\`\`\`
PUT /bucket/object
x-amz-server-side-encryption: aws:kms
x-amz-server-side-encryption-aws-kms-key-id: key-id
\`\`\`
- More control over keys
- Audit trail in CloudTrail
- Additional cost

**SSE-C** (Customer-provided keys):
\`\`\`
PUT /bucket/object
x-amz-server-side-encryption-customer-algorithm: AES256
x-amz-server-side-encryption-customer-key: <base64-key>
\`\`\`
- You manage keys
- Send key with every request
- No KMS cost

**Client-Side Encryption**:
- Encrypt before uploading
- Decrypt after downloading
- Full control, but complex

### 2. Access Control

**Bucket Policies** (resource-based):
\`\`\`json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"AWS": "arn:aws:iam::123456789:user/Alice"},
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::my-bucket/*"
  }]
}
\`\`\`

**IAM Policies** (identity-based):
\`\`\`json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "s3:*",
    "Resource": "arn:aws:s3:::my-bucket/*"
  }]
}
\`\`\`

**ACLs** (legacy, avoid):
- Object-level or bucket-level
- Limited controls
- Bucket policies preferred

**Presigned URLs** (temporary access):
\`\`\`python
# Generate presigned URL valid for 1 hour
url = s3.generate_presigned_url(
    'get_object',
    Params={'Bucket': 'my-bucket', 'Key': 'private-file.pdf'},
    ExpiresIn=3600
)
\`\`\`

### 3. Block Public Access

**Default**: All public access blocked (since 2018)

**Settings**:
- Block public ACLs
- Ignore public ACLs
- Block public bucket policies
- Restrict public buckets

**Best practice**: Keep public access blocked unless explicitly needed

---

## S3 Lifecycle Policies

**Automate transitions and expirations**:

\`\`\`xml
<LifecycleConfiguration>
  <Rule>
    <ID>Archive old logs</ID>
    <Status>Enabled</Status>
    <Prefix>logs/</Prefix>
    
    <!-- Transition to IA after 30 days -->
    <Transition>
      <Days>30</Days>
      <StorageClass>STANDARD_IA</StorageClass>
    </Transition>
    
    <!-- Transition to Glacier after 90 days -->
    <Transition>
      <Days>90</Days>
      <StorageClass>GLACIER</StorageClass>
    </Transition>
    
    <!-- Delete after 365 days -->
    <Expiration>
      <Days>365</Days>
    </Expiration>
  </Rule>
</LifecycleConfiguration>
\`\`\`

**Use cases**:
- Automatically archive old data
- Delete temporary files
- Comply with retention policies
- Save costs

---

## S3 Event Notifications

**Trigger actions on S3 events**:

\`\`\`
S3 Event → Lambda / SQS / SNS
\`\`\`

**Events**:
- Object created (PUT, POST, COPY, multipart upload)
- Object removed (DELETE)
- Object restore from Glacier
- Replication events

**Example**:
\`\`\`json
{
  "Event": "s3:ObjectCreated:Put",
  "Destination": {
    "LambdaFunctionArn": "arn:aws:lambda:us-east-1:123:function:ProcessImage"
  },
  "Filter": {
    "Key": {
      "FilterRules": [{"Name": "prefix", "Value": "images/"}]
    }
  }
}
\`\`\`

**Use cases**:
- Image thumbnail generation
- Video transcoding
- Data processing pipeline
- Backup verification

---

## S3 Replication

**Cross-Region Replication (CRR)**:
- Replicate objects to bucket in different region
- Use cases: Compliance, lower latency, disaster recovery

**Same-Region Replication (SRR)**:
- Replicate within same region
- Use cases: Log aggregation, prod/test sync

**Requirements**:
- Versioning enabled on both buckets
- Proper IAM permissions
- Can replicate encrypted objects

**Options**:
- Replicate all objects or subset (prefix/tags)
- Replicate delete markers (optional)
- Replicate previous versions (optional)
- Change storage class in destination

---

## S3 Use Cases

### 1. Static Website Hosting

\`\`\`
Enable website hosting:
- Index document: index.html
- Error document: error.html
- Access via: bucket-name.s3-website-region.amazonaws.com
\`\`\`

**Benefits**:
- ✅ Cheap (pennies per month)
- ✅ Scalable (handles any traffic)
- ✅ Integrate with CloudFront for CDN

### 2. Data Lake

\`\`\`
Raw data → S3 → Athena/Glue/EMR → Analytics
\`\`\`

**Why S3 for data lake**:
- ✅ Unlimited storage
- ✅ Low cost
- ✅ Integrates with analytics tools
- ✅ Multiple storage classes

### 3. Backup and Archive

**Backup strategy**:
- Daily backups → S3 Standard
- 30-day lifecycle → S3 Standard-IA
- 90-day lifecycle → S3 Glacier
- 7-year lifecycle → S3 Deep Archive

### 4. Application Storage

**Examples**:
- User uploads (photos, documents)
- Application logs
- Database backups
- Static assets (CSS, JS, images)

---

## S3 Cost Optimization

### 1. Use Appropriate Storage Class

\`\`\`
Frequently accessed → Standard
Monthly access → Standard-IA
Rarely accessed → Glacier
Unknown pattern → Intelligent-Tiering
\`\`\`

### 2. Lifecycle Policies

Automatically transition to cheaper classes

### 3. S3 Analytics

Enable Storage Class Analysis:
- Analyzes access patterns
- Recommends lifecycle policies
- Shows cost savings opportunities

### 4. Delete Incomplete Multipart Uploads

\`\`\`xml
<LifecycleConfiguration>
  <Rule>
    <AbortIncompleteMultipartUpload>
      <DaysAfterInitiation>7</DaysAfterInitiation>
    </AbortIncompleteMultipartUpload>
  </Rule>
</LifecycleConfiguration>
\`\`\`

### 5. Compress Data

Store compressed (gzip, brotli) to reduce storage and transfer costs

---

## Interview Tips

**Explain S3 in 2 minutes**:
"S3 is AWS's object storage with 11 nines durability and strong consistency. Objects are identified by keys (not paths) and stored across multiple AZs with erasure coding. It offers multiple storage classes from frequently-accessed Standard to rarely-accessed Glacier Deep Archive. Key features include versioning, lifecycle policies, encryption, and integration with AWS services. S3 scales automatically and handles trillions of objects. It's perfect for backups, data lakes, static websites, and application storage."

**Key trade-offs**:
- Object storage vs file storage: Simplicity vs POSIX semantics
- Durability vs cost: Standard vs One Zone-IA
- Availability vs cost: Instant access vs Glacier retrieval time
- Strong consistency vs performance: Slight write latency increase

**Common mistakes**:
- ❌ Treating keys as file paths (they're not!)
- ❌ Using versioning without lifecycle policies (cost explosion)
- ❌ Not using multipart upload for large files
- ❌ Ignoring key design for high throughput

---

## Key Takeaways

🔑 S3 = object storage with 11 nines durability, strong consistency
🔑 Objects identified by keys, stored across multiple AZs
🔑 7 storage classes for different access patterns and costs
🔑 Versioning protects against accidental deletion
🔑 Lifecycle policies automate cost optimization
🔑 Multipart upload for files > 100 MB
🔑 Encryption options: SSE-S3, SSE-KMS, SSE-C, client-side
🔑 Integrate with Lambda for event-driven processing
🔑 Perfect for backups, data lakes, static hosting, app storage
🔑 Strong consistency since Dec 2020 = simpler application logic
`,
};
