/**
 * Design Dropbox Section
 */

export const dropboxSection = {
  id: 'dropbox',
  title: 'Design Dropbox',
  content: `Dropbox is a cloud file storage and synchronization service with 700M users storing petabytes of data. The core challenges are: efficiently syncing files across devices, handling large files (up to 50 GB), detecting and resolving conflicts, and minimizing bandwidth usage through delta sync and deduplication.

## Problem Statement

Design Dropbox with:
- **File Upload/Download**: Store files in cloud, access from any device
- **File Synchronization**: Changes on one device sync to all devices automatically
- **Offline Access**: Edit files offline, sync when online
- **File Sharing**: Share files/folders with other users
- **Version History**: Restore previous versions of files
- **Conflict Resolution**: Handle concurrent edits on multiple devices
- **Large File Support**: Efficiently handle multi-GB files
- **Block-Level Deduplication**: Save storage by detecting duplicate blocks

**Scale**: 700M users, 500M files/day uploaded, 100 PB total storage

---

## Step 1: Requirements

### Functional Requirements
1. **Upload/Download**: Files up to 50 GB per file
2. **Sync**: Automatic synchronization across devices
3. **Offline Mode**: Work offline, sync when reconnected
4. **Sharing**: Share files/folders, set permissions
5. **Versioning**: Keep 30 days of file history
6. **Conflict Resolution**: Detect conflicting edits
7. **Selective Sync**: Choose which folders to sync

### Non-Functional Requirements
1. **Reliability**: 99.99% uptime, no data loss
2. **Performance**: Fast sync (< 10 sec for small files)
3. **Bandwidth Efficiency**: Delta sync (only changed blocks)
4. **Storage Efficiency**: Deduplication
5. **Scalable**: 100 PB+ storage, 700M users

---

## Step 2: High-Level Architecture

\`\`\`
[Desktop/Mobile App]
        ↓
[Sync Client] → [Watcher: Monitors file changes]
        ↓
[Chunking: Split files into 4 MB blocks]
        ↓
[Upload to Cloud]
        ↓
[Load Balancer]
        ↓
[Metadata Service] → [PostgreSQL: File metadata, blocks]
        ↓
[Block Storage] → [S3: Actual file blocks]
        ↓
[Notification Service] → [Push to other devices]
\`\`\`

---

## Step 3: File Chunking & Delta Sync

**Core Optimization**: Don't upload entire file if only few bytes changed.

**Algorithm**:
\`\`\`
1. USER EDITS FILE
   - User edits 1 MB of a 100 MB file
   
2. CHUNKING
   - Split file into 4 MB blocks (fixed size)
   - 100 MB file = 25 blocks
   - Calculate SHA-256 hash for each block
   
3. DETECT CHANGES
   - Compare new block hashes with previous version
   - Block 1-20: Same hash (unchanged)
   - Block 21: Different hash (modified)
   - Block 22-25: Same hash (unchanged)
   
4. DELTA SYNC
   - Upload only block 21 (4 MB)
   - Reuse existing blocks 1-20, 22-25 from server
   - Total upload: 4 MB instead of 100 MB (25x savings)
\`\`\`

**Implementation**:
\`\`\`python
def chunk_file(file_path, chunk_size=4*1024*1024):
    chunks = []
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            chunk_hash = hashlib.sha256(data).hexdigest()
            chunks.append({
                'hash': chunk_hash,
                'data': data,
                'offset': f.tell() - len(data)
            })
    return chunks

def upload_file(file_id, chunks):
    # Query server for existing chunks
    existing_hashes = get_existing_chunks(file_id)
    
    # Upload only new/modified chunks
    for chunk in chunks:
        if chunk['hash'] not in existing_hashes:
            upload_chunk(chunk['hash'], chunk['data'])
    
    # Update file metadata with new chunk list
    update_file_metadata(file_id, [c['hash'] for c in chunks])
\`\`\`

---

## Step 4: Block-Level Deduplication

**Problem**: Multiple users upload identical files (e.g., popular PDF, video).

**Solution**: Store each unique block once, reference multiple times.

**Example**:
\`\`\`
User A uploads video.mp4 (100 MB, 25 blocks)
- Blocks stored: block_abc123, block_def456, ..., block_xyz789
- Total storage: 100 MB

User B uploads same video.mp4
- Dropbox calculates block hashes
- All 25 blocks already exist (same hashes)
- No upload needed (instant "upload")
- Create file entry referencing existing blocks
- Additional storage: 0 MB (metadata only)

Result: 2 users, 1 copy stored (50% savings)
\`\`\`

**Database Schema**:
\`\`\`sql
CREATE TABLE files (
    file_id BIGINT PRIMARY KEY,
    user_id BIGINT,
    file_path VARCHAR(500),
    file_size BIGINT,
    created_at TIMESTAMP,
    modified_at TIMESTAMP,
    is_deleted BOOLEAN
);

CREATE TABLE file_blocks (
    file_id BIGINT,
    block_index INT,
    block_hash VARCHAR(64),
    block_size INT,
    PRIMARY KEY (file_id, block_index),
    FOREIGN KEY (block_hash) REFERENCES blocks(block_hash)
);

CREATE TABLE blocks (
    block_hash VARCHAR(64) PRIMARY KEY,
    storage_path VARCHAR(500),  -- S3 key
    block_size INT,
    ref_count INT,  -- Number of files using this block
    created_at TIMESTAMP
);
\`\`\`

**Deduplication Savings**: 20-30% for typical workloads, 80%+ for organizations (shared documents).

---

## Step 5: File Synchronization

**Sync Flow**:
\`\`\`
1. FILE CHANGED ON DEVICE A
   - User edits document.txt on laptop
   - Watcher detects file modified (inotify on Linux, FSEvents on Mac)
   
2. UPLOAD TO CLOUD
   - Chunk file, calculate hashes
   - Upload modified blocks (delta sync)
   - Update metadata server: file_id X has new version
   
3. NOTIFY OTHER DEVICES
   - Metadata service queries: Which devices/users need this file?
   - Send push notification to Device B (desktop), Device C (phone)
   
4. DEVICES DOWNLOAD
   - Device B receives notification: "document.txt changed"
   - Download new blocks
   - Reconstruct file from blocks
   - Replace local copy
   
5. USER SEES UPDATED FILE
   - Device B shows notification: "document.txt updated"
   - File automatically synced
\`\`\`

**Notification System**:
- **WebSocket** for online devices (real-time, < 1 sec)
- **Long polling** as fallback
- **Push notifications** (APNS, FCM) for mobile

---

## Step 6: Conflict Resolution

**Scenario**: User edits file on laptop (offline), also edits on desktop. Both devices come online.

**Conflict Detection**:
\`\`\`
1. Laptop uploads version 2 (based on version 1)
2. Desktop uploads version 2' (also based on version 1)
3. Server detects conflict: Two children of version 1
\`\`\`

**Resolution Strategy**:
\`\`\`
1. LAST-WRITE-WINS (timestamp-based)
   - Keep version with latest timestamp
   - Problem: Clocks may be wrong, data loss
   
2. KEEP BOTH (Dropbox approach)
   - Rename one file: document.txt → document (laptop's conflicted copy).txt
   - User manually merges
   - No data loss
   
3. OPERATIONAL TRANSFORMATION (Google Docs)
   - Merge edits automatically
   - Complex, works only for structured data
\`\`\`

**Dropbox Choice**: Keep both, let user resolve (simple, no data loss).

---

## Step 7: Versioning

**Keep 30 days of file history**:

\`\`\`sql
CREATE TABLE file_versions (
    version_id BIGINT PRIMARY KEY,
    file_id BIGINT,
    version_number INT,
    block_list JSON,  -- [block_hash1, block_hash2, ...]
    created_at TIMESTAMP,
    created_by BIGINT,
    INDEX idx_file_versions (file_id, version_number)
);
\`\`\`

**Version Retention**:
- Keep all versions for 30 days
- After 30 days: Delete metadata, but keep blocks if referenced by other files
- Premium users: Extended history (1 year)

**Storage Implications**:
- 10 versions × 100 MB = 1 GB? No!
- With delta sync: Only changed blocks stored
- Typical: 10 versions = 1.2x original size (20% overhead)

---

## Step 8: Metadata Service

**Critical Component**: Fast queries for sync operations.

**Schema Design**:
\`\`\`sql
-- User's file tree
CREATE TABLE user_files (
    user_id BIGINT,
    file_path VARCHAR(500),
    file_id BIGINT,
    is_directory BOOLEAN,
    parent_id BIGINT,
    modified_at TIMESTAMP,
    PRIMARY KEY (user_id, file_path),
    INDEX idx_user_modified (user_id, modified_at)
);

-- Device sync state
CREATE TABLE device_sync (
    device_id VARCHAR(50),
    user_id BIGINT,
    last_sync_timestamp TIMESTAMP,
    cursor VARCHAR(100),  -- For pagination
    PRIMARY KEY (device_id)
);
\`\`\`

**Sync Query** (efficient):
\`\`\`sql
-- Get all files modified since last sync
SELECT file_id, file_path, modified_at, version_id
FROM user_files
WHERE user_id = ? 
  AND modified_at > last_sync_timestamp
ORDER BY modified_at
LIMIT 1000;
\`\`\`

**Sharding**: Shard by \`user_id\` (all user's files on same shard).

---

## Step 9: Large File Handling

**Challenge**: Upload 50 GB video file over unstable network.

**Solutions**:

**1. Resumable Uploads** (similar to YouTube):
\`\`\`
- Split 50 GB into 4 MB chunks = 12,800 chunks
- Upload chunks independently
- Track progress: Redis SET upload:session_123 "chunks_uploaded: 5000/12800"
- If network fails at chunk 5000, resume from 5001
\`\`\`

**2. Parallel Uploads**:
\`\`\`
- Upload 10 chunks concurrently
- Utilize full bandwidth
- 50 GB at 100 Mbps = 67 minutes (if serial)
- With 10 parallel: ~7 minutes
\`\`\`

**3. Compression** (optional):
\`\`\`
- Compress blocks before upload
- Text files: 70% reduction
- Media files: Already compressed (skip)
\`\`\`

---

## Step 10: Sharing & Permissions

**Share Link**:
\`\`\`
POST /api/v1/files/{file_id}/share

Response:
{
  "share_link": "https://dropbox.com/s/abc123xyz",
  "expires_at": "2024-12-31",
  "permissions": "view"  // or "edit"
}
\`\`\`

**Access Control**:
\`\`\`sql
CREATE TABLE file_shares (
    share_id VARCHAR(20) PRIMARY KEY,
    file_id BIGINT,
    owner_id BIGINT,
    permissions VARCHAR(10),  -- view, edit, admin
    expires_at TIMESTAMP,
    password_hash VARCHAR(64)  -- Optional password protection
);
\`\`\`

**Team Folders** (Business):
- Shared folder with multiple users
- Permissions: Admin, Editor, Viewer
- Centralized billing and management

---

## Step 11: Optimizations

**1. Intelligent Sync**:
- Don't sync OS temp files (.DS_Store, thumbs.db)
- Skip large binaries (node_modules, .git)
- User-configurable ignore rules (.dropboxignore)

**2. Bandwidth Throttling**:
- Limit sync speed to avoid congesting network
- User-configurable: "Limit download to 1 MB/s"

**3. LAN Sync**:
- If two devices on same LAN, sync directly (P2P)
- Faster than uploading to cloud and downloading back
- Detect via Bonjour/mDNS

**4. Selective Sync**:
- User chooses which folders to sync on each device
- Work laptop: Sync only "Work" folder (save disk space)
- Personal laptop: Sync everything

**5. Smart Caching**:
- Keep frequently accessed files cached on device
- Evict old, large files not accessed in 30 days
- Placeholder files (show in folder, download on access)

---

## Step 12: Database Sharding

**Shard by \`user_id\`**:
\\\`\\\`\\\`
user_id % NUM_SHARDS = shard_number

Benefits:
- All user's files on same shard (query locality)
- User's sync queries hit single shard (fast)
- Easy to add users (distribute across shards)
\\\`\\\`\\\`

**Cross-Shard Queries** (rare):
- Shared folder with users on different shards
- Use federated query or maintain replicated metadata

---

## Step 13: Storage Architecture

**Tiered Storage**:

**Hot Tier** (Recent files, < 30 days):
- S3 Standard
- Frequent access
- $0.023/GB/month

**Warm Tier** (30-90 days):
- S3 Infrequent Access
- Occasional access
- $0.0125/GB/month (45% savings)

**Cold Tier** (>90 days, rarely accessed):
- S3 Glacier
- Archival
- $0.004/GB/month (83% savings)

**Auto-Tiering**: S3 Intelligent-Tiering moves files automatically.

**Cost**:
- 100 PB at Standard: $2.3M/month
- With tiering (50% hot, 30% warm, 20% cold): $1.2M/month (48% savings)

---

## Step 14: Monitoring

**Key Metrics**:
1. **Sync Success Rate**: % of syncs completed successfully (target: 99.9%)
2. **Sync Latency**: Time from change to synced on other device (target: < 10 sec)
3. **Upload/Download Speed**: MB/s per user
4. **Conflict Rate**: % of files with conflicts (target: < 0.1%)
5. **Deduplication Ratio**: Storage saved (target: 20-30%)
6. **Block Reuse Rate**: % of blocks deduplicated

---

## Step 15: Security

**Encryption**:
- **In Transit**: TLS 1.3 (HTTPS)
- **At Rest**: AES-256 encryption in S3
- **Zero-Knowledge** (optional, not default):
  - Client-side encryption before upload
  - Dropbox cannot decrypt (user manages keys)
  - Trade-off: Cannot recover if password lost

**Access Control**:
- OAuth 2.0 for third-party apps
- 2FA for account security
- Audit logs for business users

---

## Interview Tips

**Clarify**:
1. Scale: 100M users or 700M?
2. File size limits: 2 GB or 50 GB?
3. Versioning required?
4. Conflict resolution strategy?

**Emphasize**:
1. **Chunking & Delta Sync**: Bandwidth efficiency
2. **Block-Level Deduplication**: Storage efficiency
3. **Metadata Service**: Fast sync queries
4. **Conflict Resolution**: Keep both strategy

**Common Mistakes**:
- Uploading entire files (no delta sync)
- Not handling conflicts (data loss)
- Ignoring deduplication (wasted storage)
- Synchronous sync (blocks user)

---

## Summary

**Core Components**:
1. **Sync Client**: Watches file changes, chunks files
2. **Metadata Service**: PostgreSQL, tracks files, versions, blocks
3. **Block Storage**: S3 with tiered storage
4. **Notification Service**: Push notifications via WebSocket
5. **Conflict Resolver**: Keep both conflicted copies

**Key Decisions**:
- ✅ Fixed 4 MB block size for chunking
- ✅ Delta sync (only changed blocks uploaded)
- ✅ SHA-256 for block deduplication
- ✅ Keep both strategy for conflicts
- ✅ 30-day version history
- ✅ Shard by user_id

**Capacity**:
- 700M users, 500M files/day uploaded
- 100 PB total storage
- 20-30% deduplication savings
- < 10 sec sync latency

Dropbox's design prioritizes **bandwidth and storage efficiency** through chunking and deduplication, while maintaining **simplicity** in conflict resolution and **reliability** in sync operations.`,
};
