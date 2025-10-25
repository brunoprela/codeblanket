/**
 * Dropbox Architecture Section
 */

export const dropboxarchitectureSection = {
  id: 'dropbox-architecture',
  title: 'Dropbox Architecture',
  content: `Dropbox is a file hosting service that offers cloud storage, file synchronization, and collaboration. With over 700 million users and 2.5 billion files synced daily, Dropbox\'s architecture must handle massive scale file storage, efficient synchronization across devices, and conflict resolution. This section explores the technical systems that power Dropbox.

## Overview

Dropbox's scale and unique challenges:
- **700+ million users** worldwide
- **2.5 billion files** synced daily
- **600+ petabytes** of data stored
- **Real-time sync**: Changes propagate within seconds
- **Deduplication**: Save storage via block-level deduplication
- **Conflict resolution**: Handle simultaneous edits

### Key Challenges

1. **Synchronization**: Keep files in sync across multiple devices
2. **Deduplication**: Same file uploaded by multiple users
3. **Bandwidth**: Minimize network usage during sync
4. **Conflict resolution**: Handle simultaneous edits gracefully
5. **Scale**: Petabytes of storage, millions of concurrent syncs

---

## High-Level Architecture

\`\`\`
┌────────────────────────┐
│   Dropbox Client       │ (Desktop/Mobile App)
│   - File Watcher       │
│   - Sync Engine        │
└──────────┬─────────────┘
           │ HTTPS
           ▼
┌────────────────────────┐
│   API Gateway          │
│   (Authentication,     │
│    Rate Limiting)      │
└──────────┬─────────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌─────────┐ ┌─────────────┐
│ Metadata│ │ Block       │
│ Service │ │ Storage     │
│         │ │ Service     │
└─────────┘ └──────┬──────┘
     │             │
     ▼             ▼
┌─────────┐ ┌─────────────┐
│ Database│ │ Magic Pocket│
│ (MySQL) │ │ (Custom     │
│         │ │  Storage)   │
└─────────┘ └─────────────┘
\`\`\`

---

## Core Components

### 1. File Synchronization

Dropbox's sync engine keeps files in sync across devices in real-time.

**How Sync Works**:

**1. File Monitoring**:
- Client watches Dropbox folder for changes
- Uses OS-level APIs: inotify (Linux), FSEvents (macOS), ReadDirectoryChangesW (Windows)
- Detect: file created, modified, deleted, renamed

**2. Chunking**:
- Divide file into 4 MB blocks
- Hash each block (SHA-256)
- Send metadata to server (file name, block hashes)

**3. Server-Side Processing**:
- Check which blocks already exist (deduplication)
- Request only new blocks from client
- Store blocks in storage

**4. Propagation**:
- Notify other devices (push notification)
- Other devices download new blocks
- Reconstruct file from blocks

**Example**:
\`\`\`
User edits document.txt on Laptop:
1. File watcher detects change
2. Divide into blocks: [B1, B2, B3]
3. Hash blocks: [hash1, hash2, hash3]
4. Send metadata to server
5. Server checks: hash1 exists, hash2 new, hash3 exists
6. Client uploads only B2 (saves bandwidth!)
7. Server notifies Phone and Desktop
8. Phone/Desktop download B2, reconstruct file

Result: Only changed block uploaded/downloaded, not entire file!
\`\`\`

---

### 2. Block-Level Deduplication

Dropbox saves storage by deduplicating identical blocks across users.

**Deduplication Strategy**:

**1. Content-Addressed Storage**:
- Blocks identified by hash (SHA-256)
- Same content → Same hash → Stored once

**Example**:
\`\`\`
User A uploads photo.jpg:
- Divided into blocks: [B1, B2, B3]
- Hashes: [abc123, def456, ghi789]
- Stored in storage: abc123 → data1, def456 → data2, ghi789 → data3

User B uploads same photo.jpg:
- Divided into blocks: [B1, B2, B3]
- Hashes: [abc123, def456, ghi789]
- Server: All blocks already exist!
- No upload needed (instant "upload")
- Storage: photo stored once, referenced by both users
\`\`\`

**Benefits**:
- **Instant uploads**: If file already exists, no upload needed
- **Storage savings**: Common files (e.g., installers, libraries) stored once
- **Bandwidth savings**: Only new blocks uploaded

**Challenges**:

**1. Security**:
- Problem: Attackers could guess hashes, access files
- Solution: Files encrypted, access controlled by metadata service

**2. Partial Duplication**:
- Problem: Files mostly similar (e.g., edited document)
- Solution: Block-level deduplication catches common blocks

**Scale**:
- Deduplication saves 30-50% of storage
- For 600 PB, that's 180-300 PB saved!

---

### 3. Magic Pocket (Custom Storage System)

In 2016, Dropbox migrated from AWS S3 to **Magic Pocket**, their custom storage system.

**Why Build Custom Storage?**

- **Cost**: Cheaper than S3 at Dropbox\'s scale
- **Control**: Optimize for Dropbox's access patterns
- **Performance**: Lower latency, higher throughput

**Magic Pocket Architecture**:

**1. Storage Clusters**:
- Multiple datacenters worldwide
- Each cluster: Thousands of hard drives
- Petabyte-scale per cluster

**2. Data Layout**:
- Blocks stored on disk
- Metadata tracks: block_hash → location (disk, cluster)

**3. Replication**:
- Each block replicated 3x (different disks, different racks)
- Survives disk/rack failures

**4. Erasure Coding** (for older data):
- Instead of 3x replication (300% overhead), use erasure coding (140% overhead)
- Trade: Slightly slower reads, significant storage savings
- Used for "cold" data (not accessed frequently)

**Hardware**:
- **Custom server design**: Optimized for Dropbox workload
- **Dense storage**: Many drives per server (high capacity)
- **SMR drives**: Shingled Magnetic Recording (higher density, lower cost)

**Cost Savings**:
- Dropbox saves ~$75M/year compared to AWS
- Paid off infrastructure investment in ~2 years

**Challenges**:

**1. Operational Complexity**:
- Must manage hardware (failures, upgrades)
- Build tooling for monitoring, alerting, repair

**2. Scale**:
- Petabyte-scale migrations
- Must ensure zero data loss

**3. Global Distribution**:
- Multiple datacenters for redundancy and latency

---

### 4. Metadata Management

Metadata tracks file structure, sharing, versions.

**Metadata**:
- File name, path, size, modified time
- Owner, collaborators, permissions
- Block hashes (pointers to content)
- Version history

**Storage**: **MySQL** (sharded)

**Data Model**:
\`\`\`sql
Table: files
- file_id (primary key)
- user_id (owner)
- name
- path
- size
- modified_time
- is_deleted

Table: file_blocks
- file_id
- block_index (0, 1, 2, ...)
- block_hash (pointer to Magic Pocket)

Table: versions
- file_id
- version_id
- timestamp
- block_hashes (list of hashes for this version)
\`\`\`

**Sharding**:
- Shard by user_id (all user's files on same shard)
- Avoids cross-shard queries for most operations

**Challenges**:

**1. Consistency**:
- Ensure metadata matches storage (no orphaned blocks)
- Use transactions (MySQL ACID)

**2. Hot Users**:
- Users with millions of files (power users)
- Solution: Shard hot users separately, optimize queries

**3. Versioning**:
- Store previous versions (rewind feature)
- Challenge: Storage overhead
- Solution: Only store changed blocks (deduplication across versions)

---

### 5. Conflict Resolution

When two devices edit the same file simultaneously, conflicts arise.

**Conflict Scenarios**:

**Scenario 1: Sequential Edits** (No Conflict)
\`\`\`
Laptop edits file at 10:00 AM
Phone syncs at 10:05 AM → Gets latest version
No conflict!
\`\`\`

**Scenario 2: Simultaneous Edits** (Conflict!)
\`\`\`
Laptop edits file at 10:00 AM (offline)
Phone edits file at 10:00 AM (offline)
Both sync at 10:10 AM
Conflict detected!
\`\`\`

**Dropbox\'s Resolution Strategy**:

**1. Last-Writer-Wins** (for metadata like rename, delete)
\`\`\`
Laptop renames file.txt → report.txt at 10:00
Phone renames file.txt → document.txt at 10:01
Server receives both:
- Laptop's rename timestamp: 10:00
- Phone's rename timestamp: 10:01
Phone wins (later timestamp) → file named document.txt
\`\`\`

**2. Conflicted Copy** (for content edits)
\`\`\`
Laptop edits file.txt at 10:00 AM
Phone edits file.txt at 10:00 AM
Server receives both:
- Keep Phone\'s version as file.txt (arbitrary choice)
- Create file (conflicted copy, Laptop, 2024-10-24).txt
User resolves manually (merge or delete conflicted copy)
\`\`\`

**Why Not Automatic Merge?**
- Files are binary blobs (Dropbox doesn't understand content)
- Can't merge images, videos, zip files
- For text files, could use diff, but complexity high

**Version History**:
- All versions stored (for 30 days on free plan, indefinitely on paid)
- Users can rewind to previous version
- Helps recover from conflicts, accidental deletes

---

### 6. Sharing and Collaboration

Dropbox allows sharing files/folders with other users.

**Sharing Types**:

**1. Shared Folders**:
- Owner invites users to folder
- All users see same files, real-time sync
- Permissions: View-only, Edit

**2. Shared Links**:
- Generate public URL for file/folder
- Anyone with link can view/download
- Optionally: Password-protected, expiration date

**Architecture**:

**Data Model**:
\`\`\`sql
Table: shares
- share_id
- folder_id
- owner_id
- invitees (list of user_ids)
- permissions (view/edit)

Table: shared_links
- link_id
- file_id
- token (unique URL token)
- password_hash (optional)
- expiration_time (optional)
- view_count
\`\`\`

**Shared Folder Sync**:
\`\`\`
User A shares folder with User B:
1. A invites B (sends email)
2. B accepts invitation
3. Folder appears in B's Dropbox
4. Changes by A or B sync to both
5. Metadata service tracks: folder_id → [user_A, user_B]
6. When A edits file, notify B's devices
7. B's devices download changes
\`\`\`

**Permissions**:
- **Owner**: Full control (add/remove users, delete folder)
- **Editor**: Can add/edit/delete files
- **Viewer**: Can view/download, can't edit

**Challenges**:

**1. Access Control**:
- Ensure user has permission before serving file
- Check on every request (permissions can change)

**2. Quota**:
- Shared files count toward whose quota?
- Dropbox: Counts toward owner's quota

**3. Nested Shares**:
- Share folder within shared folder
- Complexity: Permission inheritance

---

### 7. Dropbox Paper (Collaboration)

Dropbox Paper is a collaborative document editor (like Google Docs).

**Real-Time Collaboration**:

**Operational Transform (OT)**:
- Algorithm for concurrent editing
- Ensures consistency across users

**How OT Works**:
\`\`\`
Document: "Hello"

User A types "World" at position 5 → "Hello World"
User B deletes "Hello" at position 0 → " "

Both operations sent to server:
- Operation A: Insert "World" at 5
- Operation B: Delete 0-5

Server applies OT:
1. Apply B first: " " (delete "Hello")
2. Transform A based on B: Insert "World" at 0 (position adjusted)
3. Result: "World"

Both users converge to "World" (consistency!)
\`\`\`

**Tech Stack**:
- WebSocket for real-time updates
- Server coordinates operations (central authority)
- Offline editing: Operations queued, applied when online

---

## Technology Stack

### Storage

- **Magic Pocket**: Custom storage system (replaces AWS S3)
- **MySQL**: Metadata (sharded by user_id)
- **Edgestore**: Photo storage system (also custom-built)

### Caching

- **Memcached**: Metadata cache (user profiles, file metadata)
- **Redis**: Session storage, rate limiting

### Services

- **Python**: Backend services (originally monolith, now microservices)
- **Go**: Performance-critical services (sync engine)
- **Rust**: Low-level systems (storage, networking)

### Infrastructure

- **Custom datacenters**: Magic Pocket runs in Dropbox\'s datacenters
- **Kubernetes**: Container orchestration
- **gRPC**: Inter-service communication

---

## Migration from AWS to Magic Pocket

Dropbox's migration from AWS to custom infrastructure (2016) is one of the largest cloud exits.

**Why Migrate?**

- **Cost**: Save $75M/year
- **Control**: Optimize for Dropbox workload
- **Performance**: Lower latency, higher throughput

**Migration Challenges**:

**1. Scale**:
- 600 PB of data to migrate
- Months-long process

**2. Zero Downtime**:
- Must serve 700M users during migration
- Cannot afford data loss

**3. Rollback Plan**:
- If migration fails, must revert to AWS

**Migration Strategy**:

**1. Dual-Write Phase**:
\`\`\`
New uploads:
- Write to Magic Pocket (primary)
- Write to AWS S3 (backup)

Reads:
- Try Magic Pocket first
- Fallback to S3 if not found
\`\`\`

**2. Background Migration**:
- Copy existing data from S3 to Magic Pocket
- Prioritize frequently accessed data
- Low priority for cold data

**3. Validation**:
- Verify data integrity (checksums match)
- Test with subset of users (canary deployment)

**4. Cutover**:
- Gradually shift traffic to Magic Pocket
- Monitor error rates, latency
- Complete cutover after 100% validation

**5. Decommission S3**:
- After several months of stable operation
- Delete data from S3

**Outcome**:
- Migration completed successfully (2016)
- No data loss, minimal user impact
- Savings: $75M/year

---

## Key Lessons

### 1. Block-Level Deduplication Saves Storage

Deduplicating at block level (not file level) catches partial duplicates, saves 30-50% storage.

### 2. Custom Infrastructure at Scale

At Dropbox's scale, building custom storage (Magic Pocket) is cost-effective vs cloud providers.

### 3. Conflict Resolution is Hard

Automatic merging is infeasible for binary files. Create conflicted copies, let users resolve.

### 4. Metadata and Content Separation

Store metadata (filenames, permissions) separately from content (blocks). Enables fast queries and efficient storage.

### 5. Gradual Migration

Migrating 600 PB requires dual-write phase, background migration, extensive validation. Cannot be rushed.

---

## Interview Tips

**Q: How would you design Dropbox\'s sync engine?**

A: Use block-level deduplication. Divide files into 4 MB blocks, hash each block (SHA-256). On file change: (1) Client detects via file watcher. (2) Divide file into blocks, compute hashes. (3) Send metadata to server (filename, block hashes). (4) Server checks which blocks exist. (5) Client uploads only new blocks. (6) Server stores blocks in content-addressed storage (hash → data). (7) Server updates metadata (file → list of block hashes). (8) Server notifies other devices via push notification. (9) Devices download new blocks, reconstruct file. Benefits: Bandwidth savings (only changed blocks uploaded), instant uploads (if file already exists), storage savings (blocks deduplicated across users).

**Q: How does Dropbox handle conflicts when two devices edit the same file simultaneously?**

A: Detect conflicts using timestamps and version IDs. When two devices upload different versions of the same file: (1) Server receives both edits. (2) Check timestamps: if simultaneous (within seconds), it's a conflict. (3) Resolution strategy: Keep one version as "file.txt" (arbitrary choice, e.g., last to reach server). (4) Create "file (conflicted copy, DeviceName, Date).txt" for the other version. (5) Sync both versions to all devices. (6) User resolves manually (merge or delete conflicted copy). For metadata operations (rename, delete), use last-writer-wins based on timestamp. Store all versions for 30 days (free plan) or indefinitely (paid) so users can rewind.

**Q: Why did Dropbox migrate from AWS to custom infrastructure?**

A: At Dropbox's scale (600+ PB), building custom storage is cost-effective. Savings: $75M/year compared to AWS S3. Benefits: (1) Cost: Commodity hardware + custom software cheaper than cloud. (2) Control: Optimize for Dropbox\'s access patterns (large files, high read/write ratio). (3) Performance: Lower latency (direct hardware control), higher throughput. (4) Innovation: Can experiment with new storage tech (SMR drives, erasure coding). Trade-offs: (1) Operational complexity (manage hardware, failures, upgrades). (2) Upfront cost (datacenter build-out). (3) Vendor lock-in (to own infrastructure). Migration strategy: Dual-write phase (write to both S3 and Magic Pocket), background migration, extensive validation, gradual cutover.

---

## Summary

Dropbox's architecture demonstrates building a file sync and storage platform at massive scale:

**Key Takeaways**:

1. **Block-level deduplication**: 4 MB blocks, content-addressed storage, instant uploads for duplicates
2. **Magic Pocket**: Custom storage system, saves $75M/year vs AWS
3. **Metadata separation**: MySQL for metadata, Magic Pocket for content
4. **Conflict resolution**: Conflicted copies for content edits, last-writer-wins for metadata
5. **Version history**: All versions stored, users can rewind
6. **Shared folders**: Real-time sync across multiple users, permission management
7. **Gradual migration**: Dual-write, background migration, validation for AWS → Magic Pocket

Dropbox's success came from smart deduplication, custom infrastructure at scale, and robust conflict handling for distributed file sync.
`,
};
