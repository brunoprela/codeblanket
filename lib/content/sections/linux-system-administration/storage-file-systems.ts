/**
 * Storage & File Systems Section
 * Module: Linux System Administration & DevOps Foundations
 */

export const storageFileSystemsSection = {
  id: 'storage-file-systems',
  title: 'Storage & File Systems',
  content: `# Storage & File Systems

## Introduction

Production storage management is critical for data integrity, performance, and disaster recovery. This section covers AWS EBS volumes, file system management, RAID configurations, and production backup strategies used by DevOps engineers operating systems at scale.

## AWS EBS Volume Management

### EBS Volume Types and Use Cases

\`\`\`python
"""
AWS EBS Volume Types:
1. gp3 (General Purpose SSD) - Default choice
2. gp2 (General Purpose SSD) - Legacy
3. io2/io2 Block Express - Mission-critical, high IOPS
4. st1 (Throughput Optimized HDD) - Big data, log processing
5. sc1 (Cold HDD) - Infrequent access
"""

volume_types = {
    'gp3': {
        'use_case': 'Most workloads - databases, dev/test, boot volumes',
        'iops_range': '3,000 - 16,000',
        'throughput': '125 - 1,000 MB/s',
        'size_range': '1 GB - 16 TB',
        'cost': '~$0.08/GB-month',
        'baseline': '3,000 IOPS + 125 MB/s included',
        'best_for': 'Cost-effective performance'
    },
    'io2': {
        'use_case': 'Mission-critical databases (production MySQL, PostgreSQL)',
        'iops_range': '100 - 64,000 (256,000 with Block Express)',
        'throughput': 'Up to 4,000 MB/s',
        'size_range': '4 GB - 16 TB (64 TB with Block Express)',
        'cost': '~$0.125/GB-month + $0.065/IOPS',
        'durability': '99.999% (vs 99.8-99.9% for gp3)',
        'best_for': 'When data loss is unacceptable'
    },
    'st1': {
        'use_case': 'Big data, log processing, data warehouses',
        'throughput': '40 - 500 MB/s',
        'size_range': '125 GB - 16 TB',
        'cost': '~$0.045/GB-month',
        'best_for': 'Sequential read-heavy throughput workloads'
    }
}
\`\`\`

### Creating and Attaching EBS Volumes

\`\`\`bash
# Create a gp3 volume
aws ec2 create-volume \\
    --availability-zone us-east-1a \\
    --volume-type gp3 \\
    --size 100 \\
    --iops 16000 \\
    --throughput 1000 \\
    --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=production-data},{Key=Environment,Value=production}]'

# Output will include VolumeId
# {
#     "VolumeId": "vol-0123456789abcdef0",
#     "State": "creating",
#     ...
# }

# Wait for volume to be available
aws ec2 wait volume-available --volume-ids vol-0123456789abcdef0

# Attach volume to instance
aws ec2 attach-volume \\
    --volume-id vol-0123456789abcdef0 \\
    --instance-id i-0123456789abcdef0 \\
    --device /dev/sdf

# Wait for attachment
aws ec2 wait volume-in-use --volume-ids vol-0123456789abcdef0

# On the instance, verify the device
lsblk
# NAME    MAJ:MIN RM SIZE RO TYPE MOUNTPOINT
# xvda    202:0    0  30G  0 disk
# └─xvda1 202:1    0  30G  0 part /
# xvdf    202:80   0 100G  0 disk  <- New volume (may show as xvdf instead of sdf)

# Check device details
sudo file -s /dev/xvdf
# /dev/xvdf: data  (means it's empty, needs filesystem)

# If it shows a filesystem:
# /dev/xvdf: Linux rev 1.0 ext4 filesystem data...
\`\`\`

### Creating and Mounting File Systems

\`\`\`bash
# Create ext4 filesystem
sudo mkfs.ext4 -L data-volume /dev/xvdf

# Output:
# mke2fs 1.45.5 (07-Jan-2020)
# Creating filesystem with 26214400 4k blocks and 6553600 inodes
# Filesystem UUID: 12345678-1234-1234-1234-123456789abc
# Superblock backups stored on blocks: 
#     32768, 98304, 163840, 229376, 294912, 819200, 884736, 1605632, 2654208,
#     4096000, 7962624, 11239424, 20480000, 23887872

# Create mount point
sudo mkdir -p /data

# Mount the volume
sudo mount /dev/xvdf /data

# Verify mount
df -h /data
# Filesystem      Size  Used Avail Use% Mounted on
# /dev/xvdf       99G   60M   94G   1% /data

# Check mount options
mount | grep xvdf
# /dev/xvdf on /data type ext4 (rw,relatime,data=ordered)

# Make mount persistent across reboots
# Get UUID of the filesystem
sudo blkid /dev/xvdf
# /dev/xvdf: LABEL="data-volume" UUID="12345678-1234-1234-1234-123456789abc" TYPE="ext4"

# Add to /etc/fstab
echo "UUID=12345678-1234-1234-1234-123456789abc /data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab

# Test fstab entry (IMPORTANT!)
sudo mount -a
sudo df -h /data

# Explanation of fstab fields:
# UUID=...          - Device identifier (preferred over /dev/xvdf)
# /data             - Mount point
# ext4              - File system type
# defaults,nofail   - Mount options (nofail = don't fail boot if volume missing)
# 0                 - Dump (backup) - 0 = no backup
# 2                 - fsck order - 0 = no check, 1 = root fs, 2 = other fs
\`\`\`

### Resizing EBS Volumes

\`\`\`bash
# Expand volume from 100GB to 200GB
aws ec2 modify-volume \\
    --volume-id vol-0123456789abcdef0 \\
    --size 200

# Monitor modification status
aws ec2 describe-volumes-modifications \\
    --volume-ids vol-0123456789abcdef0

# Output:
# {
#     "VolumesModifications": [{
#         "VolumeId": "vol-0123456789abcdef0",
#         "ModificationState": "optimizing",  # or "completed"
#         "Progress": 100,
#         ...
#     }]
# }

# On the instance, verify new size is recognized
lsblk
# NAME    MAJ:MIN RM SIZE RO TYPE MOUNTPOINT
# xvdf    202:80   0 200G  0 disk /data  <- Now 200GB

# Check partition
sudo lsblk -f
# If partition shows old size, may need to resize partition first

# Resize the filesystem (ext4)
sudo resize2fs /dev/xvdf
# resize2fs 1.45.5 (07-Jan-2020)
# Filesystem at /dev/xvdf is mounted on /data; on-line resizing required
# old_desc_blocks = 13, new_desc_blocks = 25
# The filesystem on /dev/xvdf is now 52428800 (4k) blocks long.

# Verify new size
df -h /data
# Filesystem      Size  Used Avail Use% Mounted on
# /dev/xvdf       197G   60M  187G   1% /data

# For XFS filesystem (Amazon Linux 2023 default)
sudo xfs_growfs /data

# Note: XFS can only grow, not shrink!
\`\`\`

### EBS Snapshot Management

\`\`\`bash
# Create snapshot
aws ec2 create-snapshot \\
    --volume-id vol-0123456789abcdef0 \\
    --description "Production data backup $(date +%Y-%m-%d)" \\
    --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=prod-backup},{Key=Created,Value='$(date -u +%Y-%m-%dT%H:%M:%SZ)'}]'

# Output:
# {
#     "SnapshotId": "snap-0123456789abcdef0",
#     "State": "pending",
#     ...
# }

# Wait for snapshot completion
aws ec2 wait snapshot-completed --snapshot-ids snap-0123456789abcdef0

# List snapshots
aws ec2 describe-snapshots \\
    --owner-ids self \\
    --filters "Name=volume-id,Values=vol-0123456789abcdef0"

# Restore from snapshot (create new volume)
aws ec2 create-volume \\
    --snapshot-id snap-0123456789abcdef0 \\
    --availability-zone us-east-1a \\
    --volume-type gp3 \\
    --iops 16000

# Automated snapshot script
cat << 'SCRIPT' > /usr/local/bin/backup-ebs-volumes.sh
#!/bin/bash
set -euo pipefail

RETENTION_DAYS=30
TAG_KEY="AutoBackup"
TAG_VALUE="enabled"

# Find volumes tagged for backup
VOLUMES=$(aws ec2 describe-volumes \\
    --filters "Name=tag:$TAG_KEY,Values=$TAG_VALUE" \\
    --query 'Volumes[].VolumeId' \\
    --output text)

for volume_id in $VOLUMES; do
    echo "Creating snapshot for $volume_id"
    
    snapshot_id=$(aws ec2 create-snapshot \\
        --volume-id "$volume_id" \\
        --description "Automated backup $(date +%Y-%m-%d)" \\
        --tag-specifications "ResourceType=snapshot,Tags=[{Key=AutoBackup,Value=enabled},{Key=VolumeId,Value=$volume_id},{Key=Created,Value=$(date -u +%Y-%m-%dT%H:%M:%SZ)}]" \\
        --query 'SnapshotId' \\
        --output text)
    
    echo "  Created snapshot: $snapshot_id"
done

# Delete old snapshots
CUTOFF_DATE=$(date -u -d "$RETENTION_DAYS days ago" +%Y-%m-%d)

OLD_SNAPSHOTS=$(aws ec2 describe-snapshots \\
    --owner-ids self \\
    --filters "Name=tag:AutoBackup,Values=enabled" \\
    --query "Snapshots[?StartTime<\`$CUTOFF_DATE\`].SnapshotId" \\
    --output text)

for snapshot_id in $OLD_SNAPSHOTS; do
    echo "Deleting old snapshot: $snapshot_id"
    aws ec2 delete-snapshot --snapshot-id "$snapshot_id"
done

echo "Backup completed"
SCRIPT

chmod +x /usr/local/bin/backup-ebs-volumes.sh

# Schedule via cron (daily at 2 AM)
# 0 2 * * * /usr/local/bin/backup-ebs-volumes.sh
\`\`\`

## LVM (Logical Volume Manager)

### LVM Concepts

\`\`\`bash
# LVM Architecture:
# Physical Volumes (PV) -> Volume Groups (VG) -> Logical Volumes (LV)

# Example:
# /dev/xvdf (100GB) + /dev/xvdg (100GB) = PVs
#   ↓
# Volume Group "data-vg" (200GB total)
#   ↓
# LV1: "mysql-lv" (80GB)
# LV2: "logs-lv" (50GB)
# LV3: "backup-lv" (70GB)

# Benefits:
# - Dynamic resizing
# - Snapshots
# - Striping across multiple disks
# - Easy to add/remove disks
\`\`\`

### Creating LVM Setup

\`\`\`bash
# Install LVM tools
sudo yum install lvm2 -y

# Attach multiple EBS volumes
# Assume /dev/xvdf and /dev/xvdg are attached

# Create physical volumes
sudo pvcreate /dev/xvdf /dev/xvdg

# Verify
sudo pvdisplay
# Output shows:
# "/dev/xvdf" is a new physical volume of "100.00 GiB"
# "/dev/xvdg" is a new physical volume of "100.00 GiB"

# Create volume group
sudo vgcreate data-vg /dev/xvdf /dev/xvdg

# Verify
sudo vgdisplay data-vg
# --- Volume group ---
# VG Name               data-vg
# VG Size               199.99 GiB
# PE Size               4.00 MiB
# Total PE              51198
# Free  PE              51198

# Create logical volumes
sudo lvcreate -L 80G -n mysql-lv data-vg
sudo lvcreate -L 50G -n logs-lv data-vg
sudo lvcreate -L 69G -n backup-lv data-vg

# Verify
sudo lvdisplay
# Shows 3 logical volumes

# List all LVM components
sudo pvs  # Physical volumes
sudo vgs  # Volume groups
sudo lvs  # Logical volumes

# Create filesystems on logical volumes
sudo mkfs.ext4 /dev/data-vg/mysql-lv
sudo mkfs.ext4 /dev/data-vg/logs-lv
sudo mkfs.ext4 /dev/data-vg/backup-lv

# Create mount points and mount
sudo mkdir -p /var/lib/mysql /var/log/app /backup
sudo mount /dev/data-vg/mysql-lv /var/lib/mysql
sudo mount /dev/data-vg/logs-lv /var/log/app
sudo mount /dev/data-vg/backup-lv /backup

# Add to /etc/fstab
cat << 'EOF' | sudo tee -a /etc/fstab
/dev/data-vg/mysql-lv  /var/lib/mysql  ext4  defaults,nofail  0  2
/dev/data-vg/logs-lv   /var/log/app    ext4  defaults,nofail  0  2
/dev/data-vg/backup-lv /backup         ext4  defaults,nofail  0  2
EOF
\`\`\`

### Resizing LVM Volumes

\`\`\`bash
# Scenario: logs-lv is getting full, need to grow it

# Check current size
df -h /var/log/app
# Filesystem                  Size  Used Avail Use% Mounted on
# /dev/mapper/data--vg-logs--lv  49G   45G  2.1G  96% /var/log/app

# Check volume group free space
sudo vgdisplay data-vg | grep "Free"
# Free  PE / Size       1024 / 4.00 GiB

# Extend logical volume by 20GB
sudo lvextend -L +20G /dev/data-vg/logs-lv
# Size of logical volume data-vg/logs-lv changed from 50.00 GiB to 70.00 GiB

# Resize filesystem (ext4)
sudo resize2fs /dev/data-vg/logs-lv
# Filesystem at /dev/data-vg/logs-lv is mounted on /var/log/app; on-line resizing required
# The filesystem on /dev/data-vg/logs-lv is now 18350080 (4k) blocks long.

# Verify new size
df -h /var/log/app
# Filesystem                    Size  Used Avail Use% Mounted on
# /dev/mapper/data--vg-logs--lv  69G   45G   22G  68% /var/log/app

# Alternative: Extend to use all free space in VG
sudo lvextend -l +100%FREE /dev/data-vg/logs-lv
sudo resize2fs /dev/data-vg/logs-lv
\`\`\`

### LVM Snapshots

\`\`\`bash
# Create snapshot for backup
sudo lvcreate -L 10G -s -n mysql-snap /dev/data-vg/mysql-lv
# Logical volume "mysql-snap" created.

# Snapshot is initially small, grows as original LV changes

# Mount snapshot for backup
sudo mkdir /mnt/snapshot
sudo mount /dev/data-vg/mysql-snap /mnt/snapshot

# Perform backup from snapshot (consistent point-in-time)
sudo tar czf /backup/mysql-backup-$(date +%Y%m%d).tar.gz -C /mnt/snapshot .

# Unmount and remove snapshot
sudo umount /mnt/snapshot
sudo lvremove -f /dev/data-vg/mysql-snap

# Complete backup script with LVM snapshot
cat << 'SCRIPT' > /usr/local/bin/backup-mysql-lvm.sh
#!/bin/bash
set -euo pipefail

LV_PATH="/dev/data-vg/mysql-lv"
SNAP_NAME="mysql-snap"
SNAP_SIZE="10G"
MOUNT_POINT="/mnt/snapshot"
BACKUP_DIR="/backup"
RETENTION_DAYS=7

echo "Creating LVM snapshot..."
lvremove -f /dev/data-vg/$SNAP_NAME 2>/dev/null || true
lvcreate -L $SNAP_SIZE -s -n $SNAP_NAME $LV_PATH

echo "Mounting snapshot..."
mkdir -p $MOUNT_POINT
mount /dev/data-vg/$SNAP_NAME $MOUNT_POINT

echo "Creating backup..."
BACKUP_FILE="$BACKUP_DIR/mysql-backup-$(date +%Y%m%d-%H%M%S).tar.gz"
tar czf $BACKUP_FILE -C $MOUNT_POINT .

echo "Cleaning up..."
umount $MOUNT_POINT
lvremove -f /dev/data-vg/$SNAP_NAME

echo "Backup completed: $BACKUP_FILE"

# Remove old backups
find $BACKUP_DIR -name "mysql-backup-*.tar.gz" -mtime +$RETENTION_DAYS -delete
SCRIPT

chmod +x /usr/local/bin/backup-mysql-lvm.sh
\`\`\`

## RAID Configurations

### RAID Levels Overview

\`\`\`python
raid_levels = {
    'RAID 0': {
        'description': 'Striping (no redundancy)',
        'min_disks': 2,
        'capacity': 'Sum of all disks',
        'performance': 'High (parallel I/O)',
        'redundancy': 'None - any disk failure loses all data',
        'use_case': 'Temporary data, high performance needed'
    },
    'RAID 1': {
        'description': 'Mirroring',
        'min_disks': 2,
        'capacity': 'Size of smallest disk',
        'performance': 'Good read, normal write',
        'redundancy': 'Survives (n-1) disk failures',
        'use_case': 'Boot drives, critical small datasets'
    },
    'RAID 5': {
        'description': 'Striping with distributed parity',
        'min_disks': 3,
        'capacity': '(n-1) × disk_size',
        'performance': 'Good read, moderate write',
        'redundancy': 'Survives 1 disk failure',
        'use_case': 'File servers, general storage'
    },
    'RAID 6': {
        'description': 'Striping with double parity',
        'min_disks': 4,
        'capacity': '(n-2) × disk_size',
        'performance': 'Good read, slower write',
        'redundancy': 'Survives 2 disk failures',
        'use_case': 'Large arrays, critical data'
    },
    'RAID 10': {
        'description': 'RAID 1+0 (mirrored striping)',
        'min_disks': 4,
        'capacity': '(n/2) × disk_size',
        'performance': 'Excellent read/write',
        'redundancy': 'Survives 1 disk per mirror',
        'use_case': 'Databases, high I/O workloads'
    }
}

# AWS Recommendation:
# Generally avoid RAID on AWS!
# Use EBS features instead:
# - Need performance? Use gp3 with high IOPS
# - Need redundancy? EBS is already replicated
# - Need capacity? Use large single volume
# Exception: RAID 0 for extreme throughput (NVMe instance storage)
\`\`\`

### Creating Software RAID (if needed)

\`\`\`bash
# Install mdadm
sudo yum install mdadm -y

# Attach 4 EBS volumes (for RAID 10)
# Assume /dev/xvdf, /dev/xvdg, /dev/xvdh, /dev/xvdi

# Create RAID 10 array
sudo mdadm --create --verbose /dev/md0 \\
    --level=10 \\
    --raid-devices=4 \\
    /dev/xvdf /dev/xvdg /dev/xvdh /dev/xvdi

# Monitor creation progress
cat /proc/mdstat
# md0 : active raid10 xvdi[3] xvdh[2] xvdg[1] xvdf[0]
#       209584128 blocks super 1.2 512K chunks 2 near-copies [4/4] [UUUU]
#       [====>................]  resync = 23.4% (49123456/209584128)

# Wait for completion
sudo mdadm --wait /dev/md0

# Create filesystem
sudo mkfs.ext4 -F /dev/md0

# Mount
sudo mkdir -p /raid-data
sudo mount /dev/md0 /raid-data

# Save RAID configuration
sudo mdadm --detail --scan | sudo tee -a /etc/mdadm.conf

# Add to fstab
echo "/dev/md0 /raid-data ext4 defaults,nofail,_netdev 0 0" | sudo tee -a /etc/fstab

# Check RAID status
sudo mdadm --detail /dev/md0
# /dev/md0:
#         Version : 1.2
#   Creation Time : Thu Oct 28 10:00:00 2024
#      Raid Level : raid10
#      Array Size : 209584128 (199.89 GiB 214.60 GB)
#   Used Dev Size : 104792064 (99.94 GiB 107.30 GB)
#    Raid Devices : 4
#   Total Devices : 4
#          State : clean
\`\`\`

## File System Performance Tuning

### Mount Options for Performance

\`\`\`bash
# Default mount options
mount | grep xvdf
# /dev/xvdf on /data type ext4 (rw,relatime,data=ordered)

# Performance-optimized mount options
sudo mount -o remount,noatime,nodiratime,data=writeback /data

# Mount options explained:
# noatime: Don't update access time (improves performance)
# nodiratime: Don't update directory access time
# data=writeback: Don't guarantee data written before metadata (faster, slightly less safe)
# data=ordered: Metadata written after data (default, safer)
# data=journal: Both data and metadata journaled (safest, slowest)

# For database volumes (ext4)
cat << 'EOF' | sudo tee -a /etc/fstab
UUID=xxx /var/lib/mysql ext4 noatime,nodiratime,data=writeback,barrier=0,commit=60 0 2
EOF

# Options for database:
# barrier=0: Disable write barriers (only if battery-backed controller)
# commit=60: Commit metadata every 60 seconds instead of 5

# For XFS (Amazon Linux 2023 default)
cat << 'EOF' | sudo tee -a /etc/fstab
UUID=xxx /data xfs noatime,nodiratime,largeio,inode64,swalloc 0 2
EOF

# XFS options:
# largeio: Optimize for large sequential I/O
# inode64: Allow inodes throughout disk (better performance on large filesystems)
# swalloc: Use stripe-width for allocations (better with RAID)

# Verify mount options
mount | grep /data
\`\`\`

### I/O Scheduler Tuning

\`\`\`bash
# Check current I/O scheduler
cat /sys/block/xvdf/queue/scheduler
# [mq-deadline] kyber bfq none

# I/O Schedulers:
# mq-deadline: Default, good for most workloads
# kyber: Low-latency, good for fast SSDs
# bfq: Budget Fair Queueing, good for interactive systems
# none: No scheduling, for NVMe with many queues

# For EBS volumes (SSD), use none or kyber
echo none | sudo tee /sys/block/xvdf/queue/scheduler

# Make persistent via udev rule
cat << 'EOF' | sudo tee /etc/udev/rules.d/60-scheduler.rules
# Set I/O scheduler for EBS volumes
ACTION=="add|change", KERNEL=="xvd[a-z]", ATTR{queue/scheduler}="none"
EOF

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Adjust queue depth for better performance
echo 1024 | sudo tee /sys/block/xvdf/queue/nr_requests

# Read-ahead tuning (for sequential workloads)
sudo blockdev --setra 8192 /dev/xvdf  # 8192 × 512 bytes = 4MB read-ahead
\`\`\`

## Storage Performance Testing

### Using fio for I/O Benchmarking

\`\`\`bash
# Install fio
sudo yum install fio -y

# Random read test (IOPS)
sudo fio --name=random-read \\
    --ioengine=libaio \\
    --iodepth=32 \\
    --rw=randread \\
    --bs=4k \\
    --direct=1 \\
    --size=1G \\
    --numjobs=4 \\
    --runtime=60 \\
    --group_reporting \\
    --filename=/data/testfile

# Random write test (IOPS)
sudo fio --name=random-write \\
    --ioengine=libaio \\
    --iodepth=32 \\
    --rw=randwrite \\
    --bs=4k \\
    --direct=1 \\
    --size=1G \\
    --numjobs=4 \\
    --runtime=60 \\
    --group_reporting \\
    --filename=/data/testfile

# Sequential read test (Throughput)
sudo fio --name=sequential-read \\
    --ioengine=libaio \\
    --iodepth=32 \\
    --rw=read \\
    --bs=1M \\
    --direct=1 \\
    --size=1G \\
    --numjobs=1 \\
    --runtime=60 \\
    --group_reporting \\
    --filename=/data/testfile

# Mixed workload (70% read, 30% write)
sudo fio --name=mixed-rw \\
    --ioengine=libaio \\
    --iodepth=32 \\
    --rw=randrw \\
    --rwmixread=70 \\
    --bs=4k \\
    --direct=1 \\
    --size=1G \\
    --numjobs=4 \\
    --runtime=60 \\
    --group_reporting \\
    --filename=/data/testfile

# Interpret results:
# IOPS: Look for "iops=" in output
# Throughput: Look for "bw=" (bandwidth)
# Latency: Look for "lat=" values (avg, p95, p99)

# Example output interpretation:
# read: IOPS=15000, BW=58.6MiB/s (61.4MB/s)
# lat (usec): min=50, max=5000, avg=2133.45
# This volume provides:
# - 15,000 read IOPS
# - 58.6 MB/s read throughput
# - ~2ms average latency

# Compare with expected EBS performance:
# gp3: 16,000 IOPS, 1,000 MB/s
# io2: Up to provisioned IOPS
\`\`\`

## Best Practices

✅ **Use gp3 as default** - Best cost/performance ratio  
✅ **Enable EBS optimization** - Dedicated bandwidth  
✅ **Use UUIDs in fstab** - More reliable than device names  
✅ **Add nofail option** - Prevent boot failures  
✅ **Monitor disk metrics** - CloudWatch alarms  
✅ **Regular snapshots** - Automated backup strategy  
✅ **Test restores** - Verify backups work  
✅ **Use LVM for flexibility** - Easy resizing  
✅ **Avoid RAID on EBS** - Use EBS features instead  
✅ **Tune for workload** - Mount options and I/O scheduler

## Key Takeaways

1. **gp3 volumes** offer best value for most workloads
2. **io2 volumes** for mission-critical databases requiring highest durability
3. **LVM provides flexibility** for dynamic storage management
4. **EBS snapshots** are incremental and efficient
5. **Mount options** significantly impact performance
6. **I/O scheduler** matters for SSD performance
7. **Test storage performance** with fio before production
8. **Monitor storage metrics** proactively

## Next Steps

In the next section, we'll cover **Networking Basics**, learning TCP/IP fundamentals, DNS configuration, firewall management, and AWS VPC networking essentials for production deployments.`,
};
