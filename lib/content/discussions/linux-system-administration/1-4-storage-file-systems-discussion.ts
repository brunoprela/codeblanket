export const storageFileSystemsDiscussion = [
  {
    id: 1,
    question:
      "Your production MySQL database on EC2 is running on a gp2 EBS volume that's experiencing performance degradation. Burst credits are at 0% and IOPS are throttled. The database is 500GB and growing 50GB/month. Design a complete migration strategy to resolve the performance issues, including volume type selection, migration approach, testing plan, and rollback procedures. Include specific AWS commands and downtime minimization strategies.",
    answer: `**Complete Migration Strategy:**

**Analysis:** gp2 volume (500GB) provides 1,500 baseline IOPS (3 IOPS/GB). With burst credits exhausted, database is throttled to baseline, causing performance issues.

**Solution:** Migrate to gp3 volume with 16,000 IOPS and 1,000 MB/s throughput.

**Migration Steps:**

\`\`\`bash
# 1. Create EBS snapshot
aws ec2 create-snapshot --volume-id vol-old123 --description "Pre-migration backup"

# 2. Create new gp3 volume from snapshot
aws ec2 create-volume --snapshot-id snap-abc123 --volume-type gp3 --iops 16000 --throughput 1000 --availability-zone us-east-1a

# 3. Stop database, detach old volume, attach new volume
sudo systemctl stop mysql
aws ec2 detach-volume --volume-id vol-old123
aws ec2 attach-volume --volume-id vol-new456 --instance-id i-xxx --device /dev/sdf

# 4. Start database and verify
sudo systemctl start mysql

# 5. Monitor performance with CloudWatch
\`\`\`

**Benefits:** 10x IOPS improvement (1,500 → 16,000), no burst credits needed, better cost efficiency.`,
  },
  {
    id: 2,
    question:
      'Design a production storage architecture for a high-traffic web application that requires: 1) OS and application binaries (30GB), 2) MySQL database (200GB, growing 20GB/month), 3) Application logs (500GB retention), 4) User uploads (1TB, growing 100GB/month). Include EBS volume types, mount strategies, backup plans, and cost optimization.',
    answer: `**Storage Architecture:**

**1. Root Volume (OS + Binaries):**
- Type: gp3, 50GB, 3,000 IOPS
- Mount: / (root)
- Backup: AMI + snapshots weekly
- Cost: ~$4/month

**2. Database Volume:**
- Type: io2, 300GB, 10,000 IOPS provisioned
- Mount: /var/lib/mysql
- Backup: Daily snapshots + replication
- Cost: ~$100/month

**3. Logs Volume:**
- Type: st1, 1TB (throughput optimized HDD)
- Mount: /var/log
- Backup: Ship to S3 + CloudWatch
- Retention: 30 days local, 1 year S3
- Cost: ~$45/month

**4. User Uploads:**
- Type: S3 (not EBS!) with CloudFront
- Benefits: Unlimited scale, 99.999999999% durability
- Lifecycle: Transition to Glacier after 90 days
- Cost: ~$25/month (vs $80 for EBS)

**Total Cost:** ~$174/month with optimal performance and durability.`,
  },
  {
    id: 3,
    question:
      'You need to perform a consistent backup of a 500GB MySQL database running on EC2 with LVM. The backup must be point-in-time consistent (no corruption) and complete in under 5 minutes to minimize application impact. Design the complete backup strategy using LVM snapshots, including all commands, timing considerations, and verification procedures.',
    answer: `**LVM Snapshot Backup Strategy:**

\`\`\`bash
#!/bin/bash
# Production MySQL backup with LVM snapshot

# 1. Flush tables and lock (ensures consistency)
mysql -e "FLUSH TABLES WITH READ LOCK; SYSTEM /usr/local/bin/create-snapshot.sh; UNLOCK TABLES;"

# create-snapshot.sh:
#!/bin/bash
# Takes <5 seconds due to COW nature
lvcreate -L 50G -s -n mysql-snap /dev/vg0/mysql-lv

# 2. Database is locked for <5 seconds during snapshot creation
# Snapshot is point-in-time consistent

# 3. Mount snapshot and backup
mount /dev/vg0/mysql-snap /mnt/snapshot
tar czf /backup/mysql-\$(date +%Y%m%d-%H%M%S).tar.gz -C /mnt/snapshot .

# 4. Verify backup integrity
tar tzf /backup/mysql-*.tar.gz > /dev/null && echo "Backup valid"

# 5. Cleanup
umount /mnt/snapshot
lvremove -f /dev/vg0/mysql-snap

# 6. Upload to S3
aws s3 cp /backup/mysql-*.tar.gz s3://prod-backups/mysql/
\`\`\`

**Benefits:** <5 second database lock, consistent backup, minimal impact, can backup 500GB in background.`,
  },
];
