/**
 * Backup & Disaster Recovery Section
 * Module: Linux System Administration & DevOps Foundations
 */

export const backupDisasterRecoverySection = {
  id: 'backup-disaster-recovery',
  title: 'Backup & Disaster Recovery',
  content: `# Backup & Disaster Recovery

## Introduction

Backup and disaster recovery (DR) planning is critical for business continuity. Understanding backup strategies (full, incremental, differential), Recovery Time Objective (RTO), Recovery Point Objective (RPO), and AWS backup services ensures you can recover from any failure scenario. This section covers comprehensive backup strategies, tools, and real-world DR implementations.

## Backup Fundamentals

### Backup Types

**Full Backup**
- Complete copy of all data
- Fastest recovery (single restore)
- Slowest backup, most storage
- Typically weekly or monthly

**Incremental Backup**
- Only data changed since last backup (full or incremental)
- Fastest backup, least storage
- Slower recovery (need full + all incrementals)
- Typically daily or hourly

**Differential Backup**
- Data changed since last full backup
- Moderate backup time and storage
- Faster recovery than incremental (full + last differential)
- Typically daily

### Backup Strategy Example

\`\`\`bash
# Weekly full + daily incremental strategy
# Sunday: Full backup
# Mon-Sat: Incremental backups

# Full backup
tar -czf /backup/full-$(date +%Y%m%d).tar.gz /var/www/html
# Size: 10GB, Time: 60min

# Incremental backup (changed files only)
tar -czf /backup/incr-$(date +%Y%m%d).tar.gz \\
  --listed-incremental=/backup/snapshot.file \\
  /var/www/html
# Size: 500MB, Time: 5min

# Recovery:
# 1. Restore full backup (Sunday)
# 2. Restore incremental backups (Mon-Sat in order)
\`\`\`

## RTO and RPO Planning

### Definitions

**RTO (Recovery Time Objective)**
- Maximum acceptable downtime
- How quickly must system be restored?
- Examples:
  - Tier 1 (Critical): RTO = 1 hour
  - Tier 2 (Important): RTO = 4 hours
  - Tier 3 (Standard): RTO = 24 hours

**RPO (Recovery Point Objective)**
- Maximum acceptable data loss
- How much data can you afford to lose?
- Examples:
  - Tier 1 (Critical): RPO = 5 minutes
  - Tier 2 (Important): RPO = 1 hour
  - Tier 3 (Standard): RPO = 24 hours

### Calculating RTO/RPO

\`\`\`
Business Impact Analysis:
1. Revenue per hour: $100,000
2. Acceptable loss: $50,000
3. RTO = $50,000 / $100,000/hour = 30 minutes

Data Loss:
1. Transactions per minute: 1,000
2. Acceptable loss: 5,000 transactions
3. RPO = 5,000 / 1,000/min = 5 minutes
\`\`\`

### Backup Strategy by Tier

\`\`\`bash
# Tier 1: RTO=1h, RPO=5min
- Full backup: Daily
- Incremental: Every 5 minutes
- Replication: Real-time (synchronous)
- DR site: Hot standby (active-active or active-passive)
- Test frequency: Monthly

# Tier 2: RTO=4h, RPO=1h
- Full backup: Weekly
- Incremental: Hourly
- Replication: Near real-time (asynchronous)
- DR site: Warm standby
- Test frequency: Quarterly

# Tier 3: RTO=24h, RPO=24h
- Full backup: Weekly
- Incremental: Daily
- Replication: Batch (daily)
- DR site: Cold standby (restore from backup)
- Test frequency: Annually
\`\`\`

## Backup Tools

### tar - Archive Files

\`\`\`bash
# Create archive
tar -czf backup.tar.gz /var/www/html
# -c: create
# -z: gzip compression
# -f: file name

# Extract archive
tar -xzf backup.tar.gz -C /restore/location

# List contents
tar -tzf backup.tar.gz

# Incremental backup with snapshot
tar -czf backup-full.tar.gz \\
  --listed-incremental=snapshot.file \\
  /var/www/html

# Subsequent incremental
tar -czf backup-incr-1.tar.gz \\
  --listed-incremental=snapshot.file \\
  /var/www/html

# Restore incremental
tar -xzf backup-full.tar.gz -C /restore
tar -xzf backup-incr-1.tar.gz -C /restore

# Verify archive
tar -tzf backup.tar.gz | head -10

# Exclude files
tar -czf backup.tar.gz \\
  --exclude='*.log' \\
  --exclude='cache/*' \\
  /var/www/html
\`\`\`

### rsync - Incremental File Sync

\`\`\`bash
# Basic sync
rsync -avz /source/ /backup/
# -a: archive mode (preserves permissions, timestamps)
# -v: verbose
# -z: compression

# Remote sync
rsync -avz /source/ user@remote:/backup/

# Incremental backup with hardlinks
rsync -avz --link-dest=/backup/previous/ \\
  /source/ /backup/current/
# Only changed files stored, rest are hardlinks

# Exclude patterns
rsync -avz \\
  --exclude='*.log' \\
  --exclude='cache/' \\
  --exclude='.git/' \\
  /source/ /backup/

# Delete files not in source
rsync -avz --delete /source/ /backup/

# Bandwidth limit
rsync -avz --bwlimit=10000 /source/ /backup/
# 10000 KB/s limit

# Dry run (test without changes)
rsync -avzn /source/ /backup/

# Show progress
rsync -avz --progress /source/ /backup/

# Partial transfer resume
rsync -avz --partial --progress /source/ /backup/
\`\`\`

### dd - Disk Cloning

\`\`\`bash
# Clone entire disk
dd if=/dev/sda of=/dev/sdb bs=4M status=progress
# if: input file (source disk)
# of: output file (destination disk)
# bs: block size
# status=progress: show progress

# Create disk image
dd if=/dev/sda of=/backup/disk.img bs=4M status=progress

# Restore from image
dd if=/backup/disk.img of=/dev/sda bs=4M status=progress

# Clone partition
dd if=/dev/sda1 of=/backup/partition.img bs=4M

# Compressed backup
dd if=/dev/sda bs=4M | gzip > /backup/disk.img.gz

# Restore compressed
gunzip -c /backup/disk.img.gz | dd of=/dev/sda bs=4M

# Verify disk copy
dd if=/dev/sda | md5sum
dd if=/dev/sdb | md5sum
# Checksums should match

# Wipe disk securely
dd if=/dev/urandom of=/dev/sda bs=4M status=progress
\`\`\`

## Database Backups

### PostgreSQL Backups

\`\`\`bash
# Logical backup (pg_dump)
pg_dump -U postgres -h localhost -d mydb > /backup/mydb-$(date +%Y%m%d).sql

# Compressed backup
pg_dump -U postgres -d mydb | gzip > /backup/mydb-$(date +%Y%m%d).sql.gz

# Custom format (parallel restore)
pg_dump -U postgres -d mydb -Fc -f /backup/mydb.dump

# All databases
pg_dumpall -U postgres > /backup/all-dbs.sql

# Restore
psql -U postgres -d mydb < /backup/mydb.sql
# OR
pg_restore -U postgres -d mydb /backup/mydb.dump

# Parallel dump (faster for large databases)
pg_dump -U postgres -d mydb -Fd -j 4 -f /backup/mydb-dir/

# Restore parallel
pg_restore -U postgres -d mydb -j 4 /backup/mydb-dir/

# Continuous archiving (WAL archiving)
# In postgresql.conf:
wal_level = replica
archive_mode = on
archive_command = 'cp %p /backup/wal/%f'

# Point-in-time recovery (PITR)
# 1. Take base backup
# 2. Archive WAL files continuously
# 3. Restore to specific timestamp
\`\`\`

### MySQL Backups

\`\`\`bash
# Logical backup (mysqldump)
mysqldump -u root -p mydb > /backup/mydb-$(date +%Y%m%d).sql

# All databases
mysqldump -u root -p --all-databases > /backup/all-dbs.sql

# Compressed backup
mysqldump -u root -p mydb | gzip > /backup/mydb.sql.gz

# Consistent backup (lock tables)
mysqldump -u root -p --single-transaction mydb > /backup/mydb.sql

# Restore
mysql -u root -p mydb < /backup/mydb.sql

# Binary backup (Percona XtraBackup)
xtrabackup --backup --target-dir=/backup/mysql-backup
xtrabackup --prepare --target-dir=/backup/mysql-backup
xtrabackup --copy-back --target-dir=/backup/mysql-backup

# Point-in-time recovery
# Enable binary logging in my.cnf:
log-bin = /var/log/mysql/mysql-bin.log
\`\`\`

## AWS Backup Services

### EBS Snapshots

\`\`\`bash
# Create snapshot (AWS CLI)
aws ec2 create-snapshot \\
  --volume-id vol-1234567890abcdef0 \\
  --description "Daily backup $(date +%Y-%m-%d)" \\
  --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=daily-backup}]'

# List snapshots
aws ec2 describe-snapshots --owner-ids self

# Create volume from snapshot
aws ec2 create-volume \\
  --snapshot-id snap-1234567890abcdef0 \\
  --availability-zone us-east-1a \\
  --volume-type gp3

# Copy snapshot to another region
aws ec2 copy-snapshot \\
  --source-region us-east-1 \\
  --source-snapshot-id snap-1234567890abcdef0 \\
  --destination-region us-west-2 \\
  --description "DR copy"

# Delete old snapshots
aws ec2 describe-snapshots \\
  --owner-ids self \\
  --query 'Snapshots[?StartTime<=\`$(date -d "30 days ago" --iso-8601)\`].SnapshotId' \\
  --output text | xargs -n1 aws ec2 delete-snapshot --snapshot-id
\`\`\`

### Terraform: Automated EBS Snapshots

\`\`\`terraform
# DLM (Data Lifecycle Manager) policy
resource "aws_dlm_lifecycle_policy" "ebs_backup" {
  description        = "EBS snapshot policy"
  execution_role_arn = aws_iam_role.dlm.arn
  state              = "ENABLED"

  policy_details {
    resource_types = ["VOLUME"]

    schedule {
      name = "Daily snapshots"

      create_rule {
        interval      = 24
        interval_unit = "HOURS"
        times         = ["03:00"]
      }

      retain_rule {
        count = 7  # Keep 7 daily snapshots
      }

      tags_to_add = {
        SnapshotType = "automated"
      }

      copy_tags = true
    }

    target_tags = {
      Backup = "true"
    }
  }
}

# IAM role for DLM
resource "aws_iam_role" "dlm" {
  name = "dlm-lifecycle-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "dlm.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "dlm" {
  role       = aws_iam_role.dlm.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSDataLifecycleManagerServiceRole"
}

# Tag volumes for backup
resource "aws_ebs_volume" "data" {
  availability_zone = "us-east-1a"
  size              = 100
  type              = "gp3"

  tags = {
    Name   = "data-volume"
    Backup = "true"  # DLM will backup this volume
  }
}
\`\`\`

### AWS Backup Service

\`\`\`terraform
# AWS Backup vault
resource "aws_backup_vault" "main" {
  name = "production-backup-vault"
}

# Backup plan
resource "aws_backup_plan" "main" {
  name = "production-backup-plan"

  rule {
    rule_name         = "daily_backup"
    target_vault_name = aws_backup_vault.main.name
    schedule          = "cron(0 3 * * ? *)"  # 3 AM daily

    lifecycle {
      delete_after = 30  # Delete after 30 days
      cold_storage_after = 7  # Move to cold storage after 7 days
    }

    copy_action {
      destination_vault_arn = aws_backup_vault.dr.arn

      lifecycle {
        delete_after = 90
      }
    }
  }

  rule {
    rule_name         = "weekly_backup"
    target_vault_name = aws_backup_vault.main.name
    schedule          = "cron(0 4 ? * 1 *)"  # 4 AM every Monday

    lifecycle {
      delete_after = 90
    }
  }
}

# Backup selection
resource "aws_backup_selection" "main" {
  name         = "production-resources"
  plan_id      = aws_backup_plan.main.id
  iam_role_arn = aws_iam_role.backup.arn

  selection_tag {
    type  = "STRINGEQUALS"
    key   = "Backup"
    value = "true"
  }

  resources = [
    "arn:aws:ec2:*:*:volume/*",
    "arn:aws:rds:*:*:db:*",
    "arn:aws:dynamodb:*:*:table/*"
  ]
}

# IAM role for AWS Backup
resource "aws_iam_role" "backup" {
  name = "aws-backup-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "backup.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "backup" {
  role       = aws_iam_role.backup.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBackupServiceRolePolicyForBackup"
}

resource "aws_iam_role_policy_attachment" "restore" {
  role       = aws_iam_role.backup.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBackupServiceRolePolicyForRestores"
}
\`\`\`

### RDS Automated Backups

\`\`\`terraform
resource "aws_db_instance" "main" {
  identifier     = "production-db"
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = "db.t3.large"

  # Automated backups
  backup_retention_period = 7     # Keep 7 days
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"
  
  # Enable automated backups
  skip_final_snapshot = false
  final_snapshot_identifier = "production-db-final-snapshot"

  # Point-in-time recovery
  enabled_cloudwatch_logs_exports = ["postgresql"]
}

# Manual snapshot
resource "null_resource" "manual_snapshot" {
  triggers = {
    always_run = timestamp()
  }

  provisioner "local-exec" {
    command = <<-EOF
      aws rds create-db-snapshot \
        --db-instance-identifier production-db \
        --db-snapshot-identifier manual-$(date +%Y%m%d-%H%M%S)
    EOF
  }
}
\`\`\`

## Backup Automation Scripts

### Comprehensive Backup Script

\`\`\`bash
#!/bin/bash
# Production backup script

set -euo pipefail

# Configuration
BACKUP_DIR="/backup"
S3_BUCKET="s3://my-backup-bucket"
RETENTION_DAYS=30
LOG_FILE="/var/log/backup.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    # Send alert (SNS, email, Slack)
    aws sns publish --topic-arn arn:aws:sns:us-east-1:123456789012:backup-alerts \\
      --message "Backup failed: $1"
    exit 1
}

# Check prerequisites
command -v aws >/dev/null 2>&1 || error_exit "AWS CLI not installed"
command -v pg_dump >/dev/null 2>&1 || error_exit "PostgreSQL client not installed"

# Create backup directory
mkdir -p "$BACKUP_DIR"
DATE=$(date +%Y%m%d-%H%M%S)

log "Starting backup process"

# Database backup
log "Backing up PostgreSQL database"
pg_dump -U postgres -h localhost mydb | gzip > "$BACKUP_DIR/mydb-$DATE.sql.gz" || \
  error_exit "Database backup failed"

# Application files backup
log "Backing up application files"
tar -czf "$BACKUP_DIR/app-$DATE.tar.gz" /var/www/html || \
  error_exit "Application backup failed"

# Configuration backup
log "Backing up configuration files"
tar -czf "$BACKUP_DIR/config-$DATE.tar.gz" /etc/myapp || \
  error_exit "Config backup failed"

# Upload to S3
log "Uploading backups to S3"
aws s3 sync "$BACKUP_DIR" "$S3_BUCKET/backups/$(date +%Y/%m/%d)/" \\
  --storage-class STANDARD_IA || \
  error_exit "S3 upload failed"

# Verify uploads
log "Verifying S3 uploads"
for file in "$BACKUP_DIR"/*-"$DATE"*; do
    filename=$(basename "$file")
    aws s3 ls "$S3_BUCKET/backups/$(date +%Y/%m/%d)/$filename" >/dev/null 2>&1 || \
      error_exit "Verification failed for $filename"
done

# Cleanup old local backups
log "Cleaning up old local backups"
find "$BACKUP_DIR" -name "*.gz" -mtime +$RETENTION_DAYS -delete

# Cleanup old S3 backups (lifecycle policy is better)
log "Cleaning up old S3 backups"
aws s3 ls "$S3_BUCKET/backups/" --recursive | \
  while read -r line; do
    createDate=$(echo "$line" | awk '{print $1" "$2}')
    createDate=$(date -d "$createDate" +%s)
    olderThan=$(date -d "$RETENTION_DAYS days ago" +%s)
    if [[ $createDate -lt $olderThan ]]; then
      fileName=$(echo "$line" | awk '{print $4}')
      aws s3 rm "s3://$S3_BUCKET/$fileName"
    fi
  done

log "Backup completed successfully"

# Send success notification
aws sns publish --topic-arn arn:aws:sns:us-east-1:123456789012:backup-alerts \\
  --subject "Backup Success" \\
  --message "Backup completed successfully at $(date)"

exit 0
\`\`\`

### Backup Verification Script

\`\`\`bash
#!/bin/bash
# Verify backups integrity

BACKUP_FILE="$1"
VERIFY_DIR="/tmp/verify-$$"

mkdir -p "$VERIFY_DIR"

# Verify archive integrity
if [[ "$BACKUP_FILE" == *.tar.gz ]]; then
    echo "Verifying tar.gz archive..."
    tar -tzf "$BACKUP_FILE" >/dev/null 2>&1
    if [[ $? -eq 0 ]]; then
        echo "✓ Archive is valid"
    else
        echo "✗ Archive is corrupted"
        exit 1
    fi
fi

# Verify SQL backup
if [[ "$BACKUP_FILE" == *.sql.gz ]]; then
    echo "Verifying SQL backup..."
    gunzip -t "$BACKUP_FILE"
    if [[ $? -eq 0 ]]; then
        echo "✓ SQL backup is valid"
        # Optional: Try to restore to test database
        # gunzip -c "$BACKUP_FILE" | psql -U postgres -h localhost testdb
    else
        echo "✗ SQL backup is corrupted"
        exit 1
    fi
fi

# Calculate checksum
echo "Calculating checksum..."
md5sum "$BACKUP_FILE" > "$BACKUP_FILE.md5"
echo "✓ Checksum saved to $BACKUP_FILE.md5"

rm -rf "$VERIFY_DIR"
echo "Verification complete"
\`\`\`

## Disaster Recovery Testing

### DR Test Plan

\`\`\`bash
#!/bin/bash
# DR Test Script

# 1. Document current state
echo "=== DR Test Started: $(date) ===" | tee dr-test-$(date +%Y%m%d).log

# 2. Stop application
systemctl stop myapp

# 3. Backup current data
tar -czf /tmp/pre-dr-backup.tar.gz /var/www/html

# 4. Simulate disaster (delete data)
rm -rf /var/www/html/*

# 5. Restore from backup
tar -xzf /backup/app-latest.tar.gz -C /

# 6. Restore database
gunzip -c /backup/mydb-latest.sql.gz | psql -U postgres mydb

# 7. Start application
systemctl start myapp

# 8. Verify functionality
curl -f http://localhost/health || exit 1

# 9. Run smoke tests
./smoke-tests.sh

# 10. Document results
echo "=== DR Test Completed: $(date) ===" | tee -a dr-test-$(date +%Y%m%d).log

# 11. Restore original if test
# (In production, you'd keep the restored state)
\`\`\`

## Best Practices

✅ **Follow 3-2-1 rule**: 3 copies, 2 different media, 1 offsite  
✅ **Test restores regularly**: Backups are useless if they don't restore  
✅ **Automate backups**: Eliminate human error  
✅ **Encrypt backups**: Protect sensitive data  
✅ **Monitor backup success**: Alert on failures  
✅ **Document procedures**: DR runbooks for teams  
✅ **Use cross-region**: Protect against regional failures  
✅ **Verify backup integrity**: Check checksums  
✅ **Plan for RTO/RPO**: Know your recovery objectives  
✅ **Lifecycle policies**: Auto-delete old backups to save costs

## Next Steps

In the next section, we'll cover **Time Synchronization & NTP**, including time drift problems, chrony, and AWS time sync service.`,
};
