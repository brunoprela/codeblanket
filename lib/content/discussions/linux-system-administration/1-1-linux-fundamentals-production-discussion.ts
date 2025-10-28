export const linuxFundamentalsProductionDiscussion = [
  {
    id: 1,
    question:
      "You're operating a production web application on AWS EC2 that suddenly starts failing with 'No space left on device' errors. When you SSH into the server and run `df -h`, you see that the root volume is only 45% full with 20GB free. What's happening, and how would you diagnose and fix this issue? Walk through your complete troubleshooting process, including the commands you'd run and how you'd prevent this from happening again.",
    answer: `## Comprehensive Answer:

This is a classic **inode exhaustion** problem - one of the most common "gotchas" in Linux production systems. The filesystem has run out of inodes (file metadata structures) even though plenty of disk space remains.

### Understanding the Problem

\`\`\`python
"""
Linux filesystems separate data storage into two components:
1. Data blocks: Store actual file contents
2. Inodes: Store file metadata (permissions, timestamps, location)

Each file/directory needs one inode, regardless of size!
A 1KB file and a 1GB file each use exactly one inode.
"""

# What's happening:
filesystem = {
    'data_blocks': {
        'total': '100GB',
        'used': '45GB',    # 45% used - plenty of space!
        'free': '55GB'
    },
    'inodes': {
        'total': 6_000_000,
        'used': 6_000_000,  # 100% exhausted!
        'free': 0           # No inodes left = can't create files
    }
}

# Common causes:
causes = [
    'Millions of small cache files (Redis, PHP sessions)',
    'Log rotation creating many small log files',
    'Application creating temporary files in /tmp',
    'Email spool with many small messages',
    'NPM/node_modules with thousands of tiny files'
]
\`\`\`

### Step-by-Step Diagnosis

**Step 1: Confirm inode exhaustion**

\`\`\`bash
# Check inode usage across all filesystems
df -i

# Expected output:
# Filesystem      Inodes  IUsed    IFree IUse% Mounted on
# /dev/xvda1     6000000 6000000       0  100% /
# devtmpfs        500000     500  499500    1% /dev
# tmpfs           500000      50  499950    1% /dev/shm

# Confirm: root filesystem at 100% inode usage while disk space is 45%
\`\`\`

**Step 2: Find the culprit directory**

\`\`\`bash
# Method 1: Count files in top-level directories
for dir in /*; do
    echo -n "$dir: "
    find "$dir" -xdev -type f | wc -l
done

# Output might show:
# /bin: 100
# /etc: 2500
# /var: 5800000  <-- FOUND IT!
# /usr: 150000
# /home: 5000

# Method 2: More detailed breakdown
for dir in /var/*; do
    echo -n "$dir: "
    find "$dir" -xdev -type f | wc -l
done

# Output:
# /var/log: 5000
# /var/lib: 5750000  <-- Drill down further
# /var/cache: 50000

# Method 3: Find exact directory
du --inodes -d 5 /var | sort -n | tail -20

# Output shows:
# 5800000  /var/lib/php/sessions
# ^^^^^^^  This is the problem!
\`\`\`

**Step 3: Investigate the problematic directory**

\`\`\`bash
# Check number of files
ls /var/lib/php/sessions | wc -l
# 5,800,000 files!

# Check file ages
ls -lt /var/lib/php/sessions | head -20
# sess_xyz123... Oct 10 10:00
# sess_abc456... Oct 10 09:58
# sess_def789... Oct 10 09:55

ls -lt /var/lib/php/sessions | tail -20
# sess_old123... Jan 15 2022  <-- Files from 2+ years ago!
# sess_old456... Jan 12 2022
# sess_old789... Jan 08 2022

# Check total size
du -sh /var/lib/php/sessions
# 2.8GB  <-- Only 2.8GB but using 5.8M inodes!

# Each session file is tiny
ls -lh /var/lib/php/sessions | head -5
# -rw------- 1 www-data www-data 512 Oct 10 10:00 sess_xyz...
#                                ^^^^ 512 bytes per file
\`\`\`

**Root Cause Analysis**:

\`\`\`python
# The problem:
issue = {
    'application': 'PHP web application',
    'session_storage': 'File-based in /var/lib/php/sessions',
    'session_lifetime': 'Never cleaned up',
    'daily_new_sessions': 8000,
    'days_running': 725,  # ~2 years
    'total_files': 8000 * 725,  # = 5.8 million
    'disk_usage': '512 bytes * 5.8M = 2.8GB',  # Minimal space
    'inode_usage': '1 inode * 5.8M = 5.8M inodes',  # EXHAUSTED!
}

# Why it happened:
explanation = """
PHP's default session garbage collection wasn't running properly.
Each visitor gets a new session file. Old sessions accumulate.
Each tiny 512-byte file consumes a full inode.
Over 2 years: millions of files = inode exhaustion.
"""
\`\`\`

### Immediate Fix (Emergency)

\`\`\`bash
# WARNING: This will log out all active users!
# Only do in emergency or maintenance window

# Step 1: Stop application (prevent new session files)
systemctl stop nginx
systemctl stop php-fpm

# Step 2: Backup a sample (for investigation)
mkdir /root/session-backup
cp /var/lib/php/sessions/sess_* /root/session-backup/ 2>/dev/null | head -100

# Step 3: Delete old session files (older than 7 days)
find /var/lib/php/sessions -type f -mtime +7 -delete

# If that's not enough, delete all sessions:
rm -rf /var/lib/php/sessions/*

# Step 4: Verify inode recovery
df -i /
# Filesystem      Inodes  IUsed    IFree IUse% Mounted on
# /dev/xvda1     6000000  200000 5800000    4% /
# ^^^^^^^ Back to normal!

# Step 5: Restart application
systemctl start php-fpm
systemctl start nginx

# Step 6: Monitor
watch -n 5 'df -i / && ls /var/lib/php/sessions | wc -l'
\`\`\`

### Proper Fix (Prevent Recurrence)

**Solution 1: Enable PHP garbage collection**

\`\`\`bash
# Edit PHP configuration
vi /etc/php/8.1/fpm/php.ini

# Configure session garbage collection
session.gc_probability = 1
session.gc_divisor = 100
# ^^ 1% chance of GC running on each request

session.gc_maxlifetime = 86400
# ^^ Delete sessions older than 24 hours (86400 seconds)

# Restart PHP-FPM
systemctl restart php-fpm

# Verify GC is working:
# Wait 24 hours, then check:
find /var/lib/php/sessions -type f -mtime +1 | wc -l
# Should be 0 or very low
\`\`\`

**Solution 2: Add cron job (more reliable)**

\`\`\`bash
# Create cleanup script
cat > /usr/local/bin/cleanup-php-sessions.sh << 'EOF'
#!/bin/bash
# Clean up PHP sessions older than 24 hours

SESSION_DIR="/var/lib/php/sessions"
MAX_AGE_DAYS=1

# Count before cleanup
BEFORE=$(find "$SESSION_DIR" -type f | wc -l)

# Delete old sessions
find "$SESSION_DIR" -type f -mtime +$MAX_AGE_DAYS -delete

# Count after cleanup
AFTER=$(find "$SESSION_DIR" -type f | wc -l)
DELETED=$((BEFORE - AFTER))

# Log results
logger -t php-session-cleanup "Cleaned $DELETED session files"
echo "$(date): Cleaned $DELETED files ($BEFORE -> $AFTER)" >> /var/log/session-cleanup.log

# Alert if session count is suspiciously high
if [ $AFTER -gt 100000 ]; then
    logger -p user.warning -t php-session-cleanup "WARNING: High session count: $AFTER"
fi
EOF

chmod +x /usr/local/bin/cleanup-php-sessions.sh

# Add to cron (run every hour)
crontab -e
# Add line:
0 * * * * /usr/local/bin/cleanup-php-sessions.sh
\`\`\`

**Solution 3: Switch to alternative session storage (best practice)**

\`\`\`php
<?php
// Option 1: Redis (recommended for production)
ini_set('session.save_handler', 'redis');
ini_set('session.save_path', 'tcp://localhost:6379?database=0');

// Option 2: Memcached
ini_set('session.save_handler', 'memcached');
ini_set('session.save_path', 'localhost:11211');

// Option 3: Database
ini_set('session.save_handler', 'mysql');
ini_set('session.save_path', 'mysql:host=localhost;dbname=sessions');

// Benefits:
// - No inode issues
// - Automatic TTL/expiry
// - Better performance
// - Easier scaling
?>
\`\`\`

**Solution 4: CloudWatch monitoring (preventive)**

\`\`\`bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
rpm -U ./amazon-cloudwatch-agent.rpm

# Configure custom metric for inode usage
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'EOF'
{
  "metrics": {
    "namespace": "CustomSystem",
    "metrics_collected": {
      "disk": {
        "measurement": [
          {
            "name": "used_percent",
            "rename": "DiskUsedPercent"
          },
          {
            "name": "inodes_free",
            "rename": "InodesFree"
          },
          {
            "name": "inodes_used_percent",
            "rename": "InodesUsedPercent"
          }
        ],
        "metrics_collection_interval": 60,
        "resources": [
          "*"
        ]
      }
    }
  }
}
EOF

# Start agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \\
  -a fetch-config \\
  -m ec2 \\
  -s \\
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

# Create CloudWatch alarm
aws cloudwatch put-metric-alarm \\
  --alarm-name high-inode-usage \\
  --alarm-description "Alert when inode usage > 80%" \\
  --metric-name InodesUsedPercent \\
  --namespace CustomSystem \\
  --statistic Average \\
  --period 300 \\
  --threshold 80 \\
  --comparison-operator GreaterThanThreshold \\
  --evaluation-periods 2 \\
  --alarm-actions arn:aws:sns:us-east-1:123456789:ops-alerts
\`\`\`

### Long-Term Architecture Improvements

\`\`\`python
# Modern session management architecture:

architecture = {
    'session_storage': 'ElastiCache Redis',
    'benefits': [
        'No filesystem impact',
        'Automatic expiry (TTL)',
        'High performance (in-memory)',
        'Easy horizontal scaling',
        'Multi-AZ failover'
    ],
    
    'terraform_example': '''
    resource "aws_elasticache_cluster" "sessions" {
      cluster_id           = "session-cache"
      engine               = "redis"
      node_type            = "cache.t3.micro"
      num_cache_nodes      = 1
      parameter_group_name = "default.redis7"
      port                 = 6379
      
      # Enable automatic failover
      automatic_failover_enabled = true
      multi_az_enabled           = true
    }
    '''
}

# Cost: ~$13/month for cache.t3.micro
# Benefit: Eliminates entire class of filesystem issues
\`\`\`

### Prevention Checklist

\`\`\`bash
# 1. Monitor inode usage
df -i

# 2. Set up alerts (CloudWatch)
# Alert at 70%, 80%, 90% inode usage

# 3. Regular cleanup cron jobs
# Daily: find /var/lib/php/sessions -mtime +1 -delete
# Weekly: find /tmp -mtime +7 -delete
# Weekly: find /var/log -name "*.gz" -mtime +30 -delete

# 4. Use appropriate storage for different data types
# Sessions: Redis/Memcached (not filesystem)
# Logs: CloudWatch Logs or centralized logging
# Temp files: tmpfs with size limits
# Cache: Redis or application-level cache

# 5. Filesystem sizing
# When creating filesystem, allocate enough inodes
mkfs.ext4 -N 20000000 /dev/xvdf  # 20M inodes

# 6. Documentation
# Document what uses inodes in your environment
# Include in runbooks
\`\`\`

### Summary

**The Issue**: Inode exhaustion caused by millions of tiny PHP session files accumulating over 2 years.

**Immediate Fix**: Delete old session files to recover inodes.

**Root Cause**: PHP garbage collection not configured properly.

**Long-Term Solution**:
1. Enable PHP GC with proper gc_maxlifetime
2. Add cron-based cleanup as backup
3. **Best practice**: Switch to Redis/Memcached for sessions
4. Set up CloudWatch monitoring and alerts
5. Document and include in incident response playbook

**Key Learning**: In Linux, disk space and inodes are separate resources. Monitor both. Production systems should always have:
- Disk space monitoring
- Inode usage monitoring  
- Automated cleanup for temporary files
- Proper session management (not filesystem-based)

This type of issue is extremely common in production web applications and is a critical skill for DevOps engineers to diagnose and prevent.
`,
  },
  {
    id: 2,
    question:
      "Your company is deploying a new high-traffic API service on EC2 instances running Amazon Linux 2023. The application team reports that during load testing, they're getting 'Too many open files' errors even though the application is only handling 200 concurrent connections. They've already increased the file descriptor limits in their application code. As the DevOps engineer, walk through how you would diagnose this issue and configure the system properly for production. Include kernel parameters, systemd service configuration, and monitoring setup.",
    answer: `## Comprehensive Answer:

This is a classic production issue involving Linux file descriptor limits - one of the most common gotchas when scaling applications. The "Too many open files" error indicates the process has hit its open file descriptor limit.

### Understanding File Descriptors

\`\`\`python
"""
In Linux, everything is a file:
- Regular files
- Network sockets
- Pipes
- Devices
- etc.

Each open 'file' consumes a file descriptor (fd).
A high-traffic API service with 200 connections might need:
- 200 socket fds (client connections)
- 50 database connection pool fds
- 20 file fds (log files, config files)
- 100 fd overhead (libraries, etc.)
= 370+ file descriptors needed
"""

# Default limits are often too low:
default_limits = {
    'per_process_soft': 1024,    # ulimit -Sn
    'per_process_hard': 4096,    # ulimit -Hn
    'system_wide': 65536,        # /proc/sys/fs/file-nr
}

# Production needs:
production_needs = {
    'high_traffic_api': 65536,
    'database_server': 65536,
    'web_server': 65536,
}
\`\`\`

### Step-by-Step Diagnosis

**Step 1: Reproduce and confirm the issue**

\`\`\`bash
# SSH to the EC2 instance
ssh ec2-user@<instance-ip>

# Check application logs
journalctl -u api-service.service -n 100
# or
tail -f /var/log/application/api.log

# Look for:
# "Too many open files"
# "java.io.IOException: Too many open files"
# "OSError: [Errno 24] Too many open files"
\`\`\`

**Step 2: Check current limits**

\`\`\`bash
# Find the application process
ps aux | grep api-service
# api-user  12345  ...  /usr/bin/java -jar api.jar

# Check limits for running process
cat /proc/12345/limits
# Limit                     Soft Limit Hard Limit Units
# Max open files            1024       4096       files
# ^^^^ FOUND IT! Only 1024 open files allowed

# Check system-wide limits
sysctl fs.file-max
# fs.file-max = 65536  (System-wide is OK)

cat /proc/sys/fs/file-nr
# 5123    0    65536
# ^^^^    ^    ^^^^^
# used    free  max
# System-wide usage is low, so this is a per-process limit issue

# Check soft/hard limits for current user
su - api-user
ulimit -Sn  # Soft limit
# 1024
ulimit -Hn  # Hard limit
# 4096

# ^^ These are too low for production!
\`\`\`

**Step 3: Check what's consuming file descriptors**

\`\`\`bash
# Count open files for the process
lsof -p 12345 | wc -l
# 1021  <-- Very close to 1024 limit!

# See what types of files are open
lsof -p 12345 | awk '{print $5}' | sort | uniq -c | sort -rn
# 850 IPv4    (network sockets)
#  50 REG     (regular files)
#  20 unix    (unix domain sockets)
# 100 other

# List actual open connections
lsof -p 12345 -i -a
# Shows all network connections

# Check socket states
ss -p | grep "pid=12345"
# Count ESTABLISHED connections:
ss -p | grep "pid=12345" | grep ESTABLISHED | wc -l
# 200  (matches application's 200 concurrent connections)

# So we need: 200 connections + 150 overhead = ~400 minimum
# But production needs headroom: recommend 65536
\`\`\`

### Complete Fix: Multi-Layer Configuration

**Layer 1: System-Wide Kernel Limits**

\`\`\`bash
# Configure system-wide file descriptor limit
sudo vi /etc/sysctl.d/99-file-limits.conf

# Add:
# System-wide maximum number of open files
fs.file-max = 2097152

# Also increase other related limits
fs.nr_open = 2097152

# Apply immediately
sudo sysctl -p /etc/sysctl.d/99-file-limits.conf

# Verify
sysctl fs.file-max
# fs.file-max = 2097152
\`\`\`

**Layer 2: User-Level Limits (PAM)**

\`\`\`bash
# Configure per-user limits
sudo vi /etc/security/limits.conf

# Add at the end:
# <domain>  <type>  <item>  <value>
*           soft    nofile  65536
*           hard    nofile  65536
api-user    soft    nofile  65536
api-user    hard    nofile  65536
root        soft    nofile  65536
root        hard    nofile  65536

# Explanation:
# * = applies to all users
# api-user = specific to application user
# soft = default limit (can be increased up to hard limit)
# hard = maximum limit (only root can increase)
# nofile = number of open files

# Note: Requires re-login to take effect!
\`\`\`

**Layer 3: systemd Service Configuration** (Most Important!)

\`\`\`bash
# Create or edit systemd service file
sudo vi /etc/systemd/system/api-service.service

[Unit]
Description=High-Traffic API Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=api-user
Group=api-user
WorkingDirectory=/opt/api-service

# FILE DESCRIPTOR LIMITS (Critical!)
LimitNOFILE=65536
# ^^ This is the key setting for systemd services

# Other resource limits for production
LimitNPROC=4096          # Max number of processes
LimitCORE=infinity       # Core dump size (for debugging)
LimitMEMLOCK=64M         # Locked memory

# Additional hardening (optional but recommended)
PrivateTmp=true          # Separate /tmp for service
NoNewPrivileges=true     # Prevent privilege escalation
ProtectSystem=strict     # Read-only /usr, /boot
ProtectHome=true         # No access to /home
ReadWritePaths=/opt/api-service/logs /opt/api-service/data

# Environment variables
Environment="JAVA_OPTS=-Xmx2g -Xms2g"
Environment="API_ENV=production"

# Startup command
ExecStart=/usr/bin/java -jar /opt/api-service/api.jar

# Restart policy
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target

# Reload systemd after changes
sudo systemctl daemon-reload

# Restart service
sudo systemctl restart api-service.service

# Verify limits are applied
sudo systemctl show api-service.service | grep LimitNOFILE
# LimitNOFILE=65536
# LimitNOFILESoft=65536

# Check running process
API_PID=$(systemctl show -p MainPID api-service.service | cut -d= -f2)
cat /proc/$API_PID/limits | grep "Max open files"
# Max open files      65536    65536    files
#                     ^^^^^    ^^^^^
#                     soft     hard
# ✓ Success! Limits are now 65536
\`\`\`

**Layer 4: Application-Level Configuration**

Some applications have their own file descriptor configuration:

\`\`\`python
# Python (using gunicorn)
# /opt/api-service/gunicorn_config.py
import multiprocessing
import resource

# Set maximum file descriptors
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
print(f"File descriptor limits: soft={soft}, hard={hard}")

if soft < 65536:
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
        print("Increased file descriptor limit to 65536")
    except Exception as e:
        print(f"Warning: Could not increase limit: {e}")

# Gunicorn configuration
bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "gevent"  # Async worker for many connections
worker_connections = 1000
max_requests = 10000
max_requests_jitter = 1000
timeout = 30
keepalive = 5

# Java (JVM options)
# /etc/systemd/system/api-service.service
# ExecStart=/usr/bin/java \\
#   -Xmx2g -Xms2g \\
#   -XX:MaxDirectMemorySize=1g \\
#   -jar /opt/api-service/api.jar

# Node.js (pm2 ecosystem)
# /opt/api-service/ecosystem.config.js
module.exports = {
  apps: [{
    name: 'api-service',
    script: './app.js',
    instances: 'max',
    exec_mode: 'cluster',
    max_memory_restart: '2G',
    env: {
      NODE_ENV: 'production',
      UV_THREADPOOL_SIZE: 128  // Increase libuv thread pool
    }
  }]
}
\`\`\`

### Verification and Testing

\`\`\`bash
# 1. Verify service started successfully
sudo systemctl status api-service.service
# Active: active (running)

# 2. Verify limits applied to process
API_PID=$(systemctl show -p MainPID api-service.service | cut -d= -f2)
cat /proc/$API_PID/limits | grep "Max open files"
# Max open files      65536    65536    files

# 3. Check actual file descriptor usage
lsof -p $API_PID | wc -l
# 450  (well below 65536 limit)

# 4. Run load test
# Use Apache Bench, wrk, or similar
wrk -t 12 -c 400 -d 30s http://localhost:8000/api/test
# Should handle 400 concurrent connections without errors

# 5. Monitor during load test
watch -n 1 "lsof -p $API_PID | wc -l"
# Observe file descriptor usage increase during test
# Should stay well below 65536

# 6. Check for errors
journalctl -u api-service.service -f
# Should NOT see "Too many open files" anymore
\`\`\`

### Production Monitoring Setup

\`\`\`bash
# 1. Create monitoring script
sudo vi /usr/local/bin/monitor-file-descriptors.sh

#!/bin/bash
# Monitor file descriptor usage for api-service

SERVICE="api-service"
PID=$(systemctl show -p MainPID $SERVICE.service | cut -d= -f2)

if [ "$PID" = "0" ] || [ -z "$PID" ]; then
    echo "Service $SERVICE is not running"
    exit 1
fi

# Get limits
SOFT_LIMIT=$(cat /proc/$PID/limits | grep "Max open files" | awk '{print $4}')
HARD_LIMIT=$(cat /proc/$PID/limits | grep "Max open files" | awk '{print $5}')

# Count open file descriptors
CURRENT=$(lsof -p $PID 2>/dev/null | wc -l)

# Calculate percentage
PERCENT=$((CURRENT * 100 / SOFT_LIMIT))

# Output in CloudWatch-compatible format
echo "FileDescriptors service=$SERVICE pid=$PID current=$CURRENT limit=$SOFT_LIMIT percent=$PERCENT"

# Alert if usage is high
if [ $PERCENT -gt 80 ]; then
    logger -p user.warning -t fd-monitor "WARNING: $SERVICE using $PERCENT% of file descriptors ($CURRENT/$SOFT_LIMIT)"
fi

if [ $PERCENT -gt 90 ]; then
    logger -p user.error -t fd-monitor "CRITICAL: $SERVICE using $PERCENT% of file descriptors ($CURRENT/$SOFT_LIMIT)"
fi

exit 0

# Make executable
sudo chmod +x /usr/local/bin/monitor-file-descriptors.sh

# Add to cron (every 5 minutes)
sudo crontab -e
*/5 * * * * /usr/local/bin/monitor-file-descriptors.sh

# 2. CloudWatch custom metric
sudo vi /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

{
  "metrics": {
    "namespace": "CustomApplication",
    "metrics_collected": {
      "statsd": {
        "service_address": ":8125",
        "metrics_collection_interval": 60,
        "metrics_aggregation_interval": 60
      }
    }
  },
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/application/api.log",
            "log_group_name": "/aws/ec2/api-service",
            "log_stream_name": "{instance_id}"
          }
        ]
      }
    }
  }
}

# Restart CloudWatch agent
sudo systemctl restart amazon-cloudwatch-agent

# 3. Create CloudWatch alarm
aws cloudwatch put-metric-alarm \\
  --alarm-name api-service-high-fd-usage \\
  --alarm-description "API service file descriptor usage > 80%" \\
  --metric-name FileDescriptorUsagePercent \\
  --namespace CustomApplication \\
  --statistic Average \\
  --period 300 \\
  --threshold 80 \\
  --comparison-operator GreaterThanThreshold \\
  --evaluation-periods 2 \\
  --alarm-actions arn:aws:sns:us-east-1:123456789:ops-alerts
\`\`\`

### AWS-Specific Considerations

\`\`\`bash
# 1. Amazon Linux 2023 comes with reasonable defaults
# But verify:
cat /etc/security/limits.d/20-nproc.conf
# Should have increased limits already

# 2. ECS/Fargate task definition
{
  "family": "api-service",
  "containerDefinitions": [{
    "name": "api",
    "image": "myapp:latest",
    "ulimits": [
      {
        "name": "nofile",
        "softLimit": 65536,
        "hardLimit": 65536
      }
    ],
    "memory": 2048,
    "cpu": 1024
  }]
}

# 3. Kubernetes pod security context
apiVersion: v1
kind: Pod
metadata:
  name: api-service
spec:
  containers:
  - name: api
    image: myapp:latest
    resources:
      limits:
        cpu: "2"
        memory: "4Gi"
    securityContext:
      # Linux capabilities
      capabilities:
        add:
        - SYS_RESOURCE  # Allows increasing limits
\`\`\`

### Complete Pre-Production Checklist

\`\`\`bash
# Before deploying to production:

# ✓ 1. System-wide limits configured
sysctl fs.file-max  # Should be >= 2097152

# ✓ 2. User limits configured
su - api-user -c "ulimit -n"  # Should be 65536

# ✓ 3. systemd service has LimitNOFILE
systemctl show api-service.service | grep LimitNOFILE  # Should be 65536

# ✓ 4. Application process has correct limits
cat /proc/<pid>/limits | grep "Max open files"  # Should be 65536

# ✓ 5. Load testing passed
# wrk -t 12 -c 1000 -d 60s http://localhost:8000/api/health

# ✓ 6. Monitoring configured
# CloudWatch metrics, alarms, dashboards

# ✓ 7. Documentation updated
# Runbook includes file descriptor troubleshooting

# ✓ 8. Alerts configured
# SNS topic for ops team

# ✓ 9. Automated scaling configured
# Auto-scaling based on connection count

# ✓ 10. Tested failure scenarios
# What happens when limit is actually hit?
\`\`\`

### Summary

**The Issue**: Default Linux file descriptor limits (1024) are too low for production high-traffic applications.

**Complete Solution** requires configuration at multiple layers:
1. **System-wide**: \`fs.file-max\` in sysctl
2. **User-level**: \`/etc/security/limits.conf\`
3. **systemd**: \`LimitNOFILE\` in service file (most important!)
4. **Application**: Verify app respects system limits

**Production Recommendation**: 
- Set \`LimitNOFILE=65536\` for all production services
- Monitor usage with CloudWatch
- Alert at 80% usage
- Document in runbooks
- Include in AMI/container image build

**Key Learning**: File descriptor limits are one of the most common production issues. Always configure them BEFORE going to production, not after encountering errors. The systemd \`LimitNOFILE\` setting is the most reliable way to ensure limits are applied correctly.
`,
  },
  {
    id: 3,
    question:
      "You're implementing a new production deployment process for a microservices application on EC2. The security team requires that you implement the principle of least privilege using Linux file permissions and ACLs. You have three user groups: 'developers' (need read access to logs and configs), 'operators' (need read/write to logs, read-only to configs), and 'app-service' (the running application, needs read/write to data directories and read-only to configs). Design the complete permission structure using both standard Unix permissions and ACLs. Include the actual commands, explain your reasoning, and describe how you'd audit and maintain this setup.",
    answer: `## Comprehensive Answer:

This scenario requires implementing defense-in-depth using a combination of standard Unix permissions, ACLs, and systemd security features. Let's design a production-grade permission system following the principle of least privilege.

### Understanding the Requirements

\`\`\`python
# Security requirements matrix
requirements = {
    'developers': {
        'configs': 'read-only',
        'logs': 'read-only',
        'data': 'no access',
        'binaries': 'no access',
        'use_case': 'Debugging production issues, viewing configs'
    },
    'operators': {
        'configs': 'read-only',
        'logs': 'read-write',  # Need to rotate, compress, clean
        'data': 'read-only',   # Backup purposes
        'binaries': 'no access',
        'use_case': 'Log management, monitoring, backups'
    },
    'app-service': {
        'configs': 'read-only',  # Application reads config
        'logs': 'read-write',    # Application writes logs
        'data': 'read-write',    # Application data storage
        'binaries': 'read-execute',  # Execute application code
        'use_case': 'Running application process'
    },
}

# Directory structure:
structure = {
    '/opt/app/': {
        'bin/': 'Application binaries',
        'config/': 'Configuration files',
        'logs/': 'Application logs',
        'data/': 'Application data (uploads, cache, etc.)',
        'tmp/': 'Temporary files',
    }
}
\`\`\`

### Step 1: Create Users and Groups

\`\`\`bash
# Create system user for the application (no login)
sudo useradd --system --no-create-home --shell /bin/false app-service

# Create groups
sudo groupadd developers
sudo groupadd operators

# Add users to groups (examples)
sudo usermod -aG developers dev1
sudo usermod -aG developers dev2
sudo usermod -aG operators ops1
sudo usermod -aG operators ops2

# Verify group membership
getent group developers
# developers:x:1001:dev1,dev2

getent group operators
# operators:x:1002:ops1,ops2

# Note: app-service user is NOT in developers or operators groups
id app-service
# uid=999(app-service) gid=999(app-service) groups=999(app-service)
\`\`\`

### Step 2: Create Directory Structure with Base Permissions

\`\`\`bash
# Create directory structure
sudo mkdir -p /opt/app/{bin,config,logs,data,tmp}

# Set ownership and base permissions
# Strategy: Owner=app-service, Group=app-service, Other=none
# Then use ACLs for developers and operators

# /opt/app - Root directory
sudo chown -R app-service:app-service /opt/app
sudo chmod 755 /opt/app  # rwxr-xr-x (others can traverse)

# /opt/app/bin - Application binaries
sudo chown -R app-service:app-service /opt/app/bin
sudo chmod 750 /opt/app/bin              # rwxr-x--- (owner+group execute)
sudo find /opt/app/bin -type f -exec chmod 550 {} \\;  # r-xr-x---

# /opt/app/config - Configuration files (SENSITIVE!)
sudo chown -R app-service:app-service /opt/app/config
sudo chmod 750 /opt/app/config           # rwxr-x---
sudo find /opt/app/config -type f -exec chmod 440 {} \\;  # r--r----- (read-only)

# Special: Protect sensitive configs
sudo chmod 400 /opt/app/config/database.conf  # r-------- (owner only)
sudo chmod 400 /opt/app/config/secrets.yaml   # r-------- (owner only)

# /opt/app/logs - Log files
sudo chown -R app-service:app-service /opt/app/logs
sudo chmod 770 /opt/app/logs             # rwxrwx---
sudo find /opt/app/logs -type f -exec chmod 660 {} \\;  # rw-rw----

# /opt/app/data - Application data
sudo chown -R app-service:app-service /opt/app/data
sudo chmod 770 /opt/app/data             # rwxrwx---
sudo find /opt/app/data -type f -exec chmod 660 {} \\;  # rw-rw----

# /opt/app/tmp - Temporary files
sudo chown -R app-service:app-service /opt/app/tmp
sudo chmod 1770 /opt/app/tmp             # rwxrwx--T (sticky bit!)
# Sticky bit: Only owner can delete their own files

# Verify base structure
tree -pug /opt/app
\`\`\`

### Step 3: Apply ACLs for Fine-Grained Access

**Enable ACL support** (usually enabled by default on modern systems):

\`\`\`bash
# Check if ACLs are enabled
mount | grep /opt
# If "acl" is not listed:
sudo mount -o remount,acl /opt

# Or make permanent in /etc/fstab:
# UUID=xxx  /opt  ext4  defaults,acl  0  2
\`\`\`

**Apply ACLs for developers group**:

\`\`\`bash
# Developers: Read-only access to logs and configs

# Config directory - Read + Execute (traverse)
sudo setfacl -R -m g:developers:rX /opt/app/config
# r = read, X = execute only if already executable or directory

# Set default ACL for new files in config
sudo setfacl -R -d -m g:developers:rX /opt/app/config

# Logs directory - Read + Execute
sudo setfacl -R -m g:developers:rX /opt/app/logs
sudo setfacl -R -d -m g:developers:rX /opt/app/logs

# Explicitly deny access to sensitive configs
sudo setfacl -m g:developers:--- /opt/app/config/database.conf
sudo setfacl -m g:developers:--- /opt/app/config/secrets.yaml

# Deny access to data directory
sudo setfacl -m g:developers:--- /opt/app/data

# Verify
sudo getfacl /opt/app/logs
# file: opt/app/logs
# owner: app-service
# group: app-service
# user::rwx
# group::rwx
# group:developers:r-x
# mask::rwx
# other::---
\`\`\`

**Apply ACLs for operators group**:

\`\`\`bash
# Operators: Read-only to configs, Read-write to logs, Read-only to data

# Config directory - Read + Execute
sudo setfacl -R -m g:operators:rX /opt/app/config
sudo setfacl -R -d -m g:operators:rX /opt/app/config

# Logs directory - Read + Write + Execute
sudo setfacl -R -m g:operators:rwX /opt/app/logs
sudo setfacl -R -d -m g:operators:rwX /opt/app/logs

# Data directory - Read + Execute (for backups)
sudo setfacl -R -m g:operators:rX /opt/app/data
sudo setfacl -R -d -m g:operators:rX /opt/app/data

# Deny access to sensitive configs
sudo setfacl -m g:operators:--- /opt/app/config/database.conf
sudo setfacl -m g:operators:--- /opt/app/config/secrets.yaml

# Verify
sudo getfacl /opt/app/logs
# file: opt/app/logs
# owner: app-service
# group: app-service
# user::rwx
# group::rwx
# group:developers:r-x
# group:operators:rwx
# mask::rwx
# other::---
\`\`\`

### Step 4: Secure Sensitive Files with Restricted ACLs

\`\`\`bash
# Database credentials - Only app-service can read
sudo setfacl -b /opt/app/config/database.conf  # Remove all ACLs
sudo chmod 400 /opt/app/config/database.conf   # r--------
sudo chown app-service:app-service /opt/app/config/database.conf

# Verify nobody else can read
sudo -u dev1 cat /opt/app/config/database.conf
# cat: /opt/app/config/database.conf: Permission denied ✓

# API keys / secrets
sudo setfacl -b /opt/app/config/secrets.yaml
sudo chmod 400 /opt/app/config/secrets.yaml
sudo chown app-service:app-service /opt/app/config/secrets.yaml

# Alternative: Use AWS Secrets Manager instead of files
# This is the best practice for production secrets!
\`\`\`

### Step 5: systemd Service Security Hardening

\`\`\`bash
# Create systemd service with security restrictions
sudo vi /etc/systemd/system/app-service.service

[Unit]
Description=Production Application Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=app-service
Group=app-service

# Working directory
WorkingDirectory=/opt/app

# File permissions mask (new files created as 640)
UMask=027

# Security hardening
NoNewPrivileges=true              # Prevent privilege escalation
PrivateTmp=true                   # Isolated /tmp
ProtectSystem=strict              # Read-only /usr, /boot, /etc
ProtectHome=true                  # No access to /home
ProtectKernelTunables=true        # Can't modify /proc/sys
ProtectKernelModules=true         # Can't load kernel modules
ProtectControlGroups=true         # Can't modify cgroups
RestrictRealtime=true             # No realtime scheduling
RestrictNamespaces=true           # Restrict namespace creation

# Read-only paths (everything except what's needed)
ReadOnlyPaths=/opt/app/bin
ReadOnlyPaths=/opt/app/config

# Read-write paths (only what's needed)
ReadWritePaths=/opt/app/logs
ReadWritePaths=/opt/app/data
ReadWritePaths=/opt/app/tmp

# Prevent writing to these
InaccessiblePaths=/root
InaccessiblePaths=/home

# System call filtering (optional but recommended)
SystemCallFilter=@system-service
SystemCallFilter=~@privileged @resources

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Environment
Environment="APP_ENV=production"
Environment="APP_CONFIG=/opt/app/config/app.conf"

# Startup
ExecStart=/opt/app/bin/application

# Restart policy
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target

# Reload and start
sudo systemctl daemon-reload
sudo systemctl enable app-service
sudo systemctl start app-service

# Verify security settings are applied
sudo systemctl show app-service | grep -i protect
sudo systemctl show app-service | grep -i private
\`\`\`

### Step 6: Testing Access Controls

\`\`\`bash
#!/bin/bash
# Test script: /root/test-permissions.sh

echo "=== Permission Testing ==="

# Test as developer
echo "\\n--- Testing as developer (dev1) ---"

echo "Attempt: Read config file"
sudo -u dev1 cat /opt/app/config/app.conf && echo "✓ Success" || echo "✗ Denied"

echo "Attempt: Read log file"
sudo -u dev1 tail /opt/app/logs/app.log && echo "✓ Success" || echo "✗ Denied"

echo "Attempt: Write to log file"
sudo -u dev1 sh -c 'echo "test" >> /opt/app/logs/app.log' && echo "✓ Success" || echo "✗ Denied (expected)"

echo "Attempt: Read data directory"
sudo -u dev1 ls /opt/app/data && echo "✓ Success" || echo "✗ Denied (expected)"

echo "Attempt: Read database.conf (sensitive)"
sudo -u dev1 cat /opt/app/config/database.conf && echo "✗ SECURITY ISSUE!" || echo "✓ Denied (expected)"

# Test as operator
echo "\\n--- Testing as operator (ops1) ---"

echo "Attempt: Read config file"
sudo -u ops1 cat /opt/app/config/app.conf && echo "✓ Success" || echo "✗ Denied"

echo "Attempt: Read log file"
sudo -u ops1 tail /opt/app/logs/app.log && echo "✓ Success" || echo "✗ Denied"

echo "Attempt: Write to log file"
sudo -u ops1 sh -c 'echo "test" >> /opt/app/logs/app.log' && echo "✓ Success" || echo "✗ Denied"

echo "Attempt: Delete log file"
sudo -u ops1 rm /opt/app/logs/old.log && echo "✓ Success" || echo "✗ Denied"

echo "Attempt: Read data directory"
sudo -u ops1 ls /opt/app/data && echo "✓ Success" || echo "✗ Denied"

echo "Attempt: Write to data directory"
sudo -u ops1 touch /opt/app/data/test.txt && echo "✗ SECURITY ISSUE!" || echo "✓ Denied (expected)"

echo "Attempt: Read database.conf (sensitive)"
sudo -u ops1 cat /opt/app/config/database.conf && echo "✗ SECURITY ISSUE!" || echo "✓ Denied (expected)"

# Test as application
echo "\\n--- Testing as app-service ---"

echo "Attempt: Read config file"
sudo -u app-service cat /opt/app/config/app.conf && echo "✓ Success" || echo "✗ Denied"

echo "Attempt: Read database.conf"
sudo -u app-service cat /opt/app/config/database.conf && echo "✓ Success" || echo "✗ Denied"

echo "Attempt: Write to logs"
sudo -u app-service sh -c 'echo "test" >> /opt/app/logs/app.log' && echo "✓ Success" || echo "✗ Denied"

echo "Attempt: Write to data"
sudo -u app-service touch /opt/app/data/test.txt && echo "✓ Success" || echo "✗ Denied"

echo "Attempt: Modify config (should fail)"
sudo -u app-service sh -c 'echo "malicious" >> /opt/app/config/app.conf' && echo "✗ SECURITY ISSUE!" || echo "✓ Denied (expected)"

echo "\\n=== Test Complete ===\"

# Run the test
sudo bash /root/test-permissions.sh
\`\`\`

### Step 7: Auditing and Maintenance

**Create ACL audit script**:

\`\`\`bash
#!/bin/bash
# /usr/local/bin/audit-acls.sh
# Audit file permissions and ACLs

APP_DIR="/opt/app"
REPORT_FILE="/var/log/acl-audit-$(date +%Y%m%d-%H%M%S).log"

echo "ACL Audit Report - $(date)" > $REPORT_FILE
echo "======================================" >> $REPORT_FILE

# Function to check directory permissions
audit_directory() {
    local dir=$1
    echo "\\n--- $dir ---" >> $REPORT_FILE
    ls -ld "$dir" >> $REPORT_FILE
    getfacl "$dir" >> $REPORT_FILE 2>&1
}

# Audit all directories
for dir in $APP_DIR $APP_DIR/bin $APP_DIR/config $APP_DIR/logs $APP_DIR/data $APP_DIR/tmp; do
    if [ -d "$dir" ]; then
        audit_directory "$dir"
    fi
done

# Check for world-readable sensitive files
echo "\\n--- World-Readable Files Check ---" >> $REPORT_FILE
find $APP_DIR -type f -perm -004 >> $REPORT_FILE

# Check for SUID/SGID files (security risk)
echo "\\n--- SUID/SGID Files (Security Risk!) ---" >> $REPORT_FILE
find $APP_DIR -type f \\( -perm -4000 -o -perm -2000 \\) -ls >> $REPORT_FILE

# Check ownership
echo "\\n--- Files Not Owned by app-service ---" >> $REPORT_FILE
find $APP_DIR ! -user app-service -ls >> $REPORT_FILE

# Summary
echo "\\n--- Summary ---" >> $REPORT_FILE
echo "Total files: $(find $APP_DIR -type f | wc -l)" >> $REPORT_FILE
echo "World-readable: $(find $APP_DIR -type f -perm -004 | wc -l)" >> $REPORT_FILE
echo "SUID/SGID: $(find $APP_DIR -type f \\( -perm -4000 -o -perm -2000 \\) | wc -l)" >> $REPORT_FILE

echo "Audit complete. Report: $REPORT_FILE"

# Alert if issues found
ISSUES=$(grep -c "SECURITY RISK" $REPORT_FILE)
if [ $ISSUES -gt 0 ]; then
    logger -p user.warning -t acl-audit "Found $ISSUES security issues in ACL audit"
fi

# Make executable
sudo chmod +x /usr/local/bin/audit-acls.sh

# Schedule monthly audit
sudo crontab -e
# Add:
0 2 1 * * /usr/local/bin/audit-acls.sh
\`\`\`

**Monitor for permission changes**:

\`\`\`bash
# Install audit daemon
sudo yum install audit  # Amazon Linux / RHEL
# or
sudo apt-get install auditd  # Ubuntu

# Configure audit rules
sudo vi /etc/audit/rules.d/app-permissions.rules

# Watch for permission changes on application directories
-w /opt/app/config/ -p wa -k config-changes
-w /opt/app/bin/ -p wa -k binary-changes
-w /opt/app/logs/ -p wa -k log-changes

# Watch for ACL changes
-w /usr/bin/setfacl -p x -k acl-changes
-w /usr/bin/chown -p x -k ownership-changes
-w /usr/bin/chmod -p x -k permission-changes

# Reload audit rules
sudo augenrules --load

# View audit logs
sudo ausearch -k config-changes
sudo ausearch -k acl-changes

# Generate report
sudo aureport --auth
\`\`\`

### Step 8: Documentation and Runbook

\`\`\`markdown
# Application Permissions Runbook

## Overview
Production application uses principle of least privilege with ACLs.

## Groups and Access

| Group      | Config | Logs | Data | Binaries |
|------------|--------|------|------|----------|
| developers | Read   | Read | None | None     |
| operators  | Read   | R/W  | Read | None     |
| app-service| Read   | R/W  | R/W  | Execute  |

## Common Tasks

### Add new developer
\`\`\`bash
sudo usermod -aG developers newdev
# Access takes effect on next login
\`\`\`

### Add new operator
\`\`\`bash
sudo usermod -aG operators newops
\`\`\`

### Rotate log files
\`\`\`bash
# Operators can rotate logs
sudo -u ops1 logrotate /etc/logrotate.d/app-service
\`\`\`

### Deploy new configuration
\`\`\`bash
# 1. Copy new config as root
sudo cp new-config.conf /opt/app/config/

# 2. Set permissions
sudo chown app-service:app-service /opt/app/config/new-config.conf
sudo chmod 440 /opt/app/config/new-config.conf

# 3. Apply ACLs
sudo setfacl -m g:developers:r /opt/app/config/new-config.conf
sudo setfacl -m g:operators:r /opt/app/config/new-config.conf

# 4. Reload application
sudo systemctl reload app-service
\`\`\`

### Troubleshooting Permission Issues

\`\`\`bash
# Check effective permissions for user
sudo -u dev1 -s
cd /opt/app/logs
ls -la  # See what user can access

# Check ACLs
getfacl /opt/app/logs/app.log

# Check process permissions
ps aux | grep app-service
# Note PID

cat /proc/<PID>/status | grep -i uid
cat /proc/<PID>/status | grep -i gid
\`\`\`
\`\`\`

### Step 9: Infrastructure as Code (Terraform)

\`\`\`hcl
# Terraform user_data script for EC2 instances

resource "aws_instance" "app_server" {
  ami           = "ami-xxxxx"  # Amazon Linux 2023
  instance_type = "t3.medium"
  
  user_data = <<-EOF
    #!/bin/bash
    set -e
    
    # Create application user and groups
    useradd --system --no-create-home --shell /bin/false app-service
    groupadd developers
    groupadd operators
    
    # Create directory structure
    mkdir -p /opt/app/{bin,config,logs,data,tmp}
    chown -R app-service:app-service /opt/app
    chmod 755 /opt/app
    
    # Set base permissions
    chmod 750 /opt/app/bin
    chmod 750 /opt/app/config
    chmod 770 /opt/app/logs
    chmod 770 /opt/app/data
    chmod 1770 /opt/app/tmp
    
    # Apply ACLs
    setfacl -R -m g:developers:rX /opt/app/config
    setfacl -R -m g:developers:rX /opt/app/logs
    setfacl -R -m g:operators:rX /opt/app/config
    setfacl -R -m g:operators:rwX /opt/app/logs
    setfacl -R -m g:operators:rX /opt/app/data
    
    # Set default ACLs for new files
    setfacl -d -R -m g:developers:rX /opt/app/logs
    setfacl -d -R -m g:operators:rwX /opt/app/logs
    
    # Deploy application
    aws s3 cp s3://my-bucket/app-binary /opt/app/bin/application
    chmod 550 /opt/app/bin/application
    
    # Deploy systemd service
    aws s3 cp s3://my-bucket/app-service.service /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable app-service
    systemctl start app-service
  EOF
  
  tags = {
    Name = "app-server-prod"
  }
}
\`\`\`

### Summary

**Complete Permission Architecture**:

1. **Base Layer**: Standard Unix permissions (owner, group, other)
2. **ACL Layer**: Fine-grained access for multiple groups
3. **systemd Layer**: Service-level security (ProtectSystem, ReadOnlyPaths, etc.)
4. **Audit Layer**: Monitoring and alerting for changes

**Key Security Principles Applied**:
- ✓ Principle of least privilege
- ✓ Defense in depth (multiple security layers)
- ✓ Separation of duties (developers ≠ operators ≠ application)
- ✓ Deny by default (explicit grants only)
- ✓ Immutable configs (read-only for application)
- ✓ Audit trail (auditd monitoring)

**Production Checklist**:
- [ ] Users and groups created
- [ ] Directory structure with base permissions
- [ ] ACLs applied and verified
- [ ] systemd service hardened
- [ ] Access controls tested
- [ ] Audit logging configured
- [ ] Monitoring and alerts set up
- [ ] Documentation written
- [ ] Team trained on procedures
- [ ] Quarterly audit scheduled

This setup provides defense-in-depth while maintaining operational flexibility for developers and operators to do their jobs without compromising security.
`,
  },
];
