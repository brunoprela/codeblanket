/**
 * Process & Resource Management Section
 * Module: Linux System Administration & DevOps Foundations
 */

export const processResourceManagementSection = {
  id: 'process-resource-management',
  title: 'Process & Resource Management',
  content: `# Process & Resource Management

## Introduction

Effective process and resource management is critical for maintaining stable, performant production systems. This section covers resource limits (ulimit), cgroups for resource isolation, OOM management, process priority, and AWS EC2 sizing strategies. You'll learn how to prevent resource exhaustion, optimize performance, and handle production workload scenarios.

## Understanding Resource Limits

### ulimit - Per-Process Limits

\`\`\`bash
# View current limits
ulimit -a
# core file size          (blocks, -c) 0
# data seg size           (kbytes, -d) unlimited
# scheduling priority             (-e) 0
# file size               (blocks, -f) unlimited
# pending signals                 (-i) 15390
# max locked memory       (kbytes, -l) 65536
# max memory size         (kbytes, -m) unlimited
# open files                      (-n) 1024
# pipe size            (512 bytes, -p) 8
# POSIX message queues     (bytes, -q) 819200
# real-time priority              (-r) 0
# stack size              (kbytes, -s) 8192
# cpu time               (seconds, -t) unlimited
# max user processes              (-u) 15390
# virtual memory          (kbytes, -v) unlimited
# file locks                      (-x) unlimited

# Set limits for current shell
ulimit -n 65536    # Open files
ulimit -u 4096     # Processes

# Soft vs hard limits
ulimit -Sn         # Soft limit (can be increased up to hard)
ulimit -Hn         # Hard limit (requires root to increase)

# Set both
ulimit -Sn 10000 -Hn 65536
\`\`\`

### System-Wide Resource Limits

\`\`\`bash
# Edit /etc/security/limits.conf
sudo vi /etc/security/limits.conf

# Format: <domain> <type> <item> <value>

# User limits
myapp     soft    nofile    10000
myapp     hard    nofile    65536
myapp     soft    nproc     4096
myapp     hard    nproc     8192

# Group limits
@developers  soft  nofile  10000
@developers  hard  nofile  65536

# Wildcard (all users)
*         soft    core      0
*         hard    core      0

# Specific limits
#  - nofile: open files
#  - nproc: processes
#  - core: core dump size
#  - memlock: locked memory
#  - cpu: CPU time
#  - nice: nice priority
#  - as: address space
#  - stack: stack size

# Apply limits (requires re-login or PAM)
# Verify with: ulimit -a
\`\`\`

### Systemd Service Limits

\`\`\`bash
# In service file
cat << 'EOF' > /etc/systemd/system/myapp.service
[Service]
# File descriptors
LimitNOFILE=65536

# Processes/threads
LimitNPROC=4096

# Core dumps
LimitCORE=infinity    # Enable core dumps
# OR
LimitCORE=0           # Disable core dumps

# Memory
LimitAS=2G            # Virtual memory
LimitDATA=2G          # Data segment
LimitSTACK=8M         # Stack size

# CPU time
LimitCPU=infinity

# File size
LimitFSIZE=infinity

# Locked memory
LimitMEMLOCK=64K

# Example: Production web app
[Unit]
Description=Production Web App

[Service]
Type=simple
User=webapp
ExecStart=/opt/webapp/server

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096
LimitCORE=0
MemoryLimit=2G
CPUQuota=200%

[Install]
WantedBy=multi-user.target
EOF
\`\`\`

## Cgroups - Control Groups

### What are Cgroups?

Cgroups provide resource isolation and limits for processes. They enable:
- CPU usage limits
- Memory limits  
- I/O bandwidth limits
- Network bandwidth (with tc)
- Device access control

### Cgroups v2 (Unified Hierarchy)

\`\`\`bash
# Check cgroup version
mount | grep cgroup
# cgroup2 on /sys/fs/cgroup type cgroup2

# View cgroup controllers
cat /sys/fs/cgroup/cgroup.controllers
# cpuset cpu io memory hugetlb pids rdma misc

# View process cgroup
cat /proc/self/cgroup
# 0::/user.slice/user-1000.slice/session-3.scope

# Systemd uses cgroups automatically
systemctl status myapp
# CGroup: /system.slice/myapp.service
#         └─1234 /opt/myapp/server

# View cgroup details
systemd-cgls
# Control group /:
# -.slice
# ├─user.slice
# │ └─user-1000.slice
# └─system.slice
#   ├─myapp.service
#   │ └─1234 /opt/myapp/server
#   └─nginx.service
#     ├─1000 nginx: master
#     ├─1001 nginx: worker
#     └─1002 nginx: worker
\`\`\`

### CPU Limits with Cgroups

\`\`\`bash
# Systemd service with CPU limits
[Service]
# CPU quota (percentage)
CPUQuota=50%        # Max 50% of 1 CPU
CPUQuota=200%       # Max 2 full CPUs

# CPU weight (shares)
CPUWeight=100       # Default (1-10000)
CPUWeight=500       # 5x more CPU than default

# CPU affinity (pin to specific CPUs)
CPUAffinity=0 1     # Use only CPUs 0 and 1

# Example: Limit to 1.5 CPUs
[Unit]
Description=CPU Limited Service

[Service]
ExecStart=/opt/myapp/server
CPUQuota=150%
CPUWeight=200

[Install]
WantedBy=multi-user.target

# Monitor CPU usage
systemd-cgtop
# Control Group                          Tasks   %CPU   Memory  
# /system.slice/myapp.service             5      75.0   512.5M
# /system.slice/nginx.service             3      12.3   128.2M

# Detailed CPU stats
systemctl show myapp -p CPUUsageNSec
# CPUUsageNSec=125430000000
\`\`\`

### Memory Limits with Cgroups

\`\`\`bash
# Systemd service with memory limits
[Service]
# Memory limits
MemoryHigh=1.8G      # Soft limit (throttling starts)
MemoryMax=2G         # Hard limit (OOM kill)
MemorySwapMax=0      # Disable swap

# Example: Memory-limited application
[Unit]
Description=Memory Limited App

[Service]
ExecStart=/opt/myapp/server
MemoryHigh=1800M
MemoryMax=2048M
MemorySwapMax=0

[Install]
WantedBy=multi-user.target

# Monitor memory
systemctl status myapp
# Memory: 1.2G (max: 2.0G available: 768.0M)

# Detailed memory stats
systemctl show myapp -p MemoryCurrent
# MemoryCurrent=1258291200

# Memory pressure
cat /sys/fs/cgroup/system.slice/myapp.service/memory.pressure
# some avg10=0.00 avg60=0.00 avg300=0.00 total=0
# full avg10=0.00 avg60=0.00 avg300=0.00 total=0
\`\`\`

### I/O Limits with Cgroups

\`\`\`bash
# Systemd service with I/O limits
[Service]
# I/O weight (relative priority)
IOWeight=100         # Default (1-10000)
IOWeight=500         # Higher I/O priority

# I/O device limits
IOReadBandwidthMax=/dev/sda 100M    # Max 100 MB/s read
IOWriteBandwidthMax=/dev/sda 50M    # Max 50 MB/s write

# IOPS limits
IOReadIOPSMax=/dev/sda 1000         # Max 1000 read IOPS
IOWriteIOPSMax=/dev/sda 500         # Max 500 write IOPS

# Example: Database with I/O limits
[Unit]
Description=Database Service

[Service]
ExecStart=/usr/bin/postgres
IOWeight=500
IOReadBandwidthMax=/dev/nvme0n1 500M
IOWriteBandwidthMax=/dev/nvme0n1 200M

[Install]
WantedBy=multi-user.target

# Monitor I/O
systemd-cgtop
# Control Group                          Tasks   %CPU   Memory  Input/s Output/s
# /system.slice/postgresql.service        12     45.2   1.2G    125.3M  45.2M
\`\`\`

## OOM (Out of Memory) Management

### Understanding OOM Killer

\`\`\`bash
# View OOM scores
ps -eo pid,comm,oom_score,oom_adj,oom_score_adj
# PID  COMMAND         OOM_SCORE  OOM_ADJ  OOM_SCORE_ADJ
# 1    systemd         0          0        -1000
# 1234 myapp           250        0        0
# 5678 postgres        180        0        -100

# OOM score calculation:
# - Higher score = more likely to be killed
# - Based on: memory usage, runtime, process tree
# - Range: 0-1000
# - -1000 = never kill (e.g., systemd)

# Adjust OOM score
echo -500 | sudo tee /proc/1234/oom_score_adj
# Negative: less likely to be killed
# Positive: more likely to be killed
# -1000: never kill
# 1000: always kill first

# Systemd service OOM adjustment
[Service]
OOMScoreAdjust=-500    # Protect from OOM killer
OOMScoreAdjust=500     # Sacrifice first
\`\`\`

### Preventing OOM Kills

\`\`\`bash
# Set memory limits to prevent OOM
[Service]
MemoryMax=2G           # Hard limit
MemoryHigh=1.8G        # Warning threshold

# Monitor for OOM events
journalctl -k | grep -i "out of memory"
# [12345.678] Out of memory: Killed process 1234 (myapp)

# dmesg for OOM events
dmesg | grep -i "killed process"
# [Mon Oct 28 10:00:00 2024] Out of memory: Killed process 1234 (myapp) total-vm:2048000kB

# Set vm.overcommit_memory
sysctl vm.overcommit_memory
# 0: Heuristic (default)
# 1: Always overcommit
# 2: Never overcommit

# For critical services, use:
sysctl -w vm.overcommit_memory=2
sysctl -w vm.overcommit_ratio=80

# Make permanent
echo "vm.overcommit_memory=2" >> /etc/sysctl.conf
echo "vm.overcommit_ratio=80" >> /etc/sysctl.conf
\`\`\`

## Process Priority and Scheduling

### Nice Values

\`\`\`bash
# Nice values: -20 (highest priority) to +19 (lowest priority)
# Default: 0

# Start process with nice value
nice -n 10 /opt/myapp/server
# Lower priority (background task)

nice -n -5 /opt/critical/service
# Higher priority (needs root for negative)

# Change nice value of running process
renice -n 5 -p 1234
# PID 1234: 0 -> 5

renice -n -10 -p 5678
# Requires root for negative values

# Systemd service with nice
[Service]
Nice=-5              # Higher priority
Nice=10              # Lower priority

# View process priorities
ps -eo pid,ni,comm
# PID  NI COMMAND
# 1234  0 myapp
# 5678 10 backup
\`\`\`

### CPU Affinity

\`\`\`bash
# Bind process to specific CPUs
taskset -c 0,1 /opt/myapp/server
# Run on CPUs 0 and 1 only

# Change affinity of running process
taskset -cp 2,3 1234
# PID 1234's current affinity list: 0-7
# PID 1234's new affinity list: 2,3

# Systemd service with CPU affinity
[Service]
CPUAffinity=0 1 2 3    # Use CPUs 0-3

# View CPU affinity
taskset -cp 1234
# pid 1234's current affinity list: 0,1,2,3
\`\`\`

### Realtime Scheduling

\`\`\`bash
# Realtime scheduling policies (requires CAP_SYS_NICE)
# SCHED_FIFO: First-in-first-out realtime
# SCHED_RR: Round-robin realtime
# SCHED_OTHER: Default time-sharing

# Set realtime priority
chrt -f 50 /opt/latency-sensitive/app
# -f: FIFO scheduling
# 50: priority (1-99)

# View scheduling policy
chrt -p 1234
# pid 1234's current scheduling policy: SCHED_OTHER
# pid 1234's current scheduling priority: 0

# Systemd service with realtime
[Service]
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=50

# Warning: Realtime can starve other processes!
# Use only for latency-critical workloads
\`\`\`

## Background Jobs and Process Management

### Job Control

\`\`\`bash
# Run in background
/opt/myapp/server &
# [1] 1234

# List background jobs
jobs
# [1]+  Running     /opt/myapp/server &

# Bring to foreground
fg %1

# Send to background (must be stopped first)
Ctrl+Z  # Stop process
bg %1   # Resume in background

# Disown job (keep running after logout)
/opt/myapp/server &
disown %1

# nohup (no hangup)
nohup /opt/myapp/server &
# Output to nohup.out

# Better: Use screen or tmux for persistent sessions
screen -S myapp
/opt/myapp/server
# Ctrl+A D to detach

# Reattach
screen -r myapp
\`\`\`

### Process Monitoring

\`\`\`bash
# Real-time process monitoring
top
htop  # Interactive
atop  # Advanced system & process monitor

# Process tree
pstree -p
pstree -p 1234  # Tree for specific process

# List processes by resource
ps aux --sort=-%cpu | head -10    # Top CPU
ps aux --sort=-%mem | head -10    # Top memory

# Watch specific process
watch -n 1 'ps -p 1234 -o pid,ppid,%cpu,%mem,cmd'

# Monitor process I/O
iotop -p 1234

# Monitor process network
nethogs

# Detailed process info
cat /proc/1234/status
cat /proc/1234/limits
cat /proc/1234/cgroup
\`\`\`

## AWS EC2 Instance Sizing

### Choosing Instance Types

\`\`\`bash
# Instance type format: <family><generation>.<size>
# Example: t3.large
#  - t: Burstable performance
#  - 3: Generation
#  - large: Size

# Instance families:
# t3/t4g: Burstable (web servers, dev environments)
# m5/m6: General purpose (balanced)
# c5/c6: Compute optimized (CPU-intensive)
# r5/r6: Memory optimized (databases, caching)
# i3/i4: Storage optimized (NoSQL, data warehouses)
# p3/p4: GPU (ML training)
# g4/g5: GPU (graphics, ML inference)
\`\`\`

### Right-Sizing Strategy

\`\`\`terraform
# Start with monitoring
resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  alarm_name          = "high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# Memory monitoring (requires CloudWatch agent)
resource "aws_cloudwatch_metric_alarm" "memory_high" {
  alarm_name          = "high-memory"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "mem_used_percent"
  namespace           = "CWAgent"
  period              = "300"
  statistic           = "Average"
  threshold           = "85"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# Right-sizing recommendations
# Use AWS Compute Optimizer or analyze metrics:
# - CPU < 40%: Downsize
# - CPU > 70%: Upsize
# - Memory < 60%: Downsize
# - Memory > 80%: Upsize
# - Network bottleneck: Use enhanced networking
# - I/O bottleneck: Use EBS-optimized, larger instance
\`\`\`

### Burstable Instances (T3/T4g)

\`\`\`bash
# T3 instances earn CPU credits
# 1 CPU credit = 1 vCPU at 100% for 1 minute

# Monitor CPU credits
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUCreditBalance \
  --dimensions Name=InstanceId,Value=i-1234567890abcdef0 \
  --start-time 2024-10-28T00:00:00Z \
  --end-time 2024-10-28T23:59:59Z \
  --period 3600 \
  --statistics Average

# T3 unlimited mode (pay for burst)
resource "aws_instance" "web" {
  ami           = "ami-12345678"
  instance_type = "t3.medium"
  
  credit_specification {
    cpu_credits = "unlimited"  # or "standard"
  }
}

# Use T3 when:
# - Workload is bursty (web servers, dev/test)
# - Average CPU < 30-40%
# - Can tolerate throttling (standard) or pay for burst (unlimited)

# Use M5/C5 when:
# - Sustained high CPU
# - Predictable performance required
# - Latency-sensitive workloads
\`\`\`

## Real-World Scenarios

### Scenario 1: Application Running Out of File Descriptors

\`\`\`bash
# Symptom
journalctl -u myapp | grep "Too many open files"

# Check current limit
cat /proc/$(pgrep myapp)/limits | grep "open files"
# Max open files  1024  1024  files

# Fix: Increase in systemd service
sudo systemctl edit myapp.service
# [Service]
# LimitNOFILE=65536

sudo systemctl daemon-reload
sudo systemctl restart myapp

# Verify
cat /proc/$(pgrep myapp)/limits | grep "open files"
# Max open files  65536  65536  files
\`\`\`

### Scenario 2: Process Consuming Too Much Memory

\`\`\`bash
# Monitor process
ps -p 1234 -o pid,vsz,rss,cmd
# PID      VSZ    RSS CMD
# 1234  2048000 1800000 /opt/myapp/server

# Set memory limit
sudo systemctl edit myapp.service
# [Service]
# MemoryMax=2G
# MemoryHigh=1.8G

# Enable swap accounting (if needed)
sudo vi /etc/default/grub
# GRUB_CMDLINE_LINUX="swapaccount=1"
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
sudo reboot

# Monitor memory usage
systemctl status myapp
# Memory: 1.6G (max: 2.0G)
\`\`\`

### Scenario 3: CPU-Intensive Process Starving Others

\`\`\`bash
# Identify CPU hog
top
# PID  USER   PR NI  %CPU  %MEM  COMMAND
# 1234 myapp  20  0  99.9  10.0  cpu-intensive-task

# Reduce priority
sudo renice -n 19 -p 1234
# Or
sudo nice -n 19 /opt/myapp/cpu-task

# Set CPU limit
sudo systemctl edit cpu-task.service
# [Service]
# CPUQuota=50%      # Max 50% of 1 CPU
# Nice=15           # Lower priority
\`\`\`

## Best Practices

✅ **Set resource limits** for all production services  
✅ **Use cgroups** for resource isolation  
✅ **Monitor resource usage** proactively  
✅ **Protect critical services** from OOM killer  
✅ **Use appropriate instance types** for workload  
✅ **Implement auto-scaling** for variable workloads  
✅ **Test resource limits** before production  
✅ **Document resource requirements** for each service  
✅ **Regular capacity planning** reviews  
✅ **Monitor CPU credits** on T3/T4g instances

## Next Steps

In the next section, we'll cover **Backup & Disaster Recovery**, including backup strategies, RTO/RPO planning, and AWS backup services.`,
};
