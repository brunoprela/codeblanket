/**
 * System Monitoring & Performance Section
 * Module: Linux System Administration & DevOps Foundations
 */

export const systemMonitoringPerformanceSection = {
  id: 'system-monitoring-performance',
  title: 'System Monitoring & Performance',
  content: `# System Monitoring & Performance

## Introduction

Production system monitoring is critical for maintaining application performance, identifying bottlenecks, and preventing outages. This section covers essential Linux monitoring tools and techniques used by DevOps engineers to diagnose and resolve performance issues on AWS EC2 instances and other production systems.

## Understanding System Performance Metrics

### The Four Golden Signals

\`\`\`python
"""
Google's Four Golden Signals for monitoring:
1. Latency - How long requests take
2. Traffic - How much demand on the system
3. Errors - Rate of failed requests
4. Saturation - How 'full' the system is
"""

golden_signals = {
    'latency': {
        'metrics': ['response_time', 'query_time', 'api_latency'],
        'target': 'p95 < 200ms',
        'tools': ['application_metrics', 'apm_tools']
    },
    'traffic': {
        'metrics': ['requests_per_second', 'concurrent_users', 'bandwidth'],
        'target': 'sustained_throughput >= design_capacity',
        'tools': ['load_balancer_metrics', 'application_logs']
    },
    'errors': {
        'metrics': ['error_rate', '5xx_responses', 'exceptions'],
        'target': 'error_rate < 0.1%',
        'tools': ['logs', 'error_tracking', 'monitoring_dashboards']
    },
    'saturation': {
        'metrics': ['cpu_utilization', 'memory_usage', 'disk_io', 'network_bandwidth'],
        'target': 'resources < 80% at peak',
        'tools': ['top', 'iostat', 'sar', 'cloudwatch']
    }
}
\`\`\`

### Linux Performance Observability Tools

\`\`\`bash
# Performance tools overview
+-------------------+------------------------+
| Tool              | Primary Use            |
+-------------------+------------------------+
| top/htop          | Overall system view    |
| vmstat            | System-wide stats      |
| iostat            | Disk I/O              |
| mpstat            | CPU per-core          |
| pidstat           | Per-process stats     |
| sar               | Historical data        |
| netstat/ss        | Network connections    |
| iotop             | Disk I/O by process   |
| perf              | CPU profiling         |
| strace            | System call tracing    |
+-------------------+------------------------+
\`\`\`

## CPU Monitoring and Analysis

### Using top for Real-Time Monitoring

\`\`\`bash
# Basic top usage
top

# Top header interpretation:
# top - 14:23:45 up 10 days,  3:45,  2 users,  load average: 2.34, 1.98, 1.75
#       ^^^^^^^^                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#       current time                            load averages: 1min, 5min, 15min

# Load average interpretation:
# For a 4-core system:
# < 4.0:  System is not fully utilized
# = 4.0:  System is perfectly utilized
# > 4.0:  System is overloaded (processes waiting)

# Top display columns:
# PID    USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND
# 12345  www-data  20   0  500000  150000  10000 R  25.0   3.7   10:25.32 nginx

# Column meanings:
# PID:   Process ID
# USER:  Process owner
# PR:    Priority (lower = higher priority)
# NI:    Nice value (-20 to 19)
# VIRT:  Virtual memory (KB)
# RES:   Resident memory (physical RAM in use)
# SHR:   Shared memory
# S:     State (R=running, S=sleeping, D=uninterruptible, Z=zombie)
# %CPU:  CPU usage percentage
# %MEM:  Memory usage percentage
# TIME+: Cumulative CPU time
# COMMAND: Process name

# Useful top commands (while running):
# 1      - Show individual CPU cores
# P      - Sort by CPU usage (default)
# M      - Sort by memory usage
# T      - Sort by time (cumulative CPU time)
# k      - Kill a process
# r      - Renice (change priority)
# c      - Show full command path
# f      - Select which columns to display
# H      - Show threads
# u      - Filter by user

# Batch mode (for logging)
top -b -n 1 > top-snapshot.txt

# Monitor specific user
top -u www-data

# Set update interval
top -d 5  # Update every 5 seconds
\`\`\`

### Using htop (Enhanced top)

\`\`\`bash
# Install htop
sudo yum install htop -y  # Amazon Linux
sudo apt-get install htop -y  # Ubuntu

# Run htop
htop

# htop advantages over top:
# - Color-coded display
# - Mouse support
# - Easier process tree view
# - Better visual representation
# - Easier to kill/renice processes

# htop keyboard shortcuts:
# F2    - Setup (customize display)
# F3    - Search for process
# F4    - Filter processes
# F5    - Tree view
# F6    - Sort by column
# F9    - Kill process
# F10   - Quit
# Space - Tag process
# U     - Show processes for specific user
# t     - Tree view toggle
# H     - Hide/show threads
# K     - Show kernel threads

# htop bars interpretation:
# CPU bar colors:
# - Blue:   Low priority (nice > 0)
# - Green:  Normal user processes
# - Red:    Kernel processes
# - Yellow: IRQ time
# - Magenta: Soft IRQ
# - Grey:   IO-wait

# Memory bar colors:
# - Green:  Used memory
# - Blue:   Buffer memory
# - Yellow: Cache memory
\`\`\`

### CPU Statistics with mpstat

\`\`\`bash
# Install sysstat package (includes mpstat, iostat, sar)
sudo yum install sysstat -y

# View per-CPU statistics
mpstat -P ALL 1

# Output explanation:
# CPU    %usr   %nice    %sys %iowait    %irq   %soft  %steal  %guest  %gnice   %idle
# all    25.0    0.0    5.0    10.0     0.0     0.5     0.0     0.0     0.0    59.5
#   0    30.0    0.0    6.0     8.0     0.0     0.3     0.0     0.0     0.0    55.7
#   1    20.0    0.0    4.0    12.0     0.0     0.7     0.0     0.0     0.0    63.3

# Column meanings:
# %usr:    User space CPU usage (application code)
# %nice:   Nice'd process CPU usage
# %sys:    Kernel space CPU usage (system calls)
# %iowait: Waiting for I/O (disk, network)
# %irq:    Hardware interrupt handling
# %soft:   Software interrupt handling
# %steal:  Stolen by hypervisor (in VMs - important for EC2!)
# %guest:  Running virtual CPUs
# %idle:   Idle time

# Key indicators:
# High %iowait (> 20%):     Disk bottleneck
# High %sys (> 20%):        Kernel overhead (syscalls, context switching)
# High %steal (> 10%):      EC2 instance too small or neighbor noise
# High %usr with low idle:  CPU-bound workload

# Monitor specific CPU core
mpstat -P 0 1  # Monitor CPU 0

# Continuous monitoring every 5 seconds
mpstat -P ALL 5

# AWS EC2 specific: Monitor steal time
# Steal time indicates your instance is waiting for CPU from the hypervisor
# Consistently high steal (>5-10%) means:
# 1. Instance type is too small
# 2. Noisy neighbor issue
# 3. Consider upgrading instance size
watch -n 5 'mpstat -P ALL 1 1 | grep -E "Average|%steal"'
\`\`\`

### Identifying CPU-Bound Processes

\`\`\`bash
# Top CPU consumers (live)
ps aux --sort=-%cpu | head -10

# Top CPU consumers with full command
ps aux --sort=-%cpu | head -10 | awk '{printf "%-8s %-6s %-6s %-50s\\n", $1, $2, $3, substr($0, index($0,$11))}'

# Find processes using > 50% CPU
ps aux | awk '$3 > 50.0 {print $0}'

# CPU usage per user
ps aux | awk '{cpu[$1]+=$3} END {for (user in cpu) printf "%-15s %6.2f%%\\n", user, cpu[user]}' | sort -k2 -rn

# Identify CPU-intensive processes with their threads
ps -eLo pid,tid,user,comm,%cpu --sort=-%cpu | head -20

# Real-world debugging script
cat << 'EOF' > /usr/local/bin/check-cpu-hogs.sh
#!/bin/bash
# Identify and log CPU-intensive processes

THRESHOLD=80
OUTPUT_FILE="/var/log/cpu-hogs.log"

CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\\([0-9.]*\\)%* id.*/\\1/" | awk '{print 100 - $1}')

if (( $(echo "$CPU_USAGE > $THRESHOLD" | bc -l) )); then
    echo "$(date): CPU usage \${CPU_USAGE}% exceeds threshold" | tee -a "$OUTPUT_FILE"
    echo "Top 10 CPU consumers:" | tee -a "$OUTPUT_FILE"
    ps aux --sort=-%cpu | head -11 | tee -a "$OUTPUT_FILE"
    echo "---" | tee -a "$OUTPUT_FILE"
fi
EOF

chmod +x /usr/local/bin/check-cpu-hogs.sh

# Run via cron every 5 minutes
# */5 * * * * /usr/local/bin/check-cpu-hogs.sh
\`\`\`

## Memory Monitoring and Analysis

### Understanding Linux Memory Management

\`\`\`bash
# View memory usage
free -h

# Output interpretation:
#               total        used        free      shared  buff/cache   available
# Mem:           15Gi       8.0Gi       1.0Gi       100Mi       6.0Gi        6.5Gi
# Swap:         8.0Gi       500Mi       7.5Gi

# Column meanings:
# total:      Total installed RAM
# used:       Memory used by processes (excluding buffers/cache)
# free:       Completely unused memory
# shared:     Shared memory (tmpfs, etc.)
# buff/cache: Buffers (metadata) + Cache (file content)
# available:  Memory available for starting new applications
#             (includes reclaimable cache)

# Key insight: Don't panic if "free" is low!
# Linux uses free memory for cache (disk cache) to improve performance
# This cache is automatically released when needed
# Look at "available" instead of "free"

# Continuous monitoring
free -h -s 5  # Update every 5 seconds

# Show total, used, and free in MB
free -m

# Show memory in human-readable format with totals
free -h --total

# Check for memory pressure
cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|SwapFree"

# Memory usage by process
ps aux --sort=-%mem | head -10

# Detailed memory usage (RSS, VSZ, etc.)
ps -eo pid,user,rss,vsz,comm --sort=-rss | head -20

# RSS: Resident Set Size (actual physical RAM used)
# VSZ: Virtual Size (includes swapped, shared, and reserved memory)

# Memory usage per user
ps aux | awk '{mem[$1]+=$4} END {for (user in mem) printf "%-15s %6.2f%%\\n", user, mem[user]}' | sort -k2 -rn
\`\`\`

### Detecting Memory Leaks

\`\`\`bash
# Monitor memory usage of specific process over time
PID=12345

watch -n 5 "ps -p $PID -o pid,user,rss,vsz,cmd"

# Log memory growth
while true; do
    MEM=$(ps -p $PID -o rss= 2>/dev/null)
    if [ -n "$MEM" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S'),$MEM" >> /tmp/mem-monitor-$PID.csv
    fi
    sleep 60
done

# Analyze memory growth
# Plot the CSV with gnuplot or analyze for linear growth

# Check for memory leaks in running process
cat << 'EOF' > /usr/local/bin/detect-memory-leak.sh
#!/bin/bash
# Detect memory leaks by monitoring RSS growth

PID=$1
DURATION=\${2:- 3600}  # Monitor for 1 hour by default
INTERVAL = 60

if [-z "$PID"]; then
    echo "Usage: $0 <PID> [duration_seconds]"
    exit 1
fi

LOGFILE = "/tmp/memory-leak-check-$PID.log"
SAMPLES = ()

echo "Monitoring PID $PID for memory leaks (\${DURATION}s)"
echo "Timestamp,RSS_KB" > "$LOGFILE"

END_TIME = $(($(date +% s) + DURATION))

while [$(date +% s) - lt $END_TIME]; do
    if !ps - p $PID > /dev/null 2 >& 1; then
        echo "Process $PID has terminated"
        exit 1
fi

RSS = $(ps - p $PID - o rss = | tr - d ' ')
TIMESTAMP = $(date '+%Y-%m-%d %H:%M:%S')
    
    echo "$TIMESTAMP,$RSS" >> "$LOGFILE"
SAMPLES += ($RSS)
    
    sleep $INTERVAL
done

# Analyze results
FIRST = \${ SAMPLES[0] }
LAST = \${ SAMPLES[-1] }
GROWTH = $((LAST - FIRST))
GROWTH_PCT = $(awk "BEGIN {printf \\" % .2f\\", ($GROWTH/$FIRST)*100}")

echo ""
echo "=== Memory Leak Analysis ==="
echo "Duration: \${DURATION}s"
echo "Initial RSS: \${FIRST} KB"
echo "Final RSS: \${LAST} KB"
echo "Growth: \${GROWTH} KB (\${GROWTH_PCT}%)"

if (($(echo "$GROWTH_PCT > 10" | bc - l))); then
    echo "WARNING: Potential memory leak detected (\${GROWTH_PCT}% growth)"
else
    echo "Memory usage appears stable"
fi

echo "Detailed log: $LOGFILE"
EOF

chmod + x / usr / local / bin / detect - memory - leak.sh

# Usage
    / usr / local / bin / detect - memory - leak.sh 12345 3600  # Monitor PID 12345 for 1 hour
\`\`\`

### OOM Killer Analysis

\`\`\`bash
# Check if OOM killer has been invoked
dmesg | grep -i "out of memory"
grep -i "out of memory" /var/log/messages
grep -i "killed process" /var/log/messages

# View OOM killer history
journalctl -k | grep -i "killed process"

# Check OOM score for processes
# Lower score = less likely to be killed
# Higher score = more likely to be killed
for pid in $(ps -eo pid --no-headers); do
    if [ -f /proc/$pid/oom_score ]; then
        score=$(cat /proc/$pid/oom_score 2>/dev/null)
        cmd=$(ps -p $pid -o comm= 2>/dev/null)
        [ -n "$score" ] && printf "%-8s %-6s %s\\n" "$pid" "$score" "$cmd"
    fi
done | sort -k2 -rn | head -20

# Adjust OOM score for critical process
# Range: -1000 (never kill) to 1000 (kill first)
echo -500 | sudo tee /proc/12345/oom_score_adj

# Make it permanent via systemd
sudo mkdir -p /etc/systemd/system/myapp.service.d/
cat << EOF | sudo tee /etc/systemd/system/myapp.service.d/oom.conf
[Service]
OOMScoreAdjust=-500
EOF

sudo systemctl daemon-reload
sudo systemctl restart myapp
\`\`\`

## Disk I/O Monitoring

### Using iostat for Disk Performance

\`\`\`bash
# Basic iostat usage
iostat

# Detailed disk statistics
iostat -x

# Output interpretation:
# Device  r/s    w/s   rkB/s   wkB/s  rrqm/s  wrqm/s  %rrqm  %wrqm r_await w_await aqu-sz rareq-sz wareq-sz svctm %util
# xvda   10.0   50.0   100.0   500.0    0.5     2.0    4.8    3.8    5.0   15.0   0.75     10.0     10.0   2.0  95.0

# Key columns:
# r/s, w/s:    Reads/writes per second
# rkB/s, wkB/s: KB read/written per second
# await:       Average wait time (ms) for I/O requests
# %util:       Device utilization (0-100%)

# Critical thresholds:
# %util > 80%:   Disk is saturated
# await > 10ms:  High latency (for SSDs)
# await > 20ms:  High latency (for HDDs)

# Monitor specific device
iostat -x xvda 5

# Extended statistics with timestamps
iostat -x -t 5

# Monitor all devices every 5 seconds
iostat -x -h 5

# AWS EBS-specific monitoring
# NVMe devices on newer instance types
iostat -x nvme0n1 5

# Check if EBS volume is performing as expected
# For gp3: Up to 16,000 IOPS and 1,000 MB/s
# For io2: Provisioned IOPS
cat << 'EOF' > /usr/local/bin/check-ebs-performance.sh
#!/bin/bash
# Monitor EBS volume performance

DEVICE=$1
EXPECTED_IOPS=\${2: -16000}  # gp3 default
DURATION = 60

if [-z "$DEVICE"]; then
    echo "Usage: $0 <device> [expected_iops]"
    exit 1
fi

echo "Monitoring $DEVICE for \${DURATION}s"

# Collect samples
SAMPLES = ()
for i in $(seq 1 $DURATION); do
    IOPS = $(iostat - x $DEVICE 1 2 | tail - 1 | awk '{print $4 + $5}')
    SAMPLES += ($IOPS)
    sleep 1
done

# Calculate average
AVG = $(printf '%s\\n' "\${SAMPLES[@]}" | awk '{sum+=$1} END {print sum/NR}')

echo "Average IOPS: $AVG"
echo "Expected IOPS: $EXPECTED_IOPS"

if (($(echo "$AVG < $EXPECTED_IOPS * 0.8" | bc - l))); then
    echo "WARNING: IOPS below 80% of expected"
fi
EOF

chmod + x / usr / local / bin / check - ebs - performance.sh
\`\`\`

### Using iotop for Per-Process I/O

\`\`\`bash
# Install iotop
sudo yum install iotop -y

# Run iotop (requires root)
sudo iotop

# Key columns:
# TID:     Thread ID
# PRIO:    I/O priority
# USER:    Process owner
# DISK READ: Disk read rate
# DISK WRITE: Disk write rate
# SWAPIN:  Swap in percentage
# IO:      I/O percentage
# COMMAND: Process name

# Show only processes doing I/O
sudo iotop -o

# Batch mode (for logging)
sudo iotop -b -n 10 > iotop-snapshot.txt

# Show accumulated I/O
sudo iotop -a

# Show specific process
sudo iotop -p 12345

# Identify I/O hogs
sudo iotop -o -b -n 1 | tail -20

# Alternative: Use pidstat for I/O monitoring
pidstat -d 5  # Disk I/O statistics every 5 seconds
\`\`\`

### Disk Space Monitoring

\`\`\`bash
# Check disk usage
df -h

# Check disk usage with inodes
df -hi

# Find large directories
du -h --max-depth=1 /var | sort -hr | head -20

# Find large files
find / -xdev -type f -size +100M -exec ls -lh {} \\; 2>/dev/null | \\
    awk '{print $5, $9}' | sort -hr | head -20

# Monitor disk usage in real-time
watch -n 5 'df -h'

# Automated disk space alert
cat << 'EOF' > /usr/local/bin/check-disk-space.sh
#!/bin/bash
# Alert on high disk usage

THRESHOLD=80
EMAIL="ops@company.com"

df -h | grep -vE '^Filesystem|tmpfs|cdrom' | awk '{print $5 " " $1 " " $6}' | \\
while read output; do
    usage=$(echo $output | awk '{print $1}' | sed 's/%//')
    partition=$(echo $output | awk '{print $2}')
    mount=$(echo $output | awk '{print $3}')
    
    if [ $usage -ge $THRESHOLD ]; then
        echo "Disk usage alert: $partition mounted on $mount is \${usage}%"
        # Send alert (integrate with your alerting system)
    fi
done
EOF

chmod +x /usr/local/bin/check-disk-space.sh

# Add to cron
# */15 * * * * /usr/local/bin/check-disk-space.sh
\`\`\`

## Network Monitoring

### Using netstat and ss

\`\`\`bash
# Show all connections
netstat -a

# Show TCP connections
netstat -t

# Show listening ports
netstat -tln

# Show with process info (requires root)
sudo netstat -tlnp

# Show connection statistics
netstat -s

# Show routing table
netstat -r

# Modern alternative: ss (socket statistics)
# ss is faster and more feature-rich than netstat

# Show all TCP connections
ss -ta

# Show listening TCP ports with process info
sudo ss -tlnp

# Show established connections
ss -ta state established

# Show connection statistics
ss -s

# Find which process is using port 80
sudo ss -tlnp | grep :80

# Show connections to specific host
ss -ta dst 10.0.1.50

# Show connections with timers
ss -to

# Real-world: Find connections in TIME_WAIT state
ss -ta state time-wait | wc -l

# If TIME_WAIT count is high (>1000), may need tuning:
# sysctl -w net.ipv4.tcp_fin_timeout=15
# sysctl -w net.ipv4.tcp_tw_reuse=1
\`\`\`

### Network Bandwidth Monitoring

\`\`\`bash
# Install iftop
sudo yum install iftop -y

# Monitor bandwidth by connection
sudo iftop -i eth0

# Show bandwidth in MB/s
sudo iftop -i eth0 -B

# Alternative: nload
sudo yum install nload -y
sudo nload eth0

# Alternative: nethogs (per-process bandwidth)
sudo yum install nethogs -y
sudo nethogs eth0

# Check network interface statistics
ifconfig eth0
ip -s link show eth0

# Monitor network errors
netstat -i
# Look for errors, drops, overruns

# AWS CloudWatch network metrics
# Monitor NetworkIn, NetworkOut, NetworkPacketsIn, NetworkPacketsOut
aws cloudwatch get-metric-statistics \\
    --namespace AWS/EC2 \\
    --metric-name NetworkIn \\
    --dimensions Name=InstanceId,Value=i-1234567890abcdef0 \\
    --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \\
    --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \\
    --period 300 \\
    --statistics Average,Maximum
\`\`\`

## Process Monitoring

### Using ps for Process Analysis

\`\`\`bash
# Show all processes
ps aux

# Show process tree
ps auxf
pstree -p

# Show processes for specific user
ps -u www-data

# Show with custom columns
ps -eo pid,user,pri,ni,vsz,rss,%mem,%cpu,comm --sort=-%cpu

# Show threads
ps -eLf

# Monitor specific process
watch -n 5 'ps -p 12345 -o pid,user,%cpu,%mem,vsz,rss,comm'

# Find parent of zombie process
ps -el | grep Z

# Show process start time
ps -eo pid,etime,comm

# Show process with full command line
ps -ef
ps auxww

# Real-world: Find all Python processes and their memory
ps aux | grep python | awk '{sum+=$6; print $2, $6, $11}' END {print "Total:", sum, "KB"}'
\`\`\`

### Using pidstat for Detailed Process Stats

\`\`\`bash
# Install sysstat (includes pidstat)
sudo yum install sysstat -y

# CPU usage per process
pidstat 5

# Show specific process
pidstat -p 12345 5

# Memory statistics
pidstat -r 5

# I/O statistics
pidstat -d 5

# Context switches
pidstat -w 5

# Show threads
pidstat -t 5

# Page faults
pidstat -R 5

# All statistics combined
pidstat -rudw 5

# Save to file
pidstat 5 12 > pidstat.log  # 12 samples, 5 seconds apart
\`\`\`

## System-Wide Monitoring with vmstat and sar

### Using vmstat

\`\`\`bash
# Basic vmstat
vmstat 5  # Update every 5 seconds

# Output interpretation:
# procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----
#  r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
#  2  0  12345  54321  12345 234567    0    0    10    50  500 1000 25  5 60 10  0

# Key columns:
# Procs:
#   r: Processes waiting for CPU (runnable)
#   b: Processes in uninterruptible sleep (blocked on I/O)

# Memory (in KB):
#   swpd: Virtual memory used
#   free: Free memory
#   buff: Buffer memory
#   cache: Cache memory

# Swap:
#   si: Memory swapped in from disk (KB/s)
#   so: Memory swapped out to disk (KB/s)

# IO:
#   bi: Blocks received from block device (blocks/s)
#   bo: Blocks sent to block device (blocks/s)

# System:
#   in: Interrupts per second
#   cs: Context switches per second

# CPU:
#   us: User time
#   sy: System time
#   id: Idle time
#   wa: I/O wait
#   st: Steal time (hypervisor)

# Key indicators:
# r > # of CPUs:       CPU saturation
# b > 0 consistently: I/O bottleneck
# si/so > 0:          Memory pressure (swapping)
# wa > 20%:           I/O bottleneck
# st > 5%:            EC2 contention (noisy neighbor)

# Show with timestamps
vmstat -t 5

# Show in MB instead of KB
vmstat -S M 5

# Show summary since boot
vmstat -s

# Show disk statistics
vmstat -d

# Show partition statistics
vmstat -p xvda1
\`\`\`

### Using sar for Historical Analysis

\`\`\`bash
# Enable sar data collection
sudo systemctl enable sysstat
sudo systemctl start sysstat

# View CPU usage from today
sar -u

# View CPU usage from specific day
sar -u -f /var/log/sa/sa28  # sa28 = 28th day of month

# View memory usage
sar -r

# View swap usage
sar -S

# View I/O statistics
sar -b

# View network statistics
sar -n DEV

# View load average
sar -q

# View all statistics
sar -A

# Specify time range
sar -u -s 10:00:00 -e 12:00:00

# Export to file
sar -u -o /tmp/sar-cpu.dat

# Generate report from file
sar -u -f /tmp/sar-cpu.dat

# Real-world: Find peak CPU times
sar -u | awk '$3 == "CPU" || $3 > 80 {print}'

# Automated performance report
cat << 'EOF' > /usr/local/bin/daily-perf-report.sh
#!/bin/bash
# Generate daily performance report

DATE=$(date +%Y%m%d)
REPORT="/var/log/performance-report-$DATE.txt"

{
    echo "Performance Report for $(date +%Y-%m-%d)"
    echo "========================================="
    echo ""
    
    echo "CPU Usage (Peak Times):"
    sar -u | tail -20
    echo ""
    
    echo "Memory Usage:"
    sar -r | tail -10
    echo ""
    
    echo "I/O Statistics:"
    sar -b | tail -10
    echo ""
    
    echo "Network Statistics:"
    sar -n DEV | grep -v lo | tail -10
    echo ""
    
    echo "Load Average:"
    sar -q | tail -10
} > "$REPORT"

echo "Performance report generated: $REPORT"
EOF

chmod +x /usr/local/bin/daily-perf-report.sh
\`\`\`

## Performance Profiling with perf

\`\`\`bash
# Install perf
sudo yum install perf -y

# Profile CPU usage (system-wide)
sudo perf top

# Record performance data
sudo perf record -a -g sleep 60  # Record for 60 seconds

# View recorded data
sudo perf report

# Profile specific process
sudo perf record -p 12345 -g sleep 30

# CPU cache misses
sudo perf stat -e cache-misses,cache-references ./myapp

# System-wide CPU cycles
sudo perf stat -a sleep 10

# Flame graph generation (requires flamegraph tools)
sudo perf record -a -g -F 99 sleep 60
sudo perf script > out.perf
./flamegraph.pl out.perf > flamegraph.svg
\`\`\`

## AWS CloudWatch Integration

\`\`\`bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
sudo rpm -U ./amazon-cloudwatch-agent.rpm

# Configure CloudWatch agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard

# Or create config manually
cat << 'EOF' | sudo tee /opt/aws/amazon-cloudwatch-agent/etc/config.json
{
  "metrics": {
    "namespace": "CustomMetrics",
    "metrics_collected": {
      "cpu": {
        "measurement": [
          {"name": "cpu_usage_idle", "rename": "CPU_IDLE"},
          {"name": "cpu_usage_iowait", "rename": "CPU_IOWAIT"}
        ],
        "metrics_collection_interval": 60,
        "totalcpu": false
      },
      "disk": {
        "measurement": [
          {"name": "used_percent", "rename": "DISK_USED"}
        ],
        "metrics_collection_interval": 60,
        "resources": ["*"]
      },
      "mem": {
        "measurement": [
          {"name": "mem_used_percent", "rename": "MEM_USED"}
        ],
        "metrics_collection_interval": 60
      }
    }
  },
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/application.log",
            "log_group_name": "/aws/ec2/application",
            "log_stream_name": "{instance_id}"
          }
        ]
      }
    }
  }
}
EOF

# Start CloudWatch agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \\
    -a fetch-config \\
    -m ec2 \\
    -s \\
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/config.json
\`\`\`

## Performance Troubleshooting Methodology

\`\`\`bash
#!/bin/bash
# Performance troubleshooting script

cat << 'EOF' > /usr/local/bin/diagnose-performance.sh
#!/bin/bash
# Quick performance diagnostic

OUTPUT="/tmp/perf-diagnostic-$(date +%Y%m%d-%H%M%S).txt"

{
    echo "Performance Diagnostic Report"
    echo "Generated: $(date)"
    echo "====================================="
    
    echo -e "\\n1. Load Average:"
    uptime
    
    echo -e "\\n2. CPU Usage:"
    mpstat 1 5
    
    echo -e "\\n3. Memory Usage:"
    free -h
    
    echo -e "\\n4. Disk I/O:"
    iostat -x 1 5
    
    echo -e "\\n5. Top Processes (CPU):"
    ps aux --sort=-%cpu | head -10
    
    echo -e "\\n6. Top Processes (Memory):"
    ps aux --sort=-%mem | head -10
    
    echo -e "\\n7. Network Connections:"
    ss -s
    
    echo -e "\\n8. Disk Usage:"
    df -h
    
    echo -e "\\n9. System Messages (Errors):"
    dmesg | tail -20
    
    echo -e "\\n10. OOM Events:"
    grep -i "out of memory" /var/log/messages | tail -5
    
} | tee "$OUTPUT"

echo -e "\\nDiagnostic report saved to: $OUTPUT"
EOF

chmod +x /usr/local/bin/diagnose-performance.sh
\`\`\`

## Best Practices

✅ **Monitor proactively** - Don't wait for issues  
✅ **Set up alerts** - CloudWatch alarms for thresholds  
✅ **Trend analysis** - Use sar for historical patterns  
✅ **Baseline performance** - Know your normal metrics  
✅ **Monitor the right metrics** - Focus on Four Golden Signals  
✅ **Automate monitoring** - Scripts for regular checks  
✅ **Document investigations** - Keep runbooks updated  
✅ **Test monitoring** - Verify alerts work  
✅ **Use proper tools** - Right tool for the job  
✅ **Integrate with CloudWatch** - Centralized monitoring

## Key Takeaways

1. **Load average** reflects system saturation, compare to CPU count
2. **High I/O wait** indicates disk bottleneck, not CPU
3. **Steal time** on EC2 indicates hypervisor contention
4. **Memory cache** is normal and beneficial, look at "available"
5. **Context switches** and interrupts indicate system overhead
6. **OOM killer** is last resort when memory exhausted
7. **Network TIME_WAIT** accumulation may need tuning
8. **Historical data** (sar) essential for trend analysis

## Next Steps

In the next section, we'll dive into **Storage & File Systems**, learning how to manage EBS volumes, configure RAID, optimize file system performance, and implement production backup strategies.`,
};
