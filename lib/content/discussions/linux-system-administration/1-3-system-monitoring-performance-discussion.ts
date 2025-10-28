export const systemMonitoringPerformanceDiscussion = [
  {
    id: 1,
    question:
      "Your production API is experiencing intermittent slowness. Users report that requests sometimes take 5-10 seconds instead of the normal 200ms. When you check the server, CPU usage is only 30% and memory is at 50%. The application logs don't show errors. Walk through your complete troubleshooting methodology using Linux performance tools. What would you check, in what order, and why? Include the specific commands you'd run and how you'd interpret the results.",
    answer: `## Comprehensive Troubleshooting Approach:

### Initial Hypothesis

With low CPU (30%) and moderate memory (50%), but slow response times, the issue is likely:
1. **Disk I/O bottleneck** (most likely)
2. **Network issues**
3. **Lock contention** (database, file locks)
4. **External dependencies** (slow third-party APIs)
5. **Context switching overhead**

### Step-by-Step Investigation

**Step 1: Check load average and I/O wait**

\`\`\`bash
# First command to run
uptime
# load average: 8.5, 7.2, 6.5

# With only 4 CPUs and 30% CPU usage, load average of 8.5 is very high!
# This indicates processes are waiting for resources (likely I/O)

# Confirm with vmstat
vmstat 1 10

# Output:
# procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----
#  r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
#  1  7      0  8000000  50000 4000000   0    0  5000 25000  800 1200 15  5 10 70  0
#     ^^                                           ^^^^^^^^             ^^^^^^^^^^
#     7 blocked                                    High I/O             70% I/O wait!

# Analysis:
# - r=1: Only 1 process waiting for CPU (normal)
# - b=7: 7 processes in uninterruptible sleep (WAITING FOR I/O!)
# - wa=70%: 70% of time spent waiting for I/O
# - bi/bo: High block in/out rates

# Conclusion: DISK I/O BOTTLENECK CONFIRMED
\`\`\`

**Step 2: Identify which disk is saturated**

\`\`\`bash
# Check disk utilization
iostat -x 1 5

# Output:
# Device   r/s   w/s  rkB/s  wkB/s  await  svctm  %util
# xvda     5.0   20.0   100    500    5.0    2.0   15.0    <- Root vol OK
# xvdf   150.0  800.0  3000  16000  250.0   50.0  100.0    <- DATA VOLUME SATURATED!

# Key findings:
# xvdf (data volume):
# - %util: 100% (fully saturated!)
# - await: 250ms (extremely high latency)
# - w/s: 800 writes/second
# - Problem: Disk can't keep up with write load

# Check what this device is mounted as
df -h | grep xvdf
# /dev/xvdf  500G  450G  50G  90%  /var/lib/mysql

# AHA! MySQL database volume is saturated
\`\`\`

**Step 3: Find which processes are doing I/O**

\`\`\`bash
# Identify processes doing heavy I/O
sudo iotop -o -b -n 3

# Output shows:
# TID  PRIO  USER     DISK READ  DISK WRITE  SWAPIN     IO    COMMAND
# 1234  be/4 mysql        0.00 B  15000.00 K  0.00 %  50.00 % mysql
# 1235  be/4 mysql        0.00 B  12000.00 K  0.00 %  45.00 % mysql
# 1236  be/4 mysql        0.00 B  10000.00 K  0.00 %  40.00 % mysql

# MySQL is doing MASSIVE write operations

# Check specific I/O patterns with pidstat
pidstat -d 2 5

# Look for:
# kB_wr/s (KB written per second)
# kB_ccwr/s (KB cancelled writes - indicates inefficiency)
\`\`\`

**Step 4: Investigate MySQL/Application behavior**

\`\`\`bash
# Connect to MySQL and check current queries
mysql -u root -p

# Show running queries
SHOW FULL PROCESSLIST;

# Look for:
# - Long-running queries
# - Queries in "Writing to tmp disk" state
# - UPDATE/INSERT queries without WHERE clause
# - Missing indexes causing table scans

# Check MySQL slow query log
tail -f /var/log/mysql/slow.log

# Example problematic query:
# Query_time: 8.5  Lock_time: 0.1  Rows_sent: 0  Rows_examined: 10000000
# UPDATE users SET last_seen = NOW();  <- NO WHERE CLAUSE!

# This query:
# 1. Scans entire table (10M rows)
# 2. Updates every row
# 3. Writes massive amounts of data to disk
# 4. Blocks other queries waiting for disk I/O

# Check MySQL status for disk writes
SHOW GLOBAL STATUS LIKE 'Innodb_data_written';
# Innodb_data_written: 50000000000  (50GB!)

SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_wait_free';
# If this is high, MySQL is waiting for buffer pool to flush to disk
\`\`\`

**Step 5: Check file system and storage configuration**

\`\`\`bash
# Check EBS volume type and IOPS
aws ec2 describe-volumes \\
    --volume-ids vol-0123456789abcdef0 \\
    --query 'Volumes[0].{Type:VolumeType,IOPS:Iops,Throughput:Throughput}'

# Output:
# {
#     "Type": "gp2",           <- Old generation!
#     "IOPS": 1500,            <- Only 1500 IOPS (3 IOPS per GB × 500GB)
#     "Throughput": null
# }

# Problem identified:
# - gp2 volume (old type)
# - Only 1500 IOPS baseline
# - Application needs ~3000 IOPS (800 writes/s × burst factor)
# - Volume is consistently hitting IOPS limit

# Check if volume has burst credits
aws cloudwatch get-metric-statistics \\
    --namespace AWS/EBS \\
    --metric-name BurstBalance \\
    --dimensions Name=VolumeId,Value=vol-0123456789abcdef0 \\
    --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \\
    --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \\
    --period 300 \\
    --statistics Average

# BurstBalance: 0%
# Volume has exhausted burst credits and is throttled to baseline IOPS!

# Check file system mount options
mount | grep xvdf
# /dev/xvdf on /var/lib/mysql type ext4 (rw,relatime,data=ordered)

# data=ordered forces sequential writes, can impact performance
# Consider data=writeback for databases (slightly less safe but faster)
\`\`\`

**Step 6: Check for lock contention**

\`\`\`bash
# Check for processes in D state (uninterruptible sleep)
ps aux | awk '$8 == "D" {print}'

# These processes are stuck waiting for I/O

# Check MySQL lock waits
mysql -e "SHOW ENGINE INNODB STATUS\\G" | grep -A 20 "TRANSACTIONS"

# Look for:
# - Lock wait timeouts
# - Deadlocks
# - Long transaction times

# Example output showing lock waits:
# ---TRANSACTION 123456, ACTIVE 45 sec
# mysql tables in use 1, locked 1
# LOCK WAIT 2 lock struct(s), heap size 360
# MySQL thread id 789, OS thread handle 0x123, query id 456 localhost root Updating
# UPDATE users SET status = 'active' WHERE id = 12345
# ------- TRX HAS BEEN WAITING 45 SEC FOR THIS LOCK TO BE GRANTED:
\`\`\`

**Step 7: Monitor in real-time**

\`\`\`bash
# Create monitoring script
cat << 'SCRIPT' > /tmp/monitor-slowness.sh
#!/bin/bash

while true; do
    echo "=== $(date) ==="
    
    echo "Load Average:"
    uptime | awk -F'load average:' '{print $2}'
    
    echo -e "\\nI/O Wait:"
    vmstat 1 2 | tail -1 | awk '{print "wa: "$15"%"}'
    
    echo -e "\\nDisk Utilization:"
    iostat -x xvdf 1 2 | tail -1 | awk '{print "util: "$14"%", "await:", $10"ms"}'
    
    echo -e "\\nMySQL Active Queries:"
    mysql -e "SELECT COUNT(*) FROM INFORMATION_SCHEMA.PROCESSLIST WHERE COMMAND != 'Sleep'" -sN
    
    echo -e "\\nProcesses in D state:"
    ps aux | awk '$8 == "D"' | wc -l
    
    echo "---"
    sleep 5
done
SCRIPT

chmod +x /tmp/monitor-slowness.sh
./tmp/monitor-slowness.sh
\`\`\`

### Root Cause Analysis

**Primary Issue**: Disk I/O bottleneck caused by:

1. **Inefficient SQL Query**
   - UPDATE without WHERE clause
   - Full table scan of 10M rows
   - Every row written to disk

2. **Insufficient IOPS**
   - gp2 volume with only 1500 IOPS
   - Burst credits exhausted
   - Application needs 3000+ IOPS

3. **Suboptimal Configuration**
   - No MySQL query cache tuning
   - InnoDB buffer pool may be undersized
   - File system mount options not optimized

### Solutions (Immediate)

\`\`\`bash
# 1. Fix the problematic query
# Add WHERE clause to only update necessary rows
UPDATE users SET last_seen = NOW() WHERE last_seen < NOW() - INTERVAL 1 HOUR;

# Or batch the updates
UPDATE users SET last_seen = NOW() WHERE id BETWEEN 1 AND 10000;
# Repeat in batches

# 2. Add index if missing
CREATE INDEX idx_last_seen ON users(last_seen);

# 3. Optimize MySQL configuration
# Edit /etc/my.cnf
[mysqld]
innodb_buffer_pool_size = 12G  # 75% of RAM
innodb_flush_log_at_trx_commit = 2  # Flush every second (faster, slightly less durable)
innodb_flush_method = O_DIRECT  # Skip OS cache
innodb_io_capacity = 2000
innodb_io_capacity_max = 4000

# Restart MySQL
sudo systemctl restart mysql
\`\`\`

### Solutions (Short-term)

\`\`\`bash
# Upgrade EBS volume to gp3
aws ec2 modify-volume \\
    --volume-id vol-0123456789abcdef0 \\
    --volume-type gp3 \\
    --iops 16000 \\
    --throughput 1000

# gp3 benefits:
# - 16,000 IOPS (vs 1500 for gp2)
# - 1,000 MB/s throughput
# - No burst credits needed
# - Cost-effective

# Monitor the modification
aws ec2 describe-volumes-modifications \\
    --volume-ids vol-0123456789abcdef0

# After modification completes, extend file system if needed
sudo resize2fs /dev/xvdf
\`\`\`

### Solutions (Long-term)

\`\`\`bash
# 1. Implement read replicas for read-heavy queries
# Offload reads to replica, master handles writes only

# 2. Add caching layer
# Redis/ElastiCache for frequently accessed data
# Reduces database queries significantly

# 3. Query optimization
# Review all slow queries
# Add proper indexes
# Optimize schema design

# 4. Monitor proactively
# Set up CloudWatch alarms:

# Alert on high I/O wait
aws cloudwatch put-metric-alarm \\
    --alarm-name high-iowait \\
    --alarm-description "Alert when I/O wait exceeds 30%" \\
    --metric-name CPUIOWait \\
    --namespace CustomMetrics \\
    --statistic Average \\
    --period 300 \\
    --threshold 30 \\
    --comparison-operator GreaterThanThreshold \\
    --evaluation-periods 2

# Alert on EBS burst balance
aws cloudwatch put-metric-alarm \\
    --alarm-name low-burst-balance \\
    --metric-name BurstBalance \\
    --namespace AWS/EBS \\
    --statistic Average \\
    --period 300 \\
    --threshold 20 \\
    --comparison-operator LessThanThreshold \\
    --dimensions Name=VolumeId,Value=vol-0123456789abcdef0 \\
    --evaluation-periods 2
\`\`\`

### Verification

\`\`\`bash
# After fixes, verify improvement:

# 1. Check I/O wait
vmstat 1 10
# Should see wa < 10%

# 2. Check disk utilization
iostat -x xvdf 1 5
# Should see %util < 80%, await < 10ms

# 3. Check application response time
# Monitor API response times
# Should return to < 200ms

# 4. Check load average
uptime
# Should be close to CPU count (4)

# 5. Verify no processes in D state
ps aux | awk '$8 == "D"' | wc -l
# Should be 0 or very low
\`\`\`

### Key Lessons

1. **Low CPU doesn't mean no bottleneck** - Check I/O wait!
2. **Load average > CPU count** indicates processes waiting
3. **%util at 100%** means disk is saturated
4. **High await times** indicate slow disk response
5. **Processes in D state** are waiting for I/O
6. **gp2 burst credits** can be depleted, causing throttling
7. **Inefficient queries** can saturate even fast disks
8. **Proper indexing** is critical for database performance

This methodical approach starting with high-level metrics (load, I/O wait) and drilling down to specific processes and queries is essential for production troubleshooting.
`,
  },
  {
    id: 2,
    question:
      "You're running a high-traffic web application on EC2 instances. Over the past week, you've noticed that memory usage keeps growing until the OOM killer eventually terminates the application process. How would you diagnose whether this is a memory leak in the application code vs. legitimate growth due to increased traffic? Design a comprehensive monitoring and analysis strategy including specific tools, metrics to track, and criteria for determining root cause.",
    answer: `## Comprehensive Memory Analysis Strategy:

### Phase 1: Initial Data Collection

**Establish Baseline and Trends**

\`\`\`bash
# 1. Check historical memory usage patterns
# Use sar to view past memory trends
sar -r | tail -50

# Look for:
# - Linear growth over time (leak indicator)
# - Sawtooth pattern (normal GC behavior)
# - Correlation with traffic patterns

# 2. Check OOM killer history
sudo journalctl -k | grep -i "killed process" | tail -20

# Example output:
# Oct 28 03:45:23 kernel: Out of memory: Killed process 12345 (node) total-vm:8GB, anon-rss:7GB
# Oct 28 15:32:10 kernel: Out of memory: Killed process 12346 (node) total-vm:8.5GB, anon-rss:7.5GB
# Oct 29 02:15:45 kernel: Out of memory: Killed process 12347 (node) total-vm:9GB, anon-rss:8GB

# Analysis: Memory is growing over time (7GB → 7.5GB → 8GB)
# This suggests accumulation, but need to correlate with traffic

# 3. Check application uptime at time of OOM
# If OOM occurs after consistent uptime (e.g., always after 6-8 hours),
# this strongly suggests a leak

# 4. Review CloudWatch metrics
aws cloudwatch get-metric-statistics \\
    --namespace AWS/EC2 \\
    --metric-name MemoryUtilization \\
    --dimensions Name=InstanceId,Value=i-xxxxx \\
    --start-time $(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%S) \\
    --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \\
    --period 3600 \\
    --statistics Average,Maximum \\
    --query 'Datapoints[*].[Timestamp,Maximum]' \\
    --output table

# Plot this data to see trend
\`\`\`

**Correlate with Traffic Patterns**

\`\`\`bash
# Get request count from application logs
awk '{print $4}' /var/log/nginx/access.log | \\
    cut -d: -f1-2 | \\
    uniq -c | \\
    awk '{print $2"  "$1}' > /tmp/hourly-requests.txt

# Get memory usage per hour
sar -r | awk '/^[0-9]/ {print $1, 100-$5}' > /tmp/hourly-memory.txt

# Compare side by side
paste /tmp/hourly-requests.txt /tmp/hourly-memory.txt

# Analysis:
# Scenario A - Memory leak:
# Time     Requests  Memory%
# 08:00    1000      45%
# 09:00    1200      52%  <- Memory grows despite similar traffic
# 10:00    1100      58%  <- Continues growing
# 11:00    1050      65%  <- Still growing
# Conclusion: Memory leak - traffic doesn't explain growth

# Scenario B - Traffic-driven:
# Time     Requests  Memory%
# 08:00    500       30%
# 09:00    1200      45%  <- Memory grows with traffic
# 10:00    2000      65%  <- Proportional growth
# 11:00    1000      40%  <- Drops with traffic
# Conclusion: Traffic-driven - memory follows load
\`\`\`

### Phase 2: Detailed Memory Profiling

**Set Up Continuous Monitoring**

\`\`\`bash
# Create comprehensive monitoring script
cat << 'SCRIPT' > /usr/local/bin/monitor-memory-leak.sh
#!/bin/bash

APP_PID=$(pgrep -f "node server.js")  # Adjust for your app
LOGDIR="/var/log/memory-analysis"
LOGFILE="$LOGDIR/memory-$(date +%Y%m%d).csv"

mkdir -p "$LOGDIR"

# CSV header
if [ ! -f "$LOGFILE" ]; then
    echo "Timestamp,RSS_KB,VSZ_KB,USS_KB,PSS_KB,HeapUsed_MB,HeapTotal_MB,External_MB,Requests_Total,Requests_Active" > "$LOGFILE"
fi

while true; do
    if [ -z "$APP_PID" ] || ! kill -0 $APP_PID 2>/dev/null; then
        APP_PID=$(pgrep -f "node server.js")
        if [ -z "$APP_PID" ]; then
            echo "$(date): Application not running" >> "$LOGDIR/errors.log"
            sleep 60
            continue
        fi
    fi
    
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Get RSS and VSZ
    MEMORY=$(ps -p $APP_PID -o rss=,vsz= | awk '{print $1","$2}')
    RSS=$(echo $MEMORY | cut -d, -f1)
    VSZ=$(echo $MEMORY | cut -d, -f2)
    
    # Get USS and PSS (unique memory) using smem
    if command -v smem &> /dev/null; then
        USS_PSS=$(smem -P node -c "uss pss" -H | tail -1)
        USS=$(echo $USS_PSS | awk '{print $1}')
        PSS=$(echo $USS_PSS | awk '{print $2}')
    else
        USS=0
        PSS=0
    fi
    
    # Get Node.js heap stats (if Node.js app)
    if command -v node &> /dev/null; then
        HEAP_STATS=$(curl -s http://localhost:9229/json/version 2>/dev/null || echo "0 0 0")
        HEAP_USED=$(echo $HEAP_STATS | jq -r '.heapUsed // 0' 2>/dev/null || echo 0)
        HEAP_TOTAL=$(echo $HEAP_STATS | jq -r '.heapTotal // 0' 2>/dev/null || echo 0)
        EXTERNAL=$(echo $HEAP_STATS | jq -r '.external // 0' 2>/dev/null || echo 0)
    else
        HEAP_USED=0
        HEAP_TOTAL=0
        EXTERNAL=0
    fi
    
    # Get request metrics from application
    REQUESTS_TOTAL=$(curl -s http://localhost:8000/metrics | grep requests_total | awk '{print $2}' 2>/dev/null || echo 0)
    REQUESTS_ACTIVE=$(curl -s http://localhost:8000/metrics | grep requests_active | awk '{print $2}' 2>/dev/null || echo 0)
    
    # Log everything
    echo "$TIMESTAMP,$RSS,$VSZ,$USS,$PSS,$HEAP_USED,$HEAP_TOTAL,$EXTERNAL,$REQUESTS_TOTAL,$REQUESTS_ACTIVE" >> "$LOGFILE"
    
    sleep 60
done
SCRIPT

chmod +x /usr/local/bin/monitor-memory-leak.sh

# Run as systemd service
cat << 'EOF' | sudo tee /etc/systemd/system/memory-monitor.service
[Unit]
Description=Memory Leak Monitoring
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/monitor-memory-leak.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable memory-monitor
sudo systemctl start memory-monitor
\`\`\`

**Analyze Memory Composition**

\`\`\`bash
# For Node.js applications, enable heap dumps
# Add to application code:
const v8 = require('v8');
const fs = require('fs');

function writeHeapSnapshot() {
    const filename = \`heap-\${Date.now()}.heapsnapshot\`;
    const heapSnapshot = v8.writeHeapSnapshot(filename);
    console.log(\`Heap snapshot written to \${heapSnapshot}\`);
}

// Trigger on signal
process.on('SIGUSR2', writeHeapSnapshot);

# Take periodic heap snapshots
kill -SIGUSR2 <pid>

# Or automate:
while true; do
    sleep 3600  # Every hour
    kill -SIGUSR2 $(pgrep -f "node server.js")
done

# For Python applications, use memory_profiler
pip install memory_profiler

# Add decorator to functions
from memory_profiler import profile

@profile
def potentially_leaking_function():
    # Function code

# Run with profiling
python -m memory_profiler myapp.py

# For Java applications, use jmap
jmap -heap <pid>
jmap -dump:format=b,file=heap.bin <pid>

# Analyze with Eclipse MAT or VisualVM
\`\`\`

### Phase 3: Differential Analysis

**Compare Memory at Different Traffic Levels**

\`\`\`bash
#!/bin/bash
# Test memory behavior under controlled load

APP_URL="http://localhost:8000"
APP_PID=$(pgrep -f "node server.js")

echo "Starting differential memory analysis"

# Baseline: No load
echo "Phase 1: Measuring baseline (no load)"
sleep 300  # Let app stabilize
BASELINE_MEM=$(ps -p $APP_PID -o rss= | tr -d ' ')
echo "Baseline memory: $BASELINE_MEM KB"

# Load test 1: Moderate load
echo "Phase 2: Moderate load (100 req/s for 5 minutes)"
ab -n 30000 -c 10 -g moderate.tsv "$APP_URL/" &
AB_PID=$!
sleep 300
kill $AB_PID 2>/dev/null || true
MODERATE_MEM=$(ps -p $APP_PID -o rss= | tr -d ' ')
echo "After moderate load: $MODERATE_MEM KB"

# Wait for GC
echo "Phase 3: Waiting for GC (5 minutes idle)"
sleep 300
AFTER_GC_MEM=$(ps -p $APP_PID -o rss= | tr -d ' ')
echo "After GC: $AFTER_GC_MEM KB"

# Load test 2: Heavy load
echo "Phase 4: Heavy load (500 req/s for 5 minutes)"
ab -n 150000 -c 50 -g heavy.tsv "$APP_URL/" &
AB_PID=$!
sleep 300
kill $AB_PID 2>/dev/null || true
HEAVY_MEM=$(ps -p $APP_PID -o rss= | tr -d ' ')
echo "After heavy load: $HEAVY_MEM KB"

# Final wait
echo "Phase 5: Final GC wait (5 minutes)"
sleep 300
FINAL_MEM=$(ps -p $APP_PID -o rss= | tr -d ' ')
echo "Final memory: $FINAL_MEM KB"

# Analysis
echo ""
echo "=== Memory Analysis ==="
echo "Baseline:            $BASELINE_MEM KB"
echo "After moderate load: $MODERATE_MEM KB (+$(($MODERATE_MEM - $BASELINE_MEM)) KB)"
echo "After first GC:      $AFTER_GC_MEM KB (+$(($AFTER_GC_MEM - $BASELINE_MEM)) KB)"
echo "After heavy load:    $HEAVY_MEM KB (+$(($HEAVY_MEM - $BASELINE_MEM)) KB)"
echo "After final GC:      $FINAL_MEM KB (+$(($FINAL_MEM - $BASELINE_MEM)) KB)"

# Determine leak vs. growth
GROWTH_AFTER_GC=$(($AFTER_GC_MEM - $BASELINE_MEM))
FINAL_GROWTH=$(($FINAL_MEM - $BASELINE_MEM))

if [ $FINAL_GROWTH -gt $((BASELINE_MEM / 10)) ]; then
    echo ""
    echo "⚠️  POTENTIAL MEMORY LEAK DETECTED"
    echo "Memory grew by $(($FINAL_GROWTH * 100 / $BASELINE_MEM))% and did not return to baseline after GC"
else
    echo ""
    echo "✓ Memory behavior appears normal"
    echo "Memory returned close to baseline after GC"
fi
\`\`\`

**Analyze Memory Leak Patterns**

\`\`\`python
# Python script to analyze memory logs
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load memory data
df = pd.read_csv('/var/log/memory-analysis/memory-20241028.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Calculate memory growth rate
df['RSS_Growth'] = df['RSS_KB'].diff()
df['Time_Delta'] = df['Timestamp'].diff().dt.total_seconds() / 3600  # hours

# Linear regression on RSS over time
X = range(len(df))
y = df['RSS_KB'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

print(f"Memory Growth Rate: {slope:.2f} KB per sample")
print(f"R-squared (fit quality): {r_value**2:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation:
# - High R-squared (>0.9) with positive slope: Strong linear growth (LEAK!)
# - Low R-squared (<0.5): Random/traffic-driven (NOT a leak)
# - Negative correlation with traffic drop: Traffic-driven (NOT a leak)

# Plot memory vs. time
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(df['Timestamp'], df['RSS_KB'], label='RSS Memory')
plt.plot(df['Timestamp'], df['Requests_Total'] * 1000, label='Requests (scaled)', alpha=0.5)
plt.xlabel('Time')
plt.ylabel('Memory (KB)')
plt.legend()
plt.title('Memory Usage Over Time')

# Plot growth rate
plt.subplot(2, 1, 2)
plt.plot(df['Timestamp'][1:], df['RSS_Growth'][1:])
plt.xlabel('Time')
plt.ylabel('Memory Growth Rate (KB/sample)')
plt.title('Memory Growth Rate')
plt.axhline(y=0, color='r', linestyle='--')

plt.tight_layout()
plt.savefig('/var/log/memory-analysis/analysis.png')
print("Analysis plot saved to /var/log/memory-analysis/analysis.png")

# Leak indicators:
leak_score = 0
if slope > 100:  # Growing > 100KB per sample
    leak_score += 3
    print("❌ Significant positive memory growth detected")

if r_value**2 > 0.8:  # Strong linear fit
    leak_score += 3
    print("❌ Memory growth is highly linear (not random)")

if p_value < 0.01:  # Statistically significant
    leak_score += 2
    print("❌ Growth is statistically significant")

# Check if memory drops with traffic
traffic_correlation = df[['RSS_KB', 'Requests_Total']].corr().iloc[0, 1]
if traffic_correlation < 0.3:
    leak_score += 2
    print("❌ Weak correlation with traffic (suggests leak, not load)")

print(f"\\nLeak Score: {leak_score}/10")
if leak_score >= 7:
    print("🚨 HIGH LIKELIHOOD OF MEMORY LEAK")
elif leak_score >= 4:
    print("⚠️  POSSIBLE MEMORY LEAK - Investigate further")
else:
    print("✓ Memory behavior appears traffic-driven")
\`\`\`

### Phase 4: Application-Specific Debugging

**Node.js Heap Analysis**

\`\`\`bash
# Take heap snapshots at different memory levels
# Install clinic.js
npm install -g clinic

# Run with clinic
clinic doctor -- node server.js

# Or use built-in profiler
node --inspect server.js

# Connect Chrome DevTools to localhost:9229
# Take heap snapshots:
# 1. At startup
# 2. After processing 1000 requests
# 3. After processing 10000 requests

# Compare snapshots to find:
# - Objects that keep growing
# - Detached DOM nodes
# - Uncleared intervals/timers
# - Event listener leaks
# - Unclosed file handles

# Common Node.js leak patterns:

# 1. Global arrays that accumulate
const cache = [];  // BAD: Grows forever
app.get('/api/data', (req, res) => {
    cache.push(req.body);  // Never cleared!
    res.json({success: true});
});

# Fix: Use LRU cache with size limit
const LRU = require('lru-cache');
const cache = new LRU({max: 1000});

# 2. Event listeners not removed
const EventEmitter = require('events');
const emitter = new EventEmitter();

app.get('/api/stream', (req, res) => {
    emitter.on('data', (data) => {  // BAD: Listener never removed
        res.write(data);
    });
});

# Fix: Remove listener on connection close
app.get('/api/stream', (req, res) => {
    const handler = (data) => res.write(data);
    emitter.on('data', handler);
    res.on('close', () => emitter.removeListener('data', handler));
});

# 3. Timers not cleared
setInterval(() => {
    // Do something
}, 1000);  // BAD: Runs forever

# Fix: Clear on shutdown
const intervalId = setInterval(() => {}, 1000);
process.on('SIGTERM', () => clearInterval(intervalId));
\`\`\`

### Phase 5: Decision Framework

**Criteria for Leak vs. Legitimate Growth**

\`\`\`bash
# Decision Tree:

# Question 1: Does memory return to baseline after traffic drops?
# YES → Likely traffic-driven (check thresholds though)
# NO → Possible leak, continue investigation

# Question 2: Is memory growth linear with time?
# YES → Likely leak
# NO → Likely traffic-driven

# Question 3: Does memory correlate with request count?
# Strong correlation (r > 0.7) → Traffic-driven
# Weak correlation (r < 0.3) → Likely leak

# Question 4: Does memory stabilize after fixed number of requests?
# YES → Warm-up caching, not a leak
# NO → Continues growing, likely leak

# Question 5: Does GC reclaim memory?
# YES (significant reduction) → Not a leak
# NO (minimal reduction) → Memory is held, likely leak

# Create automated decision script
cat << 'EOF' > /usr/local/bin/leak-diagnosis.sh
#!/bin/bash

# Analyze memory logs and provide diagnosis
LOGFILE="/var/log/memory-analysis/memory-$(date +%Y%m%d).csv"

if [ ! -f "$LOGFILE" ]; then
    echo "No memory log found for today"
    exit 1
fi

# Calculate statistics
python3 << 'PYTHON'
import pandas as pd
import numpy as np
from scipy.stats import linregress

df = pd.read_csv("$LOGFILE")

# Memory growth rate
slope, _, r_value, p_value, _ = linregress(range(len(df)), df['RSS_KB'])

# Traffic correlation
traffic_corr = df[['RSS_KB', 'Requests_Total']].corr().iloc[0, 1]

# Memory volatility
volatility = df['RSS_KB'].std() / df['RSS_KB'].mean()

# Print results
print(f"Growth Rate: {slope:.2f} KB/sample")
print(f"Linear Fit (R²): {r_value**2:.4f}")
print(f"Traffic Correlation: {traffic_corr:.4f}")
print(f"Memory Volatility: {volatility:.4f}")

# Diagnosis
if slope > 50 and r_value**2 > 0.8 and traffic_corr < 0.4:
    print("\\n🚨 DIAGNOSIS: MEMORY LEAK DETECTED")
    print("Recommended: Restart application and investigate code")
elif traffic_corr > 0.7 and volatility > 0.1:
    print("\\n✓ DIAGNOSIS: Traffic-driven memory growth")
    print("Recommended: Scale horizontally or optimize memory usage")
else:
    print("\\n⚠️  DIAGNOSIS: Inconclusive")
    print("Recommended: Continue monitoring")
PYTHON
EOF

chmod +x /usr/local/bin/leak-diagnosis.sh
\`\`\`

### Summary: Leak vs. Legitimate Growth

**Memory Leak Indicators:**
- ✓ Linear growth over time regardless of traffic
- ✓ R² > 0.8 (strong linear fit)
- ✓ Low correlation with request count (r < 0.3)
- ✓ Memory does NOT drop after traffic decreases
- ✓ GC does not reclaim significant memory
- ✓ Consistent OOM after similar uptime
- ✓ Heap snapshots show growing object retention

**Legitimate Growth Indicators:**
- ✓ Memory correlates with traffic (r > 0.7)
- ✓ Memory drops when traffic decreases
- ✓ GC effectively reclaims memory
- ✓ Memory stabilizes after warm-up period
- ✓ Growth proportional to cached data size
- ✓ High memory volatility (scales up/down)

**Action Plan:**
1. Collect 24-48 hours of detailed memory metrics
2. Correlate with traffic patterns
3. Run differential load testing
4. Analyze with statistical methods
5. Take heap snapshots if leak suspected
6. Review code for common leak patterns
7. Implement limits (LRU caches, connection pools)
8. Set up automated monitoring and alerts

This comprehensive approach definitively distinguishes between leaks and legitimate memory growth.
`,
  },
  {
    id: 3,
    question:
      'Design a comprehensive real-time performance monitoring dashboard for a production environment running 20 microservices across 50 EC2 instances. The dashboard should display the Four Golden Signals (latency, traffic, errors, saturation) and provide drill-down capabilities. Describe the complete architecture including: data collection (what tools, what metrics, what frequency), data aggregation and storage, visualization strategy, and alerting rules. Include specific implementation commands and configuration for AWS services.',
    answer: `## Complete Monitoring Architecture:

This solution implements a production-grade monitoring system using Prometheus, Grafana, CloudWatch, and custom metrics collection.

### Architecture Overview

\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                    EC2 Instances (50)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Microservice │  │ Microservice │  │ Microservice │     │
│  │   + Exporter │  │   + Exporter │  │   + Exporter │ ... │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │              │
│    ┌────▼──────────────────▼──────────────────▼────┐       │
│    │         Node Exporter (System Metrics)         │       │
│    └────────────────────────┬────────────────────────┘      │
└─────────────────────────────┼───────────────────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │   Prometheus        │
                   │   (Metrics Storage) │
                   │   + Alert Manager   │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │      Grafana        │
                   │   (Visualization)   │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │   CloudWatch        │
                   │  (AWS Integration)  │
                   └─────────────────────┘
\`\`\`

### Phase 1: Metrics Collection

**Install Prometheus on Dedicated Monitoring Instance**

\`\`\`bash
# Launch monitoring instance (t3.large recommended)
aws ec2 run-instances \\
    --image-id ami-xxxxx \\
    --instance-type t3.large \\
    --key-name monitoring-key \\
    --security-group-ids sg-monitoring \\
    --subnet-id subnet-xxxxx \\
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=monitoring-prometheus}]'

# SSH to monitoring instance
ssh ec2-user@<monitoring-instance-ip>

# Install Prometheus
PROM_VERSION="2.45.0"
wget https://github.com/prometheus/prometheus/releases/download/v\${PROM_VERSION}/prometheus-\${PROM_VERSION}.linux-amd64.tar.gz
tar xvf prometheus-\${PROM_VERSION}.linux-amd64.tar.gz
sudo mv prometheus-\${PROM_VERSION}.linux-amd64 /opt/prometheus
sudo useradd --no-create-home --shell /bin/false prometheus
sudo chown -R prometheus:prometheus /opt/prometheus

# Create data directory
sudo mkdir -p /var/lib/prometheus
sudo chown prometheus:prometheus /var/lib/prometheus

# Configure Prometheus
cat << 'EOF' | sudo tee /opt/prometheus/prometheus.yml
global:
  scrape_interval: 15s  # Scrape targets every 15s
  evaluation_interval: 15s  # Evaluate rules every 15s
  external_labels:
    cluster: 'production'
    region: 'us-east-1'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - localhost:9093

# Load rules
rule_files:
  - "/opt/prometheus/rules/*.yml"

# Scrape configurations
scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Node Exporter (system metrics) - auto-discovery via EC2
  - job_name: 'node-exporter'
    ec2_sd_configs:
      - region: us-east-1
        port: 9100
        filters:
          - name: tag:Monitoring
            values: ['enabled']
          - name: instance-state-name
            values: ['running']
    relabel_configs:
      # Use private IP
      - source_labels: [__meta_ec2_private_ip]
        target_label: __address__
        replacement: \${1}:9100
      # Add instance name
      - source_labels: [__meta_ec2_tag_Name]
        target_label: instance_name
      # Add environment
      - source_labels: [__meta_ec2_tag_Environment]
        target_label: environment

  # Application metrics - auto-discovery
  - job_name: 'microservices'
    ec2_sd_configs:
      - region: us-east-1
        port: 8080
        filters:
          - name: tag:Monitoring
            values: ['enabled']
          - name: instance-state-name
            values: ['running']
    relabel_configs:
      - source_labels: [__meta_ec2_private_ip]
        target_label: __address__
        replacement: \${1}:8080
      - source_labels: [__meta_ec2_tag_Service]
        target_label: service
      - source_labels: [__meta_ec2_tag_Environment]
        target_label: environment

  # Custom exporter for business metrics
  - job_name: 'business-metrics'
    static_configs:
      - targets:
          - 'business-metrics-exporter:9091'
EOF

# Create systemd service
cat << 'EOF' | sudo tee /etc/systemd/system/prometheus.service
[Unit]
Description=Prometheus
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/opt/prometheus/prometheus \\
    --config.file=/opt/prometheus/prometheus.yml \\
    --storage.tsdb.path=/var/lib/prometheus/ \\
    --web.console.templates=/opt/prometheus/consoles \\
    --web.console.libraries=/opt/prometheus/console_libraries \\
    --storage.tsdb.retention.time=30d \\
    --web.enable-lifecycle
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable prometheus
sudo systemctl start prometheus
\`\`\`

**Install Node Exporter on All EC2 Instances**

\`\`\`bash
# Automated deployment script
cat << 'SCRIPT' > deploy-node-exporter.sh
#!/bin/bash
set -euo pipefail

# Deploy node_exporter to all instances
INSTANCES=$(aws ec2 describe-instances \\
    --filters "Name=tag:Environment,Values=production" \\
              "Name=instance-state-name,Values=running" \\
    --query 'Reservations[].Instances[].PrivateIpAddress' \\
    --output text)

NODE_EXPORTER_VERSION="1.6.1"
PARALLEL_JOBS=10

install_node_exporter() {
    local ip=$1
    
    ssh -o StrictHostKeyChecking=no ec2-user@$ip << 'EOF'
# Download node_exporter
wget https://github.com/prometheus/node_exporter/releases/download/v1.6.1/node_exporter-1.6.1.linux-amd64.tar.gz
tar xvf node_exporter-1.6.1.linux-amd64.tar.gz
sudo mv node_exporter-1.6.1.linux-amd64/node_exporter /usr/local/bin/
sudo useradd --no-create-home --shell /bin/false node_exporter

# Create systemd service
sudo tee /etc/systemd/system/node_exporter.service << 'SERVICE'
[Unit]
Description=Node Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=node_exporter
Group=node_exporter
Type=simple
ExecStart=/usr/local/bin/node_exporter \\
    --collector.filesystem.mount-points-exclude=^/(dev|proc|sys|var/lib/docker/.+)($|/) \\
    --collector.netclass.ignored-devices=^(veth.*|docker.*|br-.*|lo)$
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable node_exporter
sudo systemctl start node_exporter

# Add Monitoring tag
INSTANCE_ID=$(ec2-metadata --instance-id | cut -d" " -f2)
aws ec2 create-tags --resources $INSTANCE_ID --tags Key=Monitoring,Value=enabled

echo "Node exporter installed on $HOSTNAME"
EOF
}

export -f install_node_exporter

# Deploy in parallel
echo "$INSTANCES" | xargs -P $PARALLEL_JOBS -I {} bash -c "install_node_exporter {}"

echo "Node exporter deployed to all instances"
SCRIPT

chmod +x deploy-node-exporter.sh
./deploy-node-exporter.sh
\`\`\`

**Instrument Microservices with Prometheus Client**

\`\`\`javascript
// Node.js example - Add to each microservice
const express = require('express');
const promClient = require('prom-client');

const app = express();

// Create a Registry
const register = new promClient.Registry();

// Add default metrics (CPU, memory, etc.)
promClient.collectDefaultMetrics({ register });

// Custom metrics for Four Golden Signals

// 1. LATENCY - Histogram of request durations
const httpRequestDuration = new promClient.Histogram({
    name: 'http_request_duration_seconds',
    help: 'Duration of HTTP requests in seconds',
    labelNames: ['method', 'route', 'status_code'],
    buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]  // Response time buckets
});
register.registerMetric(httpRequestDuration);

// 2. TRAFFIC - Counter of requests
const httpRequestsTotal = new promClient.Counter({
    name: 'http_requests_total',
    help: 'Total number of HTTP requests',
    labelNames: ['method', 'route', 'status_code']
});
register.registerMetric(httpRequestsTotal);

// 3. ERRORS - Counter of errors
const httpRequestErrors = new promClient.Counter({
    name: 'http_request_errors_total',
    help: 'Total number of HTTP request errors',
    labelNames: ['method', 'route', 'error_type']
});
register.registerMetric(httpRequestErrors);

// 4. SATURATION - Gauge of active connections
const httpActiveConnections = new promClient.Gauge({
    name: 'http_active_connections',
    help: 'Number of active HTTP connections'
});
register.registerMetric(httpActiveConnections);

// Database connection pool saturation
const dbConnectionPoolSize = new promClient.Gauge({
    name: 'db_connection_pool_size',
    help: 'Database connection pool size',
    labelNames: ['state']  // active, idle, waiting
});
register.registerMetric(dbConnectionPoolSize);

// Middleware to track metrics
app.use((req, res, next) => {
    const start = Date.now();
    httpActiveConnections.inc();
    
    res.on('finish', () => {
        const duration = (Date.now() - start) / 1000;
        
        // Record latency
        httpRequestDuration.observe(
            { method: req.method, route: req.route?.path || req.path, status_code: res.statusCode },
            duration
        );
        
        // Record traffic
        httpRequestsTotal.inc({
            method: req.method,
            route: req.route?.path || req.path,
            status_code: res.statusCode
        });
        
        // Record errors
        if (res.statusCode >= 500) {
            httpRequestErrors.inc({
                method: req.method,
                route: req.route?.path || req.path,
                error_type: 'server_error'
            });
        }
        
        httpActiveConnections.dec();
    });
    
    next();
});

// Expose metrics endpoint
app.get('/metrics', async (req, res) => {
    res.set('Content-Type', register.contentType);
    res.end(await register.metrics());
});

// Monitor database pool (example with pg)
const pool = new Pool({
    max: 20,
    idleTimeoutMillis: 30000
});

setInterval(() => {
    dbConnectionPoolSize.set({ state: 'total' }, pool.totalCount);
    dbConnectionPoolSize.set({ state: 'idle' }, pool.idleCount);
    dbConnectionPoolSize.set({ state: 'waiting' }, pool.waitingCount);
}, 5000);

app.listen(8080);
\`\`\`

### Phase 2: Alerting Rules

\`\`\`yaml
# /opt/prometheus/rules/alerts.yml
groups:
  - name: golden_signals
    interval: 30s
    rules:
      # LATENCY - High p95 response time
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
          signal: latency
        annotations:
          summary: "High latency detected on {{ $labels.instance }}"
          description: "95th percentile response time is {{ $value }}s (threshold: 1s)"

      # TRAFFIC - Sudden traffic spike
      - alert: TrafficSpike
        expr: rate(http_requests_total[5m]) > 2 * avg_over_time(rate(http_requests_total[5m])[1h:5m])
        for: 5m
        labels:
          severity: warning
          signal: traffic
        annotations:
          summary: "Traffic spike detected on {{ $labels.instance }}"
          description: "Request rate {{ $value }} req/s is 2x normal"

      # TRAFFIC - Traffic drop (potential outage)
      - alert: TrafficDrop
        expr: rate(http_requests_total[5m]) < 0.2 * avg_over_time(rate(http_requests_total[5m])[1h:5m])
        for: 10m
        labels:
          severity: critical
          signal: traffic
        annotations:
          summary: "Traffic drop detected on {{ $labels.instance }}"
          description: "Request rate {{ $value }} req/s is significantly below normal"

      # ERRORS - High error rate
      - alert: HighErrorRate
        expr: rate(http_request_errors_total[5m]) / rate(http_requests_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
          signal: errors
        annotations:
          summary: "High error rate on {{ $labels.instance }}"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 1%)"

      # SATURATION - High CPU usage
      - alert: HighCPU
        expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 10m
        labels:
          severity: warning
          signal: saturation
        annotations:
          summary: "High CPU usage on {{ $labels.instance }}"
          description: "CPU usage is {{ $value }}% (threshold: 80%)"

      # SATURATION - High memory usage
      - alert: HighMemory
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85
        for: 10m
        labels:
          severity: warning
          signal: saturation
        annotations:
          summary: "High memory usage on {{ $labels.instance }}"
          description: "Memory usage is {{ $value }}% (threshold: 85%)"

      # SATURATION - High disk I/O
        - alert: HighDiskIO
        expr: rate(node_disk_io_time_seconds_total[5m]) > 0.8
        for: 10m
        labels:
          severity: warning
          signal: saturation
        annotations:
          summary: "High disk I/O on {{ $labels.instance }}"
          description: "Disk utilization is {{ $value | humanizePercentage }}"

      # SATURATION - Connection pool saturation
      - alert: ConnectionPoolSaturated
        expr: db_connection_pool_size{state="waiting"} > 5
        for: 5m
        labels:
          severity: critical
          signal: saturation
        annotations:
          summary: "Database connection pool saturated"
          description: "{{ $value }} connections waiting"

      # SATURATION - High network bandwidth
      - alert: HighNetworkBandwidth
        expr: rate(node_network_transmit_bytes_total[5m]) > 100000000  # 100 MB/s
        for: 10m
        labels:
          severity: warning
          signal: saturation
        annotations:
          summary: "High network bandwidth on {{ $labels.instance }}"
          description: "Network transmit rate: {{ $value | humanize }}B/s"
\`\`\`

### Phase 3: Grafana Dashboard

\`\`\`bash
# Install Grafana
sudo yum install -y https://dl.grafana.com/oss/release/grafana-10.1.0-1.x86_64.rpm
sudo systemctl enable grafana-server
sudo systemctl start grafana-server

# Access Grafana at http://<monitoring-ip>:3000
# Default credentials: admin/admin

# Add Prometheus data source
curl -X POST http://admin:admin@localhost:3000/api/datasources \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "Prometheus",
    "type": "prometheus",
    "url": "http://localhost:9090",
    "access": "proxy",
    "isDefault": true
  }'

# Create dashboard (save as dashboard.json)
cat << 'JSON' > golden-signals-dashboard.json
{
  "dashboard": {
    "title": "Four Golden Signals - Production Overview",
    "tags": ["golden-signals", "production"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Request Latency (p50, p95, p99)",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))",
            "legendFormat": "p50 - {{service}}"
          },
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))",
            "legendFormat": "p95 - {{service}}"
          },
          {
            "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))",
            "legendFormat": "p99 - {{service}}"
          }
        ],
        "yaxes": [{"format": "s"}],
        "alert": {
          "name": "High Latency Alert",
          "conditions": [
            {
              "evaluator": {"params": [1], "type": "gt"},
              "query": {"params": ["A", "5m", "now"]},
              "type": "query"
            }
          ]
        }
      },
      {
        "title": "Request Rate (Traffic)",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (service)",
            "legendFormat": "{{service}}"
          }
        ],
        "yaxes": [{"format": "reqps"}]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "sum(rate(http_request_errors_total[5m])) by (service) / sum(rate(http_requests_total[5m])) by (service) * 100",
            "legendFormat": "{{service}}"
          }
        ],
        "yaxes": [{"format": "percent"}],
        "thresholds": [{"value": 1, "colorMode": "critical"}]
      },
      {
        "title": "System Saturation",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "targets": [
          {
            "expr": "100 - (avg by(instance) (rate(node_cpu_seconds_total{mode='idle'}[5m])) * 100)",
            "legendFormat": "CPU - {{instance}}"
          },
          {
            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
            "legendFormat": "Memory - {{instance}}"
          }
        ],
        "yaxes": [{"format": "percent"}]
      }
    ]
  }
}
JSON

# Import dashboard
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \\
  -H "Content-Type: application/json" \\
  -d @golden-signals-dashboard.json
\`\`\`

### Phase 4: CloudWatch Integration

\`\`\`bash
# Export Prometheus metrics to CloudWatch
pip install prometheus-aws-cloudwatch

# Configure exporter
cat << 'EOF' > /opt/prometheus/cloudwatch-exporter.yml
region: us-east-1
namespace: CustomMetrics/MicroservicesPerformance
metrics:
  - name: http_request_duration_seconds
    type: histogram
    statistic: [p95, p99]
  - name: http_requests_total
    type: counter
  - name: http_request_errors_total
    type: counter
EOF

# Run exporter
prometheus-aws-cloudwatch \\
  --prometheus-url http://localhost:9090 \\
  --config /opt/prometheus/cloudwatch-exporter.yml

# Create CloudWatch dashboard
aws cloudwatch put-dashboard \\
  --dashboard-name MicroservicesPerformance \\
  --dashboard-body file://cloudwatch-dashboard.json
\`\`\`

### Summary

This comprehensive monitoring architecture provides:

1. **Metrics Collection**: Prometheus scraping from 50 instances every 15s
2. **Auto-Discovery**: EC2 service discovery for dynamic scaling
3. **Four Golden Signals**: Latency, Traffic, Errors, Saturation
4. **Real-Time Alerts**: Alertmanager with multi-channel notifications
5. **Visualization**: Grafana dashboards with drill-down capabilities
6. **Cloud Integration**: CloudWatch export for AWS-native tools
7. **Scalability**: Supports hundreds of instances
8. **Cost-Effective**: ~$200/month for monitoring infrastructure

**Total monthly cost**:
- Monitoring instance (t3.large): ~$60
- Prometheus storage (500GB EBS): ~$50
- CloudWatch custom metrics: ~$30
- Data transfer: ~$40
- Total: **~$180/month**

This provides enterprise-grade monitoring for production microservices at scale.
`,
  },
];
