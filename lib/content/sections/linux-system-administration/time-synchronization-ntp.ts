/**
 * Time Synchronization & NTP Section
 * Module: Linux System Administration & DevOps Foundations
 */

export const timeSynchronizationNtpSection = {
  id: 'time-synchronization-ntp',
  title: 'Time Synchronization & NTP',
  content: `# Time Synchronization & NTP

## Introduction

Accurate time synchronization is critical for distributed systems, logging, security, and compliance. Time drift can cause authentication failures, log correlation issues, database replication problems, and certificate validation errors. This section covers time management, NTP/chrony, timezone handling, and AWS Time Sync Service.

## Why Time Synchronization Matters

### Problems Caused by Time Drift

\`\`\`
1. **Authentication Failures**
   - Kerberos tickets invalid (±5 min tolerance)
   - SSL/TLS certificate validation fails
   - API token expiration issues
   - Two-factor authentication (TOTP) fails

2. **Distributed Systems**
   - Database replication conflicts
   - Distributed transactions fail
   - Cache invalidation issues
   - Message queue ordering problems

3. **Logging & Monitoring**
   - Can't correlate logs across servers
   - Metrics timestamps incorrect
   - Alert timing inaccurate
   - Audit trails unreliable

4. **Financial & Compliance**
   - Trading timestamps legally required
   - Audit logs must be accurate
   - PCI-DSS compliance requires time sync
   - HIPAA audit trails

5. **Debugging**
   - Unable to trace request flow
   - Can't determine event causality
   - Performance analysis inaccurate
\`\`\`

### Time Drift Example

\`\`\`bash
# Server time vs NTP server
date && ntpdate -q pool.ntp.org
# Mon Oct 28 10:15:23 UTC 2024
# server 192.168.1.1, stratum 2, offset -0.123456, delay 0.02345
# 28 Oct 10:15:23 ntpdate[1234]: adjust time server 192.168.1.1 offset -0.123456 sec

# offset -0.123456 sec = -123 ms drift
# Acceptable: < 100ms
# Warning: 100-500ms
# Critical: > 500ms
\`\`\`

## NTP (Network Time Protocol)

### NTP Basics

\`\`\`bash
# Install NTP
sudo yum install ntp  # RHEL/CentOS 7
sudo apt install ntp  # Ubuntu/Debian

# NTP configuration
sudo vi /etc/ntp.conf

# Use public NTP servers
server 0.pool.ntp.org iburst
server 1.pool.ntp.org iburst
server 2.pool.ntp.org iburst
server 3.pool.ntp.org iburst

# Or AWS Time Sync (169.254.169.123)
server 169.254.169.123 prefer iburst

# Restrict access
restrict default kod nomodify notrap nopeer noquery
restrict -6 default kod nomodify notrap nopeer noquery
restrict 127.0.0.1
restrict ::1

# Enable NTP
sudo systemctl enable ntpd
sudo systemctl start ntpd

# Check NTP status
ntpq -p
#      remote           refid      st t when poll reach   delay   offset  jitter
# ==============================================================================
# *time.aws.com    .GPS.            1 u   64   64  377    0.123  -0.045   0.012
# +pool-1.ntp.org  .CDMA.           1 u   32   64  377    5.234  -0.123   0.234

# * = current sync source
# + = candidate
# - = outlier
# x = falseticker

# Check synchronization
ntpstat
# synchronised to NTP server (169.254.169.123) at stratum 2
#    time correct to within 12 ms
#    polling server every 64 s
\`\`\`

## Chrony (Modern NTP Alternative)

### Why Chrony?

- Faster initial sync
- Better for virtual machines
- Better for intermittent network
- Better for mobile devices
- Used by default in RHEL 8+, Amazon Linux 2023

### Chrony Configuration

\`\`\`bash
# Install chrony
sudo yum install chrony  # RHEL/Amazon Linux
sudo apt install chrony  # Ubuntu/Debian

# Chrony configuration
sudo vi /etc/chrony.conf

# NTP servers (AWS Time Sync preferred on EC2)
server 169.254.169.123 prefer iburst minpoll 4 maxpoll 4

# Or public NTP pools
pool pool.ntp.org iburst maxsources 4
pool time.google.com iburst maxsources 4

# Allow large time corrections on startup
makestep 1.0 3

# Drift file
driftfile /var/lib/chrony/drift

# Logging
logdir /var/log/chrony

# Enable chrony
sudo systemctl enable chronyd
sudo systemctl start chronyd

# Check status
chronyc tracking
# Reference ID    : A9FEA97B (169.254.169.123)
# Stratum         : 2
# Ref time (UTC)  : Mon Oct 28 10:15:23 2024
# System time     : 0.000012456 seconds fast of NTP time
# Last offset     : -0.000023456 seconds
# RMS offset      : 0.000034567 seconds
# Frequency       : 12.345 ppm slow
# Residual freq   : -0.001 ppm
# Skew            : 0.123 ppm
# Root delay      : 0.000123 seconds
# Root dispersion : 0.000234 seconds
# Update interval : 16.0 seconds
# Leap status     : Normal

# Show sources
chronyc sources -v
# MS Name/IP address         Stratum Poll Reach LastRx Last sample
# ===============================================================================
# ^* 169.254.169.123                2   4   377    15   -123us[ -156us] +/-  1234us
# ^+ time.google.com                1   6   377    23   -234us[ -267us] +/-  2345us

# * = current best source
# + = combined
# - = not combined
# x = falseticker
# ? = unreachable

# Force time sync (if needed)
sudo chronyc makestep
# 200 OK

# View current time sources
chronyc activity
# 4 sources online
# 0 sources offline
# 0 sources doing burst (return to online)
# 0 sources doing burst (return to offline)
# 0 sources with unknown address
\`\`\`

## AWS Time Sync Service

### EC2 Time Synchronization

\`\`\`bash
# Amazon Time Sync Service
# IP: 169.254.169.123
# Stratum 1 time source (directly connected to atomic clock)
# Available in all regions
# No additional cost

# Configure chrony for AWS
sudo vi /etc/chrony.conf
# Replace with:
server 169.254.169.123 prefer iburst minpoll 4 maxpoll 4
makestep 1.0 3
driftfile /var/lib/chrony/drift

# Amazon Linux 2023 default configuration
cat /etc/chrony.conf
# server 169.254.169.123 prefer iburst minpoll 4 maxpoll 4
# leapsectz right/UTC
# makestep 1.0 3
# driftfile /var/lib/chrony/drift
# logdir /var/log/chrony

# Restart chrony
sudo systemctl restart chronyd

# Verify using AWS Time Sync
chronyc sources
# MS Name/IP address         Stratum Poll Reach LastRx Last sample
# ===============================================================================
# ^* 169.254.169.123                1   4   377     8   -12us[ -15us] +/- 100us

# Benefits of AWS Time Sync:
# 1. Stratum 1 accuracy
# 2. No network egress charges
# 3. Low latency (link-local)
# 4. Leap second smearing (no sudden jumps)
# 5. Consistent across all AZs in region
\`\`\`

### User Data for Time Sync

\`\`\`bash
#!/bin/bash
# EC2 instance user data

# Configure chrony for AWS Time Sync
cat << 'EOF' > /etc/chrony.conf
server 169.254.169.123 prefer iburst minpoll 4 maxpoll 4
makestep 1.0 3
driftfile /var/lib/chrony/drift
logdir /var/log/chrony
EOF

# Restart chrony
systemctl restart chronyd
systemctl enable chronyd

# Wait for time sync
timeout 60 sh -c 'until chronyc tracking | grep -q "Leap status.*Normal"; do sleep 2; done'

# Continue with application setup
# ...
\`\`\`

## Timezone Management

### Setting Timezone

\`\`\`bash
# View current timezone
timedatectl
#                Local time: Mon 2024-10-28 10:15:23 UTC
#            Universal time: Mon 2024-10-28 10:15:23 UTC
#                  RTC time: Mon 2024-10-28 10:15:23
#                 Time zone: UTC (UTC, +0000)
# System clock synchronized: yes
#               NTP service: active
#           RTC in local TZ: no

# List available timezones
timedatectl list-timezones
# America/New_York
# America/Chicago
# America/Denver
# America/Los_Angeles
# Europe/London
# Asia/Tokyo

# Set timezone
sudo timedatectl set-timezone America/New_York

# Verify
timedatectl
#                Local time: Mon 2024-10-28 06:15:23 EDT
#            Universal time: Mon 2024-10-28 10:15:23 UTC
#                  RTC time: Mon 2024-10-28 10:15:23
#                 Time zone: America/New_York (EDT, -0400)

# Alternative: symlink method
sudo ln -sf /usr/share/zoneinfo/America/New_York /etc/localtime

# Verify timezone file
ls -l /etc/localtime
# lrwxrwxrwx 1 root root 38 Oct 28 10:15 /etc/localtime -> /usr/share/zoneinfo/America/New_York

# Set timezone in environment
export TZ="America/New_York"
date
# Mon Oct 28 06:15:23 EDT 2024
\`\`\`

### Best Practices for Timezones

\`\`\`bash
# Production servers: USE UTC
# Advantages:
# 1. No daylight saving time confusion
# 2. Consistent across all servers
# 3. Easy log correlation
# 4. Database timestamps unambiguous
# 5. API timestamps standard

# Set all servers to UTC
sudo timedatectl set-timezone UTC

# Application handles user timezone
# Store: UTC in database
# Display: User's local timezone in UI

# Example: PostgreSQL
# Store as:
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

# Query in user's timezone:
SELECT created_at AT TIME ZONE 'America/New_York' AS local_time
FROM events;
\`\`\`

## Time Commands and Utilities

### Date Command

\`\`\`bash
# Current date/time
date
# Mon Oct 28 10:15:23 UTC 2024

# Custom format
date '+%Y-%m-%d %H:%M:%S'
# 2024-10-28 10:15:23

# ISO 8601 format
date -I
# 2024-10-28

date -Iseconds
# 2024-10-28T10:15:23+00:00

# Unix timestamp
date +%s
# 1698491723

# Convert timestamp to human-readable
date -d @1698491723
# Mon Oct 28 10:15:23 UTC 2024

# Date arithmetic
date -d "tomorrow"
date -d "next Monday"
date -d "2 weeks ago"
date -d "+3 days"

# Set system time (requires root, not recommended if using NTP)
sudo date --set="2024-10-28 10:15:23"
\`\`\`

### Hardware Clock (RTC)

\`\`\`bash
# View hardware clock
sudo hwclock --show
# 2024-10-28 10:15:23.456789+00:00

# Sync hardware clock to system time
sudo hwclock --systohc

# Sync system time to hardware clock
sudo hwclock --hctosys

# Check if RTC is in UTC or local time
timedatectl | grep "RTC in local TZ"
# RTC in local TZ: no

# Best practice: RTC should be in UTC
\`\`\`

## Monitoring Time Synchronization

### Monitoring Script

\`\`\`bash
#!/bin/bash
# Monitor time synchronization

check_time_sync() {
    # Check if chronyd is running
    if ! systemctl is-active chronyd >/dev/null 2>&1; then
        echo "CRITICAL: chronyd is not running"
        return 2
    fi
    
    # Check sync status
    if ! chronyc tracking | grep -q "Leap status.*Normal"; then
        echo "WARNING: Time not synchronized"
        return 1
    fi
    
    # Check time offset
    offset=$(chronyc tracking | grep "System time" | awk '{print $4}')
    offset_abs=$(echo "$offset" | tr -d '-')
    
    if (( $(echo "$offset_abs > 0.1" | bc -l) )); then
        echo "WARNING: Time offset \${offset}s exceeds 100ms"
        return 1
    fi
    
    echo "OK: Time synchronized, offset \${offset}s"
    return 0
}

check_time_sync
exit $?
\`\`\`

### CloudWatch Metrics

\`\`\`bash
# Send time offset to CloudWatch
#!/bin/bash

INSTANCE_ID=$(ec2-metadata --instance-id | awk '{print $2}')
OFFSET=$(chronyc tracking | grep "System time" | awk '{print $4}' | tr -d 'seconds')

aws cloudwatch put-metric-data \\
  --namespace "CustomMetrics/TimeSync" \\
  --metric-name "TimeOffset" \\
  --value "$OFFSET" \\
  --unit Seconds \\
  --dimensions Instance=$INSTANCE_ID
\`\`\`

### CloudWatch Alarm

\`\`\`terraform
resource "aws_cloudwatch_metric_alarm" "time_drift" {
  alarm_name          = "time-drift-\${var.instance_id
}"
comparison_operator = "GreaterThanThreshold"
evaluation_periods = "2"
metric_name = "TimeOffset"
namespace = "CustomMetrics/TimeSync"
period = "300"
statistic = "Maximum"
threshold = "0.1"  # 100ms
alarm_description = "Time drift exceeds 100ms"
alarm_actions = [aws_sns_topic.alerts.arn]

dimensions = {
    Instance = var.instance_id
}
}
\`\`\`

## Troubleshooting Time Issues

### Time Not Syncing

\`\`\`bash
# Check chronyd status
systemctl status chronyd
# Active: active (running)

# Check sources
chronyc sources -v
# All sources have "?"? Network connectivity issue

# Test NTP server connectivity
nc -vuz 169.254.169.123 123
# Connection to 169.254.169.123 123 port [udp/ntp] succeeded!

# Check firewall
sudo iptables -L -n | grep 123
# Should allow UDP port 123

# Check chrony logs
sudo tail -f /var/log/chrony/tracking.log

# Force sync
sudo chronyc makestep
sudo chronyc -a burst 4/4

# Restart chronyd
sudo systemctl restart chronyd
\`\`\`

### Large Time Offset

\`\`\`bash
# If time difference > 1000s, chrony won't sync automatically

# Option 1: Allow large step
sudo vi /etc/chrony.conf
# Add:
makestep 1.0 -1  # Allow step at any time

sudo systemctl restart chronyd

# Option 2: Manual time set then enable chrony
sudo chronyd -q
sudo systemctl start chronyd

# Option 3: Use ntpdate (deprecated but works)
sudo systemctl stop chronyd
sudo ntpdate 169.254.169.123
sudo systemctl start chronyd
\`\`\`

## Production Scenarios

### Scenario 1: SSL Certificate Validation Failing

\`\`\`bash
# Symptom
curl https://api.example.com
# SSL certificate problem: certificate is not yet valid

# Check system time
date
# Thu Oct 26 10:15:23 UTC 2024  # 2 days behind!

# Check chronyd
systemctl status chronyd
# inactive (dead)

# Start chronyd
sudo systemctl start chronyd
sudo systemctl enable chronyd

# Wait for sync
sleep 10

# Verify time
chronyc tracking
# System time: 0.000012 seconds fast of NTP time

# Retry SSL connection
curl https://api.example.com
# Success!
\`\`\`

### Scenario 2: Log Timestamps Out of Order

\`\`\`bash
# Multiple servers with time drift
# Server 1: 10:15:23
# Server 2: 10:14:45  # 38 seconds behind
# Server 3: 10:15:58  # 35 seconds ahead

# Configure all servers with AWS Time Sync
for server in server{1..3}; do
    ssh $server "sudo vi /etc/chrony.conf"
    # server 169.254.169.123 prefer iburst minpoll 4 maxpoll 4
    
    ssh $server "sudo systemctl restart chronyd"
done

# Wait for sync
sleep 30

# Verify sync across all servers
for server in server{1..3}; do
    echo "$server: $(ssh $server 'date +%s')"
done
# All timestamps within 1 second
\`\`\`

## Best Practices

✅ **Use AWS Time Sync Service** on EC2 instances  
✅ **Set all production servers to UTC timezone**  
✅ **Monitor time offset** with CloudWatch  
✅ **Alert on time drift** > 100ms  
✅ **Use chrony** over ntpd (modern, better for VMs)  
✅ **Configure makestep** for initial large corrections  
✅ **Multiple NTP sources** for redundancy  
✅ **Store timestamps in UTC** in databases  
✅ **Log time sync issues** for debugging  
✅ **Test time sync** during DR drills

## Key Takeaways

1. **Time sync is critical** - authentication, logging, compliance depend on it
2. **AWS Time Sync is best** for EC2 - Stratum 1 accuracy, no cost, low latency
3. **Use UTC everywhere** in production - avoid timezone complexity
4. **Monitor time drift** proactively - don't wait for failures
5. **Chrony > NTP** - faster sync, better for VMs
6. **Test time sync** - include in health checks and monitoring

## Next Steps

In the final section, we'll cover **Debugging Production Issues**, including strace, tcpdump, perf, and systematic troubleshooting methodologies.`,
};
