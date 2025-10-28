/**
 * Systemd Service Management Section
 * Module: Linux System Administration & DevOps Foundations
 */

export const systemdServiceManagementSection = {
  id: 'systemd-service-management',
  title: 'Systemd Service Management',
  content: `# Systemd Service Management

## Introduction

Systemd is the init system and service manager for modern Linux distributions, replacing the traditional SysV init. It's responsible for bootstrapping user space, managing services, handling dependencies, and supervising processes. Understanding systemd is fundamental for running production applications, configuring automatic restarts, implementing health checks, and troubleshooting service issues.

In this comprehensive section, we'll cover systemd from fundamentals to advanced production patterns, including service configuration, resource limits, security hardening, timers, and real-world troubleshooting.

## Systemd Architecture and Concepts

### What is Systemd?

Systemd is not just an init system - it's a suite of system management daemons, libraries, and utilities:

- **systemd**: The main system and service manager (PID 1)
- **journald**: Logging service
- **logind**: Login and session management
- **resolved**: DNS resolution
- **timedated**: Time synchronization
- **networkd**: Network configuration

\`\`\`bash
# Check systemd version
systemctl --version
# systemd 252 (252.17-1+deb12u1)
# +PAM +AUDIT +SELINUX +APPARMOR +IMA +SMACK +SECCOMP +GCRYPT ...

# View system status
systemctl status
# ● ip-10-0-1-50
#     State: running
#      Jobs: 0 queued
#    Failed: 0 units
#     Since: Mon 2024-10-28 10:00:00 UTC; 5 days ago

# List all units
systemctl list-units

# List only services
systemctl list-units --type=service

# List all running services
systemctl list-units --type=service --state=running

# List failed services
systemctl list-units --type=service --state=failed

# Show all unit types
systemctl list-units --type=help
# service, socket, target, device, mount, timer, path, slice, scope
\`\`\`

### Unit Files and Locations

Unit files define how systemd manages services, sockets, and other resources.

\`\`\`bash
# Unit file locations (in order of precedence):
# 1. /etc/systemd/system/         - Administrator overrides
# 2. /run/systemd/system/          - Runtime units (tmpfs)
# 3. /lib/systemd/system/          - Distribution-provided units

# View unit file
systemctl cat nginx.service

# List all unit files
systemctl list-unit-files

# Check if service is enabled
systemctl is-enabled nginx.service

# Check if service is active
systemctl is-active nginx.service
\`\`\`

## Basic Service Operations

\`\`\`bash
# Start a service
sudo systemctl start nginx

# Stop a service
sudo systemctl stop nginx

# Restart a service
sudo systemctl restart nginx

# Reload configuration without stopping
sudo systemctl reload nginx

# Reload or restart if reload not supported
sudo systemctl reload-or-restart nginx

# Check service status
systemctl status nginx

# Enable service to start on boot
sudo systemctl enable nginx

# Disable service from starting on boot
sudo systemctl disable nginx

# Enable and start in one command
sudo systemctl enable --now nginx

# Prevent a service from being started (mask)
sudo systemctl mask nginx

# Unmask a service
sudo systemctl unmask nginx

# View service logs
journalctl -u nginx

# Follow logs in real-time
journalctl -u nginx -f
\`\`\`

## Creating Production Services

### Basic Service Structure

\`\`\`bash
# Example: Node.js application service
sudo cat << 'EOF' > /etc/systemd/system/nodeapp.service
[Unit]
Description=Node.js Web Application
Documentation=https://docs.example.com
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=nodeapp
Group=nodeapp
WorkingDirectory=/opt/nodeapp

Environment="NODE_ENV=production"
Environment="PORT=3000"

ExecStart=/usr/bin/node /opt/nodeapp/server.js

Restart=always
RestartSec=10

StandardOutput=journal
StandardError=journal
SyslogIdentifier=nodeapp

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Start and enable
sudo systemctl enable --now nodeapp.service
\`\`\`

### Production-Grade Service Configuration

\`\`\`bash
# Complete production service
sudo cat << 'EOF' > /etc/systemd/system/webapp.service
[Unit]
Description=Production Web Application
Documentation=https://wiki.example.com/webapp
After=network-online.target postgresql.service redis.service
Wants=network-online.target
Requires=postgresql.service

[Service]
####################
# Process Management
####################
Type=simple
# Types: simple, forking, oneshot, notify, idle

####################
# User/Group
####################
User=webapp
Group=webapp
SupplementaryGroups=ssl-cert

####################
# Working Directory
####################
WorkingDirectory=/opt/webapp

####################
# Environment
####################
Environment="NODE_ENV=production"
Environment="PORT=8000"
EnvironmentFile=/etc/webapp/config.env

####################
# Execution
####################
ExecStartPre=/usr/local/bin/webapp-preflight.sh
ExecStart=/usr/bin/node /opt/webapp/server.js
ExecReload=/bin/kill -HUP $MAINPID
ExecStop=/bin/kill -TERM $MAINPID

TimeoutStartSec=60
TimeoutStopSec=30

####################
# Restart Policy
####################
Restart=always
RestartSec=10

# Prevent restart storms
StartLimitIntervalSec=300
StartLimitBurst=5

####################
# Resource Limits
####################
MemoryLimit=2G
MemoryHigh=1.8G
CPUQuota=200%
LimitNOFILE=65536
LimitNPROC=4096

####################
# Security Hardening
####################
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/webapp/logs /opt/webapp/data

ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictRealtime=true
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX

####################
# Logging
####################
StandardOutput=journal
StandardError=journal
SyslogIdentifier=webapp

####################
# Watchdog
####################
WatchdogSec=30

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now webapp.service
\`\`\`

## Service Dependencies

### Understanding Dependencies

\`\`\`bash
# View dependency tree
systemctl list-dependencies webapp.service

# Reverse dependencies
systemctl list-dependencies --reverse postgresql.service

# Critical chain analysis
systemd-analyze critical-chain webapp.service

# Analyze boot time
systemd-analyze

# Show slowest services
systemd-analyze blame
\`\`\`

### Complex Dependency Example

\`\`\`bash
# Multi-tier: Database → App → Load Balancer

# 1. Database service
cat << 'EOF' > /etc/systemd/system/mydb.service
[Unit]
Description=Application Database
After=network-online.target
Wants=network-online.target

[Service]
Type=forking
ExecStart=/usr/local/bin/mydb-server
PIDFile=/var/run/mydb.pid
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 2. Application (depends on DB)
cat << 'EOF' > /etc/systemd/system/myapp.service
[Unit]
Description=Application Server
After=mydb.service
Requires=mydb.service

[Service]
Type=notify
ExecStart=/usr/local/bin/myapp-server
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 3. Load balancer (depends on app)
cat << 'EOF' > /etc/systemd/system/mylb.service
[Unit]
Description=Load Balancer
After=myapp.service
Requires=myapp.service

[Service]
Type=forking
ExecStart=/usr/sbin/nginx
PIDFile=/var/run/nginx.pid
Restart=always

[Install]
WantedBy=multi-user.target
EOF
\`\`\`

## Advanced Service Types

### Type=oneshot (One-Time Tasks)

\`\`\`bash
cat << 'EOF' > /etc/systemd/system/db-migration.service
[Unit]
Description=Database Migration
Before=myapp.service
After=postgresql.service
Requires=postgresql.service

[Service]
Type=oneshot
ExecStart=/opt/myapp/bin/migrate-database
RemainAfterExit=true
User=myapp

[Install]
WantedBy=multi-user.target
EOF
\`\`\`

### Type=notify (Modern Applications)

\`\`\`bash
cat << 'EOF' > /etc/systemd/system/notify-app.service
[Unit]
Description=Application with sd_notify

[Service]
Type=notify
ExecStart=/usr/local/bin/my-app
NotifyAccess=main
TimeoutStartSec=60
Restart=always

[Install]
WantedBy=multi-user.target
EOF
\`\`\`

Python with sd_notify:

\`\`\`python
#!/usr/bin/env python3
import systemd.daemon
import time

def main():
    print("Starting application...")
    time.sleep(5)
    
    # Notify systemd we're ready
    systemd.daemon.notify('READY=1')
    print("Application ready!")
    
    while True:
        time.sleep(10)
        # Send watchdog keepalive
        systemd.daemon.notify('WATCHDOG=1')

if __name__ == '__main__':
    main()
\`\`\`

## Systemd Timers (Cron Replacement)

### Why Timers Over Cron?

1. Better logging (journalctl)
2. Dependencies on services
3. Persistent (run missed executions)
4. Randomization support
5. Monitoring with systemctl
6. Flexible triggers

### Creating a Timer

\`\`\`bash
# 1. Service file
cat << 'EOF' > /etc/systemd/system/backup.service
[Unit]
Description=Daily Backup

[Service]
Type=oneshot
ExecStart=/usr/local/bin/backup.sh
User=backup
StandardOutput=journal
EOF

# 2. Timer file
cat << 'EOF' > /etc/systemd/system/backup.timer
[Unit]
Description=Run backup daily at 2 AM

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
EOF

# Enable and start timer
sudo systemctl enable --now backup.timer

# List all timers
systemctl list-timers

# Check timer status
systemctl status backup.timer
\`\`\`

### Timer Schedules

\`\`\`bash
# OnCalendar examples
OnCalendar=daily                  # Daily at midnight
OnCalendar=weekly                 # Weekly on Monday
OnCalendar=monthly                # Monthly on 1st
OnCalendar=*-*-* 00:00:00        # Daily at midnight
OnCalendar=Mon-Fri 09:00:00      # Weekdays at 9 AM
OnCalendar=Mon,Wed,Fri 10:00:00  # MWF at 10 AM
OnCalendar=*-*-01 00:00:00       # 1st of month
OnCalendar=*:0/15                # Every 15 minutes

# Monotonic timers
OnBootSec=15min                   # 15 min after boot
OnUnitActiveSec=1h                # 1h after last activation
OnUnitInactiveSec=30min           # 30min after last deactivation

# Verify calendar expression
systemd-analyze calendar "Mon-Fri *-*-* 09:00:00"
\`\`\`

## Troubleshooting Services

### Service Won't Start

\`\`\`bash
# Check status
sudo systemctl status myapp
# Shows: failed, exit code, last log lines

# View full logs
sudo journalctl -u myapp -n 100

# Check configuration
sudo systemd-analyze verify myapp.service

# Test without starting
sudo systemd-run --collect --wait /path/to/binary

# Debug mode
SYSTEMD_LOG_LEVEL=debug sudo systemctl restart myapp

# Check dependencies
systemctl list-dependencies myapp

# Common issues:
# 1. Wrong User/Group
# 2. Missing WorkingDirectory
# 3. Incorrect ExecStart path
# 4. Missing dependencies
# 5. Permission issues
\`\`\`

### Service Crashes Repeatedly

\`\`\`bash
# Increase StartLimitBurst temporarily for debugging
sudo systemctl edit myapp.service
# [Service]
# StartLimitBurst=10

# Monitor in real-time
watch -n 1 'systemctl status myapp'

# Check if hitting resource limits
journalctl -u myapp | grep -i "killed\|oom\|memory"

# View restart count
systemctl show myapp -p NRestarts

# Reset restart counter
sudo systemctl reset-failed myapp
\`\`\`

### Performance Issues

\`\`\`bash
# Check resource usage
systemctl status myapp
# Shows: Memory, CPU, Tasks

# Detailed resource info
systemd-cgtop

# Service accounting
systemctl show myapp -p CPUUsageNSec
systemctl show myapp -p MemoryCurrent

# Set resource limits
sudo systemctl edit myapp.service
# [Service]
# CPUQuota=150%
# MemoryLimit=1G
\`\`\`

## Real-World Production Scenarios

### Scenario 1: Service Restart Storm

**Problem**: Service crashes and restarts infinitely, filling logs.

\`\`\`bash
# Symptom
sudo systemctl status myapp
# Active: activating (auto-restart) (Result: exit-code)

# Check restart count
systemctl show myapp -p NRestarts
# NRestarts=247

# View crash pattern
journalctl -u myapp --since "1 hour ago" | grep -i error

# Solution: Implement exponential backoff
sudo systemctl edit myapp.service
# [Service]
# Restart=on-failure
# RestartSec=10
# StartLimitIntervalSec=300
# StartLimitBurst=5
# StartLimitAction=none

# After 5 restarts in 5 minutes, stop trying
\`\`\`

### Scenario 2: Graceful Shutdown Timeout

**Problem**: Service takes too long to stop, systemd kills it forcefully.

\`\`\`bash
# Symptom
journalctl -u myapp | grep "Killing\|SIGKILL"

# Solution: Increase timeout
sudo systemctl edit myapp.service
# [Service]
# TimeoutStopSec=90
# KillMode=mixed
# KillSignal=SIGTERM

# Application should handle SIGTERM gracefully
\`\`\`

Python graceful shutdown:

\`\`\`python
import signal
import sys

def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    # Close database connections
    db.close()
    # Finish in-flight requests
    server.shutdown()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
\`\`\`

### Scenario 3: Dependency Not Ready

**Problem**: Service starts before database is ready.

\`\`\`bash
# Symptom
journalctl -u myapp
# Connection refused to database

# Solution: Add readiness check
sudo systemctl edit myapp.service
# [Unit]
# After=postgresql.service
# Requires=postgresql.service
#
# [Service]
# ExecStartPre=/usr/local/bin/wait-for-postgres.sh

# wait-for-postgres.sh
cat << 'EOF' > /usr/local/bin/wait-for-postgres.sh
#!/bin/bash
for i in {1..30}; do
    if pg_isready -h localhost -p 5432; then
        exit 0
    fi
    sleep 1
done
echo "Database not ready after 30 seconds"
exit 1
EOF

chmod +x /usr/local/bin/wait-for-postgres.sh
\`\`\`

### Scenario 4: Memory Leak Detection

**Problem**: Service memory grows over time.

\`\`\`bash
# Monitor memory usage
watch -n 1 'systemctl status myapp | grep Memory'

# Set memory limits
sudo systemctl edit myapp.service
# [Service]
# MemoryHigh=1.8G   # Warning threshold
# MemoryMax=2G      # Hard limit (OOM kill)

# Log when hitting limits
journalctl -u myapp | grep -i "memory\|oom"

# Automatic restart on high memory
sudo systemctl edit myapp.service
# [Service]
# MemoryMax=2G
# Restart=always
# When OOM killed, systemd will restart
\`\`\`

## Security Best Practices

### Sandboxing Services

\`\`\`bash
# Full sandboxing example
cat << 'EOF' > /etc/systemd/system/sandboxed.service
[Unit]
Description=Sandboxed Service

[Service]
Type=simple
ExecStart=/usr/local/bin/myapp

# User isolation
User=myapp
Group=myapp
DynamicUser=yes  # Ephemeral user/group

# Filesystem isolation
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ReadOnlyPaths=/
ReadWritePaths=/var/lib/myapp

# Network isolation (if not needed)
PrivateNetwork=yes

# Device isolation
PrivateDevices=yes

# Kernel protection
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes

# System call filtering
SystemCallFilter=@system-service
SystemCallFilter=~@privileged @resources

# No new privileges
NoNewPrivileges=yes

# Restrict namespaces
RestrictNamespaces=yes

# Restrict realtime
RestrictRealtime=yes

[Install]
WantedBy=multi-user.target
EOF
\`\`\`

### Principle of Least Privilege

\`\`\`bash
# Run as non-root user
User=appuser
Group=appgroup

# Drop all capabilities
CapabilityBoundingSet=

# If need to bind to privileged port
AmbientCapabilities=CAP_NET_BIND_SERVICE
CapabilityBoundingSet=CAP_NET_BIND_SERVICE

# Restrict address families
RestrictAddressFamilies=AF_INET AF_INET6

# Lock down filesystem
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/lib/myapp
\`\`\`

## Monitoring and Observability

### Service Health Checks

\`\`\`bash
# Check service status
systemctl is-active myapp
# active, inactive, failed

# Machine-readable status
systemctl show myapp -p ActiveState,SubState,Result

# Export metrics for monitoring
cat << 'EOF' > /usr/local/bin/service-metrics.sh
#!/bin/bash
for service in myapp nginx postgresql; do
    active=$(systemctl is-active $service)
    status=$([[ "$active" == "active" ]] && echo 1 || echo 0)
    echo "service_up{service=\"$service\"} $status"
    
    # Restart count
    restarts=$(systemctl show $service -p NRestarts --value)
    echo "service_restarts{service=\"$service\"} $restarts"
    
    # Memory usage
    mem=$(systemctl show $service -p MemoryCurrent --value)
    echo "service_memory_bytes{service=\"$service\"} $mem"
done
EOF
\`\`\`

### Integration with CloudWatch

\`\`\`bash
# Send service failures to CloudWatch
cat << 'EOF' > /etc/systemd/system/cloudwatch-alert@.service
[Unit]
Description=CloudWatch Alert for %i

[Service]
Type=oneshot
ExecStart=/usr/local/bin/send-cloudwatch-alert.sh %i
EOF

# Hook into service failures
sudo systemctl edit myapp.service
# [Unit]
# OnFailure=cloudwatch-alert@%n.service
\`\`\`

## Best Practices

✅ **Use Type=simple** for most applications  
✅ **Run as non-root** user  
✅ **Set Restart=always** for production services  
✅ **Configure resource limits** (Memory, CPU, file descriptors)  
✅ **Enable security hardening** (ProtectSystem, NoNewPrivileges)  
✅ **Implement graceful shutdown** (handle SIGTERM)  
✅ **Use ExecStartPre** for readiness checks  
✅ **Set proper timeouts** (TimeoutStartSec, TimeoutStopSec)  
✅ **Prevent restart storms** (StartLimitBurst, StartLimitInterval)  
✅ **Monitor service health** (automated checks)  
✅ **Use timers instead of cron**  
✅ **Log to journald**  
✅ **Document dependencies** clearly  
✅ **Test service files** before deployment

## Key Takeaways

1. **systemd is powerful** - use its features (dependencies, resource limits, security)
2. **Always handle SIGTERM** gracefully in applications
3. **Restart policies matter** - configure properly to avoid storms
4. **Resource limits prevent runaway processes**
5. **Security hardening is built-in** - use it
6. **Timers are better than cron** for new services
7. **Monitor service health** proactively
8. **journalctl is your friend** for debugging

## Next Steps

In the next section, we'll cover **Log Management**, including log rotation, centralized logging, structured logging, and AWS CloudWatch Logs integration.`,
};
