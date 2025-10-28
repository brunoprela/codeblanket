/**
 * Linux Fundamentals for Production Section
 * Module: Linux System Administration & DevOps Foundations
 */

export const linuxFundamentalsProductionSection = {
  id: 'linux-fundamentals-production',
  title: 'Linux Fundamentals for Production',
  content: `# Linux Fundamentals for Production

## Introduction

Production Linux administration requires deep understanding of core operating system concepts that affect system stability, performance, and security. This section covers essential Linux fundamentals you'll need to operate production systems reliably on AWS EC2 instances and other cloud environments.

## File Systems: ext4, xfs, and Inodes

### File System Architecture

Linux file systems organize data on disk using **inodes** (index nodes) - data structures that store metadata about files and directories.

\`\`\`bash
# View inode information
stat /etc/passwd
# Output shows:
#  - File: /etc/passwd
#  - Size: 2847
#  - Inode: 1234567
#  - Links: 1
#  - Access/Modify/Change times

# Display file with inode number
ls -li /etc/passwd
# 1234567 -rw-r--r-- 1 root root 2847 Oct 15 10:30 /etc/passwd

# Check inode usage
df -i
# Shows total and available inodes per filesystem
\`\`\`

**Key Concept**: Files are not their names! A filename is just a pointer to an inode. This enables hard links and explains why you can delete a file while a process still has it open.

### ext4 File System

**ext4** (Fourth Extended Filesystem) is the default for most Linux distributions including Ubuntu.

\`\`\`bash
# Create an ext4 filesystem
mkfs.ext4 /dev/xvdf

# Mount with options
mount -t ext4 -o noatime,errors=remount-ro /dev/xvdf /mnt/data

# Check filesystem status
tune2fs -l /dev/xvdf | grep -i "mount\|state"

# Resize ext4 filesystem (online)
resize2fs /dev/xvdf
\`\`\`

**Production Features**:
- **Journaling**: Metadata journaling prevents corruption during crashes
- **Delayed allocation**: Better performance and less fragmentation
- **Extents**: More efficient than block mapping for large files
- **Online defragmentation**: \`e4defrag\` can defragment mounted filesystems

**Common Mount Options**:
\`\`\`bash
# /etc/fstab entry for production
/dev/xvdf /var/lib/mysql ext4 defaults,noatime,data=ordered 0 2

# noatime: Don't update access time (performance)
# data=ordered: Metadata journaled before data written (safety)
# relatime: Update access time only if older than mtime (compromise)
\`\`\`

### XFS File System

**XFS** excels with large files and high-performance workloads. Default on Amazon Linux 2023 and RHEL/CentOS.

\`\`\`bash
# Create XFS filesystem
mkfs.xfs /dev/xvdf

# Mount with performance options
mount -t xfs -o noatime,largeio,inode64 /dev/xvdf /data

# Check XFS filesystem
xfs_info /dev/xvdf

# XFS-specific: Grow filesystem (online only)
xfs_growfs /data

# XFS repair (must be unmounted)
xfs_repair /dev/xvdf
\`\`\`

**When to Choose XFS**:
- Large files (multi-GB logs, databases, media files)
- High throughput requirements (data analytics, video streaming)
- Need for filesystem growth without downtime
- RHEL/CentOS environments

**XFS Limitations**:
- Cannot shrink filesystem
- Slightly higher memory usage
- Less widely understood than ext4

### Inode Management in Production

**Inode Exhaustion**: You can run out of inodes even with free disk space!

\`\`\`bash
# Check inode usage
df -i
# Filesystem      Inodes  IUsed   IFree IUse% Mounted on
# /dev/xvda1     6000000 200000 5800000    4% /

# Find directories with many files
find /var -xdev -type d -exec sh -c \
  'echo -n "{}: "; ls -1 "{}" | wc -l' \; | \
  sort -t: -k2 -nr | head -20

# Common cause: thousands of small cache/session files
ls /var/lib/php/sessions | wc -l
# 850000 files!

# Solution: Clean up or increase inodes at filesystem creation
mkfs.ext4 -N 10000000 /dev/xvdf  # Specify inode count
\`\`\`

**Production Scenario**: A web application creating millions of session files exhausted inodes while disk was only 30% full, causing "No space left on device" errors despite free space.

## Process Management with systemd

### Understanding systemd

**systemd** is the init system and service manager for modern Linux distributions.

\`\`\`bash
# Check systemd version
systemctl --version

# View system state
systemctl status

# List all running services
systemctl list-units --type=service --state=running

# Service dependency tree
systemctl list-dependencies nginx.service

# Analyze boot time
systemd-analyze
systemd-analyze blame  # Services sorted by init time
\`\`\`

### Process Lifecycle

\`\`\`bash
# View all processes with full details
ps aux
# a: all users
# u: user-oriented format  
# x: include processes without controlling terminal

# More detailed process view
ps auxf  # ASCII art process tree

# Modern alternative: htop
htop  # Interactive process viewer

# Process states:
# R: Running
# S: Sleeping (waiting for event)
# D: Uninterruptible sleep (I/O wait)
# Z: Zombie (terminated but parent hasn't reaped)
# T: Stopped (SIGSTOP)
\`\`\`

**Key Process Concepts**:

1. **Parent-Child Relationships**:
\`\`\`bash
# View process tree
pstree -p

# systemd (PID 1)
#   ├─sshd(1234)
#   │   └─sshd(5678)───bash(5679)───vim(5680)
#   └─nginx(2000)
#       ├─nginx(2001)
#       └─nginx(2002)
\`\`\`

2. **Zombie Processes**:
\`\`\`bash
# Find zombie processes
ps aux | awk '$8=="Z"'

# Zombies occur when:
# - Child exits but parent doesn't call wait()
# - Parent must be fixed to prevent zombies

# Find parent of zombie
ps -o ppid= -p <zombie_pid>

# If parent is broken, kill parent (zombies auto-cleanup)
kill -9 <parent_pid>
\`\`\`

3. **Orphan Processes**:
- When parent dies before child, systemd (PID 1) adopts the orphan
- This is normal and expected behavior

### Signals and Process Control

\`\`\`bash
# Common signals
kill -l  # List all signals

# Key signals:
# SIGTERM (15): Polite shutdown request (default)
# SIGKILL (9): Immediate termination (cannot be caught)
# SIGHUP (1): Reload configuration
# SIGSTOP (19): Pause process
# SIGCONT (18): Resume paused process

# Graceful shutdown
kill -TERM <pid>
# or
kill <pid>  # SIGTERM is default

# Force kill (last resort)
kill -9 <pid>

# Reload application config without restart
kill -HUP <pid>
# Common for nginx, syslog, etc.

# Kill all processes by name
pkill -9 python
killall nginx

# Send signal to process group
kill -TERM -<pgid>
\`\`\`

**Production Best Practice**: Always try SIGTERM first, wait 30 seconds, then SIGKILL if needed.

\`\`\`bash
#!/bin/bash
# Graceful process termination
PID=$1
echo "Sending SIGTERM to $PID"
kill -TERM $PID

for i in {1..30}; do
    if ! kill -0 $PID 2>/dev/null; then
        echo "Process terminated gracefully"
        exit 0
    fi
    sleep 1
done

echo "Process didn't terminate, forcing with SIGKILL"
kill -9 $PID
\`\`\`

## File Permissions, Users, and Groups

### Standard Unix Permissions

\`\`\`bash
# Permission format: type, owner, group, other
ls -l /etc/passwd
# -rw-r--r-- 1 root root 2847 Oct 15 10:30 /etc/passwd
# │││││││││││
# ││││││││││└─ other: read
# │││││││││└── other: no write
# ││││││││└─── other: no execute
# │││││││└──── group: read
# ││││││└───── group: no write
# │││││└────── group: no execute
# ││││└─────── owner: read
# │││└──────── owner: write
# ││└───────── owner: no execute
# │└────────── file type: regular file
# └─────────── (d=directory, l=symlink, b=block device, c=char device)

# Numeric permissions
chmod 644 file.txt     # rw-r--r--
chmod 755 script.sh    # rwxr-xr-x
chmod 600 id_rsa       # rw-------
chmod 700 ~/.ssh       # rwx------

# Symbolic permissions
chmod u+x script.sh           # Add execute for user
chmod go-w file.txt           # Remove write for group and other
chmod a+r public.txt          # Add read for all
chmod u=rwx,go=rx dir/        # Set explicitly

# Change ownership
chown user:group file.txt
chown -R www-data:www-data /var/www/html

# Change only group
chgrp developers project/
\`\`\`

### Access Control Lists (ACLs)

Standard permissions only support one owner and one group. ACLs provide fine-grained control.

\`\`\`bash
# Install ACL tools
apt-get install acl  # Ubuntu/Debian
yum install acl      # RHEL/CentOS

# Enable ACL on filesystem
mount -o remount,acl /

# View ACLs
getfacl /var/www/html

# Grant user-specific permission
setfacl -m u:developer:rwx /var/log/app.log

# Grant group-specific permission
setfacl -m g:developers:rx /opt/app

# Remove specific ACL
setfacl -x u:developer /var/log/app.log

# Remove all ACLs
setfacl -b /var/log/app.log

# Recursive ACL
setfacl -R -m u:www-data:rwx /var/www

# Default ACL (inherited by new files)
setfacl -d -m g:developers:rwx /opt/projects
\`\`\`

**Production Use Case**: Multiple teams need different access to application logs:

\`\`\`bash
# Base permissions
chown app-user:app-group /var/log/app.log
chmod 640 /var/log/app.log

# Add ACL for security team (read-only)
setfacl -m g:security-team:r /var/log/app.log

# Add ACL for dev team (read-write)
setfacl -m g:dev-team:rw /var/log/app.log

# Verify
getfacl /var/log/app.log
# file: var/log/app.log
# owner: app-user
# group: app-group
# user::rw-
# group::r--
# group:security-team:r--
# group:dev-team:rw-
# mask::rw-
# other::---
\`\`\`

### Special Permissions

\`\`\`bash
# SUID (Set User ID): Run as file owner
chmod u+s /usr/bin/passwd
chmod 4755 /usr/bin/passwd
# -rwsr-xr-x (note the 's' in owner execute position)

# SGID (Set Group ID): Run as file group, or inherit directory group
chmod g+s /var/www/html
chmod 2775 /var/www/html
# drwxrwsr-x (note the 's' in group execute position)

# Sticky bit: Only owner can delete files in directory
chmod +t /tmp
chmod 1777 /tmp
# drwxrwxrwt (note the 't' in other execute position)

# Combined:
chmod 6755 binary  # SUID + SGID
chmod 1755 dir/    # Sticky bit
chmod 7755 file    # All three (rare)
\`\`\`

**Security Note**: SUID binaries run with escalated privileges - audit regularly:

\`\`\`bash
# Find all SUID files
find / -type f -perm -4000 -ls 2>/dev/null

# Find all SGID files
find / -type f -perm -2000 -ls 2>/dev/null

# Review for unauthorized SUID binaries (potential security risk)
\`\`\`

## Package Management

### APT (Debian/Ubuntu)

\`\`\`bash
# Update package lists
apt-get update

# Upgrade all packages
apt-get upgrade          # Safe: won't remove packages
apt-get dist-upgrade     # May remove packages for dependencies

# Install package
apt-get install nginx

# Remove package
apt-get remove nginx      # Keeps config files
apt-get purge nginx       # Removes config files

# Search for packages
apt-cache search nginx
apt search nginx

# Show package details
apt-cache show nginx
apt show nginx

# List installed packages
dpkg -l
apt list --installed

# Find which package provides a file
dpkg -S /bin/ls
# coreutils: /bin/ls

# List files in package
dpkg -L nginx

# Clean up
apt-get autoremove        # Remove unused dependencies
apt-get autoclean         # Remove old .deb files
apt-get clean             # Remove all .deb files
\`\`\`

### YUM/DNF (RHEL/CentOS/Amazon Linux)

\`\`\`bash
# Update package lists
yum check-update

# Upgrade packages
yum update                # All packages
yum update nginx          # Specific package

# Install package
yum install nginx

# Remove package
yum remove nginx

# Search packages
yum search nginx

# Show package info
yum info nginx

# List installed packages
yum list installed

# Find which package provides a file
yum provides /usr/bin/python3

# List files in package
rpm -ql nginx

# Package groups
yum grouplist
yum groupinstall "Development Tools"

# Clean cache
yum clean all
\`\`\`

### Amazon Linux 2023 Specifics

Amazon Linux 2023 uses **dnf** (next-gen YUM):

\`\`\`bash
# Update Amazon Linux 2023
dnf update

# Install from Amazon Linux repos
dnf install python3.11

# Enable/disable repositories
dnf config-manager --enable epel
dnf repolist

# Transaction history
dnf history
dnf history undo 5        # Undo transaction #5
\`\`\`

## System Calls Basics

Understanding system calls helps debug issues and understand performance.

### What Are System Calls?

System calls are the interface between userspace applications and the kernel.

\`\`\`bash
# Trace system calls
strace ls /tmp
# execve("/bin/ls", ["ls", "/tmp"], ...)
# openat(AT_FDCWD, "/tmp", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
# getdents64(3, ...) = 512
# write(1, "file1\nfile2\n", 13) = 13

# Count syscalls
strace -c ls /tmp
# % time     seconds  usecs/call     calls    errors syscall
# ------ ----------- ----------- --------- --------- ----------------
#  33.33    0.000010           2         5           read
#  33.33    0.000010           2         5           write
#  ...

# Trace specific syscall
strace -e open ls /tmp
strace -e trace=network curl google.com

# Attach to running process
strace -p <pid>
\`\`\`

### Common System Calls

\`\`\`bash
# File operations
open(), read(), write(), close()
openat(), stat(), fstat()

# Process management
fork(), exec(), wait(), exit()
clone() (for threads)

# Network
socket(), bind(), listen(), accept()
connect(), send(), recv()

# Memory
mmap(), munmap(), brk()

# Signals
kill(), signal(), sigaction()
\`\`\`

**Production Debugging Example**:

\`\`\`bash
# Application hanging? Check what it's waiting on
strace -p <pid>
# Stuck in:
# futex(0x7f8a9c0012d0, FUTEX_WAIT_PRIVATE, 2, NULL

# Indicates waiting for mutex/lock - likely deadlock

# Application slow? Count system calls
strace -c -p <pid>
# High number of stat() calls? Checking files repeatedly
# Many failed open() calls? Looking for missing files
\`\`\`

## Kernel Parameters and Tuning

### sysctl: Runtime Kernel Configuration

\`\`\`bash
# View all kernel parameters
sysctl -a

# View specific parameter
sysctl net.ipv4.ip_forward

# Set parameter temporarily (until reboot)
sysctl -w net.ipv4.ip_forward=1

# Persistent configuration
vi /etc/sysctl.conf
# Add:
net.ipv4.ip_forward = 1

# Apply changes
sysctl -p

# Alternative: per-file configuration
echo 1 > /proc/sys/net/ipv4/ip_forward
\`\`\`

### Production Tuning Parameters

**Network Tuning**:

\`\`\`bash
# /etc/sysctl.d/99-network-tuning.conf

# Increase connection queue for high traffic
net.core.somaxconn = 4096
net.ipv4.tcp_max_syn_backlog = 4096

# Increase buffer sizes for high throughput
net.core.rmem_max = 134217728       # 128 MB
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 67108864
net.ipv4.tcp_wmem = 4096 65536 67108864

# Enable TCP window scaling
net.ipv4.tcp_window_scaling = 1

# Reduce TIME_WAIT connections
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_tw_reuse = 1

# Increase port range
net.ipv4.ip_local_port_range = 10000 65000

# Apply:
sysctl -p /etc/sysctl.d/99-network-tuning.conf
\`\`\`

**File System Tuning**:

\`\`\`bash
# /etc/sysctl.d/99-filesystem-tuning.conf

# Increase file handles
fs.file-max = 2097152

# Increase inotify limits (for file watching)
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 512

# Reduce swappiness (prefer RAM over swap)
vm.swappiness = 10

# Dirty page ratios for write performance
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
\`\`\`

**Database Server Tuning**:

\`\`\`bash
# /etc/sysctl.d/99-database-tuning.conf

# Shared memory for PostgreSQL/MySQL
kernel.shmmax = 68719476736  # 64 GB
kernel.shmall = 4294967296

# Increase connection tracking
net.netfilter.nf_conntrack_max = 1000000

# TCP keepalive for long connections
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_keepalive_probes = 5
net.ipv4.tcp_keepalive_intvl = 15
\`\`\`

## AWS-Specific Linux Considerations

### Amazon Linux 2023

\`\`\`bash
# Check Amazon Linux version
cat /etc/os-release
# NAME="Amazon Linux"
# VERSION="2023"
# ID="amzn"

# Amazon Linux uses:
# - XFS as default filesystem
# - SELinux enabled by default
# - dnf package manager
# - systemd init system
# - kernel 6.1+ with AWS optimizations

# AWS-optimized kernel settings
uname -r
# 6.1.19-30.43.amzn2023.x86_64

# Pre-installed AWS tools
aws --version
ec2-metadata --help
\`\`\`

### EC2-Specific Considerations

\`\`\`bash
# Enhanced Networking (SR-IOV) enabled?
ethtool -i eth0 | grep driver
# driver: ena (Elastic Network Adapter)

# Check instance metadata
curl http://169.254.169.254/latest/meta-data/instance-type
# t3.large

# EBS volume optimization
# Check if EBS-optimized
aws ec2 describe-instances --instance-ids i-xxx \\
  --query 'Reservations[0].Instances[0].EbsOptimized'

# Monitor EBS performance
iostat -x 1

# NVMe EBS volumes (newer instances)
lsblk
# nvme0n1 (root volume)
# nvme1n1 (attached EBS volume)
\`\`\`

## Real-World Production Scenarios

### Scenario 1: Server Running Out of Inodes

**Symptom**: "No space left on device" but \`df -h\` shows space available.

\`\`\`bash
# Check inode usage
df -i
# /dev/xvda1     6000000 5999999      1  100% /

# Find culprit
for dir in /*; do
  echo "$dir: $(find "$dir" -xdev | wc -l)"
done
# /var: 5900000

for dir in /var/*; do
  echo "$dir: $(find "$dir" -xdev | wc -l)"
done
# /var/lib/php/sessions: 5800000

# Solution: Clean old session files
find /var/lib/php/sessions -type f -mtime +7 -delete

# Preventive: Add cron job
crontab -e
# 0 3 * * * find /var/lib/php/sessions -type f -mtime +7 -delete
\`\`\`

### Scenario 2: Application Won't Start After Reboot

**Symptom**: Application auto-starts on boot but fails.

\`\`\`bash
# Check systemd status
systemctl status myapp.service
# Failed to start

# Check logs
journalctl -u myapp.service -b
# "Permission denied"

# Issue: Application expects /var/run/myapp directory
# /var/run is tmpfs - cleared on boot!

# Solution: Use RuntimeDirectory in service file
vi /etc/systemd/system/myapp.service
# [Service]
# RuntimeDirectory=myapp
# RuntimeDirectoryMode=0755

# systemd will automatically create /var/run/myapp on start
\`\`\`

### Scenario 3: High Load Average but Low CPU Usage

**Symptom**: Load average is 50 but CPU usage is 10%.

\`\`\`bash
# Check load average
uptime
# load average: 50.23, 48.15, 45.67

# But CPU is idle
top
# %Cpu(s): 10.2 us,  2.3 sy,  0.0 ni, 85.0 id

# Check for I/O wait
top
# %Cpu(s): 10.2 us,  2.3 sy,  0.0 ni, 10.0 id, 75.0 wa
#                                               ^^^^^ HIGH I/O WAIT!

# Find processes in D state (uninterruptible sleep - I/O wait)
ps aux | awk '$8 == "D"'

# Check disk I/O
iostat -x 1
# Device r/s  w/s  %util
# xvda   0.0  5000.0  100.00

# Issue: Disk saturated with writes
# Solution: Check application doing excessive I/O
iotop -o  # Show only processes doing I/O
\`\`\`

## Key Takeaways

1. **Inodes are separate from disk space** - monitor both
2. **systemd manages modern Linux** - master service management
3. **Permissions beyond rwx** - ACLs provide fine-grained control
4. **Kernel tuning matters** - especially for high-performance workloads
5. **AWS optimizations** - Amazon Linux includes AWS-specific enhancements
6. **strace is your friend** - for debugging system call issues
7. **Zombie processes are symptoms** - fix the parent process
8. **Always SIGTERM before SIGKILL** - allow graceful shutdown

## Best Practices for Production

✅ **Monitor inode usage** alongside disk space  
✅ **Use specific Amazon Linux versions** in production  
✅ **Apply security updates promptly** but test first  
✅ **Document custom sysctl parameters** with rationale  
✅ **Use systemd service hardening** (RestrictSUIDs, PrivateTmp, etc.)  
✅ **Implement log rotation** to prevent disk fills  
✅ **Set up AWS CloudWatch agent** for metrics collection  
✅ **Use ACLs for complex permission requirements**  
✅ **Regular security audits** of SUID binaries  
✅ **Kernel parameter tuning** based on workload profiling

## Next Steps

In the next section, we'll dive deep into **Shell Scripting for Automation**, learning how to write production-ready Bash scripts that automate deployments, backups, and system maintenance tasks.`,
};
