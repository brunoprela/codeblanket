/**
 * Debugging Production Issues Section
 * Module: Linux System Administration & DevOps Foundations
 */

export const debuggingProductionIssuesSection = {
  id: 'debugging-production-issues',
  title: 'Debugging Production Issues',
  content: `# Debugging Production Issues

## Introduction

Production debugging requires systematic approaches, specialized tools, and deep Linux knowledge. This section covers essential debugging tools (strace, ltrace, tcpdump, perf), performance profiling, network troubleshooting, core dump analysis, and methodologies for diagnosing complex issues in production environments.

## Systematic Debugging Approach

### The Production Debugging Methodology

\`\`\`
1. **Gather Information**
   - What changed recently? (deployments, config, traffic)
   - When did it start? (exact time)
   - Who is affected? (all users, specific subset)
   - What's the impact? (error rate, latency, throughput)

2. **Reproduce**
   - Can you reproduce locally?
   - Can you reproduce in staging?
   - What are the exact steps?

3. **Isolate**
   - Is it application, database, network, or infrastructure?
   - Single instance or cluster-wide?
   - Specific requests or all traffic?

4. **Hypothesize**
   - Form testable hypotheses
   - Rank by likelihood and impact

5. **Test**
   - Test hypotheses systematically
   - Collect data, don't guess
   - Use metrics, logs, traces

6. **Fix**
   - Implement fix in staging first
   - Deploy to production
   - Monitor for regression

7. **Document**
   - Root cause
   - Resolution steps
   - Prevention measures
\`\`\`

## strace - System Call Tracing

### strace Basics

\`\`\`bash
# Trace system calls of a command
strace ls /tmp
# execve("/bin/ls", ["ls", "/tmp"], ...)
# brk(NULL) = 0x55e8f9c0a000
# openat(AT_FDCWD, "/tmp", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
# getdents64(3, ...) = 512
# write(1, "file1\nfile2\n", 13) = 13
# close(3) = 0

# Trace specific system calls
strace -e open,read,write cat /etc/passwd
# Only shows open, read, write calls

# Trace open/openat
strace -e trace=open,openat ls
strace -e trace=file ls  # All file-related syscalls

# Trace network calls
strace -e trace=network curl google.com
# socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 3
# connect(3, {sa_family=AF_INET, sin_port=htons(80), ...}) = 0
# sendto(3, "GET / HTTP/1.1\r\n...", ...) = 78

# Attach to running process
strace -p 1234
# Ctrl+C to stop

# Count syscalls
strace -c ls /tmp
# % time     seconds  usecs/call     calls    errors syscall
# ------ ----------- ----------- --------- --------- ----------------
#  35.71    0.000050          10         5           read
#  28.57    0.000040           8         5           write
#  14.29    0.000020          10         2           openat

# Save to file
strace -o /tmp/trace.log myapp

# Timestamps
strace -t ls  # Time of day
strace -tt ls # Microseconds
strace -T ls  # Time spent in each syscall

# Follow forks
strace -f ./parent-process

# String length (default 32)
strace -s 256 cat /etc/passwd  # Show 256 chars
\`\`\`

### Production Debugging with strace

\`\`\`bash
# Application hanging - find what it's waiting on
strace -p $(pgrep myapp)
# futex(0x7f8a9c0012d0, FUTEX_WAIT_PRIVATE, 2, NULL
# → Waiting for mutex lock (potential deadlock)

# Application slow - find bottleneck
strace -c -p $(pgrep myapp)
# Run for 30 seconds, Ctrl+C
# % time     seconds  usecs/call     calls    errors syscall
# ------ ----------- ----------- --------- --------- ----------------
#  89.23    5.234000     523400        10           fsync
# → Excessive fsync calls causing slowness

# Find files being accessed
strace -e trace=open,openat -p $(pgrep myapp) 2>&1 | grep -v ENOENT
# openat(AT_FDCWD, "/etc/config.json", O_RDONLY) = 3
# → App reading config frequently

# Network issues
strace -e trace=network -p $(pgrep myapp)
# connect(3, {sa_family=AF_INET, sin_port=htons(443), ...}) = -1 ETIMEDOUT
# → Connection timeout to external API

# File not found errors
strace -e trace=open,openat myapp 2>&1 | grep ENOENT
# openat(AT_FDCWD, "/etc/myapp/missing.conf", O_RDONLY) = -1 ENOENT
# → Missing configuration file

# Permission denied
strace -e trace=open,openat myapp 2>&1 | grep EACCES
# openat(AT_FDCWD, "/var/log/myapp/app.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = -1 EACCES
# → Can't write to log file
\`\`\`

## ltrace - Library Call Tracing

\`\`\`bash
# Trace library calls
ltrace ls /tmp
# __libc_start_main(0x403c30, 2, 0x7ffc6e5b9e88, 0x40a2e0 <unfinished ...>
# malloc(120) = 0x1c4e010
# strcpy(0x1c4e010, "/tmp") = 0x1c4e010

# Trace specific library
ltrace -l libssl.so curl https://example.com

# Count library calls
ltrace -c ls /tmp

# Attach to running process
ltrace -p 1234

# Useful for:
# - Debugging library interactions
# - Finding which library function is called
# - Memory allocation issues
# - String manipulation bugs
\`\`\`

## tcpdump - Network Packet Capture

### tcpdump Basics

\`\`\`bash
# Capture all traffic on interface
sudo tcpdump -i eth0
# 10:15:23.123456 IP 10.0.1.50.54321 > 10.0.1.100.80: Flags [S], seq 123456

# Capture specific port
sudo tcpdump -i eth0 port 80
sudo tcpdump -i eth0 port 443

# Capture specific host
sudo tcpdump -i eth0 host 10.0.1.100

# Capture to file (pcap format)
sudo tcpdump -i eth0 -w /tmp/capture.pcap

# Read from file
tcpdump -r /tmp/capture.pcap

# Show packet contents
sudo tcpdump -i eth0 -X  # Hex and ASCII
sudo tcpdump -i eth0 -A  # ASCII only

# Filter by protocol
sudo tcpdump -i eth0 tcp
sudo tcpdump -i eth0 udp
sudo tcpdump -i eth0 icmp

# Complex filters
sudo tcpdump -i eth0 'tcp port 80 and host 10.0.1.100'
sudo tcpdump -i eth0 'tcp[tcpflags] & (tcp-syn|tcp-ack) != 0'  # SYN or ACK

# Capture only HTTP GET requests
sudo tcpdump -i eth0 -s 0 -A 'tcp port 80 and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0)' | grep -i "GET"

# Limit packet count
sudo tcpdump -i eth0 -c 100 port 80

# Rotate capture files
sudo tcpdump -i eth0 -w /tmp/capture.pcap -C 100 -W 5
# -C 100: 100MB per file
# -W 5: Keep 5 files
\`\`\`

### Production Network Debugging

\`\`\`bash
# Debug connection timeout
sudo tcpdump -i eth0 -nn host 10.0.1.100
# 10:15:23.123 IP 10.0.1.50.54321 > 10.0.1.100.443: Flags [S]
# 10:15:24.123 IP 10.0.1.50.54321 > 10.0.1.100.443: Flags [S]  # Retransmit
# 10:15:26.123 IP 10.0.1.50.54321 > 10.0.1.100.443: Flags [S]  # Retransmit
# → No SYN-ACK response: firewall blocking or server down

# Debug slow API
sudo tcpdump -i eth0 -tttt port 8000 | head -20
# 2024-10-28 10:15:23.000000 SYN sent
# 2024-10-28 10:15:23.001000 SYN-ACK received  # 1ms connection
# 2024-10-28 10:15:23.002000 ACK sent
# 2024-10-28 10:15:23.500000 HTTP request sent
# 2024-10-28 10:15:25.500000 HTTP response received  # 2s processing!
# → Server processing is slow, not network

# Capture database queries
sudo tcpdump -i lo -A port 5432 | grep -i SELECT

# Check for packet loss
sudo tcpdump -i eth0 -nn -c 1000 port 80 | grep -c Retransmit

# SSL/TLS handshake
sudo tcpdump -i eth0 -nn 'tcp port 443 and (tcp[tcpflags] & tcp-syn != 0)'
\`\`\`

## perf - Performance Profiling

### perf Basics

\`\`\`bash
# Install perf
sudo yum install perf  # Amazon Linux
sudo apt install linux-tools-generic  # Ubuntu

# CPU profiling
sudo perf top
# Samples: 10K of event 'cycles', 4000 Hz
#   15.23%  myapp           [.] compute_hash
#   10.45%  myapp           [.] process_request
#    8.34%  libc-2.31.so    [.] malloc

# Record performance data
sudo perf record -g -p $(pgrep myapp)
# ^C after 30 seconds

# Analyze recorded data
sudo perf report
# Shows call graph with hotspots

# Profile specific function
sudo perf record -e cpu-clock -g -- /path/to/binary

# System-wide profiling
sudo perf record -a -g sleep 30
sudo perf report

# CPU cache misses
sudo perf stat -e cache-misses,cache-references myapp

# Context switches
sudo perf stat -e context-switches myapp

# Flamegraph (requires perf-tools)
sudo perf record -F 99 -a -g -- sleep 30
sudo perf script | ./stackcollapse-perf.pl | ./flamegraph.pl > flamegraph.svg
\`\`\`

### Production Performance Analysis

\`\`\`bash
# Find CPU hotspot
sudo perf top -p $(pgrep myapp)
# 45.23% - json_parse()  # JSON parsing is bottleneck

# Memory allocation profiling
sudo perf record -e kmem:kmalloc -p $(pgrep myapp)
sudo perf report

# I/O profiling
sudo perf record -e block:block_rq_issue -a -g

# Lock contention
sudo perf record -e lock:lock_acquire -p $(pgrep myapp)
\`\`\`

## lsof - List Open Files

\`\`\`bash
# List all open files
sudo lsof

# Files opened by process
sudo lsof -p 1234

# Processes using specific file
sudo lsof /var/log/myapp/app.log

# Network connections
sudo lsof -i
sudo lsof -i :80  # Port 80
sudo lsof -i TCP  # TCP connections

# Find process using port
sudo lsof -i :8000
# COMMAND  PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
# node    1234 app   10u  IPv4  12345      0t0  TCP *:8000 (LISTEN)

# Deleted files still held open (disk space not freed)
sudo lsof +L1
# Lists files with link count 0 (deleted but still open)

# Find file descriptor leak
watch -n 1 'lsof -p $(pgrep myapp) | wc -l'
# If count keeps increasing, file descriptor leak

# TCP connections by state
sudo lsof -i -sTCP:LISTEN  # Listening
sudo lsof -i -sTCP:ESTABLISHED  # Established
\`\`\`

## Core Dump Analysis

### Enabling Core Dumps

\`\`\`bash
# Check current limit
ulimit -c
# 0 (disabled)

# Enable core dumps
ulimit -c unlimited

# System-wide configuration
sudo vi /etc/security/limits.conf
# * soft core unlimited
# * hard core unlimited

# Core dump pattern
sudo vi /etc/sysctl.conf
# kernel.core_pattern = /tmp/core-%e-%p-%t
# %e: executable name
# %p: PID
# %t: timestamp

# Apply
sudo sysctl -p

# Systemd service with core dumps
[Service]
LimitCORE=infinity

# Generate core dump (for testing)
kill -ABRT $(pgrep myapp)
\`\`\`

### Analyzing Core Dumps

\`\`\`bash
# Analyze with gdb
gdb /path/to/binary /tmp/core-myapp-1234-1698491723

# (gdb) backtrace  # Call stack
# (gdb) info threads  # Thread information
# (gdb) thread 1  # Switch to thread 1
# (gdb) frame 3  # Switch to frame 3
# (gdb) print variable  # Print variable value
# (gdb) list  # Show source code

# Quick backtrace
gdb -batch -ex "thread apply all bt full" /path/to/binary /tmp/core.1234

# Find segfault location
gdb /path/to/binary /tmp/core.1234
# (gdb) where
# #0  0x00007f8a9c001234 in strcmp () from /lib/x86_64-linux-gnu/libc.so.6
# #1  0x0000000000401234 in process_string (str=0x0) at myapp.c:123
# → NULL pointer dereference at myapp.c:123
\`\`\`

## Debugging Tools Comparison

\`\`\`
Tool       Purpose                   Performance Impact    When to Use
--------   ------------------------  -------------------   ---------------------------
strace     System call tracing       HIGH (2-100x slower)  Process hangs, file access issues
ltrace     Library call tracing      HIGH                  Library interaction debugging
tcpdump    Network packet capture    MEDIUM                Network connectivity issues
perf       CPU profiling            LOW-MEDIUM            Performance bottlenecks
lsof       Open files/sockets       LOW                   File descriptor leaks, port conflicts
gdb        Interactive debugging     N/A (stopped)         Core dumps, segfaults
sar        System activity report    LOW                   Historical performance data
vmstat     Virtual memory stats      LOW                   Memory and CPU trends
iostat     I/O statistics           LOW                   Disk performance issues
\`\`\`

## Production Debugging Scenarios

### Scenario 1: Application Hanging

\`\`\`bash
# 1. Check if process is alive
ps aux | grep myapp
# Running, not consuming CPU

# 2. Check what it's doing
sudo strace -p $(pgrep myapp)
# futex(FUTEX_WAIT) = ?
# → Waiting for lock

# 3. Get thread backtrace
sudo gdb -p $(pgrep myapp) -batch -ex "thread apply all bt"
# Thread 1: waiting on mutex at database.c:234
# Thread 2: holding mutex at cache.c:123
# → Deadlock identified

# 4. Generate thread dump (if supported)
kill -USR1 $(pgrep myapp)
# Check log for thread dump

# 5. Check for resource exhaustion
lsof -p $(pgrep myapp) | wc -l
# 65535 open files - at limit!
# → File descriptor exhaustion
\`\`\`

### Scenario 2: High CPU Usage

\`\`\`bash
# 1. Identify process
top
# PID  USER  %CPU  %MEM  COMMAND
# 1234 app   99.9  10.0  myapp

# 2. Profile with perf
sudo perf top -p 1234
# 75% CPU in regex_match()
# → Regular expression performance issue

# 3. Check for infinite loop
sudo strace -c -p 1234
# calls   syscall
# 100000  futex
# 50000   poll
# → Busy loop in futex

# 4. Get call stack
sudo perf record -g -p 1234
sudo perf report
# → Shows call graph to bottleneck
\`\`\`

### Scenario 3: Memory Leak

\`\`\`bash
# 1. Monitor memory growth
watch -n 1 'ps -p $(pgrep myapp) -o pid,vsz,rss,cmd'
# RSS growing by 1MB/second

# 2. Check for file descriptor leak
watch -n 1 'lsof -p $(pgrep myapp) | wc -l'
# Growing steadily → FD leak

# 3. Use valgrind (not in production!)
valgrind --leak-check=full --log-file=valgrind.log myapp

# 4. Enable malloc debugging
export MALLOC_CHECK_=3
myapp

# 5. Generate heap dump
gcore $(pgrep myapp)
# Analyze with gdb

# 6. Use memory profiler
# - heaptrack
# - massif (valgrind)
# - jemalloc profiling
\`\`\`

### Scenario 4: Network Connectivity Issue

\`\`\`bash
# 1. Test basic connectivity
ping 10.0.1.100
# Success

# 2. Test specific port
telnet 10.0.1.100 443
# Connection refused
# → Port not listening or firewall

# 3. Capture packets
sudo tcpdump -i eth0 -nn host 10.0.1.100
# SYN sent, no SYN-ACK
# → Firewall blocking

# 4. Check local firewall
sudo iptables -L -n | grep 10.0.1.100

# 5. Check security group (AWS)
aws ec2 describe-security-groups --group-ids sg-12345

# 6. Trace route
traceroute 10.0.1.100

# 7. Check DNS
dig api.example.com
nslookup api.example.com
\`\`\`

## Best Practices

✅ **Reproduce before fixing** - understand the problem first  
✅ **Use the right tool** - strace for syscalls, tcpdump for network  
✅ **Minimize performance impact** - avoid strace in high-traffic prod  
✅ **Collect data systematically** - logs, metrics, traces  
✅ **Form hypotheses** - don't randomly try things  
✅ **Test in staging first** - whenever possible  
✅ **Document findings** - build knowledge base  
✅ **Implement monitoring** - catch issues earlier next time  
✅ **Root cause analysis** - don't just treat symptoms  
✅ **Share learnings** - team-wide postmortems

## Key Takeaways

1. **Systematic approach beats guessing** - follow the methodology
2. **Right tool for the job** - each tool has specific use cases
3. **Performance impact matters** - strace can slow apps 100x
4. **Production debugging is an art** - requires experience and intuition
5. **Monitor proactively** - catch issues before users do
6. **Document everything** - help future debugging
7. **Learn from failures** - implement preventive measures

## Module Complete!

Congratulations! You've completed **Module 1: Linux System Administration & DevOps Foundations**. You now have production-ready Linux skills for AWS environments.

### What You've Learned

✅ Linux fundamentals (filesystems, processes, permissions)  
✅ Shell scripting for automation  
✅ System monitoring and performance tuning  
✅ Storage management (EBS, EFS, FSx)  
✅ Networking basics and troubleshooting  
✅ Systemd service management  
✅ Log management and centralized logging  
✅ SSH hardening and remote administration  
✅ Security hardening and compliance  
✅ Package management and automated updates  
✅ Process and resource management  
✅ Backup and disaster recovery strategies  
✅ Time synchronization with AWS Time Sync  
✅ Production debugging with strace, tcpdump, perf

### Next Steps

Continue to **Module 2: Networking Deep Dive for AWS** to master:
- VPC architecture and design
- Load balancing (ALB, NLB)
- DNS and Route 53
- CloudFront and CDN
- VPN and Direct Connect
- Network security and compliance
- Multi-region networking
- Production network troubleshooting

**Keep learning and building!** 🚀`,
};
