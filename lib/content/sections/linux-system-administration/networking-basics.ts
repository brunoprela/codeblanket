/**
 * Networking Basics Section
 * Module: Linux System Administration & DevOps Foundations
 */

export const networkingBasicsSection = {
  id: 'networking-basics',
  title: 'Networking Basics',
  content: `# Networking Basics

## Introduction

Understanding Linux networking is essential for DevOps engineers managing production systems on AWS. This section covers TCP/IP fundamentals, DNS configuration, firewall management with iptables, and AWS VPC networking concepts necessary for deploying secure, scalable applications.

## TCP/IP Stack Fundamentals

### OSI Model and TCP/IP

\`\`\`python
"""
OSI Model (7 Layers) vs TCP/IP Model (4 Layers)

OSI                     TCP/IP              Examples
-----------------------------------------------------------
7. Application  ┐
6. Presentation ├──→  Application      HTTP, DNS, SSH, SMTP
5. Session      ┘

4. Transport    ────→  Transport        TCP, UDP

3. Network      ────→  Internet         IP, ICMP, ARP

2. Data Link    ┐
1. Physical     └──→  Network Access   Ethernet, WiFi
"""

layers = {
    'Application': {
        'protocols': ['HTTP/HTTPS', 'DNS', 'SSH', 'FTP', 'SMTP'],
        'port_examples': {
            'HTTP': 80,
            'HTTPS': 443,
            'SSH': 22,
            'DNS': 53,
            'MySQL': 3306,
            'PostgreSQL': 5432,
            'Redis': 6379
        }
    },
    'Transport': {
        'TCP': 'Connection-oriented, reliable, ordered delivery',
        'UDP': 'Connectionless, fast, no guarantees',
        'use_tcp': ['HTTP', 'SSH', 'databases', 'file transfers'],
        'use_udp': ['DNS queries', 'video streaming', 'VoIP', 'game servers']
    },
    'Internet': {
        'IPv4': '32-bit (4.3 billion addresses) - e.g., 192.168.1.1',
        'IPv6': '128-bit (340 undecillion addresses) - e.g., 2001:0db8::1',
        'routing': 'Finds path from source to destination'
    }
}
\`\`\`

### IP Addressing and CIDR

\`\`\`bash
# IPv4 address structure
192.168.1.100
# ^^^ ^^^ ^ ^^^
# Network  Host

# CIDR notation: IP/prefix
# 10.0.0.0/16 means:
# - Network: 10.0
# - Hosts: 0.0 - 255.255
# - Total IPs: 2^16 = 65,536

# Common CIDR blocks:
# /32 = 1 IP (255.255.255.255)
# /24 = 256 IPs (255.255.255.0) - Class C
# /16 = 65,536 IPs (255.255.0.0) - Class B
# /8 = 16,777,216 IPs (255.0.0.0) - Class A

# Calculate CIDR details
ipcalc 10.0.1.0/24
# Output:
# Address:   10.0.1.0
# Netmask:   255.255.255.0 = 24
# Wildcard:  0.0.0.255
# Network:   10.0.1.0/24
# HostMin:   10.0.1.1
# HostMax:   10.0.1.254
# Broadcast: 10.0.1.255
# Hosts/Net: 254

# AWS VPC CIDR planning
# VPC: 10.0.0.0/16 (65,536 IPs)
#   Subnet 1 (Public):  10.0.1.0/24 (256 IPs)
#   Subnet 2 (Private): 10.0.2.0/24 (256 IPs)
#   Subnet 3 (DB):      10.0.3.0/24 (256 IPs)
\`\`\`

## Network Configuration

### Network Interfaces

\`\`\`bash
# Modern command: ip (replaces ifconfig)
ip addr show
# Output:
# 1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536
#     inet 127.0.0.1/8 scope host lo
# 2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9001
#     inet 10.0.1.50/24 brd 10.0.1.255 scope global eth0

# Show specific interface
ip addr show eth0

# Show brief summary
ip -br addr
# lo    UNKNOWN  127.0.0.1/8
# eth0  UP       10.0.1.50/24

# Add secondary IP
sudo ip addr add 10.0.1.51/24 dev eth0

# Remove IP
sudo ip addr del 10.0.1.51/24 dev eth0

# Enable/disable interface
sudo ip link set eth0 up
sudo ip link set eth0 down

# Legacy ifconfig (still works)
ifconfig eth0
\`\`\`

### Routing Tables

\`\`\`bash
# View routing table
ip route show
# default via 10.0.1.1 dev eth0
# 10.0.1.0/24 dev eth0 proto kernel scope link src 10.0.1.50

# Add route
sudo ip route add 192.168.1.0/24 via 10.0.1.1 dev eth0

# Add default gateway
sudo ip route add default via 10.0.1.1

# Delete route
sudo ip route del 192.168.1.0/24

# Route-specific to interface
sudo ip route add 172.16.0.0/16 dev eth1

# Make routing persistent
cat << 'EOF' | sudo tee /etc/sysconfig/network-scripts/route-eth0
192.168.1.0/24 via 10.0.1.1
EOF

# Test connectivity
ping -c 4 8.8.8.8
traceroute google.com
mtr google.com  # Real-time traceroute
\`\`\`

## DNS Configuration

\`\`\`bash
# DNS resolver configuration
cat /etc/resolv.conf
# nameserver 10.0.0.2  # AWS VPC DNS
# search ec2.internal

# Test DNS resolution
nslookup google.com
dig google.com
host google.com

# Detailed dig query
dig google.com +short  # Just the IP
dig google.com ANY     # All records
dig google.com MX      # Mail servers
dig @8.8.8.8 google.com  # Query specific DNS server

# Check DNS resolution path
dig +trace google.com

# Local DNS cache (systemd-resolved)
sudo systemd-resolve --status
sudo systemd-resolve --flush-caches

# /etc/hosts for static mappings
cat << 'EOF' | sudo tee -a /etc/hosts
10.0.1.50  app1.internal
10.0.1.51  app2.internal
10.0.2.10  db1.internal
EOF
\`\`\`

## Firewall Management with iptables

\`\`\`bash
# List current rules
sudo iptables -L -n -v
# -L: list rules
# -n: numeric output (no DNS lookups)
# -v: verbose

# Default policy (DROP or ACCEPT)
sudo iptables -P INPUT DROP    # Drop all incoming by default
sudo iptables -P FORWARD DROP  # Drop all forwarding
sudo iptables -P OUTPUT ACCEPT # Allow all outgoing

# Allow loopback
sudo iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (port 22)
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP/HTTPS
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow from specific IP
sudo iptables -A INPUT -s 10.0.1.0/24 -j ACCEPT

# Allow specific service from specific IP
sudo iptables -A INPUT -p tcp -s 10.0.1.100 --dport 3306 -j ACCEPT

# Block specific IP
sudo iptables -A INPUT -s 192.168.1.100 -j DROP

# Delete rule
sudo iptables -D INPUT -s 192.168.1.100 -j DROP

# Save rules (persistent)
# RHEL/CentOS
sudo iptables-save > /etc/sysconfig/iptables

# Ubuntu
sudo iptables-save > /etc/iptables/rules.v4

# Restore rules
sudo iptables-restore < /etc/sysconfig/iptables

# Production iptables script
cat << 'SCRIPT' > /etc/iptables-rules.sh
#!/bin/bash
# Flush existing rules
iptables -F
iptables -X
iptables -Z

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT

# Allow established
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow ping
iptables -A INPUT -p icmp --icmp-type echo-request -j ACCEPT

# Log dropped packets
iptables -A INPUT -j LOG --log-prefix "IPTables-Dropped: "

# Save rules
iptables-save > /etc/sysconfig/iptables
SCRIPT

chmod +x /etc/iptables-rules.sh
\`\`\`

## Network Troubleshooting

\`\`\`bash
# Check connectivity
ping -c 4 google.com

# Check if port is open
telnet google.com 80
nc -zv google.com 80  # netcat

# Check listening ports
sudo netstat -tlnp  # TCP listening with process
sudo ss -tlnp       # Faster alternative

# Check established connections
sudo netstat -an | grep ESTABLISHED
sudo ss -ta state established

# Network statistics
netstat -s
ss -s

# Check route to host
traceroute google.com
tracepath google.com

# Real-time network monitoring
iftop -i eth0      # Bandwidth by connection
nethogs eth0       # Bandwidth by process

# Packet capture
sudo tcpdump -i eth0 port 80
sudo tcpdump -i eth0 -w capture.pcap
sudo tcpdump -r capture.pcap

# Check DNS
dig +short google.com
nslookup google.com

# Test HTTP endpoint
curl -I https://google.com
wget --spider https://google.com
\`\`\`

## AWS VPC Networking

\`\`\`bash
# VPC CIDR
aws ec2 describe-vpcs --vpc-ids vpc-xxx

# Subnets
aws ec2 describe-subnets --filters "Name=vpc-id,Values=vpc-xxx"

# Route tables
aws ec2 describe-route-tables --filters "Name=vpc-id,Values=vpc-xxx"

# Security groups
aws ec2 describe-security-groups --group-ids sg-xxx

# Network ACLs
aws ec2 describe-network-acls --filters "Name=vpc-id,Values=vpc-xxx"

# Internet Gateway
aws ec2 describe-internet-gateways --filters "Name=attachment.vpc-id,Values=vpc-xxx"

# NAT Gateways
aws ec2 describe-nat-gateways --filter "Name=vpc-id,Values=vpc-xxx"

# Elastic IPs
aws ec2 describe-addresses

# Network interfaces
aws ec2 describe-network-interfaces --filters "Name=vpc-id,Values=vpc-xxx"
\`\`\`

## Security Groups vs NACLs

\`\`\`python
comparison = {
    'Security Groups': {
        'level': 'Instance level (ENI)',
        'rules': 'Allow rules only',
        'stateful': 'Yes - return traffic auto-allowed',
        'evaluation': 'All rules evaluated',
        'default': 'Deny all inbound, allow all outbound',
        'use_case': 'Instance-specific access control'
    },
    'Network ACLs': {
        'level': 'Subnet level',
        'rules': 'Allow and Deny rules',
        'stateful': 'No - must explicitly allow return traffic',
        'evaluation': 'Rules processed in order',
        'default': 'Allow all inbound and outbound',
        'use_case': 'Subnet-wide protection, additional security layer'
    }
}

# Example Security Group (web server)
{
    'Inbound': [
        {'Port': 80, 'Protocol': 'TCP', 'Source': '0.0.0.0/0'},
        {'Port': 443, 'Protocol': 'TCP', 'Source': '0.0.0.0/0'},
        {'Port': 22, 'Protocol': 'TCP', 'Source': '10.0.0.0/16'}
    ],
    'Outbound': [
        {'All traffic': 'Allowed'}  # Default
    ]
}
\`\`\`

## Best Practices

✅ **Use Security Groups** for instance-level access control  
✅ **Use NACLs** as additional subnet-level protection  
✅ **Document firewall rules** with comments  
✅ **Principle of least privilege** - only open necessary ports  
✅ **Use CIDR blocks** carefully - avoid 0.0.0.0/0 when possible  
✅ **Monitor network metrics** - CloudWatch, VPC Flow Logs  
✅ **Test firewall changes** before applying to production  
✅ **Use Route 53** for DNS management  
✅ **Enable VPC Flow Logs** for troubleshooting  
✅ **Regular security group audits** - remove unused rules

## Key Takeaways

1. **TCP/IP stack** has 4 layers in practical use
2. **CIDR notation** defines network ranges efficiently
3. **ip command** replaces legacy ifconfig
4. **iptables** controls host-level firewall
5. **Security Groups** are stateful, NACLs are stateless
6. **DNS resolution** critical for service discovery
7. **VPC networking** provides isolation and security
8. **Network troubleshooting** uses ping, traceroute, tcpdump
9. **Port management** requires understanding of protocols
10. **AWS networking** abstracts but requires understanding fundamentals

## Next Steps

In the next section, we'll cover **Systemd Service Management**, learning how to create, manage, and troubleshoot systemd services for running applications in production.`,
};
