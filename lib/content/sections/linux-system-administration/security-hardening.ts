/**
 * Security Hardening Section
 * Module: Linux System Administration & DevOps Foundations
 */

export const securityHardeningSection = {
  id: 'security-hardening',
  title: 'Security Hardening',
  content: `# Security Hardening

## Introduction

Security hardening involves reducing the attack surface by removing unnecessary services, enforcing strong authentication, implementing firewalls, and following the principle of least privilege. This section covers production-grade security practices for Linux systems.

## Firewall Configuration (iptables & firewalld)

\`\`\`bash
# iptables - Traditional firewall
# View current rules
sudo iptables -L -v -n

# Allow SSH
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP/HTTPS
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow established connections
sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow loopback
sudo iptables -A INPUT -i lo -j ACCEPT

# Drop everything else
sudo iptables -P INPUT DROP
sudo iptables -P FORWARD DROP
sudo iptables -P OUTPUT ACCEPT

# Save rules (RHEL/Amazon Linux)
sudo service iptables save
# Or
sudo iptables-save > /etc/sysconfig/iptables

# firewalld - Modern firewall (RHEL 7+)
sudo systemctl start firewalld
sudo systemctl enable firewalld

# Check status
sudo firewall-cmd --state
sudo firewall-cmd --list-all

# Add services
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --permanent --add-port=8080/tcp

# Remove service
sudo firewall-cmd --permanent --remove-service=http

# Restrict SSH to specific IP
sudo firewall-cmd --permanent --add-rich-rule='rule family="ipv4" source address="1.2.3.4/32" port port=22 protocol=tcp accept'

# Reload
sudo firewall-cmd --reload

# Zones
sudo firewall-cmd --get-active-zones
sudo firewall-cmd --zone=public --list-all
\`\`\`

## SELinux (Security-Enhanced Linux)

\`\`\`bash
# Check SELinux status
getenforce  # Enforcing, Permissive, or Disabled
sestatus

# Set mode temporarily
sudo setenforce 0  # Permissive
sudo setenforce 1  # Enforcing

# Set mode permanently
sudo vi /etc/selinux/config
# SELINUX=enforcing

# View SELinux contexts
ls -Z /var/www/html
ps -eZ | grep nginx

# Change context
sudo chcon -t httpd_sys_content_t /var/www/html/index.html

# Restore default context
sudo restorecon -Rv /var/www/html

# SELinux booleans
sudo getsebool -a | grep httpd
sudo setsebool -P httpd_can_network_connect on

# Troubleshooting denials
sudo cat /var/log/audit/audit.log | grep denied
sudo ausearch -m avc -ts recent
sudo audit2why < /var/log/audit/audit.log

# Generate policy from denials
sudo audit2allow -a -M mypolicy
sudo semodule -i mypolicy.pp
\`\`\`

## AppArmor (Alternative to SELinux)

\`\`\`bash
# Ubuntu/Debian default

# Check status
sudo aa-status

# Profiles location
ls /etc/apparmor.d/

# Modes
sudo aa-enforce /etc/apparmor.d/usr.sbin.nginx  # Enforce
sudo aa-complain /etc/apparmor.d/usr.sbin.nginx  # Complain (log only)
sudo aa-disable /etc/apparmor.d/usr.sbin.nginx  # Disable

# Reload profiles
sudo systemctl reload apparmor
\`\`\`

## Fail2Ban - Brute Force Protection

\`\`\`bash
# Install
sudo yum install -y fail2ban  # Amazon Linux/RHEL
sudo apt install -y fail2ban  # Ubuntu/Debian

# Configure
sudo cat << 'EOF' > /etc/fail2ban/jail.local
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
destemail = admin@example.com
sendername = Fail2Ban
action = %(action_mwl)s

[sshd]
enabled = true
port = ssh
logpath = /var/log/secure
maxretry = 3
bantime = 86400

[nginx-limit-req]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 10

[nginx-noscript]
enabled = true
port = http,https
logpath = /var/log/nginx/access.log
maxretry = 6
EOF

# Start service
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Check status
sudo fail2ban-client status
sudo fail2ban-client status sshd

# Unban IP
sudo fail2ban-client set sshd unbanip 1.2.3.4

# View banned IPs
sudo iptables -L -n | grep REJECT
\`\`\`

## System Updates & Patch Management

\`\`\`bash
# Amazon Linux 2023
sudo dnf update -y
sudo dnf upgrade -y

# Enable automatic security updates
sudo dnf install -y dnf-automatic
sudo systemctl enable --now dnf-automatic.timer

# Ubuntu
sudo apt update && sudo apt upgrade -y

# Unattended upgrades (Ubuntu)
sudo apt install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades

# Check for CVEs
rpm -qa --changelog | grep CVE  # RHEL/Amazon Linux
\`\`\`

## User & Permission Hardening

\`\`\`bash
# Disable unused users
sudo usermod -L nobody
sudo usermod -s /sbin/nologin nobody

# Password policies
sudo vi /etc/login.defs
# PASS_MAX_DAYS 90
# PASS_MIN_DAYS 7
# PASS_MIN_LEN 12
# PASS_WARN_AGE 14

# PAM password requirements
sudo vi /etc/security/pwquality.conf
# minlen = 12
# dcredit = -1  # At least 1 digit
# ucredit = -1  # At least 1 uppercase
# lcredit = -1  # At least 1 lowercase
# ocredit = -1  # At least 1 special

# Sudo without password (careful!)
echo "appuser ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/appuser

# Sudo with password
echo "appuser ALL=(ALL) ALL" | sudo tee /etc/sudoers.d/appuser

# Audit sudo access
sudo tail -f /var/log/secure | grep sudo
\`\`\`

## Security Scanning

\`\`\`bash
# Lynis - Security auditing
sudo yum install -y lynis
sudo lynis audit system

# ClamAV - Antivirus
sudo yum install -y clamav clamav-update
sudo freshclam  # Update virus definitions
sudo clamscan -r /home

# AIDE - File integrity monitoring
sudo yum install -y aide
sudo aide --init
sudo mv /var/lib/aide/aide.db.new.gz /var/lib/aide/aide.db.gz
sudo aide --check

# RKHunter - Rootkit detection
sudo yum install -y rkhunter
sudo rkhunter --update
sudo rkhunter --check
\`\`\`

## AWS Security Best Practices

\`\`\`terraform
# Security groups - principle of least privilege
resource "aws_security_group" "web_servers" {
  name        = "web-servers-sg"
  vpc_id      = aws_vpc.main.id
  
  # Only allow HTTPS from ALB
  ingress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
    description     = "HTTPS from ALB only"
  }
  
  # No direct SSH - use Session Manager
  
  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS outbound for updates"
  }
}

# IMDSv2 enforcement (prevents SSRF attacks)
resource "aws_launch_template" "web" {
  name = "web-launch-template"
  
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"  # IMDSv2 only
    http_put_response_hop_limit = 1
  }
  
  # Other config...
}

# EBS encryption
resource "aws_ebs_volume" "data" {
  availability_zone = "us-east-1a"
  size              = 100
  encrypted         = true
  kms_key_id        = aws_kms_key.ebs.arn
}

# S3 bucket security
resource "aws_s3_bucket" "private" {
  bucket = "my-private-bucket"
}

resource "aws_s3_bucket_public_access_block" "private" {
  bucket = aws_s3_bucket.private.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "private" {
  bucket = aws_s3_bucket.private.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.s3.arn
    }
  }
}
\`\`\`

## Compliance & Hardening Guides

- **CIS Benchmarks**: Industry-standard security configurations
- **STIG (Security Technical Implementation Guide)**: DoD security standards
- **PCI DSS**: Payment card industry standards
- **HIPAA**: Healthcare data protection
- **NIST Cybersecurity Framework**

\`\`\`bash
# Apply CIS benchmark (automated)
sudo yum install -y openscap scap-security-guide
sudo oscap xccdf eval --profile xccdf_org.ssgproject.content_profile_cis \
  --results-arf arf.xml --report report.html \
  /usr/share/xml/scap/ssg/content/ssg-amzn2-ds.xml
\`\`\`

## Security Checklist

✅ **Firewall** enabled and configured  
✅ **SELinux/AppArmor** in enforcing mode  
✅ **Fail2ban** protecting SSH  
✅ **Regular security updates** automated  
✅ **SSH hardened** (no root, key-based only)  
✅ **Unnecessary services** disabled  
✅ **File integrity monitoring** (AIDE)  
✅ **Intrusion detection** (fail2ban, CloudWatch)  
✅ **Security scanning** (Lynis, vulnerability scanners)  
✅ **Audit logging** enabled and monitored  
✅ **Least privilege** IAM policies  
✅ **Encryption** at rest and in transit  
✅ **IMDSv2** enforced on EC2  
✅ **Security groups** locked down  
✅ **Regular security audits**

## Next Steps

In the next section, we'll cover **Package Management & Updates**, including yum/dnf, apt, package repositories, and automated patching strategies.`,
};
