/**
 * SSH & Remote Administration Section
 * Module: Linux System Administration & DevOps Foundations
 */

export const sshRemoteAdministrationSection = {
  id: 'ssh-remote-administration',
  title: 'SSH & Remote Administration',
  content: `# SSH & Remote Administration

## Introduction

Secure Shell (SSH) is the standard protocol for secure remote administration of Linux systems. Understanding SSH configuration, key-based authentication, hardening, and AWS-specific considerations (Session Manager, bastion hosts) is essential for production DevOps.

## SSH Basics

\`\`\`bash
# Connect to server
ssh user@hostname
ssh user@192.168.1.100
ssh -p 2222 user@hostname  # Custom port

# Specify private key
ssh -i ~/.ssh/mykey.pem user@hostname

# Execute single command
ssh user@hostname 'uptime'
ssh user@hostname 'sudo systemctl restart nginx'

# SSH config file
cat << 'EOF' > ~/.ssh/config
Host prod-web
    HostName 10.0.1.50
    User ec2-user
    Port 22
    IdentityFile ~/.ssh/prod-key.pem
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    
Host *.example.com
    User admin
    IdentityFile ~/.ssh/company-key.pem
    ForwardAgent yes
    
Host bastion
    HostName bastion.example.com
    User ec2-user
    IdentityFile ~/.ssh/bastion-key.pem
    
Host private-server
    HostName 10.0.2.100
    User ec2-user
    IdentityFile ~/.ssh/prod-key.pem
    ProxyJump bastion
EOF

chmod 600 ~/.ssh/config

# Now connect with aliases
ssh prod-web
ssh private-server  # Automatically goes through bastion
\`\`\`

## Key-Based Authentication

\`\`\`bash
# Generate SSH key pair
ssh-keygen -t ed25519 -C "myapp-production" -f ~/.ssh/prod-key
# Or RSA (older)
ssh-keygen -t rsa -b 4096 -C "myapp-production" -f ~/.ssh/prod-key-rsa

# Copy public key to server
ssh-copy-id -i ~/.ssh/prod-key.pub user@hostname

# Manual installation
cat ~/.ssh/prod-key.pub | ssh user@hostname 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'

# Set correct permissions on server
ssh user@hostname 'chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys'

# Test key-based auth
ssh -i ~/.ssh/prod-key user@hostname

# Disable password authentication (after keys work!)
sudo vi /etc/ssh/sshd_config
# Set: PasswordAuthentication no
sudo systemctl restart sshd
\`\`\`

## SSH Hardening

\`\`\`bash
# /etc/ssh/sshd_config - Production hardening
sudo cat << 'EOF' > /etc/ssh/sshd_config
# Port and protocol
Port 22
Protocol 2
AddressFamily inet

# Authentication
PermitRootLogin no                    # NEVER allow root SSH
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
PasswordAuthentication no             # Only key-based auth
ChallengeResponseAuthentication no
UsePAM yes

# Kerberos and GSSAPI
KerberosAuthentication no
GSSAPIAuthentication no

# Timeout and session
ClientAliveInterval 300               # 5 minutes
ClientAliveCountMax 2                 # Disconnect after 2 missed keepalives
LoginGraceTime 60                     # 60s to complete login
MaxStartups 10:30:60                  # Connection rate limiting
MaxSessions 10                        # Max sessions per connection

# Access control
AllowUsers ec2-user appuser admin     # Whitelist users
DenyUsers baduser
AllowGroups ssh-users
# Or use Match blocks for complex rules

# Security
PermitEmptyPasswords no
X11Forwarding no                      # Disable unless needed
PermitUserEnvironment no
HostbasedAuthentication no
IgnoreRhosts yes

# Cryptography
Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com,aes128-gcm@openssh.com
MACs hmac-sha2-512-etm@openssh.com,hmac-sha2-256-etm@openssh.com
KexAlgorithms curve25519-sha256,curve25519-sha256@libssh.org,diffie-hellman-group16-sha512

# Logging
SyslogFacility AUTHPRIV
LogLevel VERBOSE                      # Log key fingerprints

# Subsystems
Subsystem sftp /usr/libexec/openssh/sftp-server

# Banner
Banner /etc/ssh/banner.txt
EOF

sudo systemctl restart sshd

# Verify config
sudo sshd -t

# Monitor SSH attempts
sudo tail -f /var/log/secure  # RHEL/Amazon Linux
sudo tail -f /var/log/auth.log  # Debian/Ubuntu
\`\`\`

## Bastion Host Architecture

\`\`\`
Internet → ALB/NLB → Bastion (Public Subnet) → Private EC2 Instances (Private Subnet)
\`\`\`

\`\`\`terraform
# Bastion host in public subnet
resource "aws_instance" "bastion" {
  ami           = data.aws_ami.amazon_linux_2023.id
  instance_type = "t3.micro"
  subnet_id     = aws_subnet.public.id
  
  vpc_security_group_ids = [aws_security_group.bastion.id]
  
  key_name = aws_key_pair.bastion.key_name
  
  iam_instance_profile = aws_iam_instance_profile.bastion.name
  
  user_data = <<-EOF
              #!/bin/bash
              yum update -y
              # Harden SSH
              sed -i 's/^#PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
              sed -i 's/^PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
              systemctl restart sshd
              
              # Install fail2ban
              yum install -y fail2ban
              systemctl enable fail2ban
              systemctl start fail2ban
              EOF
  
  tags = {
    Name = "bastion-host"
  }
}

# Security group for bastion
resource "aws_security_group" "bastion" {
  name        = "bastion-sg"
  description = "Security group for bastion host"
  vpc_id      = aws_vpc.main.id
  
  # SSH from specific IPs only
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["1.2.3.4/32", "5.6.7.8/32"]  # Your office IPs
    description = "SSH from office"
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Security group for private instances
resource "aws_security_group" "private_instances" {
  name        = "private-instances-sg"
  vpc_id      = aws_vpc.main.id
  
  # SSH from bastion only
  ingress {
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    security_groups = [aws_security_group.bastion.id]
    description     = "SSH from bastion"
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# SSH connection through bastion
# ~/.ssh/config
/*
Host bastion
    HostName <bastion-public-ip>
    User ec2-user
    IdentityFile ~/.ssh/bastion-key.pem

Host private-*
    User ec2-user
    IdentityFile ~/.ssh/private-key.pem
    ProxyJump bastion
*/
\`\`\`

## AWS Systems Manager Session Manager

\`\`\`terraform
# NO SSH needed! Connect via Session Manager
resource "aws_iam_role" "ssm_role" {
  name = "ssm-instance-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm_policy" {
  role       = aws_iam_role.ssm_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ssm_profile" {
  name = "ssm-instance-profile"
  role = aws_iam_role.ssm_role.name
}

resource "aws_instance" "private" {
  ami           = data.aws_ami.amazon_linux_2023.id
  instance_type = "t3.micro"
  subnet_id     = aws_subnet.private.id
  
  iam_instance_profile = aws_iam_instance_profile.ssm_profile.name
  
  # NO security group ingress needed for SSH!
  vpc_security_group_ids = [aws_security_group.private_no_ssh.id]
  
  tags = {
    Name = "private-instance"
  }
}

# Connect via Session Manager (AWS CLI)
# aws ssm start-session --target i-1234567890abcdef0

# Or via AWS Console → Systems Manager → Session Manager → Start session
\`\`\`

## SSH Tunneling & Port Forwarding

\`\`\`bash
# Local port forwarding (access remote service locally)
ssh -L 8080:localhost:80 user@remote-server
# Now browse to http://localhost:8080 (connects to remote:80)

# Remote port forwarding (expose local service remotely)
ssh -R 8080:localhost:3000 user@remote-server
# Remote users can access your local:3000 via remote:8080

# Dynamic port forwarding (SOCKS proxy)
ssh -D 1080 user@remote-server
# Configure browser to use localhost:1080 as SOCKS proxy

# Database tunnel
ssh -L 5432:database.internal:5432 user@bastion
# Now connect to localhost:5432 (routes through bastion to database)

# Keep tunnel open
autossh -M 0 -L 5432:database.internal:5432 user@bastion
\`\`\`

## Multiplexing & Session Management

\`\`\`bash
# tmux - persistent sessions
tmux new -s work
tmux attach -t work
tmux ls
tmux kill-session -t work

# Common tmux commands (Ctrl+b prefix)
Ctrl+b c    # New window
Ctrl+b n    # Next window
Ctrl+b p    # Previous window
Ctrl+b d    # Detach session
Ctrl+b %    # Split vertically
Ctrl+b "    # Split horizontally
Ctrl+b [    # Scroll mode

# SSH multiplexing (reuse connections)
cat << 'EOF' >> ~/.ssh/config
Host *
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600
EOF

mkdir -p ~/.ssh/sockets
\`\`\`

## Best Practices

✅ **Use key-based authentication** (never passwords)  
✅ **Disable root login** (PermitRootLogin no)  
✅ **Restrict SSH access** by IP (security groups)  
✅ **Use bastion hosts** for private instances  
✅ **Consider Session Manager** (no public SSH)  
✅ **Enable SSH logging** (LogLevel VERBOSE)  
✅ **Implement fail2ban** (brute force protection)  
✅ **Use SSH config** for complex setups  
✅ **Regular key rotation** (every 90 days)  
✅ **Monitor SSH sessions** (who, last, w commands)

## Security Considerations

- **Never commit private keys** to git
- **Use separate keys** per environment
- **Implement MFA** for critical systems
- **Audit SSH access** logs regularly
- **Use certificate-based authentication** for large fleets
- **Implement session recording** (Teleport, asciinema)
- **Time-bound access** (temporary keys)
- **Principle of least privilege** (AllowUsers)

## Next Steps

In the next section, we'll cover **Security Hardening**, including firewall configuration, SELinux/AppArmor, intrusion detection, and security scanning.`,
};
