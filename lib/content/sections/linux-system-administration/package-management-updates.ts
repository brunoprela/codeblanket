/**
 * Package Management & Updates Section
 * Module: Linux System Administration & DevOps Foundations
 */

export const packageManagementUpdatesSection = {
  id: 'package-management-updates',
  title: 'Package Management & Updates',
  content: `# Package Management & Updates

## Introduction

Package management is central to maintaining secure, stable, and up-to-date Linux systems. Understanding package managers (yum/dnf, apt), repository management, version pinning, security updates, and automated patching is essential for production DevOps. This section covers comprehensive package management strategies for Amazon Linux, RHEL, and Debian-based systems.

## Package Managers Overview

### DNF/YUM (RHEL/CentOS/Amazon Linux)

**DNF** (Dandified YUM) is the next-generation package manager for RPM-based distributions. Amazon Linux 2023 uses DNF by default.

\`\`\`bash
# Check version
dnf --version
# OR
yum --version

# Update package lists
dnf check-update
yum check-update

# List installed packages
dnf list installed
yum list installed

# Search for packages
dnf search nginx
yum search nginx

# Show package information
dnf info nginx
yum info nginx

# Install package
sudo dnf install nginx
sudo yum install nginx

# Install specific version
sudo dnf install nginx-1.20.1

# Install local RPM
sudo dnf install ./package.rpm
sudo rpm -ivh package.rpm

# Remove package
sudo dnf remove nginx
sudo yum remove nginx

# Autoremove unused dependencies
sudo dnf autoremove
sudo yum autoremove

# Clean cache
sudo dnf clean all
sudo yum clean all

# List available updates
dnf list updates
yum list updates

# Update all packages
sudo dnf update
sudo yum update

# Update specific package
sudo dnf update nginx
sudo yum update nginx

# Downgrade package
sudo dnf downgrade nginx

# View transaction history
dnf history
yum history

# Undo transaction
sudo dnf history undo 5
sudo yum history undo 5

# Redo transaction
sudo dnf history redo 5
\`\`\`

### APT (Debian/Ubuntu)

**APT** (Advanced Package Tool) is the package manager for Debian-based distributions.

\`\`\`bash
# Update package lists
sudo apt update

# Upgrade all packages (safe)
sudo apt upgrade

# Full upgrade (may remove packages)
sudo apt full-upgrade
sudo apt dist-upgrade

# Install package
sudo apt install nginx

# Install specific version
sudo apt install nginx=1.18.0-0ubuntu1

# Remove package (keep config)
sudo apt remove nginx

# Remove package and config
sudo apt purge nginx

# Autoremove unused dependencies
sudo apt autoremove

# Clean downloaded packages
sudo apt clean
sudo apt autoclean

# Search packages
apt search nginx
apt-cache search nginx

# Show package info
apt show nginx
apt-cache show nginx

# List installed packages
apt list --installed
dpkg -l

# Find which package provides a file
dpkg -S /usr/bin/nginx
# nginx-core: /usr/bin/nginx

# List files in package
dpkg -L nginx

# Check if package is installed
dpkg -l | grep nginx

# Hold package version (prevent updates)
sudo apt-mark hold nginx

# Unhold package
sudo apt-mark unhold nginx

# Show held packages
apt-mark showhold
\`\`\`

## Repository Management

### DNF/YUM Repositories

\`\`\`bash
# List enabled repositories
dnf repolist
yum repolist

# List all repositories (including disabled)
dnf repolist --all
yum repolist all

# Enable repository
sudo dnf config-manager --enable epel
sudo yum-config-manager --enable epel

# Disable repository
sudo dnf config-manager --disable epel

# Add repository
sudo dnf config-manager --add-repo https://example.com/repo.repo

# Install from specific repository
sudo dnf install --enablerepo=epel package-name

# View repository configuration
cat /etc/yum.repos.d/epel.repo
\`\`\`

### Amazon Linux 2023 Repositories

\`\`\`bash
# Default AL2023 repositories
cat /etc/yum.repos.d/amazonlinux.repo

[amazonlinux]
name=Amazon Linux 2023 repository
baseurl=https://cdn.amazonlinux.com/al2023/core/mirrors/$releasever/$basearch/
enabled=1
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-amazon-linux-2023

# Enable EPEL for AL2023
sudo dnf install -y epel-release

# Verify EPEL
dnf repolist | grep epel
\`\`\`

### APT Repositories

\`\`\`bash
# Repository configuration
cat /etc/apt/sources.list

# deb http://archive.ubuntu.com/ubuntu/ jammy main restricted
# deb http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted
# deb http://security.ubuntu.com/ubuntu/ jammy-security main restricted

# Add repository (modern method)
sudo add-apt-repository ppa:nginx/stable
sudo apt update

# Add repository manually
echo "deb [signed-by=/usr/share/keyrings/nginx-keyring.gpg] http://nginx.org/packages/ubuntu jammy nginx" | \
  sudo tee /etc/apt/sources.list.d/nginx.list

# Add GPG key
curl -fsSL https://nginx.org/keys/nginx_signing.key | \
  sudo gpg --dearmor -o /usr/share/keyrings/nginx-keyring.gpg

# Update after adding repository
sudo apt update

# Remove repository
sudo add-apt-repository --remove ppa:nginx/stable

# Or delete file
sudo rm /etc/apt/sources.list.d/nginx.list
\`\`\`

## Version Management and Pinning

### DNF/YUM Version Pinning

\`\`\`bash
# Install specific version
sudo dnf install nginx-1.20.1-1.el9

# Exclude package from updates
echo "exclude=nginx*" | sudo tee -a /etc/dnf/dnf.conf

# Or per-command
sudo dnf update --exclude=nginx

# Use versionlock plugin
sudo dnf install python3-dnf-plugin-versionlock

# Lock package version
sudo dnf versionlock add nginx

# List locked packages
dnf versionlock list

# Remove lock
sudo dnf versionlock delete nginx

# Clear all locks
sudo dnf versionlock clear
\`\`\`

### APT Version Pinning

\`\`\`bash
# Hold package at current version
sudo apt-mark hold nginx

# Unhold package
sudo apt-mark unhold nginx

# Show held packages
apt-mark showhold

# Advanced pinning with preferences
sudo cat << 'EOF' > /etc/apt/preferences.d/nginx
Package: nginx
Pin: version 1.18.*
Pin-Priority: 1001
EOF

# Pin priorities:
# < 0:    Never install
# 0-100:  Install only if no other version
# 100-500: Install if not currently installed
# 500-990: Install unless from target release
# 990-1000: Install even if downgrade
# > 1000:  Install even if downgrade, override

# Prevent specific version
sudo cat << 'EOF' > /etc/apt/preferences.d/nginx-block
Package: nginx
Pin: version 1.19.*
Pin-Priority: -1
EOF

sudo apt update
\`\`\`

## Security Updates

### Automatic Security Updates (Amazon Linux)

\`\`\`bash
# Install dnf-automatic
sudo dnf install -y dnf-automatic

# Configure for security-only updates
sudo vi /etc/dnf/automatic.conf
# [commands]
# upgrade_type = security
# download_updates = yes
# apply_updates = yes
#
# [emitters]
# emit_via = stdio
# email_from = root@localhost
# email_to = admin@example.com

# Enable timer
sudo systemctl enable --now dnf-automatic.timer

# Check status
systemctl status dnf-automatic.timer

# List timers
systemctl list-timers | grep dnf-automatic

# Manual security update
sudo dnf update --security
sudo dnf update-minimal --security

# List security advisories
dnf updateinfo list security

# Show details
dnf updateinfo info ALAS-2023-1234

# Update for specific CVE
dnf update --cve CVE-2023-12345
\`\`\`

### Automatic Security Updates (Ubuntu)

\`\`\`bash
# Install unattended-upgrades
sudo apt install -y unattended-upgrades

# Configure
sudo dpkg-reconfigure -plow unattended-upgrades

# Manual configuration
sudo vi /etc/apt/apt.conf.d/50unattended-upgrades
# Unattended-Upgrade::Allowed-Origins {
#     "\${distro_id}:\${distro_codename}-security";
#     "\${distro_id}ESMApps:\${distro_codename}-apps-security";
# };
#
# Unattended-Upgrade::AutoFixInterruptedDpkg "true";
# Unattended-Upgrade::MinimalSteps "true";
# Unattended-Upgrade::Remove-Unused-Dependencies "true";
# Unattended-Upgrade::Automatic-Reboot "false";
# Unattended-Upgrade::Automatic-Reboot-Time "02:00";

# Enable automatic updates
sudo vi /etc/apt/apt.conf.d/20auto-upgrades
# APT::Periodic::Update-Package-Lists "1";
# APT::Periodic::Download-Upgradeable-Packages "1";
# APT::Periodic::AutocleanInterval "7";
# APT::Periodic::Unattended-Upgrade "1";

# Test unattended-upgrades
sudo unattended-upgrades --dry-run --debug

# View logs
sudo tail -f /var/log/unattended-upgrades/unattended-upgrades.log

# List security updates
apt list --upgradable | grep security
\`\`\`

## Production Update Strategies

### Blue-Green Deployment for Updates

\`\`\`bash
# Terraform: Create new launch template with updated AMI
resource "aws_launch_template" "app_v2" {
  name_prefix   = "app-v2-"
  image_id      = data.aws_ami.updated_ami.id  # New AMI with updates
  instance_type = "t3.medium"
  
  user_data = base64encode(<<-EOF
    #!/bin/bash
    dnf update -y
    dnf install -y my-app-latest
  EOF
  )
}

# Create new Auto Scaling Group
resource "aws_autoscaling_group" "app_v2" {
  name                = "app-asg-v2"
  vpc_zone_identifier = aws_subnet.private[*].id
  target_group_arns   = [aws_lb_target_group.app.arn]
  health_check_type   = "ELB"
  
  min_size = 3
  max_size = 10
  
  launch_template {
    id      = aws_launch_template.app_v2.id
    version = "$Latest"
  }
  
  tag {
    key                 = "Version"
    value               = "v2"
    propagate_at_launch = true
  }
}

# After validation, scale down v1 ASG
# Then delete v1 resources
\`\`\`

### Rolling Updates with ASG

\`\`\`terraform
resource "aws_autoscaling_group" "app" {
  name                = "app-asg"
  vpc_zone_identifier = aws_subnet.private[*].id
  target_group_arns   = [aws_lb_target_group.app.arn]
  health_check_type   = "ELB"
  health_check_grace_period = 300
  
  min_size         = 3
  max_size         = 10
  desired_capacity = 3
  
  launch_template {
    id      = aws_launch_template.app.id
    version = "$Latest"
  }
  
  # Instance refresh for rolling updates
  instance_refresh {
    strategy = "Rolling"
    preferences {
      min_healthy_percentage = 66  # Keep 2/3 healthy
      instance_warmup        = 300
    }
  }
  
  # Update strategy
  lifecycle {
    create_before_destroy = true
  }
}

# Trigger instance refresh when launch template changes
resource "null_resource" "trigger_instance_refresh" {
  triggers = {
    launch_template_version = aws_launch_template.app.latest_version
  }
  
  provisioner "local-exec" {
    command = <<-EOF
      aws autoscaling start-instance-refresh \
        --auto-scaling-group-name \${aws_autoscaling_group.app.name} \
        --preferences MinHealthyPercentage=66,InstanceWarmup=300
    EOF
  }
}
\`\`\`

### Staged Update Process

\`\`\`bash
#!/bin/bash
# Staged update script for production

set -e

echo "=== Production Update Process ==="

# 1. Update dev environment
echo "Step 1: Updating DEV environment..."
ssh dev-server "sudo dnf update -y && sudo systemctl restart myapp"
sleep 300  # Wait 5 minutes

# 2. Run smoke tests on dev
echo "Step 2: Running smoke tests on DEV..."
curl -f http://dev-server/health || exit 1

# 3. Update staging
echo "Step 3: Updating STAGING environment..."
for server in staging-{1..3}; do
    ssh $server "sudo dnf update -y && sudo systemctl restart myapp"
    sleep 60
done
sleep 600  # Wait 10 minutes

# 4. Run integration tests on staging
echo "Step 4: Running integration tests on STAGING..."
./run-integration-tests.sh || exit 1

# 5. Update production (canary)
echo "Step 5: Updating PRODUCTION (1 instance)..."
ssh prod-1 "sudo dnf update -y && sudo systemctl restart myapp"
sleep 600  # Wait 10 minutes

# 6. Monitor canary
echo "Step 6: Monitoring canary..."
./monitor-canary.sh prod-1 || exit 1

# 7. Update remaining production instances (rolling)
echo "Step 7: Rolling update to remaining PRODUCTION instances..."
for server in prod-{2..10}; do
    echo "Updating $server..."
    # Drain connections
    aws elbv2 modify-target-group-attributes \
      --target-group-arn $TG_ARN \
      --attributes Key=deregistration_delay.timeout_seconds,Value=30
    
    # Deregister from load balancer
    aws elbv2 deregister-targets \
      --target-group-arn $TG_ARN \
      --targets Id=$server
    
    sleep 45  # Wait for drain
    
    # Update
    ssh $server "sudo dnf update -y && sudo systemctl restart myapp"
    
    # Re-register
    aws elbv2 register-targets \
      --target-group-arn $TG_ARN \
      --targets Id=$server
    
    # Wait for healthy
    sleep 120
done

echo "=== Update complete ==="
\`\`\`

## Package Building and Custom RPMs

### Creating Custom RPM

\`\`\`bash
# Install build tools
sudo dnf install -y rpm-build rpmdevtools

# Setup build environment
rpmdev-setuptree
# Creates: ~/rpmbuild/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

# Create spec file
cat << 'EOF' > ~/rpmbuild/SPECS/myapp.spec
Name:           myapp
Version:        1.0.0
Release:        1%{?dist}
Summary:        My Production Application

License:        MIT
URL:            https://example.com/myapp
Source0:        myapp-1.0.0.tar.gz

BuildRequires:  nodejs >= 16
Requires:       nodejs >= 16

%description
My production application

%prep
%setup -q

%build
npm install --production

%install
mkdir -p %{buildroot}/opt/myapp
cp -r * %{buildroot}/opt/myapp/

mkdir -p %{buildroot}/etc/systemd/system
cat << 'UNIT' > %{buildroot}/etc/systemd/system/myapp.service
[Unit]
Description=My Application
After=network.target

[Service]
Type=simple
User=myapp
WorkingDirectory=/opt/myapp
ExecStart=/usr/bin/node /opt/myapp/server.js
Restart=always

[Install]
WantedBy=multi-user.target
UNIT

%files
/opt/myapp
/etc/systemd/system/myapp.service

%pre
getent group myapp >/dev/null || groupadd -r myapp
getent passwd myapp >/dev/null || useradd -r -g myapp -s /sbin/nologin myapp

%post
systemctl daemon-reload
systemctl enable myapp

%preun
if [ $1 -eq 0 ]; then
    systemctl stop myapp
    systemctl disable myapp
fi

%postun
systemctl daemon-reload

%changelog
* Mon Oct 28 2024 Admin <admin@example.com> - 1.0.0-1
- Initial release
EOF

# Build RPM
rpmbuild -ba ~/rpmbuild/SPECS/myapp.spec

# RPM created at:
# ~/rpmbuild/RPMS/x86_64/myapp-1.0.0-1.el9.x86_64.rpm

# Install
sudo dnf install ~/rpmbuild/RPMS/x86_64/myapp-1.0.0-1.el9.x86_64.rpm

# Sign RPM
rpmsign --addsign myapp-1.0.0-1.el9.x86_64.rpm
\`\`\`

### Creating Custom DEB

\`\`\`bash
# Install build tools
sudo apt install -y build-essential devscripts debhelper

# Create package directory
mkdir -p myapp-1.0.0/{DEBIAN,opt/myapp,etc/systemd/system}

# Create control file
cat << 'EOF' > myapp-1.0.0/DEBIAN/control
Package: myapp
Version: 1.0.0
Section: web
Priority: optional
Architecture: amd64
Depends: nodejs (>= 16)
Maintainer: Admin <admin@example.com>
Description: My Production Application
 Production web application
EOF

# Create postinst script
cat << 'EOF' > myapp-1.0.0/DEBIAN/postinst
#!/bin/bash
set -e

# Create user
if ! getent passwd myapp > /dev/null; then
    useradd -r -s /bin/false myapp
fi

# Enable service
systemctl daemon-reload
systemctl enable myapp

exit 0
EOF

chmod 755 myapp-1.0.0/DEBIAN/postinst

# Copy application files
cp -r src/* myapp-1.0.0/opt/myapp/

# Create systemd unit
cat << 'EOF' > myapp-1.0.0/etc/systemd/system/myapp.service
[Unit]
Description=My Application

[Service]
Type=simple
User=myapp
ExecStart=/usr/bin/node /opt/myapp/server.js
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Build package
dpkg-deb --build myapp-1.0.0

# Install
sudo dpkg -i myapp-1.0.0.deb
\`\`\`

## Monitoring and Reporting

### Package Update Monitoring

\`\`\`bash
# Check for available updates
dnf check-update --quiet
echo $?  # 100 if updates available

# List security updates
dnf updateinfo list security --available

# Create update report
cat << 'EOF' > /usr/local/bin/update-report.sh
#!/bin/bash
# Generate update report

OUTPUT="/var/log/update-report-$(date +%Y%m%d).txt"

{
    echo "=== Package Update Report ==="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo ""
    
    echo "=== Available Updates ==="
    dnf list updates
    echo ""
    
    echo "=== Security Updates ==="
    dnf updateinfo list security
    echo ""
    
    echo "=== CVE Information ==="
    dnf updateinfo list --cve
    echo ""
    
    echo "=== Package Hold Status ==="
    dnf versionlock list
    echo ""
} > "$OUTPUT"

# Send to monitoring
curl -X POST https://monitoring.example.com/reports \
  -H "Content-Type: application/json" \
  -d @"$OUTPUT"
EOF

chmod +x /usr/local/bin/update-report.sh

# Schedule weekly report
cat << 'EOF' > /etc/systemd/system/update-report.timer
[Unit]
Description=Weekly Update Report

[Timer]
OnCalendar=Mon *-*-* 09:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo systemctl enable --now update-report.timer
\`\`\`

## Best Practices

✅ **Use package managers** - never install from source in production  
✅ **Enable automatic security updates** - with proper testing  
✅ **Pin critical package versions** - prevent unexpected breaks  
✅ **Test updates in staging first** - never update prod directly  
✅ **Implement rolling updates** - minimize downtime  
✅ **Monitor for CVEs** - subscribe to security advisories  
✅ **Keep AMIs updated** - bake updates into golden images  
✅ **Use repository mirrors** - for reliability and speed  
✅ **Document update procedures** - runbooks for teams  
✅ **Maintain update logs** - audit trail for compliance

## Next Steps

In the next section, we'll cover **Process & Resource Management**, including CPU/memory management, cgroups, ulimits, and performance tuning for production workloads.`,
};
