/**
 * Shell Scripting for Automation Section
 * Module: Linux System Administration & DevOps Foundations
 */

export const shellScriptingAutomationSection = {
  id: 'shell-scripting-automation',
  title: 'Shell Scripting for Automation',
  content: `# Shell Scripting for Automation

## Introduction

Production DevOps relies heavily on shell scripting to automate repetitive tasks, deployments, and system maintenance. Well-written Bash scripts are idempotent, handle errors gracefully, and provide clear feedback. This section covers production-grade shell scripting techniques used in real AWS deployments.

## Bash Script Fundamentals

### Script Structure and Best Practices

\`\`\`bash
#!/bin/bash
#
# Script: deploy-application.sh
# Purpose: Deploy application to production
# Author: DevOps Team
# Date: 2024-10-28
#
# Usage: ./deploy-application.sh [environment] [version]
# Example: ./deploy-application.sh production v1.2.3

# Strict mode: Exit on error, undefined variables, and pipe failures
set -euo pipefail

# Set IFS to handle spaces in filenames correctly
IFS=$'\\n\\t'

# Script metadata
readonly SCRIPT_NAME=$(basename "$0")
readonly SCRIPT_DIR="$(cd "$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_PID=$$
readonly SCRIPT_START_TIME=$(date +%s)

# Configuration
readonly APP_NAME="myapp"
readonly DEPLOY_USER="deploy"
readonly LOG_DIR="/var/log/deployments"
readonly LOG_FILE="$LOG_DIR/deploy-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
readonly COLOR_RED='\\033[0;31m'
readonly COLOR_GREEN='\\033[0;32m'
readonly COLOR_YELLOW='\\033[1;33m'
readonly COLOR_NC='\\033[0m'  # No Color

# Logging functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "$LOG_FILE" >&2
}

log_warning() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [WARNING] $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo "\${COLOR_GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] [SUCCESS] $*\${COLOR_NC}" | tee -a "$LOG_FILE"
}

# Cleanup function (runs on exit)
cleanup() {
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - SCRIPT_START_TIME))
    
    if [ $exit_code -eq 0 ]; then
        log_success "Script completed successfully in \${duration}s"
    else
        log_error "Script failed with exit code $exit_code after \${duration}s"
    fi
    
    # Cleanup temp files
    rm -f /tmp/deploy-*-$$
}

# Set trap to call cleanup on exit
trap cleanup EXIT

# Trap errors
error_handler() {
    local line_number=$1
    local command=$2
    log_error "Error on line $line_number: $command"
}

trap 'error_handler \${LINENO} "$BASH_COMMAND"' ERR

# Main script logic
main() {
    log "Starting deployment script"
    # Your deployment logic here
}

# Run main function
main "$@"
\`\`\`

### Understanding Bash Options

\`\`\`bash
# set -e: Exit immediately if a command exits with non-zero status
# Example without set -e:
#!/bin/bash
false  # Exits with code 1
echo "This still runs"  # BAD: Script continues after error

# Example with set -e:
#!/bin/bash
set -e
false  # Exits with code 1
echo "This never runs"  # GOOD: Script stops after error

# set -u: Treat unset variables as errors
#!/bin/bash
set -u
echo "$UNDEFINED_VAR"  # Error: UNDEFINED_VAR: unbound variable

# set -o pipefail: Return exit code of rightmost failed command in pipeline
#!/bin/bash
set -o pipefail
false | true  # Without pipefail: exits 0. With pipefail: exits 1

# Combined (recommended for production):
set -euo pipefail
\`\`\`

## Error Handling and Exit Codes

### Proper Exit Code Usage

\`\`\`bash
#!/bin/bash
set -euo pipefail

# Exit code constants (make code self-documenting)
readonly EXIT_SUCCESS=0
readonly EXIT_ERROR_GENERAL=1
readonly EXIT_ERROR_USAGE=2
readonly EXIT_ERROR_PERMISSION=3
readonly EXIT_ERROR_NOT_FOUND=4
readonly EXIT_ERROR_NETWORK=5

# Function with error handling
backup_database() {
    local db_name=$1
    local backup_dir=$2
    local backup_file="\${backup_dir}/\${db_name}-$(date +%Y%m%d-%H%M%S).sql"
    
    log "Backing up database: $db_name"
    
    # Check if directory exists
    if [[ ! -d "$backup_dir" ]]; then
        log_error "Backup directory does not exist: $backup_dir"
        return $EXIT_ERROR_NOT_FOUND
    fi
    
    # Check write permission
    if [[ ! -w "$backup_dir" ]]; then
        log_error "No write permission for directory: $backup_dir"
        return $EXIT_ERROR_PERMISSION
    fi
    
    # Perform backup with error handling
    if ! mysqldump "$db_name" > "$backup_file" 2>/dev/null; then
        log_error "Database backup failed for: $db_name"
        rm -f "$backup_file"  # Clean up partial backup
        return $EXIT_ERROR_GENERAL
    fi
    
    # Verify backup file was created and is not empty
    if [[ ! -s "$backup_file" ]]; then
        log_error "Backup file is empty: $backup_file"
        rm -f "$backup_file"
        return $EXIT_ERROR_GENERAL
    fi
    
    log_success "Database backup completed: $backup_file"
    return $EXIT_SUCCESS
}

# Usage with error handling
if backup_database "production_db" "/backups"; then
    log_success "Backup succeeded"
else
    exit_code=$?
    log_error "Backup failed with exit code: $exit_code"
    exit $exit_code
fi
\`\`\`

### Try-Catch Pattern in Bash

\`\`\`bash
#!/bin/bash

# Bash doesn't have try-catch, but we can simulate it
try() {
    [[ $- = *e* ]]; SAVED_OPT_E=$?
    set +e
}

catch() {
    export exception_code=$?
    (( SAVED_OPT_E )) && set +e
    return $exception_code
}

# Usage example
try
(
    # Commands in subshell
    cd /nonexistent/directory
    echo "This won't execute if cd fails"
)
catch || {
    case $exception_code in
        1)
            log_error "Directory not found"
            ;;
        2)
            log_error "Permission denied"
            ;;
        *)
            log_error "Unknown error: $exception_code"
            ;;
    esac
}
\`\`\`

## Functions and Modular Scripts

### Writing Reusable Functions

\`\`\`bash
#!/bin/bash
set -euo pipefail

# Function library for AWS operations

# Validate AWS CLI is installed and configured
validate_aws_cli() {
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed"
        return 1
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS CLI is not configured or credentials are invalid"
        return 1
    fi
    
    log "AWS CLI validation successful"
    return 0
}

# Get EC2 instance ID from metadata service
get_instance_id() {
    local instance_id
    
    # Try IMDSv2 first (more secure)
    local token
    token=$(curl -X PUT "http://169.254.169.254/latest/api/token" \\
        -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" \\
        -s --connect-timeout 2 2>/dev/null)
    
    if [[ -n "$token" ]]; then
        instance_id=$(curl -H "X-aws-ec2-metadata-token: $token" \\
            -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null)
    else
        # Fall back to IMDSv1
        instance_id=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null)
    fi
    
    if [[ -z "$instance_id" ]]; then
        log_error "Failed to get instance ID from metadata service"
        return 1
    fi
    
    echo "$instance_id"
    return 0
}

# Get tag value from EC2 instance
get_instance_tag() {
    local instance_id=$1
    local tag_key=$2
    local tag_value
    
    tag_value=$(aws ec2 describe-tags \\
        --filters "Name=resource-id,Values=$instance_id" \\
                  "Name=key,Values=$tag_key" \\
        --query 'Tags[0].Value' \\
        --output text 2>/dev/null)
    
    if [[ -z "$tag_value" ]] || [[ "$tag_value" == "None" ]]; then
        log_warning "Tag not found: $tag_key"
        return 1
    fi
    
    echo "$tag_value"
    return 0
}

# Check if S3 bucket exists
s3_bucket_exists() {
    local bucket_name=$1
    
    if aws s3api head-bucket --bucket "$bucket_name" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Upload file to S3 with retry logic
s3_upload_with_retry() {
    local local_file=$1
    local s3_path=$2
    local max_attempts=3
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log "Uploading to S3 (attempt $attempt/$max_attempts): $s3_path"
        
        if aws s3 cp "$local_file" "$s3_path" --no-progress 2>&1 | tee -a "$LOG_FILE"; then
            log_success "Upload successful: $s3_path"
            return 0
        fi
        
        log_warning "Upload failed, attempt $attempt/$max_attempts"
        attempt=$((attempt + 1))
        
        if [[ $attempt -le $max_attempts ]]; then
            local sleep_time=$((attempt * 5))
            log "Retrying in \${sleep_time}s..."
            sleep "$sleep_time"
        fi
    done
    
    log_error "Upload failed after $max_attempts attempts: $s3_path"
    return 1
}

# Example usage
main() {
    validate_aws_cli || exit 1
    
    local instance_id
    instance_id=$(get_instance_id) || exit 1
    log "Running on instance: $instance_id"
    
    local environment
    environment=$(get_instance_tag "$instance_id" "Environment") || environment="unknown"
    log "Environment: $environment"
    
    # Perform operations based on environment
    case "$environment" in
        production)
            log "Running production deployment"
            ;;
        staging)
            log "Running staging deployment"
            ;;
        *)
            log_error "Unknown environment: $environment"
            exit 1
            ;;
    esac
}

main "$@"
\`\`\`

## Text Processing with awk, sed, and grep

### grep: Pattern Matching

\`\`\`bash
#!/bin/bash

# Basic grep usage
grep "ERROR" /var/log/application.log

# Case-insensitive search
grep -i "error" /var/log/application.log

# Show line numbers
grep -n "ERROR" /var/log/application.log

# Count occurrences
grep -c "ERROR" /var/log/application.log

# Inverted match (lines that DON'T match)
grep -v "DEBUG" /var/log/application.log

# Multiple patterns (OR)
grep -E "ERROR|FATAL|CRITICAL" /var/log/application.log

# Recursive search in directory
grep -r "TODO" /opt/application/

# Show context (lines before/after match)
grep -B 3 -A 3 "ERROR" /var/log/application.log
# -B: before context
# -A: after context
# -C: both (same as -B N -A N)

# Extended regex
grep -E "^[0-9]{4}-[0-9]{2}-[0-9]{2}" /var/log/application.log

# Quiet mode (just return exit code)
if grep -q "ERROR" /var/log/application.log; then
    echo "Errors found in log"
fi

# Real-world example: Extract error count by hour
extract_hourly_errors() {
    local log_file=$1
    
    grep "ERROR" "$log_file" | \\
        awk '{print $1, $2}' | \\
        cut -d: -f1 | \\
        sort | \\
        uniq -c | \\
        sort -rn
    
    # Output:
    #  45 2024-10-28 14
    #  23 2024-10-28 13
    #  12 2024-10-28 15
}
\`\`\`

### sed: Stream Editor

\`\`\`bash
#!/bin/bash

# Basic substitution
echo "Hello World" | sed 's/World/Universe/'
# Output: Hello Universe

# Global substitution (all occurrences on line)
echo "foo foo foo" | sed 's/foo/bar/g'
# Output: bar bar bar

# In-place editing (modify file directly)
sed -i 's/old_value/new_value/g' config.txt

# In-place with backup
sed -i.bak 's/old_value/new_value/g' config.txt

# Delete lines matching pattern
sed '/^#/d' file.txt  # Delete comment lines

# Delete empty lines
sed '/^$/d' file.txt

# Print specific lines
sed -n '10,20p' file.txt  # Print lines 10-20

# Multiple operations
sed -e 's/foo/bar/g' -e 's/baz/qux/g' file.txt

# Replace only on lines matching pattern
sed '/^server/s/80/8080/' nginx.conf

# Real-world example: Update configuration file
update_config() {
    local config_file=$1
    local old_db_host=$2
    local new_db_host=$3
    
    # Backup original
    cp "$config_file" "\${config_file}.backup"
    
    # Update database host
    sed -i "s/^DB_HOST=.*/DB_HOST=$new_db_host/" "$config_file"
    
    # Verify change was made
    if grep -q "^DB_HOST=$new_db_host" "$config_file"; then
        log_success "Configuration updated successfully"
        rm "\${config_file}.backup"
        return 0
    else
        log_error "Configuration update failed"
        mv "\${config_file}.backup" "$config_file"
        return 1
    fi
}

# Advanced: Comment out lines
sed -i '/^DEBUG_MODE=/s/^/# /' config.txt

# Remove comments
sed 's/#.*//' config.txt

# Extract values from config
get_config_value() {
    local config_file=$1
    local key=$2
    
    grep "^\${key}=" "$config_file" | sed "s/^\${key}=//" | tr -d '"'"'"
}

# Usage
db_host=$(get_config_value "config.txt" "DB_HOST")
\`\`\`

### awk: Text Processing Powerhouse

\`\`\`bash
#!/bin/bash

# Print specific columns
ps aux | awk '{print $1, $11}'  # Print user and command

# Print with custom delimiter
awk -F: '{print $1, $3}' /etc/passwd  # Print username and UID

# Conditional printing
awk '$3 > 1000 {print $1}' /etc/passwd  # Users with UID > 1000

# Sum column values
ps aux | awk '{sum+=$3} END {print "Total CPU:", sum "%"}'

# Average
ps aux | awk '{sum+=$4; count++} END {print "Avg Memory:", sum/count "%"}'

# Pattern matching
awk '/ERROR/ {print $0}' application.log

# Multiple conditions
awk '$3 > 50 && $4 > 10 {print $0}' process_stats.txt

# Print line numbers
awk '{print NR, $0}' file.txt

# BEGIN and END blocks
awk 'BEGIN {print "Starting..."} {print $0} END {print "Done"}' file.txt

# Real-world example: Parse AWS CLI JSON output
parse_ec2_instances() {
    aws ec2 describe-instances \\
        --filters "Name=tag:Environment,Values=production" \\
        --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PrivateIpAddress]' \\
        --output text | \\
    awk '{
        printf "Instance: %-20s State: %-10s IP: %s\\n", $1, $2, $3
    }'
    
    # Output:
    # Instance: i-0123456789abcdef0 State: running    IP: 10.0.1.50
    # Instance: i-abcdef0123456789 State: running    IP: 10.0.1.51
}

# Calculate disk usage percentage by directory
analyze_disk_usage() {
    du -sh /var/* | \\
    awk '{
        # Extract size with unit
        size=$1
        dir=$2
        
        # Store for sorting
        data[dir] = size
    }
    END {
        # Sort and print
        for (dir in data) {
            print data[dir], dir
        }
    }' | sort -hr | head -10
}

# Log analysis: Count HTTP status codes
analyze_http_logs() {
    local log_file=$1
    
    awk '{
        # Assuming Apache/Nginx combined log format
        # Status code is typically field 9
        status = $9
        status_counts[status]++
        total++
    }
    END {
        print "HTTP Status Code Distribution:"
        print "================================"
        for (status in status_counts) {
            percentage = (status_counts[status] / total) * 100
            printf "Status %s: %d requests (%.2f%%)\\n", \\
                status, status_counts[status], percentage
        }
    }' "$log_file"
}

# Advanced: Calculate percentiles
calculate_percentiles() {
    local metric_file=$1
    
    awk '{
        values[NR] = $1
        sum += $1
        count++
    }
    END {
        # Sort values
        asort(values)
        
        # Calculate statistics
        mean = sum / count
        p50 = values[int(count * 0.50)]
        p95 = values[int(count * 0.95)]
        p99 = values[int(count * 0.99)]
        
        printf "Count: %d\\n", count
        printf "Mean: %.2f\\n", mean
        printf "p50: %.2f\\n", p50
        printf "p95: %.2f\\n", p95
        printf "p99: %.2f\\n", p99
    }' "$metric_file"
}
\`\`\`

## JSON Processing with jq

### jq Fundamentals

\`\`\`bash
#!/bin/bash

# Basic jq usage

# Pretty print JSON
echo '{"name":"John","age":30}' | jq '.'

# Extract field
echo '{"name":"John","age":30}' | jq '.name'
# Output: "John"

# Raw output (without quotes)
echo '{"name":"John","age":30}' | jq -r '.name'
# Output: John

# Array indexing
echo '["apple","banana","cherry"]' | jq '.[1]'
# Output: "banana"

# Array slicing
echo '[1,2,3,4,5]' | jq '.[1:3]'
# Output: [2,3]

# Filter arrays
echo '[{"name":"John","age":30},{"name":"Jane","age":25}]' | \\
    jq '.[] | select(.age > 26)'

# Map over array
echo '[1,2,3]' | jq 'map(. * 2)'
# Output: [2,4,6]

# Real-world AWS examples

# Extract instance IDs
get_instance_ids() {
    aws ec2 describe-instances \\
        --filters "Name=tag:Environment,Values=production" \\
        --output json | \\
    jq -r '.Reservations[].Instances[].InstanceId'
}

# Get instances with specific tags
get_tagged_instances() {
    local tag_key=$1
    local tag_value=$2
    
    aws ec2 describe-instances \\
        --filters "Name=tag:\${tag_key},Values=\${tag_value}" \\
        --output json | \\
    jq -r '.Reservations[].Instances[] | {
        InstanceId: .InstanceId,
        State: .State.Name,
        PrivateIP: .PrivateIpAddress,
        Name: (.Tags[]? | select(.Key=="Name") | .Value)
    }'
}

# Extract S3 bucket names and sizes
analyze_s3_buckets() {
    aws s3api list-buckets --output json | \\
    jq -r '.Buckets[] | .Name' | \\
    while read -r bucket; do
        size=$(aws s3 ls "s3://$bucket" --recursive --summarize 2>/dev/null | \\
               grep "Total Size" | awk '{print $3}')
        
        if [[ -n "$size" ]]; then
            echo "$bucket: $size bytes"
        fi
    done
}

# Complex: Build deployment manifest from AWS resources
generate_deployment_manifest() {
    local environment=$1
    
    jq -n \\
        --arg env "$environment" \\
        --argjson instances "$(aws ec2 describe-instances \\
            --filters "Name=tag:Environment,Values=$environment" \\
            --query 'Reservations[].Instances[].{id:InstanceId,ip:PrivateIpAddress}' \\
            --output json)" \\
        --argjson rds "$(aws rds describe-db-instances \\
            --query 'DBInstances[].{id:DBInstanceIdentifier,endpoint:Endpoint.Address}' \\
            --output json)" \\
        '{
            environment: $env,
            timestamp: now | strftime("%Y-%m-%d %H:%M:%S"),
            instances: $instances,
            databases: $rds
        }'
}
\`\`\`

## Idempotent Scripts

### Writing Idempotent Operations

\`\`\`bash
#!/bin/bash
set -euo pipefail

# Idempotent: Can run multiple times with same result

# BAD: Not idempotent
install_package_bad() {
    apt-get install nginx
    # Fails if already installed
}

# GOOD: Idempotent
install_package_good() {
    if ! dpkg -l | grep -q "^ii  nginx "; then
        log "Installing nginx..."
        apt-get install -y nginx
    else
        log "nginx already installed, skipping"
    fi
}

# Idempotent file creation
create_config_file() {
    local config_file="/etc/myapp/config.conf"
    local config_dir=$(dirname "$config_file")
    
    # Create directory if it doesn't exist
    if [[ ! -d "$config_dir" ]]; then
        log "Creating directory: $config_dir"
        mkdir -p "$config_dir"
    fi
    
    # Create config file if it doesn't exist
    if [[ ! -f "$config_file" ]]; then
        log "Creating config file: $config_file"
        cat > "$config_file" << 'EOF'
# Application Configuration
DB_HOST=localhost
DB_PORT=5432
APP_PORT=8000
EOF
        chmod 644 "$config_file"
    else
        log "Config file already exists: $config_file"
    fi
}

# Idempotent configuration update
update_config_value() {
    local config_file=$1
    local key=$2
    local value=$3
    
    # Check if key exists
    if grep -q "^\${key}=" "$config_file"; then
        # Key exists - check if value needs updating
        current_value=$(grep "^\${key}=" "$config_file" | cut -d= -f2)
        
        if [[ "$current_value" != "$value" ]]; then
            log "Updating $key: $current_value -> $value"
            sed -i "s/^\${key}=.*/\${key}=\${value}/" "$config_file"
        else
            log "$key already set to $value, skipping"
        fi
    else
        # Key doesn't exist - add it
        log "Adding $key=$value to config"
        echo "\${key}=\${value}" >> "$config_file"
    fi
}

# Idempotent systemd service management
ensure_service_running() {
    local service_name=$1
    
    if systemctl is-active --quiet "$service_name"; then
        log "Service $service_name is already running"
    else
        log "Starting service: $service_name"
        systemctl start "$service_name"
    fi
    
    if systemctl is-enabled --quiet "$service_name"; then
        log "Service $service_name is already enabled"
    else
        log "Enabling service: $service_name"
        systemctl enable "$service_name"
    fi
}

# Idempotent AWS security group rule
ensure_security_group_rule() {
    local group_id=$1
    local port=$2
    local cidr=$3
    
    # Check if rule exists
    if aws ec2 describe-security-groups \\
        --group-ids "$group_id" \\
        --output json | \\
        jq -e ".SecurityGroups[].IpPermissions[] | \\
               select(.FromPort==$port and .ToPort==$port and \\
                      .IpRanges[].CidrIp==\\"$cidr\\")" > /dev/null; then
        log "Security group rule already exists"
    else
        log "Adding security group rule: port $port from $cidr"
        aws ec2 authorize-security-group-ingress \\
            --group-id "$group_id" \\
            --protocol tcp \\
            --port "$port" \\
            --cidr "$cidr"
    fi
}
\`\`\`

## AWS CLI Scripting Patterns

### Common AWS Operations

\`\`\`bash
#!/bin/bash
set -euo pipefail

# Get current region
get_aws_region() {
    aws configure get region || echo "us-east-1"
}

# Get account ID
get_aws_account_id() {
    aws sts get-caller-identity --query Account --output text
}

# Wait for EC2 instance to be running
wait_for_instance() {
    local instance_id=$1
    local max_wait=300  # 5 minutes
    local waited=0
    
    log "Waiting for instance $instance_id to be running..."
    
    while [[ $waited -lt $max_wait ]]; do
        local state
        state=$(aws ec2 describe-instances \\
            --instance-ids "$instance_id" \\
            --query 'Reservations[0].Instances[0].State.Name' \\
            --output text)
        
        if [[ "$state" == "running" ]]; then
            log_success "Instance is running"
            return 0
        fi
        
        log "Instance state: $state (waited \${waited}s)"
        sleep 10
        waited=$((waited + 10))
    done
    
    log_error "Timeout waiting for instance to start"
    return 1
}

# Safely stop EC2 instance with checks
stop_instance_safely() {
    local instance_id=$1
    
    # Check if instance exists and is running
    local state
    state=$(aws ec2 describe-instances \\
        --instance-ids "$instance_id" \\
        --query 'Reservations[0].Instances[0].State.Name' \\
        --output text 2>/dev/null) || {
        log_error "Instance not found: $instance_id"
        return 1
    }
    
    if [[ "$state" != "running" ]]; then
        log "Instance is not running (state: $state), skipping stop"
        return 0
    fi
    
    # Check for protection
    local termination_protection
    termination_protection=$(aws ec2 describe-instance-attribute \\
        --instance-id "$instance_id" \\
        --attribute disableApiTermination \\
        --query 'DisableApiTermination.Value' \\
        --output text)
    
    if [[ "$termination_protection" == "true" ]]; then
        log_warning "Instance has termination protection enabled"
    fi
    
    # Stop instance
    log "Stopping instance: $instance_id"
    aws ec2 stop-instances --instance-ids "$instance_id" --output json
    
    # Wait for stopped state
    aws ec2 wait instance-stopped --instance-ids "$instance_id"
    log_success "Instance stopped: $instance_id"
}

# Create AMI with tagging
create_ami_with_tags() {
    local instance_id=$1
    local ami_name=$2
    local description=$3
    
    log "Creating AMI from instance: $instance_id"
    
    # Create AMI
    local ami_id
    ami_id=$(aws ec2 create-image \\
        --instance-id "$instance_id" \\
        --name "$ami_name" \\
        --description "$description" \\
        --no-reboot \\
        --query 'ImageId' \\
        --output text)
    
    log "AMI creation initiated: $ami_id"
    
    # Tag AMI
    aws ec2 create-tags \\
        --resources "$ami_id" \\
        --tags \\
            "Key=Name,Value=$ami_name" \\
            "Key=CreatedBy,Value=$(whoami)" \\
            "Key=CreatedAt,Value=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \\
            "Key=SourceInstance,Value=$instance_id"
    
    # Wait for AMI to be available
    log "Waiting for AMI to become available..."
    aws ec2 wait image-available --image-ids "$ami_id"
    
    log_success "AMI created and available: $ami_id"
    echo "$ami_id"
}

# Rotate CloudWatch log exports to S3
export_cloudwatch_logs_to_s3() {
    local log_group=$1
    local bucket_name=$2
    local prefix=$3
    local from_timestamp=$4  # Unix timestamp
    local to_timestamp=$5
    
    log "Exporting CloudWatch logs to S3"
    log "  Log group: $log_group"
    log "  S3 bucket: s3://$bucket_name/$prefix"
    
    local task_id
    task_id=$(aws logs create-export-task \\
        --log-group-name "$log_group" \\
        --from "$from_timestamp" \\
        --to "$to_timestamp" \\
        --destination "$bucket_name" \\
        --destination-prefix "$prefix" \\
        --query 'taskId' \\
        --output text)
    
    log "Export task created: $task_id"
    
    # Wait for export to complete
    while true; do
        local status
        status=$(aws logs describe-export-tasks \\
            --task-id "$task_id" \\
            --query 'exportTasks[0].status.code' \\
            --output text)
        
        case "$status" in
            COMPLETED)
                log_success "Log export completed"
                return 0
                ;;
            FAILED|CANCELLED)
                log_error "Log export failed with status: $status"
                return 1
                ;;
            *)
                log "Export status: $status"
                sleep 10
                ;;
        esac
    done
}
\`\`\`

## Production Deployment Script Example

Here's a complete, production-ready deployment script:

\`\`\`bash
#!/bin/bash
#
# Production Deployment Script
# Deploys application to EC2 instances behind ALB
#

set -euo pipefail

# Script configuration
readonly SCRIPT_NAME=$(basename "$0")
readonly SCRIPT_DIR="$(cd "$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_DIR="/var/log/deployments"
readonly LOG_FILE="$LOG_DIR/deploy-$(date +%Y%m%d-%H%M%S).log"

# Application configuration
readonly APP_NAME="myapp"
readonly APP_USER="appuser"
readonly APP_DIR="/opt/\${APP_NAME}"
readonly DEPLOY_BUCKET="s3://my-company-deployments"

# Deployment configuration
readonly MAX_PARALLEL_DEPLOYS=5
readonly HEALTH_CHECK_RETRIES=10
readonly HEALTH_CHECK_INTERVAL=6

# Logging setup
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "$LOG_FILE" >&2
}

log_success() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [SUCCESS] $*" | tee -a "$LOG_FILE"
}

# Cleanup on exit
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code: $exit_code"
        send_slack_notification "❌ Deployment failed" "error"
    fi
}
trap cleanup EXIT

# Validate prerequisites
validate_prerequisites() {
    log "Validating prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found"
        return 1
    fi
    
    # Check jq
    if ! command -v jq &> /dev/null; then
        log_error "jq not found"
        return 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        return 1
    fi
    
    log_success "Prerequisites validated"
}

# Get instances behind target group
get_target_group_instances() {
    local target_group_arn=$1
    
    aws elbv2 describe-target-health \\
        --target-group-arn "$target_group_arn" \\
        --query 'TargetHealthDescriptions[?TargetHealth.State==\`healthy\`].Target.Id' \\
        --output json | jq -r '.[]'
}

# Deregister instance from target group
deregister_from_target_group() {
    local target_group_arn=$1
    local instance_id=$2
    
    log "Deregistering $instance_id from target group"
    
    aws elbv2 deregister-targets \\
        --target-group-arn "$target_group_arn" \\
        --targets Id="$instance_id"
    
    # Wait for draining
    log "Waiting for connection draining..."
    while true; do
        local state
        state=$(aws elbv2 describe-target-health \\
            --target-group-arn "$target_group_arn" \\
            --targets Id="$instance_id" \\
            --query 'TargetHealthDescriptions[0].TargetHealth.State' \\
            --output text 2>/dev/null) || break
        
        if [[ "$state" == "unused" ]] || [[ -z "$state" ]]; then
            break
        fi
        
        log "  Draining state: $state"
        sleep 5
    done
    
    log_success "Instance deregistered and drained"
}

# Register instance to target group
register_to_target_group() {
    local target_group_arn=$1
    local instance_id=$2
    
    log "Registering $instance_id to target group"
    
    aws elbv2 register-targets \\
        --target-group-arn "$target_group_arn" \\
        --targets Id="$instance_id"
    
    # Wait for healthy state
    log "Waiting for health checks..."
    local attempt=0
    while [[ $attempt -lt $HEALTH_CHECK_RETRIES ]]; do
        local state
        state=$(aws elbv2 describe-target-health \\
            --target-group-arn "$target_group_arn" \\
            --targets Id="$instance_id" \\
            --query 'TargetHealthDescriptions[0].TargetHealth.State' \\
            --output text)
        
        if [[ "$state" == "healthy" ]]; then
            log_success "Instance is healthy"
            return 0
        fi
        
        log "  Health check state: $state (attempt $((attempt+1))/$HEALTH_CHECK_RETRIES)"
        sleep "$HEALTH_CHECK_INTERVAL"
        attempt=$((attempt+1))
    done
    
    log_error "Instance failed health checks"
    return 1
}

# Deploy to single instance
deploy_to_instance() {
    local instance_id=$1
    local version=$2
    local target_group_arn=$3
    
    log "===== Deploying to $instance_id ====="
    
    # Deregister from load balancer
    deregister_from_target_group "$target_group_arn" "$instance_id" || return 1
    
    # Download new version
    log "Downloading version $version from S3"
    local artifact_path="\${DEPLOY_BUCKET}/\${APP_NAME}/\${version}/\${APP_NAME}.tar.gz"
    local temp_file="/tmp/\${APP_NAME}-\${version}.tar.gz"
    
    if ! aws s3 cp "$artifact_path" "$temp_file"; then
        log_error "Failed to download artifact"
        register_to_target_group "$target_group_arn" "$instance_id"  # Re-register old version
        return 1
    fi
    
    # Backup current version
    log "Backing up current version"
    if [[ -d "$APP_DIR" ]]; then
        sudo tar czf "/var/backups/\${APP_NAME}-backup-$(date +%Y%m%d-%H%M%S).tar.gz" \\
            -C "$(dirname "$APP_DIR")" "$(basename "$APP_DIR")"
    fi
    
    # Stop application
    log "Stopping application"
    sudo systemctl stop "$APP_NAME"
    
    # Extract new version
    log "Extracting new version"
    sudo mkdir -p "$APP_DIR"
    sudo tar xzf "$temp_file" -C "$APP_DIR"
    sudo chown -R "$APP_USER:$APP_USER" "$APP_DIR"
    
    # Run migrations (if applicable)
    if [[ -f "\${APP_DIR}/migrate.sh" ]]; then
        log "Running database migrations"
        sudo -u "$APP_USER" "\${APP_DIR}/migrate.sh"
    fi
    
    # Start application
    log "Starting application"
    sudo systemctl start "$APP_NAME"
    
    # Wait for application to be ready
    sleep 5
    
    # Verify application started
    if ! sudo systemctl is-active --quiet "$APP_NAME"; then
        log_error "Application failed to start"
        sudo systemctl status "$APP_NAME"
        
        # Rollback attempt
        log "Attempting rollback..."
        sudo systemctl stop "$APP_NAME"
        # Restore from backup would go here
        
        return 1
    fi
    
    # Re-register to load balancer
    register_to_target_group "$target_group_arn" "$instance_id" || return 1
    
    # Cleanup
    rm -f "$temp_file"
    
    log_success "Deployment to $instance_id completed successfully"
    return 0
}

# Send Slack notification
send_slack_notification() {
    local message=$1
    local level=\${2:- info}
    local slack_webhook = \${ SLACK_WEBHOOK_URL: -}

if [[-z "$slack_webhook"]]; then
        log "Slack webhook not configured, skipping notification"
return 0
fi
    
    local color
    case "$level" in
    error) color = "danger";;
        success) color = "good";;
        *) color = "#439FE0";;
esac
    
    local payload
payload = $(jq - n \\
    --arg text "$message" \\
    --arg color "$color" \\
    '{
            attachments: [{
    color: $color,
    text: $text,
    footer: "Deployment Script",
    ts: now
}]
        }')
    
    curl - X POST - H 'Content-type: application/json' \\
    --data "$payload" \\
    "$slack_webhook" 2 > /dev/null || log "Failed to send Slack notification"
}

# Main deployment function
    main() {
    local target_group_arn = $1
    local version = $2
    
    log "========================================="
    log "Starting deployment"
    log "Target Group: $target_group_arn"
    log "Version: $version"
    log "========================================="

    validate_prerequisites || exit 1
    
    # Get instances
    local instances
    instances = $(get_target_group_instances "$target_group_arn")
    local instance_count
    instance_count = $(echo "$instances" | wc - l)
    
    log "Found $instance_count instances to deploy"
    
    # Deploy to each instance(one at a time for safety)
    local failed_instances = ()
    for instance_id in $instances; do
        if deploy_to_instance "$instance_id" "$version" "$target_group_arn"; then
            log_success "Deployment to $instance_id successful"
        else
            log_error "Deployment to $instance_id failed"
    failed_instances += ("$instance_id")
    fi
    done
    
    # Final status
    if [[\${ #failed_instances[@] } - eq 0]]; then
        log_success "========================================="
        log_success "Deployment completed successfully!"
        log_success "========================================="
        send_slack_notification "✅ Deployment of $version completed successfully" "success"
        exit 0
    else
        log_error "========================================="
        log_error "Deployment completed with failures"
        log_error "Failed instances: \${failed_instances[*]}"
        log_error "========================================="
        send_slack_notification "⚠️  Deployment of $version partially failed" "error"
        exit 1
    fi
}

# Argument parsing
if [[$# - ne 2]]; then
    echo "Usage: $SCRIPT_NAME <target-group-arn> <version>"
    echo "Example: $SCRIPT_NAME arn:aws:elasticloadbalancing:us-east-1:123456789:targetgroup/my-app/abc123 v1.2.3"
    exit 2
fi

main "$@"
\`\`\`

## Best Practices Summary

✅ **Always use strict mode**: \`set -euo pipefail\`  
✅ **Implement comprehensive logging**: Info, warning, error levels  
✅ **Handle errors gracefully**: Exit codes and cleanup  
✅ **Make scripts idempotent**: Can run multiple times safely  
✅ **Use functions**: Modular and reusable code  
✅ **Validate prerequisites**: Check dependencies upfront  
✅ **Document thoroughly**: Usage examples and comments  
✅ **Test before production**: Dry-run mode when possible  
✅ **Version control**: Keep scripts in Git  
✅ **Security**: Never hardcode credentials, use AWS IAM roles

## Next Steps

In the next section, we'll cover **System Monitoring & Performance**, learning how to use top, htop, iostat, and other tools to diagnose performance issues on production systems.`,
};
