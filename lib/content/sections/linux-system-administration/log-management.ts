/**
 * Log Management Section
 * Module: Linux System Administration & DevOps Foundations
 */

export const logManagementSection = {
  id: 'log-management',
  title: 'Log Management',
  content: `# Log Management

## Introduction

Effective log management is crucial for debugging, security auditing, compliance, and operational insights. In production environments, logs can grow rapidly and consume significant storage. This section covers log rotation, centralized logging, log analysis, and AWS CloudWatch Logs integration.

## Log Rotation with Logrotate

\`\`\`bash
# Logrotate configuration
sudo cat /etc/logrotate.conf

# Application-specific configuration
sudo cat << 'EOF' > /etc/logrotate.d/myapp
/var/log/myapp/*.log {
    daily                    # Rotate daily
    rotate 14                # Keep 14 days
    compress                 # Gzip old logs
    delaycompress            # Compress on next rotation
    missingok                # Don't error if log missing
    notifempty               # Don't rotate if empty
    create 0640 appuser appgroup  # Create new log with permissions
    sharedscripts            # Run scripts once for all logs
    postrotate
        systemctl reload myapp > /dev/null 2>&1 || true
    endscript
}

/var/log/myapp/access.log {
    daily
    rotate 30                # Keep 30 days for access logs
    compress
    delaycompress
    missingok
    notifempty
    create 0640 appuser appgroup
    dateext                  # Add date to filename
    dateformat -%Y%m%d
    maxsize 1G               # Rotate if exceeds 1GB
    postrotate
        kill -USR1 $(cat /var/run/myapp.pid) > /dev/null 2>&1 || true
    endscript
}

/var/log/myapp/critical.log {
    weekly                   # Rotate weekly
    rotate 52                # Keep 1 year
    compress
    delaycompress
    missingok
    notifempty
    create 0640 appuser appgroup
    sharedscripts
    postrotate
        # Email alert for critical logs
        /usr/local/bin/alert-critical-logs.sh
    endscript
}
EOF

# Test configuration
sudo logrotate -d /etc/logrotate.d/myapp

# Force rotation
sudo logrotate -f /etc/logrotate.d/myapp

# Check logrotate status
sudo cat /var/lib/logrotate/status
\`\`\`

## Systemd Journal Management

\`\`\`bash
# View journal disk usage
journalctl --disk-usage

# Vacuum by time
sudo journalctl --vacuum-time=30d   # Keep 30 days
sudo journalctl --vacuum-time=1week

# Vacuum by size
sudo journalctl --vacuum-size=1G    # Max 1GB
sudo journalctl --vacuum-size=500M

# Vacuum by files
sudo journalctl --vacuum-files=10   # Keep 10 journal files

# Configure persistent storage
sudo mkdir -p /var/log/journal
sudo systemd-tmpfiles --create --prefix /var/log/journal

# Journal configuration
sudo cat << 'EOF' > /etc/systemd/journald.conf
[Journal]
# Storage location
Storage=persistent         # persistent, volatile, auto, none

# Size limits
SystemMaxUse=1G           # Max disk space
SystemKeepFree=2G         # Keep this much free
SystemMaxFileSize=100M    # Max single file size
SystemMaxFiles=100        # Max number of files

# Time limits
MaxRetentionSec=30day     # Keep 30 days
MaxFileSec=1week          # New file weekly

# Forward to syslog
ForwardToSyslog=no
ForwardToKMsg=no
ForwardToConsole=no
ForwardToWall=yes

# Rate limiting
RateLimitInterval=30s
RateLimitBurst=1000
EOF

# Apply changes
sudo systemctl restart systemd-journald

# Verify logs
sudo journalctl --verify
\`\`\`

## Centralized Logging Architecture

\`\`\`
Application → Local Log Files/Journald → Log Agent (CloudWatch Agent/Fluentd) → CloudWatch Logs → Analysis/Alerts
\`\`\`

## AWS CloudWatch Logs Agent

\`\`\`bash
# Install CloudWatch Agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
sudo rpm -U ./amazon-cloudwatch-agent.rpm

# Configuration file
sudo cat << 'EOF' > /opt/aws/amazon-cloudwatch-agent/etc/config.json
{
  "agent": {
    "metrics_collection_interval": 60,
    "run_as_user": "cwagent"
  },
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/myapp/application.log",
            "log_group_name": "/aws/ec2/myapp",
            "log_stream_name": "{instance_id}/application",
            "timezone": "UTC",
            "timestamp_format": "%Y-%m-%d %H:%M:%S",
            "multi_line_start_pattern": "{timestamp_format}"
          },
          {
            "file_path": "/var/log/myapp/error.log",
            "log_group_name": "/aws/ec2/myapp",
            "log_stream_name": "{instance_id}/error",
            "timezone": "UTC"
          },
          {
            "file_path": "/var/log/messages",
            "log_group_name": "/aws/ec2/system",
            "log_stream_name": "{instance_id}/messages"
          }
        ]
      },
      "journald": {
        "log_group_name": "/aws/ec2/journal",
        "log_stream_name": "{instance_id}",
        "unit_whitelist": ["myapp.service", "nginx.service"]
      }
    },
    "log_stream_name": "{instance_id}"
  },
  "metrics": {
    "namespace": "MyApp/EC2",
    "metrics_collected": {
      "disk": {
        "measurement": [
          {"name": "used_percent", "rename": "DiskUsedPercent"}
        ],
        "metrics_collection_interval": 60,
        "resources": ["*"]
      },
      "mem": {
        "measurement": [
          {"name": "mem_used_percent", "rename": "MemUsedPercent"}
        ],
        "metrics_collection_interval": 60
      }
    }
  }
}
EOF

# Start agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -s \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/config.json

# Check status
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a query \
  -m ec2 \
  -s

# Enable on boot
sudo systemctl enable amazon-cloudwatch-agent
\`\`\`

## Terraform: CloudWatch Log Groups

\`\`\`terraform
resource "aws_cloudwatch_log_group" "application" {
  name              = "/aws/ec2/myapp"
  retention_in_days = 30

  tags = {
    Environment = "production"
    Application = "myapp"
  }
}

resource "aws_cloudwatch_log_group" "error" {
  name              = "/aws/ec2/myapp/error"
  retention_in_days = 90  # Keep errors longer
}

# Metric filter for errors
resource "aws_cloudwatch_log_metric_filter" "error_count" {
  name           = "ErrorCount"
  log_group_name = aws_cloudwatch_log_group.application.name
  pattern        = "[timestamp, request_id, level=ERROR*, ...]"

  metric_transformation {
    name      = "ErrorCount"
    namespace = "MyApp"
    value     = "1"
    default_value = 0
  }
}

# Alarm on error rate
resource "aws_cloudwatch_metric_alarm" "high_error_rate" {
  alarm_name          = "myapp-high-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "ErrorCount"
  namespace           = "MyApp"
  period              = "300"
  statistic           = "Sum"
  threshold           = "10"
  alarm_description   = "Triggers when error count exceeds 10 in 5 minutes"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# IAM role for EC2 to write logs
resource "aws_iam_role" "ec2_cloudwatch" {
  name = "ec2-cloudwatch-logs-role"

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

resource "aws_iam_role_policy" "cloudwatch_logs_policy" {
  name = "cloudwatch-logs-policy"
  role = aws_iam_role.ec2_cloudwatch.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "ec2-cloudwatch-profile"
  role = aws_iam_role.ec2_cloudwatch.name
}
\`\`\`

## Structured Logging

\`\`\`python
# Python structured logging (JSON)
import json
import logging
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms
            
        # Add exception info
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

# Configure logging
logger = logging.getLogger('myapp')
handler = logging.FileHandler('/var/log/myapp/application.log')
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info('User logged in', extra={'user_id': '123', 'request_id': 'abc'})
logger.error('Database connection failed', extra={'database': 'postgres', 'retries': 3})
\`\`\`

## Log Analysis

\`\`\`bash
# CloudWatch Insights queries
# Error rate over time
fields @timestamp, level, message
| filter level = "ERROR"
| stats count() by bin(5m)

# Slowest requests
fields @timestamp, duration_ms, endpoint
| filter duration_ms > 1000
| sort duration_ms desc
| limit 20

# User activity
fields @timestamp, user_id, action
| filter user_id = "user123"
| sort @timestamp desc

# Top error messages
fields message
| filter level = "ERROR"
| stats count() by message
| sort count desc
| limit 10
\`\`\`

## Best Practices

✅ **Structured logging** with JSON for easy parsing  
✅ **Include context** (request_id, user_id, trace_id)  
✅ **Use appropriate log levels** (DEBUG, INFO, WARN, ERROR, FATAL)  
✅ **Rotate logs** to prevent disk exhaustion  
✅ **Centralize logs** for distributed systems  
✅ **Set retention policies** based on compliance requirements  
✅ **Create alerts** for error patterns  
✅ **Sanitize sensitive data** (PII, credentials)  
✅ **Monitor log volume** and costs  
✅ **Use log sampling** for high-traffic applications

## Security Considerations

- **Never log credentials, tokens, or PII**
- **Restrict log file permissions** (0640 or 0600)
- **Encrypt logs at rest** (CloudWatch Logs encryption)
- **Encrypt logs in transit** (TLS for log shipping)
- **Implement log retention policies** for compliance
- **Audit log access** (who viewed what logs)
- **Protect against log injection** attacks

## Next Steps

In the next section, we'll cover **SSH & Remote Administration**, including key-based authentication, SSH hardening, bastion hosts, and session management.`,
};
