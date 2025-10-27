import { Content } from '@/lib/types';

const productionBacktestingEngineProjectDiscussion: Content = {
  title: 'Production Backtesting Engine - Discussion Questions',
  description:
    'Capstone discussion questions on building, deploying, and operating production-grade backtesting infrastructure',
  sections: [
    {
      title: 'Discussion Questions',
      content: `
# Discussion Questions: Production Backtesting Engine

## Question 1: Disaster Recovery and System Resilience

**Scenario**: It's 3 AM. Your production backtesting system—which runs critical daily optimizations for $500M in live capital—has gone down. The database corrupted during a failed deployment. No backups exist from the last 6 hours (20 completed backtests lost).

Your on-call engineer needs guidance on:
1. Immediate recovery steps
2. What data can/cannot be recovered
3. How to prevent this in the future

**Design a comprehensive disaster recovery strategy for a production backtesting system, including backup procedures, incident response playbook, and prevention measures.**

### Comprehensive Answer

Production systems require robust disaster recovery planning.

**Immediate Recovery Playbook:**

\`\`\`python
class DisasterRecoveryManager:
    """
    Disaster recovery procedures for production backtesting system
    """
    
    def __init__(self):
        self.incident_log = []
        self.recovery_status = {}
    
    async def execute_recovery_procedure(self, incident_type: str):
        """
        Execute disaster recovery based on incident type
        """
        
        procedures = {
            'database_corruption': self.recover_from_db_corruption,
            'cache_failure': self.recover_from_cache_failure,
            'application_crash': self.recover_from_app_crash,
            'network_partition': self.recover_from_network_partition
        }
        
        procedure = procedures.get(incident_type)
        if procedure:
            await procedure()
        else:
            await self.escalate_to_senior_engineer(incident_type)
    
    async def recover_from_db_corruption(self):
        """
        Database corruption recovery
        
        Priority: Restore service, minimize data loss
        """
        
        print("\\n🚨 DATABASE CORRUPTION RECOVERY")
        print("="*80)
        
        steps = [
            ("1. STOP ALL WRITES", self.stop_all_writes),
            ("2. ASSESS DAMAGE", self.assess_database_damage),
            ("3. RESTORE FROM BACKUP", self.restore_from_backup),
            ("4. REPLAY WAL LOGS", self.replay_write_ahead_logs),
            ("5. VERIFY INTEGRITY", self.verify_database_integrity),
            ("6. RERUN LOST BACKTESTS", self.rerun_lost_backtests),
            ("7. RESUME SERVICE", self.resume_service)
        ]
        
        for step_name, step_func in steps:
            print(f"\\n{step_name}...")
            try:
                await step_func()
                print(f"✓ {step_name} complete")
            except Exception as e:
                print(f"✗ {step_name} failed: {e}")
                await self.escalate_to_senior_engineer(f"Recovery failed at {step_name}")
                return
        
        print("\\n✓ RECOVERY COMPLETE")
        print("="*80)
    
    async def stop_all_writes(self):
        """Immediately stop all write operations"""
        # Set read-only mode
        # Reject new backtest submissions
        # Wait for in-flight transactions
        pass
    
    async def assess_database_damage(self):
        """Determine extent of corruption"""
        # Check which tables affected
        # Identify last valid transaction
        # Calculate data loss window
        pass
    
    async def restore_from_backup(self):
        """Restore database from most recent backup"""
        # Options:
        # 1. Point-in-time recovery (AWS RDS)
        # 2. Snapshot restore
        # 3. Streaming replication failover
        pass
    
    async def replay_write_ahead_logs(self):
        """Replay WAL to recover recent transactions"""
        # PostgreSQL WAL replay
        # Recovers transactions after backup
        pass
    
    async def verify_database_integrity(self):
        """Verify database is healthy"""
        # Run ANALYZE
        # Check constraints
        # Validate indexes
        pass
    
    async def rerun_lost_backtests(self):
        """Rerun backtests lost during outage"""
        # Query job queue for pending/running jobs during outage
        # Resubmit with original parameters
        # High priority queue
        pass
    
    async def resume_service(self):
        """Bring system back online"""
        # Remove read-only mode
        # Enable new submissions
        # Monitor closely for 24 hours
        pass


class BackupManager:
    """
    Automated backup management
    """
    
    def __init__(self, db_url: str, s3_bucket: str):
        self.db_url = db_url
        self.s3_bucket = s3_bucket
    
    async def execute_backup_schedule(self):
        """
        Automated backup schedule
        
        - Continuous WAL archiving (every 5 min)
        - Snapshot backups (every 6 hours)
        - Full backups (daily at 2 AM)
        - Weekly backups (retained 4 weeks)
        - Monthly backups (retained 1 year)
        """
        
        schedule = {
            'continuous_wal': ('*/5 * * * *', self.archive_wal),
            'snapshot': ('0 */6 * * *', self.create_snapshot),
            'daily_full': ('0 2 * * *', self.full_backup),
            'weekly': ('0 2 * * 0', self.weekly_backup),
            'monthly': ('0 2 1 * *', self.monthly_backup)
        }
        
        # Scheduled via Kubernetes CronJob or AWS EventBridge
        pass
    
    async def archive_wal(self):
        """Archive Write-Ahead Logs to S3"""
        # Continuous archiving enables point-in-time recovery
        # pg_wal files copied to S3 every 5 minutes
        pass
    
    async def create_snapshot(self):
        """Create database snapshot"""
        # AWS RDS snapshot (instant, no downtime)
        # Or pg_dump for smaller databases
        pass
    
    async def verify_backups(self):
        """
        Periodically verify backups are restorable
        
        Critical: Backups are useless if they can't be restored
        """
        # Monthly: restore backup to staging environment
        # Run integrity checks
        # Document recovery time
        pass


class IncidentResponse:
    """
    Incident response procedures
    """
    
    @staticmethod
    def create_incident_playbook() -> Dict:
        """Incident response playbook"""
        
        return {
            'severity_levels': {
                'P0_CRITICAL': {
                    'description': 'Production down, live trading affected',
                    'response_time': '< 5 minutes',
                    'escalation': 'Page CTO, send to all-hands channel',
                    'examples': ['Database corruption', 'Live trading system down']
                },
                'P1_HIGH': {
                    'description': 'Core functionality impaired',
                    'response_time': '< 30 minutes',
                    'escalation': 'Page on-call engineer',
                    'examples': ['Backtest engine down', 'Optimization failures']
                },
                'P2_MEDIUM': {
                    'description': 'Degraded performance',
                    'response_time': '< 2 hours',
                    'escalation': 'Alert engineering team',
                    'examples': ['Slow API responses', 'Cache misses']
                }
            },
            
            'response_steps': [
                '1. Acknowledge incident in PagerDuty/Slack',
                '2. Assess severity and scope',
                '3. Execute appropriate recovery procedure',
                '4. Communicate status updates every 30 min',
                '5. Document actions taken',
                '6. Restore service',
                '7. Schedule post-mortem within 48 hours'
            ],
            
            'post_mortem_template': {
                'incident_summary': 'What happened?',
                'root_cause': 'Why did it happen?',
                'impact': 'Who/what was affected?',
                'timeline': 'Sequence of events',
                'resolution': 'How was it fixed?',
                'prevention': 'How to prevent recurrence?',
                'action_items': 'Follow-up tasks with owners and deadlines'
            }
        }
\`\`\`

**Prevention Measures:**

1. **Database**:
   - Continuous WAL archiving (5-min RPO)
   - Automated snapshots every 6 hours
   - Streaming replication to standby (hot failover)
   - Monthly restore testing

2. **Application**:
   - Blue-green deployments (zero downtime)
   - Canary releases (gradual rollout)
   - Automated rollback on errors
   - Feature flags for risky changes

3. **Monitoring**:
   - Health checks every 30 seconds
   - Alert on database lag, slow queries, errors
   - PagerDuty integration
   - Runbook automation

4. **Testing**:
   - Chaos engineering (random failures in staging)
   - Disaster recovery drills quarterly
   - Load testing before major releases

---

## Question 2: Scaling to Handle 1000+ Concurrent Backtests

**Your firm is scaling research from 20 quants to 200 quants. Backtest volume will increase 50x (from 20/day to 1000+/day). Your current single-server system cannot handle this.**

**Design a scalable architecture that:**
1. Handles 1000+ concurrent backtests
2. Maintains <5 second response time for status queries
3. Costs <$10,000/month in AWS
4. Scales automatically based on load

### Comprehensive Answer

\`\`\`python
"""
Scalable Architecture Design

Components:
1. API Gateway (Load Balancer)
2. Application Tier (Auto-scaling ECS/Fargate)
3. Job Queue (SQS/RabbitMQ)
4. Worker Pool (Spot Instances)
5. Data Layer (RDS + Redis + S3)
6. Monitoring (CloudWatch/Datadog)
"""

# Auto-scaling configuration
auto_scaling_config = {
    'api_tier': {
        'min_instances': 2,
        'max_instances': 20,
        'target_cpu': 70,  # Scale up at 70% CPU
        'scale_up_cooldown': 60,  # seconds
        'scale_down_cooldown': 300
    },
    
    'worker_tier': {
        'min_instances': 5,
        'max_instances': 100,
        'target_queue_depth': 10,  # Scale up if queue > 10 per worker
        'use_spot_instances': True,  # 70% cost savings
        'spot_price_limit': 0.10  # Max $0.10/hour
    },
    
    'database': {
        'instance_type': 'db.r5.xlarge',  # 4 vCPU, 32GB RAM
        'read_replicas': 3,  # Distribute read load
        'auto_scaling_storage': True
    },
    
    'cache': {
        'instance_type': 'cache.r5.large',  # 2 vCPU, 13GB RAM
        'num_nodes': 3  # Cluster mode
    }
}

# Cost breakdown
monthly_costs = {
    'API Gateway': 100,  # $3.50/million requests
    'ECS Fargate (avg 10 tasks)': 720,  # $0.04048/hour
    'Workers (avg 50 spot instances)': 1800,  # $0.05/hour spot
    'RDS (r5.xlarge + replicas)': 2400,
    'ElastiCache (r5.large cluster)': 450,
    'S3 Storage (10TB)': 230,
    'Data Transfer': 500,
    'CloudWatch': 200,
    'Total': 6400  # Well under $10k
}

class ScalableBacktestOrchestrator:
    """
    Orchestrates distributed backtesting at scale
    """
    
    def __init__(self):
        self.job_queue = SQS(queue_url='https://sqs.../backtest-queue')
        self.status_cache = Redis()
        self.worker_pool = ECSWorkerPool()
    
    async def submit_backtest(self, request: BacktestRequest) -> str:
        """
        Submit backtest (O(1) operation)
        """
        job_id = generate_job_id()
        
        # Add to SQS queue
        await self.job_queue.send_message({
            'job_id': job_id,
            'request': request.dict()
        })
        
        # Cache initial status
        await self.status_cache.set(
            f"status:{job_id}",
            json.dumps({'status': 'queued', 'position': await self.job_queue.approximate_size()}),
            ex=86400  # 24 hour expiry
        )
        
        # Trigger auto-scaling if needed
        await self.check_scaling_trigger()
        
        return job_id
    
    async def get_status(self, job_id: str) -> Dict:
        """
        Get backtest status (<5ms from Redis cache)
        """
        cached = await self.status_cache.get(f"status:{job_id}")
        
        if cached:
            return json.loads(cached)
        
        # Fallback to database (slower)
        return await self.get_status_from_db(job_id)
    
    async def check_scaling_trigger(self):
        """
        Trigger auto-scaling based on queue depth
        """
        queue_size = await self.job_queue.approximate_size()
        active_workers = await self.worker_pool.get_active_count()
        
        target_workers = min(
            max(
                queue_size / 10,  # 1 worker per 10 jobs
                auto_scaling_config['worker_tier']['min_instances']
            ),
            auto_scaling_config['worker_tier']['max_instances']
        )
        
        if target_workers > active_workers:
            await self.worker_pool.scale_up(int(target_workers - active_workers))
        elif target_workers < active_workers - 5:  # Hysteresis
            await self.worker_pool.scale_down(int(active_workers - target_workers))


class ECSWorkerPool:
    """
    Manages ECS Fargate worker pool
    """
    
    async def scale_up(self, count: int):
        """Launch additional workers"""
        for _ in range(count):
            await self.launch_worker()
    
    async def scale_down(self, count: int):
        """Terminate idle workers gracefully"""
        # Only terminate workers not currently processing
        pass
    
    async def launch_worker(self):
        """Launch ECS Fargate task"""
        ecs_client.run_task(
            cluster='backtest-cluster',
            taskDefinition='backtest-worker',
            launchType='FARGATE',
            networkConfiguration={...},
            capacityProviderStrategy=[
                {'capacityProvider': 'FARGATE_SPOT', 'weight': 1}
            ]
        )
\`\`\`

**Key Architectural Decisions:**

1. **Decouple API from Workers**: API handles submissions/queries, workers process jobs
2. **Use Job Queue**: SQS provides durability, auto-scaling trigger, retry logic
3. **Spot Instances**: 70% cost savings for workers (acceptable for batch processing)
4. **Read Replicas**: Distribute query load across multiple database instances
5. **Redis Caching**: Sub-5ms status queries from cache
6. **Auto-scaling**: Responds to load automatically, no manual intervention

This architecture handles 10,000+ concurrent backtests and scales down to zero cost during idle periods.

---

## Question 3: Monitoring and Observability

**How do you know your production backtesting system is healthy? Design a comprehensive monitoring and alerting strategy.**

### Comprehensive Answer

\`\`\`python
# Key metrics to monitor

monitoring_metrics = {
    'System Health': {
        'api_latency_p99': {'threshold': '500ms', 'severity': 'HIGH'},
        'error_rate': {'threshold': '1%', 'severity': 'CRITICAL'},
        'worker_cpu_utilization': {'threshold': '80%', 'severity': 'MEDIUM'},
        'queue_depth': {'threshold': '1000', 'severity': 'HIGH'}
    },
    
    'Database': {
        'connection_count': {'threshold': '80%', 'severity': 'HIGH'},
        'replication_lag': {'threshold': '10s', 'severity': 'CRITICAL'},
        'slow_query_count': {'threshold': '10/min', 'severity': 'MEDIUM'},
        'storage_utilization': {'threshold': '85%', 'severity': 'HIGH'}
    },
    
    'Business Metrics': {
        'backtest_success_rate': {'threshold': '95%', 'severity': 'HIGH'},
        'avg_backtest_duration': {'threshold': '300s', 'severity': 'MEDIUM'},
        'daily_backtest_volume': {'threshold': '1000', 'severity': 'LOW'}
    },
    
    'Cost': {
        'daily_aws_spend': {'threshold': '$500', 'severity': 'MEDIUM'},
        'cost_per_backtest': {'threshold': '$0.50', 'severity': 'LOW'}
    }
}

# Implement monitoring
class MonitoringService:
    """Comprehensive monitoring and alerting"""
    
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.pagerduty = PagerDuty(api_key='...')
    
    async def emit_metric(self, metric_name: str, value: float):
        """Emit metric to CloudWatch"""
        self.cloudwatch.put_metric_data(
            Namespace='BacktestEngine',
            MetricData=[{
                'MetricName': metric_name,
                'Value': value,
                'Timestamp': datetime.now(),
                'Unit': 'Count'
            }]
        )
    
    async def check_health(self):
        """Comprehensive health check"""
        checks = {
            'database': await self.check_database(),
            'cache': await self.check_cache(),
            'queue': await self.check_queue(),
            'workers': await self.check_workers()
        }
        
        all_healthy = all(checks.values())
        
        if not all_healthy:
            await self.alert_on_call(checks)
        
        return {'healthy': all_healthy, 'details': checks}
\`\`\`

**Recommended Stack:**
- **Metrics**: CloudWatch + Datadog
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Tracing**: AWS X-Ray
- **Alerting**: PagerDuty + Slack
- **Dashboards**: Grafana

**Congratulations on completing Module 10!** You now have the skills to build and operate production-grade backtesting infrastructure.
`,
    },
  ],
};

export default productionBacktestingEngineProjectDiscussion;
