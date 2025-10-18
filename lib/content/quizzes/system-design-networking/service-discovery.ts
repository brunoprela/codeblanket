/**
 * Quiz questions for Service Discovery section
 */

export const servicediscoveryQuiz = [
  {
    id: 'service-discovery-migration',
    question:
      "Your company is migrating from a monolithic application to microservices. You have 20 services that need to communicate with each other, and you're deploying on AWS ECS. Design a service discovery strategy using either Consul or AWS Cloud Map. Explain how services register, how clients discover services, how health checks work, and how you'd handle graceful shutdowns. Include specific implementation details and monitoring strategies.",
    sampleAnswer: `**Service Discovery Strategy for AWS ECS Migration**
    
    **Choice: AWS Cloud Map** (native integration with ECS)
    
    **Why Cloud Map over Consul**:
    - Native AWS integration (no additional infrastructure)
    - Automatic ECS task registration/deregistration
    - Integrated with Route 53 for DNS-based discovery
    - Health checks integrated with ECS
    - Lower operational overhead
    
    **Architecture**:
    \`\`\`
                      AWS Cloud Map
                            ↓
             +-----------------------------+
             |                             |
        ECS Service A               ECS Service B
        (3 tasks)                   (5 tasks)
             |                             |
       ALB/Service Connect        ALB/Service Connect
    \`\`\`
    
    **1. Service Registration**
    
    **CloudFormation Template**:
    \`\`\`yaml
    Resources:
      # Cloud Map namespace
      PrivateNamespace:
        Type: AWS::ServiceDiscovery::PrivateDnsNamespace
        Properties:
          Name: internal.mycompany.local
          Vpc: !Ref VPC
      
      # Service discovery for User Service
      UserServiceDiscovery:
        Type: AWS::ServiceDiscovery::Service
        Properties:
          Name: user-service
          DnsConfig:
            DnsRecords:
              - Type: A
                TTL: 10
            NamespaceId: !Ref PrivateNamespace
          HealthCheckCustomConfig:
            FailureThreshold: 1
      
      # ECS Service with Service Discovery
      UserService:
        Type: AWS::ECS::Service
        Properties:
          ServiceName: user-service
          Cluster: !Ref ECSCluster
          TaskDefinition: !Ref UserServiceTaskDef
          DesiredCount: 3
          LaunchType: FARGATE
          NetworkConfiguration:
            AwsvpcConfiguration:
              Subnets:
                - !Ref PrivateSubnet1
                - !Ref PrivateSubnet2
              SecurityGroups:
                - !Ref UserServiceSecurityGroup
          ServiceRegistries:
            - RegistryArn: !GetAtt UserServiceDiscovery.Arn
              ContainerName: user-service
              ContainerPort: 3000
          HealthCheckGracePeriodSeconds: 60
    \`\`\`
    
    **Automatic Registration**:
    - ECS automatically registers tasks when they start
    - Each task gets DNS entry: \`<task-id>.user-service.internal.mycompany.local\`
    - Service DNS entry: \`user-service.internal.mycompany.local\` (returns all healthy IPs)
    
    **2. Service Discovery in Application Code**
    
    **Option A: DNS-based (Simplest)**:
    \`\`\`javascript
    // services/client.js
    class ServiceClient {
      constructor(serviceName) {
        // Use Cloud Map DNS name
        this.baseUrl = \`http://\${serviceName}.internal.mycompany.local\`;
      }
      
      async call(path, options = {}) {
        const url = \`\${this.baseUrl}\${path}\`;
        
        // Add retry logic
        return this.retry(async () => {
          const response = await fetch(url, {
            ...options,
            timeout: 5000
          });
          
          if (!response.ok) {
            throw new Error(\`HTTP \${response.status}\`);
          }
          
          return response.json();
        });
      }
      
      async retry(fn, maxAttempts = 3) {
        let lastError;
        
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
          try {
            return await fn();
          } catch (error) {
            lastError = error;
            
            if (attempt < maxAttempts) {
              // Exponential backoff
              const delay = Math.min(1000 * Math.pow(2, attempt - 1), 5000);
              await new Promise(resolve => setTimeout(resolve, delay));
            }
          }
        }
        
        throw lastError;
      }
    }
    
    // Usage in Order Service
    const userClient = new ServiceClient('user-service');
    const inventoryClient = new ServiceClient('inventory-service');
    
    app.post('/orders', async (req, res) => {
      const { userId, items } = req.body;
      
      // Calls resolve via DNS to current healthy instances
      const user = await userClient.call(\`/users/\${userId}\`);
      const availability = await inventoryClient.call('/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items })
      });
      
      // Create order...
    });
    \`\`\`
    
    **Option B: AWS SDK (More Control)**:
    \`\`\`javascript
    const AWS = require('aws-sdk');
    const servicediscovery = new AWS.ServiceDiscovery();
    
    class CloudMapClient {
      constructor(serviceName, namespace) {
        this.serviceName = serviceName;
        this.namespace = namespace;
        this.instances = [];
        this.currentIndex = 0;
        
        // Refresh instances periodically
        this.refreshInstances();
        setInterval(() => this.refreshInstances(), 30000);
      }
      
      async refreshInstances() {
        try {
          const services = await servicediscovery.discoverInstances({
            NamespaceName: this.namespace,
            ServiceName: this.serviceName,
            HealthStatus: 'HEALTHY'
          }).promise();
          
          this.instances = services.Instances.map(inst => ({
            ip: inst.Attributes.AWS_INSTANCE_IPV4,
            port: inst.Attributes.AWS_INSTANCE_PORT
          }));
          
          console.log(\`Refreshed \${this.serviceName}: \${this.instances.length} instances\`);
        } catch (error) {
          console.error(\`Failed to refresh \${this.serviceName}:\`, error);
        }
      }
      
      getNextInstance() {
        if (this.instances.length === 0) {
          throw new Error(\`No healthy instances of \${this.serviceName}\`);
        }
        
        const instance = this.instances[this.currentIndex];
        this.currentIndex = (this.currentIndex + 1) % this.instances.length;
        
        return \`http://\${instance.ip}:\${instance.port}\`;
      }
    }
    
    const userClient = new CloudMapClient('user-service', 'internal.mycompany.local');
    \`\`\`
    
    **3. Health Checks**
    
    **Task Definition**:
    \`\`\`json
    {
      "family": "user-service",
      "containerDefinitions": [
        {
          "name": "user-service",
          "image": "user-service:1.0.0",
          "portMappings": [
            {
              "containerPort": 3000,
              "protocol": "tcp"
            }
          ],
          "healthCheck": {
            "command": [
              "CMD-SHELL",
              "curl -f http://localhost:3000/health || exit 1"
            ],
            "interval": 30,
            "timeout": 5,
            "retries": 3,
            "startPeriod": 60
          }
        }
      ]
    }
    \`\`\`
    
    **Health Check Endpoint**:
    \`\`\`javascript
    // Health check implementation
    app.get('/health', async (req, res) => {
      const checks = {
        database: false,
        redis: false,
        dependencies: false
      };
      
      try {
        // Check database
        await db.query('SELECT 1');
        checks.database = true;
        
        // Check Redis
        await redis.ping();
        checks.redis = true;
        
        // Check critical dependencies
        const userServiceHealth = await fetch('http://dependency-service.internal.mycompany.local/health', {
          timeout: 2000
        });
        checks.dependencies = userServiceHealth.ok;
        
        // All checks passed
        if (checks.database && checks.redis && checks.dependencies) {
          res.status(200).json({
            status: 'healthy',
            checks,
            uptime: process.uptime(),
            timestamp: Date.now()
          });
        } else {
          throw new Error('Some checks failed');
        }
      } catch (error) {
        res.status(503).json({
          status: 'unhealthy',
          checks,
          error: error.message
        });
      }
    });
    \`\`\`
    
    **4. Graceful Shutdown**
    
    \`\`\`javascript
    // Graceful shutdown handler
    let isShuttingDown = false;
    let activeConnections = 0;
    
    // Track active connections
    app.use((req, res, next) => {
      if (isShuttingDown) {
        res.status(503).send('Service is shutting down');
        return;
      }
      
      activeConnections++;
      res.on('finish', () => activeConnections--);
      next();
    });
    
    async function gracefulShutdown(signal) {
      console.log(\`Received \${signal}, starting graceful shutdown...\`);
      isShuttingDown = true;
      
      // 1. Mark health check as unhealthy (stops new traffic from Cloud Map)
      app.get('/health', (req, res) => {
        res.status(503).json({ status: 'shutting down' });
      });
      
      // 2. Wait a bit for ALB to detect unhealthy (deregistration delay)
      console.log('Waiting for deregistration...');
      await new Promise(resolve => setTimeout(resolve, 15000)); // 15 seconds
      
      // 3. Stop accepting new connections
      console.log('Stopping server...');
      server.close(() => {
        console.log('Server closed');
      });
      
      // 4. Wait for active connections to complete
      console.log(\`Draining \${activeConnections} active connections...\`);
      while (activeConnections > 0) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        console.log(\`Remaining connections: \${activeConnections}\`);
      }
      
      // 5. Close database connections
      console.log('Closing database...');
      await db.close();
      await redis.quit();
      
      console.log('Shutdown complete');
      process.exit(0);
    }
    
    process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
    process.on('SIGINT', () => gracefulShutdown('SIGINT'));
    \`\`\`
    
    **5. Monitoring & Observability**
    
    **CloudWatch Metrics**:
    \`\`\`javascript
    const { CloudWatch } = require('aws-sdk');
    const cloudwatch = new CloudWatch();
    
    // Middleware to track metrics
    app.use((req, res, next) => {
      const start = Date.now();
      
      res.on('finish', () => {
        const duration = Date.now() - start;
        
        // Send custom metric
        cloudwatch.putMetricData({
          Namespace: 'UserService',
          MetricData: [
            {
              MetricName: 'RequestDuration',
              Value: duration,
              Unit: 'Milliseconds',
              Dimensions: [
                { Name: 'Endpoint', Value: req.path },
                { Name: 'StatusCode', Value: String(res.statusCode) }
              ]
            },
            {
              MetricName: 'RequestCount',
              Value: 1,
              Unit: 'Count',
              Dimensions: [
                { Name: 'Endpoint', Value: req.path },
                { Name: 'StatusCode', Value: String(res.statusCode) }
              ]
            }
          ]
        }).promise().catch(console.error);
      });
      
      next();
    });
    \`\`\`
    
    **Service Discovery Metrics**:
    \`\`\`javascript
    // Track service discovery health
    setInterval(async () => {
      const services = ['user-service', 'inventory-service', 'payment-service'];
      
      for (const service of services) {
        try {
          const response = await servicediscovery.discoverInstances({
            NamespaceName: 'internal.mycompany.local',
            ServiceName: service,
            HealthStatus: 'HEALTHY'
          }).promise();
          
          await cloudwatch.putMetricData({
            Namespace: 'ServiceDiscovery',
            MetricData: [
              {
                MetricName: 'HealthyInstances',
                Value: response.Instances.length,
                Unit: 'Count',
                Dimensions: [{ Name: 'Service', Value: service }]
              }
            ]
          }).promise();
        } catch (error) {
          console.error(\`Failed to check \${service}:\`, error);
        }
      }
    }, 60000); // Every minute
    \`\`\`
    
    **Alerts**:
    \`\`\`yaml
    # CloudWatch Alarm
    ServiceHealthyInstancesAlarm:
      Type: AWS::CloudWatch::Alarm
      Properties:
        AlarmName: user-service-low-healthy-instances
        AlarmDescription: Alert when healthy instances drop below 2
        Namespace: ServiceDiscovery
        MetricName: HealthyInstances
        Dimensions:
          - Name: Service
            Value: user-service
        Statistic: Average
        Period: 60
        EvaluationPeriods: 2
        Threshold: 2
        ComparisonOperator: LessThanThreshold
        AlarmActions:
          - !Ref SNSTopic
    \`\`\`
    
    **6. Testing Strategy**
    
    **Integration Test**:
    \`\`\`javascript
    describe('Service Discovery', () => {
      it('should discover user-service instances', async () => {
        const client = new ServiceClient('user-service');
        const response = await client.call('/health');
        expect(response.status).toBe('healthy');
      });
      
      it('should handle service unavailability gracefully', async () => {
        const client = new ServiceClient('nonexistent-service');
        await expect(client.call('/test')).rejects.toThrow();
      });
      
      it('should retry on transient failures', async () => {
        const client = new ServiceClient('flaky-service');
        // Mock: First 2 calls fail, 3rd succeeds
        const response = await client.call('/test');
        expect(response).toBeDefined();
      });
    });
    \`\`\`
    
    **Expected Results**:
    
    | **Metric** | **Target** | **Implementation** |
    |------------|-----------|-------------------|
    | **Service Discovery Latency** | <50ms | DNS caching, Cloud Map |
    | **Health Check Interval** | 30s | ECS health checks |
    | **Deregistration Time** | <30s | Health check + grace period |
    | **Failed Request Rate** | <0.1% | Retry logic, health checks |
    | **Graceful Shutdown Time** | <60s | Connection draining |
    
    **Key Takeaways**:
    
    1. **AWS Cloud Map** integrates natively with ECS for automatic registration
    2. **DNS-based discovery** is simplest (service-name.namespace.local)
    3. **Health checks** prevent routing to unhealthy tasks
    4. **Graceful shutdown** critical: mark unhealthy → stop accepting → drain → exit
    5. **Retry logic** handles transient failures
    6. **Monitor** healthy instance count, discovery latency, failed requests
    7. **TTL** of 10 seconds balances freshness and query load
    8. **Connection draining** prevents failed requests during deployments`,
    keyPoints: [
      'Consul for service discovery: DNS (simple) or HTTP API (advanced)',
      'Health checks critical: HTTP endpoint checking dependencies (DB, Redis, etc.)',
      'Client-side caching with 10-second TTL balances freshness and query load',
      'Connection draining during deployment: health check fails → removed from registry → existing connections finish',
      'Monitor: healthy instance count, discovery latency, failed request rate',
      'TTL trade-off: Lower = fresher data but more queries, Higher = less load but stale data',
    ],
  },
  {
    id: 'service-discovery-consul-cluster',
    question:
      'Design a highly available Consul cluster for service discovery in a production environment with 100+ microservices across 3 data centers. Include cluster topology, quorum requirements, network configuration, backup/recovery strategy, monitoring, and how to handle split-brain scenarios. Provide specific configurations and operational procedures.',
    sampleAnswer: `**Highly Available Consul Cluster Design**
    
    **1. Cluster Topology**
    
    **Architecture: Multi-Datacenter with WAN Federation**
    
    \`\`\`
    Datacenter 1 (us-east-1)          Datacenter 2 (us-west-2)          Datacenter 3 (eu-west-1)
    ├── Consul Server 1 (leader)      ├── Consul Server 4                ├── Consul Server 7
    ├── Consul Server 2               ├── Consul Server 5                ├── Consul Server 8
    ├── Consul Server 3               ├── Consul Server 6                ├── Consul Server 9
    └── Consul Clients (100+)         └── Consul Clients (100+)          └── Consul Clients (100+)
                ↓                                  ↓                                  ↓
        WAN Gossip Protocol ←→ WAN Gossip Protocol ←→ WAN Gossip Protocol
    \`\`\`
    
    **Quorum Requirements**:
    - **Consul uses Raft consensus** - requires majority (N/2 + 1) for writes
    - **3 servers per DC**: Tolerates 1 failure (quorum: 2/3)
    - **5 servers per DC**: Tolerates 2 failures (quorum: 3/5)
    - **Recommended**: 3 or 5 servers per DC (never even numbers!)
    
    **Why 3 DCs**:
    - Tolerates 1 entire DC failure
    - Majority still available: 2 DCs with 6 servers (quorum: 4/9)
    
    **2. Consul Server Configuration**
    
    **server.hcl** (Datacenter 1):
    \`\`\`hcl
    # Basic server config
    datacenter = "us-east-1"
    node_name = "consul-server-1"
    data_dir = "/opt/consul/data"
    log_level = "INFO"
    
    # Server mode
    server = true
    bootstrap_expect = 3
    
    # Network
    bind_addr = "10.0.1.10"      # Private IP
    advertise_addr = "10.0.1.10"
    client_addr = "0.0.0.0"      # Listen on all interfaces
    
    # Join other servers in same DC
    retry_join = ["10.0.1.11", "10.0.1.12"]
    
    # UI
    ui_config {
      enabled = true
    }
    
    # Performance tuning
    performance {
      raft_multiplier = 1  # Default (lower = faster, higher = more stable)
    }
    
    # Encryption
    encrypt = "base64-encoded-32-byte-key"
    encrypt_verify_incoming = true
    encrypt_verify_outgoing = true
    
    # TLS
    tls {
      defaults {
        ca_file = "/etc/consul/ca.pem"
        cert_file = "/etc/consul/server.pem"
        key_file = "/etc/consul/server-key.pem"
        verify_incoming = true
        verify_outgoing = true
      }
    }
    
    # ACLs
    acl {
      enabled = true
      default_policy = "deny"
      enable_token_persistence = true
      tokens {
        initial_management = "bootstrap-token"
        agent = "agent-token"
      }
    }
    
    # Autopilot (automatic operator-friendly management)
    autopilot {
      cleanup_dead_servers = true
      last_contact_threshold = "200ms"
      max_trailing_logs = 250
      server_stabilization_time = "10s"
    }
    
    # Telemetry
    telemetry {
      prometheus_retention_time = "60s"
      disable_hostname = false
    }
    \`\`\`
    
    **3. WAN Federation (Cross-DC)**
    
    **Primary DC (us-east-1) server.hcl**:
    \`\`\`hcl
    primary_datacenter = "us-east-1"
    
    # WAN join addresses
    retry_join_wan = [
      "consul-server-4.us-west-2.example.com",
      "consul-server-5.us-west-2.example.com",
      "consul-server-7.eu-west-1.example.com",
      "consul-server-8.eu-west-1.example.com"
    ]
    
    # WAN gossip encryption
    encrypt_wan = "base64-encoded-32-byte-key"
    \`\`\`
    
    **Secondary DC (us-west-2) server.hcl**:
    \`\`\`hcl
    datacenter = "us-west-2"
    primary_datacenter = "us-east-1"  # Replicates ACLs from primary
    
    retry_join_wan = [
      "consul-server-1.us-east-1.example.com",
      "consul-server-2.us-east-1.example.com",
      "consul-server-7.eu-west-1.example.com",
      "consul-server-8.eu-west-1.example.com"
    ]
    \`\`\`
    
    **4. Consul Client Configuration**
    
    **client.hcl** (on application servers):
    \`\`\`hcl
    datacenter = "us-east-1"
    node_name = "app-server-1"
    data_dir = "/opt/consul/data"
    
    # Client mode
    server = false
    
    # Join servers in same DC
    retry_join = [
      "consul-server-1.internal",
      "consul-server-2.internal",
      "consul-server-3.internal"
    ]
    
    bind_addr = "{{ GetPrivateIP }}"
    
    # Encryption
    encrypt = "base64-encoded-32-byte-key"
    
    # TLS
    tls {
      defaults {
        ca_file = "/etc/consul/ca.pem"
        verify_incoming = false  # Clients don't need incoming verification
        verify_outgoing = true
      }
    }
    
    # ACLs
    acl {
      enabled = true
      default_policy = "deny"
      tokens {
        agent = "agent-token"
        default = "service-token"
      }
    }
    \`\`\`
    
    **5. Service Registration**
    
    **service.hcl** (user-service):
    \`\`\`hcl
    service {
      name = "user-service"
      id = "user-service-1"
      port = 3000
      tags = ["v1", "http"]
      
      # Health check
      check {
        id = "user-service-health"
        name = "HTTP Health Check"
        http = "http://localhost:3000/health"
        interval = "10s"
        timeout = "5s"
        deregister_critical_service_after = "1m"
      }
      
      # Service metadata
      meta {
        version = "1.2.3"
        environment = "production"
      }
      
      # Connect (service mesh)
      connect {
        sidecar_service {}
      }
    }
    \`\`\`
    
    **6. Split-Brain Prevention**
    
    **Problem**: Network partition causes two groups of servers to elect separate leaders.
    
    **Solution 1: Odd Number of Servers**
    \`\`\`
    3 servers: Partition → 2 (quorum) vs 1 (no quorum)
    5 servers: Partition → 3 (quorum) vs 2 (no quorum)
    
    NEVER 4 servers: Partition → 2 vs 2 (both lose quorum!)
    \`\`\`
    
    **Solution 2: Datacenter-Aware Placement**
    - Deploy servers across availability zones
    - Never 2 servers in same AZ (for 3-server cluster)
    
    **Solution 3: Monitoring for Split**
    \`\`\`bash
    # Check for multiple leaders
    curl http://localhost:8500/v1/status/leader
    
    # Should return same leader across all servers
    # If different leaders returned, split-brain detected!
    \`\`\`
    
    **Solution 4: Automatic Recovery**
    \`\`\`hcl
    autopilot {
      cleanup_dead_servers = true
      # Servers unreachable for >72h automatically removed
      # Prevents split-brain from persisting
    }
    \`\`\`
    
    **7. Backup & Recovery**
    
    **Automated Snapshots**:
    \`\`\`bash
    #!/bin/bash
    # /usr/local/bin/consul-backup.sh
    
    # Take snapshot
    consul snapshot save \\
      -token=\${CONSUL_TOKEN} \\
      /backups/consul-snapshot-$(date +%Y%m%d-%H%M%S).snap
    
    # Upload to S3
    aws s3 cp /backups/consul-snapshot-*.snap \\
      s3://my-consul-backups/$(date +%Y/%m/%d)/
    
    # Retention: Keep last 30 days
    find /backups -name "consul-snapshot-*.snap" -mtime +30 -delete
    \`\`\`
    
    **Cron**:
    \`\`\`
    # Every 6 hours
    0 */6 * * * /usr/local/bin/consul-backup.sh
    \`\`\`
    
    **Disaster Recovery**:
    \`\`\`bash
    # Restore from snapshot
    consul snapshot restore \\
      -token=\${CONSUL_TOKEN} \\
      /backups/consul-snapshot-20250101-120000.snap
    
    # Verify
    consul catalog services
    consul catalog nodes
    \`\`\`
    
    **Restore Procedure**:
    1. Stop all Consul servers
    2. Clear data directories: \`rm -rf /opt/consul/data/*\`
    3. Start one server with restored snapshot
    4. Wait for it to become leader
    5. Start remaining servers (they'll sync from leader)
    
    **8. Monitoring**
    
    **Key Metrics**:
    \`\`\`yaml
    # Consul exposes Prometheus metrics at /v1/agent/metrics
    
    metrics_to_monitor:
      # Cluster health
      - consul_raft_peers: Number of Raft peers (should equal server count)
      - consul_raft_leader: Is this server the leader? (1 = yes)
      - consul_raft_apply_time: Time to apply Raft log (should be <10ms)
      
      # Performance
      - consul_rpc_request_time: RPC request latency
      - consul_serf_member_flap: Member flapping (joining/leaving rapidly)
      - consul_dns_query_time: DNS query latency
      
      # Service discovery
      - consul_catalog_services_total: Number of registered services
      - consul_health_node_status: Node health status
      - consul_health_service_status: Service health status
    \`\`\`
    
    **Prometheus Config**:
    \`\`\`yaml
    scrape_configs:
      - job_name: 'consul'
        metrics_path: '/v1/agent/metrics'
        params:
          format: ['prometheus']
        static_configs:
          - targets:
              - 'consul-server-1:8500'
              - 'consul-server-2:8500'
              - 'consul-server-3:8500'
    \`\`\`
    
    **Alerting Rules**:
    \`\`\`yaml
    groups:
      - name: consul
        rules:
          # No leader elected
          - alert: ConsulNoLeader
            expr: sum(consul_raft_leader) == 0
            for: 1m
            annotations:
              summary: "Consul cluster has no leader"
          
          # Raft peer mismatch
          - alert: ConsulRaftPeerMismatch
            expr: consul_raft_peers != 3
            for: 5m
            annotations:
              summary: "Expected 3 Raft peers, found {{ $value }}"
          
          # High RPC latency
          - alert: ConsulHighRPCLatency
            expr: histogram_quantile(0.99, consul_rpc_request_time) > 1000
            for: 5m
            annotations:
              summary: "Consul RPC p99 latency > 1s"
          
          # Service unhealthy
          - alert: ServiceUnhealthy
            expr: consul_health_service_status{status="critical"} > 0
            for: 2m
            annotations:
              summary: "Service {{ $labels.service }} is unhealthy"
    \`\`\`
    
    **9. Operational Procedures**
    
    **Adding a Server**:
    \`\`\`bash
    # 1. Deploy new server with same config
    # 2. It auto-joins via retry_join
    # 3. Autopilot promotes it to voter after stabilization
    
    # Verify
    consul operator raft list-peers
    \`\`\`
    
    **Removing a Server**:
    \`\`\`bash
    # Graceful removal (allows transfer of leadership)
    consul leave -id=consul-server-3
    
    # Verify
    consul operator raft list-peers
    \`\`\`
    
    **Rolling Restart**:
    \`\`\`bash
    # 1. Restart non-leader servers first
    for server in consul-server-2 consul-server-3; do
      ssh $server "systemctl restart consul"
      sleep 60  # Wait for it to rejoin
    done
    
    # 2. Transfer leadership
    consul operator raft transfer-leader -id=consul-server-2
    
    # 3. Restart old leader
    ssh consul-server-1 "systemctl restart consul"
    \`\`\`
    
    **10. Security Best Practices**
    
    **Encryption**:
    \`\`\`bash
    # Generate gossip encryption key
    consul keygen  # Base64-encoded 32-byte key
    
    # Generate TLS certificates
    consul tls ca create
    consul tls cert create -server -dc=us-east-1
    consul tls cert create -client
    \`\`\`
    
    **ACL Bootstrap**:
    \`\`\`bash
    # Bootstrap ACLs
    consul acl bootstrap
    
    # Output:
    # AccessorID:   2b778dd9-f5f1-6f29-b4b4-9a5fa948757a
    # SecretID:     6a1253d2-1785-24fd-91c2-f8e78c745511
    
    # Create policies
    consul acl policy create \\
      -name "service-policy" \\
      -rules @service-policy.hcl
    
    # Create tokens
    consul acl token create \\
      -description "User Service Token" \\
      -policy-name "service-policy"
    \`\`\`
    
    **service-policy.hcl**:
    \`\`\`hcl
    service "user-service" {
      policy = "write"
    }
    
    service_prefix "" {
      policy = "read"
    }
    
    node_prefix "" {
      policy = "read"
    }
    \`\`\`
    
    **Key Takeaways**:
    
    1. **3 or 5 servers per DC** (never even numbers) - tolerates N/2 failures
    2. **Odd number of DCs** (3 recommended) - tolerates 1 DC failure
    3. **WAN federation** connects DCs for global service discovery
    4. **Gossip encryption + TLS** for security
    5. **ACLs with deny-default** prevent unauthorized access
    6. **Autopilot** for automatic dead server cleanup
    7. **Automated snapshots** every 6 hours to S3
    8. **Monitor Raft peers, leader, and latency**
    9. **Split-brain prevented** by quorum requirements
    10. **Graceful operations**: leave before removing, transfer leadership before restart`,
    keyPoints: [
      'HA Consul cluster: 5 servers (tolerates 2 failures) across 3 datacenters',
      'Raft consensus requires quorum: (N/2)+1 servers must agree',
      'Client agents on every node: cache queries, reduce load on servers',
      'WAN gossip for cross-datacenter communication',
      'Split-brain prevented by quorum requirements',
      'Backup strategy: Consul snapshots every 6 hours, test restoration monthly',
      'Monitor: Raft peers, leader stability, commit latency',
      'Graceful operations: leave before removing node, transfer leadership before restart',
    ],
  },
  {
    id: 'service-discovery-performance',
    question:
      'Your service discovery system is experiencing performance issues: DNS queries are timing out, service registrations are slow, and clients are getting stale service information. Debug and optimize the system. Include specific diagnostic steps, performance bottlenecks to check, caching strategies, and configuration tuning. Provide before/after metrics showing improvements.',
    sampleAnswer: `**Service Discovery Performance Optimization**
    
    **Symptoms**:
    - DNS queries timing out (>1000ms)
    - Service registration taking 5-10 seconds
    - Clients routing to dead instances
    - Consul CPU at 90%
    
    **1. Diagnostic Steps**
    
    **Step 1: Check Consul Cluster Health**
    \`\`\`bash
    # Check leader and peers
    consul operator raft list-peers
    
    # Output should show:
    # Node         ID         Address       State     Voter
    # server-1     server-1   10.0.1.10:8300  leader    true
    # server-2     server-2   10.0.1.11:8300  follower  true
    # server-3     server-3   10.0.1.12:8300  follower  true
    
    # Check for frequent leader elections (bad!)
    consul monitor --log-level=info | grep "leader election"
    
    # Get metrics
    curl http://localhost:8500/v1/agent/metrics?format=prometheus
    \`\`\`
    
    **Step 2: Check DNS Query Performance**
    \`\`\`bash
    # Time DNS query
    time dig @localhost -p 8600 user-service.service.consul
    
    # Should be <50ms
    # If >500ms, DNS is the bottleneck
    
    # Check DNS cache hit rate
    consul monitor | grep "dns.query_time"
    \`\`\`
    
    **Step 3: Check Service Count and Churn**
    \`\`\`bash
    # How many services?
    consul catalog services | wc -l
    
    # How many instances?
    consul catalog nodes | wc -l
    
    # Check registration rate
    consul monitor | grep "catalog.register"
    
    # High churn (frequent register/deregister) is expensive
    \`\`\`
    
    **Step 4: Check Network Latency**
    \`\`\`bash
    # Measure round-trip time to Consul server
    ping -c 10 consul-server-1
    
    # Should be <10ms within same region
    # If >50ms, network is bottleneck
    
    # Check for packet loss
    mtr consul-server-1
    \`\`\`
    
    **Step 5: Analyze Metrics**
    \`\`\`bash
    # Key metrics to check:
    consul_raft_apply_time         # Should be <10ms
    consul_rpc_request_time        # Should be <100ms  
    consul_dns_query_time          # Should be <50ms
    consul_catalog_register_time   # Should be <100ms
    \`\`\`
    
    **2. Root Cause Analysis**
    
    **Issue 1: DNS Query Overload**
    
    **Problem**: Clients query Consul DNS on every request
    
    \`\`\`javascript
    // Bad: DNS query on every request
    async function callUserService() {
      // DNS query happens here (1000ms!)
      const response = await fetch('http://user-service.service.consul/users/123');
    }
    
    // Called 1000 times/second = 1000 DNS queries/sec
    \`\`\`
    
    **Solution: Client-Side DNS Caching**
    
    \`\`\`javascript
    const dns = require('dns');
    const { promisify } = require('util');
    const resolve4 = promisify(dns.resolve4);
    
    class CachedDNSResolver {
      constructor(ttl = 30000) {
        this.cache = new Map();
        this.ttl = ttl;
      }
      
      async resolve(hostname) {
        const now = Date.now();
        const cached = this.cache.get(hostname);
        
        // Return cached if fresh
        if (cached && now - cached.timestamp < this.ttl) {
          return cached.addresses;
        }
        
        // Resolve and cache
        const addresses = await resolve4(hostname);
        this.cache.set(hostname, {
          addresses,
          timestamp: now
        });
        
        return addresses;
      }
    }
    
    const resolver = new CachedDNSResolver(30000); // 30 second cache
    
    async function callUserService() {
      // DNS query cached for 30 seconds
      const addresses = await resolver.resolve('user-service.service.consul');
      const address = addresses[Math.floor(Math.random() * addresses.length)];
      
      const response = await fetch(\`http://\${address}:3000/users/123\`);
    }
    \`\`\`
    
    **Result**: DNS queries reduced from 1000/sec to 33/sec (97% reduction)
    
    **Issue 2: Consul DNS Configuration**
    
    **Problem**: Default DNS config not optimized
    
    **Before**:
    \`\`\`hcl
    # Default config
    dns_config {
      allow_stale = false  # Always query leader (slow!)
      max_stale = "0s"
    }
    \`\`\`
    
    **After (Optimized)**:
    \`\`\`hcl
    dns_config {
      # Allow stale reads from followers (fast!)
      allow_stale = true
      max_stale = "5s"
      
      # Enable DNS caching
      node_ttl = "30s"
      service_ttl = {
        "*" = "30s"
        "user-service" = "10s"  # Lower TTL for critical services
      }
      
      # Performance tuning
      udp_answer_limit = 3  # Limit UDP response size
      enable_truncate = true
    }
    \`\`\`
    
    **Result**: DNS query latency reduced from 500ms to 10ms (98% improvement)
    
    **Issue 3: Excessive Service Churn**
    
    **Problem**: Services register/deregister frequently
    
    \`\`\`javascript
    // Bad: Register on every request
    app.get('/users/:id', async (req, res) => {
      await consul.agent.service.register({...});
      // Handle request
      await consul.agent.service.deregister('user-service-1');
    });
    \`\`\`
    
    **Solution**: Register once on startup
    
    \`\`\`javascript
    // Register once
    await consul.agent.service.register({
      id: 'user-service-1',
      name: 'user-service',
      port: 3000,
      check: {
        http: 'http://localhost:3000/health',
        interval: '10s',
        timeout: '5s',
        deregister_critical_service_after: '1m'
      }
    });
    
    // Deregister only on shutdown
    process.on('SIGTERM', async () => {
      await consul.agent.service.deregister('user-service-1');
      process.exit(0);
    });
    \`\`\`
    
    **Issue 4: Health Check Intervals Too Aggressive**
    
    **Before**:
    \`\`\`hcl
    check {
      http = "http://localhost:3000/health"
      interval = "1s"    # Check every second (expensive!)
      timeout = "500ms"
    }
    \`\`\`
    
    **After**:
    \`\`\`hcl
    check {
      http = "http://localhost:3000/health"
      interval = "10s"   # Check every 10 seconds
      timeout = "5s"
      
      # Critical: Automatically deregister if unhealthy for 1 min
      deregister_critical_service_after = "1m"
    }
    \`\`\`
    
    **Result**: Health check load reduced by 90%
    
    **Issue 5: Raft Apply Latency**
    
    **Problem**: Raft consensus slow (writes to all replicas)
    
    **Diagnosis**:
    \`\`\`bash
    # Check Raft metrics
    curl localhost:8500/v1/agent/metrics | grep raft_apply
    
    # If consul_raft_apply_time >100ms, Raft is slow
    \`\`\`
    
    **Solution 1: Use Stale Reads for Read-Heavy Workloads**
    \`\`\`javascript
    // Query with stale=true (reads from followers, no consensus)
    const services = await consul.health.service({
      service: 'user-service',
      passing: true,
      stale: true  // Much faster!
    });
    \`\`\`
    
    **Solution 2: Tune Raft Multiplier**
    \`\`\`hcl
    performance {
      # Lower = faster (but less stable)
      # Higher = more stable (but slower)
      raft_multiplier = 1  # Default: 5
    }
    \`\`\`
    
    **Solution 3: Upgrade Network**
    - Ensure low-latency network between servers (<10ms)
    - Use placement groups or same AZ
    
    **Issue 6: Client Configuration**
    
    **Problem**: Clients not configured optimally
    
    **Before**:
    \`\`\`javascript
    // Bad: New connection per query
    async function getService(name) {
      const consul = new Consul();
      return await consul.health.service({ service: name });
    }
    \`\`\`
    
    **After**:
    \`\`\`javascript
    // Good: Reuse connection
    const consul = new Consul({
      host: 'consul.service.consul',
      port: 8500,
      promisify: true,
      defaults: {
        stale: true,  // Allow stale reads
        token: process.env.CONSUL_TOKEN
      }
    });
    
    // Cache service locations
    class ServiceCache {
      constructor(refreshInterval = 30000) {
        this.cache = new Map();
        setInterval(() => this.refresh(), refreshInterval);
      }
      
      async get(serviceName) {
        if (!this.cache.has(serviceName)) {
          await this.refresh(serviceName);
        }
        
        const instances = this.cache.get(serviceName);
        return instances[Math.floor(Math.random() * instances.length)];
      }
      
      async refresh(serviceName = null) {
        const services = serviceName ? [serviceName] : Array.from(this.cache.keys());
        
        for (const service of services) {
          const instances = await consul.health.service({
            service,
            passing: true,
            stale: true
          });
          
          this.cache.set(service, instances.map(i => ({
            address: i.Service.Address,
            port: i.Service.Port
          })));
        }
      }
    }
    
    const serviceCache = new ServiceCache(30000);
    \`\`\`
    
    **3. Performance Improvements Summary**
    
    | **Metric** | **Before** | **After** | **Improvement** |
    |------------|-----------|---------|----------------|
    | **DNS Query Latency (p99)** | 500ms | 10ms | 98% faster |
    | **DNS Query Rate** | 1000/sec | 33/sec | 97% reduction |
    | **Service Registration Time** | 5s | 200ms | 96% faster |
    | **Consul CPU Usage** | 90% | 25% | 72% reduction |
    | **Failed Requests (stale data)** | 5% | 0.1% | 98% reduction |
    | **Health Check Load** | 1000 checks/sec | 100 checks/sec | 90% reduction |
    
    **4. Monitoring Dashboard**
    
    \`\`\`yaml
    # Grafana dashboard queries
    
    # DNS query latency
    histogram_quantile(0.99, 
      rate(consul_dns_query_time_bucket[5m])
    )
    
    # Service cache hit rate
    rate(service_cache_hits[5m]) / 
      (rate(service_cache_hits[5m]) + rate(service_cache_misses[5m]))
    
    # Raft apply latency
    consul_raft_apply_time
    
    # Registration rate
    rate(consul_catalog_register[5m])
    \`\`\`
    
    **Key Takeaways**:
    
    1. **Client-side DNS caching** (30s TTL) reduces queries by 97%
    2. **Allow stale reads** for followers improves latency by 98%
    3. **Service TTL configuration** enables DNS caching
    4. **Health check intervals** should be 10s (not 1s)
    5. **Reuse Consul client** connections
    6. **Cache service locations** and refresh periodically
    7. **Raft multiplier tuning** for write-heavy workloads
    8. **Monitor**: DNS latency, query rate, Raft apply time, CPU
    9. **Graceful registration**: once on startup, not per request
    10. **Use stale=true** for read-heavy workloads (99% of queries)`,
    keyPoints: [
      'Root cause: DNS cache disabled, anti-entropy interval too short, no client-side caching',
      'Enable DNS caching with 10-second TTL on Consul servers',
      'Reduce anti-entropy sync interval from 1s to 60s',
      'Client-side caching with 10-second TTL to reduce DNS queries by 90%',
      'Use stale=true for DNS queries to avoid leader bottleneck',
      'Monitor: DNS query latency, query rate, Raft apply time, CPU usage',
      'Expected improvements: 100ms → 5ms DNS latency, 90% reduction in DNS queries',
    ],
  },
];
