/**
 * Quiz questions for Service Discovery & Registry section
 */

export const servicediscoveryQuiz = [
  {
    id: 'q1-discovery',
    question:
      'Design a service discovery system for a Kubernetes-based microservices platform with 50+ services. Explain how services register, how clients discover services, how you handle health checks, load balancing strategies, and failure scenarios. Include specific implementation details for both DNS-based and API-based discovery.',
    sampleAnswer: `**Service Discovery System for Kubernetes**

**1. Architecture Overview**

\`\`\`
┌─────────────────────────────────────────────┐
│         Kubernetes Cluster                  │
│                                             │
│  ┌─────────────────────┐                    │
│  │  Kubernetes DNS     │                    │
│  │  (CoreDNS)          │                    │
│  └─────────────────────┘                    │
│           ↑                                 │
│           │                                 │
│  ┌────────┴────────┐                        │
│  │  kube-apiserver │                        │
│  └────────┬────────┘                        │
│           ↓                                 │
│  ┌────────────────────┐                     │
│  │   Service Registry │                     │
│  │   (etcd)           │                     │
│  └────────────────────┘                     │
│           ↑                                 │
│           │                                 │
│  ┌────────┴─────────────┐                   │
│  │  Services & Endpoints│                   │
│  │                      │                   │
│  │  order-service       │                   │
│  │  ├─ pod-1 (10.0.1.5) │                   │
│  │  ├─ pod-2 (10.0.1.6) │                   │
│  │  └─ pod-3 (10.0.1.7) │                   │
│  │                      │                   │
│  │  payment-service     │                   │
│  │  ├─ pod-1 (10.0.2.3) │                   │
│  │  └─ pod-2 (10.0.2.4) │                   │
│  └──────────────────────┘                   │
└─────────────────────────────────────────────┘
\`\`\`

---

**2. Service Registration**

**Kubernetes Service Definition**:

\`\`\`yaml
# order-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: order-service
  namespace: production
  labels:
    app: order-service
    version: v1
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
spec:
  selector:
    app: order-service  # Matches pods with this label
  ports:
    - name: http
      protocol: TCP
      port: 80        # Service port
      targetPort: 8080 # Pod port
    - name: grpc
      protocol: TCP
      port: 50051
      targetPort: 50051
  type: ClusterIP  # Internal service (not exposed externally)
  sessionAffinity: None
---
apiVersion: v1
kind: Endpoints
metadata:
  name: order-service
  namespace: production
subsets:
  - addresses:
      - ip: 10.0.1.5
        nodeName: node-1
        targetRef:
          kind: Pod
          name: order-service-pod-1
          namespace: production
      - ip: 10.0.1.6
        nodeName: node-2
        targetRef:
          kind: Pod
          name: order-service-pod-2
          namespace: production
    ports:
      - name: http
        port: 8080
        protocol: TCP
      - name: grpc
        port: 50051
        protocol: TCP
\`\`\`

**Automatic Registration** (Kubernetes handles this):

When you deploy a pod:
1. **Pod starts** → kubelet reports to kube-apiserver
2. **kube-apiserver** → updates etcd with pod IP
3. **Endpoint Controller** → watches for pods matching Service selector
4. **Endpoints object created** → lists all pod IPs
5. **CoreDNS updated** → DNS resolves service name to ClusterIP

---

**3. DNS-Based Discovery**

**CoreDNS Configuration**:

\`\`\`
# CoreDNS Corefile
.:53 {
    errors
    health {
        lameduck 5s
    }
    ready
    kubernetes cluster.local in-addr.arpa ip6.arpa {
        pods insecure
        fallthrough in-addr.arpa ip6.arpa
        ttl 30
    }
    prometheus :9153
    forward . /etc/resolv.conf {
        max_concurrent 1000
    }
    cache 30
    loop
    reload
    loadbalance
}
\`\`\`

**Client Discovery via DNS**:

\`\`\`javascript
// Simple DNS lookup (returns ClusterIP)
const dns = require('dns');

dns.resolve4('order-service.production.svc.cluster.local', (err, addresses) => {
  if (err) throw err;
  console.log('Service IP:', addresses[0]); // ClusterIP: e.g., 10.96.0.10
});

// Kubernetes routes ClusterIP to one of the pod IPs via iptables
\`\`\`

**DNS Service Name Patterns**:

\`\`\`
<service-name>.<namespace>.svc.<cluster-domain>

Examples:
- order-service.production.svc.cluster.local  (full FQDN)
- order-service.production                    (within same cluster)
- order-service                               (within same namespace)
\`\`\`

**Advantages**:
✅ Universal (all languages support DNS)
✅ No client library needed
✅ Built-in to Kubernetes

**Disadvantages**:
❌ DNS caching can cause stale IPs
❌ Limited load balancing (round-robin only)
❌ No advanced routing (canary, A/B testing)

---

**4. API-Based Discovery (Kubernetes API)**

**Direct API Access**:

\`\`\`javascript
const k8s = require('@kubernetes/client-node');

const kc = new k8s.KubeConfig();
kc.loadFromDefault();

const k8sApi = kc.makeApiClient(k8s.CoreV1Api);

// Get service endpoints
async function getServiceEndpoints(serviceName, namespace) {
  try {
    const response = await k8sApi.readNamespacedEndpoints(serviceName, namespace);
    
    const endpoints = [];
    response.body.subsets.forEach(subset => {
      subset.addresses.forEach(address => {
        subset.ports.forEach(port => {
          endpoints.push({
            ip: address.ip,
            port: port.port,
            nodeName: address.nodeName,
            podName: address.targetRef?.name
          });
        });
      });
    });
    
    return endpoints;
  } catch (error) {
    console.error('Failed to get endpoints:', error);
    throw error;
  }
}

// Usage
const endpoints = await getServiceEndpoints('order-service', 'production');
console.log(endpoints);
// [
//   { ip: '10.0.1.5', port: 8080, nodeName: 'node-1', podName: 'order-service-pod-1' },
//   { ip: '10.0.1.6', port: 8080, nodeName: 'node-2', podName: 'order-service-pod-2' }
// ]
\`\`\`

**Watch for Changes** (real-time updates):

\`\`\`javascript
const watch = new k8s.Watch(kc);

watch.watch('/api/v1/namespaces/production/endpoints',
  {},
  (type, apiObj, watchObj) => {
    console.log('Event type:', type); // ADDED, MODIFIED, DELETED
    console.log('Endpoints:', apiObj);
    
    if (type === 'MODIFIED') {
      // Update local cache
      updateServiceRegistry(apiObj);
    }
  },
  (err) => {
    console.error('Watch error:', err);
  }
);
\`\`\`

---

**5. Health Checks**

**Liveness & Readiness Probes**:

\`\`\`yaml
apiVersion: v1
kind: Pod
metadata:
  name: order-service-pod
spec:
  containers:
  - name: order-service
    image: order-service:v1
    ports:
    - containerPort: 8080
    
    # Liveness probe: Is the app running?
    livenessProbe:
      httpGet:
        path: /health/live
        port: 8080
      initialDelaySeconds: 30
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3
    
    # Readiness probe: Is the app ready to serve traffic?
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 5
      timeoutSeconds: 3
      failureThreshold: 2
    
    # Startup probe: Has the app started yet?
    startupProbe:
      httpGet:
        path: /health/startup
        port: 8080
      initialDelaySeconds: 0
      periodSeconds: 5
      failureThreshold: 30  # Allow up to 150 seconds to start
\`\`\`

**Health Endpoint Implementation**:

\`\`\`javascript
const express = require('express');
const app = express();

let isReady = false;

// Liveness: Is the process alive?
app.get('/health/live', (req, res) => {
  // Simple check - just return 200 if process is running
  res.status(200).json({ status: 'alive' });
});

// Readiness: Can the app serve traffic?
app.get('/health/ready', async (req, res) => {
  try {
    // Check dependencies
    await checkDatabaseConnection();
    await checkRedisConnection();
    await checkKafkaConnection();
    
    res.status(200).json({
      status: 'ready',
      dependencies: {
        database: 'ok',
        redis: 'ok',
        kafka: 'ok'
      }
    });
  } catch (error) {
    res.status(503).json({
      status: 'not ready',
      error: error.message
    });
  }
});

// Startup: Has the app finished initializing?
app.get('/health/startup', (req, res) => {
  if (isReady) {
    res.status(200).json({ status: 'started' });
  } else {
    res.status(503).json({ status: 'starting' });
  }
});

// After initialization
async function initialize() {
  await loadConfiguration();
  await connectToDatabase();
  await warmupCache();
  isReady = true;
}

initialize();
\`\`\`

**Behavior**:
- **Liveness fails** → Kubernetes **restarts** the pod
- **Readiness fails** → Kubernetes **removes pod from Endpoints** (no traffic)
- **Startup fails** → Pod marked as failed (after threshold)

---

**6. Load Balancing Strategies**

**Built-in Kubernetes Load Balancing** (kube-proxy):

\`\`\`bash
# iptables mode (default)
# kube-proxy creates iptables rules that redirect ClusterIP traffic to pod IPs

# Check iptables rules
sudo iptables-save | grep order-service

# Example rule:
# -A KUBE-SERVICES -d 10.96.0.10/32 -p tcp -m tcp --dport 80 -j KUBE-SVC-ORDER
# -A KUBE-SVC-ORDER -m statistic --mode random --probability 0.33 -j KUBE-SEP-POD1
# -A KUBE-SVC-ORDER -m statistic --mode random --probability 0.50 -j KUBE-SEP-POD2
# -A KUBE-SVC-ORDER -j KUBE-SEP-POD3
\`\`\`

**Custom Client-Side Load Balancing**:

\`\`\`javascript
class ServiceDiscoveryClient {
  constructor(serviceName, namespace) {
    this.serviceName = serviceName;
    this.namespace = namespace;
    this.endpoints = [];
    this.currentIndex = 0;
    
    // Watch for endpoint changes
    this.startWatch();
  }
  
  async startWatch() {
    const k8sApi = kc.makeApiClient(k8s.CoreV1Api);
    
    // Initial fetch
    await this.refreshEndpoints();
    
    // Watch for changes
    const watch = new k8s.Watch(kc);
    watch.watch(
      \`/api/v1/namespaces/\${this.namespace}/endpoints/\${this.serviceName}\`,
      {},
      (type, apiObj) => {
        if (type === 'MODIFIED' || type === 'ADDED') {
          this.updateEndpoints(apiObj);
        }
      },
      (err) => console.error('Watch error:', err)
    );
  }
  
  async refreshEndpoints() {
    const k8sApi = kc.makeApiClient(k8s.CoreV1Api);
    const response = await k8sApi.readNamespacedEndpoints(
      this.serviceName,
      this.namespace
    );
    this.updateEndpoints(response.body);
  }
  
  updateEndpoints(endpointsObj) {
    this.endpoints = [];
    endpointsObj.subsets?.forEach(subset => {
      subset.addresses?.forEach(address => {
        subset.ports?.forEach(port => {
          this.endpoints.push({
            ip: address.ip,
            port: port.port,
            url: \`http://\${address.ip}:\${port.port}\`
          });
        });
      });
    });
    console.log(\`Updated endpoints for \${this.serviceName}:\`, this.endpoints.length);
  }
  
  // Round-robin load balancing
  getNextEndpoint() {
    if (this.endpoints.length === 0) {
      throw new Error('No healthy endpoints available');
    }
    
    const endpoint = this.endpoints[this.currentIndex];
    this.currentIndex = (this.currentIndex + 1) % this.endpoints.length;
    return endpoint;
  }
  
  // Random selection
  getRandomEndpoint() {
    if (this.endpoints.length === 0) {
      throw new Error('No healthy endpoints available');
    }
    
    const index = Math.floor(Math.random() * this.endpoints.length);
    return this.endpoints[index];
  }
  
  // Least connections (would need connection tracking)
  getLeastConnectionsEndpoint() {
    // Track active connections per endpoint
    // Return endpoint with fewest connections
  }
}

// Usage
const orderService = new ServiceDiscoveryClient('order-service', 'production');

async function callOrderService(data) {
  const endpoint = orderService.getNextEndpoint();
  
  try {
    const response = await fetch(\`\${endpoint.url}/api/orders\`, {
      method: 'POST',
      body: JSON.stringify(data)
    });
    return await response.json();
  } catch (error) {
    console.error(\`Failed to call \${endpoint.url}:\`, error);
    
    // Retry with different endpoint
    const retryEndpoint = orderService.getNextEndpoint();
    const response = await fetch(\`\${retryEndpoint.url}/api/orders\`, {
      method: 'POST',
      body: JSON.stringify(data)
    });
    return await response.json();
  }
}
\`\`\`

---

**7. Failure Scenarios**

**Scenario 1: Pod Crashes**

\`\`\`
1. Pod crashes
2. Liveness probe fails (after 3 attempts)
3. Kubernetes restarts pod
4. New pod starts
5. Readiness probe fails initially
6. Pod NOT added to Endpoints (no traffic yet)
7. After warmup, readiness succeeds
8. Pod added to Endpoints
9. Traffic resumes
\`\`\`

**Scenario 2: Database Connection Lost**

\`\`\`
1. Database connection lost
2. Readiness probe fails (can't connect to DB)
3. Kubernetes removes pod from Endpoints
4. Existing connections continue (TCP still alive)
5. New requests go to other pods
6. App reconnects to database
7. Readiness probe succeeds
8. Pod re-added to Endpoints
\`\`\`

**Scenario 3: DNS Caching Issues**

\`\`\`
Problem: Client cached old IP, pod is dead

Solution:
- Set low DNS TTL (30 seconds)
- Implement retry logic
- Use client-side discovery (watch API)
\`\`\`

**Scenario 4: Rolling Deployment**

\`\`\`yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2        # Can have 12 pods during rollout
      maxUnavailable: 1  # Max 1 pod down at a time
  template:
    spec:
      containers:
      - name: order-service
        image: order-service:v2
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]  # Grace period
\`\`\`

**Behavior**:
1. New pod (v2) starts
2. Readiness probe passes
3. v2 pod added to Endpoints
4. v1 pod receives SIGTERM
5. 15-second grace period (finish in-flight requests)
6. v1 pod shuts down
7. Repeat until all pods are v2

---

**Key Takeaways**:

1. **DNS-based discovery**: Simple, universal, but limited
2. **API-based discovery**: Real-time updates, advanced routing
3. **Health checks**: Liveness (restart), Readiness (traffic), Startup (initialization)
4. **Load balancing**: kube-proxy (iptables), client-side (custom logic)
5. **Failure handling**: Automatic pod removal from Endpoints when unhealthy
6. **Rolling deployments**: Zero-downtime updates with preStop hooks
7. **Watch API**: Real-time endpoint updates for client-side discovery`,
    keyPoints: [
      'DNS-based discovery: CoreDNS resolves service names to ClusterIPs (simple but limited)',
      'API-based discovery: Watch Kubernetes Endpoints API for real-time pod IP updates',
      'Health checks: Liveness (restart pod), Readiness (add/remove from Endpoints), Startup (initialization)',
      'Load balancing: kube-proxy iptables (random) or client-side (round-robin, least connections)',
      'Failure handling: Unhealthy pods automatically removed from Endpoints',
      'Rolling deployments: Use maxSurge, maxUnavailable, and preStop hooks for zero downtime',
    ],
  },
  {
    id: 'q2-discovery',
    question:
      'Your service discovery system is experiencing issues where clients occasionally connect to terminated pods, causing 5% of requests to fail. Diagnose the root cause and propose solutions. Consider DNS caching, endpoint propagation delays, graceful shutdown, and client retry logic.',
    sampleAnswer: `**Diagnosing Stale Endpoint Connections**

**Problem**: Clients connecting to terminated pods → 5% request failure rate

---

**Root Cause Analysis**

**1. DNS TTL Caching**

\`\`\`javascript
// Client caches DNS lookup
const dns = require('dns');

dns.resolve4('order-service.production.svc.cluster.local', (err, addresses) => {
  console.log('Cached IP:', addresses[0]); // May be stale!
});

// DNS TTL: 30 seconds (default)
// If pod terminates, client still uses cached IP for up to 30 seconds
\`\`\`

**Issue**: Client cached ClusterIP → kube-proxy cached iptables rules → routes to dead pod

---

**2. Endpoint Propagation Delay**

\`\`\`
Timeline of pod termination:

T+0s:   kubectl delete pod order-service-pod-1
T+0.1s: kubelet receives SIGTERM
T+0.1s: Pod status → "Terminating"
T+0.2s: kube-apiserver updates etcd
T+0.5s: Endpoint Controller removes pod from Endpoints
T+1s:   kube-proxy updates iptables rules
T+2s:   CoreDNS cache expires

Problem: Between T+0s and T+2s, clients can still route to dead pod!
\`\`\`

---

**3. Lack of Graceful Shutdown**

\`\`\`javascript
// Server without graceful shutdown
const server = app.listen(8080);

// On SIGTERM → immediate shutdown
process.on('SIGTERM', () => {
  server.close(); // Closes immediately, drops in-flight requests
  process.exit(0);
});

// Problem: Active connections terminated abruptly
\`\`\`

---

**4. No Client Retry Logic**

\`\`\`javascript
// Client without retry
const response = await fetch('http://order-service/api/orders');
// If pod just terminated → Connection refused → Request fails
// No retry → 5% failure rate
\`\`\`

---

**Solutions**

**Solution 1: Implement Graceful Shutdown**

\`\`\`javascript
const express = require('express');
const app = express();

const server = app.listen(8080, () => {
  console.log('Server started on port 8080');
});

let isShuttingDown = false;

// Health check endpoint
app.get('/health/ready', (req, res) => {
  if (isShuttingDown) {
    // Signal to Kubernetes that we're not ready for new traffic
    return res.status(503).json({ status: 'shutting down' });
  }
  res.status(200).json({ status: 'ready' });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('SIGTERM received, starting graceful shutdown...');
  
  // Step 1: Stop accepting new connections
  isShuttingDown = true;
  
  // Step 2: Wait for Kubernetes to remove pod from Endpoints (5-10 seconds)
  console.log('Waiting for Kubernetes to remove pod from load balancer...');
  await sleep(10000); // 10 seconds
  
  // Step 3: Close server (finish in-flight requests, reject new ones)
  server.close(() => {
    console.log('All connections closed');
    
    // Step 4: Clean up resources
    disconnectFromDatabase();
    disconnectFromRedis();
    
    // Step 5: Exit
    process.exit(0);
  });
  
  // Step 6: Force exit after timeout (prevent hanging)
  setTimeout(() => {
    console.error('Forced shutdown after timeout');
    process.exit(1);
  }, 30000); // 30 seconds max
});

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
\`\`\`

**Kubernetes Pod Spec**:

\`\`\`yaml
apiVersion: v1
kind: Pod
metadata:
  name: order-service-pod
spec:
  containers:
  - name: order-service
    image: order-service:v1
    
    lifecycle:
      preStop:
        exec:
          # Sleep before shutdown to allow endpoint removal
          command: ["/bin/sh", "-c", "sleep 15"]
    
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8080
      periodSeconds: 5
  
  terminationGracePeriodSeconds: 30  # Max time for graceful shutdown
\`\`\`

**Shutdown Timeline**:

\`\`\`
T+0s:   kubectl delete pod
T+0s:   Kubernetes sends SIGTERM to container
T+0s:   preStop hook executes (sleep 15)
T+0s:   App sets isShuttingDown = true
T+0s:   Readiness probe fails (returns 503)
T+5s:   Kubernetes removes pod from Endpoints
T+6s:   kube-proxy updates iptables (no new traffic)
T+15s:  preStop hook completes
T+15s:  App receives SIGTERM
T+15s:  App closes server (finishes in-flight requests)
T+20s:  All connections closed
T+20s:  Process exits
\`\`\`

**Result**: Zero dropped requests!

---

**Solution 2: Client-Side Retry with Exponential Backoff**

\`\`\`javascript
async function retryableRequest(url, options = {}, maxRetries = 3) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(url, options);
      
      // Retry on specific status codes
      if (response.status >= 500 && response.status < 600) {
        throw new Error(\`Server error: \${response.status}\`);
      }
      
      if (response.status === 429) {
        // Rate limited - retry after delay
        const retryAfter = response.headers.get('Retry-After') || 5;
        await sleep(retryAfter * 1000);
        continue;
      }
      
      return response;
      
    } catch (error) {
      console.error(\`Attempt \${attempt + 1} failed:\`, error.message);
      
      // Don't retry on last attempt
      if (attempt === maxRetries - 1) {
        throw error;
      }
      
      // Retry only on network errors or 5xx errors
      if (error.code === 'ECONNREFUSED' ||
          error.code === 'ECONNRESET' ||
          error.code === 'ETIMEDOUT' ||
          error.message.includes('Server error')) {
        
        // Exponential backoff: 100ms, 200ms, 400ms
        const delay = Math.pow(2, attempt) * 100;
        console.log(\`Retrying in \${delay}ms...\`);
        await sleep(delay);
        continue;
      }
      
      // Don't retry on client errors (4xx)
      throw error;
    }
  }
}

// Usage
const response = await retryableRequest('http://order-service/api/orders', {
  method: 'POST',
  body: JSON.stringify({ items: [...] })
});
\`\`\`

---

**Solution 3: Reduce DNS TTL**

\`\`\`yaml
# CoreDNS ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: coredns
  namespace: kube-system
data:
  Corefile: |
    .:53 {
        kubernetes cluster.local in-addr.arpa ip6.arpa {
            ttl 10  # Reduce from 30s to 10s
        }
        cache 10    # Reduce cache duration
    }
\`\`\`

**Trade-off**: More frequent DNS lookups → higher load on CoreDNS

---

**Solution 4: Client-Side Service Discovery**

Instead of DNS, watch Kubernetes Endpoints API directly:

\`\`\`javascript
const k8s = require('@kubernetes/client-node');

class ServiceDiscoveryClient {
  constructor(serviceName, namespace) {
    this.serviceName = serviceName;
    this.namespace = namespace;
    this.endpoints = [];
    this.watch();
  }
  
  async watch() {
    const kc = new k8s.KubeConfig();
    kc.loadFromDefault();
    
    const watch = new k8s.Watch(kc);
    
    watch.watch(
      \`/api/v1/namespaces/\${this.namespace}/endpoints/\${this.serviceName}\`,
      {},
      (type, apiObj) => {
        console.log('Endpoint event:', type);
        
        if (type === 'MODIFIED' || type === 'ADDED') {
          this.updateEndpoints(apiObj);
        } else if (type === 'DELETED') {
          this.endpoints = [];
        }
      },
      (err) => console.error('Watch error:', err)
    );
  }
  
  updateEndpoints(endpointsObj) {
    this.endpoints = [];
    
    endpointsObj.subsets?.forEach(subset => {
      subset.addresses?.forEach(address => {
        subset.ports?.forEach(port => {
          this.endpoints.push({
            ip: address.ip,
            port: port.port,
            url: \`http://\${address.ip}:\${port.port}\`
          });
        });
      });
    });
    
    console.log(\`Updated endpoints: \${this.endpoints.length} healthy pods\`);
  }
  
  getEndpoint() {
    if (this.endpoints.length === 0) {
      throw new Error('No healthy endpoints');
    }
    
    // Round-robin
    const endpoint = this.endpoints[Math.floor(Math.random() * this.endpoints.length)];
    return endpoint;
  }
}

// Usage
const orderService = new ServiceDiscoveryClient('order-service', 'production');

async function callOrderService(data) {
  const endpoint = orderService.getEndpoint();
  
  const response = await fetch(\`\${endpoint.url}/api/orders\`, {
    method: 'POST',
    body: JSON.stringify(data)
  });
  
  return await response.json();
}
\`\`\`

**Benefits**:
✅ Real-time endpoint updates (no DNS caching)
✅ Immediate removal of unhealthy pods
✅ Custom load balancing strategies

---

**Solution 5: Connection Draining with Readiness Gates**

\`\`\`yaml
apiVersion: v1
kind: Pod
metadata:
  name: order-service-pod
spec:
  readinessGates:
  - conditionType: "example.com/connection-drained"
  
  containers:
  - name: order-service
    image: order-service:v1
    
    lifecycle:
      preStop:
        httpGet:
          path: /shutdown
          port: 8080
\`\`\`

\`\`\`javascript
app.post('/shutdown', async (req, res) => {
  isShuttingDown = true;
  
  // Wait for active connections to finish
  const activeConnections = getActiveConnectionCount();
  console.log(\`Waiting for \${activeConnections} connections to finish...\`);
  
  // Poll until all connections closed
  while (getActiveConnectionCount() > 0) {
    await sleep(1000);
  }
  
  res.status(200).json({ status: 'ready to shutdown' });
});
\`\`\`

---

**Monitoring & Validation**

\`\`\`javascript
// Track connection failures
const prometheus = require('prom-client');

const connectionFailures = new prometheus.Counter({
  name: 'http_connection_failures_total',
  help: 'Total number of connection failures',
  labelNames: ['service', 'error']
});

async function monitoredRequest(url, options) {
  try {
    return await fetch(url, options);
  } catch (error) {
    if (error.code === 'ECONNREFUSED') {
      connectionFailures.inc({ service: 'order-service', error: 'refused' });
    }
    throw error;
  }
}

// Alert on high failure rate
// Alert: http_connection_failures_total > 5% of requests
\`\`\`

---

**Summary of Solutions**

| Solution | Impact | Complexity |
|----------|--------|-----------|
| Graceful shutdown | ✅ Eliminates 90% of errors | Low |
| Client retry | ✅ Handles remaining 10% | Low |
| Reduce DNS TTL | ⚠️ Reduces DNS caching issues | Low |
| Client-side discovery | ✅ Real-time updates, no DNS lag | High |
| Connection draining | ✅ Zero dropped connections | Medium |

**Recommended Approach**:
1. Implement graceful shutdown (must-have)
2. Add client retry logic (must-have)
3. Reduce DNS TTL to 10s (nice-to-have)
4. Consider client-side discovery for critical services (optional)`,
    keyPoints: [
      'Root cause: DNS caching + endpoint propagation delay + lack of graceful shutdown',
      'Graceful shutdown: preStop hook (sleep 15s) + fail readiness probe + finish in-flight requests',
      'Client retry: Exponential backoff for ECONNREFUSED, ECONNRESET, 5xx errors (3 retries max)',
      'Reduce DNS TTL from 30s to 10s to minimize stale IP caching',
      'Client-side discovery: Watch Kubernetes Endpoints API for real-time pod updates',
      'Monitoring: Track connection failures with Prometheus, alert on >5% failure rate',
    ],
  },
  {
    id: 'q3-discovery',
    question:
      'Design a multi-region service discovery system for a global application deployed across AWS us-east-1, eu-west-1, and ap-southeast-1. Explain how you would handle cross-region discovery, latency-based routing, regional failover, health checks, and data consistency. Include specific implementation details for both Kubernetes and external service registries like Consul.',
    sampleAnswer: `**Multi-Region Service Discovery System**

**1. Architecture Overview**

\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                   Global Load Balancer                       │
│                  (AWS Route53 / Cloudflare)                  │
│              Latency-based routing + Health checks           │
└───────────────┬────────────────┬────────────────────────────┘
                │                │                             
    ┌───────────┴───────┐   ┌────┴────────┐   ┌──────────────┴──────┐
    │   us-east-1       │   │  eu-west-1  │   │   ap-southeast-1    │
    │   (Primary)       │   │  (Regional) │   │   (Regional)        │
    │                   │   │             │   │                     │
    │  ┌─────────────┐  │   │ ┌─────────┐ │   │  ┌─────────────┐   │
    │  │ Kubernetes  │  │   │ │Kubernetes│ │  │  │ Kubernetes  │   │
    │  │  Cluster    │  │   │ │ Cluster │ │   │  │  Cluster    │   │
    │  └─────────────┘  │   │ └─────────┘ │   │  └─────────────┘   │
    │                   │   │             │   │                     │
    │  ┌─────────────┐  │   │ ┌─────────┐ │   │  ┌─────────────┐   │
    │  │   Consul    │◄─┼───┼─┤ Consul  │◄┼───┼──┤   Consul    │   │
    │  │   (Leader)  │──┼───┼─▶(Follower)├─┼───┼─▶│  (Follower) │   │
    │  └─────────────┘  │   │ └─────────┘ │   │  └─────────────┘   │
    │                   │   │             │   │                     │
    └───────────────────┘   └─────────────┘   └─────────────────────┘
                │                │                      │
                └────────────────┴──────────────────────┘
                         WAN Gossip (Consul)
                         Cross-region replication
\`\`\`

---

**2. Kubernetes Setup (Per Region)**

**Multi-Cluster Service Mesh (Istio)**:

\`\`\`yaml
# us-east-1 cluster
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: istio-controlplane
spec:
  values:
    global:
      multiCluster:
        clusterName: us-east-1
      network: us-east-network
      meshID: global-mesh
\`\`\`

**ServiceEntry for Cross-Region Discovery**:

\`\`\`yaml
# Define external service in eu-west-1
apiVersion: networking.istio.io/v1beta1
kind: ServiceEntry
metadata:
  name: order-service-eu
  namespace: production
spec:
  hosts:
  - order-service.eu-west-1.global
  ports:
  - number: 80
    name: http
    protocol: HTTP
  location: MESH_EXTERNAL
  resolution: DNS
  endpoints:
  - address: order-service.eu-west-1.svc.cluster.local
    ports:
      http: 80
    locality: eu-west-1
    labels:
      region: eu-west-1
---
# Define external service in ap-southeast-1
apiVersion: networking.istio.io/v1beta1
kind: ServiceEntry
metadata:
  name: order-service-ap
  namespace: production
spec:
  hosts:
  - order-service.ap-southeast-1.global
  ports:
  - number: 80
    name: http
    protocol: HTTP
  location: MESH_EXTERNAL
  resolution: DNS
  endpoints:
  - address: order-service.ap-southeast-1.svc.cluster.local
    ports:
      http: 80
    locality: ap-southeast-1
    labels:
      region: ap-southeast-1
\`\`\`

**DestinationRule for Locality-Based Routing**:

\`\`\`yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: order-service-global
  namespace: production
spec:
  host: order-service.global
  trafficPolicy:
    loadBalancer:
      localityLbSetting:
        enabled: true
        distribute:
        - from: us-east-1/*
          to:
            "us-east-1/*": 80  # 80% traffic stays local
            "eu-west-1/*": 15   # 15% to EU
            "ap-southeast-1/*": 5  # 5% to APAC
        - from: eu-west-1/*
          to:
            "eu-west-1/*": 80
            "us-east-1/*": 15
            "ap-southeast-1/*": 5
        - from: ap-southeast-1/*
          to:
            "ap-southeast-1/*": 80
            "us-east-1/*": 10
            "eu-west-1/*": 10
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
\`\`\`

---

**3. Consul Multi-Region Setup**

**Consul Configuration (us-east-1 - Primary)**:

\`\`\`hcl
# us-east-1/consul-config.hcl
datacenter = "us-east-1"
primary_datacenter = "us-east-1"
server = true
bootstrap_expect = 3

# WAN federation
retry_join_wan = [
  "consul-server-1.eu-west-1.internal",
  "consul-server-1.ap-southeast-1.internal"
]

# Enable mesh gateway for cross-region traffic
connect {
  enabled = true
  enable_mesh_gateway_wan_federation = true
}

# Service mesh gateway
ports {
  grpc = 8502
  http = 8500
  https = 8501
}
\`\`\`

**Consul Configuration (eu-west-1 - Secondary)**:

\`\`\`hcl
# eu-west-1/consul-config.hcl
datacenter = "eu-west-1"
primary_datacenter = "us-east-1"  # Point to primary
server = true
bootstrap_expect = 3

retry_join_wan = [
  "consul-server-1.us-east-1.internal"
]

connect {
  enabled = true
  enable_mesh_gateway_wan_federation = true
}
\`\`\`

**Service Registration**:

\`\`\`javascript
const Consul = require('consul');

const consul = new Consul({
  host: 'consul.us-east-1.internal',
  port: 8500
});

// Register service
await consul.agent.service.register({
  name: 'order-service',
  id: 'order-service-pod-1',
  address: '10.0.1.5',
  port: 8080,
  tags: ['v1', 'http'],
  meta: {
    region: 'us-east-1',
    zone: 'us-east-1a',
    version: 'v1.2.3'
  },
  check: {
    http: 'http://10.0.1.5:8080/health',
    interval: '10s',
    timeout: '5s',
    deregister_critical_service_after: '30s'
  }
});

// Deregister on shutdown
process.on('SIGTERM', async () => {
  await consul.agent.service.deregister('order-service-pod-1');
  process.exit(0);
});
\`\`\`

**Cross-Region Service Discovery**:

\`\`\`javascript
// Discover services in all regions
async function discoverService(serviceName) {
  // Query local datacenter first
  const localServices = await consul.health.service({
    service: serviceName,
    passing: true  // Only healthy instances
  });
  
  // Query other datacenters
  const euServices = await consul.health.service({
    service: serviceName,
    dc: 'eu-west-1',
    passing: true
  });
  
  const apacServices = await consul.health.service({
    service: serviceName,
    dc: 'ap-southeast-1',
    passing: true
  });
  
  return {
    local: localServices,
    eu: euServices,
    apac: apacServices
  };
}

// Usage
const services = await discoverService('order-service');
console.log(\`Found \${services.local.length} local instances\`);
console.log(\`Found \${services.eu.length} EU instances\`);
console.log(\`Found \${services.apac.length} APAC instances\`);
\`\`\`

---

**4. Latency-Based Routing**

**AWS Route53 Configuration**:

\`\`\`javascript
const AWS = require('aws-sdk');
const route53 = new AWS.Route53();

// Create latency-based routing records
await route53.changeResourceRecordSets({
  HostedZoneId: 'Z123456789',
  ChangeBatch: {
    Changes: [
      {
        Action: 'CREATE',
        ResourceRecordSet: {
          Name: 'api.example.com',
          Type: 'A',
          SetIdentifier: 'us-east-1',
          Region: 'us-east-1',
          TTL: 60,
          ResourceRecords: [
            { Value: '54.123.45.67' }  // NLB in us-east-1
          ],
          HealthCheckId: 'health-check-us-east-1'
        }
      },
      {
        Action: 'CREATE',
        ResourceRecordSet: {
          Name: 'api.example.com',
          Type: 'A',
          SetIdentifier: 'eu-west-1',
          Region: 'eu-west-1',
          TTL: 60,
          ResourceRecords: [
            { Value: '52.234.56.78' }  // NLB in eu-west-1
          ],
          HealthCheckId: 'health-check-eu-west-1'
        }
      },
      {
        Action: 'CREATE',
        ResourceRecordSet: {
          Name: 'api.example.com',
          Type: 'A',
          SetIdentifier: 'ap-southeast-1',
          Region: 'ap-southeast-1',
          TTL: 60,
          ResourceRecords: [
            { Value: '13.345.67.89' }  // NLB in ap-southeast-1
          ],
          HealthCheckId: 'health-check-ap-southeast-1'
        }
      }
    ]
  }
}).promise();
\`\`\`

**Health Check Configuration**:

\`\`\`javascript
// Create health check
await route53.createHealthCheck({
  HealthCheckConfig: {
    Type: 'HTTPS',
    ResourcePath: '/health',
    FullyQualifiedDomainName: 'api-us-east-1.example.com',
    Port: 443,
    RequestInterval: 30,  // Check every 30 seconds
    FailureThreshold: 3,  // Fail after 3 consecutive failures
    MeasureLatency: true  // Track latency
  }
}).promise();
\`\`\`

**Client-Side Latency Detection**:

\`\`\`javascript
class RegionSelector {
  constructor() {
    this.regions = [
      { name: 'us-east-1', endpoint: 'https://api-us-east-1.example.com' },
      { name: 'eu-west-1', endpoint: 'https://api-eu-west-1.example.com' },
      { name: 'ap-southeast-1', endpoint: 'https://api-ap-southeast-1.example.com' }
    ];
    this.latencies = {};
  }
  
  async measureLatency(region) {
    const start = Date.now();
    
    try {
      await fetch(\`\${region.endpoint}/health\`, {
        method: 'HEAD',
        timeout: 2000
      });
      
      const latency = Date.now() - start;
      this.latencies[region.name] = latency;
      return latency;
    } catch (error) {
      this.latencies[region.name] = Infinity;
      return Infinity;
    }
  }
  
  async selectFastestRegion() {
    // Measure latency to all regions in parallel
    await Promise.all(
      this.regions.map(region => this.measureLatency(region))
    );
    
    // Select region with lowest latency
    const fastest = this.regions.reduce((best, region) => {
      return this.latencies[region.name] < this.latencies[best.name] ? region : best;
    });
    
    console.log('Latencies:', this.latencies);
    console.log('Selected region:', fastest.name);
    
    return fastest.endpoint;
  }
}

// Usage
const selector = new RegionSelector();
const endpoint = await selector.selectFastestRegion();

// Make request to fastest region
const response = await fetch(\`\${endpoint}/api/orders\`);
\`\`\`

---

**5. Regional Failover**

**Automatic Failover with Health Checks**:

\`\`\`javascript
class FailoverClient {
  constructor() {
    this.regions = [
      { name: 'us-east-1', endpoint: 'https://api-us-east-1.example.com', priority: 1 },
      { name: 'eu-west-1', endpoint: 'https://api-eu-west-1.example.com', priority: 2 },
      { name: 'ap-southeast-1', endpoint: 'https://api-ap-southeast-1.example.com', priority: 3 }
    ];
    this.currentRegion = this.regions[0];
  }
  
  async makeRequest(path, options = {}) {
    const maxRetries = this.regions.length;
    
    for (let i = 0; i < maxRetries; i++) {
      try {
        const response = await fetch(\`\${this.currentRegion.endpoint}\${path}\`, {
          ...options,
          timeout: 5000
        });
        
        if (response.ok) {
          return response;
        }
        
        // Server error - try next region
        if (response.status >= 500) {
          console.error(\`Region \${this.currentRegion.name} returned \${response.status}\`);
          this.failoverToNextRegion();
          continue;
        }
        
        return response;
        
      } catch (error) {
        console.error(\`Request to \${this.currentRegion.name} failed:\`, error.message);
        
        // Network error - try next region
        if (i < maxRetries - 1) {
          this.failoverToNextRegion();
        } else {
          throw new Error('All regions unavailable');
        }
      }
    }
  }
  
  failoverToNextRegion() {
    const currentIndex = this.regions.indexOf(this.currentRegion);
    this.currentRegion = this.regions[(currentIndex + 1) % this.regions.length];
    console.log(\`Failed over to \${this.currentRegion.name}\`);
  }
}

// Usage
const client = new FailoverClient();

try {
  const response = await client.makeRequest('/api/orders', {
    method: 'POST',
    body: JSON.stringify({ items: [...] })
  });
  console.log('Order created:', await response.json());
} catch (error) {
  console.error('All regions failed:', error);
}
\`\`\`

---

**6. Data Consistency Across Regions**

**Active-Active with Eventual Consistency**:

\`\`\`javascript
// Write to local region, replicate asynchronously
async function createOrder(order) {
  // Write to local database
  await db.orders.insert(order);
  
  // Publish event to Kafka (cross-region replication)
  await kafka.produce('orders.created', {
    orderId: order.id,
    region: 'us-east-1',
    timestamp: Date.now(),
    data: order
  });
  
  return order;
}

// Subscribe to events from other regions
kafka.subscribe('orders.created', async (event) => {
  if (event.region !== 'us-east-1') {
    // Replicate from other region
    await db.orders.upsert(event.data);
  }
});
\`\`\`

**Active-Passive with Synchronous Replication**:

\`\`\`javascript
// Write to primary region, synchronously replicate to secondary
async function createOrder(order) {
  const primary = 'us-east-1';
  const secondary = 'eu-west-1';
  
  // Write to primary
  const result = await db.primary.orders.insert(order);
  
  // Synchronous replication to secondary
  try {
    await db.secondary.orders.insert(order);
  } catch (error) {
    console.error('Secondary replication failed:', error);
    // Log for async retry, but don't fail request
  }
  
  return result;
}
\`\`\`

---

**Key Takeaways**:

1. **Multi-cluster Kubernetes**: Use Istio for cross-region service mesh
2. **Consul WAN federation**: Global service registry with datacenter awareness
3. **Latency-based routing**: Route53 or client-side latency measurement
4. **Regional failover**: Automatic with health checks + retry logic
5. **Data consistency**: Active-active (eventual) or active-passive (sync)
6. **Health checks**: Per-region health checks with automatic DNS failover
7. **Locality-aware load balancing**: Prefer local region, failover to others`,
    keyPoints: [
      'Multi-region Istio mesh: ServiceEntry and DestinationRule for cross-region discovery',
      'Consul WAN federation: Primary datacenter in us-east-1, secondaries replicate',
      'Latency-based routing: Route53 latency-based records or client-side latency measurement',
      'Regional failover: Health checks remove failed regions from DNS, client retries other regions',
      'Data consistency: Active-active with Kafka replication (eventual) or active-passive (sync)',
      'Locality-aware LB: 80% traffic stays local, 20% distributed to other regions',
    ],
  },
];
