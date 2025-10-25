/**
 * Service Discovery & Registry Section
 */

export const servicediscoverySection = {
  id: 'service-discovery',
  title: 'Service Discovery & Registry',
  content: `In a microservices architecture, services need to find and communicate with each other dynamically. Service discovery solves the problem of locating service instances in a constantly changing environment.

## The Problem

In a monolith, components call each other via function calls. In microservices, services are distributed across multiple machines with dynamic IP addresses and ports.

**Challenges**:
- Service instances have dynamic IPs (containers, autoscaling)
- Number of instances changes (scale up/down)
- Instances fail and get replaced
- Services move between hosts
- Manual configuration doesn't scale

**Example**:
\`\`\`
Order Service needs to call Payment Service
❌ Hardcoded: http://10.0.1.5:8080/payment
   Problem: What if Payment Service moves? Scales to 10 instances?
   
✅ Service Discovery: http://payment-service/payment
   Service registry resolves to healthy instance automatically
\`\`\`

---

## Service Registry

A **service registry** is a database of available service instances and their locations.

**Core Operations**:
1. **Register**: Service instance registers itself on startup
2. **Deregister**: Service unregisters on shutdown (or detected as down)
3. **Discover**: Clients query registry to find service instances
4. **Health Check**: Registry monitors instance health

**Example: Service Registry Data**
\`\`\`json
{
  "payment-service": [
    {
      "instanceId": "payment-1",
      "host": "10.0.1.5",
      "port": 8080,
      "status": "UP",
      "metadata": {
        "version": "2.1.0",
        "zone": "us-east-1a"
      }
    },
    {
      "instanceId": "payment-2",
      "host": "10.0.1.8",
      "port": 8080,
      "status": "UP",
      "metadata": {
        "version": "2.1.0",
        "zone": "us-east-1b"
      }
    }
  ],
  "order-service": [...]
}
\`\`\`

---

## Client-Side Discovery

**Pattern**: Client queries service registry directly and chooses an instance.

**Flow**:
\`\`\`
1. Order Service queries registry: "Where is payment-service?"
2. Registry returns: [10.0.1.5:8080, 10.0.1.8:8080]
3. Order Service chooses instance (load balancing logic)
4. Order Service calls chosen instance directly
\`\`\`

**Advantages**:
✅ Simple architecture (no intermediary)
✅ Client controls load balancing algorithm
✅ Low latency (direct communication)

**Disadvantages**:
❌ Client must implement discovery logic
❌ Tight coupling to registry
❌ Logic duplicated across clients (every language)

**Implementation**: Netflix Eureka

**Example Code**:
\`\`\`java
// Spring Cloud with Eureka
@Autowired
private DiscoveryClient discoveryClient;

public String callPaymentService() {
    // Get instances from registry
    List<ServiceInstance> instances = 
        discoveryClient.getInstances("payment-service");
    
    // Choose instance (round robin, random, etc.)
    ServiceInstance instance = chooseInstance (instances);
    
    // Make HTTP call
    String url = instance.getUri() + "/charge";
    return restTemplate.postForObject (url, payment, String.class);
}
\`\`\`

---

## Server-Side Discovery

**Pattern**: Client sends request to load balancer, which queries registry and routes request.

**Flow**:
\`\`\`
1. Order Service calls load balancer: http://payment-service-lb/charge
2. Load balancer queries registry
3. Load balancer chooses healthy instance
4. Load balancer forwards request to instance
5. Response returns through load balancer
\`\`\`

**Advantages**:
✅ Simple clients (just call load balancer)
✅ Discovery logic centralized
✅ No client-side dependencies

**Disadvantages**:
❌ Load balancer is single point of failure
❌ Extra network hop (adds latency)
❌ Load balancer must scale

**Implementation**: AWS ELB with service registry, Kubernetes Service

**Example: Kubernetes**:
\`\`\`yaml
# Kubernetes Service (built-in service discovery)
apiVersion: v1
kind: Service
metadata:
  name: payment-service
spec:
  selector:
    app: payment
  ports:
  - port: 80
    targetPort: 8080
\`\`\`

Clients simply call \`http://payment-service\`. Kubernetes DNS resolves it and load balances automatically.

---

## DNS-Based Discovery

**Pattern**: Use DNS to resolve service names to IP addresses.

**How it works**:
\`\`\`
1. Service registers with DNS server
2. Client performs DNS lookup: payment-service.namespace.svc.cluster.local
3. DNS returns IP(s)
4. Client connects directly
\`\`\`

**Advantages**:
✅ Universal (all languages support DNS)
✅ Simple integration
✅ No special libraries needed

**Disadvantages**:
❌ DNS caching issues (TTL)
❌ Limited load balancing options
❌ No health checks (returns dead instances)

**Used by**: Kubernetes (CoreDNS), Consul

---

## Popular Service Discovery Tools

### 1. **Netflix Eureka**

**Type**: Client-side discovery

**Features**:
- Self-registration
- Heartbeat-based health checks
- Client-side load balancing (Ribbon)
- Zone awareness
- Peer-to-peer replication (no single point of failure)

**Example**:
\`\`\`java
// Service registers itself
@EnableEurekaClient
@SpringBootApplication
public class PaymentServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(PaymentServiceApplication.class, args);
    }
}
\`\`\`

**Used by**: Netflix (obviously), many Spring Cloud users

### 2. **Consul (by HashiCorp)**

**Type**: Both client-side and server-side

**Features**:
- Service registry
- Health checks (HTTP, TCP, script-based)
- DNS interface
- Key-value store
- Multi-datacenter support
- Service mesh capabilities

**Example**:
\`\`\`bash
# Register service
curl -X PUT -d '{
  "ID": "payment-1",
  "Name": "payment-service",
  "Address": "10.0.1.5",
  "Port": 8080,
  "Check": {
    "HTTP": "http://10.0.1.5:8080/health",
    "Interval": "10s"
  }
}' http://consul-server:8500/v1/agent/service/register

# Discover service via DNS
dig @consul-server payment-service.service.consul
\`\`\`

### 3. **etcd**

**Type**: Key-value store used for discovery

**Features**:
- Strongly consistent (Raft consensus)
- Watch mechanism (real-time updates)
- TTL for automatic expiration
- Used by Kubernetes

**Example**:
\`\`\`bash
# Register service
etcdctl put /services/payment-service/instance-1 \
  '{"host":"10.0.1.5","port":8080}'

# Discover
etcdctl get --prefix /services/payment-service
\`\`\`

### 4. **Kubernetes Service Discovery**

**Type**: Server-side + DNS

**Features**:
- Built-in (no external tool needed)
- DNS-based
- Service objects abstract pods
- Automatic load balancing

**Automatic**: When you create pods with labels, Kubernetes Service automatically discovers and load balances.

---

## Health Checks

Service registry must know which instances are healthy.

### Types of Health Checks:

**1. Heartbeat**
- Service sends periodic heartbeat to registry
- If heartbeat stops, instance marked as down
- Example: Eureka (30-second interval)

\`\`\`java
// Spring Boot health endpoint
@RestController
public class HealthController {
    @GetMapping("/health")
    public String health() {
        // Check database, dependencies, etc.
        return "OK";
    }
}
\`\`\`

**2. Polling**
- Registry actively polls service health endpoint
- Example: Consul HTTP checks

\`\`\`json
{
  "check": {
    "http": "http://10.0.1.5:8080/health",
    "interval": "10s",
    "timeout": "2s"
  }
}
\`\`\`

**3. Active Health Checks**
- Load balancer sends test requests
- Removes instance if fails N consecutive checks
- Example: NGINX health checks

**Health Check Best Practices**:
- Check critical dependencies (database, downstream services)
- But don't fail if non-critical dependency down
- Keep checks lightweight (< 100ms)
- Return detailed status for debugging

---

## Load Balancing Algorithms

When multiple instances are available, how to choose?

**1. Round Robin**
- Rotate through instances sequentially
- Simple, fair distribution
- Doesn't account for instance load

**2. Random**
- Pick random instance
- Simple, good enough for most cases

**3. Least Connections**
- Route to instance with fewest active connections
- Better for long-lived connections

**4. Weighted Round Robin**
- Assign weights to instances
- More powerful instances get more traffic

**5. Zone-Aware**
- Prefer instances in same availability zone
- Reduces latency and cross-AZ data transfer costs

**Example: Ribbon (Netflix)**:
\`\`\`java
// Configure load balancing rule
@Bean
public IRule ribbonRule() {
    return new ZoneAvoidanceRule(); // Zone-aware
}
\`\`\`

---

## Service Registration Patterns

### Self-Registration

Service instance registers itself with registry on startup.

**Flow**:
\`\`\`
1. Service starts
2. Service calls registry API: "I'm payment-service at 10.0.1.5:8080"
3. Service sends periodic heartbeats
4. Service unregisters on shutdown
\`\`\`

**Advantages**: Simple, service controls its registration

**Disadvantages**: Tight coupling to registry, must implement in every service

### Third-Party Registration

External process registers services.

**Flow**:
\`\`\`
1. Service starts
2. Service registrar (sidecar or orchestrator) detects new service
3. Registrar registers service in registry
4. Registrar monitors health and updates registry
\`\`\`

**Example**: Kubernetes automatically registers pods as services.

**Advantages**: Services don't know about registry, works with legacy apps

**Disadvantages**: Extra component to manage

---

## Real-World Example: Netflix

**Netflix Architecture**:
- **Eureka**: Service registry
- **Ribbon**: Client-side load balancing
- **Hystrix**: Circuit breaker (integrated with Eureka)

**Flow**:
\`\`\`
1. API Service needs to call Recommendation Service
2. Ribbon queries Eureka: "Where is recommendation-service?"
3. Eureka returns healthy instances
4. Ribbon chooses instance using zone-aware algorithm
5. Hystrix wraps call with circuit breaker
6. Request sent to chosen instance
\`\`\`

**Why it works at Netflix scale**:
- 700+ microservices
- 1000s of instances
- Constant deployments
- Instances come and go frequently
- Automatic discovery handles all this

---

## Decision Framework

**Use Client-Side Discovery When**:
✅ Need control over load balancing
✅ Performance critical (avoid extra hop)
✅ Using Netflix OSS stack

**Use Server-Side Discovery When**:
✅ Want simple clients
✅ Polyglot environment (many languages)
✅ Using Kubernetes or cloud load balancers

**Use DNS-Based Discovery When**:
✅ Simple requirements
✅ No special libraries wanted
✅ Using Kubernetes (built-in)

---

## Interview Tips

**Red Flags**:
❌ Hardcoded service URLs
❌ Not mentioning health checks
❌ Ignoring failure scenarios

**Good Responses**:
✅ Explain client-side vs server-side discovery
✅ Mention specific tools (Eureka, Consul, Kubernetes)
✅ Discuss health checks and load balancing
✅ Consider trade-offs

**Sample Answer**:
*"I'd use service discovery to allow services to find each other dynamically. For a Kubernetes-based deployment, I'd leverage built-in service discovery via Kubernetes Services, which provides DNS-based discovery and automatic load balancing. Each service registers automatically when pods are created. For health checks, I'd implement /health endpoints that verify critical dependencies. If not using Kubernetes, I'd use Consul for its robust health checking and multi-datacenter support."*

---

## Key Takeaways

1. **Service discovery** enables dynamic service communication
2. **Client-side discovery**: Client queries registry directly (Eureka)
3. **Server-side discovery**: Load balancer queries registry (Kubernetes, ELB)
4. **DNS-based**: Simple, universal, but limited (Kubernetes DNS)
5. **Health checks**: Critical for removing unhealthy instances
6. **Load balancing**: Choose algorithm based on requirements
7. **Self-registration vs third-party**: Trade-off between control and simplicity
8. **Don't hardcode IPs**: Use service names that resolve dynamically`,
};
