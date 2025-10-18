/**
 * Service Mesh Section
 */

export const servicemeshSection = {
  id: 'service-mesh',
  title: 'Service Mesh',
  content: `A service mesh is an infrastructure layer that handles service-to-service communication via sidecar proxies. It moves cross-cutting concerns (observability, security, traffic management) out of application code into the infrastructure.

## The Problem: Cross-Cutting Concerns at Scale

**Without Service Mesh**:

Each service must implement:
- Service discovery
- Load balancing  
- Circuit breakers
- Retries and timeouts
- Metrics and tracing
- mTLS encryption
- Rate limiting

**Problems**:
❌ Logic duplicated across services
❌ Different implementations in different languages (Java, Node, Go)
❌ Hard to enforce policies consistently
❌ Code changes needed for infrastructure concerns

**With Service Mesh**:

All communication routed through sidecar proxies that provide:
✅ Service discovery
✅ Load balancing
✅ Circuit breakers
✅ Retries
✅ Metrics
✅ mTLS
✅ Rate limiting

Without code changes!

---

## Architecture

### Sidecar Proxy Pattern

**Each service instance has a sidecar proxy**:

\`\`\`
┌─────────────────┐
│  Order Service  │
│      :8080      │
└────────┬────────┘
         │ localhost
    ┌────▼──────┐
    │  Envoy    │  (Sidecar Proxy)
    │  Proxy    │
    └─────┬─────┘
          │ Network
          ▼
    ┌─────────────┐
    │  Envoy      │  (Sidecar Proxy)
    │  Proxy      │
    └─────┬───────┘
          │ localhost
    ┌─────▼────────┐
    │Payment Service│
    │     :8080     │
    └───────────────┘
\`\`\`

**Service calls localhost proxy, proxy handles everything**.

### Two Planes

**1. Data Plane**: Sidecar proxies (Envoy)
- Handle all network traffic
- Enforce policies
- Collect metrics

**2. Control Plane**: Management layer (Istio, Linkerd)
- Configure proxies
- Collect telemetry
- Provide APIs

\`\`\`
                  ┌──────────────────┐
                  │  Control Plane   │
                  │  (Istio/Linkerd) │
                  └────────┬─────────┘
                           │ Configuration
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼────┐ ┌────▼─────┐ ┌───▼──────┐
        │  Proxy   │ │  Proxy   │ │  Proxy   │  (Data Plane)
        └─────┬────┘ └────┬─────┘ └───┬──────┘
              │           │           │
        ┌─────▼────┐ ┌───▼──────┐ ┌──▼───────┐
        │ Service  │ │ Service  │ │ Service  │
        │    A     │ │    B     │ │    C     │
        └──────────┘ └──────────┘ └──────────┘
\`\`\`

---

## Popular Service Meshes

### 1. Istio

**Most popular**, feature-rich service mesh.

**Components**:
- **Envoy**: Sidecar proxy (data plane)
- **Pilot**: Service discovery and configuration
- **Citadel**: Certificate management (mTLS)
- **Galley**: Configuration validation
- **Mixer** (deprecated): Telemetry and policy

**Installation**:
\`\`\`bash
# Install Istio
curl -L https://istio.io/downloadIstio | sh -
cd istio-*
export PATH=$PWD/bin:$PATH

# Install on Kubernetes
istioctl install --set profile=demo

# Enable sidecar injection for namespace
kubectl label namespace default istio-injection=enabled
\`\`\`

**Deploy service**:
\`\`\`yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
spec:
  replicas: 2
  template:
    metadata:
      labels:
        app: order-service
    spec:
      containers:
      - name: order-service
        image: order-service:v1
        ports:
        - containerPort: 8080
\`\`\`

**Istio automatically injects sidecar proxy!**

After deployment:
\`\`\`bash
# Two containers: app + envoy proxy
kubectl get pods
NAME                             READY   STATUS
order-service-abc123             2/2     Running
\`\`\`

### 2. Linkerd

**Lightweight, fast, simpler than Istio**.

**Pros**:
✅ Lower resource usage
✅ Easier to learn
✅ Faster data plane
✅ Written in Rust

**Installation**:
\`\`\`bash
# Install Linkerd CLI
curl --proto '=https' --tlsv1.2 -sSfL https://run.linkerd.io/install | sh

# Install on Kubernetes
linkerd install | kubectl apply -f -

# Inject sidecar into deployment
kubectl get deploy order-service -o yaml | linkerd inject - | kubectl apply -f -
\`\`\`

### 3. Consul Connect

**By HashiCorp**, integrates with Consul service registry.

**Pros**:
✅ Works outside Kubernetes (VMs, bare metal)
✅ Built-in service discovery
✅ Multi-datacenter support

### 4. AWS App Mesh

**Managed** service mesh for AWS.

**Pros**:
✅ Integrated with AWS services
✅ No control plane to manage
✅ Works with ECS, EKS, EC2

---

## Key Features

### 1. Automatic mTLS

**Problem**: Services communicate over plain HTTP

**Solution**: Service mesh auto-encrypts all traffic

**Istio example**:
\`\`\`yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: default
spec:
  mtls:
    mode: STRICT  # Require mTLS for all services
\`\`\`

**What happens**:
1. Istio injects certificates into each proxy
2. Proxies establish mTLS automatically
3. Services talk to localhost (proxy handles encryption)
4. Certificates rotate automatically

**No code changes needed!**

### 2. Traffic Management

#### Canary Deployments

**Gradually** shift traffic to new version.

\`\`\`yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: order-service
spec:
  hosts:
  - order-service
  http:
  - match:
    - headers:
        user-type:
          exact: beta-tester
    route:
    - destination:
        host: order-service
        subset: v2
  - route:
    - destination:
        host: order-service
        subset: v1
      weight: 90
    - destination:
        host: order-service
        subset: v2
      weight: 10  # 10% traffic to v2
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: order-service
spec:
  host: order-service
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
\`\`\`

**Result**: 90% traffic to v1, 10% to v2, beta-testers always get v2.

#### Circuit Breakers

\`\`\`yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: payment-service
spec:
  host: payment-service
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        maxRequestsPerConnection: 2
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
\`\`\`

**Proxy automatically**:
- Limits connections
- Ejects unhealthy instances
- No code changes!

#### Retries

\`\`\`yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: order-service
spec:
  hosts:
  - order-service
  http:
  - route:
    - destination:
        host: order-service
    retries:
      attempts: 3
      perTryTimeout: 2s
      retryOn: 5xx,reset,connect-failure
\`\`\`

### 3. Observability

#### Distributed Tracing

**Automatic** trace propagation.

**Without service mesh**:
\`\`\`javascript
// Must manually propagate trace headers
const response = await axios.post('http://payment-service/charge', data, {
    headers: {
        'x-request-id': req.headers['x-request-id'],
        'x-b3-traceid': req.headers['x-b3-traceid'],
        'x-b3-spanid': req.headers['x-b3-spanid'],
        // ... more headers
    }
});
\`\`\`

**With service mesh**:
\`\`\`javascript
// Just make the call - proxy handles tracing!
const response = await axios.post('http://payment-service/charge', data);
\`\`\`

**View in Jaeger**:
\`\`\`
Request: GET /orders/123
├─ order-service: 245ms
│  ├─ payment-service: 102ms
│  │  └─ fraud-service: 45ms
│  └─ inventory-service: 87ms
└─ shipping-service: 156ms

Total: 401ms
\`\`\`

#### Metrics

**Automatic** RED metrics (Rate, Errors, Duration).

**Prometheus metrics exported**:
\`\`\`
istio_requests_total{source_app="order-service",destination_app="payment-service",response_code="200"} 1247
istio_request_duration_milliseconds{source_app="order-service",destination_app="payment-service",quantile="0.95"} 102.5
\`\`\`

**Grafana dashboards** show:
- Request rates per service
- Error rates
- Latency (p50, p95, p99)
- Success rates

#### Traffic Visualization

**Kiali** dashboard shows service topology:
\`\`\`
         ┌──────┐
         │ User │
         └───┬──┘
             │
        ┌────▼─────┐
        │   API    │
        │ Gateway  │
        └────┬─────┘
             │
    ┌────────┼────────┐
    │        │        │
┌───▼───┐ ┌──▼──┐ ┌──▼──────┐
│ Order │ │User │ │ Product │
└───┬───┘ └─────┘ └─────────┘
    │
┌───┼─────────┐
│   │         │
▼   ▼         ▼
Payment   Inventory  Shipping
\`\`\`

With **live metrics** on each edge!

### 4. Security

#### Authorization Policies

**Control** which services can talk to which.

\`\`\`yaml
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: payment-service-policy
spec:
  selector:
    matchLabels:
      app: payment-service
  action: ALLOW
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/default/sa/order-service"]
    to:
    - operation:
        methods: ["POST"]
        paths: ["/charge"]
\`\`\`

**Result**: Only Order Service can call Payment Service's /charge endpoint.

#### Rate Limiting

\`\`\`yaml
apiVersion: networking.istio.io/v1beta1
kind: EnvoyFilter
metadata:
  name: ratelimit
spec:
  workloadSelector:
    labels:
      app: api-gateway
  configPatches:
  - applyTo: HTTP_FILTER
    match:
      context: SIDECAR_INBOUND
    patch:
      operation: INSERT_BEFORE
      value:
        name: envoy.filters.http.local_ratelimit
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.http.local_ratelimit.v3.LocalRateLimit
          stat_prefix: http_local_rate_limiter
          token_bucket:
            max_tokens: 100
            tokens_per_fill: 100
            fill_interval: 60s
\`\`\`

---

## When to Use Service Mesh

**Use service mesh when**:
✅ 15+ microservices (complexity justifies overhead)
✅ Polyglot environment (Java, Node, Go, Python)
✅ Strict security requirements (mTLS, authorization)
✅ Need advanced traffic management (canary, A/B testing)
✅ Running on Kubernetes

**Skip service mesh when**:
❌ < 5 microservices (not worth complexity)
❌ Team lacks operational maturity
❌ Simple communication patterns
❌ Performance-critical (service mesh adds latency)

---

## Performance Overhead

**Service mesh adds latency** (each request goes through 2 proxies).

**Typical overhead**:
- **Linkerd**: +1-2ms per hop
- **Istio**: +2-5ms per hop

**Example**:
\`\`\`
Without mesh:
Order Service → Payment Service: 50ms

With mesh:
Order Service → Proxy → Proxy → Payment Service: 54ms
\`\`\`

**Is it worth it?**
- For most services: Yes (4ms is negligible vs benefits)
- For ultra-low-latency: Maybe not (HFT, gaming)

---

## Best Practices

### 1. Start Small

Don't enable mesh for all services at once.

**Gradual rollout**:
1. Start with non-critical services
2. Monitor metrics and performance
3. Gradually expand
4. Enable for critical services last

### 2. Use Proper Resource Limits

Proxies consume resources.

\`\`\`yaml
apiVersion: v1
kind: Pod
metadata:
  name: order-service
spec:
  containers:
  - name: order-service
    resources:
      requests:
        cpu: 500m
        memory: 512Mi
  - name: istio-proxy
    resources:
      requests:
        cpu: 100m  # Proxy resources
        memory: 128Mi
\`\`\`

### 3. Monitor Control Plane

Control plane outage = can't change config (but data plane continues working).

**Monitor**:
- Control plane CPU/memory
- API response times
- Configuration sync lag

### 4. Secure the Mesh

**Don't** expose mesh components to internet.

**Do**:
- Use NetworkPolicies to isolate control plane
- Rotate certificates regularly
- Enable RBAC for mesh operations

---

## Alternatives to Service Mesh

### 1. Library-Based Approach

**Use libraries** like Netflix OSS (Hystrix, Ribbon, Eureka).

**Pros**: No infrastructure complexity
**Cons**: Language-specific, code changes needed

### 2. API Gateway

**Centralize** cross-cutting concerns in API Gateway.

**Pros**: Simpler than mesh
**Cons**: Single point of failure, doesn't handle east-west traffic

### 3. Build Your Own

**Custom** middleware/interceptors.

**Pros**: Full control
**Cons**: Lots of work, reinventing wheel

---

## Interview Tips

**Red Flags**:
❌ Saying "always use service mesh"
❌ Not mentioning overhead/complexity
❌ Confusing service mesh with API gateway

**Good Responses**:
✅ Explain sidecar proxy pattern
✅ Discuss data plane vs control plane
✅ Mention specific tools (Istio, Linkerd)
✅ Talk about trade-offs (benefits vs overhead)

**Sample Answer**:
*"I'd use a service mesh when we have 15+ microservices and need consistent observability, security, and traffic management across a polyglot environment. Istio or Linkerd would inject sidecar proxies that handle mTLS, circuit breaking, retries, and metrics without code changes. The proxies form the data plane, while the control plane configures them. Trade-offs: adds 2-5ms latency per hop and operational complexity, but provides powerful features that would be hard to implement consistently across services. For fewer than 10 services, I'd start with simpler solutions like libraries or API gateway."*

---

## Key Takeaways

1. **Service mesh**: Infrastructure layer for service-to-service communication
2. **Sidecar proxies**: Handle traffic, enforce policies, collect metrics
3. **Data plane**: Proxies (Envoy). Control plane: Management (Istio/Linkerd)
4. **Benefits**: mTLS, circuit breakers, retries, tracing, rate limiting without code changes
5. **Use when**: 15+ services, polyglot, strict security, Kubernetes
6. **Skip when**: < 5 services, simple patterns, performance-critical
7. **Overhead**: 2-5ms latency, operational complexity
8. **Start small**: Gradual rollout, monitor carefully`,
};
