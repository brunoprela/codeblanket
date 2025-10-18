/**
 * Vertical vs Horizontal Scaling Section
 */

export const verticalvshorizontalscalingSection = {
  id: 'vertical-vs-horizontal-scaling',
  title: 'Vertical vs Horizontal Scaling',
  content: `Scaling is inevitable as your system grows. The choice between vertical and horizontal scaling fundamentally impacts your architecture, costs, and operational complexity.

## Definitions

**Vertical Scaling** (Scale Up):
- Add more resources to a **single machine**
- Increase CPU, RAM, disk, network bandwidth
- Examples: Upgrade from 8GB to 64GB RAM, 4 cores to 32 cores

**Horizontal Scaling** (Scale Out):
- Add more **machines** to your pool of resources
- Distribute load across multiple servers
- Examples: Add 10 more web servers, expand from 3 to 20 database replicas

---

## Vertical Scaling in Detail

### How It Works

**Before**:
\`\`\`
Single Server: 8GB RAM, 4 CPU cores → Handling 1,000 req/s
\`\`\`

**After Vertical Scaling**:
\`\`\`
Same Server: 64GB RAM, 32 CPU cores → Handling 8,000 req/s
\`\`\`

### Advantages

✅ **Simplicity**: No code changes, no distributed system complexity
✅ **No data consistency issues**: Single database, no replication/sharding
✅ **Lower latency**: No network calls between servers
✅ **Easier debugging**: Logs, profiling on single machine
✅ **Licensing**: Some software licensed per server (cheaper with fewer servers)

### Disadvantages

❌ **Hard limits**: CPUs, RAM have physical limits (can't scale infinitely)
❌ **Downtime**: Must stop server to upgrade hardware
❌ **Cost inefficiency**: High-end hardware exponentially expensive
❌ **Single point of failure**: If server dies, entire system down
❌ **Risk**: All eggs in one basket

### Real-World Example: Database Vertical Scaling

**Scenario**: PostgreSQL database struggling with load

**Vertical scaling approach**:
1. Current: 16GB RAM, 8 cores → Query time: 500ms
2. Upgrade to 128GB RAM, 64 cores
3. Cost: $500/month → $5,000/month
4. Query time improves: 500ms → 100ms
5. Works until you hit limits again

**When this works**:
- Database read:write ratio is 1:1 (can't easily read scale)
- Queries are CPU/memory intensive
- Don't need high availability (can tolerate brief downtime for upgrades)

---

## Horizontal Scaling in Detail

### How It Works

**Before**:
\`\`\`
1 Web Server → Handling 1,000 req/s → Max capacity
\`\`\`

**After Horizontal Scaling**:
\`\`\`
10 Web Servers → Each handling 100 req/s → Total 1,000 req/s
Load Balancer distributes traffic across 10 servers
\`\`\`

**Key**: Each server is identical, load balancer distributes requests

### Advantages

✅ **Nearly unlimited scale**: Add as many servers as needed
✅ **High availability**: If one server fails, others continue serving
✅ **No downtime**: Add/remove servers without stopping system
✅ **Cost efficiency**: Use commodity hardware, cheaper than high-end single server
✅ **Flexibility**: Scale up during traffic spikes, scale down during low traffic

### Disadvantages

❌ **Complexity**: Distributed system challenges (consistency, network latency)
❌ **Code changes**: Application must be stateless or use shared state (Redis, database)
❌ **Data consistency**: Replication lag, eventual consistency, cache invalidation
❌ **Debugging**: Logs spread across multiple servers, need centralized logging
❌ **Network overhead**: Cross-server communication adds latency

### Real-World Example: Web Server Horizontal Scaling

**Scenario**: E-commerce site with traffic spikes

**Horizontal scaling approach**:
\`\`\`
Normal traffic: 3 servers, 1,000 req/s each = 3,000 req/s total
Black Friday: Auto-scale to 30 servers = 30,000 req/s total
After Black Friday: Scale down to 3 servers
\`\`\`

**Requirements**:
- Stateless servers (sessions in Redis, not server memory)
- Load balancer (AWS ALB, NGINX)
- Shared database/cache
- Auto-scaling group (AWS Auto Scaling, Kubernetes HPA)

---

## Cost Comparison

### Vertical Scaling Cost

**Example: AWS EC2 Pricing** (approximate, 2024):
- t3.small (2 vCPU, 2GB RAM): $15/month
- t3.medium (2 vCPU, 4GB RAM): $30/month
- t3.xlarge (4 vCPU, 16GB RAM): $120/month
- t3.2xlarge (8 vCPU, 32GB RAM): $240/month
- c5.24xlarge (96 vCPU, 192GB RAM): $3,500/month

**Notice**: Cost grows exponentially, not linearly. 24xlarge is ~233x more expensive than small, but only 48x more CPU.

### Horizontal Scaling Cost

**Example**: Same budget, different approach

**Option 1 (Vertical)**: 1x c5.24xlarge = $3,500/month = 96 vCPU total

**Option 2 (Horizontal)**: 116x t3.xlarge = $3,480/month = 464 vCPU total

**Horizontal gives 4.8x more CPU for same price!**

**Considerations**:
- Horizontal needs load balancer ($15-30/month)
- Horizontal needs shared state (Redis: $50-200/month)
- But still more cost-effective at scale

---

## When to Use Each

### Use Vertical Scaling When:

**1. Database Primary/Master**
- Single-master databases can't horizontally scale writes
- Vertical scaling is only option for write scaling (until sharding)
- Example: PostgreSQL, MySQL primary

**2. Stateful Applications**
- Applications with in-memory state difficult to distribute
- Legacy applications not designed for horizontal scaling
- Example: Monolithic apps with session state

**3. Small to Medium Scale**
- <10,000 req/s
- Don't need high availability yet
- Simplicity more important than scale

**4. Quick Fix**
- Temporary solution until refactor for horizontal scaling
- Performance issue needs immediate fix

**5. Software Licensing**
- Licensed per server (Oracle, some enterprise software)
- Cheaper to have 1 powerful server than 10 weak servers

### Use Horizontal Scaling When:

**1. Stateless Applications**
- Web servers, API servers, microservices
- No local state, or state stored externally (Redis, database)

**2. Read-Heavy Workloads**
- Database read replicas
- Cache servers (multiple Redis instances)
- CDN edge servers

**3. High Availability Required**
- Mission-critical systems
- Can't afford downtime
- Need redundancy

**4. Large Scale**
- >10,000 req/s
- Need to handle traffic spikes
- Global user base (multi-region)

**5. Cost Optimization**
- Need elasticity (scale up/down based on demand)
- Prefer commodity hardware over expensive high-end servers

---

## Hybrid Approach (Most Common)

Most systems use **both** vertical and horizontal scaling:

### Example: E-commerce Architecture

**Vertically scaled components**:
- Primary database (PostgreSQL): Scale up to 128GB RAM, 64 cores
- Cache coordinator (Redis Cluster master): Scale up to 64GB RAM

**Horizontally scaled components**:
- Web servers: 10-100 instances (auto-scaling)
- API servers: 20-200 instances (auto-scaling)
- Database read replicas: 5-10 replicas
- Background job workers: 50-500 workers (auto-scaling)

**Why hybrid**:
- Vertical for components that can't horizontally scale (primary database writes)
- Horizontal for components that can scale out (stateless web servers)
- Best of both worlds: Simple where possible, scalable where needed

---

## Database Scaling Patterns

### Pattern 1: Vertical Primary + Horizontal Replicas

**Architecture**:
\`\`\`
Primary (writes): Vertically scaled (128GB RAM)
Replicas (reads): Horizontally scaled (10 replicas, 16GB RAM each)
\`\`\`

**Use case**: Read-heavy applications (90% reads, 10% writes)

**Example**: Social media, content sites

---

### Pattern 2: Sharding (Horizontal Database Scaling)

**Architecture**:
\`\`\`
Shard 1: Users 1-1M
Shard 2: Users 1M-2M
Shard 3: Users 2M-3M
...
Each shard: Vertically scaled + read replicas
\`\`\`

**Use case**: Write-heavy at massive scale

**Example**: Instagram, Facebook

**Complexity**: High (cross-shard queries, rebalancing)

---

## Cloud Auto-Scaling

Modern cloud platforms enable automatic horizontal scaling:

### AWS Auto Scaling Example

\`\`\`yaml
Auto Scaling Group:
  Min instances: 3
  Max instances: 30
  Desired: 10
  Scale up trigger: CPU > 70% for 5 minutes → Add 5 instances
  Scale down trigger: CPU < 30% for 10 minutes → Remove 2 instances
\`\`\`

**Benefits**:
- Automatic response to traffic changes
- Cost optimization (only pay for what you use)
- High availability (always min 3 instances)

**Limitations**:
- Cold start time (takes 2-5 minutes to launch new instances)
- Need application to be stateless
- Must handle instances being terminated (graceful shutdown)

---

## Common Mistakes

### ❌ Mistake 1: Premature Horizontal Scaling

**Problem**: Building distributed system before you need it

**Example**: Startup with 10 users deploys Kubernetes cluster with microservices

**Cost**: High complexity, slower development, no benefit

**Better**: Start with single server (vertical scaling), scale horizontally when needed (>10,000 req/s)

---

### ❌ Mistake 2: Ignoring Vertical Scaling Limits

**Problem**: Assuming you can always scale up

**Example**: Database at 2TB RAM, 256 cores → Can't scale up further

**Reality**: Vertical scaling has hard limits. Plan horizontal approach (sharding) before hitting limits.

---

### ❌ Mistake 3: Horizontal Scaling with Stateful Apps

**Problem**: Horizontally scaling application with local session state

**Example**:
\`\`\`
User logs in → Session stored in server A memory
Next request → Load balancer routes to server B → User "not logged in"
\`\`\`

**Solution**: Use sticky sessions (non-ideal) or externalize state (Redis)

---

## Best Practices

### ✅ 1. Start Vertical, Plan Horizontal

Begin with single server. Optimize code. Scale vertically. Plan for horizontal scaling before hitting limits.

### ✅ 2. Make Applications Stateless

Store session state in Redis, not server memory. Enable horizontal scaling.

### ✅ 3. Use Read Replicas

Database read replicas = horizontal scaling for reads. Much easier than sharding.

### ✅ 4. Implement Health Checks

Load balancer must detect unhealthy instances and stop routing traffic.

### ✅ 5. Graceful Shutdown

Handle termination signals. Finish in-flight requests before shutting down.

### ✅ 6. Monitor and Alert

Track CPU, memory, network. Alert before hitting capacity. Auto-scale based on metrics.

---

## Interview Tips

### Strong Answer Pattern

"For this system, I'd use a **hybrid approach**:

**Vertical scaling for**:
- Primary database (single master for writes)
- Scale to [64GB/128GB RAM] before considering sharding
- Trade-off: Simpler, but has limits

**Horizontal scaling for**:
- Web/API servers (stateless)
- Database read replicas (read scaling)
- Background job workers
- Auto-scaling based on CPU/request count

**Reasoning**: [95% reads, 5% writes] → Read replicas handle most load. Primary database can scale vertically for foreseeable future. Web servers are stateless, easy to horizontally scale.

**At scale** (if needed): Consider sharding database by [user_id, region] when vertical limits reached."

---

## Summary Table

| Aspect | Vertical Scaling | Horizontal Scaling |
|--------|-----------------|-------------------|
| **Method** | Bigger machine | More machines |
| **Complexity** | Simple | Complex (distributed) |
| **Limits** | Hard limits (hardware) | Nearly unlimited |
| **Downtime** | Yes (during upgrade) | No (add/remove servers) |
| **Cost** | Expensive at high-end | Cost-efficient (commodity) |
| **Availability** | Single point of failure | High availability |
| **Use Cases** | Databases, stateful | Web servers, APIs |
| **Code Changes** | None needed | Must be stateless |
| **Debugging** | Easy (one server) | Complex (distributed logs) |

---

## Key Takeaways

✅ Vertical scaling: Upgrade single machine (simple but limited)
✅ Horizontal scaling: Add more machines (complex but unlimited)
✅ Vertical is cost-inefficient at high-end (exponential pricing)
✅ Horizontal enables high availability and elasticity
✅ Most systems use hybrid: Vertical for databases, horizontal for web servers
✅ Start simple (vertical), scale horizontally when needed
✅ Applications must be stateless for horizontal scaling
✅ Database reads scale horizontally (replicas), writes scale vertically (until sharding)`,
};
