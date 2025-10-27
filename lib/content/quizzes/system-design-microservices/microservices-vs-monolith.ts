/**
 * Quiz questions for Microservices vs Monolith section
 */

export const microservicesvsmonolithQuiz = [
  {
    id: 'q1',
    question:
      'You\'re the tech lead at a 50-person engineering company. The CTO wants to migrate from your monolith to microservices "because that\'s what Netflix does." How do you respond? What factors would you consider, and what would be your recommendation?',
    sampleAnswer: `I would caution against blindly copying Netflix\'s architecture and instead focus on our specific needs and constraints.

**Key Considerations:**1. **Organizational Maturity**: Do we have the DevOps culture and tooling (CI/CD, monitoring, distributed tracing, on-call rotations) to support microservices? With 50 engineers, we might be at the threshold, but we need to honestly assess our operational readiness.

2. **Current Pain Points**: What problems are we trying to solve? If teams are blocked waiting for deployments, or we have vastly different scaling needs per component, microservices might help. But if the main issue is code quality or technical debt, microservices will just distribute the mess.

3. **Domain Clarity**: Do we have clear, stable domain boundaries? Microservices work best when services map cleanly to business capabilities. If our domains are still evolving rapidly, premature decomposition will lead to many cross-service changes.

4. **Cost-Benefit Analysis**: Microservices bring distributed system complexity. Will the benefits (independent deployment, scaling, technology choice) outweigh the costs (operational overhead, data consistency challenges, testing complexity)?

**My Recommendation**: I would propose a hybrid approach:

1. **First, improve the monolith**: Introduce clear module boundaries, implement modular monolith patterns, improve testing and deployment pipelines. This gives many benefits without the complexity.

2. **Identify 2-3 service candidates**: Look for components that have clear boundaries, different scaling needs, or would benefit from different technology (e.g., real-time services in Go, ML services in Python).

3. **Extract one service using Strangler Fig**: Start with a low-risk service (like notifications or email) to validate our approach and build operational muscle.

4. **Evaluate after 3-6 months**: Did we get the expected benefits? What challenges emerged? Use learnings to decide next steps.

5. **Only continue if benefits are clear**: Don't extract services just because we can. Each service should solve a real problem.

**To the CTO**: "Netflix has 100x our engineering team and operates at global scale with different technology needs per service. They also invested heavily in tooling (Zuul, Eureka, Hystrix). Let's solve our specific problems rather than copy their architecture. I propose we start with a modular monolith and extract services strategically where we have clear pain points."`,
    keyPoints: [
      'Assess organizational readiness (DevOps maturity, team size)',
      'Identify specific pain points that microservices would solve',
      'Propose incremental approach (Strangler Fig) rather than big rewrite',
      'Start with modular monolith improvements',
      'Extract 1-2 services to validate approach and learn',
    ],
  },
  {
    id: 'q2',
    question:
      'Design an e-commerce system for 10M users. Would you use microservices or monolith? Justify your choice and outline your architecture.',
    sampleAnswer: `This decision depends heavily on the team size and organizational context, but I'll outline both approaches for 10M users.

**Assumption**: We have a 40-50 person engineering team with decent DevOps maturity.

**My Recommendation: Start with Modular Monolith, Extract Key Services**

Here\'s my reasoning and architecture:

**Phase 1: Modular Monolith Foundation (Months 0-6)**

Core monolith handling:
- User management (authentication, profiles)
- Product catalog (browsing, search)
- Shopping cart
- Order management
- Basic checkout flow

**Benefits**: Fast development, simple deployment, ACID transactions for orders, easy testing.

**Architecture**:
\`\`\`
Load Balancer
    ↓
API Gateway
    ↓
Monolith Application (modular boundaries)
    ↓
Primary Database (PostgreSQL)
Read Replicas
\`\`\`

**Phase 2: Extract High-Value Services (Months 6-12)**

Based on 10M users, I would extract these services early:

**1. Payment Service (Separate from Day 1 if possible)**
- Reason: Security isolation, PCI compliance, different tech stack
- Critical: Must be reliable, needs separate audit trail
- Technology: Java/Go for reliability

**2. Search Service**
- Reason: Different scaling needs, different tech (Elasticsearch)
- 10M users = millions of searches/day
- CPU-intensive, needs horizontal scaling independent of checkout

**3. Notification Service**
- Reason: Async by nature, can tolerate failures without impacting checkout
- Handles email, SMS, push notifications
- Can be down without impacting core business

**4. Inventory Service**
- Reason: High write load, different consistency requirements
- Needs real-time updates
- Potential for race conditions in monolith

**Phase 2 Architecture**:
\`\`\`
Users
  ↓
Load Balancer / CDN
  ↓
API Gateway
  ├→ Monolith (User, Product, Cart, Orders)
  │   └→ PostgreSQL Primary + Replicas
  │
  ├→ Search Service (Elasticsearch)
  ├→ Payment Service (PCI-compliant, isolated)
  ├→ Inventory Service (high write load)
  └→ Notification Service (async queue)

Message Queue (RabbitMQ/Kafka)
  └→ Async communication between services
\`\`\`

**What Stays in Monolith**:
- User management (low write volume)
- Product catalog (mostly reads, cached)
- Cart management (session-based)
- Order orchestration (benefits from transactions)

**Why This Hybrid Approach**:

1. **Pragmatic**: Get benefits of microservices (independent scaling for search/inventory, isolation for payments) without full complexity

2. **Scalability**: With 10M users, we'll likely have:
   - ~1M daily active users
   - ~100K concurrent users at peak
   - ~10K orders/hour at peak
   
   This is manageable with properly scaled monolith + key services

3. **Operational Complexity**: 4-5 services is manageable, 20+ would be overkill

4. **Evolution Path**: If we grow to 50M+ users, we can further decompose the monolith

**Alternative: Full Microservices (If Team is 100+ Engineers)**

If we had a much larger team, I'd go full microservices from the start:
- User Service
- Product Service
- Cart Service
- Order Service
- Payment Service
- Inventory Service
- Shipping Service
- Notification Service
- Recommendation Service
- Review Service

But with 40-50 engineers, this would create too much operational overhead.

**Key Point for Interview**: The answer depends on team size, operational maturity, and specific requirements. There\'s no one-size-fits-all. Always justify your choice with specific reasoning.`,
    keyPoints: [
      'Decision depends on team size and operational maturity',
      'Hybrid approach balances benefits and complexity',
      'Extract services with clear benefits (Payment for security, Search for scale)',
      'Keep cohesive business logic in monolith',
      'Consider evolutionary architecture (start simple, extract as needed)',
    ],
  },
  {
    id: 'q3',
    question:
      'What are the most common mistakes teams make when migrating from monolith to microservices? How would you avoid them?',
    sampleAnswer: `Based on real-world migrations, here are the critical mistakes and how to avoid them:

**Mistake 1: Big Bang Rewrite**

**What happens**: Team decides to rewrite entire monolith to microservices in one go, estimating 6-12 months. Project takes 18-24 months, business features stop, morale suffers, and often the project is abandoned.

**Why it fails**: 
- Underestimate complexity
- Business can't wait years for new features
- Learn painful lessons too late (can't pivot easily)

**Solution: Strangler Fig Pattern**
- Extract ONE service at a time
- Run in parallel with monolith initially
- Validate approach, learn lessons
- Iterate based on learnings
- Feature flag everything for easy rollback

**Mistake 2: Creating a Distributed Monolith**

**What happens**: Services share the same database, are tightly coupled via synchronous calls, and can't be deployed independently. You get microservices complexity without the benefits.

**Red flags**:
- "We need to deploy Service A and B together"
- All services connect to same PostgreSQL database
- Synchronous chain: A → B → C → D (all must be up)

**Solution**:
- Database per service (truly independent data)
- Async communication where possible (events/messaging)
- Design for service failure (circuit breakers)
- Clear service boundaries based on domains

**Mistake 3: Wrong Service Boundaries**

**What happens**: Services are decomposed incorrectly, leading to constant cross-service changes and chatty communication.

**Example**: E-commerce split by "technical layers"
- Frontend Service
- Business Logic Service
- Database Service
(Every feature touches all three services)

**Better approach**: Domain-driven design
- Order Service (handles entire order domain)
- Product Service (handles entire product domain)
- User Service (handles entire user domain)

**Solution**:
- Follow domain boundaries, not technical layers
- Each service should be independently changeable
- Minimize cross-service transactions
- Test: "Can I add a feature touching only one service?"

**Mistake 4: Underestimating Operational Complexity**

**What happens**: Team focuses on development benefits, ignores operational reality. Production becomes a nightmare of debugging distributed systems.

**Missing pieces**:
- No distributed tracing (can't find which service caused error)
- No centralized logging (can't correlate logs)
- No service mesh (mTLS, retry logic, circuit breakers)
- No standardized monitoring

**Solution: Invest in Platform First**
Before extracting services, build:
1. Service discovery (Consul, Eureka)
2. API Gateway (Kong, Ambassador)
3. Distributed tracing (Jaeger, Zipkin)
4. Centralized logging (ELK, Splunk)
5. Metrics/monitoring (Prometheus + Grafana)
6. CI/CD pipeline per service
7. Infrastructure as code

**Mistake 5: Ignoring Data Consistency Challenges**

**What happens**: Team realizes too late that transactions don't work across services. Data inconsistencies cause business problems.

**Example**: Order checkout
1. Inventory service decrements stock ✓
2. Payment service fails ✗
3. Result: Stock decremented, but no payment (unhappy customer)

**Solution**:
- Use Saga pattern (orchestration or choreography)
- Implement compensating transactions
- Event sourcing for critical flows
- Accept eventual consistency where business allows
- Test failure scenarios extensively

**Mistake 6: Too Many Services, Too Soon**

**What happens**: Team extracts 30+ services for 10-person team. Cognitive overhead kills productivity.

**Problems**:
- Can't remember what services do
- Deployment takes hours (30 services to deploy)
- Debugging requires jumping between 10 services
- Every feature touches 5 services

**Solution**:
- Start with 3-5 services
- Extract only when clear benefit
- Each service should have clear owner
- Rule of thumb: Max 2-3 services per team

**Mistake 7: No Migration Validation**

**What happens**: Team migrates service, hopes it works the same, finds subtle bugs in production.

**Solution: Shadowing/Dark Launch**1. Deploy new service
2. Route traffic to BOTH old monolith and new service
3. Return monolith response to user
4. Compare responses in background
5. Fix discrepancies
6. Only fully cutover when behavior identical

**Mistake 8: Cargo-Culting Netflix/Google Architecture**

**What happens**: Small company with 20 engineers tries to implement Netflix\'s 700-service architecture.

**Reality**: 
- Netflix has 1000s of engineers
- They built custom tooling (Zuul, Eureka, Hystrix)
- Took years to get there
- Had organizational support

**Solution**:
- Design for YOUR scale and team size
- Borrow patterns, not exact architecture
- Start simple, evolve as you grow
- Be honest about operational maturity

**How I'd Lead a Migration:**1. **Assess Readiness** (2 weeks)
   - Team size, DevOps maturity, pain points

2. **Build Platform** (2-3 months)
   - Service discovery, API gateway, observability

3. **Extract First Service** (1 month)
   - Choose low-risk service (notifications)
   - Validate approach

4. **Learn and Iterate** (1 month)
   - What worked? What didn't?
   - Adjust approach

5. **Extract More Services** (6-12 months)
   - Based on learnings
   - Continue only if benefits are clear

6. **Measure Success**
   - Deployment frequency increased?
   - Team velocity improved?
   - Operational burden manageable?
   - If no, stop extracting

**Key Insight**: Many teams would be better off with a well-designed modular monolith than poorly implemented microservices. Complexity should be justified.`,
    keyPoints: [
      'Avoid big bang rewrites (use Strangler Fig pattern)',
      "Don't create distributed monolith (database per service)",
      'Get service boundaries right (domain-driven, not technical layers)',
      'Build platform/tooling first (observability, service discovery)',
      'Address data consistency (Saga pattern, eventual consistency)',
      'Start with few services (3-5), not 30+',
      'Validate migrations (shadow traffic, compare responses)',
      "Don't cargo-cult other companies' architectures",
      "Measure success and be willing to stop if benefits don't materialize",
    ],
  },
];
