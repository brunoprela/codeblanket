import { Module } from '@/lib/types';

export const systemDesignMicroservicesModule: Module = {
  id: 'system-design-microservices',
  title: 'Microservices Architecture',
  description:
    'Master microservices patterns, decomposition strategies, and distributed system challenges',
  icon: '🔬',
  category: 'System Design',
  difficulty: 'Advanced',
  estimatedTime: '4-5 hours',
  sections: [
    {
      id: 'microservices-vs-monolith',
      title: 'Microservices vs Monolith',
      content: `Choosing between microservices and monolithic architectures is one of the most consequential decisions in system design. Let's explore both approaches, their trade-offs, and when to use each.

## What is a Monolith?

A **monolithic architecture** is a single, unified application where all components are tightly integrated and deployed as one unit.

**Characteristics**:
- Single codebase
- Single deployment unit
- Shared database
- Components communicate via function calls
- Scale entire application together

**Example Structure**:
\`\`\`
Monolithic E-commerce App
├── User Management
├── Product Catalog
├── Shopping Cart
├── Order Processing
├── Payment
├── Inventory
├── Shipping
└── Notifications
→ All deployed together as one WAR/JAR/binary
\`\`\`

---

## What are Microservices?

**Microservices architecture** decomposes an application into small, independent services that communicate over a network.

**Characteristics**:
- Multiple codebases (one per service)
- Independent deployment
- Separate databases (database per service)
- Services communicate via APIs (HTTP/gRPC/messaging)
- Scale services independently

**Example Structure**:
\`\`\`
Microservices E-commerce
├── User Service        (separate deployment)
├── Product Service     (separate deployment)
├── Cart Service        (separate deployment)
├── Order Service       (separate deployment)
├── Payment Service     (separate deployment)
├── Inventory Service   (separate deployment)
├── Shipping Service    (separate deployment)
└── Notification Service (separate deployment)
→ Each deployed independently
\`\`\`

---

## Benefits of Monolithic Architecture

### 1. Simplicity
- **Single codebase**: Easier to navigate and understand
- **Simple deployment**: One deployment pipeline
- **No network complexity**: Function calls, not API calls
- **Easy to debug**: All code in one place, single stack trace

### 2. Development Speed (Initially)
- **Faster feature development**: No service boundaries to cross
- **Easy refactoring**: Can change anything without API versioning
- **Simple testing**: Integration tests in one application
- **Quick setup**: New developers get productive faster

### 3. Performance
- **No network overhead**: Function calls are microseconds vs milliseconds for API calls
- **Transactions**: ACID transactions across entire application
- **No serialization**: No JSON/protobuf encoding/decoding

### 4. Operational Simplicity
- **Single deployment**: Deploy once, not 50 times
- **Easier monitoring**: One application to monitor
- **Simpler infrastructure**: One server (initially)
- **Lower costs**: Fewer resources needed

---

## Challenges of Monolithic Architecture

### 1. Scalability Limitations
**Problem**: Must scale entire application even if only one component is a bottleneck.

**Example**: If your product search is CPU-intensive but checkout is memory-intensive, you need servers with both high CPU AND memory. Wasteful.

**Horizontal scaling challenges**:
- Stateless design required for load balancing
- Session management complexity
- Shared database becomes bottleneck

### 2. Deployment Risk
**Problem**: Every deployment is high-risk because entire application redeploys.

**Impact**:
- Bug in payment system brings down product catalog too
- Small change requires full regression testing
- Deployment windows become large
- Fear of deployment leads to infrequent releases

### 3. Technology Lock-in
**Problem**: Stuck with initial technology choices.

**Example**: Started with Ruby on Rails in 2015, now in 2024 you want to use Go for high-performance services. You're stuck rewriting entire application or living with suboptimal performance.

### 4. Team Coordination Overhead
**Problem**: All teams work in same codebase, requiring coordination.

**Issues**:
- Merge conflicts
- Code review bottlenecks
- "Broken main branch" blocks everyone
- Hard to parallelize work across teams

### 5. Long-Term Maintainability
**"Big Ball of Mud"**: Over time, monoliths tend to become tightly coupled and hard to reason about.

**Issues**:
- Dependencies between modules become unclear
- Changes have unexpected side effects
- Technical debt accumulates
- Onboarding new engineers takes weeks

---

## Benefits of Microservices

### 1. Independent Scalability
**Scale only what needs scaling**.

**Example**: Netflix
- Streaming service (video delivery): Needs 1000s of instances
- User profile service: Needs 10 instances
- Billing service: Needs 5 instances

**Cost savings**: Don't over-provision everything.

### 2. Technology Flexibility
**Choose the right tool for each job**.

**Example**: Uber
- Dispatch system: Go (high performance, low latency)
- Data analytics: Python (rich ML libraries)
- Mobile API: Node.js (good for I/O)
- Fraud detection: Java (mature ecosystem)

### 3. Fault Isolation
**A failure in one service doesn't crash entire system**.

**Example**: Amazon
- Recommendation engine down → You can still browse and buy
- Reviews service slow → Rest of product page loads fine

**Implementation**: Circuit breakers prevent cascading failures.

### 4. Independent Deployment
**Deploy services independently without coordinating**.

**Benefits**:
- Checkout team deploys 10x/day
- Product team deploys 3x/week
- No deployment coordination needed
- Smaller change sets = lower risk

### 5. Team Autonomy
**Conway's Law**: "Organizations design systems that mirror their communication structure."

**Microservices alignment**:
- Team owns service end-to-end (full-stack ownership)
- No coordination with other teams for deployment
- Choose tech stack
- Define API contracts
- Move fast independently

### 6. Easier to Understand
**Each service is small and focused**.

**Example**: Order Service is 5,000 lines vs Monolith is 500,000 lines.

New engineer can understand Order Service in days vs weeks for monolith.

---

## Challenges of Microservices

### 1. Distributed System Complexity

**Network is unreliable**:
- Services can be down
- Network latency (milliseconds vs microseconds)
- Timeouts and retries
- Partial failures

**Example**: Order checkout calls 5 services:
\`\`\`
Order Service → Inventory Service (timeout!)
              → Payment Service (success)
              → User Service (success)
              → Shipping Service (down!)
              → Notification Service (success)
\`\`\`
Now what? Partial success? Compensating transactions?

### 2. Data Consistency Challenges

**Problem**: Each service has its own database. No ACID transactions across services.

**Example**: E-commerce order
1. Order Service: Create order ✓
2. Inventory Service: Decrement stock ✗ (fails)
3. Payment Service: Charge card ✓

**Result**: Customer charged but no inventory reserved. Inconsistent state.

**Solutions**: 
- Saga pattern (compensating transactions)
- Event sourcing
- Eventual consistency

### 3. Operational Complexity

**More moving parts**:
- 50 services = 50 deployments, 50 monitoring dashboards, 50 log streams
- Service discovery
- Load balancing
- API gateway
- Distributed tracing (what service caused the error?)
- Centralized logging

**Tooling required**: Kubernetes, service mesh, distributed tracing (Jaeger), centralized logging (ELK), metrics (Prometheus/Grafana).

### 4. Testing Complexity

**Integration testing is hard**:
- Need to spin up multiple services
- Mock external dependencies
- End-to-end tests are slow and flaky
- Contract testing required

**Example**: Testing checkout flow requires:
- User Service
- Product Service
- Inventory Service
- Payment Service
- Order Service
- Notification Service
- Test database for each

### 5. Organizational Overhead

**Requirements**:
- Mature DevOps culture
- Well-defined API contracts
- Service ownership model
- On-call rotations
- More communication overhead

**Not suitable for**: Small teams (< 10 engineers) or startups finding product-market fit.

### 6. Performance Overhead

**Network calls are expensive**:
- Function call: ~1 microsecond
- HTTP call within datacenter: ~1-10 milliseconds
- Serialization/deserialization overhead

**Example**: Monolith page load → 5ms total
Microservices page load → 50ms (10 service calls × 5ms each)

### 7. Distributed Transactions

**No simple ACID transactions**:
- 2-Phase Commit (2PC) is slow and fragile
- Saga pattern is complex
- Eventual consistency is hard to reason about

---

## When to Use Monolith

### ✅ Good Fit For:

**1. Startups (Pre-Product-Market Fit)**
- Need speed to iterate
- Small team (< 10 engineers)
- Uncertain requirements
- Need to pivot quickly

**2. Simple Applications**
- CRUD applications
- Internal tools
- MVPs
- Low traffic applications

**3. Small Teams**
- Team can understand entire codebase
- Don't have DevOps maturity
- Limited operational bandwidth

**4. Tight Performance Requirements**
- Low-latency requirements (trading systems)
- Need ACID transactions
- Complex queries across domains

### 🏢 Companies Using Monoliths Successfully:

- **Shopify**: Massive Ruby on Rails monolith serving millions of stores
- **Basecamp**: Rails monolith, 60 employees, millions of users
- **Stack Overflow**: .NET monolith, 10M+ monthly visitors

**Key**: Monoliths can scale! Good architecture matters more than microservices.

---

## When to Use Microservices

### ✅ Good Fit For:

**1. Large Organizations**
- Multiple teams (> 50 engineers)
- Need team autonomy
- Different teams own different domains

**2. Scale Requirements**
- Different scaling needs per component
- Global distribution
- High availability requirements

**3. Different Technology Needs**
- ML models (Python)
- Real-time services (Go)
- Complex business logic (Java)

**4. Mature DevOps Culture**
- Strong CI/CD
- Infrastructure as code
- Monitoring/observability
- On-call culture

### 🏢 Companies Using Microservices:

- **Netflix**: 700+ microservices
- **Uber**: 2,000+ microservices
- **Amazon**: "Two-pizza team" rule → microservices
- **Airbnb**: 1,000+ microservices

---

## Migration Strategy: Strangler Fig Pattern

**Don't rewrite monolith to microservices in one go!**

**Strangler Fig Pattern**: Incrementally replace monolith pieces.

### Migration Steps:

**1. Add API Gateway**
\`\`\`
Users → API Gateway → Monolith
\`\`\`

**2. Extract One Service (lowest risk)**
\`\`\`
Users → API Gateway → Monolith
                    ↘
                     Notification Service (new)
\`\`\`

**3. Route traffic to new service**
\`\`\`
Users → API Gateway → Monolith (notifications disabled)
                    ↘
                     Notification Service (handles all notifications)
\`\`\`

**4. Repeat for other services**
\`\`\`
Users → API Gateway → Monolith (smaller)
                    ↘ User Service
                    ↘ Notification Service
                    ↘ Payment Service
\`\`\`

**5. Eventually, monolith is gone**
\`\`\`
Users → API Gateway → User Service
                    ↘ Order Service
                    ↘ Product Service
                    ↘ Payment Service
                    ... (10+ services)
\`\`\`

### Tips for Migration:

1. **Start with low-risk services**: Notifications, email sending
2. **Avoid data-heavy services first**: Don't start with "User" or "Order"
3. **Extract vertically**: Service + database + UI
4. **Feature flag everything**: Easy rollback
5. **Run in parallel**: Dual-write pattern initially
6. **Monitor closely**: Compare monolith vs microservice behavior

---

## The Distributed Monolith Anti-Pattern

**Worst of both worlds**: Microservices architecture with monolith dependencies.

### Characteristics:

1. **Tight coupling**: Services can't be deployed independently
2. **Shared database**: All services use same database
3. **Synchronous chains**: Service A → B → C → D (all must be up)
4. **No ownership boundaries**: Teams work across all services

**Example**:
\`\`\`
User Service → (calls) → Order Service → (calls) → Inventory Service
                                      ↘
                                       Payment Service
                                          ↓
                                    (All use same MySQL database)
\`\`\`

**Problems**:
- Can't deploy independently (breaking changes across services)
- Database becomes bottleneck
- Network overhead with no benefits
- Complex deployment dependencies

**Solution**: 
- True service boundaries (database per service)
- Async communication
- Independent deployments

---

## Real-World Examples

### Example 1: Amazon's Journey

**1990s**: Monolithic application
**Problem**: Teams blocked each other, slow deployment

**Solution**: Two-pizza team rule
- If team needs more than 2 pizzas, it's too big
- Each team owns a service end-to-end
- API-first culture

**Result**: 
- Deploy every 11.6 seconds
- Independent team velocity

### Example 2: Netflix

**Monolith (2008)**: DVD rental business
**Challenge**: Moving to streaming, need to scale

**Microservices (2012+)**: 
- 700+ microservices
- Independent scaling
- Fault isolation (recommendation down ≠ streaming down)

**Key**: Invested heavily in tooling (Eureka, Zuul, Hystrix)

### Example 3: Shopify's Successful Monolith

**2023**: Ruby on Rails monolith
**Scale**: Powers millions of stores

**Why monolith works**:
- Well-architected modular monolith
- Strong team ownership of modules
- Good testing practices
- Can extract services when needed

**Lesson**: Monolith ≠ bad if well-designed

---

## Decision Framework

### Choose Monolith If:
- ✅ Team < 10 engineers
- ✅ Startup / uncertain product
- ✅ Simple domain
- ✅ Limited operational maturity
- ✅ Need fast iteration

### Choose Microservices If:
- ✅ Team > 50 engineers
- ✅ Clear domain boundaries
- ✅ Different scaling needs
- ✅ Mature DevOps culture
- ✅ Multiple technology requirements

### Start as Monolith, Extract Services When:
- ✅ Team growing
- ✅ Deployment coordination painful
- ✅ Clear scaling bottlenecks
- ✅ Technology limitations

---

## Interview Tips

### Red Flags:
❌ "Always use microservices" (no justification)
❌ "Monoliths don't scale" (false)
❌ Ignoring operational complexity
❌ No migration strategy

### Good Responses:
✅ "It depends on team size, scale, and domain clarity"
✅ Discuss trade-offs clearly
✅ Mention operational requirements
✅ Suggest starting simple, evolving as needed
✅ Reference real companies

### Sample Answer:
*"For a startup with 5 engineers, I'd recommend starting with a well-architected monolith. We can move fast, avoid operational complexity, and don't yet know which parts will need independent scaling. As we grow to 30+ engineers with clear domain boundaries, we can use the Strangler Fig pattern to gradually extract microservices for components with different scaling needs."*

---

## Key Takeaways

1. **No silver bullet**: Both architectures have trade-offs
2. **Microservices = distributed systems complexity**: Only worth it if benefits outweigh costs
3. **Start simple**: Monolith → extract services as needed
4. **Team size matters**: Microservices require organizational maturity
5. **Well-designed monolith > poorly designed microservices**
6. **Migration > rewrite**: Use Strangler Fig pattern
7. **Avoid distributed monolith**: Worst of both worlds

The question isn't "*Should we use microservices?*" but "*Do the benefits outweigh the complexity for our specific context?*"`,
      multipleChoice: [
        {
          id: 'mc-microservices-1',
          question:
            'Your startup has 8 engineers and is still finding product-market fit. Users are growing 50% month-over-month. What architecture should you choose?',
          options: [
            'Microservices from day 1 to prepare for scale',
            'Well-designed monolith with clear module boundaries',
            'Serverless functions for everything',
            'Distributed monolith with shared database',
          ],
          correctAnswer: 1,
          explanation:
            'With a small team (8 engineers) and uncertain product direction (still finding PMF), a well-designed modular monolith is ideal. It allows fast iteration, avoids operational complexity of microservices, and can be extracted into services later using the Strangler Fig pattern. Premature microservices would slow down development and add unnecessary complexity. The 50% growth, while impressive, can be handled by vertical/horizontal scaling of a monolith.',
        },
        {
          id: 'mc-microservices-2',
          question:
            'Which of the following is NOT a benefit of microservices architecture?',
          options: [
            'Independent scalability of different components',
            'Lower operational complexity than monoliths',
            'Technology flexibility across services',
            'Fault isolation between services',
          ],
          correctAnswer: 1,
          explanation:
            "Microservices have HIGHER operational complexity than monoliths, not lower. You must manage service discovery, distributed tracing, multiple deployments, network failures, and eventual consistency. The other options are genuine benefits: independent scaling (scale only what needs it), technology flexibility (use Go for performance, Python for ML), and fault isolation (one service failure doesn't crash everything).",
        },
        {
          id: 'mc-microservices-3',
          question:
            'What is a "distributed monolith" and why is it problematic?',
          options: [
            'A monolith deployed across multiple regions for low latency',
            'Microservices architecture where services share a database and are tightly coupled',
            'A monolith that uses distributed caching for performance',
            'A microservice that handles multiple domains',
          ],
          correctAnswer: 1,
          explanation:
            "A distributed monolith is the worst of both worlds: you have the operational complexity of microservices (network calls, multiple deployments, distributed system challenges) but without the benefits (services can't deploy independently due to tight coupling, shared database creates bottleneck). It's an anti-pattern that often results from poorly designed microservices migration. Signs include: shared database, synchronous coupling chains, inability to deploy services independently.",
        },
        {
          id: 'mc-microservices-4',
          question:
            'Your company has 100 engineers across 15 teams. Deployment coordination is painful, with teams blocked waiting for others. Different components have vastly different scaling needs. What should you do?',
          options: [
            'Keep monolith but improve coordination processes',
            'Rewrite entire monolith to microservices in 6 months',
            'Use Strangler Fig pattern to gradually extract services',
            'Create a distributed monolith with shared database',
          ],
          correctAnswer: 2,
          explanation:
            'The Strangler Fig pattern is the safe approach: incrementally extract services starting with low-risk components (like notifications), validate the approach, then continue. This avoids the high-risk "big bang" rewrite while allowing teams to gain microservices experience gradually. With 100 engineers and clear pain points (deployment blocking, different scaling needs), you have the organizational maturity for microservices. Don\'t improve coordination (scales poorly) or create distributed monolith (anti-pattern). Never do big rewrites (high failure rate).',
        },
        {
          id: 'mc-microservices-5',
          question:
            'Which company successfully operates at massive scale with a monolithic architecture?',
          options: [
            'Netflix (700+ microservices)',
            'Shopify (Rails monolith powering millions of stores)',
            'Uber (2,000+ microservices)',
            'Amazon (two-pizza team microservices)',
          ],
          correctAnswer: 1,
          explanation:
            "Shopify is a famous example of a successful monolith at scale. Despite powering millions of stores, they maintain a well-architected Rails monolith with clear module boundaries, strong ownership, and good testing. They extract services only when truly needed. This proves that monoliths CAN scale with good architecture. Netflix, Uber, and Amazon all use extensive microservices. The lesson: architecture quality matters more than whether it's monolith or microservices.",
        },
      ],
      quiz: [
        {
          id: 'q1',
          question:
            'You\'re the tech lead at a 50-person engineering company. The CTO wants to migrate from your monolith to microservices "because that\'s what Netflix does." How do you respond? What factors would you consider, and what would be your recommendation?',
          sampleAnswer: `I would caution against blindly copying Netflix's architecture and instead focus on our specific needs and constraints.

**Key Considerations:**

1. **Organizational Maturity**: Do we have the DevOps culture and tooling (CI/CD, monitoring, distributed tracing, on-call rotations) to support microservices? With 50 engineers, we might be at the threshold, but we need to honestly assess our operational readiness.

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

Here's my reasoning and architecture:

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

**Key Point for Interview**: The answer depends on team size, operational maturity, and specific requirements. There's no one-size-fits-all. Always justify your choice with specific reasoning.`,
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

**Solution: Shadowing/Dark Launch**
1. Deploy new service
2. Route traffic to BOTH old monolith and new service
3. Return monolith response to user
4. Compare responses in background
5. Fix discrepancies
6. Only fully cutover when behavior identical

**Mistake 8: Cargo-Culting Netflix/Google Architecture**

**What happens**: Small company with 20 engineers tries to implement Netflix's 700-service architecture.

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

**How I'd Lead a Migration:**

1. **Assess Readiness** (2 weeks)
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
      ],
    },
    {
      id: 'service-decomposition',
      title: 'Service Decomposition Strategies',
      content: `Breaking down a monolith into microservices is one of the hardest problems in system design. Poor service boundaries lead to distributed monoliths, while good boundaries enable team autonomy and independent deployment.

## The Challenge of Decomposition

**Why It's Hard**:
- Business logic is deeply intertwined
- Data dependencies are unclear
- Existing code reflects years of quick fixes
- No obvious boundaries
- Fear of breaking production

**Cost of Getting It Wrong**:
- Services that must always deploy together
- Constant cross-service changes
- "Chatty" services (N+1 calls)
- Distributed monolith (worst of both worlds)

---

## Decomposition Strategy 1: By Business Capability

**Concept**: Organize services around what the business does, not how it's implemented technically.

**Example: E-commerce**

Instead of:
\`\`\`
❌ Technical Decomposition (Bad):
- Frontend Service
- Backend Service
- Database Service
- Cache Service
\`\`\`

Use business capabilities:
\`\`\`
✅ Business Capability Decomposition (Good):
- Product Catalog Service (manages products)
- Shopping Cart Service (handles cart operations)
- Order Management Service (processes orders)
- Payment Service (handles payments)
- Shipping Service (manages shipments)
- Customer Service (manages customer data)
- Inventory Service (tracks stock)
\`\`\`

**Why This Works**:
- Each service maps to business function
- Business changes are localized
- Easy for non-technical stakeholders to understand
- Clear ownership ("Payments team owns Payment Service")

**Real Example: Amazon**

Amazon's organization structure:
- Retail team → Retail services
- AWS team → AWS services
- Kindle team → Kindle services
- Prime Video team → Video services

**Conway's Law**: "Organizations design systems that mirror their communication structure."

Amazon designed services to match organizational boundaries.

---

## Decomposition Strategy 2: Domain-Driven Design (DDD)

**Concept**: Decompose by business domains and subdomains.

### Key DDD Concepts:

**1. Bounded Context**

A **bounded context** is a clear boundary within which a domain model is defined.

**Example: "Customer" means different things in different contexts**:
- **Sales Context**: Customer = potential buyer, has leads, opportunities
- **Shipping Context**: Customer = delivery address, shipping preferences
- **Support Context**: Customer = ticket history, support level
- **Billing Context**: Customer = payment methods, invoice history

Each context has its own model of "Customer" with different attributes and behaviors.

**2. Subdomain**

**Core Domain**: Your competitive advantage (e.g., recommendation algorithm for Netflix)
**Supporting Domain**: Necessary but not differentiating (e.g., user authentication)
**Generic Domain**: Can buy off the shelf (e.g., email sending via SendGrid)

**Strategy**:
- Invest heavily in Core Domain
- Minimal effort on Supporting Domain
- Buy Generic Domain solutions

**3. Ubiquitous Language**

Use the same terms in code as business uses.

**Example**:
- Business says "Order" → Code has Order entity (not "Transaction" or "Purchase")
- Business says "Fulfillment" → Service called FulfillmentService (not "ShippingProcessor")

### DDD Decomposition Example: Insurance Company

**Bounded Contexts**:

1. **Policy Management Context**
   - Entities: Policy, Coverage, Premium
   - Operations: CreatePolicy, RenewPolicy, ModifyPolicy
   - Owner: Policy Management team

2. **Claims Context**
   - Entities: Claim, Adjuster, Settlement
   - Operations: FileClaim, AssignAdjuster, ProcessSettlement
   - Owner: Claims team

3. **Billing Context**
   - Entities: Invoice, Payment, PaymentMethod
   - Operations: GenerateInvoice, ProcessPayment, HandleRefund
   - Owner: Billing team

4. **Underwriting Context**
   - Entities: RiskAssessment, Quote, Application
   - Operations: AssessRisk, GenerateQuote, ApproveApplication
   - Owner: Underwriting team

**Key Insight**: "Policy" exists in multiple contexts, but means different things and has different attributes in each.

**Service Communication**:
\`\`\`
Policy Service (creates policy)
    ↓ [event: PolicyCreated]
Billing Service (creates invoice)
    ↓ [event: InvoiceGenerated]
Notification Service (sends email to customer)
\`\`\`

Each service has its own database with its own representation of data.

---

## Decomposition Strategy 3: By Subdomain (DDD)

**Core Domain**: Focus your best engineers here. This is your competitive advantage.

**Example: Netflix**
- **Core Domain**: Recommendation algorithm (differentiator)
- **Supporting Domain**: User profiles, viewing history
- **Generic Domain**: Payment processing (use Stripe), email (use SendGrid)

**Service Investment**:
- Recommendation Service: Custom ML models, A/B testing, high investment
- Profile Service: Standard CRUD, minimal investment
- Payment Service: Third-party integration (Stripe)

**Strategy**:
- Build Core Domain services in-house with best tech
- Keep Supporting Domain simple
- Buy or use SaaS for Generic Domain

---

## Decomposition Strategy 4: By Transaction

**Concept**: Group operations that need to be ACID transactions.

**Example: E-commerce Checkout**

**Option 1: Single Order Service**
\`\`\`
Order Service (handles full transaction):
1. Validate inventory
2. Reserve stock
3. Process payment
4. Create order
5. Send notification
→ All in single ACID transaction
\`\`\`

**Option 2: Separate Services**
\`\`\`
API Gateway → Inventory Service (reserve stock)
           → Payment Service (charge card)
           → Order Service (create order)
           → Notification Service (send email)
→ Requires distributed transaction (Saga pattern)
\`\`\`

**Trade-off**:
- **Option 1**: Simpler (ACID), but less flexibility
- **Option 2**: Independent scaling, but complex (eventual consistency)

**Rule**: If operations MUST be atomic and consistent, keep them in same service.

---

## Decomposition Strategy 5: By Scalability Needs

**Concept**: Extract services that have different scaling characteristics.

**Example: Social Media Platform**

**Different Scaling Needs**:
1. **Feed Service**: Read-heavy, 1M reads/second
2. **Post Service**: Write-heavy, 10K writes/second
3. **Profile Service**: Low traffic, 1K requests/second
4. **Notification Service**: Burst traffic, 100K/second at peak

**Architecture**:
\`\`\`
Feed Service:
- 50 read replicas
- Aggressive caching (Redis)
- CDN for popular feeds

Post Service:
- Write-optimized database (SSD)
- Message queue for async processing
- 10 write instances

Profile Service:
- 2 instances (low traffic)
- Simple database

Notification Service:
- Autoscaling (1-100 instances)
- Queue-based (SQS)
- Burst-friendly
\`\`\`

**Benefit**: Each service scaled independently. Don't over-provision Profile Service just because Feed Service needs 50 instances.

---

## Decomposition Strategy 6: By Team Ownership

**Concept**: Services match team structure (Conway's Law).

**Example: Company with 60 Engineers**

**Team Structure**:
- **Customer Experience Team** (12 engineers) → Frontend + User Service + Profile Service
- **Commerce Team** (15 engineers) → Product Service + Cart Service + Checkout Service
- **Logistics Team** (10 engineers) → Inventory Service + Shipping Service
- **Payments Team** (8 engineers) → Payment Service + Billing Service
- **Platform Team** (15 engineers) → API Gateway + Auth Service + Notification Service

**Benefits**:
- Clear ownership ("Payments team on-call for Payment Service")
- Independent deployment ("Commerce team doesn't block Customer Experience team")
- Team autonomy ("Logistics team chooses PostgreSQL, Payments team chooses DynamoDB")

**Anti-pattern**: Services that span teams
\`\`\`
❌ Order Service owned by:
- Commerce team (order creation)
- Payments team (payment logic)
- Logistics team (shipping logic)
→ Three teams must coordinate for every change
\`\`\`

---

## How to Identify Service Boundaries

### 1. Event Storming Workshop

**Process**:
1. Gather domain experts and engineers
2. Identify domain events ("OrderPlaced", "PaymentProcessed", "ItemShipped")
3. Group related events
4. Draw boundaries around groups
5. These boundaries → services

**Example Workshop for E-commerce**:

**Events Identified**:
- UserRegistered, UserLoggedIn, ProfileUpdated
- ProductAdded, ProductUpdated, ProductSearched
- ItemAddedToCart, CartCheckout
- OrderPlaced, OrderCancelled, OrderShipped
- PaymentAuthorized, PaymentCaptured, RefundProcessed
- InventoryReserved, StockReplenished
- EmailSent, SMSSent, PushNotificationSent

**Grouping**:
\`\`\`
[User Service]
- UserRegistered
- UserLoggedIn
- ProfileUpdated

[Product Service]
- ProductAdded
- ProductUpdated
- ProductSearched

[Cart Service]
- ItemAddedToCart
- CartCheckout

[Order Service]
- OrderPlaced
- OrderCancelled
- OrderShipped

[Payment Service]
- PaymentAuthorized
- PaymentCaptured
- RefundProcessed

[Inventory Service]
- InventoryReserved
- StockReplenished

[Notification Service]
- EmailSent
- SMSSent
- PushNotificationSent
\`\`\`

### 2. Analyze Data Ownership

**Look for**:
- Tables that are always joined together → Same service
- Tables that are never joined → Different services
- Foreign keys → Potential coupling

**Example**:
\`\`\`
Users table
Orders table (foreign key to Users)
OrderItems table (foreign key to Orders)
Products table (referenced by OrderItems)
\`\`\`

**Question**: Should Order and Product be in same service?

**Analysis**:
- Orders are frequently joined with Users → Keep User + Order together?
- Products are referenced by many orders, but product data rarely changes when order is placed
- Product has its own lifecycle (added by admin, updated by inventory team)

**Decision**:
- Order Service: Users, Orders, OrderItems
- Product Service: Products, Categories, ProductImages
- Order Service stores ProductID as reference (eventual consistency acceptable)

### 3. Look for Verb-Noun Patterns

**Verbs → Operations**
**Nouns → Entities**

**Example User Stories**:
- "Customer **views** product catalog" → ProductService.getProducts()
- "Customer **adds** item to cart" → CartService.addItem()
- "Customer **places** order" → OrderService.createOrder()
- "Admin **updates** inventory" → InventoryService.updateStock()
- "System **sends** notification" → NotificationService.send()

**Group by nouns**:
- Product-related: ProductService
- Cart-related: CartService
- Order-related: OrderService
- Inventory-related: InventoryService
- Notification-related: NotificationService

---

## Common Decomposition Mistakes

### Mistake 1: Too Granular (Nanoservices)

**Example**:
\`\`\`
❌ Over-decomposed:
- UserRegistrationService
- UserLoginService
- UserProfileService
- UserPasswordService
- UserEmailService
→ 5 services for "User" domain
\`\`\`

**Problems**:
- Network overhead for simple flows
- Distributed transaction complexity
- Operational burden (5 deployments, 5 monitors)

**Better**:
\`\`\`
✅ Single User Service:
- Handles registration, login, profile, password, email
→ Cohesive domain in one service
\`\`\`

**Rule**: If services always deploy together, they should be one service.

### Mistake 2: Not Granular Enough

**Example**:
\`\`\`
❌ Too coarse:
- EcommerceService (handles products, orders, payments, shipping, notifications)
→ Basically still a monolith
\`\`\`

**Problems**:
- Can't scale parts independently
- Can't deploy independently
- Multiple teams stepping on each other

**Better**: Decompose into separate services (Product, Order, Payment, Shipping, Notification)

### Mistake 3: Decomposition by Technical Layer

**Example**:
\`\`\`
❌ Technical layers:
- API Gateway Service
- Business Logic Service
- Data Access Service
- Database Service
\`\`\`

**Problem**: Every feature touches all layers. No independence.

**Better**: Vertical slices (each service has its own API + logic + data)

### Mistake 4: Shared Database

**Example**:
\`\`\`
❌ Shared database:
Order Service    Product Service    Inventory Service
        ↓              ↓                  ↓
        └──────── PostgreSQL ────────────┘
\`\`\`

**Problems**:
- Schema changes affect all services
- Can't deploy independently
- Database becomes coupling point
- Basically a distributed monolith

**Better**: Database per service
\`\`\`
✅ Database per service:
Order Service → PostgreSQL (orders)
Product Service → PostgreSQL (products)
Inventory Service → DynamoDB (inventory)
\`\`\`

### Mistake 5: Chatty Services

**Example**:
\`\`\`
API Gateway → Order Service
                ↓
              User Service (get user)
                ↓
              Product Service (get product details)
                ↓
              Inventory Service (check stock)
                ↓
              Pricing Service (get price)
                ↓
              Discount Service (get discount)
                ↓
              Tax Service (calculate tax)
→ 7 network calls for single order page
\`\`\`

**Impact**: Latency adds up (7 × 10ms = 70ms just in network)

**Solution 1: Aggregation Service**
\`\`\`
API Gateway → Order Aggregation Service
                ↓ (parallel calls)
              User, Product, Inventory, Pricing, Discount, Tax
                ↓
              Returns combined response
\`\`\`

**Solution 2: Denormalization**
\`\`\`
Order Service stores:
- ProductID (reference)
- ProductName (denormalized)
- Price (denormalized at order time)
→ Single call to Order Service
\`\`\`

---

## Decision Framework: When to Extract a Service

### Extract When:

✅ **Independent Scalability**: Component has very different scaling needs
   - Example: Video encoding service (CPU-heavy) vs API service (I/O-heavy)

✅ **Different Technology**: Component would benefit from different tech stack
   - Example: ML service needs Python, API service uses Go

✅ **Clear Domain Boundary**: Component has clear, stable interface
   - Example: Payment processing is self-contained

✅ **Different Release Cadence**: Component deploys much more/less frequently
   - Example: Experimental recommendation service (10 deploys/day) vs stable billing service (1 deploy/month)

✅ **Team Ownership**: Clear team owns this domain
   - Example: Payments team owns Payment Service

✅ **Security Isolation**: Component needs stricter security
   - Example: PCI-compliant payment service

### Don't Extract When:

❌ **Tight Coupling**: Components always change together
   - Keep in same service

❌ **ACID Transactions**: Operations need strong consistency
   - Distributed transactions are complex

❌ **No Clear Boundary**: Unclear where one ends and other begins
   - Wait until domain is clearer

❌ **Premature Optimization**: "We might need to scale this later"
   - Extract when you actually have the problem

❌ **Small Team**: < 10 engineers
   - Operational complexity not worth it

---

## Real-World Example: Spotify's Decomposition

**Evolution**:

**2010: Monolith** (10 engineers)
- Single Rails application
- All music features

**2012: First Services** (50 engineers)
- Extracted: Playlist Service, Social Service
- Reason: Different scaling needs

**2015: Domain Services** (200 engineers)
- User Service, Playlist Service, Social Service, Artist Service, Search Service, Player Service
- Reason: Team boundaries

**2020: 100+ Services** (1000+ engineers)
- Microservices aligned with "squads" (small teams)
- Each squad owns 1-3 services
- Services per domain: Playlists (5 services), Social (8 services), Discovery (10 services)

**Key Lesson**: Started simple, evolved as team grew and domains became clearer.

---

## Interview Tips

### Red Flags:
❌ "Create 20 microservices for everything"
❌ No justification for service boundaries
❌ Ignoring data dependencies
❌ Not considering team structure

### Good Responses:
✅ Explain domain-driven design
✅ Justify each service boundary
✅ Discuss data ownership
✅ Consider team organization
✅ Acknowledge trade-offs

### Sample Answer:
*"I would decompose this e-commerce system by business capability: Product, Order, Payment, Inventory, and Shipping services. These align with clear domain boundaries and likely team structure. Payment needs security isolation (PCI compliance), Inventory has high write load requiring different database optimization, and Shipping integrates with external APIs. I'd use event-driven architecture for cross-service communication to avoid tight coupling. Each service owns its data, using eventual consistency where acceptable to the business."*

---

## Key Takeaways

1. **Business capabilities** → best starting point for decomposition
2. **Domain-driven design** → bounded contexts become services
3. **Team structure matters** → services should align with teams (Conway's Law)
4. **Database per service** → true independence
5. **Avoid nanoservices** → balance granularity
6. **Extract strategically** → don't decompose everything
7. **Data dependencies** → biggest challenge in decomposition
8. **Start coarse, refine** → easier to split than merge

**Golden Rule**: Service boundaries should match domain boundaries, not technical implementation details.`,
      multipleChoice: [
        {
          id: 'mc-decomposition-1',
          question:
            "You're decomposing an e-commerce monolith. Which decomposition is BEST aligned with business capabilities?",
          options: [
            'Frontend Service, Backend Service, Database Service, Cache Service',
            'Product Service, Order Service, Payment Service, Shipping Service, Customer Service',
            'UserRegistrationService, UserLoginService, UserProfileService, UserPasswordService',
            'Read Service, Write Service, Analytics Service',
          ],
          correctAnswer: 1,
          explanation:
            'Option 2 decomposes by business capability (what the business does). Each service maps to a clear business function. Option 1 is technical decomposition (how, not what) - every feature would touch all layers. Option 3 is over-granular (nanoservices) - all User-related functions should be one service. Option 4 is technical (CQRS pattern) not business-driven. Business capability decomposition is recommended because it aligns with how business stakeholders think and typically matches organizational structure.',
        },
        {
          id: 'mc-decomposition-2',
          question: 'In Domain-Driven Design, what is a "bounded context"?',
          options: [
            'A security boundary that limits access to sensitive data',
            'A clear boundary within which a domain model is defined and consistent',
            'The maximum size (lines of code) that a microservice should have',
            'A transaction boundary where ACID properties are guaranteed',
          ],
          correctAnswer: 1,
          explanation:
            'A bounded context is a key DDD concept: it\'s a boundary within which a domain model is consistent and has specific meaning. For example, "Customer" in the Sales context (leads, opportunities) is different from "Customer" in the Billing context (payment methods, invoices). Each bounded context typically becomes a microservice. It\'s not about security (option 1), size limits (option 3), or transactions (option 4), but about semantic boundaries in the business domain.',
        },
        {
          id: 'mc-decomposition-3',
          question:
            'You have User, Order, and OrderItems tables that are frequently joined. Product table is referenced by OrderItems but rarely joined. How should you decompose?',
          options: [
            'One service for all tables (User, Order, OrderItems, Product)',
            'User Service (User table) and Order Service (Order, OrderItems, Product tables)',
            'Each table gets its own service (4 services)',
            'Order Service (User, Order, OrderItems) and Product Service (Product), with ProductID as reference in Orders',
          ],
          correctAnswer: 3,
          explanation:
            "Option 4 correctly recognizes that User/Order/OrderItems are tightly coupled (always accessed together) and should stay together, while Product has an independent lifecycle managed by a different team. Order Service stores ProductID as a reference, accepting eventual consistency (product details might change after order is placed, which is usually acceptable). Option 1 (monolithic) doesn't give microservices benefits. Option 2 incorrectly couples Product with Order. Option 3 (nanoservices) creates too much granularity and requires distributed joins.",
        },
        {
          id: 'mc-decomposition-4',
          question:
            'What is the main problem with this architecture: Order Service, Payment Service, and Inventory Service all connecting to the same PostgreSQL database?',
          options: [
            "PostgreSQL can't handle connections from multiple services",
            "It creates a distributed monolith - services can't deploy independently and are coupled through database",
            'It violates security best practices for payment data',
            'PostgreSQL is not suitable for microservices architecture',
          ],
          correctAnswer: 1,
          explanation:
            'Shared database is a critical anti-pattern in microservices. When services share a database: (1) schema changes affect all services, (2) services can\'t deploy independently, (3) services are coupled through database schemas/tables, (4) you lose technology flexibility per service. This creates a "distributed monolith" - microservices complexity without the benefits. The solution is "database per service" pattern where each service owns its data. Option 1 is factually wrong (PostgreSQL handles multiple connections fine). Option 3, while potentially true, isn\'t the main architectural problem. Option 4 is false - PostgreSQL works fine with microservices if each service has its own database.',
        },
        {
          id: 'mc-decomposition-5',
          question:
            'Your API Gateway calls Order Service, which calls User Service, which calls Product Service, which calls Inventory Service (4 sequential network calls). What pattern should you use to reduce latency?',
          options: [
            'Merge all services back into a monolith',
            'Use caching at every layer to avoid repeated calls',
            'Create an Order Aggregation Service that makes parallel calls to User, Product, and Inventory services',
            'Implement database replication to speed up queries',
          ],
          correctAnswer: 2,
          explanation:
            "Sequential network calls create latency (waterfall effect). An Aggregation Service (Backend for Frontend pattern) makes parallel calls and combines results, reducing total time. Example: instead of 4 sequential calls (40ms total), make 3 parallel calls from aggregation service (10ms total). Option 1 (merge to monolith) is too drastic. Option 2 (caching) helps but doesn't solve sequential dependency. Option 4 (database replication) doesn't address service-to-service communication latency. Aggregation services are common for mobile/web BFFs (Backend for Frontend) where UI needs data from multiple services.",
        },
      ],
      quiz: [
        {
          id: 'q1',
          question:
            "You're designing a food delivery platform (like DoorDash/Uber Eats). How would you decompose it into microservices? Justify your service boundaries.",
          sampleAnswer: `I would decompose the food delivery platform using a combination of business capability and domain-driven design approaches. Let me outline the services and rationale:

**Core Services:**

**1. Restaurant Service**
- Manages restaurant data (menu, hours, location, photos)
- Restaurant onboarding
- Menu management

*Justification*: Restaurants are a core domain with independent lifecycle. Restaurant team manages this data. Can scale independently (thousands of restaurants).

**2. Customer Service**
- Customer profiles, preferences
- Addresses, payment methods
- Order history

*Justification*: Clear bounded context around customer data. Privacy/security concerns warrant isolation. Different team (customer experience) owns this.

**3. Order Service (Orchestrator)**
- Order placement
- Order status tracking
- Order history

*Justification*: Order is the core business transaction. Orchestrates workflow between multiple services. Needs strong consistency for order state.

**4. Cart Service**
- Shopping cart management
- Cart items, modifications

*Justification*: Session-based, high traffic, can tolerate eventual consistency. Different scaling characteristics (many carts, fewer orders).

**5. Delivery Service**
- Driver management
- Real-time location tracking
- Driver assignment/dispatch
- Route optimization

*Justification*: Complex domain requiring real-time data, geospatial queries. Likely uses different tech (Go for performance, geospatial databases). Driver team owns this.

**6. Payment Service**
- Payment processing
- Refunds
- Billing

*Justification*: PCI compliance requires isolation. Critical service needing high reliability. Likely uses third-party (Stripe) requiring its own service boundary.

**7. Pricing Service**
- Dynamic pricing (surge pricing)
- Promotions, discounts
- Delivery fee calculation

*Justification*: Complex business logic, frequently changing. Experimentation-heavy (A/B tests). Business team wants independent deploy cycle.

**8. Notification Service**
- Push notifications, SMS, email
- Real-time order updates

*Justification*: Async by nature, can tolerate failures without affecting core flow. Generic capability used by all services.

**9. Search Service**
- Restaurant search
- Filtering, sorting
- Personalized recommendations

*Justification*: Different technology (Elasticsearch), different scaling needs (read-heavy), CPU-intensive. Can be down without breaking order flow.

**10. Analytics Service**
- Real-time dashboard
- Driver metrics, restaurant metrics
- Business intelligence

*Justification*: Async, read-only, different database (columnar for analytics). Batch processing use case.

**Service Communication Flow (Order Placement):**

\`\`\`
1. User adds items to cart
   → Cart Service (session-based, fast)

2. User clicks "Place Order"
   → API Gateway → Order Service (orchestrator)

3. Order Service (saga orchestration):
   a) Validate Cart
      → Cart Service: Get cart items
   
   b) Validate Restaurant
      → Restaurant Service: Check if open, items available
   
   c) Calculate Total
      → Pricing Service: Get delivery fee, apply promos
   
   d) Process Payment
      → Payment Service: Charge customer
   
   e) Assign Driver
      → Delivery Service: Find available driver
   
   f) Create Order
      → Order Service: Save order to DB
   
   g) Notify
      → Notification Service: Notify customer, restaurant, driver

4. If any step fails → Compensating transactions (saga pattern)
\`\`\`

**Data Ownership:**

Each service has its own database:
- Restaurant Service: PostgreSQL (structured menu data)
- Customer Service: PostgreSQL (GDPR compliance, relational)
- Order Service: PostgreSQL (transactions, consistency)
- Cart Service: Redis (session data, TTL)
- Delivery Service: MongoDB + PostGIS (geospatial)
- Payment Service: PostgreSQL (audit trail, ACID)
- Pricing Service: PostgreSQL
- Notification Service: DynamoDB (high write, eventually consistent)
- Search Service: Elasticsearch
- Analytics Service: ClickHouse (columnar)

**What I Would NOT Decompose:**

❌ Don't create separate services for:
- CustomerRegistration, CustomerLogin, CustomerProfile → Keep as one Customer Service
- OrderCreation, OrderCancellation, OrderHistory → Keep in Order Service
- RestaurantMenu, RestaurantHours, RestaurantLocation → Keep in Restaurant Service

**Trade-offs:**

**Complexity**: 10 services is complex but manageable. For a startup with 20 engineers, I'd start with 4-5 services (Customer, Restaurant, Order+Cart+Payment, Delivery) and extract later.

**Consistency**: Using Saga pattern for distributed transactions (order placement). Eventual consistency acceptable for analytics/notifications but not for payments.

**Latency**: Order placement requires multiple service calls. Mitigated by:
- Parallel calls where possible
- Caching (restaurant menus)
- Async for non-critical (notifications)

**Operational**: Requires mature DevOps (service discovery, distributed tracing, centralized logging). Investment justified for this domain.

**Key Insight**: Service boundaries follow domain boundaries (Restaurant, Customer, Delivery) not technical boundaries (Frontend, Backend, Database). This allows teams to own services end-to-end and deploy independently.`,
          keyPoints: [
            'Decompose by business capability (Restaurant, Customer, Order, Delivery)',
            'Each service has clear domain boundary and owner team',
            'Database per service (choose tech based on use case)',
            'Order Service orchestrates saga pattern for distributed transaction',
            'Separate services with different scaling needs (Search, Cart)',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain Domain-Driven Design\'s "bounded context" concept using a real-world example. How does it help with microservices decomposition?',
          sampleAnswer: `Domain-Driven Design's "bounded context" is one of the most important concepts for microservices decomposition. Let me explain with a concrete example.

**Bounded Context Definition:**

A bounded context is a boundary within which a particular domain model is defined and consistent. The same term (like "Customer" or "Product") can mean different things in different contexts.

**Real-World Example: Healthcare System**

Let's consider the term "Patient" across different contexts:

**1. Scheduling Context (Appointments)**
\`\`\`
Patient {
  patientId
  name
  phoneNumber
  appointmentHistory
  preferredDoctors
  noShowCount
}

Operations:
- scheduleAppointment()
- cancelAppointment()
- sendReminder()
\`\`\`

*Focus*: Patient as someone who books appointments. Cares about scheduling history and communication preferences.

**2. Clinical Context (Medical Records)**
\`\`\`
Patient {
  patientId
  name
  dateOfBirth
  medicalHistory
  allergies
  currentMedications
  diagnoses
  labResults
}

Operations:
- recordDiagnosis()
- prescribeMedication()
- orderTest()
- viewMedicalHistory()
\`\`\`

*Focus*: Patient as medical entity. Cares about clinical data and treatment.

**3. Billing Context**
\`\`\`
Patient {
  patientId
  name
  billingAddress
  insuranceProvider
  policyNumber
  outstandingBalance
  paymentHistory
}

Operations:
- processPayment()
- submitInsuranceClaim()
- generateInvoice()
- applyPaymentPlan()
\`\`\`

*Focus*: Patient as payer. Cares about financial data and payment methods.

**4. Emergency Context (ER Triage)**
\`\`\`
Patient {
  patientId
  name
  emergencyContact
  bloodType
  criticalAllergies
  currentLocation
  triagePriority
}

Operations:
- triagePatient()
- notifyEmergencyContact()
- checkCriticalConditions()
\`\`\`

*Focus*: Patient as emergency case. Cares only about critical, life-saving information.

**Key Insight:**

"Patient" is NOT a single, universal model. It means different things in different contexts:
- Scheduling cares about phone numbers, not blood type
- Clinical cares about medical history, not billing address
- Billing cares about insurance, not lab results
- Emergency cares about blood type, not appointment history

Each bounded context can have different:
- **Attributes**: Different fields are relevant
- **Operations**: Different business logic
- **Rules**: Different validation and constraints
- **Lifecycle**: Managed by different teams
- **Storage**: Different database schemas/types

**How Bounded Contexts Map to Microservices:**

**Each bounded context becomes a microservice:**

\`\`\`
Healthcare System Microservices:

1. Appointment Service (Scheduling Context)
   → PostgreSQL: patients, appointments, schedules

2. Medical Records Service (Clinical Context)
   → PostgreSQL (HIPAA-compliant): medical_history, diagnoses, prescriptions

3. Billing Service (Billing Context)
   → PostgreSQL: billing_accounts, invoices, payments

4. Emergency Service (Emergency Context)
   → Redis (fast access): patient_triage, emergency_contacts
\`\`\`

**Service Communication Example:**

When a patient arrives for appointment:

\`\`\`
Frontend → API Gateway

1. Check appointment
   → Appointment Service
   Returns: { appointment_id, doctor, time, patient_name }

2. Pull up medical record
   → Medical Records Service
   Returns: { medical_history, allergies, current_medications }

3. Check insurance
   → Billing Service
   Returns: { insurance_valid, copay_amount }
\`\`\`

Each service has its OWN representation of "patient" with only the data it needs.

**Benefits for Microservices:**

**1. Clear Boundaries**
- No ambiguity about what belongs where
- Medical Records Service clearly owns clinical data
- Billing Service clearly owns payment data

**2. Independent Evolution**
- Scheduling team can add "preferred appointment times" without affecting Billing
- Clinical team can add "genomic data" without affecting Scheduling

**3. Team Ownership**
- Clinical staff team → Medical Records Service
- Administrative staff team → Appointment Service
- Finance team → Billing Service

**4. Data Privacy**
- Emergency Service doesn't have access to billing data
- Appointment Service doesn't have access to full medical history
- Security boundaries match bounded contexts

**5. Different Technology Choices**
- Medical Records: PostgreSQL (HIPAA compliance, ACID)
- Emergency: Redis (millisecond response time)
- Appointment: PostgreSQL
- Analytics: Elasticsearch (search/reporting)

**Common Mistake: Shared "Patient" Database**

❌ **Anti-pattern: Shared Patient Table**
\`\`\`
patients table {
  patient_id
  name
  phone
  address
  date_of_birth
  blood_type
  allergies
  medical_history
  insurance_provider
  billing_address
  appointment_history
  ...
  (100+ columns)
}

All services → same patients table
\`\`\`

**Problems**:
- Schema changes affect all services
- Can't deploy independently
- Privacy nightmare (every service sees all data)
- Performance (100+ column table)
- Conflicting needs (Scheduling needs fast inserts, Billing needs complex joins)

✅ **Better: Database per Service with Denormalization**

\`\`\`
Appointment Service DB:
patients_scheduling {
  patient_id (reference)
  name (denormalized)
  phone (denormalized)
  email (denormalized)
  appointment_preferences
}

Medical Records Service DB:
patients_clinical {
  patient_id (reference)
  name (denormalized)
  date_of_birth (denormalized)
  full medical data...
}

Billing Service DB:
patients_billing {
  patient_id (reference)
  name (denormalized)
  billing addresses, payment methods...
}
\`\`\`

Each service has:
- Only the patient data it needs
- Denormalized for read performance
- Independent schema evolution

**Synchronization**: Use events
\`\`\`
Patient Registration Service (master):
1. Patient registers
2. Publish PatientRegistered event {patient_id, name, basic_info}
3. Each service subscribes and stores relevant data
\`\`\`

**How Bounded Contexts Help Decomposition:**

**Process:**
1. **Identify Domains**: What business capabilities exist? (Scheduling, Clinical, Billing, Emergency)

2. **Find Ubiquitous Language**: What terms does business use? (Patient, Appointment, Diagnosis, Invoice)

3. **Discover Multiple Meanings**: Does "Patient" mean same thing everywhere? (No!)

4. **Draw Boundaries**: Where does meaning change? → That's a bounded context boundary

5. **Map to Services**: Each bounded context → One microservice

**Interview Approach:**

When asked to design a complex system:

1. "I'll use Domain-Driven Design to identify bounded contexts"
2. "The term 'Customer' likely means different things in different parts of the system"
3. "In the Sales context, Customer is a lead. In Support context, Customer is a ticket owner. In Billing context, Customer is a payer."
4. "Each bounded context becomes a microservice with its own data model"
5. "Services communicate via events or API calls, with eventual consistency"

**Key Takeaways:**

1. Bounded contexts define where domain models are consistent
2. Same term (Patient, Customer, Product) can mean different things in different contexts
3. Each bounded context → microservice boundary
4. Prevents "God object" models (100-column Patient table)
5. Enables independent evolution and team ownership
6. Requires accepting some denormalization and eventual consistency

Bounded contexts are the most principled way to decompose systems because they follow natural business boundaries, not technical convenience.`,
          keyPoints: [
            'Bounded context = boundary where domain model is consistent',
            'Same term (Patient, Customer) means different things in different contexts',
            'Each bounded context becomes a microservice',
            'Services have their own representation of shared entities',
            'Enables clear ownership and independent evolution',
          ],
        },
        {
          id: 'q3',
          question:
            'You have a monolithic e-commerce application. Walk me through how you would identify the first service to extract. What factors would you consider?',
          sampleAnswer: `Extracting the first service from a monolith is critical because it sets precedent and teaches lessons for future extractions. Let me walk through a systematic approach.

**Step 1: Analyze Current Monolith (1-2 weeks)**

First, understand what you have:

**A. Map the domains:**
\`\`\`
E-commerce Monolith Contains:
- User management (registration, login, profiles)
- Product catalog (products, categories, search)
- Shopping cart
- Order processing
- Payment processing
- Inventory management
- Shipping/fulfillment
- Customer service (returns, support tickets)
- Reviews and ratings
- Recommendations
- Analytics/reporting
- Email notifications
\`\`\`

**B. Analyze coupling:**

Create a dependency matrix:
\`\`\`
           User Product Cart Order Payment Inventory Ship Notify
User       -    Low    Med  High  Med     Low       Low  High
Product    Low  -      High High  Low     High      Low  Med
Cart       Med  High   -    High  Low     High      Low  Med
Order      High High   High -     High    High      High High
Payment    Med  Low    Low  High  -       Low       Low  High
Inventory  Low  High   High High  Low     -         High Low
Ship       Low  Low    Low  High  Low     High      -    High
Notify     High Med    Med  High  High    Low       High -
\`\`\`

**"Order" is highly coupled → Extract LAST, not first**

**C. Identify pain points:**

Survey engineering team:
- What's slow to deploy?
- What causes production issues?
- What's hard to scale?
- What requires different tech?

Example survey results:
\`\`\`
Notifications: 
  - Slow deploys (not critical path)
  - Currently synchronous (blocks checkout)
  - Could use async queue
  
Search:
  - CPU-intensive
  - Would benefit from Elasticsearch
  - Currently slows down application servers
  
Payment:
  - Regulatory compliance (PCI)
  - Team wants to isolate for security
  - Rarely changes
\`\`\`

**Step 2: Score Service Candidates**

Use a scoring rubric (1-5 scale):

\`\`\`
Candidate: Notification Service
✅ Low coupling (5/5): Used by others but doesn't depend on much
✅ Clear boundary (5/5): Well-defined interface (send notification)
✅ Low risk (5/5): Can be down without breaking checkout
✅ Async friendly (5/5): Naturally asynchronous
❌ Business value (2/5): Doesn't directly generate revenue
✅ Learning value (4/5): Good first service to learn from
✅ Team ready (4/5): Team comfortable with this domain

TOTAL: 30/35

Candidate: Payment Service
✅ Low coupling (4/5): Interfaces with few services
✅ Clear boundary (5/5): Well-defined payment operations
❌ Risk (2/5): If broken, customers can't buy
✅ Business critical (5/5): Directly impacts revenue
❌ Complexity (2/5): PCI compliance, high stakes
✅ Security benefit (5/5): Isolation reduces compliance scope
❌ Team experience (3/5): Team hasn't built microservices before

TOTAL: 26/35

Candidate: Search Service
✅ Low coupling (4/5): Reads product data, that's it
✅ Clear boundary (5/5): Search API is well-defined
✅ Low risk (4/5): Fallback to simple search if down
✅ Tech benefit (5/5): Elasticsearch would be huge improvement
✅ Performance (5/5): Currently a bottleneck
✅ Learning value (3/5): Involves new tech (Elasticsearch)
❌ Data sync (3/5): Need to keep product index updated

TOTAL: 29/35

Candidate: Order Service
❌ Coupling (1/5): Touches everything
❌ Clear boundary (2/5): Orchestrates across many domains
❌ Risk (1/5): Core business logic
❌ Transactions (1/5): Requires distributed transactions
❌ Data (1/5): Complex data relationships

TOTAL: 6/35
\`\`\`

**Recommendation: Start with Notification Service or Search Service**

**Step 3: Detailed Plan for First Extraction (Notification Service)**

**Why Notification Service is ideal first extraction:**

✅ **Low Risk**: If broken, core business (ordering) still works
✅ **Async Nature**: Already should be async, easy to queue
✅ **Low Coupling**: Receives events, doesn't call other services
✅ **Clear Interface**: sendEmail(), sendSMS(), sendPush()
✅ **Learning Opportunity**: Safe place to learn microservices patterns
✅ **Quick Win**: Can extract in 2-4 weeks

**Phase 1: Preparation (Week 1)**

1. **Build platform infrastructure:**
   - Set up service discovery (Consul/Eureka)
   - Message queue (RabbitMQ/SQS)
   - Monitoring (Prometheus + Grafana)
   - Distributed tracing (Jaeger)
   - Centralized logging (ELK/Splunk)

2. **Define service contract:**
\`\`\`
Notification Service API:

POST /notifications/email
{
  "recipient": "user@example.com",
  "template": "order_confirmation",
  "data": { "order_id": "12345", ... }
}

POST /notifications/sms
POST /notifications/push

Event Subscriptions:
- OrderPlaced → Send order confirmation
- OrderShipped → Send shipping notification
- UserRegistered → Send welcome email
\`\`\`

3. **Set up new service skeleton:**
   - Repository
   - CI/CD pipeline
   - Database (notification history)
   - Monitoring dashboards

**Phase 2: Parallel Run (Weeks 2-3)**

1. **Implement Notification Service:**
   - Build service
   - Integrate with SendGrid/Twilio
   - Store notification history

2. **Dual-write from monolith:**
\`\`\`
Monolith:
  orderService.placeOrder() {
    // ... order logic ...
    
    // OLD: Send email directly
    emailService.send(...)
    
    // NEW: Also publish event
    messageQueue.publish('OrderPlaced', {...})
    
    // Feature flag: if (useNewService)
  }

Notification Service:
  Subscribes to 'OrderPlaced' event
  Sends email
\`\`\`

3. **Shadow mode:**
   - Both monolith and new service send notifications
   - Compare results
   - Fix discrepancies
   - Monitor errors

**Phase 3: Cutover (Week 4)**

1. **Feature flag to route % of traffic:**
   - 10% → Notification Service
   - Monitor for errors
   - 50% → Notification Service
   - 100% → Notification Service

2. **Remove notification code from monolith:**
   - Monolith only publishes events
   - Doesn't send emails directly anymore

3. **Celebrate!** 🎉 First service extracted

**Phase 4: Lessons Learned (Week 5)**

Document learnings:
- What went well?
- What was harder than expected?
- Platform gaps identified?
- Team skill gaps?
- Next service easier or harder?

**What I Would NOT Extract First:**

❌ **Order Service**: Too coupled, core business logic, requires distributed transactions
❌ **User Service**: Everything depends on it, schema changes affect all
❌ **Payment Service**: Too risky for first extraction (though high value)
❌ **Product Service**: Too many dependencies, complex data model

**Alternative First Service: Search Service**

If Notification Service doesn't exist or isn't a bottleneck, Search would be my second choice:

**Why:**
- Clear benefit (Elasticsearch much better than SQL LIKE)
- Read-only (doesn't write to critical data)
- Can fail without breaking orders
- Independent scaling needs

**Approach:**
1. Set up Elasticsearch
2. Index product data
3. Keep SQL search as fallback
4. Route 10% of searches to Elasticsearch
5. Gradually increase to 100%
6. Remove SQL search

**Decision Framework Summary:**

**Extract First When Service Has:**
1. ✅ Low coupling (few dependencies)
2. ✅ Clear boundary (well-defined interface)
3. ✅ Low risk (failure doesn't break core business)
4. ✅ Async friendly (or read-only)
5. ✅ Quick extraction (2-4 weeks)
6. ✅ Learning value (teaches patterns for next services)

**Don't Extract First When Service:**
1. ❌ Highly coupled (touches everything)
2. ❌ Core business logic (high risk)
3. ❌ Requires distributed transactions
4. ❌ Complex data relationships
5. ❌ Team not ready for complexity

**Interview Tips:**

When asked about service extraction:
1. "I'd start by mapping domains and analyzing coupling"
2. "I'd identify low-risk, low-coupling candidates"
3. "I'd use Strangler Fig pattern with parallel run"
4. "I'd extract Notification or Search first to learn"
5. "I'd save high-coupling services like Order for last"
6. "I'd measure success and iterate"

**Key Insight**: First service teaches you how to do microservices. Choose something safe where you can learn without risking the business. Don't boil the ocean - extract incrementally.`,
          keyPoints: [
            'Analyze monolith to map domains and understand coupling',
            'Score candidates on: coupling, boundary clarity, risk, business value',
            'Choose low-risk service for first extraction (Notification, Search)',
            'Use Strangler Fig pattern with parallel run and gradual cutover',
            'Build platform infrastructure first (discovery, queue, monitoring)',
          ],
        },
      ],
    },
    {
      id: 'inter-service-communication',
      title: 'Inter-Service Communication',
      content: `One of the biggest challenges in microservices is how services communicate with each other. The choices you make here fundamentally impact performance, reliability, and complexity.

## Communication Patterns Overview

Microservices can communicate in two fundamental ways:

**1. Synchronous Communication**
- Request/response
- Client waits for response
- Examples: HTTP/REST, gRPC

**2. Asynchronous Communication**
- Fire-and-forget or publish/subscribe
- Client doesn't wait
- Examples: Message queues, event streaming

---

## Synchronous Communication

### HTTP/REST

**Most common pattern** for microservices communication.

**How it works**:
\`\`\`
Order Service ---HTTP POST---> Payment Service
              <--- Response ---
\`\`\`

**Advantages**:
✅ Simple and intuitive
✅ Wide tooling support
✅ Easy to debug (request/response visible)
✅ Strong typing with OpenAPI/Swagger
✅ Immediate feedback

**Disadvantages**:
❌ Tight coupling (Order Service blocks waiting for Payment)
❌ Cascading failures (if Payment down, Order fails)
❌ Synchronous chain latency adds up
❌ Less resilient to network issues

**Example: Order Checkout**
\`\`\`javascript
// Order Service
async function createOrder(orderData) {
  // 1. Validate inventory (synchronous call)
  const inventory = await inventoryService.checkStock(orderData.items);
  if (!inventory.available) {
    throw new Error('Out of stock');
  }
  
  // 2. Process payment (synchronous call)
  const payment = await paymentService.charge({
    amount: orderData.total,
    customerId: orderData.customerId
  });
  
  if (!payment.success) {
    throw new Error('Payment failed');
  }
  
  // 3. Create order
  const order = await db.orders.create(orderData);
  
  // 4. Send notification (synchronous call)
  await notificationService.sendEmail({
    to: orderData.customerEmail,
    template: 'order_confirmation',
    data: order
  });
  
  return order;
}
\`\`\`

**Problems with this approach**:
- If notification service is down, entire order creation fails
- If payment service is slow, user waits
- If any service times out, need retry logic
- Network latency: 3 services × 10ms = 30ms minimum

**When to use REST**:
✅ Need immediate response (user waiting)
✅ Simple request/response
✅ CRUD operations
✅ Low latency requirements
✅ Small number of services

---

### gRPC

**Modern RPC framework** using HTTP/2 and Protocol Buffers.

**Advantages over REST**:
✅ **Performance**: Binary protocol (protobuf) faster than JSON
✅ **Streaming**: Bidirectional streaming support
✅ **Type safety**: Strong schema enforcement
✅ **Efficient**: HTTP/2 multiplexing

**Disadvantages**:
❌ Less human-readable (binary)
❌ Steeper learning curve
❌ Browser support limited
❌ Debugging harder

**Example Service Definition**:
\`\`\`protobuf
// payment.proto
service PaymentService {
  rpc ProcessPayment(PaymentRequest) returns (PaymentResponse);
  rpc GetPaymentStatus(PaymentId) returns (PaymentStatus);
  rpc RefundPayment(RefundRequest) returns (RefundResponse);
}

message PaymentRequest {
  string customer_id = 1;
  double amount = 2;
  string currency = 3;
  string payment_method = 4;
}

message PaymentResponse {
  string payment_id = 1;
  PaymentStatus status = 2;
  string message = 3;
}
\`\`\`

**When to use gRPC**:
✅ Internal microservices (not public API)
✅ High-performance requirements
✅ Streaming data (real-time updates)
✅ Strong typing needed
✅ Polyglot environments (generate client libraries)

**Real-world**: Google uses gRPC extensively for internal services.

---

## Asynchronous Communication

### Message Queues (Point-to-Point)

**Pattern**: Producer sends message to queue, single consumer processes it.

\`\`\`
Producer → [Queue] → Consumer
\`\`\`

**Example: Order Processing**
\`\`\`javascript
// Order Service (Producer)
async function createOrder(orderData) {
  // 1. Create order immediately
  const order = await db.orders.create(orderData);
  
  // 2. Publish to queue (non-blocking)
  await queue.publish('order.created', {
    orderId: order.id,
    customerId: order.customerId,
    items: order.items,
    total: order.total
  });
  
  // 3. Return immediately (don't wait for processing)
  return order;
}

// Payment Service (Consumer)
queue.subscribe('order.created', async (message) => {
  try {
    await processPayment(message.orderId, message.total);
  } catch (error) {
    // Retry logic handled by queue
    throw error;
  }
});
\`\`\`

**Advantages**:
✅ **Decoupling**: Producer doesn't know/care about consumer
✅ **Async**: Producer doesn't wait
✅ **Reliability**: Message persisted, retries handled
✅ **Load leveling**: Consumers process at their own pace
✅ **Fault tolerance**: If consumer down, messages queued

**Disadvantages**:
❌ **No immediate response**: Producer doesn't know result
❌ **Complexity**: Need to handle eventual consistency
❌ **Debugging**: Harder to trace async flows
❌ **Message ordering**: Can be challenging
❌ **Duplicate processing**: Need idempotency

**Technologies**: RabbitMQ, AWS SQS, Azure Service Bus

**When to use**:
✅ Operations can be async (emails, notifications)
✅ Load leveling needed (burst traffic)
✅ Retry logic important
✅ Decoupling services

---

### Pub/Sub (Publish/Subscribe)

**Pattern**: Producer publishes event, multiple consumers can subscribe.

\`\`\`
Producer → [Topic] → Consumer 1
                   → Consumer 2
                   → Consumer 3
\`\`\`

**Example: Order Placed Event**
\`\`\`javascript
// Order Service (Publisher)
async function createOrder(orderData) {
  const order = await db.orders.create(orderData);
  
  // Publish event to topic (multiple subscribers)
  await eventBus.publish('order.placed', {
    orderId: order.id,
    customerId: order.customerId,
    items: order.items,
    total: order.total,
    timestamp: new Date()
  });
  
  return order;
}

// Payment Service (Subscriber 1)
eventBus.subscribe('order.placed', async (event) => {
  await processPayment(event.orderId, event.total);
});

// Inventory Service (Subscriber 2)
eventBus.subscribe('order.placed', async (event) => {
  await reserveInventory(event.items);
});

// Notification Service (Subscriber 3)
eventBus.subscribe('order.placed', async (event) => {
  await sendOrderConfirmation(event.customerId, event.orderId);
});

// Analytics Service (Subscriber 4)
eventBus.subscribe('order.placed', async (event) => {
  await trackOrderEvent(event);
});
\`\`\`

**Advantages**:
✅ **Loose coupling**: Publisher doesn't know subscribers
✅ **Extensibility**: Add new subscribers without changing publisher
✅ **Parallel processing**: All subscribers process simultaneously
✅ **Event-driven architecture**: Natural fit for domain events

**Disadvantages**:
❌ **No response**: Publisher doesn't know if subscribers succeeded
❌ **Eventual consistency**: Data may be temporarily inconsistent
❌ **Debugging**: Hard to trace event flows
❌ **Message ordering**: Subscribers may process out of order

**Technologies**: Apache Kafka, AWS SNS, Google Pub/Sub, RabbitMQ (with topic exchanges)

**When to use**:
✅ Multiple services need same event
✅ Event-driven architecture
✅ Auditability (event log)
✅ Decoupling critical

---

## Hybrid Pattern: Synchronous + Asynchronous

**Best practice**: Mix both patterns based on requirements.

**Example: E-commerce Checkout**
\`\`\`javascript
async function checkout(orderData) {
  // SYNCHRONOUS: Need immediate result
  // 1. Validate inventory (must succeed before proceeding)
  const inventoryCheck = await inventoryService.reserve(orderData.items);
  if (!inventoryCheck.success) {
    return { error: 'Out of stock' };
  }
  
  // 2. Process payment (must succeed before confirming)
  const payment = await paymentService.charge(orderData.amount);
  if (!payment.success) {
    // Compensate: release inventory
    await inventoryService.release(orderData.items);
    return { error: 'Payment failed' };
  }
  
  // 3. Create order
  const order = await db.orders.create({
    ...orderData,
    status: 'confirmed',
    paymentId: payment.id
  });
  
  // ASYNCHRONOUS: Fire and forget
  // 4. Send notification (don't wait)
  await queue.publish('order.confirmed', order);
  
  // 5. Update analytics (don't wait)
  await eventBus.publish('order.placed', order);
  
  // 6. Trigger fulfillment (don't wait)
  await queue.publish('order.fulfillment', order);
  
  // Return immediately to user
  return { success: true, orderId: order.id };
}
\`\`\`

**Decision Tree**:
\`\`\`
Does user need immediate response?
├─ Yes → Synchronous (REST/gRPC)
└─ No → Asynchronous (Queue/Event)

Is operation critical to request success?
├─ Yes → Synchronous (payment, inventory check)
└─ No → Asynchronous (notification, analytics)

Do multiple services need this data?
├─ Yes → Pub/Sub (event)
└─ No → Queue (single consumer)
\`\`\`

---

## Service Mesh

**Problem**: With many microservices, managing communication becomes complex:
- Service discovery
- Load balancing
- Retries and timeouts
- Circuit breakers
- Mutual TLS
- Distributed tracing

**Solution: Service Mesh** (like Istio, Linkerd)

**Architecture**:
\`\`\`
Service A → [Sidecar Proxy A] → [Sidecar Proxy B] → Service B
               ↓                      ↓
           Control Plane (manages all proxies)
\`\`\`

**Key Features**:

**1. Traffic Management**
- Load balancing (round robin, least request)
- Retry logic (automatic retries)
- Timeouts (prevent hanging requests)
- Circuit breaking (fail fast when service unhealthy)

**2. Security**
- Mutual TLS (mTLS) - automatic encryption
- Authentication/authorization between services
- Certificate management

**3. Observability**
- Automatic distributed tracing
- Metrics collection (latency, error rate)
- Service topology mapping

**Example: Istio Configuration**
\`\`\`yaml
# Retry configuration
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: payment-service
spec:
  hosts:
  - payment-service
  http:
  - route:
    - destination:
        host: payment-service
    retries:
      attempts: 3
      perTryTimeout: 2s
      retryOn: 5xx,reset,connect-failure
    timeout: 10s
\`\`\`

**When to use Service Mesh**:
✅ Many microservices (> 10)
✅ Need sophisticated traffic management
✅ Security requirements (mTLS)
✅ Polyglot environment
✅ Using Kubernetes

**When NOT to use**:
❌ Few services (< 5) - overhead not worth it
❌ Team lacks operational maturity
❌ Simple communication patterns

---

## API Contracts and Backward Compatibility

**Problem**: Service A depends on Service B's API. Service B changes API. Service A breaks.

**Solution: Versioning and Backward Compatibility**

### Strategy 1: URL Versioning
\`\`\`
/v1/payments
/v2/payments
\`\`\`

**Pros**: Clear, explicit
**Cons**: Must maintain multiple versions

### Strategy 2: Header Versioning
\`\`\`
GET /payments
Header: API-Version: 2
\`\`\`

**Pros**: Single endpoint
**Cons**: Less visible, harder to route

### Strategy 3: Backward-Compatible Changes
\`\`\`javascript
// Version 1: Original API
{
  "amount": 100.00,
  "currency": "USD"
}

// Version 2: Add fields (backward compatible)
{
  "amount": 100.00,
  "currency": "USD",
  "tax": 10.00,      // NEW: optional field
  "discount": 5.00   // NEW: optional field
}

// Consumers on v1 API ignore new fields (works fine)
// Consumers on v2 API can use new fields
\`\`\`

**Backward-Compatible Changes**:
✅ Add optional fields
✅ Add new endpoints
✅ Deprecate (but don't remove) fields
✅ Make required fields optional

**Breaking Changes** (require versioning):
❌ Remove fields
❌ Rename fields
❌ Change field types
❌ Change validation rules
❌ Remove endpoints

---

## Communication Anti-Patterns

### Anti-Pattern 1: Chatty Services
\`\`\`
❌ API Gateway → User Service
               → Product Service (needs user data)
                 → User Service (again!)
                   → Order Service (needs user data)
                     → User Service (again!)
\`\`\`

**Problem**: 3 calls to User Service for same data. Network overhead.

**Solution**: Pass data, don't fetch repeatedly
\`\`\`
✅ API Gateway → User Service (get user)
               → Product Service (pass user data)
                 → Order Service (pass user data)
\`\`\`

Or use API Gateway to aggregate:
\`\`\`
✅ API Gateway → User, Product, Order (parallel calls)
               → Aggregate response
               → Return to client
\`\`\`

### Anti-Pattern 2: Synchronous Chains
\`\`\`
❌ Service A → Service B → Service C → Service D
\`\`\`

**Problem**: 
- If any service down, entire chain fails
- Latency adds up (4 × 10ms = 40ms)
- Tight coupling

**Solution**: Use events
\`\`\`
✅ Service A → Publish Event
             ↓
    [Event Bus]
             ↓ (parallel subscriptions)
    Service B, Service C, Service D
\`\`\`

### Anti-Pattern 3: Distributed Monolith via Communication
\`\`\`
❌ Every operation requires calling 5+ services
   Services can't operate independently
   Synchronous coupling everywhere
\`\`\`

**Solution**: Redesign service boundaries (see Section 2)

---

## Real-World Example: Netflix

**Netflix Communication Patterns**:

1. **User requests video**
   - API Gateway → User Service (REST)
   - API Gateway → Video Service (REST)
   - Response to user (synchronous)

2. **Video playback starts**
   - Video Service → Publish event: "PlaybackStarted"
   - Analytics Service subscribes (for reporting)
   - Recommendation Service subscribes (for ML)
   - Billing Service subscribes (for usage tracking)
   (asynchronous, pub/sub)

3. **Encoding new video**
   - Upload Service → Queue: "VideoEncoding"
   - Encoding Workers pull from queue (async, load leveling)
   - Progress updates via events

**Key Insight**: Netflix uses BOTH sync and async based on requirements.

---

## Decision Framework

**Use Synchronous (REST/gRPC) When**:
✅ Need immediate response
✅ User is waiting
✅ Operation must complete before proceeding
✅ Simple request/response
✅ Low latency critical

**Use Asynchronous (Queue) When**:
✅ Operation can be delayed
✅ Load leveling needed
✅ Retry logic important
✅ Single consumer

**Use Pub/Sub (Events) When**:
✅ Multiple services need same data
✅ Audit trail needed
✅ Event-driven architecture
✅ Loose coupling critical

**Use Service Mesh When**:
✅ Many services (> 10)
✅ Need traffic management (retries, circuit breakers)
✅ Security requirements (mTLS)
✅ Kubernetes-based

---

## Interview Tips

### Red Flags:
❌ "Always use REST" or "Always use events"
❌ No mention of trade-offs
❌ Ignoring latency implications
❌ Not discussing failure scenarios

### Good Responses:
✅ Explain both sync and async patterns
✅ Justify choice based on requirements
✅ Discuss retry logic and failure handling
✅ Mention backward compatibility
✅ Reference real-world examples (Netflix, Uber)

### Sample Answer:
*"For the checkout flow, I'd use synchronous REST calls for inventory check and payment processing since we need immediate results and these are critical for order confirmation. However, for notifications, analytics, and fulfillment, I'd use asynchronous message queues since users don't need to wait for these to complete. I'd implement retry logic with exponential backoff for async operations and circuit breakers for synchronous calls to prevent cascading failures. For service-to-service authentication, I'd use mutual TLS via a service mesh like Istio."*

---

## Key Takeaways

1. **No one-size-fits-all**: Mix sync and async based on requirements
2. **Synchronous**: Simple but tight coupling, use when response needed
3. **Asynchronous**: Complex but resilient, use when operations can be delayed
4. **Pub/Sub**: Best for events that multiple services need
5. **Service Mesh**: Solves cross-cutting concerns (security, observability, traffic management)
6. **Backward compatibility**: Critical for independent deployment
7. **Avoid chatty services**: Minimize network calls
8. **Design for failure**: Timeouts, retries, circuit breakers essential

The choice of communication pattern fundamentally impacts your microservices architecture's performance, reliability, and operational complexity.`,
      quiz: [
        {
          id: 'q1-communication',
          question:
            'Design the communication strategy for an e-commerce checkout flow involving Order Service, Inventory Service, Payment Service, and Notification Service. Explain which communications should be synchronous vs asynchronous, how you would handle failures, implement idempotency, and ensure data consistency. Include specific examples of message formats and error handling.',
          sampleAnswer: `**E-Commerce Checkout Communication Strategy**

**1. Service Flow Overview**

\`\`\`
User Checkout
    ↓
Order Service (orchestrator)
    ├→ Inventory Service (sync)
    ├→ Payment Service (sync)
    ├→ Fulfillment Service (async)
    ├→ Notification Service (async)
    └→ Analytics Service (async)
\`\`\`

**2. Synchronous Communication**

**Inventory Check** (Synchronous - REST/gRPC):

\`\`\`javascript
// Order Service → Inventory Service
async function reserveInventory(orderId, items) {
  try {
    const response = await httpClient.post(
      'http://inventory-service/api/reserve',
      {
        orderId: orderId,
        items: items.map(item => ({
          productId: item.productId,
          quantity: item.quantity
        })),
        reservationTimeout: 300 // 5 minutes
      },
      {
        timeout: 2000, // 2 second timeout
        headers: {
          'X-Request-ID': generateRequestId(),
          'X-Idempotency-Key': \`reserve-\${orderId}\`
        }
      }
    );
    
    return {
      success: true,
      reservationId: response.data.reservationId
    };
    
  } catch (error) {
    if (error.code === 'OUT_OF_STOCK') {
      throw new OutOfStockError(error.message);
    } else if (error.code === 'TIMEOUT') {
      // Retry once
      return await reserveInventory(orderId, items);
    }
    throw error;
  }
}
\`\`\`

**Why Synchronous**: Need immediate confirmation before proceeding to payment.

**Payment Processing** (Synchronous - REST):

\`\`\`javascript
async function processPayment(orderId, paymentDetails) {
  const idempotencyKey = \`payment-\${orderId}\`;
  
  try {
    const response = await httpClient.post(
      'http://payment-service/api/charge',
      {
        orderId: orderId,
        amount: paymentDetails.amount,
        currency: 'USD',
        paymentMethod: paymentDetails.paymentMethodId
      },
      {
        timeout: 10000, // 10 seconds (payment can take time)
        headers: {
          'X-Request-ID': generateRequestId(),
          'X-Idempotency-Key': idempotencyKey
        }
      }
    );
    
    return {
      success: true,
      transactionId: response.data.transactionId
    };
    
  } catch (error) {
    if (error.code === 'PAYMENT_DECLINED') {
      // Release inventory reservation
      await releaseInventory(orderId);
      throw new PaymentDeclinedError();
    }
    throw error;
  }
}
\`\`\`

**Why Synchronous**: User must know immediately if payment succeeded.

**3. Asynchronous Communication**

**Order Confirmation** (Async - Message Queue):

\`\`\`javascript
// Order Service publishes event
async function publishOrderCreatedEvent(order) {
  const event = {
    eventId: generateUUID(),
    eventType: 'OrderCreated',
    timestamp: new Date().toISOString(),
    aggregateId: order.id,
    version: 1,
    data: {
      orderId: order.id,
      userId: order.userId,
      items: order.items,
      totalAmount: order.totalAmount,
      paymentTransactionId: order.transactionId
    }
  };
  
  await messageQueue.publish('orders.created', event, {
    messageId: event.eventId,
    persistent: true
  });
}

// Notification Service subscribes
messageQueue.subscribe('orders.created', async (message) => {
  const order = message.data;
  
  // Send email
  await emailService.send({
    to: order.userEmail,
    subject: 'Order Confirmation',
    body: \`Your order #\${order.orderId} has been confirmed\`
  });
  
  // Acknowledge message
  message.ack();
});
\`\`\`

**Why Asynchronous**: User doesn't need to wait for email to be sent.

**4. Complete Checkout Implementation**

\`\`\`javascript
class OrderService {
  async createOrder(userId, items, paymentDetails) {
    const orderId = generateOrderId();
    let reservationId = null;
    let transactionId = null;
    
    try {
      // Step 1: Reserve inventory (SYNC)
      console.log('Reserving inventory...');
      const inventoryResult = await this.reserveInventory(orderId, items);
      reservationId = inventoryResult.reservationId;
      
      // Step 2: Process payment (SYNC)
      console.log('Processing payment...');
      const paymentResult = await this.processPayment(orderId, paymentDetails);
      transactionId = paymentResult.transactionId;
      
      // Step 3: Confirm inventory (SYNC)
      console.log('Confirming inventory...');
      await this.confirmInventory(reservationId);
      
      // Step 4: Create order in database
      const order = await this.orderRepository.create({
        id: orderId,
        userId: userId,
        items: items,
        status: 'CONFIRMED',
        paymentTransactionId: transactionId,
        createdAt: new Date()
      });
      
      // Step 5: Publish events (ASYNC)
      await this.publishOrderCreatedEvent(order);
      
      return {
        success: true,
        orderId: orderId,
        message: 'Order created successfully'
      };
      
    } catch (error) {
      // Compensating transactions
      console.error('Order creation failed:', error);
      
      if (reservationId && !transactionId) {
        // Payment failed - release inventory
        await this.releaseInventory(reservationId);
      }
      
      if (transactionId) {
        // Rare case: payment succeeded but confirmation failed
        // Mark for manual review
        await this.createManualReviewTask({
          orderId,
          transactionId,
          reservationId,
          error: error.message
        });
      }
      
      throw error;
    }
  }
}
\`\`\`

**5. Idempotency Implementation**

\`\`\`javascript
// Idempotency middleware
async function idempotencyMiddleware(req, res, next) {
  const idempotencyKey = req.headers['x-idempotency-key'];
  
  if (!idempotencyKey) {
    return res.status(400).json({ error: 'Idempotency-Key header required' });
  }
  
  // Check if we've seen this key before
  const cached = await redis.get(\`idempotency:\${idempotencyKey}\`);
  
  if (cached) {
    // Return cached response
    return res.status(200).json(JSON.parse(cached));
  }
  
  // Store original response
  const originalSend = res.send;
  res.send = function(data) {
    // Cache for 24 hours
    redis.setex(\`idempotency:\${idempotencyKey}\`, 86400, data);
    originalSend.call(this, data);
  };
  
  next();
}

app.post('/api/orders', idempotencyMiddleware, async (req, res) => {
  // Order creation logic
});
\`\`\`

**6. Error Handling Strategy**

**Retry Logic with Exponential Backoff**:

\`\`\`javascript
async function retryWithBackoff(fn, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      
      // Retry on transient errors
      if (['TIMEOUT', 'CONNECTION_ERROR', 'SERVICE_UNAVAILABLE'].includes(error.code)) {
        const delay = Math.pow(2, i) * 1000; // 1s, 2s, 4s
        await sleep(delay);
        continue;
      }
      
      // Don't retry on client errors
      throw error;
    }
  }
}
\`\`\`

**Circuit Breaker**:

\`\`\`javascript
class CircuitBreaker {
  constructor(threshold = 5, timeout = 60000) {
    this.state = 'CLOSED';
    this.failureCount = 0;
    this.threshold = threshold;
    this.timeout = timeout;
  }
  
  async execute(fn) {
    if (this.state === 'OPEN') {
      throw new Error('Circuit breaker is OPEN');
    }
    
    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }
  
  onSuccess() {
    this.failureCount = 0;
    this.state = 'CLOSED';
  }
  
  onFailure() {
    this.failureCount++;
    if (this.failureCount >= this.threshold) {
      this.state = 'OPEN';
      setTimeout(() => {
        this.state = 'HALF_OPEN';
        this.failureCount = 0;
      }, this.timeout);
    }
  }
}
\`\`\`

**7. Data Consistency**

**Event-Driven Updates**:

\`\`\`javascript
// Notification Service maintains read model
messageQueue.subscribe('orders.created', async (message) => {
  const order = message.data;
  
  // Update local read model
  await notificationDB.orders.upsert({
    orderId: order.orderId,
    userId: order.userId,
    status: 'CONFIRMED',
    lastUpdated: new Date()
  });
  
  message.ack();
});

messageQueue.subscribe('orders.shipped', async (message) => {
  // Update read model
  await notificationDB.orders.update(
    { orderId: message.data.orderId },
    { status: 'SHIPPED' }
  );
  
  message.ack();
});
\`\`\`

**Key Takeaways**:

1. **Sync for critical path**: Inventory + Payment (user must know result)
2. **Async for non-blocking**: Notifications, analytics, fulfillment
3. **Idempotency**: Use idempotency keys for all mutations
4. **Compensating transactions**: Release inventory if payment fails
5. **Retry with backoff**: Automatic retry for transient errors
6. **Circuit breakers**: Prevent cascading failures
7. **Event sourcing**: Publish events for async consumers
8. **Eventual consistency**: Accept that notifications may be delayed`,
          keyPoints: [
            'Synchronous (REST/gRPC): Inventory check and payment processing (need immediate response)',
            'Asynchronous (message queue): Notifications, analytics, fulfillment (user can wait)',
            'Idempotency keys: Prevent duplicate orders on retry (store in Redis for 24h)',
            'Compensating transactions: Release inventory if payment fails',
            'Circuit breakers + retry with exponential backoff for fault tolerance',
            'Event-driven: Publish OrderCreated event for async processing by multiple services',
          ],
        },
        {
          id: 'q2-communication',
          question:
            'Compare REST, gRPC, and message queues for microservice communication. For each approach, discuss performance characteristics, use cases, error handling, backward compatibility, and operational complexity. Which would you choose for different scenarios and why?',
          sampleAnswer: `**Comprehensive Comparison: REST vs gRPC vs Message Queues**

---

## **1. REST (Representational State Transfer)**

**Protocol**: HTTP/HTTPS with JSON/XML payloads

**Example**:
\`\`\`javascript
// REST API call
const response = await fetch('https://api.example.com/users/123', {
  method: 'GET',
  headers: {
    'Authorization': 'Bearer token',
    'Accept': 'application/json'
  }
});
const user = await response.json();
\`\`\`

### **Performance**

**Throughput**: ~1,000-5,000 requests/second per instance
- Text-based (JSON) → larger payloads
- HTTP/1.1 → one request per connection (unless HTTP/2)
- Parsing overhead for JSON

**Latency**: ~10-50ms typical
- Network + JSON serialization/deserialization

**Bandwidth**: ~500 bytes - 2KB per request (JSON overhead)

### **Use Cases**

✅ **Public APIs** (RESTful, cacheable, curl-testable)
✅ **Browser clients** (native fetch support)
✅ **CRUD operations** (GET, POST, PUT, DELETE map naturally)
✅ **Third-party integrations** (universal support)

### **Error Handling**

\`\`\`javascript
try {
  const response = await fetch('/api/orders', {
    method: 'POST',
    body: JSON.stringify(order)
  });
  
  if (!response.ok) {
    if (response.status === 429) {
      // Rate limited - retry after delay
      throw new RateLimitError();
    } else if (response.status === 503) {
      // Service unavailable - transient error
      throw new TransientError();
    }
  }
  
  return await response.json();
} catch (error) {
  // Handle network errors
  throw error;
}
\`\`\`

### **Backward Compatibility**

✅ Easy with versioning (/api/v1/, /api/v2/)
✅ Additive changes safe (new fields don't break old clients)
❌ Removing fields breaks clients

### **Operational Complexity**

**Low** - Well-understood, simple debugging, standard tooling

---

## **2. gRPC (Google Remote Procedure Call)**

**Protocol**: HTTP/2 with Protocol Buffers (protobuf)

**Example**:
\`\`\`protobuf
// user.proto
service UserService {
  rpc GetUser (GetUserRequest) returns (User);
  rpc StreamUsers (StreamUsersRequest) returns (stream User);
}

message User {
  int64 id = 1;
  string name = 2;
  string email = 3;
}
\`\`\`

\`\`\`javascript
// gRPC client
const client = new UserServiceClient('localhost:50051');

client.getUser({ id: 123 }, (error, user) => {
  if (error) {
    console.error(error);
    return;
  }
  console.log(user);
});
\`\`\`

### **Performance**

**Throughput**: ~10,000-50,000 requests/second per instance
- Binary protocol (protobuf) → ~5-10x smaller than JSON
- HTTP/2 multiplexing → multiple streams per connection
- No parsing overhead

**Latency**: ~1-10ms typical
- Faster serialization than JSON

**Bandwidth**: ~50-200 bytes per request (protobuf efficiency)

### **Use Cases**

✅ **Internal microservices** (high performance needed)
✅ **Real-time streaming** (bi-directional streams)
✅ **Polyglot environments** (protobuf generates code for all languages)
✅ **Mobile backends** (bandwidth efficiency)

### **Error Handling**

\`\`\`javascript
client.getUser({ id: 123 }, (error, user) => {
  if (error) {
    switch (error.code) {
      case grpc.status.NOT_FOUND:
        // User not found
        break;
      case grpc.status.UNAVAILABLE:
        // Service unavailable - retry
        break;
      case grpc.status.DEADLINE_EXCEEDED:
        // Timeout
        break;
      default:
        // Unknown error
    }
  }
});
\`\`\`

### **Backward Compatibility**

✅ Excellent - protobuf designed for evolution
✅ Adding fields safe (field numbers never reused)
✅ Removing fields safe (clients ignore unknown fields)
✅ No versioning needed (schema evolution built-in)

### **Operational Complexity**

**Medium** - Requires protobuf compilation, harder to debug (binary), limited browser support

---

## **3. Message Queues (RabbitMQ, Kafka, SQS)**

**Protocol**: AMQP, Kafka protocol, or cloud-specific

**Example**:
\`\`\`javascript
// Publisher
await messageQueue.publish('orders.created', {
  orderId: '123',
  userId: 'user-456',
  items: [...]
});

// Subscriber
messageQueue.subscribe('orders.created', async (message) => {
  const order = message.data;
  await processOrder(order);
  message.ack();
});
\`\`\`

### **Performance**

**Throughput**: ~10,000-1,000,000 messages/second (Kafka)
- Kafka: highest throughput (log-based)
- RabbitMQ: ~10,000-50,000 messages/second
- SQS: ~3,000 messages/second per queue

**Latency**: ~5-50ms
- Depends on queue implementation
- Kafka: batch processing → higher throughput, slightly higher latency

**Bandwidth**: Variable (depends on message size)

### **Use Cases**

✅ **Asynchronous processing** (don't need immediate response)
✅ **Event-driven architecture** (pub/sub patterns)
✅ **Decoupling services** (temporal decoupling)
✅ **Load leveling** (buffer spikes in traffic)
✅ **Guaranteed delivery** (at-least-once with acknowledgments)

### **Error Handling**

\`\`\`javascript
messageQueue.subscribe('orders.created', async (message) => {
  try {
    await processOrder(message.data);
    message.ack(); // Success - remove from queue
  } catch (error) {
    if (error.isTransient) {
      // Retry after delay
      message.nack({ requeue: true, delay: 5000 });
    } else {
      // Permanent failure - send to dead letter queue
      message.nack({ requeue: false });
      await deadLetterQueue.publish('orders.failed', {
        originalMessage: message.data,
        error: error.message
      });
    }
  }
});
\`\`\`

### **Backward Compatibility**

✅ Good - consumer reads only fields it knows
⚠️ Need schema registry for Kafka (Avro/Protobuf)
⚠️ Old consumers might ignore new event types

### **Operational Complexity**

**High** - Requires queue infrastructure, monitoring, dead letter queues, handling duplicates

---

## **Comparison Table**

| Aspect | REST | gRPC | Message Queue |
|--------|------|------|---------------|
| **Performance** | Moderate | High (5-10x faster) | High (async) |
| **Latency** | 10-50ms | 1-10ms | 5-50ms + processing |
| **Throughput** | 1K-5K req/s | 10K-50K req/s | 10K-1M msg/s |
| **Coupling** | Tight (sync) | Tight (sync) | Loose (async) |
| **Use Case** | Public APIs, CRUD | Internal services | Events, async |
| **Debugging** | Easy (text) | Hard (binary) | Medium |
| **Browser Support** | Native | Limited (gRPC-Web) | None (backend only) |
| **Backward Compat** | Versioning | Protobuf evolution | Schema registry |
| **Failure Handling** | Retry | Retry | DLQ + requeue |
| **Ops Complexity** | Low | Medium | High |

---

## **Decision Framework**

### **Choose REST when:**
- **Public-facing APIs** (third-party developers)
- **Browser clients** (fetch API works natively)
- **Simple CRUD** (GET, POST, PUT, DELETE)
- **Team familiarity** (most devs know REST)
- **Debugging ease** (curl-testable)

### **Choose gRPC when:**
- **Internal microservices** (backend-to-backend)
- **High performance required** (low latency, high throughput)
- **Streaming** (server streaming, client streaming, bidirectional)
- **Polyglot** (multiple languages)
- **Mobile apps** (bandwidth efficiency)

### **Choose Message Queues when:**
- **Asynchronous processing** (user doesn't wait)
- **Event-driven architecture** (multiple consumers)
- **Decoupling** (services don't call each other directly)
- **Load leveling** (buffer traffic spikes)
- **Guaranteed delivery** (at-least-once semantics)

---

## **Real-World Example: E-Commerce Platform**

\`\`\`
┌─────────────────────────────────────────────┐
│           Client (Browser/Mobile)           │
└──────────────────┬──────────────────────────┘
                   │
                   │ REST (public API)
                   ↓
          ┌────────────────┐
          │  API Gateway   │
          └────────────────┘
                   │
                   ├─ gRPC ─→ User Service
                   ├─ gRPC ─→ Product Service  
                   ├─ gRPC ─→ Order Service
                   └─ gRPC ─→ Payment Service
                          │
                          │ Publish events
                          ↓
                   ┌─────────────┐
                   │ Message Bus │
                   └─────────────┘
                          │
                          ├─→ Notification Service (email)
                          ├─→ Analytics Service (metrics)
                          ├─→ Fulfillment Service (shipping)
                          └─→ Inventory Service (stock update)
\`\`\`

**Reasoning**:
1. **REST**: Browser → API Gateway (universal support)
2. **gRPC**: Internal services (performance)
3. **Message Queue**: Notifications, analytics (async, decoupled)

---

## **My Recommendation**

For most production microservices:

**Frontend ↔ Backend**: REST (simplicity, browser support)
**Backend ↔ Backend**: gRPC (performance, type safety)
**Async/Events**: Message Queue (decoupling, reliability)

This hybrid approach gives you the best of all worlds!`,
          keyPoints: [
            'REST: Best for public APIs and browser clients (easy to use, universally supported)',
            'gRPC: Best for internal microservices (5-10x faster, type-safe, streaming support)',
            'Message Queues: Best for async processing and event-driven architecture (decoupling, guaranteed delivery)',
            'Performance: gRPC fastest (1-10ms), REST moderate (10-50ms), MQ async but high throughput',
            'Operational complexity: REST (low), gRPC (medium), Message Queue (high)',
            'Hybrid approach recommended: REST for public, gRPC internal, MQ for events',
          ],
        },
        {
          id: 'q3-communication',
          question:
            'Design a request tracing system for microservices that allows debugging distributed transactions across 20+ services. Explain how you would propagate trace context, collect spans, handle sampling, store trace data, and build a query interface. Include specific implementation details for instrumentation and visualization.',
          sampleAnswer: `**Distributed Tracing System Design**

**1. Architecture Overview**

\`\`\`
Client Request
    ↓
API Gateway (generates trace_id)
    ├→ Service A (adds span)
    │   ├→ Service B (adds span)
    │   └→ Service C (adds span)
    │       └→ Service D (adds span)
    └→ Service E (adds span)
         │
         └ All spans sent to Collector
              ↓
         Trace Storage (Cassandra/Elasticsearch)
              ↓
         Query API & UI (Jaeger/Zipkin)
\`\`\`

**Key Components**:
1. **Trace Context Propagation** (W3C Trace Context standard)
2. **Span Collection** (OpenTelemetry)
3. **Sampling** (Head-based and tail-based)
4. **Storage** (Cassandra for traces, Elasticsearch for search)
5. **Query Interface** (Jaeger UI)

---

**2. Trace Context Propagation**

**W3C Trace Context Format**:

\`\`\`
traceparent: 00-{trace-id}-{span-id}-{flags}
tracestate: vendor1=value1,vendor2=value2
\`\`\`

**Implementation**:

\`\`\`javascript
// API Gateway (Entry Point)
const tracer = require('@opentelemetry/api').trace.getTracer('api-gateway');

app.use((req, res, next) => {
  // Start new trace or continue existing
  const span = tracer.startSpan('http_request', {
    kind: SpanKind.SERVER,
    attributes: {
      'http.method': req.method,
      'http.url': req.url,
      'http.target': req.path,
      'http.host': req.hostname
    }
  });
  
  // Extract or generate trace context
  const traceId = req.headers['traceparent'] 
    ? extractTraceId(req.headers['traceparent'])
    : generateTraceId();
  
  // Inject into request context
  req.traceId = traceId;
  req.spanId = span.spanContext().spanId;
  
  // Set response header
  res.setHeader('X-Trace-ID', traceId);
  
  // Continue
  span.end();
  next();
});

// Service-to-Service propagation
async function callServiceA(data) {
  const span = tracer.startSpan('call_service_a');
  
  try {
    const response = await httpClient.post('http://service-a/api/process', data, {
      headers: {
        'traceparent': \`00-\${traceId}-\${span.spanContext().spanId}-01\`,
        'X-Request-ID': generateRequestId()
      }
    });
    
    span.setStatus({ code: SpanStatusCode.OK });
    return response.data;
  } catch (error) {
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: error.message
    });
    span.recordException(error);
    throw error;
  } finally {
    span.end();
  }
}
\`\`\`

---

**3. Span Collection with OpenTelemetry**

**Instrumentation**:

\`\`\`javascript
const { NodeTracerProvider } = require('@opentelemetry/sdk-trace-node');
const { BatchSpanProcessor } = require('@opentelemetry/sdk-trace-base');
const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');
const { Resource } = require('@opentelemetry/resources');
const { SemanticResourceAttributes } = require('@opentelemetry/semantic-conventions');

// Initialize tracer
const provider = new NodeTracerProvider({
  resource: new Resource({
    [SemanticResourceAttributes.SERVICE_NAME]: 'order-service',
    [SemanticResourceAttributes.SERVICE_VERSION]: '1.2.3',
    [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: 'production'
  })
});

// Configure exporter
const exporter = new JaegerExporter({
  endpoint: 'http://jaeger-collector:14268/api/traces',
  maxPacketSize: 65000
});

// Batch spans for efficiency
provider.addSpanProcessor(new BatchSpanProcessor(exporter, {
  maxQueueSize: 2048,
  maxExportBatchSize: 512,
  scheduledDelayMillis: 5000
}));

provider.register();

// Auto-instrumentation
const { registerInstrumentations } = require('@opentelemetry/instrumentation');
const { HttpInstrumentation } = require('@opentelemetry/instrumentation-http');
const { ExpressInstrumentation } = require('@opentelemetry/instrumentation-express');
const { PgInstrumentation } = require('@opentelemetry/instrumentation-pg');

registerInstrumentations({
  instrumentations: [
    new HttpInstrumentation(),
    new ExpressInstrumentation(),
    new PgInstrumentation()
  ]
});

// Manual instrumentation example
const tracer = require('@opentelemetry/api').trace.getTracer('order-service');

async function processOrder(orderId) {
  const span = tracer.startSpan('process_order', {
    attributes: {
      'order.id': orderId,
      'operation': 'process_order'
    }
  });
  
  try {
    // Business logic
    await reserveInventory(orderId);
    await processPayment(orderId);
    await confirmOrder(orderId);
    
    span.addEvent('order_processed_successfully');
    span.setStatus({ code: SpanStatusCode.OK });
    
  } catch (error) {
    span.recordException(error);
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: error.message
    });
    throw error;
  } finally {
    span.end();
  }
}
\`\`\`

---

**4. Sampling Strategy**

**Head-Based Sampling** (at trace start):

\`\`\`javascript
const { TraceIdRatioBasedSampler, ParentBasedSampler } = require('@opentelemetry/sdk-trace-base');

// Sample 10% of traces
const sampler = new ParentBasedSampler({
  root: new TraceIdRatioBasedSampler(0.1) // 10%
});

const provider = new NodeTracerProvider({
  sampler: sampler
});
\`\`\`

**Tail-Based Sampling** (after trace completes):

\`\`\`javascript
// Collector config (tail-based sampling)
processors:
  tail_sampling:
    decision_wait: 10s
    num_traces: 100000
    expected_new_traces_per_sec: 1000
    policies:
      - name: errors-policy
        type: status_code
        status_code: {status_codes: [ERROR]}
      - name: slow-traces-policy
        type: latency
        latency: {threshold_ms: 1000}
      - name: random-policy
        type: probabilistic
        probabilistic: {sampling_percentage: 1}
\`\`\`

**Adaptive Sampling**:

\`\`\`javascript
class AdaptiveSampler {
  constructor(targetRate = 100) { // 100 traces/sec
    this.targetRate = targetRate;
    this.currentRate = 0;
    this.samplingProbability = 1.0;
  }
  
  shouldSample(traceId) {
    // Always sample errors
    if (this.hasError(traceId)) {
      return true;
    }
    
    // Always sample slow traces
    if (this.isSlow(traceId)) {
      return true;
    }
    
    // Probabilistic sampling for others
    return Math.random() < this.samplingProbability;
  }
  
  adjustRate() {
    // Adjust sampling rate every minute
    if (this.currentRate > this.targetRate) {
      this.samplingProbability *= 0.9;
    } else if (this.currentRate < this.targetRate * 0.8) {
      this.samplingProbability = Math.min(1.0, this.samplingProbability * 1.1);
    }
  }
}
\`\`\`

---

**5. Trace Storage**

**Schema**:

\`\`\`sql
-- Cassandra schema
CREATE TABLE traces (
    trace_id text,
    span_id text,
    parent_span_id text,
    service_name text,
    operation_name text,
    start_time timestamp,
    duration bigint,
    tags map<text, text>,
    logs list<frozen<log_entry>>,
    PRIMARY KEY (trace_id, start_time, span_id)
) WITH CLUSTERING ORDER BY (start_time DESC);

-- Index for queries
CREATE INDEX ON traces (service_name);
CREATE INDEX ON traces (operation_name);
CREATE INDEX ON traces (duration);
\`\`\`

**Elasticsearch for Search**:

\`\`\`json
{
  "mappings": {
    "properties": {
      "traceId": { "type": "keyword" },
      "spanId": { "type": "keyword" },
      "serviceName": { "type": "keyword" },
      "operationName": { "type": "keyword" },
      "startTime": { "type": "date" },
      "duration": { "type": "long" },
      "tags": {
        "type": "nested",
        "properties": {
          "key": { "type": "keyword" },
          "value": { "type": "text" }
        }
      },
      "status": { "type": "keyword" }
    }
  }
}
\`\`\`

---

**6. Query Interface**

**API**:

\`\`\`javascript
// Query API
app.get('/api/traces', async (req, res) => {
  const {
    service,
    operation,
    minDuration,
    maxDuration,
    tags,
    startTime,
    endTime,
    limit = 20
  } = req.query;
  
  const query = {
    bool: {
      must: []
    }
  };
  
  if (service) {
    query.bool.must.push({ term: { serviceName: service } });
  }
  
  if (operation) {
    query.bool.must.push({ term: { operationName: operation } });
  }
  
  if (minDuration || maxDuration) {
    query.bool.must.push({
      range: {
        duration: {
          gte: minDuration || 0,
          lte: maxDuration || Infinity
        }
      }
    });
  }
  
  const traces = await elasticsearch.search({
    index: 'traces',
    body: { query },
    size: limit
  });
  
  res.json(traces.hits.hits);
});

// Get trace by ID
app.get('/api/traces/:traceId', async (req, res) => {
  const spans = await cassandra.execute(
    'SELECT * FROM traces WHERE trace_id = ?',
    [req.params.traceId]
  );
  
  // Build trace tree
  const trace = buildTraceTree(spans.rows);
  res.json(trace);
});
\`\`\`

**7. Visualization**

Use **Jaeger UI** for visualization:
- Gantt chart of spans (timeline view)
- Service dependency graph
- Span details (tags, logs, events)
- Trace comparison

---

**Key Takeaways**:

1. **W3C Trace Context** for standardized propagation
2. **OpenTelemetry** for vendor-neutral instrumentation
3. **Sampling**: Head-based (simple) or tail-based (smarter)
4. **Storage**: Cassandra for traces, Elasticsearch for search
5. **Always trace errors and slow requests**
6. **Jaeger/Zipkin** for visualization
7. **Auto-instrumentation** for frameworks, manual for business logic
8. **Batch span exports** for efficiency (5-second batches)`,
          keyPoints: [
            'Propagate trace context using W3C Trace Context standard (traceparent header)',
            'Instrument with OpenTelemetry (vendor-neutral, auto-instrumentation for HTTP/DB)',
            'Sampling: Always sample errors and slow requests, probabilistic for others',
            'Storage: Cassandra for traces (scalable writes), Elasticsearch for search queries',
            'Batch span exports every 5 seconds for efficiency (reduce network calls)',
            'Visualization: Jaeger UI for Gantt charts, service graphs, and trace comparison',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-communication-1',
          question:
            'For an e-commerce checkout flow, you need to check inventory, process payment, and send confirmation email. Which communication pattern is BEST?',
          options: [
            'All synchronous REST calls (inventory, payment, email)',
            'All asynchronous message queue (inventory, payment, email)',
            'Synchronous for inventory and payment, asynchronous for email',
            'Asynchronous for inventory and payment, synchronous for email',
          ],
          correctAnswer: 2,
          explanation:
            "Option 3 is correct because: (1) Inventory check and payment processing are critical to checkout success and need immediate results - user is waiting and can't proceed without confirmation, so these should be synchronous. (2) Email notification can be sent asynchronously - user doesn't need to wait for email to be sent, and if email service is temporarily down, it shouldn't block the order. Option 1 is worse because email failure would block order. Option 2 is wrong because user needs immediate confirmation of inventory/payment. Option 4 makes no sense (why would email be synchronous but payment async?).",
        },
        {
          id: 'mc-communication-2',
          question:
            'Your Order Service calls User Service, which calls Product Service, which calls Inventory Service in sequence. What is the main problem?',
          options: [
            'Too many services - should be combined into one',
            'Sequential network calls create latency waterfall and tight coupling',
            'REST is the wrong protocol - should use gRPC',
            'Services need to share a database instead',
          ],
          correctAnswer: 1,
          explanation:
            "Sequential (synchronous chain) communication creates two major problems: (1) Latency adds up in a waterfall (if each call is 10ms, total is 40ms), and (2) Tight coupling - if any service in the chain is down or slow, the entire flow fails. Solutions include: making calls in parallel where possible, using an aggregation service (BFF pattern), or redesigning to use asynchronous events. Option 1 (merge services) defeats microservices benefits. Option 3 (gRPC) would be slightly faster but doesn't solve fundamental problem. Option 4 (shared database) is an anti-pattern.",
        },
        {
          id: 'mc-communication-3',
          question:
            'When an order is placed, 5 different services need to be notified (Payment, Inventory, Shipping, Analytics, Notification). What pattern should you use?',
          options: [
            'Synchronous REST calls from Order Service to all 5 services in sequence',
            'Synchronous REST calls from Order Service to all 5 services in parallel',
            'Order Service publishes "OrderPlaced" event to message broker, services subscribe',
            'All 5 services poll Order Service database every minute for new orders',
          ],
          correctAnswer: 2,
          explanation:
            'Pub/Sub (publish/subscribe) pattern is ideal when multiple services need to react to the same event. Order Service publishes a single "OrderPlaced" event, and each service subscribes independently. Benefits: (1) Loose coupling - Order Service doesn\'t know/care who subscribes, (2) Easy to add new subscribers without changing Order Service, (3) Parallel processing - all services process simultaneously, (4) Each service can fail/retry independently. Option 1 (sequential REST) creates tight coupling and latency. Option 2 (parallel REST) is better but still couples Order Service to all 5 services. Option 4 (polling) is inefficient and adds latency.',
        },
        {
          id: 'mc-communication-4',
          question:
            'What is the main benefit of a service mesh like Istio for microservices communication?',
          options: [
            'Service mesh replaces HTTP with faster binary protocol',
            'Service mesh handles cross-cutting concerns (retries, circuit breakers, mTLS, tracing) without changing service code',
            'Service mesh eliminates the need for asynchronous communication',
            'Service mesh combines all microservices into a single deployment',
          ],
          correctAnswer: 1,
          explanation:
            "Service mesh's primary value is handling cross-cutting concerns at the infrastructure layer via sidecar proxies: automatic retries, circuit breakers, timeouts, mutual TLS encryption, distributed tracing, and load balancing - all without developers writing code for these in each service. This is huge for polyglot environments where services are written in different languages. Option 1 is false (service mesh works with HTTP/gRPC). Option 3 is false (you still need async for some use cases). Option 4 is completely wrong (service mesh is for distributed microservices).",
        },
        {
          id: 'mc-communication-5',
          question:
            'Which API change is backward-compatible and does NOT require API versioning?',
          options: [
            'Removing the "phoneNumber" field from User response',
            'Changing "amount" field from number to string',
            'Adding a new optional field "email" to User response',
            'Renaming "userId" field to "id"',
          ],
          correctAnswer: 2,
          explanation:
            'Adding optional fields is backward-compatible because: (1) Old clients will ignore the new field (JSON parsers skip unknown fields), (2) New clients can use the new field, (3) No existing functionality breaks. Options 1, 2, and 4 are all breaking changes: removing fields breaks clients expecting them, changing types breaks parsers, renaming fields breaks all references. Backward-compatible changes include: adding optional fields, adding new endpoints, making required fields optional, adding enum values. Breaking changes require versioning (/v1/ vs /v2/) or deprecation periods.',
        },
      ],
      discussion: [
        {
          question:
            'Design the communication architecture for a ride-sharing app like Uber. Include: ride requests, driver matching, real-time location updates, and payments. Choose synchronous vs asynchronous patterns and justify your choices.',
          answer: `I'll design a hybrid communication architecture mixing synchronous and asynchronous patterns based on specific requirements for each use case.

**Architecture Overview:**

**Core Services:**
- Rider Service
- Driver Service  
- Matching Service (dispatch)
- Location Service (real-time tracking)
- Pricing Service
- Payment Service
- Notification Service
- Trip Service

**Use Case 1: Ride Request Flow**

**Pattern: Synchronous → Asynchronous Hybrid**

\\\`\\\`\\\`
1. Rider opens app, requests ride
   → API Gateway → Rider Service (REST, synchronous)
   Returns: Request acknowledged immediately
   
2. Rider Service → Publish event: "RideRequested"
   → Event Bus (Kafka topic)
   
3. Matching Service subscribes to "RideRequested"
   → Find available drivers nearby (geospatial query)
   → Publish event: "DriversMatched"
   
4. Driver Service subscribes to "DriversMatched"
   → Send push notifications to matched drivers
   → First driver to accept wins
\\\`\\\`\\\`

**Why This Pattern:**
- ✅ Rider gets immediate acknowledgment (good UX)
- ✅ Matching algorithm can take time (5-30 seconds) without blocking
- ✅ Can try multiple drivers in parallel
- ✅ If Matching Service slow, doesn't hang rider app

**Use Case 2: Real-Time Location Updates**

**Pattern: WebSocket (bidirectional streaming)**

\\\`\\\`\\\`
Driver App ←→ WebSocket ←→ Location Service
Rider App ←→ WebSocket ←→ Location Service

Every 3 seconds:
- Driver app sends GPS coordinates
- Location Service broadcasts to rider
- Rider app updates map
\\\`\\\`\\\`

**Why WebSocket (not REST polling):**
- ✅ Real-time: sub-second latency
- ✅ Efficient: single persistent connection (not 20 requests/minute)
- ✅ Bidirectional: both driver and rider get updates
- ❌ More complex than REST
- ❌ Requires WebSocket infrastructure

**Alternative for Backend**: Publish location updates to Kafka topic
\\\`\\\`\\\`
Location Service → Kafka topic: "location.updates"
↓
Subscriptions:
- ETA Service (recalculate arrival time)
- Surge Pricing Service (detect traffic patterns)  
- Analytics Service (heat maps)
\\\`\\\`\\\`

**Use Case 3: Payment Processing**

**Pattern: Synchronous with Circuit Breaker**

\\\`\\\`\\\`
Trip ends → Trip Service
          ↓ (REST call with circuit breaker)
        Payment Service
          ↓
        Stripe API
          ↓
        Response (success/failure)
\\\`\\\`\\\`

**Why Synchronous:**
- ✅ Need immediate result (did payment succeed?)
- ✅ Trip can't complete without payment confirmation
- ✅ User needs to know payment status before closing app

Mix patterns based on specific requirements. There's no single "best" communication pattern for all use cases.`,
        },
        {
          question:
            'Explain the trade-offs between synchronous (REST) and asynchronous (message queue) communication in microservices. When would you choose each, and what are the operational implications?',
          answer:
            'This is one of the most important architectural decisions in microservices. Synchronous communication (REST/gRPC) is simple and provides immediate feedback with strong consistency, but creates tight coupling and cascading failures. Asynchronous communication (message queues/events) provides decoupling, resilience, and load leveling, but adds complexity and eventual consistency challenges. Choose synchronous when users need immediate responses and strong consistency is required. Choose asynchronous when operations can be delayed, multiple consumers exist, or retry logic is critical. Most real systems use a hybrid approach - synchronous for critical paths and asynchronous for non-critical operations.',
        },
        {
          question:
            'What is a service mesh (like Istio or Linkerd)? What problems does it solve in microservices communication, and when should you use one?',
          answer:
            "A service mesh is an infrastructure layer that handles service-to-service communication via sidecar proxies. It solves cross-cutting concerns like service discovery, load balancing, retries, circuit breakers, mutual TLS, and distributed tracing without code changes. Use a service mesh when you have 15+ microservices, a polyglot environment, strict security requirements, or complex traffic management needs (canary deployments, A/B testing). Don't use it for fewer than 5 services, simple communication patterns, or teams lacking operational maturity. The trade-off is reduced application complexity at the cost of increased operational complexity.",
        },
      ],
    },
    {
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
    ServiceInstance instance = chooseInstance(instances);
    
    // Make HTTP call
    String url = instance.getUri() + "/charge";
    return restTemplate.postForObject(url, payment, String.class);
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
      quiz: [
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
      ],
      multipleChoice: [
        {
          id: 'mc-discovery-1',
          question:
            'What is the main advantage of client-side service discovery over server-side discovery?',
          options: [
            'Simpler client implementation',
            'No need for health checks',
            'Lower latency (no extra network hop) and client controls load balancing',
            'Works with any programming language',
          ],
          correctAnswer: 2,
          explanation:
            'Client-side discovery has lower latency because clients call service instances directly without going through a load balancer (no extra hop). Clients also have full control over the load balancing algorithm. However, this comes at the cost of more complex client implementation. Option 1 is wrong (client-side is actually more complex). Option 2 is wrong (health checks are still needed). Option 4 is wrong (server-side discovery is more language-agnostic).',
        },
        {
          id: 'mc-discovery-2',
          question:
            'Your payment service autoscales from 2 to 20 instances during Black Friday. How does service discovery handle this?',
          options: [
            'Manual configuration update required for all clients',
            'New instances self-register with registry; clients automatically discover them',
            'Need to restart all client services',
            'Load balancer must be manually reconfigured',
          ],
          correctAnswer: 1,
          explanation:
            'Service discovery automatically handles dynamic scaling. New instances self-register with the service registry on startup (or are registered by an orchestrator like Kubernetes). Clients query the registry and automatically get the updated list of instances. No manual intervention needed. This is the whole point of service discovery - handling dynamic infrastructure. Options 1, 3, and 4 all describe manual processes that service discovery eliminates.',
        },
        {
          id: 'mc-discovery-3',
          question:
            'What happens if a service instance fails but remains registered in the service registry?',
          options: [
            'Nothing - the system continues to work normally',
            'Clients receive errors when trying to call the dead instance',
            'The service registry automatically detects and removes it after N failed health checks',
            'All instances of that service must be restarted',
          ],
          correctAnswer: 2,
          explanation:
            'Health checks prevent routing to dead instances. The service registry periodically checks instance health (heartbeat or active polling). After N consecutive failed health checks, the registry marks the instance as DOWN and stops returning it in discovery queries. Clients never see the dead instance. Option 2 can happen temporarily, but proper health checking minimizes this window. Option 1 is wrong (dead instances cause problems). Option 4 is unnecessary overkill.',
        },
        {
          id: 'mc-discovery-4',
          question: 'In Kubernetes, how does built-in service discovery work?',
          options: [
            'Kubernetes uses Netflix Eureka internally',
            'Services must manually register with etcd',
            'Kubernetes creates DNS entries for Services; pods are automatically discovered via label selectors',
            'Each pod must implement health check endpoints',
          ],
          correctAnswer: 2,
          explanation:
            "Kubernetes has built-in service discovery: You create a Service resource with label selectors. Kubernetes automatically discovers all pods matching those labels and adds them to the Service. CoreDNS creates DNS entries (service-name.namespace.svc.cluster.local) that resolve to the Service's ClusterIP, which load balances to healthy pods. It's completely automatic - no manual registration needed. Option 1 is false (Kubernetes doesn't use Eureka). Option 2 is false (automatic via label matching). Option 4 is good practice but not required for basic discovery.",
        },
        {
          id: 'mc-discovery-5',
          question:
            'Why is DNS-based service discovery considered simpler but more limited than registry-based discovery?',
          options: [
            'DNS is slower than registries',
            'DNS requires special client libraries',
            'DNS has caching issues (TTL) and limited load balancing options, but is universally supported',
            'DNS only works with HTTP services',
          ],
          correctAnswer: 2,
          explanation:
            "DNS-based discovery is simpler because every programming language has built-in DNS support - no special libraries needed. However, it's limited: (1) DNS caching (TTL) means clients may get stale data, (2) DNS provides limited load balancing (usually just round-robin A records), (3) DNS doesn't remove unhealthy instances quickly. Registry-based discovery (Eureka, Consul) provides real-time updates, sophisticated load balancing, and immediate health-based routing. Option 1 is debatable. Option 2 is backwards (DNS needs no libraries). Option 4 is false (DNS works with any protocol).",
        },
      ],
    },
    {
      id: 'api-gateway',
      title: 'API Gateway Pattern',
      content: `The API Gateway pattern is a single entry point for all client requests in a microservices architecture. It acts as a reverse proxy, routing requests to appropriate microservices.

## The Problem

**Without API Gateway**:
\`\`\`
Mobile App ─────┬──→ Auth Service (port 8001)
                ├──→ Order Service (port 8002)
                ├──→ Payment Service (port 8003)
                ├──→ Product Service (port 8004)
                └──→ User Service (port 8005)
\`\`\`

**Problems**:
- Clients must know about all services and their locations
- Each service may have different authentication mechanisms
- Can't enforce rate limiting across services
- CORS issues with multiple origins
- Client makes multiple round trips (slow on mobile)
- Protocol translation (HTTP → gRPC) handled by each client

**With API Gateway**:
\`\`\`
Mobile App ───→ API Gateway:443 ───┬──→ Auth Service
Web App    ───→                     ├──→ Order Service
                                    ├──→ Payment Service
                                    ├──→ Product Service
                                    └──→ User Service
\`\`\`

Single entry point, handles routing, auth, rate limiting, etc.

---

## Core Responsibilities

### 1. **Request Routing**

Route requests to appropriate backend services.

**Example**:
\`\`\`
GET  /api/products      → Product Service
POST /api/orders        → Order Service
GET  /api/users/me      → User Service
POST /api/auth/login    → Auth Service
\`\`\`

**Path-Based Routing**:
\`\`\`nginx
# NGINX configuration
location /api/products {
    proxy_pass http://product-service:8080;
}

location /api/orders {
    proxy_pass http://order-service:8080;
}
\`\`\`

**Intelligent Routing** (based on headers, query params):
\`\`\`javascript
// Route to different service versions
if (request.headers['X-Client-Version'] === '2.0') {
    route to product-service-v2
} else {
    route to product-service-v1
}
\`\`\`

### 2. **Request Aggregation (Backend for Frontend - BFF)**

Combine multiple backend calls into one response.

**Problem**: Mobile app needs user profile + recent orders + notifications = 3 API calls

**Solution**: API Gateway makes 3 calls internally, returns combined response

**Example**:
\`\`\`javascript
// API Gateway endpoint: GET /api/home
async function getHomeData(userId) {
    // Make 3 parallel calls to backend services
    const [user, orders, notifications] = await Promise.all([
        userService.getUser(userId),
        orderService.getRecentOrders(userId),
        notificationService.getUnread(userId)
    ]);
    
    // Return combined response
    return {
        user,
        recentOrders: orders,
        unreadNotifications: notifications
    };
}
\`\`\`

**Benefit**: Mobile app makes 1 request instead of 3, reducing latency

### 3. **Authentication & Authorization**

Centralize auth logic instead of duplicating in each service.

**Flow**:
\`\`\`
1. Client sends request with JWT token to API Gateway
2. Gateway validates token (signature, expiration)
3. Gateway extracts user ID and roles
4. Gateway adds user context to request headers
5. Gateway forwards request to backend service
6. Backend service trusts headers (doesn't re-validate)
\`\`\`

**Example**:
\`\`\`javascript
// API Gateway middleware
async function authenticate(request) {
    const token = request.headers['Authorization'].replace('Bearer ', '');
    
    // Validate JWT
    const decoded = jwt.verify(token, JWT_SECRET);
    
    // Add user context to request
    request.headers['X-User-Id'] = decoded.userId;
    request.headers['X-User-Roles'] = decoded.roles.join(',');
    
    // Forward to backend
    return proxy(request);
}
\`\`\`

**Benefits**:
- Backend services don't need JWT validation logic
- Can enforce authorization policies centrally
- Easy to switch auth mechanisms

### 4. **Rate Limiting & Throttling**

Protect backend services from abuse.

**Example**:
\`\`\`javascript
// Rate limit: 100 requests per minute per user
const rateLimiter = new RateLimiter({
    windowMs: 60 * 1000, // 1 minute
    max: 100 // requests
});

app.use(async (req, res, next) => {
    const userId = req.headers['X-User-Id'];
    
    if (await rateLimiter.isBlocked(userId)) {
        return res.status(429).json({
            error: 'Too many requests'
        });
    }
    
    next();
});
\`\`\`

**Different limits for different tiers**:
\`\`\`
Free tier:    100 req/min
Pro tier:   1,000 req/min
Enterprise: 10,000 req/min
\`\`\`

### 5. **Protocol Translation**

Translate between different protocols.

**Example**: Client uses HTTP/REST, backend uses gRPC

\`\`\`javascript
// API Gateway
app.post('/api/orders', async (req, res) => {
    // Receive HTTP/JSON request
    const orderData = req.body;
    
    // Call backend gRPC service
    const grpcClient = getGrpcClient('order-service');
    const result = await grpcClient.createOrder(orderData);
    
    // Return HTTP/JSON response
    res.json(result);
});
\`\`\`

**Other translations**:
- GraphQL → multiple REST calls
- REST → message queue
- WebSocket → HTTP polling

### 6. **Response Transformation**

Modify responses before returning to client.

**Example**: Remove internal fields, add pagination metadata

\`\`\`javascript
async function getProducts(req, res) {
    // Call backend service
    const products = await productService.getAll();
    
    // Transform response
    const transformed = products.map(p => ({
        id: p.id,
        name: p.name,
        price: p.price,
        // Remove internal fields
        // internalCost: p.internalCost  ❌
        // supplierId: p.supplierId      ❌
    }));
    
    // Add metadata
    res.json({
        data: transformed,
        total: transformed.length,
        page: 1
    });
}
\`\`\`

### 7. **Caching**

Cache responses to reduce backend load.

**Example**:
\`\`\`javascript
const redis = require('redis').createClient();

app.get('/api/products/:id', async (req, res) => {
    const cacheKey = \`product:\${req.params.id}\`;
    
    // Check cache
    const cached = await redis.get(cacheKey);
    if (cached) {
        return res.json(JSON.parse(cached));
    }
    
    // Cache miss - call backend
    const product = await productService.get(req.params.id);
    
    // Store in cache (5 minutes)
    await redis.setex(cacheKey, 300, JSON.stringify(product));
    
    res.json(product);
});
\`\`\`

### 8. **Load Balancing**

Distribute requests across service instances.

**Example: Round Robin**
\`\`\`javascript
const instances = [
    'http://order-service-1:8080',
    'http://order-service-2:8080',
    'http://order-service-3:8080'
];
let currentIndex = 0;

function getNextInstance() {
    const instance = instances[currentIndex];
    currentIndex = (currentIndex + 1) % instances.length;
    return instance;
}
\`\`\`

---

## API Gateway Implementations

### 1. **Kong**

Open-source API Gateway with extensive plugin ecosystem.

**Features**:
- Load balancing
- Authentication (JWT, OAuth 2.0, API Keys)
- Rate limiting
- Request/response transformation
- Logging and monitoring

**Example**:
\`\`\`bash
# Add service
curl -i -X POST http://localhost:8001/services \\
  --data name=order-service \\
  --data url=http://order-service:8080

# Add route
curl -i -X POST http://localhost:8001/services/order-service/routes \\
  --data 'paths[]=/api/orders'

# Add JWT auth plugin
curl -X POST http://localhost:8001/services/order-service/plugins \\
  --data name=jwt
\`\`\`

### 2. **AWS API Gateway**

Fully managed service for creating, publishing, and managing APIs.

**Features**:
- Automatic scaling
- Integration with AWS Lambda
- API versioning
- Usage plans and API keys
- CloudWatch logging

**Example**:
\`\`\`yaml
# SAM template
Resources:
  MyApi:
    Type: AWS::Serverless::Api
    Properties:
      StageName: prod
      Auth:
        ApiKeyRequired: true
      
  GetProduct:
    Type: AWS::Serverless::Function
    Properties:
      Handler: index.handler
      Runtime: nodejs18.x
      Events:
        GetProductApi:
          Type: Api
          Properties:
            RestApiId: !Ref MyApi
            Path: /products/{id}
            Method: get
\`\`\`

### 3. **NGINX / NGINX Plus**

High-performance web server that can act as API Gateway.

**Example**:
\`\`\`nginx
upstream product_service {
    server product-1:8080;
    server product-2:8080;
}

server {
    listen 443 ssl;
    
    location /api/products {
        # Authentication
        auth_request /auth;
        
        # Rate limiting
        limit_req zone=api_limit burst=10;
        
        # Proxy to backend
        proxy_pass http://product_service;
        proxy_set_header X-User-Id $http_x_user_id;
    }
    
    location = /auth {
        internal;
        proxy_pass http://auth-service/validate;
    }
}
\`\`\`

### 4. **Spring Cloud Gateway**

Built on Spring Boot, reactive gateway for microservices.

**Example**:
\`\`\`java
@Configuration
public class GatewayConfig {
    @Bean
    public RouteLocator customRouteLocator(RouteLocatorBuilder builder) {
        return builder.routes()
            .route("products", r -> r.path("/api/products/**")
                .filters(f -> f
                    .stripPrefix(2)
                    .addRequestHeader("X-Gateway", "true")
                    .circuitBreaker(config -> config
                        .setName("productsCircuitBreaker")
                        .setFallbackUri("/fallback/products")))
                .uri("lb://product-service"))
            .build();
    }
}
\`\`\`

### 5. **Traefik**

Cloud-native reverse proxy and load balancer.

**Features**:
- Automatic service discovery
- Let's Encrypt SSL
- Middleware (auth, rate limit, retry)
- Kubernetes ingress controller

**Example**:
\`\`\`yaml
# docker-compose.yml
services:
  traefik:
    image: traefik:v2.9
    command:
      - "--providers.docker=true"
      - "--entrypoints.web.address=:80"
    labels:
      - "traefik.http.routers.api.rule=Host(\`api.example.com\`)"
      - "traefik.http.routers.api.middlewares=auth,ratelimit"
  
  product-service:
    image: product-service:latest
    labels:
      - "traefik.http.routers.products.rule=PathPrefix(\`/api/products\`)"
      - "traefik.http.routers.products.middlewares=auth"
\`\`\`

---

## Backend for Frontend (BFF) Pattern

Different clients (mobile, web, IoT) have different needs. BFF creates dedicated gateways for each.

**Architecture**:
\`\`\`
Mobile App → Mobile BFF ─┐
Web App    → Web BFF    ─┼──→ Microservices
IoT Device → IoT BFF    ─┘
\`\`\`

**Example**:

**Mobile BFF** (optimized for bandwidth):
\`\`\`javascript
// Returns minimal data
GET /api/products
{
    products: [
        {id: 1, name: "iPhone", price: 999, thumb: "url"}
    ]
}
\`\`\`

**Web BFF** (more details):
\`\`\`javascript
// Returns full data
GET /api/products
{
    products: [
        {
            id: 1,
            name: "iPhone 15 Pro",
            description: "...",
            price: 999,
            images: ["url1", "url2", "url3"],
            specs: {...},
            reviews: [...]
        }
    ]
}
\`\`\`

**Advantages**:
✅ Optimized for each client type
✅ Can evolve independently
✅ Better separation of concerns

**Disadvantages**:
❌ More code to maintain
❌ Potential duplication

---

## API Composition Pattern

Gateway composes data from multiple services.

**Example: Order Details Page**

Needs:
1. Order info (Order Service)
2. Product details (Product Service)
3. Shipping status (Shipping Service)
4. Payment status (Payment Service)

**API Gateway implementation**:
\`\`\`javascript
async function getOrderDetails(orderId) {
    // Get order
    const order = await orderService.getOrder(orderId);
    
    // Get related data in parallel
    const [products, shipping, payment] = await Promise.all([
        productService.getByIds(order.productIds),
        shippingService.getStatus(orderId),
        paymentService.getStatus(order.paymentId)
    ]);
    
    // Compose response
    return {
        order: {
            id: order.id,
            status: order.status,
            total: order.total,
            items: order.items.map(item => ({
                ...item,
                product: products.find(p => p.id === item.productId)
            }))
        },
        shipping: {
            status: shipping.status,
            estimatedDelivery: shipping.estimatedDelivery,
            trackingNumber: shipping.trackingNumber
        },
        payment: {
            status: payment.status,
            method: payment.method
        }
    };
}
\`\`\`

---

## API Gateway Anti-Patterns

### ❌ Smart Gateway, Dumb Services

**Problem**: Gateway contains too much business logic

**Example**:
\`\`\`javascript
// BAD: Business logic in gateway
app.post('/api/orders', async (req, res) => {
    // Gateway calculates tax, applies discounts, validates inventory
    const tax = calculateTax(req.body.items, req.body.shippingAddress);
    const discount = applyPromotions(req.body.items, req.user.id);
    const total = calculateTotal(req.body.items, tax, discount);
    
    // Then calls service
    await orderService.create({...req.body, total});
});
\`\`\`

**Why it's bad**: Logic duplicated if you add another gateway, hard to test

**Better**: Gateway only routes, service handles logic
\`\`\`javascript
// GOOD: Gateway just routes
app.post('/api/orders', async (req, res) => {
    // Service handles all business logic
    const result = await orderService.create(req.body, req.user.id);
    res.json(result);
});
\`\`\`

### ❌ Chatty Gateway

**Problem**: Gateway makes too many backend calls for single request

**Example**:
\`\`\`javascript
// BAD: 10 sequential calls
for (const productId of order.productIds) {
    const product = await productService.get(productId); // 1 call per product
}
\`\`\`

**Better**: Batch requests
\`\`\`javascript
// GOOD: 1 call
const products = await productService.getByIds(order.productIds);
\`\`\`

### ❌ God Gateway

**Problem**: Single gateway for everything (becomes bottleneck)

**Better**: Multiple gateways per domain (e.g., public API gateway, internal gateway, admin gateway)

---

## Security Considerations

### SSL Termination

Gateway terminates SSL, talks to backend over HTTP.

\`\`\`
Client (HTTPS) → Gateway (HTTPS) → Services (HTTP)
\`\`\`

**Advantage**: Services don't need SSL certificates

**Risk**: Unencrypted internal traffic (use VPC/private network)

### API Keys

\`\`\`javascript
app.use((req, res, next) => {
    const apiKey = req.headers['X-API-Key'];
    
    if (!isValidApiKey(apiKey)) {
        return res.status(401).json({error: 'Invalid API key'});
    }
    
    next();
});
\`\`\`

### CORS Handling

\`\`\`javascript
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', 'https://example.com');
    res.header('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE');
    res.header('Access-Control-Allow-Headers', 'Content-Type,Authorization');
    next();
});
\`\`\`

---

## Performance Optimization

### Connection Pooling

Reuse connections to backend services.

\`\`\`javascript
const axios = require('axios');
const http = require('http');
const https = require('https');

const httpAgent = new http.Agent({
    keepAlive: true,
    maxSockets: 50
});

const httpsAgent = new https.Agent({
    keepAlive: true,
    maxSockets: 50
});

const client = axios.create({
    httpAgent,
    httpsAgent
});
\`\`\`

### Request Coalescing

Combine duplicate requests.

\`\`\`javascript
const cache = new Map();

async function getProduct(id) {
    // If request in flight, wait for it
    if (cache.has(id)) {
        return cache.get(id);
    }
    
    // Start new request
    const promise = productService.get(id);
    cache.set(id, promise);
    
    // Clean up when done
    const result = await promise;
    setTimeout(() => cache.delete(id), 1000);
    
    return result;
}
\`\`\`

---

## Decision Framework

**Use API Gateway When**:
✅ Multiple client types (mobile, web, partners)
✅ Need centralized auth/rate limiting
✅ Want to hide backend complexity
✅ Microservices architecture

**Skip API Gateway When**:
❌ Simple monolith application
❌ Only one client type
❌ Team too small to maintain it
❌ Latency-critical (extra hop)

---

## Interview Tips

**Red Flags**:
❌ Saying "API Gateway is required for microservices"
❌ Putting business logic in gateway
❌ Not mentioning trade-offs

**Good Responses**:
✅ Explain core responsibilities (routing, auth, aggregation)
✅ Mention specific tools (Kong, AWS API Gateway, NGINX)
✅ Discuss BFF pattern for different clients
✅ Acknowledge trade-offs (single point of failure, latency)

**Sample Answer**:
*"I'd use an API Gateway as the single entry point for all client requests. It would handle authentication, rate limiting, and request routing. For our mobile app, I'd implement the BFF pattern with request aggregation to minimize round trips. We'd use Kong for its plugin ecosystem or AWS API Gateway if we're fully on AWS. The gateway would be stateless and horizontally scalable. We'd monitor latency carefully since it adds an extra hop."*

---

## Key Takeaways

1. **API Gateway** is a single entry point for all clients
2. **Core functions**: routing, auth, rate limiting, aggregation
3. **BFF pattern**: Dedicated gateway per client type
4. **API composition**: Combine multiple service calls
5. **Don't** put business logic in gateway
6. **Tools**: Kong, AWS API Gateway, NGINX, Traefik
7. **Trade-off**: Simplifies clients but adds latency and single point of failure
8. **Scale horizontally** and use health checks`,
      quiz: [
        {
          id: 'q1-api-gateway',
          question:
            'Your mobile app makes 5 separate API calls when loading the home screen, causing slow load times. How would you solve this using API Gateway? What pattern would you use?',
          sampleAnswer:
            "I would implement the API Composition pattern in the API Gateway. Create a new endpoint like GET /api/home that internally makes all 5 backend calls in parallel using Promise.all() and returns a single combined response. This follows the BFF (Backend for Frontend) pattern, where the gateway is optimized for the mobile client's needs. This reduces network round trips from 5 to 1, dramatically improving performance on mobile networks. The gateway should make the backend calls in parallel, not sequentially, and handle partial failures gracefully (e.g., if notifications service is down, still return the rest of the data).",
          keyPoints: [
            'Use API Composition pattern to aggregate multiple backend calls',
            'Create dedicated BFF endpoint optimized for mobile',
            'Make backend calls in parallel (Promise.all) not sequential',
            'Reduces round trips from N to 1 (critical for mobile)',
            'Handle partial failures gracefully',
          ],
        },
        {
          id: 'q2-api-gateway',
          question:
            'What are the trade-offs of using an API Gateway? When might you NOT want to use one?',
          sampleAnswer:
            'API Gateway trade-offs: Adds latency (extra network hop), becomes a single point of failure (mitigated by horizontal scaling and health checks), increases operational complexity (another component to monitor), and can become a bottleneck if not scaled properly. Skip API Gateway when: (1) Simple monolith with one client type - no need for the complexity, (2) Team too small to maintain it - operational overhead not justified, (3) Latency-critical applications where every millisecond matters, (4) Internal services only - might not need centralized auth/rate limiting. Start without it and add later if needed (YAGNI principle).',
          keyPoints: [
            'Adds latency (extra network hop)',
            'Single point of failure (needs HA setup)',
            'Operational complexity',
            'Can become bottleneck',
            'Skip for simple apps, small teams, or latency-critical systems',
          ],
        },
        {
          id: 'q3-api-gateway',
          question:
            'Explain the Backend for Frontend (BFF) pattern. Why would you use separate gateways for mobile and web?',
          sampleAnswer:
            'BFF pattern creates dedicated API Gateways for each client type (mobile BFF, web BFF, IoT BFF, etc.). Mobile needs: minimal data (bandwidth constraints), aggregated responses (fewer round trips), optimized images (thumbnails not full res). Web needs: richer data, multiple images, detailed pagination. By having separate BFFs, each can be optimized for its client without compromising others. Mobile BFF might aggregate 5 calls into 1 and return 100KB, while Web BFF returns 1MB with full details. BFFs can also handle client-specific auth (mobile uses OAuth, web uses cookies, partners use API keys). The trade-off is more code to maintain, but better user experience and separation of concerns.',
          keyPoints: [
            'Separate gateway per client type (mobile, web, IoT)',
            'Mobile BFF: minimal data, aggregated calls, optimized for bandwidth',
            'Web BFF: richer data, multiple images, detailed responses',
            'Each can evolve independently',
            'Trade-off: more code vs better UX and separation of concerns',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-gateway-1',
          question:
            'What is the primary purpose of an API Gateway in microservices?',
          options: [
            'Store user session data',
            'Single entry point that handles routing, authentication, and request aggregation',
            'Replace the need for load balancers',
            'Store business logic to keep services simple',
          ],
          correctAnswer: 1,
          explanation:
            'The API Gateway serves as a single entry point for all client requests. It handles cross-cutting concerns like routing to appropriate services, authentication/authorization, rate limiting, request aggregation, and protocol translation. It simplifies the client by hiding backend complexity. Option 1 is wrong (gateways are usually stateless). Option 3 is wrong (gateways often work WITH load balancers). Option 4 is an anti-pattern (business logic belongs in services, not gateway).',
        },
        {
          id: 'mc-gateway-2',
          question:
            'Your mobile app loads the home screen slowly because it makes 8 separate API calls. What API Gateway pattern should you use?',
          options: [
            'Rate Limiting',
            'Circuit Breaker',
            'API Composition (aggregate multiple calls into one)',
            'Service Discovery',
          ],
          correctAnswer: 2,
          explanation:
            "API Composition (also called request aggregation or BFF pattern) solves this problem. The gateway makes multiple backend calls internally and returns a single combined response to the client. This reduces network round trips from 8 to 1, dramatically improving load times on mobile networks. Option 1 (Rate Limiting) prevents abuse but doesn't help with performance. Option 2 (Circuit Breaker) handles failures but doesn't reduce calls. Option 4 (Service Discovery) is for services finding each other, not client optimization.",
        },
        {
          id: 'mc-gateway-3',
          question: 'Which of these is an API Gateway anti-pattern?',
          options: [
            'Implementing authentication in the gateway',
            'Routing requests to different microservices',
            'Implementing complex business logic (tax calculation, discount rules) in the gateway',
            'Caching responses to reduce backend load',
          ],
          correctAnswer: 2,
          explanation:
            'Putting business logic in the gateway is an anti-pattern called "Smart Gateway, Dumb Services". Business logic should live in services, not the gateway. If you add another gateway or bypass it, the logic is missing. Gateway should handle cross-cutting concerns (auth, routing, caching) but not business rules. Options 1, 2, and 4 are legitimate gateway responsibilities. Keep the gateway focused on infrastructure concerns, not business logic.',
        },
        {
          id: 'mc-gateway-4',
          question: 'What is the Backend for Frontend (BFF) pattern?',
          options: [
            'Having a single API Gateway that serves all clients equally',
            'Creating separate API Gateways optimized for each client type (mobile, web, IoT)',
            'Using a load balancer in front of services',
            'Implementing circuit breakers in the frontend',
          ],
          correctAnswer: 1,
          explanation:
            "BFF pattern creates dedicated gateways for each client type. Mobile BFF returns minimal data optimized for bandwidth; Web BFF returns richer data with more details. Each BFF can aggregate different calls, transform responses differently, and evolve independently. This provides better user experience at the cost of maintaining multiple gateways. Option 1 is wrong (that's a single gateway anti-pattern when clients have very different needs). Option 3 is unrelated. Option 4 makes no sense (circuit breakers are backend, not frontend).",
        },
        {
          id: 'mc-gateway-5',
          question:
            'API Gateway increases latency by adding an extra network hop. When might you skip using one?',
          options: [
            'When you have 50+ microservices that need centralized auth',
            'When you have mobile, web, and partner clients with different needs',
            'When you have a simple monolith with one client and latency is critical',
            'When you need rate limiting and request aggregation',
          ],
          correctAnswer: 2,
          explanation:
            "Skip API Gateway for simple applications (monolith or few services) with one client type, especially when latency is critical. The operational complexity isn't justified. Start simple and add gateway later if needed. Options 1, 2, and 4 are perfect use cases FOR an API Gateway (many services, multiple clients, cross-cutting concerns). Don't over-engineer when you don't need it (YAGNI principle).",
        },
      ],
    },
    {
      id: 'distributed-transactions-saga',
      title: 'Distributed Transactions & Saga Pattern',
      content: `Maintaining data consistency across multiple microservices is one of the hardest challenges in distributed systems. Traditional ACID transactions don't work across service boundaries.

## The Problem: Distributed Transactions

**Monolith with local transaction**:
\`\`\`sql
BEGIN TRANSACTION;
  INSERT INTO orders (...);
  UPDATE inventory SET quantity = quantity - 1;
  INSERT INTO payments (...);
  UPDATE loyalty_points SET points = points + 100;
COMMIT;
\`\`\`

Either all succeed or all rollback atomically.

**Microservices - each has its own database**:
\`\`\`
Order Service (orders DB)
Inventory Service (inventory DB)  
Payment Service (payments DB)
Loyalty Service (loyalty DB)
\`\`\`

**Cannot use a single database transaction across services!**

**Failure scenario**:
\`\`\`
1. Order Service: Create order ✅
2. Inventory Service: Decrease stock ✅  
3. Payment Service: Charge card ❌ FAILS
4. Loyalty Service: Add points ❌ NOT EXECUTED

Result: Order created, inventory decreased, but no payment!
Customer gets free product 💸
\`\`\`

---

## Why 2PC (Two-Phase Commit) Doesn't Work

**Two-Phase Commit** is a traditional distributed transaction protocol:

**Phase 1 - Prepare**:
\`\`\`
Coordinator: "Can you all commit?"
Service A: "Yes, I'm ready"
Service B: "Yes, I'm ready"  
Service C: "Yes, I'm ready"
\`\`\`

**Phase 2 - Commit**:
\`\`\`
Coordinator: "OK, everyone commit now!"
All services: Commit simultaneously
\`\`\`

**Problems in microservices**:

1. **Blocking**: Services hold locks while waiting for coordinator, reducing throughput
2. **Single point of failure**: If coordinator crashes, all services are stuck
3. **Not supported**: Many modern databases (NoSQL, cloud) don't support 2PC
4. **Latency**: Synchronous protocol adds significant latency
5. **Reduced availability**: System availability = product of all service availabilities

**Example**:
\`\`\`
If each service has 99.9% uptime:
4 services with 2PC = 0.999^4 = 99.6% uptime

Without 2PC (eventual consistency) = 99.9% uptime
\`\`\`

**Microservices prefer availability over consistency** (CAP theorem).

---

## The Saga Pattern

A **saga** is a sequence of local transactions where each service performs its work and publishes events. If one step fails, compensating transactions undo the previous steps.

**Key principles**:
1. Each service performs its local transaction
2. On success, publishes event to trigger next step
3. On failure, executes compensating transactions to rollback

**Two implementations**:
1. **Choreography**: Decentralized, event-driven
2. **Orchestration**: Centralized coordinator

---

## Choreography-Based Saga

Services communicate via events. Each service listens for events, does work, publishes next event.

**Example: E-commerce order**

**Happy path**:
\`\`\`
1. Order Service:     CreateOrder() → OrderCreated event
2. Inventory Service: (listens) → ReserveInventory() → InventoryReserved event
3. Payment Service:   (listens) → ChargeCard() → PaymentCompleted event
4. Shipping Service:  (listens) → ShipOrder() → OrderShipped event
\`\`\`

**Failure path** (payment fails):
\`\`\`
1. Order Service:     CreateOrder() → OrderCreated event
2. Inventory Service: ReserveInventory() → InventoryReserved event
3. Payment Service:   ChargeCard() → ❌ FAILS → PaymentFailed event
4. Inventory Service: (listens) → CancelReservation() → InventoryReleased event
5. Order Service:     (listens) → CancelOrder() → OrderCancelled event
\`\`\`

**Implementation**:
\`\`\`javascript
// Order Service
async function createOrder(orderData) {
    // Local transaction
    const order = await db.orders.insert({
        ...orderData,
        status: 'PENDING'
    });
    
    // Publish event
    await eventBus.publish('OrderCreated', {
        orderId: order.id,
        items: order.items,
        customerId: order.customerId
    });
    
    return order;
}

// Listen for failure events
eventBus.on('PaymentFailed', async (event) => {
    // Compensating transaction
    await db.orders.update(event.orderId, {
        status: 'CANCELLED'
    });
    
    await eventBus.publish('OrderCancelled', {
        orderId: event.orderId
    });
});

// Inventory Service
eventBus.on('OrderCreated', async (event) => {
    try {
        // Reserve inventory
        await db.inventory.update({
            productId: event.items[0].productId,
            reserved: { $inc: event.items[0].quantity }
        });
        
        await eventBus.publish('InventoryReserved', {
            orderId: event.orderId,
            items: event.items
        });
    } catch (error) {
        await eventBus.publish('InventoryReservationFailed', {
            orderId: event.orderId,
            reason: 'Out of stock'
        });
    }
});

// Compensating transaction
eventBus.on('PaymentFailed', async (event) => {
    await db.inventory.update({
        productId: event.items[0].productId,
        reserved: { $dec: event.items[0].quantity }
    });
    
    await eventBus.publish('InventoryReleased', {
        orderId: event.orderId
    });
});
\`\`\`

**Advantages**:
✅ Decentralized (no single point of failure)
✅ Services are loosely coupled
✅ Simple for basic flows

**Disadvantages**:
❌ Hard to understand flow (scattered across services)
❌ Difficult to debug and monitor
❌ Risk of cyclic dependencies
❌ Harder to add new steps

---

## Orchestration-Based Saga

A **saga orchestrator** coordinates the saga, telling each service what to do and handling failures.

**Example: Order orchestrator**

**Happy path**:
\`\`\`
Orchestrator:
  1. Tell Order Service: CreateOrder()
  2. Tell Inventory Service: ReserveInventory()
  3. Tell Payment Service: ChargeCard()
  4. Tell Shipping Service: ShipOrder()
  5. Mark saga COMPLETED
\`\`\`

**Failure path**:
\`\`\`
Orchestrator:
  1. Tell Order Service: CreateOrder() ✅
  2. Tell Inventory Service: ReserveInventory() ✅
  3. Tell Payment Service: ChargeCard() ❌ FAILS
  4. Rollback:
     - Tell Inventory Service: CancelReservation()
     - Tell Order Service: CancelOrder()
  5. Mark saga FAILED
\`\`\`

**Implementation**:
\`\`\`javascript
// Saga Orchestrator
class OrderSagaOrchestrator {
    async execute(orderData) {
        const sagaId = generateId();
        const sagaState = {
            id: sagaId,
            status: 'STARTED',
            steps: []
        };
        
        try {
            // Step 1: Create order
            const order = await orderService.createOrder(orderData);
            sagaState.steps.push({step: 'CreateOrder', status: 'COMPLETED', data: order});
            
            // Step 2: Reserve inventory
            await inventoryService.reserveInventory(order.items);
            sagaState.steps.push({step: 'ReserveInventory', status: 'COMPLETED'});
            
            // Step 3: Charge payment
            const payment = await paymentService.charge({
                amount: order.total,
                cardToken: orderData.paymentToken
            });
            sagaState.steps.push({step: 'ChargePayment', status: 'COMPLETED', data: payment});
            
            // Step 4: Ship order
            await shippingService.ship(order.id);
            sagaState.steps.push({step: 'ShipOrder', status: 'COMPLETED'});
            
            // Success
            sagaState.status = 'COMPLETED';
            await sagaRepository.save(sagaState);
            return order;
            
        } catch (error) {
            // Rollback in reverse order
            sagaState.status = 'ROLLING_BACK';
            await this.rollback(sagaState);
            throw error;
        }
    }
    
    async rollback(sagaState) {
        // Execute compensating transactions in reverse order
        const completedSteps = sagaState.steps.filter(s => s.status === 'COMPLETED').reverse();
        
        for (const step of completedSteps) {
            switch (step.step) {
                case 'CreateOrder':
                    await orderService.cancelOrder(step.data.id);
                    break;
                case 'ReserveInventory':
                    await inventoryService.releaseInventory(step.data.items);
                    break;
                case 'ChargePayment':
                    await paymentService.refund(step.data.paymentId);
                    break;
                // No compensation needed for ShipOrder (we failed before this)
            }
        }
        
        sagaState.status = 'ROLLED_BACK';
        await sagaRepository.save(sagaState);
    }
}
\`\`\`

**Advantages**:
✅ Centralized logic (easy to understand)
✅ Easy to add/remove steps
✅ Better monitoring and debugging
✅ Can implement timeouts, retries centrally

**Disadvantages**:
❌ Orchestrator can become single point of failure (mitigate with HA)
❌ Services coupled to orchestrator
❌ Orchestrator can become complex (god object)

---

## Compensating Transactions

**Compensating transaction**: Undo the effect of a previous transaction.

**Example**:

| Forward Transaction | Compensating Transaction |
|---------------------|-------------------------|
| Create order | Cancel order |
| Reserve inventory | Release inventory |
| Charge credit card | Refund |
| Send email | Send cancellation email |

**Important**: Compensations should be **idempotent** (can be executed multiple times safely).

**Semantic rollback vs Physical rollback**:
- **Physical**: Delete the order record (not recommended - lose history)
- **Semantic**: Mark order as CANCELLED (preferred - audit trail)

**Example**:
\`\`\`sql
-- ❌ Physical rollback
DELETE FROM orders WHERE id = 123;

-- ✅ Semantic rollback
UPDATE orders SET status = 'CANCELLED', cancelled_at = NOW() WHERE id = 123;
\`\`\`

---

## Handling Failures

### Retryable Failures (Transient)

Temporary issues that might succeed on retry:
- Network timeout
- Database temporarily unavailable
- Rate limit exceeded

**Solution**: Retry with exponential backoff

\`\`\`javascript
async function withRetry(fn, maxRetries = 3) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await fn();
        } catch (error) {
            if (i === maxRetries - 1) throw error;
            
            // Exponential backoff
            const delay = Math.pow(2, i) * 1000;
            await sleep(delay);
        }
    }
}

// Usage
await withRetry(() => paymentService.charge(payment));
\`\`\`

### Non-Retryable Failures (Permanent)

Business rule violations that won't succeed on retry:
- Insufficient funds
- Invalid card
- Out of stock

**Solution**: Execute compensating transactions immediately

---

## Saga State Machine

Track saga state to handle crashes and recovery.

**States**:
\`\`\`
STARTED → INVENTORY_RESERVED → PAYMENT_COMPLETED → SHIPPED → COMPLETED
                ↓                      ↓                ↓
              FAILED             COMPENSATING     COMPENSATING
                                      ↓                ↓
                                  ROLLED_BACK    ROLLED_BACK
\`\`\`

**Store state in database**:
\`\`\`sql
CREATE TABLE sagas (
    id UUID PRIMARY KEY,
    order_id UUID,
    status VARCHAR(50), -- STARTED, COMPLETED, FAILED, ROLLING_BACK, ROLLED_BACK
    current_step VARCHAR(100),
    steps JSONB, -- Array of completed steps
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
\`\`\`

**Recovery**: If saga orchestrator crashes, another instance can pick up and continue from last saved state.

---

## Saga Execution Coordinator (SEC) Pattern

When using orchestration, implement a **Saga Execution Coordinator**:

**Responsibilities**:
1. Execute saga steps in order
2. Store saga state after each step
3. Handle failures and trigger compensations
4. Implement timeouts
5. Provide monitoring and visibility

**Example (using state machine)**:
\`\`\`javascript
class SagaExecutionCoordinator {
    async run(sagaDefinition, data) {
        const saga = await this.createSaga(sagaDefinition, data);
        
        for (const step of sagaDefinition.steps) {
            try {
                // Execute step
                const result = await this.executeStep(step, saga);
                
                // Save state
                saga.steps.push({name: step.name, status: 'COMPLETED', result});
                saga.currentStep = step.name;
                await this.saveSaga(saga);
                
            } catch (error) {
                // Trigger compensation
                saga.status = 'FAILED';
                await this.compensate(saga);
                throw error;
            }
        }
        
        saga.status = 'COMPLETED';
        await this.saveSaga(saga);
        return saga;
    }
    
    async compensate(saga) {
        const completedSteps = saga.steps.filter(s => s.status === 'COMPLETED').reverse();
        
        for (const step of completedSteps) {
            const compensation = sagaDefinition.compensations[step.name];
            if (compensation) {
                await this.executeCompensation(compensation, step.result);
            }
        }
        
        saga.status = 'ROLLED_BACK';
        await this.saveSaga(saga);
    }
}
\`\`\`

---

## Real-World Example: Amazon Order

**Saga steps**:
1. **Order Service**: Create order (status: PENDING)
2. **Inventory Service**: Reserve items
3. **Payment Service**: Authorize payment (not capture yet)
4. **Shipping Service**: Calculate shipping, reserve slot
5. **Payment Service**: Capture payment (now that shipping confirmed)
6. **Notification Service**: Send confirmation email
7. **Order Service**: Mark order CONFIRMED

**Compensations if step 5 fails**:
- Shipping: Release slot
- Payment: Cancel authorization (it will expire anyway)
- Inventory: Release reserved items
- Order: Mark as CANCELLED
- Notification: Send cancellation email

**Why authorize then capture?**
- Don't charge customer until we're sure we can ship
- Authorization holds the funds but doesn't transfer them

---

## Eventual Consistency

Sagas provide **eventual consistency**, not immediate consistency.

**During saga execution**:
\`\`\`
Time T0: Order created (status: PENDING)
Time T1: Inventory reserved
Time T2: Payment processing...
Time T3: Payment succeeded (status: CONFIRMED)
\`\`\`

**Between T0 and T3, order is in inconsistent state** (created but not paid).

**Handling this**:
1. **Don't expose intermediate states to users**: Show "Processing..." to customer
2. **Status field**: Use status field to track saga progress
3. **Read your writes**: Query same service that wrote the data
4. **Version/timestamp**: Use to detect stale reads

---

## Saga vs 2PC Comparison

| Aspect | Two-Phase Commit (2PC) | Saga |
|--------|----------------------|------|
| **Isolation** | Locks held during transaction | No locks (eventual consistency) |
| **Availability** | Lower (blocking) | Higher (non-blocking) |
| **Consistency** | Strong (ACID) | Eventual |
| **Latency** | Higher (synchronous) | Lower (asynchronous) |
| **Rollback** | Automatic | Manual (compensating transactions) |
| **Complexity** | Database handles it | Application handles it |
| **Use case** | Monoliths, local services | Microservices, distributed systems |

---

## Interview Tips

**Red Flags**:
❌ Suggesting distributed ACID transactions
❌ Not mentioning compensating transactions
❌ Ignoring failure scenarios

**Good Responses**:
✅ Explain Saga pattern (choreography vs orchestration)
✅ Discuss eventual consistency trade-off
✅ Mention specific tools (temporal.io, Netflix Conductor)
✅ Explain compensating transactions

**Sample Answer**:
*"I'd use the Saga pattern with orchestration for complex flows. The order service would coordinate steps: create order, reserve inventory, charge payment, ship order. Each step is a local transaction in its service. If any step fails, we execute compensating transactions in reverse order (refund payment, release inventory, cancel order). We'd store saga state in a database for crash recovery. This provides eventual consistency rather than ACID, but gives us better availability and scalability. For simpler flows, I'd use choreography with event-driven communication."*

---

## Key Takeaways

1. **Distributed transactions** across microservices require special patterns
2. **2PC doesn't work** well in microservices (blocking, reduces availability)
3. **Saga pattern**: Sequence of local transactions with compensating transactions
4. **Choreography**: Decentralized, event-driven (good for simple flows)
5. **Orchestration**: Centralized coordinator (better for complex flows, easier to understand)
6. **Compensating transactions**: Semantic rollback, must be idempotent
7. **Eventual consistency**: Accept that system is temporarily inconsistent during saga
8. **Store saga state**: Enable crash recovery and monitoring`,
      quiz: [
        {
          id: 'q1-saga',
          question:
            "Explain why you can't use traditional database transactions across microservices. What problems would you encounter?",
          sampleAnswer:
            "Traditional database transactions (ACID) don't work across microservices because each service has its own database. You can't have a BEGIN TRANSACTION that spans multiple databases across the network. Problems: (1) Database locks can't span services, (2) Two-phase commit is blocking and reduces availability (if one service is down, all are blocked), (3) Many modern databases (NoSQL, cloud) don't support distributed transactions, (4) CAP theorem - must choose between consistency and availability; microservices prefer availability. Instead, use Saga pattern with eventual consistency and compensating transactions.",
          keyPoints: [
            'Each microservice has its own database (database per service pattern)',
            'Cannot use single transaction across network boundaries',
            '2PC is blocking and reduces system availability',
            'CAP theorem: microservices prefer availability over strong consistency',
            'Solution: Saga pattern with eventual consistency',
          ],
        },
        {
          id: 'q2-saga',
          question:
            'When would you use choreography vs orchestration for a saga? Give a specific example for each.',
          sampleAnswer:
            'Use choreography for simple, linear flows with few steps where services are already event-driven. Example: User registration - User Service creates user → publishes UserCreated → Email Service sends welcome email → Analytics Service tracks signup. Only 3 services, linear flow. Use orchestration for complex flows with conditional logic, multiple branches, or when you need centralized monitoring. Example: E-commerce checkout - Order Service orchestrator coordinates: create order → check inventory (if out of stock, cancel) → authorize payment (if fails, release inventory) → calculate shipping → capture payment → send confirmation. 6+ services with complex failure handling. Orchestration makes this easier to understand and debug.',
          keyPoints: [
            'Choreography: Simple, linear flows with few steps',
            'Choreography: Already event-driven architecture',
            'Orchestration: Complex flows with conditional logic',
            'Orchestration: Need centralized monitoring and debugging',
            'Orchestration: Easier to modify flow later',
          ],
        },
        {
          id: 'q3-saga',
          question:
            'Your saga fails at step 3 of 5. How do you handle rollback? What challenges might you face with compensating transactions?',
          sampleAnswer:
            "Execute compensating transactions in reverse order for completed steps: Step 2 compensation → Step 1 compensation. Store saga state after each step to know what to rollback. Challenges: (1) Compensations must be idempotent (if rollback crashes halfway, need to restart safely), (2) Some operations can't be fully compensated (email sent - can send \"sorry, cancelled\" but can't unsend), (3) Compensation might fail (payment refund fails - need retry with backoff), (4) Race conditions (user trying to use inventory that's being released), (5) Semantic vs physical compensation (mark order CANCELLED vs deleting order record). Store full audit trail, make compensations idempotent, implement retry logic, and prefer semantic compensation.",
          keyPoints: [
            'Execute compensating transactions in reverse order',
            'Store saga state to know what needs compensation',
            'Compensations must be idempotent (safe to retry)',
            "Some operations can't be fully compensated",
            'Prefer semantic compensation (status change) over physical (delete)',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-saga-1',
          question:
            "Why doesn't Two-Phase Commit (2PC) work well for microservices?",
          options: [
            "It's too simple and doesn't provide enough features",
            "It's blocking, reduces availability, and many modern databases don't support it",
            'It only works with SQL databases',
            "It's faster than Saga pattern",
          ],
          correctAnswer: 1,
          explanation:
            "2PC is blocking (services hold locks while waiting for coordinator), which reduces throughput and availability. If coordinator crashes, all services are stuck. Modern NoSQL and cloud databases often don't support 2PC. In distributed systems, availability is usually more important than strong consistency (CAP theorem). Saga pattern provides better availability with eventual consistency. Option 1 is wrong (2PC is actually complex). Option 3 is partially true but not the main reason. Option 4 is wrong (2PC is slower due to synchronous coordination).",
        },
        {
          id: 'mc-saga-2',
          question: 'In a Saga, what is a compensating transaction?',
          options: [
            'A faster version of a transaction that compensates for slow performance',
            'An undo operation that reverses the effects of a previous transaction',
            'A transaction that adds extra features to compensate for missing ones',
            'A payment to developers for extra work',
          ],
          correctAnswer: 1,
          explanation:
            'A compensating transaction is an undo operation that semantically reverses a previous transaction. Example: if the forward transaction is "Reserve Inventory", the compensating transaction is "Release Inventory". If "Charge Card", then "Refund Card". This is how Sagas handle failures - by rolling back completed steps using compensations. Compensations should be idempotent (safe to execute multiple times) and use semantic rollback (mark as CANCELLED) rather than physical rollback (DELETE record). Option 1 is nonsense. Option 3 confuses "compensating" with "complementing". Option 4 is a joke.',
        },
        {
          id: 'mc-saga-3',
          question:
            "What's the main difference between choreography-based and orchestration-based sagas?",
          options: [
            'Choreography is faster than orchestration',
            'Choreography is decentralized (event-driven), orchestration has a central coordinator',
            'Choreography only works with 2 services',
            "Orchestration can't handle failures",
          ],
          correctAnswer: 1,
          explanation:
            'Choreography is decentralized: services communicate via events, each service knows its next step. No central coordinator. Orchestration is centralized: an orchestrator tells each service what to do, coordinates the flow, and handles failures. Choreography is simpler for basic flows but harder to understand as complexity grows. Orchestration is easier to understand, debug, and modify, but requires a coordinator service (potential single point of failure). Option 1 is debatable (both can be fast). Option 3 is wrong (choreography works with any number). Option 4 is wrong (orchestration handles failures very well).',
        },
        {
          id: 'mc-saga-4',
          question:
            'Your e-commerce saga: CreateOrder → ReserveInventory → ChargePayment → Ship. Payment fails. What happens?',
          options: [
            'System crashes and needs manual intervention',
            'Execute compensating transactions: ReleaseInventory → CancelOrder',
            'Retry payment indefinitely until it succeeds',
            'Keep the order as-is and notify the admin',
          ],
          correctAnswer: 1,
          explanation:
            'When a saga step fails, execute compensating transactions in reverse order for all completed steps. PaymentFailed → execute compensation for ReserveInventory (ReleaseInventory) → execute compensation for CreateOrder (CancelOrder). This leaves the system in a consistent state. The order is marked CANCELLED, inventory is released, and customer is notified. Option 1 is wrong (sagas handle failures automatically). Option 3 is wrong (payment failure might be non-retryable like "insufficient funds"). Option 4 leaves inconsistent state (order without payment).',
        },
        {
          id: 'mc-saga-5',
          question: 'What type of consistency does the Saga pattern provide?',
          options: [
            'Strong consistency (ACID)',
            'Eventual consistency',
            'No consistency',
            'Immediate consistency',
          ],
          correctAnswer: 1,
          explanation:
            "Saga pattern provides eventual consistency. During saga execution, the system is temporarily in an inconsistent state (e.g., order created but payment not yet processed). Eventually, when the saga completes (either successfully or via compensations), the system reaches a consistent state. This is a trade-off: we sacrifice immediate consistency for better availability and scalability. Option 1 is wrong (sagas don't provide ACID guarantees). Option 3 is wrong (eventual consistency is still a form of consistency). Option 4 is wrong (there's a delay before consistency is achieved).",
        },
      ],
    },
    {
      id: 'data-management-microservices',
      title: 'Data Management in Microservices',
      content: `Data management is one of the most challenging aspects of microservices. The "database per service" pattern provides autonomy but introduces complexity in querying and consistency.

## Database Per Service Pattern

**Core Principle**: Each microservice owns its data and database. Other services cannot access it directly.

**Monolith**: Single shared database
\`\`\`
┌─────────────────────────────────┐
│    Application (Monolith)       │
└────────────┬────────────────────┘
             │
    ┌────────▼───────────┐
    │  Shared Database   │
    │ ┌────────────────┐ │
    │ │ orders         │ │
    │ │ users          │ │
    │ │ payments       │ │
    │ │ products       │ │
    │ └────────────────┘ │
    └────────────────────┘
\`\`\`

**Microservices**: Database per service
\`\`\`
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Order   │  │   User   │  │ Payment  │  │ Product  │
│ Service  │  │ Service  │  │ Service  │  │ Service  │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │              │              │
┌────▼────┐   ┌───▼────┐    ┌───▼────┐    ┌───▼────┐
│ Orders  │   │ Users  │    │Payments│    │Products│
│   DB    │   │   DB   │    │   DB   │    │   DB   │
└─────────┘   └────────┘    └────────┘    └────────┘
\`\`\`

**Why?**
✅ **Loose coupling**: Services can evolve databases independently
✅ **Tech diversity**: Use best database for each use case (SQL, NoSQL, graph)
✅ **Scalability**: Scale databases independently
✅ **Failure isolation**: Database failure affects only one service

**Challenges**:
❌ Can't use JOIN across services
❌ Distributed transactions (already covered via Saga)
❌ Data duplication
❌ Eventual consistency

---

## Implementing Database Per Service

### Option 1: Separate Database Instances

Each service has its own database server.

**Pros**: Complete isolation, can scale database independently
**Cons**: Higher cost, more operational overhead

\`\`\`
Order Service  → MySQL instance 1
User Service   → PostgreSQL instance 2
Product Service → MongoDB instance 3
\`\`\`

### Option 2: Separate Schemas

Services share database server but have separate schemas/databases.

**Pros**: Lower cost, easier to manage
**Cons**: Less isolation, database becomes coupling point

\`\`\`
MySQL Server:
├── order_db (Order Service)
├── user_db (User Service)
└── product_db (Product Service)
\`\`\`

**Best Practice**: Start with separate schemas, move to separate instances as you scale.

---

## Handling Cross-Service Queries

Without JOIN, how do you query data across services?

### Problem: Display Order with Product Details

**In Monolith**:
\`\`\`sql
SELECT o.*, p.name, p.price, p.image
FROM orders o
JOIN products p ON o.product_id = p.id
WHERE o.user_id = 123;
\`\`\`

**In Microservices**: Can't JOIN across services!

### Solution 1: API Composition

Application makes multiple API calls and combines results.

\`\`\`javascript
// Get orders from Order Service
const orders = await orderService.getOrdersByUser(userId);

// Extract product IDs
const productIds = orders.map(o => o.productId);

// Get product details from Product Service
const products = await productService.getByIds(productIds);

// Combine in application
const ordersWithProducts = orders.map(order => ({
    ...order,
    product: products.find(p => p.id === order.productId)
}));
\`\`\`

**Pros**: Simple, maintains service boundaries
**Cons**: Multiple round trips (latency), N+1 query problem

### Solution 2: Data Duplication (CQRS)

Store denormalized data for queries.

**Idea**: Order Service stores product name and price (not just ID)

\`\`\`javascript
// When creating order
const order = {
    id: generateId(),
    userId: userId,
    productId: product.id,
    productName: product.name,      // Duplicated!
    productPrice: product.price,    // Duplicated!
    productImage: product.image,    // Duplicated!
    status: 'PENDING'
};
\`\`\`

**Pros**: Fast queries (no extra calls), single service read
**Cons**: Data duplication, staleness (what if product name changes?)

**When to use**: For data that rarely changes or staleness is acceptable

**Updating duplicated data**:
- Product Service publishes ProductUpdated event
- Order Service listens and updates its copies

\`\`\`javascript
// Product Service
await productService.updateProduct(productId, {name: 'New Name'});
await eventBus.publish('ProductUpdated', {
    productId,
    name: 'New Name',
    price: 999
});

// Order Service (listener)
eventBus.on('ProductUpdated', async (event) => {
    // Update all orders with this product
    await db.orders.updateMany(
        {productId: event.productId},
        {
            productName: event.name,
            productPrice: event.price
        }
    );
});
\`\`\`

### Solution 3: CQRS with Read Models

Create specialized read databases optimized for queries.

**Pattern**: Command Query Responsibility Segregation (CQRS)

**Architecture**:
\`\`\`
Write Side:
  Order Service  → Orders DB (write)
  Product Service → Products DB (write)

Read Side:
  Events ↓
  OrderViewService → Order View DB (read-only, denormalized)
    Contains: orders + product details + user details
\`\`\`

**Implementation**:
\`\`\`javascript
// Order View Service
eventBus.on('OrderCreated', async (event) => {
    // Get additional data
    const product = await productService.get(event.productId);
    const user = await userService.get(event.userId);
    
    // Create denormalized read model
    await orderViewDB.insert({
        orderId: event.orderId,
        orderStatus: event.status,
        orderTotal: event.total,
        // Product details
        productId: product.id,
        productName: product.name,
        productPrice: product.price,
        productImage: product.image,
        // User details
        userId: user.id,
        userName: user.name,
        userEmail: user.email
    });
});

// Queries use read model
async function getOrderDetailsForUser(userId) {
    return await orderViewDB.find({userId});
    // Returns everything in one query!
}
\`\`\`

**Pros**: Fast queries, optimized for specific use cases
**Cons**: Eventual consistency, complexity, storage overhead

---

## Choosing the Right Database Per Service

Different services have different data needs.

### Relational (SQL)

**Use for**: Structured data, complex queries, transactions, strong consistency

**Examples**: PostgreSQL, MySQL

**Good for**:
- Order Service (needs ACID for order creation)
- Payment Service (financial transactions)
- User Service (structured user data)

\`\`\`sql
-- Order Service (PostgreSQL)
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    total DECIMAL(10,2),
    status VARCHAR(50),
    created_at TIMESTAMP
);

CREATE TABLE order_items (
    id UUID PRIMARY KEY,
    order_id UUID REFERENCES orders(id),
    product_id UUID,
    quantity INT,
    price DECIMAL(10,2)
);
\`\`\`

### Document (NoSQL)

**Use for**: Flexible schema, nested data, high write throughput

**Examples**: MongoDB, DynamoDB

**Good for**:
- Product Catalog Service (flexible product attributes)
- Content Service (blog posts, comments)
- Session Service (user sessions)

\`\`\`javascript
// Product Service (MongoDB)
{
    _id: "prod_123",
    name: "iPhone 15 Pro",
    price: 999,
    category: "Electronics",
    attributes: {
        // Flexible schema
        color: "Black",
        storage: "256GB",
        processor: "A17 Pro"
    },
    images: ["url1", "url2"],
    reviews: [
        {user: "user_1", rating: 5, comment: "Great phone!"}
    ]
}
\`\`\`

### Key-Value

**Use for**: Simple lookups, caching, sessions

**Examples**: Redis, DynamoDB

**Good for**:
- Cart Service (shopping carts)
- Session Service
- Cache Service

\`\`\`javascript
// Cart Service (Redis)
SET cart:user_123 '{"items": [{"productId": "prod_1", "quantity": 2}]}'
EXPIRE cart:user_123 3600  // Auto-delete after 1 hour
\`\`\`

### Graph

**Use for**: Relationships, recommendations, social graphs

**Examples**: Neo4j, Amazon Neptune

**Good for**:
- Social Network Service
- Recommendation Service

\`\`\`cypher
// Recommendation Service (Neo4j)
CREATE (u:User {id: 'user_123', name: 'John'})
CREATE (p:Product {id: 'prod_456', name: 'iPhone'})
CREATE (u)-[:PURCHASED]->(p)

// Find recommendations
MATCH (u:User {id: 'user_123'})-[:PURCHASED]->(p:Product)
      <-[:PURCHASED]-(other:User)-[:PURCHASED]->(recommendation:Product)
WHERE NOT (u)-[:PURCHASED]->(recommendation)
RETURN recommendation.name
\`\`\`

### Time-Series

**Use for**: Metrics, logs, IoT data

**Examples**: InfluxDB, TimescaleDB

**Good for**:
- Metrics Service
- Logging Service
- IoT Service

---

## Data Ownership and Boundaries

**Rule**: Each entity is owned by exactly ONE service.

**Example**:
- User Service owns users table
- Order Service owns orders table
- Product Service owns products table

**What about foreign keys?**

**In Monolith**:
\`\`\`sql
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),  -- Foreign key
    product_id UUID REFERENCES products(id)  -- Foreign key
);
\`\`\`

**In Microservices**: Can't have foreign keys across services!

**Solution**: Store IDs without foreign key constraints

\`\`\`sql
-- Order Service database
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,  -- No REFERENCES
    product_id UUID NOT NULL,  -- No REFERENCES
    status VARCHAR(50)
);
\`\`\`

**Referential integrity**: Handled by application logic, not database

\`\`\`javascript
// Before creating order, verify user and product exist
const user = await userService.getUser(userId);
if (!user) throw new Error('User not found');

const product = await productService.getProduct(productId);
if (!product) throw new Error('Product not found');

// Now create order
await orderService.createOrder({userId, productId, ...});
\`\`\`

---

## Handling Data Changes Across Services

### Problem: Product price changes

**Scenario**:
1. Order Service stores product price when order created: $99
2. Product Service updates price to $89
3. Old orders still show $99

**Is this a problem?** Usually NO! Historical orders should reflect price at purchase time.

**But what if we need updated data?**

### Solution: Event-Driven Updates

**Pattern**: Services subscribe to relevant events

\`\`\`javascript
// Product Service
async function updateProductPrice(productId, newPrice) {
    await db.products.update(productId, {price: newPrice});
    
    await eventBus.publish('ProductPriceChanged', {
        productId,
        oldPrice: 99,
        newPrice: 89,
        changedAt: new Date()
    });
}

// Order Service (if it needs current prices)
eventBus.on('ProductPriceChanged', async (event) => {
    // Update cached product info for pending orders
    await db.orders.updateMany(
        {productId: event.productId, status: 'PENDING'},
        {currentProductPrice: event.newPrice}
    );
});

// Inventory Service (adjust reorder calculations)
eventBus.on('ProductPriceChanged', async (event) => {
    await recalculateReorderPoint(event.productId);
});
\`\`\`

---

## Shared Data Anti-Patterns

### ❌ Shared Database

**Problem**: Multiple services access the same database

\`\`\`
Order Service  ─┐
User Service   ─┼─→ Shared Database
Product Service─┘
\`\`\`

**Why it's bad**:
- Tight coupling (schema change breaks all services)
- Can't choose different database types
- Scaling nightmare
- Single point of failure

### ❌ Shared Tables

**Problem**: Multiple services read/write same table

\`\`\`
Order Service writes orders table
Reporting Service reads orders table
\`\`\`

**Why it's bad**: Breaks encapsulation, creates hidden dependencies

**Better**: Reporting Service subscribes to OrderCreated events

---

## Transaction Boundaries

**Rule**: Transactions should not span services.

**Example**:

❌ **Bad** (distributed transaction):
\`\`\`
BEGIN TRANSACTION;
  INSERT INTO orders (...);  -- Order Service DB
  UPDATE inventory (...);     -- Inventory Service DB
  INSERT INTO payments (...); -- Payment Service DB
COMMIT;
\`\`\`

✅ **Good** (Saga pattern with local transactions):
\`\`\`
// Order Service
BEGIN TRANSACTION;
  INSERT INTO orders (...);
COMMIT;
Publish OrderCreated event

// Inventory Service
Listen to OrderCreated
BEGIN TRANSACTION;
  UPDATE inventory (...);
COMMIT;
Publish InventoryReserved event

// Payment Service
Listen to InventoryReserved
BEGIN TRANSACTION;
  INSERT INTO payments (...);
COMMIT;
Publish PaymentCompleted event
\`\`\`

---

## Data Migration Strategies

Moving from monolith to microservices?

### Strangler Fig Pattern

**Gradually** extract services while keeping shared database initially.

**Phase 1**: Extract service but keep shared DB
\`\`\`
Monolith     ─┐
Order Service─┼─→ Shared Database
\`\`\`

**Phase 2**: Replicate data to service's own DB
\`\`\`
Monolith     ─┬─→ Shared Database
Order Service─┼─→ Shared Database
              └─→ Orders DB (read-only copy)
\`\`\`

**Phase 3**: Service writes to its own DB, syncs to shared DB
\`\`\`
Monolith     ───→ Shared Database ←─┐
Order Service──→ Orders DB ─────────┘ (sync)
\`\`\`

**Phase 4**: Cut over completely
\`\`\`
Monolith     ───→ Shared Database
Order Service──→ Orders DB (independent)
\`\`\`

---

## Interview Tips

**Red Flags**:
❌ Suggesting shared database for microservices
❌ Not mentioning eventual consistency
❌ Using JOINs across services

**Good Responses**:
✅ Explain database per service pattern
✅ Discuss trade-offs (autonomy vs consistency)
✅ Mention solutions (API composition, CQRS, events)
✅ Talk about data ownership

**Sample Answer**:
*"I'd implement the database per service pattern, where each microservice owns its data and database. This provides loose coupling and allows us to choose the best database for each use case - PostgreSQL for Order Service (ACID), MongoDB for Product Catalog (flexible schema), Redis for Cart Service (fast lookups). For cross-service queries like 'orders with product details', I'd use API composition for simple cases or CQRS with read models for complex queries. Data consistency is eventual, not immediate, which is acceptable for this use case. Services communicate data changes via events."*

---

## Key Takeaways

1. **Database per service**: Each service owns its data and database
2. **No JOINs**: Can't query across services directly
3. **Solutions**: API composition, data duplication, CQRS
4. **Choose right DB**: SQL for structured, NoSQL for flexible, graph for relationships
5. **Event-driven**: Services communicate data changes via events
6. **Eventual consistency**: Accept temporary inconsistency
7. **No shared databases**: Anti-pattern that creates tight coupling
8. **Local transactions**: Transactions don't span services (use Saga)`,
      quiz: [
        {
          id: 'q1-data',
          question:
            "Why can't you use database JOINs in microservices? How do you query data that spans multiple services?",
          sampleAnswer:
            "JOINs don't work in microservices because each service has its own database - you can't JOIN across network boundaries between separate databases. Solutions: (1) API Composition - application makes multiple API calls and combines results in memory (simple but has latency), (2) Data Duplication - store denormalized data in each service (Order Service stores product name/price, not just ID), update via events, (3) CQRS with Read Models - create specialized read databases that aggregate data from multiple services via events. Choose based on query patterns and consistency requirements. API composition for simple queries, CQRS for complex dashboards, data duplication for frequently accessed data.",
          keyPoints: [
            'Each service has own database (database per service)',
            "Can't JOIN across network/database boundaries",
            'API Composition: multiple calls, combine in app',
            'Data Duplication: store denormalized copies, update via events',
            'CQRS: dedicated read models for complex queries',
          ],
        },
        {
          id: 'q2-data',
          question:
            'What are the trade-offs of the "database per service" pattern? When might you bend this rule?',
          sampleAnswer:
            'Trade-offs: Pros: (1) Loose coupling (services evolve independently), (2) Technology diversity (best DB for each use case), (3) Independent scaling, (4) Fault isolation. Cons: (1) No JOINs across services, (2) Distributed transactions require Saga pattern, (3) Data duplication, (4) Eventual consistency instead of immediate. Bend the rule when: Starting from monolith (Strangler Fig - extract services gradually while sharing DB initially), read-only sharing for analytics/reporting (but better to use events), very early startup (pragmatic to start simple). However, plan migration path to true database per service as you scale.',
          keyPoints: [
            'Pros: autonomy, tech diversity, scaling, isolation',
            'Cons: no JOINs, distributed transactions, duplication, eventual consistency',
            'Bend for: monolith migration (Strangler Fig), analytics',
            'Always plan migration to true database per service',
            "Don't share for write operations (breaks encapsulation)",
          ],
        },
        {
          id: 'q3-data',
          question:
            'Your Order Service stores product price. Product Service updates the price. Should old orders update? How do you handle this?',
          sampleAnswer:
            "Usually old orders should NOT update - they should show the historical price at purchase time (auditing, receipts). For pending orders, it depends on business rules. Implementation: (1) Store snapshot of product data when order created (productName, productPrice at time of order), (2) Product Service publishes ProductPriceChanged event, (3) Order Service can listen and update pending orders if needed, but completed orders stay unchanged. Use status field to determine: ORDER_STATUS = COMPLETED → don't update (historical), ORDER_STATUS = PENDING → optionally update if business requires. This is data duplication with event-driven updates - eventual consistency is fine here.",
          keyPoints: [
            'Historical orders should show price at purchase time',
            'Store snapshot of product data when creating order',
            'Product Service publishes ProductPriceChanged event',
            'Order Service updates based on order status (pending vs completed)',
            'Event-driven updates maintain eventual consistency',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-data-1',
          question:
            'What is the main advantage of the "database per service" pattern?',
          options: [
            'Makes queries easier with JOINs across services',
            'Provides loose coupling - services can evolve databases independently',
            'Eliminates the need for backups',
            'Guarantees strong consistency across services',
          ],
          correctAnswer: 1,
          explanation:
            'Database per service provides loose coupling - each service owns its data and can evolve its schema independently without breaking other services. Services can also choose different database technologies (SQL, NoSQL, graph) based on their needs. Option 1 is wrong (database per service makes JOINs impossible). Option 3 is wrong (still need backups). Option 4 is wrong (database per service provides eventual consistency, not strong consistency).',
        },
        {
          id: 'mc-data-2',
          question:
            'You need to display orders with product details. Each service has its own database. What approach should you use?',
          options: [
            'Use SQL JOIN across both databases',
            'Share the database between Order and Product services',
            'Use API composition (get orders, then get products) or data duplication (store product details in Order Service)',
            'Give Order Service direct access to Product database',
          ],
          correctAnswer: 2,
          explanation:
            'Use API composition (Order Service calls Product Service to get details) for simple cases, or data duplication (Order Service stores product name/price when order created) for better performance. Update duplicated data via events when products change. Option 1 is impossible (different databases). Option 2 breaks database per service pattern. Option 4 violates encapsulation and creates tight coupling. API composition or data duplication are the correct microservices patterns.',
        },
        {
          id: 'mc-data-3',
          question:
            'What is CQRS in the context of microservices data management?',
          options: [
            'A type of database that supports microservices',
            'Command Query Responsibility Segregation - separate read and write models',
            'A security pattern for encrypting data',
            'A caching layer for databases',
          ],
          correctAnswer: 1,
          explanation:
            'CQRS (Command Query Responsibility Segregation) separates read and write models. Write side: services write to their own databases. Read side: dedicated read models (denormalized databases) optimized for queries, updated via events from write side. Example: OrderViewService subscribes to OrderCreated, ProductUpdated events and maintains a denormalized view combining order + product + user data for fast queries. Solves the "no JOIN" problem in microservices. Option 1 is wrong (it\'s a pattern, not a database). Options 3 and 4 are unrelated concepts.',
        },
        {
          id: 'mc-data-4',
          question:
            'Which database type would you choose for a Product Catalog service with flexible attributes that vary by product type?',
          options: [
            'Relational (PostgreSQL) - for strong schema',
            'Document (MongoDB) - for flexible schema',
            'Graph (Neo4j) - for relationships',
            'Time-series (InfluxDB) - for metrics',
          ],
          correctAnswer: 1,
          explanation:
            'Document databases like MongoDB are perfect for flexible schemas where attributes vary significantly. Example: electronics have (processor, RAM), clothing has (size, color, material), books have (author, ISBN). With MongoDB, each product can have different attributes without schema changes. PostgreSQL would require either EAV pattern (slow) or JSON columns (less structured). Graph databases are for relationship-heavy data. Time-series is for temporal data. Choose the database that matches your data model.',
        },
        {
          id: 'mc-data-5',
          question: 'What is the Strangler Fig pattern in data migration?',
          options: [
            'Delete old data after migrating to new database',
            'Gradually extract services while initially sharing database, then split databases incrementally',
            'Replicate all data to new database immediately',
            'Use database triggers to sync data',
          ],
          correctAnswer: 1,
          explanation:
            "Strangler Fig is a gradual migration pattern: Phase 1 - extract service but keep shared DB, Phase 2 - replicate data to service's own DB, Phase 3 - service writes to own DB and syncs back, Phase 4 - cut over completely. This allows safe, incremental migration from monolith to microservices without big-bang rewrites. Named after strangler fig vines that gradually replace host trees. Option 1 loses data. Option 3 is risky (big bang). Option 4 creates tight coupling.",
        },
      ],
    },
    {
      id: 'circuit-breaker',
      title: 'Circuit Breaker Pattern',
      content: `The Circuit Breaker pattern prevents cascading failures in distributed systems by failing fast when a service is down, rather than waiting for timeouts.

## The Problem: Cascading Failures

**Scenario**: Payment Service is down

**Without Circuit Breaker**:
\`\`\`
1. User clicks "Checkout"
2. Order Service calls Payment Service
3. Payment Service is down
4. Order Service waits 30 seconds (timeout)
5. Returns error to user
6. 100 users click checkout simultaneously
7. 100 threads in Order Service blocked for 30 seconds
8. Order Service runs out of threads → crashes
9. Now both Order and Payment Service are down! 💥
\`\`\`

This is a **cascading failure** - one service failure brings down others.

**With Circuit Breaker**:
\`\`\`
1. First few requests to Payment Service fail
2. Circuit breaker "opens" - stops calling Payment Service
3. Subsequent requests fail immediately (no waiting)
4. Order Service stays healthy
5. After timeout, circuit breaker "half-opens" - allows test request
6. If test succeeds, circuit "closes" - normal operation resumes
\`\`\`

---

## Circuit Breaker States

Circuit breaker acts like an electrical circuit breaker.

### State Machine:

\`\`\`
        ┌──────────┐
        │  CLOSED  │  (Normal operation)
        │  ✅      │
        └────┬─────┘
             │ Failures exceed threshold
             ▼
        ┌──────────┐
        │   OPEN   │  (Fail fast)
        │  ❌      │
        └────┬─────┘
             │ After timeout period
             ▼
        ┌──────────┐
        │HALF-OPEN │  (Testing)
        │  ⚠️      │
        └────┬─────┘
          │      │
    Success│      │More failures
          │      │
          ▼      ▼
      CLOSED   OPEN
\`\`\`

### 1. CLOSED (Normal Operation)

**Behavior**: All requests pass through to service

**Tracking**: Count failures

**Transition**: If failures exceed threshold → OPEN

\`\`\`javascript
// Example: 5 failures in 10 seconds
if (failureCount >= 5 && timeWindow <= 10s) {
    state = OPEN;
}
\`\`\`

### 2. OPEN (Failing Fast)

**Behavior**: All requests fail immediately (don't call service)

**Response**: Return error or fallback value

**Transition**: After timeout (e.g., 30 seconds) → HALF_OPEN

\`\`\`javascript
// Fail immediately
if (state === OPEN) {
    throw new Error('Circuit breaker open');
}
\`\`\`

### 3. HALF_OPEN (Testing)

**Behavior**: Allow limited requests through (test if service recovered)

**Transition**: 
- If test succeeds → CLOSED (resume normal operation)
- If test fails → OPEN (back to failing fast)

\`\`\`javascript
if (state === HALF_OPEN) {
    if (testRequest succeeds) {
        state = CLOSED;
        resetFailureCount();
    } else {
        state = OPEN;
        startTimeout();
    }
}
\`\`\`

---

## Implementation

### Basic Circuit Breaker

\`\`\`javascript
class CircuitBreaker {
    constructor(options = {}) {
        this.failureThreshold = options.failureThreshold || 5;
        this.timeout = options.timeout || 60000; // 60 seconds
        this.resetTimeout = options.resetTimeout || 30000; // 30 seconds
        
        this.state = 'CLOSED';
        this.failureCount = 0;
        this.nextAttempt = Date.now();
        this.successCount = 0;
    }
    
    async call(fn) {
        // Check state
        if (this.state === 'OPEN') {
            if (Date.now() < this.nextAttempt) {
                throw new Error('Circuit breaker is OPEN');
            }
            // Time to try again
            this.state = 'HALF_OPEN';
        }
        
        try {
            // Execute function
            const result = await this.executeWithTimeout(fn, this.timeout);
            
            // Success!
            this.onSuccess();
            return result;
            
        } catch (error) {
            // Failure
            this.onFailure();
            throw error;
        }
    }
    
    async executeWithTimeout(fn, timeout) {
        return Promise.race([
            fn(),
            new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Timeout')), timeout)
            )
        ]);
    }
    
    onSuccess() {
        this.failureCount = 0;
        
        if (this.state === 'HALF_OPEN') {
            this.successCount++;
            // Require N successes to fully close
            if (this.successCount >= 3) {
                this.state = 'CLOSED';
                this.successCount = 0;
            }
        }
    }
    
    onFailure() {
        this.failureCount++;
        this.successCount = 0;
        
        if (this.failureCount >= this.failureThreshold) {
            this.state = 'OPEN';
            this.nextAttempt = Date.now() + this.resetTimeout;
        }
    }
    
    getState() {
        return this.state;
    }
}
\`\`\`

### Usage

\`\`\`javascript
const paymentServiceBreaker = new CircuitBreaker({
    failureThreshold: 5,     // Open after 5 failures
    timeout: 3000,           // Request timeout: 3s
    resetTimeout: 30000      // Try again after 30s
});

async function chargePayment(paymentData) {
    try {
        return await paymentServiceBreaker.call(async () => {
            return await paymentService.charge(paymentData);
        });
    } catch (error) {
        if (error.message === 'Circuit breaker is OPEN') {
            // Handle gracefully
            return { status: 'PENDING', message: 'Payment service temporarily unavailable' };
        }
        throw error;
    }
}
\`\`\`

---

## Fallback Strategies

When circuit breaker is OPEN, what do you return?

### 1. Default Value

Return sensible default.

\`\`\`javascript
async function getRecommendations(userId) {
    try {
        return await circuitBreaker.call(() => 
            recommendationService.get(userId)
        );
    } catch (error) {
        // Fallback: return popular products
        return await productService.getPopular();
    }
}
\`\`\`

### 2. Cached Value

Return last known good value.

\`\`\`javascript
const cache = new Map();

async function getProductPrice(productId) {
    try {
        const price = await circuitBreaker.call(() => 
            pricingService.getPrice(productId)
        );
        cache.set(productId, price);  // Update cache
        return price;
    } catch (error) {
        // Fallback: return cached price
        const cachedPrice = cache.get(productId);
        if (cachedPrice) {
            return { ...cachedPrice, stale: true };
        }
        throw error;
    }
}
\`\`\`

### 3. Degraded Functionality

Reduce features rather than complete failure.

\`\`\`javascript
async function getProductDetails(productId) {
    const product = await productService.get(productId);
    
    // Try to get recommendations (non-critical)
    try {
        product.recommendations = await circuitBreaker.call(() =>
            recommendationService.getRelated(productId)
        );
    } catch (error) {
        // Degrade gracefully - product still works without recommendations
        product.recommendations = [];
    }
    
    return product;
}
\`\`\`

### 4. Queue for Later

Queue request for processing when service recovers.

\`\`\`javascript
async function sendNotification(notification) {
    try {
        return await circuitBreaker.call(() => 
            notificationService.send(notification)
        );
    } catch (error) {
        // Fallback: queue for later
        await notificationQueue.enqueue(notification);
        return { status: 'QUEUED' };
    }
}
\`\`\`

---

## Real-World Example: Netflix Hystrix

**Hystrix** is Netflix's circuit breaker library (now in maintenance mode, but concepts remain).

**Features**:
- Circuit breaker
- Timeouts
- Fallbacks
- Request collapsing
- Real-time monitoring dashboard

**Example (Java)**:
\`\`\`java
@HystrixCommand(
    fallbackMethod = "getDefaultRecommendations",
    commandProperties = {
        @HystrixProperty(name = "execution.isolation.thread.timeoutInMilliseconds", value = "3000"),
        @HystrixProperty(name = "circuitBreaker.requestVolumeThreshold", value = "20"),
        @HystrixProperty(name = "circuitBreaker.errorThresholdPercentage", value = "50"),
        @HystrixProperty(name = "circuitBreaker.sleepWindowInMilliseconds", value = "60000")
    }
)
public List<Movie> getRecommendations(String userId) {
    return recommendationService.getForUser(userId);
}

public List<Movie> getDefaultRecommendations(String userId) {
    // Fallback: return popular movies
    return movieService.getPopular();
}
\`\`\`

**Configuration Explained**:
- timeout: 3 seconds
- requestVolumeThreshold: Minimum 20 requests before circuit can open
- errorThresholdPercentage: Open circuit if 50% of requests fail
- sleepWindowInMilliseconds: Wait 60s before trying again (HALF_OPEN)

---

## Modern Alternative: Resilience4j

**Resilience4j** is a lightweight alternative to Hystrix.

**Features**:
- Circuit breaker
- Rate limiter
- Bulkhead (limit concurrent calls)
- Retry
- Time limiter

**Example (Java)**:
\`\`\`java
CircuitBreakerConfig config = CircuitBreakerConfig.custom()
    .failureRateThreshold(50)                    // 50% failure rate
    .waitDurationInOpenState(Duration.ofSeconds(30))  // Wait 30s
    .slidingWindowSize(20)                       // Last 20 requests
    .build();

CircuitBreaker circuitBreaker = CircuitBreaker.of("paymentService", config);

// Decorate function
Supplier<PaymentResult> decoratedSupplier = CircuitBreaker
    .decorateSupplier(circuitBreaker, () -> paymentService.charge(payment));

// Execute
try {
    PaymentResult result = decoratedSupplier.get();
} catch (CallNotPermittedException e) {
    // Circuit breaker is open
    return handleFallback();
}
\`\`\`

**JavaScript equivalent** (using opossum library):
\`\`\`javascript
const CircuitBreaker = require('opossum');

const options = {
    timeout: 3000,              // 3 second timeout
    errorThresholdPercentage: 50,  // Open at 50% failure
    resetTimeout: 30000         // Try again after 30s
};

const breaker = new CircuitBreaker(paymentService.charge, options);

// Add fallback
breaker.fallback(() => ({status: 'PENDING', message: 'Service unavailable'}));

// Add listeners
breaker.on('open', () => console.log('Circuit opened!'));
breaker.on('halfOpen', () => console.log('Circuit half-open, testing...'));
breaker.on('close', () => console.log('Circuit closed!'));

// Use it
const result = await breaker.fire(paymentData);
\`\`\`

---

## Monitoring Circuit Breakers

**Metrics to track**:
1. **Circuit state**: CLOSED, OPEN, HALF_OPEN
2. **Success rate**: % of successful requests
3. **Failure rate**: % of failed requests
4. **Latency**: Response time distribution (p50, p95, p99)
5. **Timeout rate**: % of requests that timeout
6. **Fallback rate**: % of requests using fallback

**Dashboard Example**:
\`\`\`
Service: payment-service
State: OPEN ❌
Failures: 47 / 50 (94%)
Uptime: 12min since last transition
Next attempt: in 18 seconds

Last 100 requests:
✅✅✅❌❌❌❌❌❌❌ ... (mostly failures)

Fallback responses: 235
\`\`\`

**Alerting**:
\`\`\`
ALERT: Circuit breaker for payment-service has been OPEN for > 5 minutes
Action: Investigate payment-service health
\`\`\`

---

## Circuit Breaker vs Retry

**Circuit Breaker**: Stop calling failing service
**Retry**: Try again on failure

**They complement each other!**

\`\`\`javascript
// Retry THEN circuit breaker
const retryPolicy = {
    maxRetries: 3,
    backoff: 'exponential'
};

async function callServiceWithResilience(data) {
    return await circuitBreaker.call(async () => {
        return await retryWithBackoff(async () => {
            return await service.call(data);
        }, retryPolicy);
    });
}
\`\`\`

**When to use each**:

| Pattern | Use Case |
|---------|----------|
| **Retry** | Transient failures (network blip) |
| **Circuit Breaker** | Service down or degraded |
| **Both** | Retry for transient issues, circuit breaker to stop retry storm |

---

## Best Practices

### 1. Tune Thresholds

Don't use default values blindly.

**Consider**:
- **Request volume**: Don't open circuit on 2 failures if you only get 10 req/min
- **Error rate**: 50% might be too aggressive for some services
- **Recovery time**: How long does service need to recover?

**Example**:
\`\`\`javascript
// Critical service - more lenient
const paymentBreaker = new CircuitBreaker({
    failureThreshold: 10,      // Higher threshold
    resetTimeout: 10000        // Retry sooner
});

// Non-critical service - stricter
const recommendationBreaker = new CircuitBreaker({
    failureThreshold: 3,       // Lower threshold
    resetTimeout: 60000        // Wait longer
});
\`\`\`

### 2. Bulkheading

Isolate circuit breakers per dependency.

❌ **Bad**: Single circuit breaker for entire service
\`\`\`javascript
// All external calls share one breaker
const breaker = new CircuitBreaker();
await breaker.call(() => paymentService.charge());
await breaker.call(() => emailService.send());
\`\`\`

**Problem**: Email service failure opens circuit for payment too!

✅ **Good**: Separate breakers per dependency
\`\`\`javascript
const paymentBreaker = new CircuitBreaker();
const emailBreaker = new CircuitBreaker();

await paymentBreaker.call(() => paymentService.charge());
await emailBreaker.call(() => emailService.send());
\`\`\`

### 3. Fail Fast

Don't retry when circuit is open.

❌ **Bad**:
\`\`\`javascript
for (let i = 0; i < 3; i++) {
    try {
        return await breaker.call(fn);
    } catch (error) {
        // Circuit is open, but we keep retrying! 🤦
    }
}
\`\`\`

✅ **Good**:
\`\`\`javascript
try {
    return await breaker.call(fn);
} catch (error) {
    if (error.message === 'Circuit breaker is OPEN') {
        return fallback();  // Don't retry
    }
    throw error;
}
\`\`\`

### 4. Monitor and Alert

Circuit breaker state changes are important signals.

\`\`\`javascript
breaker.on('open', () => {
    logger.error('Circuit breaker opened for payment-service');
    metrics.increment('circuit_breaker.opened');
    alerting.notify('Payment service is down!');
});

breaker.on('halfOpen', () => {
    logger.info('Circuit breaker half-open, testing payment-service');
});

breaker.on('close', () => {
    logger.info('Circuit breaker closed, payment-service recovered');
    metrics.increment('circuit_breaker.closed');
});
\`\`\`

---

## Interview Tips

**Red Flags**:
❌ Not knowing about cascading failures
❌ Thinking circuit breaker is same as retry
❌ Not mentioning fallback strategies

**Good Responses**:
✅ Explain the three states (CLOSED, OPEN, HALF_OPEN)
✅ Discuss cascading failure prevention
✅ Mention fallback strategies
✅ Talk about monitoring

**Sample Answer**:
*"I'd implement circuit breakers to prevent cascading failures. When Payment Service fails repeatedly (e.g., 5 failures in 10 seconds), the circuit breaker opens and fails fast instead of waiting for timeouts. This prevents Order Service from running out of threads. The circuit stays open for 30 seconds, then enters half-open state to test if the service recovered. For fallbacks, I'd queue payment requests or show 'Payment pending' to users. We'd monitor circuit breaker state changes and alert on-call engineers when circuits open. I'd use a library like Resilience4j (Java) or Opossum (Node.js) rather than building from scratch."*

---

## Key Takeaways

1. **Circuit breaker** prevents cascading failures by failing fast
2. **Three states**: CLOSED (normal), OPEN (failing fast), HALF_OPEN (testing)
3. **Fail fast**: Don't wait for timeout when service is known to be down
4. **Fallbacks**: Default values, cached data, degraded functionality, or queuing
5. **Monitor**: Track state changes, failure rates, fallback usage
6. **Tune thresholds**: Based on traffic patterns and service criticality
7. **Bulkhead**: Separate circuit breakers per dependency
8. **Use with retry**: Retry for transient failures, circuit breaker for sustained outages`,
      quiz: [
        {
          id: 'q1-circuit',
          question:
            'Explain how circuit breakers prevent cascading failures. What would happen WITHOUT a circuit breaker when a downstream service fails?',
          sampleAnswer:
            "Without circuit breaker: When Payment Service goes down, Order Service keeps trying to call it. Each request waits for timeout (e.g., 30 seconds). With 100 concurrent users, 100 threads get blocked waiting. Order Service runs out of threads → can't handle new requests → crashes. Now both services are down (cascading failure). WITH circuit breaker: After N failures, circuit opens → subsequent requests fail immediately (no waiting) → Order Service stays healthy → circuit half-opens after timeout → tests if Payment Service recovered → closes if test succeeds. Circuit breaker isolates failure to one service, prevents thread pool exhaustion, and allows graceful degradation.",
          keyPoints: [
            'Without CB: threads block waiting for timeout → thread pool exhaustion',
            'Cascading failure: one service failure brings down others',
            'Circuit breaker fails fast → no thread blocking',
            'Allows graceful degradation with fallbacks',
            'Auto-recovery via half-open state testing',
          ],
        },
        {
          id: 'q2-circuit',
          question:
            'Describe the three states of a circuit breaker and the transitions between them.',
          sampleAnswer:
            'CLOSED (normal): All requests pass through. Track failures. Transition: If failures >= threshold → OPEN. OPEN (failing fast): All requests fail immediately without calling service. Saves resources. Transition: After timeout period → HALF_OPEN. HALF_OPEN (testing): Allow limited test requests through. Transition: If test succeeds → CLOSED (resume normal), if fails → OPEN (back to failing fast). Example: threshold=5 failures, timeout=30s. After 5 failures, circuit opens. For 30 seconds, all requests fail fast. Then circuit half-opens, sends test request. If test succeeds, circuit closes; otherwise back to open for another 30s.',
          keyPoints: [
            'CLOSED: normal operation, tracking failures',
            'OPEN: fail fast, no calls to service',
            'HALF_OPEN: testing recovery with limited requests',
            'Transitions based on failure threshold and timeout',
            'Auto-recovery mechanism prevents manual intervention',
          ],
        },
        {
          id: 'q3-circuit',
          question:
            'What are fallback strategies for when a circuit breaker is OPEN? Give examples for different scenarios.',
          sampleAnswer:
            'Fallback strategies: (1) Default/Popular values - Recommendation service down → return popular products instead of personalized. (2) Cached data - Pricing service down → return last known prices (mark as potentially stale). (3) Degraded functionality - Product page works but without recommendations. (4) Queue for later - Notification service down → queue emails for retry when service recovers. (5) Error with retry - Critical operation → return error, ask user to try again. Choose based on: Is operation critical? (payment = yes, recommendations = no), Can we show stale data? (prices = maybe with disclaimer, inventory = risky), Can we defer? (notifications = yes, checkout = no).',
          keyPoints: [
            'Default/popular values for non-critical features',
            'Cached data with staleness indicators',
            'Degraded functionality (reduce features)',
            'Queue for asynchronous operations',
            'Choose based on criticality and data freshness requirements',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-circuit-1',
          question: 'What is the primary purpose of a circuit breaker pattern?',
          options: [
            'To retry failed requests automatically',
            'To prevent cascading failures by failing fast when a service is down',
            'To load balance requests across multiple service instances',
            'To cache responses from slow services',
          ],
          correctAnswer: 1,
          explanation:
            "Circuit breaker prevents cascading failures by detecting when a service is failing and failing fast (returning error immediately) instead of waiting for timeouts. This prevents thread pool exhaustion in the calling service. When a downstream service fails repeatedly, the circuit \"opens\" and stops calling it, allowing the calling service to stay healthy. Option 1 is wrong (that's retry pattern). Option 3 is wrong (that's load balancer). Option 4 is wrong (that's caching layer). Circuit breaker is specifically about preventing cascading failures.",
        },
        {
          id: 'mc-circuit-2',
          question:
            'A circuit breaker is in OPEN state. What happens when a new request arrives?',
          options: [
            'Request is queued until service recovers',
            'Request is retried with exponential backoff',
            'Request fails immediately without calling the service',
            'Request is sent to service normally',
          ],
          correctAnswer: 2,
          explanation:
            "When circuit is OPEN, requests fail immediately (fail fast) without calling the downstream service. This prevents wasting resources (threads, connections) on calls that will likely fail. The circuit stays open for a configured timeout period, then transitions to HALF_OPEN to test if service recovered. Option 1 is not automatic (you could implement this as a fallback). Option 2 is wrong (no retries when circuit is open). Option 4 is wrong (that's CLOSED state).",
        },
        {
          id: 'mc-circuit-3',
          question:
            'Your Order Service calls Payment Service (critical) and Recommendation Service (non-critical). How should you configure circuit breakers?',
          options: [
            'Use one circuit breaker for both services',
            "Don't use circuit breakers for critical services",
            'Use separate circuit breakers: lenient for Payment (higher threshold, retry sooner), stricter for Recommendations (lower threshold, wait longer)',
            'Only use circuit breaker for Payment Service',
          ],
          correctAnswer: 2,
          explanation:
            'Use separate circuit breakers per dependency (bulkheading pattern). Configure based on criticality: Payment Service gets more lenient settings (higher failure threshold, shorter reset timeout) because we want to give it more chances before opening. Recommendations get stricter settings because failures are less critical. Option 1 is wrong (email failure would open circuit for payment too). Option 2 is wrong (critical services especially need circuit breakers). Option 4 is incomplete (recommendations also benefit from circuit breaker).',
        },
        {
          id: 'mc-circuit-4',
          question:
            'What is the difference between circuit breaker HALF_OPEN and CLOSED states?',
          options: [
            'No difference, they are the same',
            'HALF_OPEN allows limited test requests to check recovery; CLOSED allows all requests normally',
            'HALF_OPEN blocks all requests; CLOSED allows all requests',
            'HALF_OPEN is faster than CLOSED',
          ],
          correctAnswer: 1,
          explanation:
            'HALF_OPEN is a testing state after circuit has been OPEN. It allows limited test requests through to check if the downstream service has recovered. If tests succeed, circuit transitions to CLOSED (normal operation, all requests allowed). If tests fail, circuit goes back to OPEN. This prevents thundering herd problem (all requests hitting recovering service at once). CLOSED is normal operation where all requests pass through while monitoring for failures. Option 1 is wrong (very different states). Option 3 confuses HALF_OPEN with OPEN. Option 4 makes no sense.',
        },
        {
          id: 'mc-circuit-5',
          question: 'When should you NOT use a circuit breaker?',
          options: [
            'When calling external payment gateway (critical service)',
            'When calling internal recommendation service across network',
            'For in-process function calls within the same service',
            'When calling notification service',
          ],
          correctAnswer: 2,
          explanation:
            "Don't use circuit breakers for in-process function calls within the same service. Circuit breakers are for network calls between services to prevent cascading failures due to timeouts and resource exhaustion. Local function calls don't have these issues. Options 1, 2, and 4 are all valid use cases for circuit breakers (external services, internal services across network, and non-critical services all benefit from circuit breakers). Only skip circuit breakers for local calls.",
        },
      ],
    },
    {
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
      quiz: [
        {
          id: 'q1-mesh',
          question:
            'What is a service mesh and what problems does it solve? How is it different from an API gateway?',
          sampleAnswer:
            'A service mesh is an infrastructure layer that handles service-to-service communication (east-west traffic) via sidecar proxies. It solves cross-cutting concerns: (1) Observability - automatic tracing, metrics, logging, (2) Security - automatic mTLS encryption, authorization policies, (3) Resilience - circuit breakers, retries, timeouts, (4) Traffic management - canary deployments, A/B testing. All WITHOUT code changes. API Gateway handles north-south traffic (client to services) and provides single entry point, authentication, rate limiting. Service mesh handles east-west (service-to-service) and provides per-service proxies. They complement each other: API Gateway at edge, service mesh for internal communication.',
          keyPoints: [
            'Service mesh: infrastructure layer for service-to-service (east-west) communication',
            'Sidecar proxies provide observability, security, resilience, traffic management',
            'No code changes needed (vs libraries)',
            'API Gateway: edge (north-south), Service Mesh: internal (east-west)',
            'They complement each other',
          ],
        },
        {
          id: 'q2-mesh',
          question:
            'When would you choose NOT to use a service mesh? What are the trade-offs?',
          sampleAnswer:
            "Skip service mesh when: (1) < 5-10 microservices - overhead not justified, simpler solutions work, (2) Performance-critical applications - service mesh adds 2-5ms latency per hop, (3) Team lacks operational maturity - service mesh adds significant complexity, (4) Simple communication patterns - don't need advanced traffic management. Trade-offs: PROS: mTLS, circuit breakers, retries, tracing, rate limiting without code changes, consistent policies across polyglot services. CONS: adds latency (2-5ms), operational complexity (another thing to manage), resource overhead (proxy containers), learning curve. Start with libraries (Resilience4j) or API gateway, graduate to service mesh as complexity grows.",
          keyPoints: [
            'Skip for: < 10 services, performance-critical, simple patterns, immature ops team',
            'Pros: powerful features without code changes, consistent policies',
            'Cons: latency (+2-5ms), operational complexity, resource overhead',
            'Trade-off: features vs complexity',
            'Start simple, graduate to mesh as needed',
          ],
        },
        {
          id: 'q3-mesh',
          question:
            'Explain the sidecar proxy pattern. How does it enable mTLS without code changes?',
          sampleAnswer:
            "Sidecar proxy pattern: Each service instance has a companion proxy container (Envoy) deployed alongside it. Service communicates with localhost:port (its sidecar), sidecar handles all network communication. For mTLS: (1) Control plane (Istio) provisions certificates to each sidecar, (2) When Order Service calls Payment Service, it calls localhost:15001 (its sidecar), (3) Order's sidecar establishes mTLS connection with Payment's sidecar, (4) Payment's sidecar decrypts and forwards to localhost:8080 (Payment Service), (5) Certificates rotate automatically. Service code unchanged - just calls http://payment-service, proxy intercepts and encrypts. This is powerful because: no library dependencies, works with any language, centralized certificate management, automatic rotation.",
          keyPoints: [
            'Sidecar proxy: companion container per service instance',
            'Service talks to localhost (proxy), proxy handles network',
            'Control plane provisions certificates to proxies',
            'Proxies establish mTLS between themselves',
            'Service code unchanged (no library, works with any language)',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-mesh-1',
          question: 'What is the primary purpose of a service mesh?',
          options: [
            'To replace Kubernetes',
            'To handle service-to-service communication with observability, security, and traffic management via sidecar proxies',
            'To store data in a distributed database',
            'To replace the need for load balancers',
          ],
          correctAnswer: 1,
          explanation:
            "Service mesh handles service-to-service (east-west) communication via sidecar proxies. It provides observability (tracing, metrics), security (mTLS, authorization), resilience (circuit breakers, retries), and traffic management (canary, A/B testing) without code changes. Option 1 is wrong (service mesh runs ON Kubernetes). Option 3 is unrelated. Option 4 is wrong (service mesh includes load balancing but doesn't replace external LBs).",
        },
        {
          id: 'mc-mesh-2',
          question:
            'How does a service mesh provide mTLS encryption without changing application code?',
          options: [
            'It modifies the application binary at runtime',
            'Services must import a special library',
            'Sidecar proxies handle encryption/decryption, services talk to localhost',
            'It uses a special compiler',
          ],
          correctAnswer: 2,
          explanation:
            'Service mesh uses sidecar proxy pattern: each service has a proxy container. Service talks to its localhost proxy, which handles mTLS encryption with the destination proxy. Control plane provisions certificates to proxies automatically. Service code unchanged - just calls http://payment-service, proxy intercepts and encrypts. This works with any programming language. Option 1 is wrong (no binary modification). Option 2 is wrong (no libraries needed). Option 4 makes no sense.',
        },
        {
          id: 'mc-mesh-3',
          question:
            'What are the two main components of a service mesh architecture?',
          options: [
            'Frontend and Backend',
            'Master and Worker nodes',
            'Data Plane (proxies) and Control Plane (management)',
            'Load Balancer and Database',
          ],
          correctAnswer: 2,
          explanation:
            'Service mesh has two planes: (1) Data Plane - sidecar proxies (Envoy) that handle all network traffic, enforce policies, collect metrics, (2) Control Plane - management layer (Istio/Linkerd) that configures proxies, collects telemetry, provides APIs. Data plane does the work, control plane tells it what to do. Options 1, 2, and 4 are unrelated concepts.',
        },
        {
          id: 'mc-mesh-4',
          question: 'When should you consider using a service mesh?',
          options: [
            'Always use service mesh from day one',
            'Only for external APIs',
            'When you have 15+ microservices, polyglot environment, strict security needs, or complex traffic management',
            "Never, it's deprecated technology",
          ],
          correctAnswer: 2,
          explanation:
            'Use service mesh when: (1) 15+ microservices (complexity justifies overhead), (2) Polyglot environment (Java, Node, Go), (3) Strict security requirements, (4) Need advanced traffic management, (5) Running on Kubernetes. For < 10 services, simpler solutions often suffice. Option 1 is wrong (adds complexity for small deployments). Option 2 confuses service mesh with API gateway. Option 4 is wrong (service mesh is actively used by major companies).',
        },
        {
          id: 'mc-mesh-5',
          question:
            'What is the typical latency overhead of a service mesh per request?',
          options: [
            'No overhead',
            '2-5 milliseconds (proxies add small latency)',
            '500+ milliseconds',
            '10+ seconds',
          ],
          correctAnswer: 1,
          explanation:
            "Service mesh adds 2-5ms latency per hop because each request goes through 2 proxies (source sidecar and destination sidecar). Linkerd is faster (~1-2ms), Istio ~2-5ms. For most applications, this is negligible compared to benefits. For ultra-low-latency systems (HFT, real-time gaming), this might matter. Option 1 is wrong (there's always some overhead). Options 3 and 4 are way too high (that would be unacceptable).",
        },
      ],
    },
    {
      id: 'microservices-testing',
      title: 'Microservices Testing',
      content: `Testing microservices is fundamentally different from testing monoliths. With distributed services, you need multiple testing strategies at different levels.

## Testing Pyramid for Microservices

The testing pyramid still applies, but with microservices-specific considerations:

\`\`\`
         ╱╲
        ╱  ╲      End-to-End Tests (Few)
       ╱────╲     - Full system integration
      ╱      ╲    - Expensive, slow, brittle
     ╱────────╲
    ╱ Contract ╲   Contract Tests (Some)
   ╱   Tests    ╲  - Service boundaries
  ╱──────────────╲ - Provider/Consumer
 ╱  Integration   ╲ Integration Tests (More)
╱      Tests       ╲ - Service + dependencies
────────────────────
   Unit Tests        Unit Tests (Most)
   (Many)            - Fast, isolated, cheap
\`\`\`

**Key difference**: Add **Contract Tests** layer to verify service boundaries.

---

## 1. Unit Tests

**What**: Test individual functions/classes in isolation.

**Scope**: Single service, no external dependencies.

**Example**:
\`\`\`javascript
// order-service/calculateTotal.test.js
describe('calculateTotal', () => {
    it('should calculate order total correctly', () => {
        const items = [
            { price: 10, quantity: 2 },
            { price: 5, quantity: 3 }
        ];
        
        const total = calculateTotal(items);
        
        expect(total).toBe(35); // (10*2) + (5*3)
    });
    
    it('should apply discount correctly', () => {
        const items = [{ price: 100, quantity: 1 }];
        const discount = 0.1; // 10%
        
        const total = calculateTotal(items, discount);
        
        expect(total).toBe(90);
    });
});
\`\`\`

**Best Practices**:
- Mock external dependencies (database, other services)
- Fast execution (< 100ms per test)
- High coverage (80%+)
- Run on every code change

---

## 2. Integration Tests

**What**: Test service with its dependencies (database, message queue, cache).

**Scope**: Single service + immediate dependencies.

### Database Integration Tests

**Use test database or Docker containers**:

\`\`\`javascript
// order-service/orderRepository.integration.test.js
describe('OrderRepository Integration', () => {
    let db;
    
    beforeAll(async () => {
        // Start test database (Docker)
        db = await startTestDatabase();
        await db.migrate();
    });
    
    afterAll(async () => {
        await db.close();
    });
    
    afterEach(async () => {
        await db.clearAll(); // Clean between tests
    });
    
    it('should create and retrieve order', async () => {
        const order = {
            userId: 'user-123',
            items: [{ productId: 'prod-1', quantity: 2 }],
            total: 50
        };
        
        const createdOrder = await orderRepository.create(order);
        const retrievedOrder = await orderRepository.findById(createdOrder.id);
        
        expect(retrievedOrder).toMatchObject(order);
    });
    
    it('should update order status', async () => {
        const order = await orderRepository.create({...});
        
        await orderRepository.updateStatus(order.id, 'SHIPPED');
        
        const updated = await orderRepository.findById(order.id);
        expect(updated.status).toBe('SHIPPED');
    });
});
\`\`\`

### Message Queue Integration Tests

\`\`\`javascript
describe('Order Event Publisher Integration', () => {
    let rabbitMQ;
    
    beforeAll(async () => {
        rabbitMQ = await startTestRabbitMQ();
    });
    
    it('should publish OrderCreated event', async () => {
        const order = { id: 'order-123', userId: 'user-1' };
        
        await eventPublisher.publishOrderCreated(order);
        
        // Verify message published
        const messages = await rabbitMQ.getMessages('order-events');
        expect(messages).toHaveLength(1);
        expect(messages[0]).toMatchObject({
            eventType: 'OrderCreated',
            data: order
        });
    });
});
\`\`\`

---

## 3. Contract Tests

**Problem**: Service A depends on Service B. If Service B changes its API, Service A breaks.

**Solution**: Contract tests verify that provider (Service B) fulfills the contract that consumer (Service A) expects.

### Consumer-Driven Contract Testing

**Tool**: Pact

**Flow**:
1. **Consumer** (Order Service) defines expected contract
2. **Consumer** generates Pact file
3. **Provider** (Payment Service) verifies it can fulfill contract

**Example**:

**Consumer side** (Order Service):
\`\`\`javascript
// order-service/payment.contract.test.js
const { Pact } = require('@pact-foundation/pact');
const { chargePayment } = require('./paymentClient');

describe('Payment Service Contract', () => {
    const provider = new Pact({
        consumer: 'OrderService',
        provider: 'PaymentService',
        port: 8989
    });
    
    beforeAll(() => provider.setup());
    afterAll(() => provider.finalize());
    
    describe('charge payment endpoint', () => {
        beforeAll(() => {
            // Define expected interaction
            return provider.addInteraction({
                state: 'user has sufficient balance',
                uponReceiving: 'a charge payment request',
                withRequest: {
                    method: 'POST',
                    path: '/charge',
                    body: {
                        userId: 'user-123',
                        amount: 50,
                        orderId: 'order-456'
                    }
                },
                willRespondWith: {
                    status: 200,
                    body: {
                        transactionId: Matchers.like('txn-789'),
                        status: 'SUCCESS'
                    }
                }
            });
        });
        
        it('should charge payment successfully', async () => {
            const result = await chargePayment({
                userId: 'user-123',
                amount: 50,
                orderId: 'order-456'
            });
            
            expect(result.status).toBe('SUCCESS');
            expect(result.transactionId).toBeTruthy();
        });
    });
});
\`\`\`

This generates a **Pact file** (JSON contract).

**Provider side** (Payment Service):
\`\`\`javascript
// payment-service/pact-verification.test.js
const { Verifier } = require('@pact-foundation/pact');

describe('Pact Verification', () => {
    it('should validate all consumer contracts', async () => {
        await new Verifier({
            provider: 'PaymentService',
            providerBaseUrl: 'http://localhost:8080',
            pactUrls: [
                './pacts/OrderService-PaymentService.json'
            ],
            publishVerificationResult: true
        }).verifyProvider();
    });
});
\`\`\`

**Benefits**:
✅ Catch breaking changes before deployment
✅ Consumer defines contract (knows what it needs)
✅ Automated testing of service boundaries
✅ No need to run both services together

---

## 4. Component Tests

**What**: Test entire service in isolation with mocked dependencies.

**Approach**: Run service, mock external services, test via API.

\`\`\`javascript
// order-service/component.test.js
describe('Order Service Component Tests', () => {
    let app;
    let mockPaymentService;
    let mockInventoryService;
    
    beforeAll(async () => {
        // Start mock services
        mockPaymentService = startMockServer(8081);
        mockInventoryService = startMockServer(8082);
        
        // Start order service (points to mocks)
        app = await startOrderService({
            paymentServiceUrl: 'http://localhost:8081',
            inventoryServiceUrl: 'http://localhost:8082'
        });
    });
    
    afterAll(async () => {
        await app.stop();
        await mockPaymentService.stop();
        await mockInventoryService.stop();
    });
    
    it('should create order end-to-end', async () => {
        // Setup mocks
        mockInventoryService.on('POST', '/reserve', () => ({
            status: 200,
            body: { reserved: true }
        }));
        
        mockPaymentService.on('POST', '/charge', () => ({
            status: 200,
            body: { transactionId: 'txn-123', status: 'SUCCESS' }
        }));
        
        // Test order creation
        const response = await request(app)
            .post('/orders')
            .send({
                userId: 'user-123',
                items: [{ productId: 'prod-1', quantity: 2 }]
            });
        
        expect(response.status).toBe(201);
        expect(response.body.status).toBe('PENDING');
        
        // Verify calls to dependencies
        expect(mockInventoryService.requests).toHaveLength(1);
        expect(mockPaymentService.requests).toHaveLength(1);
    });
    
    it('should handle payment failure', async () => {
        mockInventoryService.on('POST', '/reserve', () => ({
            status: 200,
            body: { reserved: true }
        }));
        
        // Payment fails
        mockPaymentService.on('POST', '/charge', () => ({
            status: 400,
            body: { error: 'Insufficient funds' }
        }));
        
        const response = await request(app)
            .post('/orders')
            .send({...});
        
        expect(response.status).toBe(400);
        expect(response.body.error).toContain('Payment failed');
        
        // Should have called inventory release (compensating transaction)
        expect(mockInventoryService.requests).toContainEqual(
            expect.objectContaining({ path: '/release' })
        );
    });
});
\`\`\`

---

## 5. End-to-End Tests

**What**: Test entire system with all services running.

**Challenges**:
❌ Slow (seconds to minutes per test)
❌ Flaky (network issues, timing problems)
❌ Expensive to maintain
❌ Hard to debug failures

**Keep them minimal!**

### Critical Path Testing

**Only test happy paths and critical failures**:

\`\`\`javascript
// e2e/order-flow.test.js
describe('Order Flow E2E', () => {
    beforeAll(async () => {
        // Start all services (Docker Compose)
        await exec('docker-compose up -d');
        await waitForServices();
    });
    
    afterAll(async () => {
        await exec('docker-compose down');
    });
    
    it('should complete full order flow', async () => {
        // 1. Create user
        const user = await createUser({
            email: 'test@example.com',
            password: 'password123'
        });
        
        // 2. Add to cart
        await addToCart(user.id, { productId: 'prod-1', quantity: 2 });
        
        // 3. Checkout
        const order = await checkout(user.id, {
            paymentMethod: 'credit_card'
        });
        
        expect(order.status).toBe('CONFIRMED');
        
        // 4. Wait for async processing
        await sleep(2000);
        
        // 5. Verify order status
        const orderStatus = await getOrderStatus(order.id);
        expect(orderStatus.status).toBe('SHIPPED');
        
        // 6. Verify email sent
        const emails = await getTestEmails(user.email);
        expect(emails).toContainEqual(
            expect.objectContaining({
                subject: 'Order Confirmed',
                orderId: order.id
            })
        );
    });
});
\`\`\`

**Best Practices**:
- Run in dedicated environment (not production!)
- Use test data that's isolated
- Clean up after tests
- Run less frequently (nightly, not on every commit)

---

## Test Data Management

### Problem: Test Data Conflicts

Multiple tests running in parallel can interfere:

\`\`\`
Test 1: Create user with email "test@example.com"
Test 2: Create user with email "test@example.com" ❌ CONFLICT
\`\`\`

### Solutions:

**1. Generate Unique Data**:
\`\`\`javascript
const userId = \`user-\${uuidv4()}\`;
const email = \`test-\${Date.now()}@example.com\`;
\`\`\`

**2. Use Test Database Per Test**:
\`\`\`javascript
beforeEach(async () => {
    testDb = await createTestDatabase();
});

afterEach(async () => {
    await testDb.drop();
});
\`\`\`

**3. Database Transactions (Rollback)**:
\`\`\`javascript
beforeEach(async () => {
    await db.beginTransaction();
});

afterEach(async () => {
    await db.rollback(); // Undo all changes
});
\`\`\`

---

## Testing Asynchronous Communication

### Testing Event-Driven Services

**Challenge**: Service A publishes event, Service B consumes it asynchronously.

**Solution**: Poll and wait for expected state:

\`\`\`javascript
describe('Order Created Event Processing', () => {
    it('should send email when order created', async () => {
        // Publish event
        await eventBus.publish('OrderCreated', {
            orderId: 'order-123',
            userId: 'user-456'
        });
        
        // Poll for email sent (eventual consistency)
        await waitFor(async () => {
            const emails = await getEmailsSent('user-456');
            return emails.some(e => e.orderId === 'order-123');
        }, { timeout: 5000 });
        
        // Verify email content
        const emails = await getEmailsSent('user-456');
        const orderEmail = emails.find(e => e.orderId === 'order-123');
        expect(orderEmail.subject).toContain('Order Confirmed');
    });
});

// Helper
async function waitFor(condition, options = {}) {
    const timeout = options.timeout || 5000;
    const interval = options.interval || 100;
    const start = Date.now();
    
    while (Date.now() - start < timeout) {
        if (await condition()) {
            return;
        }
        await sleep(interval);
    }
    
    throw new Error('Condition not met within timeout');
}
\`\`\`

---

## Testing Resilience Patterns

### Circuit Breaker Tests

\`\`\`javascript
describe('Circuit Breaker', () => {
    it('should open after threshold failures', async () => {
        // Make service fail
        mockPaymentService.alwaysFail();
        
        // Trigger failures
        for (let i = 0; i < 5; i++) {
            await expect(chargePayment({})).rejects.toThrow();
        }
        
        // Circuit should be open now
        await expect(chargePayment({})).rejects.toThrow('Circuit breaker open');
        
        // Verify no requests sent (circuit is open)
        expect(mockPaymentService.requestCount).toBe(5); // Not 6
    });
    
    it('should half-open after timeout', async () => {
        // Open circuit
        mockPaymentService.alwaysFail();
        for (let i = 0; i < 5; i++) {
            await chargePayment({}).catch(() => {});
        }
        
        // Wait for reset timeout
        await sleep(30000);
        
        // Fix service
        mockPaymentService.alwaysSucceed();
        
        // Should allow test request (half-open)
        const result = await chargePayment({});
        expect(result.status).toBe('SUCCESS');
        
        // Circuit should be closed now
        const result2 = await chargePayment({});
        expect(result2.status).toBe('SUCCESS');
    });
});
\`\`\`

### Retry Tests

\`\`\`javascript
describe('Retry Logic', () => {
    it('should retry on transient failure', async () => {
        // Fail twice, then succeed
        mockPaymentService
            .onCall(1).fail()
            .onCall(2).fail()
            .onCall(3).succeed();
        
        const result = await chargePaymentWithRetry({});
        
        expect(result.status).toBe('SUCCESS');
        expect(mockPaymentService.requestCount).toBe(3);
    });
    
    it('should not retry on permanent failure', async () => {
        // 400 Bad Request (permanent failure)
        mockPaymentService.respondWith(400, { error: 'Invalid card' });
        
        await expect(chargePaymentWithRetry({})).rejects.toThrow('Invalid card');
        
        // Should not retry 400 errors
        expect(mockPaymentService.requestCount).toBe(1);
    });
});
\`\`\`

---

## Performance Testing

### Load Testing

**Tool**: k6, Artillery, Gatling

\`\`\`javascript
// k6 load test
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
    stages: [
        { duration: '1m', target: 50 },   // Ramp up to 50 users
        { duration: '3m', target: 50 },   // Stay at 50 users
        { duration: '1m', target: 100 },  // Ramp up to 100
        { duration: '3m', target: 100 },  // Stay at 100
        { duration: '1m', target: 0 },    // Ramp down
    ],
    thresholds: {
        http_req_duration: ['p(95)<500'], // 95% of requests < 500ms
        http_req_failed: ['rate<0.01'],   // < 1% failure rate
    },
};

export default function () {
    const res = http.post('http://api-gateway/orders', JSON.stringify({
        userId: 'user-123',
        items: [{ productId: 'prod-1', quantity: 2 }]
    }), {
        headers: { 'Content-Type': 'application/json' },
    });
    
    check(res, {
        'status is 201': (r) => r.status === 201,
        'response time < 500ms': (r) => r.timings.duration < 500,
    });
    
    sleep(1);
}
\`\`\`

---

## Chaos Testing

**What**: Intentionally break things to verify resilience.

**Tool**: Chaos Monkey, Gremlin

**Examples**:
- Randomly kill service instances
- Inject network latency
- Fill up disk space
- Corrupt databases

\`\`\`javascript
describe('Chaos Tests', () => {
    it('should handle payment service failure', async () => {
        // Kill payment service
        await killService('payment-service');
        
        // Order creation should still work (degrade gracefully)
        const response = await createOrder({...});
        expect(response.status).toBe(202); // Accepted
        expect(response.body.status).toBe('PENDING_PAYMENT');
    });
    
    it('should recover when service comes back', async () => {
        await killService('payment-service');
        
        // Wait for Kubernetes to restart
        await sleep(10000);
        
        // Should work again
        const response = await createOrder({...});
        expect(response.status).toBe(201);
    });
});
\`\`\`

---

## Testing in Production

### Synthetic Monitoring

Run automated tests against production.

\`\`\`javascript
// Runs every 5 minutes in production
describe('Production Smoke Tests', () => {
    it('should create test order', async () => {
        const order = await createOrder({
            userId: 'SYNTHETIC_USER',
            items: [{ productId: 'TEST_PRODUCT', quantity: 1 }]
        });
        
        expect(order.status).toBe('CONFIRMED');
        
        // Clean up
        await deleteOrder(order.id);
    });
});
\`\`\`

---

## Best Practices

1. **Test Pyramid**: Most unit tests, fewer integration, minimal E2E
2. **Contract Tests**: Verify service boundaries
3. **Independent Tests**: No shared state between tests
4. **Fast Feedback**: Unit tests run on every commit
5. **Isolate Services**: Use mocks, stubs, or Docker
6. **Test Data**: Generate unique data, clean up after tests
7. **Chaos Engineering**: Test failure scenarios
8. **Monitor in Production**: Synthetic tests, real user monitoring

---

## Interview Tips

**Red Flags**:
❌ Only unit tests (ignoring integration)
❌ Too many E2E tests (slow, brittle)
❌ Not testing service boundaries

**Good Responses**:
✅ Explain testing pyramid for microservices
✅ Mention contract testing (Pact)
✅ Discuss trade-offs (speed vs confidence)
✅ Talk about chaos engineering

**Sample Answer**:
*"For microservices testing, I follow the testing pyramid: majority unit tests for fast feedback, integration tests for service + dependencies, contract tests to verify service boundaries, and minimal E2E tests for critical paths. I'd use Pact for consumer-driven contract testing to catch breaking changes early. For async communication, I'd use polling with timeouts. Tests should be independent with unique test data. I'd also implement chaos testing to verify resilience and synthetic monitoring in production."*

---

## Key Takeaways

1. **Testing pyramid** applies but with contract tests added
2. **Contract tests** verify service boundaries without running both services
3. **Minimal E2E tests** (slow, expensive, flaky)
4. **Integration tests** for service + immediate dependencies
5. **Async testing** requires polling and timeouts
6. **Chaos engineering** verifies resilience
7. **Test data** must be unique and isolated
8. **Synthetic monitoring** tests production continuously`,
      quiz: [
        {
          id: 'q1-testing',
          question:
            'Why are contract tests important in microservices? How do they differ from integration tests?',
          sampleAnswer:
            'Contract tests verify service boundaries without running both services together. Consumer (Order Service) defines expected contract (what it needs from Payment Service), provider (Payment Service) verifies it can fulfill that contract. Benefits: (1) Catch breaking changes before deployment, (2) No need to run both services (faster), (3) Consumer-driven (knows what it needs), (4) Automated boundary testing. Integration tests test service WITH its dependencies running (database, message queue). Contract tests test service boundaries (API contracts) WITHOUT running the other service. Example: Pact lets Order Service define "when I call POST /charge with {amount: 50}, I expect {status: SUCCESS}". Payment Service verifies it fulfills this. If Payment changes response format, contract test fails.',
          keyPoints: [
            'Contract tests verify service API boundaries',
            'Consumer defines contract, provider verifies',
            'Catches breaking changes before deployment',
            "Doesn't require running both services (vs integration)",
            'Tool: Pact for consumer-driven contract testing',
          ],
        },
        {
          id: 'q2-testing',
          question:
            'How do you test asynchronous event-driven communication between microservices?',
          sampleAnswer:
            'Testing async communication requires polling/waiting for eventual consistency. Approach: (1) Publish event to message bus, (2) Poll consumer service for expected state change with timeout, (3) Verify outcome. Example: Order Service publishes OrderCreated event → wait for Email Service to send email → verify email sent. Use helper function waitFor() that polls condition every 100ms up to 5s timeout. Challenges: (1) Timing - need generous timeouts, (2) Flakiness - network delays, (3) Test isolation - clean up events between tests. Alternative: Mock message bus for faster testing, but sacrifice realism. For integration tests, use real message bus (RabbitMQ/Kafka in Docker) with test queues.',
          keyPoints: [
            'Async testing requires polling for eventual consistency',
            'Publish event → wait for outcome → verify',
            'Use waitFor() helper with timeout (e.g., 5 seconds)',
            'Trade-off: real message bus (slow, realistic) vs mocks (fast, less realistic)',
            'Clean up events/queues between tests for isolation',
          ],
        },
        {
          id: 'q3-testing',
          question:
            'What is the testing pyramid for microservices? Why keep E2E tests minimal?',
          sampleAnswer:
            'Testing pyramid (bottom to top): (1) Unit tests (most) - fast, isolated, cheap, high coverage, (2) Integration tests (more) - service + dependencies (DB, cache), slower but still manageable, (3) Contract tests (some) - service boundaries, faster than E2E, (4) E2E tests (few) - full system, only critical paths. Keep E2E minimal because: (1) Slow - seconds to minutes per test (vs milliseconds for unit), (2) Flaky - network issues, timing problems, (3) Expensive - require all services running, (4) Hard to debug - failure could be anywhere in system. Instead, push testing down: more unit tests, integration tests for each service, contract tests for boundaries. E2E only for critical happy paths (checkout flow) and major failure scenarios.',
          keyPoints: [
            'Pyramid: many unit tests, fewer integration, some contract, minimal E2E',
            'E2E tests are slow, flaky, expensive, hard to debug',
            'Push testing down the pyramid for faster feedback',
            'E2E only for critical paths and major failures',
            'Contract tests reduce need for E2E (verify boundaries)',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-testing-1',
          question:
            'What is the main purpose of contract testing in microservices?',
          options: [
            'To test database performance',
            'To verify service API boundaries without running both services together',
            'To replace all integration tests',
            'To test UI components',
          ],
          correctAnswer: 1,
          explanation:
            "Contract testing (e.g., Pact) verifies that a provider service fulfills the API contract that a consumer service expects, without actually running both services together. Consumer defines expectations, generates contract file, provider verifies it meets the contract. This catches breaking changes before deployment and is faster than integration tests requiring both services. Option 1 is unrelated. Option 3 is wrong (contract tests complement, don't replace integration tests). Option 4 is wrong (that's UI testing).",
        },
        {
          id: 'mc-testing-2',
          question:
            'Why should you minimize end-to-end (E2E) tests in microservices?',
          options: [
            'E2E tests are not important',
            'E2E tests are slow, flaky, expensive to maintain, and hard to debug',
            "E2E tests don't test the full system",
            'E2E tests are only for monoliths',
          ],
          correctAnswer: 1,
          explanation:
            "E2E tests require all services running, making them slow (seconds to minutes), flaky (network issues, timing problems), expensive (infrastructure costs, maintenance), and hard to debug (failure could be anywhere). Instead, follow the testing pyramid: push most testing down to unit/integration/contract tests, use E2E only for critical happy paths. Option 1 is wrong (E2E tests ARE important, just use sparingly). Option 3 is wrong (they do test full system, that's why they're slow). Option 4 is wrong (E2E tests apply to both).",
        },
        {
          id: 'mc-testing-3',
          question: 'How do you test asynchronous event-driven communication?',
          options: [
            "You can't test async communication",
            'Only with manual testing',
            'Publish event, poll for expected outcome with timeout',
            'Synchronous tests work fine',
          ],
          correctAnswer: 2,
          explanation:
            "Testing async communication requires: (1) Publish event to message bus, (2) Poll consumer service for expected state change (e.g., every 100ms), (3) Verify outcome within timeout (e.g., 5 seconds). This accounts for eventual consistency. Example: publish OrderCreated → wait for email sent → verify email. Use helper functions like waitFor() with polling and timeout. Option 1 is defeatist. Option 2 ignores automation. Option 4 doesn't work (need to wait for eventual consistency).",
        },
        {
          id: 'mc-testing-4',
          question: 'What is a component test in microservices?',
          options: [
            'Testing UI components',
            'Testing a single function',
            'Testing an entire service in isolation with mocked external dependencies',
            'Testing hardware components',
          ],
          correctAnswer: 2,
          explanation:
            'Component testing tests an entire microservice (all endpoints, business logic) in isolation with external services mocked. Start the service, mock dependencies (other services, external APIs), send requests via API, verify responses. This is faster than E2E (no need to start all services) but more comprehensive than unit tests (tests full service). Example: start Order Service, mock Payment/Inventory services, test order creation flow. Option 1 is UI testing. Option 2 is unit testing. Option 4 is hardware testing.',
        },
        {
          id: 'mc-testing-5',
          question: 'What is chaos testing (chaos engineering)?',
          options: [
            'Testing without any plan',
            'Intentionally injecting failures (kill services, add latency) to verify system resilience',
            'Testing during chaotic deployments',
            'Random testing without assertions',
          ],
          correctAnswer: 1,
          explanation:
            'Chaos testing (chaos engineering) intentionally injects failures into the system to verify resilience: kill service instances, inject network latency, fill disk space, corrupt data. Verify system degrades gracefully, recovers automatically, and alerts work. Tools: Chaos Monkey (Netflix), Gremlin. Example: kill Payment Service → Order Service should still work (create orders as PENDING) → Payment Service restarts → orders process. Option 1 is wrong (chaos testing is very planned). Options 3 and 4 misunderstand the concept.',
        },
      ],
    },
    {
      id: 'microservices-deployment',
      title: 'Microservices Deployment',
      content: `Deploying microservices requires orchestration, containerization, and CI/CD automation. The deployment strategy significantly impacts reliability, scalability, and development velocity.

## Containerization

**Why containers?** Consistency across environments, isolation, portability.

### Docker

**Package service as container image**:

\`\`\`dockerfile
# Dockerfile for Order Service
FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy source
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=3s \\
  CMD node healthcheck.js || exit 1

EXPOSE 8080

CMD ["node", "server.js"]
\`\`\`

**Build and run**:
\`\`\`bash
docker build -t order-service:v1.2.3 .
docker run -p 8080:8080 order-service:v1.2.3
\`\`\`

**Best practices**:
- Multi-stage builds (reduce image size)
- Layer caching (faster builds)
- Non-root user (security)
- Small base images (alpine)
- Health checks

---

## Kubernetes

**De facto** orchestration platform for microservices.

### Core Concepts

**1. Pod**: Smallest deployable unit (1+ containers)
**2. Deployment**: Manages replica Pods
**3. Service**: Stable network endpoint
**4. Ingress**: External access
**5. ConfigMap/Secret**: Configuration

### Example Deployment

\`\`\`yaml
# order-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
  labels:
    app: order-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: order-service
  template:
    metadata:
      labels:
        app: order-service
        version: v1.2.3
    spec:
      containers:
      - name: order-service
        image: myregistry/order-service:v1.2.3
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: order-service-secrets
              key: database-url
        - name: PAYMENT_SERVICE_URL
          value: "http://payment-service"
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: order-service
spec:
  selector:
    app: order-service
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
\`\`\`

**Deploy**:
\`\`\`bash
kubectl apply -f order-service-deployment.yaml
kubectl get pods
kubectl logs order-service-abc123
\`\`\`

---

## Deployment Strategies

### 1. Rolling Update (Default)

**Gradually** replace old pods with new ones.

\`\`\`yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # Max extra pods during update
      maxUnavailable: 0  # Min available pods during update
\`\`\`

**Flow**:
\`\`\`
v1: [Pod1] [Pod2] [Pod3]
    [Pod1] [Pod2] [Pod3] [Pod4-v2]  # Create new pod
    [Pod1] [Pod2] [Pod4-v2]          # Delete old pod
    [Pod1] [Pod2] [Pod4-v2] [Pod5-v2]
    [Pod2] [Pod4-v2] [Pod5-v2]
    [Pod2] [Pod4-v2] [Pod5-v2] [Pod6-v2]
    [Pod4-v2] [Pod5-v2] [Pod6-v2]
\`\`\`

**Pros**: Zero downtime, automatic rollback
**Cons**: Mixed versions during deployment

### 2. Blue-Green Deployment

**Two identical environments**, switch traffic instantly.

\`\`\`yaml
# Blue (current)
apiVersion: v1
kind: Service
metadata:
  name: order-service
spec:
  selector:
    app: order-service
    version: blue
  ports:
  - port: 80
    targetPort: 8080
---
# Deploy Green
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service-green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: order-service
      version: green
  template:
    metadata:
      labels:
        app: order-service
        version: green
    spec:
      containers:
      - name: order-service
        image: order-service:v1.2.3
\`\`\`

**Switch traffic** (update Service selector):
\`\`\`yaml
spec:
  selector:
    app: order-service
    version: green  # Changed from blue
\`\`\`

**Pros**: Instant rollback, testing before switch
**Cons**: 2x resources, database migration challenges

### 3. Canary Deployment

**Gradually** shift traffic to new version.

Using Istio:
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
        subset: v1
      weight: 90  # 90% to old version
    - destination:
        host: order-service
        subset: v2
      weight: 10  # 10% to new version (canary)
\`\`\`

**Increase gradually**: 10% → 25% → 50% → 100%

**Pros**: Low risk, real user testing
**Cons**: Requires service mesh, slow rollout

---

## CI/CD Pipeline

**Automated** build, test, deploy.

### Example Pipeline (GitLab CI)

\`\`\`yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

build:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

unit-test:
  stage: test
  script:
    - npm install
    - npm test

integration-test:
  stage: test
  script:
    - docker-compose up -d
    - npm run test:integration
    - docker-compose down

contract-test:
  stage: test
  script:
    - npm run test:pact

deploy-staging:
  stage: deploy
  script:
    - kubectl set image deployment/order-service \\
        order-service=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA \\
        -n staging
  environment:
    name: staging
  only:
    - develop

deploy-production:
  stage: deploy
  script:
    - kubectl set image deployment/order-service \\
        order-service=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA \\
        -n production
  environment:
    name: production
  when: manual  # Require approval
  only:
    - main
\`\`\`

---

## Configuration Management

**Never** hardcode config. Use environment variables or config files.

### ConfigMaps

\`\`\`yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: order-service-config
data:
  PAYMENT_SERVICE_URL: "http://payment-service"
  LOG_LEVEL: "info"
  MAX_RETRIES: "3"
\`\`\`

### Secrets

\`\`\`yaml
apiVersion: v1
kind: Secret
metadata:
  name: order-service-secrets
type: Opaque
data:
  database-url: cG9zdGdyZXM6Ly8uLi4=  # base64 encoded
  api-key: YWJjZGVm  # base64 encoded
\`\`\`

**Use in Pod**:
\`\`\`yaml
env:
- name: DATABASE_URL
  valueFrom:
    secretKeyRef:
      name: order-service-secrets
      key: database-url
- name: PAYMENT_SERVICE_URL
  valueFrom:
    configMapKeyRef:
      name: order-service-config
      key: PAYMENT_SERVICE_URL
\`\`\`

---

## Service Mesh Deployment

**Istio automatically** injects sidecar proxies.

**Enable injection**:
\`\`\`bash
kubectl label namespace default istio-injection=enabled
\`\`\`

**Deploy service** (Istio adds proxy automatically):
\`\`\`bash
kubectl apply -f order-service-deployment.yaml

# Pod now has 2 containers: order-service + istio-proxy
kubectl get pods
NAME                             READY   STATUS
order-service-abc123             2/2     Running
\`\`\`

---

## Database Migrations

**Challenge**: Deploy new service version with database changes.

### Backward-Compatible Migrations

**Step 1**: Add new column (optional)
\`\`\`sql
ALTER TABLE orders ADD COLUMN shipping_address TEXT;
\`\`\`

**Step 2**: Deploy new service version (uses new column)

**Step 3**: Backfill data
\`\`\`sql
UPDATE orders SET shipping_address = address WHERE shipping_address IS NULL;
\`\`\`

**Step 4**: Make column required (next deployment)
\`\`\`sql
ALTER TABLE orders ALTER COLUMN shipping_address SET NOT NULL;
\`\`\`

**Step 5**: Remove old column (next deployment)
\`\`\`sql
ALTER TABLE orders DROP COLUMN address;
\`\`\`

**Key**: Each step is backward-compatible.

---

## Monitoring and Observability

**Essential for production** microservices.

### Health Checks

**Liveness probe**: Is service alive? (restart if fails)
**Readiness probe**: Is service ready for traffic? (remove from load balancer if fails)

\`\`\`javascript
// /health endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

// /ready endpoint (check dependencies)
app.get('/ready', async (req, res) => {
    try {
        await database.ping();
        await paymentService.healthCheck();
        res.json({ status: 'ready' });
    } catch (error) {
        res.status(503).json({ status: 'not ready', error: error.message });
    }
});
\`\`\`

### Metrics

**Export Prometheus metrics**:
\`\`\`javascript
const promClient = require('prom-client');
const register = new promClient.Registry();

// Metrics
const httpRequestDuration = new promClient.Histogram({
    name: 'http_request_duration_seconds',
    help: 'Duration of HTTP requests in seconds',
    labelNames: ['method', 'route', 'status_code'],
    registers: [register]
});

// Middleware
app.use((req, res, next) => {
    const end = httpRequestDuration.startTimer();
    res.on('finish', () => {
        end({ method: req.method, route: req.route?.path, status_code: res.statusCode });
    });
    next();
});

// Metrics endpoint
app.get('/metrics', async (req, res) => {
    res.set('Content-Type', register.contentType);
    res.end(await register.metrics());
});
\`\`\`

---

## Scaling

### Horizontal Pod Autoscaler

**Automatically** scale based on CPU/memory:

\`\`\`yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: order-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: order-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
\`\`\`

**Kubernetes automatically** adds/removes pods to maintain target CPU/memory.

---

## Best Practices

1. **Containerize** all services
2. **Use Kubernetes** for orchestration
3. **Automate** CI/CD pipeline
4. **Canary deployments** for low-risk rollouts
5. **Health checks** (liveness + readiness)
6. **Resource limits** (CPU, memory)
7. **ConfigMaps/Secrets** for configuration
8. **Backward-compatible** database migrations
9. **Monitor** everything (metrics, logs, traces)
10. **Autoscale** based on load

---

## Key Takeaways

1. **Containers** provide consistency and isolation
2. **Kubernetes** is standard for microservices orchestration
3. **Deployment strategies**: Rolling, Blue-Green, Canary
4. **CI/CD** automates build, test, deploy
5. **Health checks** enable automatic recovery
6. **Database migrations** must be backward-compatible
7. **Configuration** via ConfigMaps/Secrets
8. **Autoscaling** handles variable load`,
      quiz: [
        {
          id: 'q1-deploy',
          question:
            'Compare Rolling Update, Blue-Green, and Canary deployment strategies. When would you use each?',
          sampleAnswer:
            'Rolling Update: Gradually replace pods one by one (v1→v1→v1 becomes v2→v1→v1 becomes v2→v2→v1...). Pros: zero downtime, automatic rollback. Use for: normal deployments, low-risk changes. Blue-Green: Two environments (Blue=current, Green=new), switch instantly. Pros: instant rollback, test before switch. Use for: high-stakes deployments, need quick rollback. Canary: Gradually shift traffic (10%→25%→50%→100%). Pros: real user testing, lowest risk. Use for: risky changes, want real user feedback. Example: Rolling for bug fixes, Blue-Green for major releases, Canary for algorithm changes where you monitor metrics before full rollout.',
          keyPoints: [
            'Rolling: gradual pod replacement, zero downtime, automatic rollback',
            'Blue-Green: two environments, instant switch, 2x resources',
            'Canary: gradual traffic shift, lowest risk, requires service mesh',
            'Choose based on risk level and resource availability',
            'All strategies enable zero-downtime deployments',
          ],
        },
        {
          id: 'q2-deploy',
          question:
            'How do you handle database migrations in microservices without causing downtime?',
          sampleAnswer:
            'Use backward-compatible migrations in multiple steps: (1) Add new column as NULLABLE, deploy service that uses it optionally, (2) Backfill data, (3) Make column NOT NULL in next deployment, (4) Remove old column in another deployment. Example: Renaming "address" to "shipping_address": Add shipping_address (nullable) → deploy service writing to both → backfill data → deploy service using only shipping_address → make NOT NULL → remove address column. Each step is backward-compatible so old and new service versions coexist during rolling deployment. NEVER make breaking changes in single step (adding NOT NULL column immediately breaks old service version).',
          keyPoints: [
            'Backward-compatible migrations in multiple steps',
            'Add column as nullable first',
            'Deploy service, then backfill, then enforce constraints',
            'Each step must work with previous service version',
            'Multiple deployments needed for breaking changes',
          ],
        },
        {
          id: 'q3-deploy',
          question:
            'Explain liveness vs readiness probes in Kubernetes. Why do you need both?',
          sampleAnswer:
            "Liveness probe: Is container alive? If fails, Kubernetes restarts the container. Checks if service is deadlocked or crashed. Readiness probe: Is container ready for traffic? If fails, Kubernetes removes pod from service load balancer (doesn't restart). Checks if dependencies are available. Need both because: Liveness prevents permanent failures (restart if stuck), Readiness prevents routing traffic to unhealthy instances (remove from LB temporarily). Example: Service starting up - readiness fails (not ready yet), liveness passes (not stuck). Service running but database down - readiness fails (can't serve requests), liveness passes (service itself is ok).",
          keyPoints: [
            'Liveness: Is service alive? (restart if fails)',
            'Readiness: Is service ready for traffic? (remove from LB if fails)',
            'Liveness prevents permanent failures (deadlocks, crashes)',
            'Readiness prevents routing to unhealthy instances',
            'Both needed for automatic recovery without manual intervention',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-deploy-1',
          question:
            'What is the main advantage of containerization for microservices?',
          options: [
            'Containers are faster than virtual machines',
            'Containers provide consistency across environments and isolation between services',
            'Containers eliminate the need for testing',
            'Containers are only useful in production',
          ],
          correctAnswer: 1,
          explanation:
            'Containers provide consistency ("works on my machine" → works everywhere) and isolation (dependencies don\'t conflict). Package service with all dependencies, run same image in dev/staging/prod. Docker image includes OS, runtime, libraries, code. Option 1 is partially true but not the main advantage. Option 3 is wrong (testing still needed). Option 4 is wrong (containers used in all environments).',
        },
        {
          id: 'mc-deploy-2',
          question: 'What is a canary deployment?',
          options: [
            'Deploying to a bird-themed environment',
            'Gradually shifting traffic to new version (10% → 25% → 50% → 100%)',
            'Deploying all at once',
            'Only deploying to production',
          ],
          correctAnswer: 1,
          explanation:
            'Canary deployment gradually shifts traffic to new version while monitoring metrics. Start with 10% of traffic to new version, monitor for errors/performance, gradually increase if all is well. Named after "canary in coal mine" (early warning). This minimizes risk by testing with real users before full rollout. Requires service mesh (Istio) or load balancer that supports weighted routing. Option 1 is a joke. Option 3 is big bang. Option 4 doesn\'t describe a strategy.',
        },
        {
          id: 'mc-deploy-3',
          question:
            'What is the difference between a liveness probe and a readiness probe?',
          options: [
            'They are the same thing',
            'Liveness restarts container if fails; Readiness removes from load balancer if fails',
            'Liveness is for production; Readiness is for development',
            'Readiness restarts container; Liveness removes from load balancer',
          ],
          correctAnswer: 1,
          explanation:
            "Liveness probe checks if container is alive (e.g., /health endpoint). If fails, Kubernetes RESTARTS the container (fixes deadlocks, crashes). Readiness probe checks if container is ready for traffic (e.g., /ready endpoint checking database connection). If fails, Kubernetes REMOVES pod from service load balancer but doesn't restart (temporary issue, will recover). Need both for proper health management. Option 1 is wrong (different purposes). Option 3 is wrong (both used in all environments). Option 4 is backwards.",
        },
        {
          id: 'mc-deploy-4',
          question:
            'Why must database migrations be backward-compatible in microservices?',
          options: [
            'For better performance',
            'Because old and new service versions coexist during rolling deployment',
            'To reduce database size',
            'For security reasons',
          ],
          correctAnswer: 1,
          explanation:
            "During rolling deployment, old and new service versions run simultaneously. If you add a NOT NULL column, old service version (doesn't know about it) will fail to write to database. Solution: backward-compatible migrations in steps - add as nullable, deploy service, backfill, then enforce NOT NULL. Each step works with both old and new service versions. Option 1 is unrelated. Option 3 is unrelated. Option 4 is not the main reason.",
        },
        {
          id: 'mc-deploy-5',
          question:
            'What is the purpose of Horizontal Pod Autoscaler (HPA) in Kubernetes?',
          options: [
            'To manually scale pods',
            'To automatically add/remove pods based on CPU/memory metrics',
            'To rotate pods regularly',
            'To distribute pods across nodes',
          ],
          correctAnswer: 1,
          explanation:
            "HPA automatically scales the number of pods based on metrics (CPU, memory, custom metrics). Define min/max replicas and target utilization (e.g., 70% CPU). Kubernetes adds pods when load increases, removes when load decreases. This handles variable load without manual intervention. Example: Black Friday traffic spike → HPA scales from 3 to 20 pods → traffic drops → scales back to 3. Option 1 is wrong (automatic, not manual). Option 3 is wrong (that's pod disruption budget). Option 4 is wrong (that's node affinity).",
        },
      ],
    },
    {
      id: 'microservices-security',
      title: 'Microservices Security',
      content: `Security in microservices is more complex than monoliths due to the distributed nature. With many services communicating over the network, the attack surface increases significantly.

## Security Challenges in Microservices

**Compared to monolith**:

**Monolith**:
- Single security perimeter (firewall)
- Internal function calls (no network exposure)
- One authentication point
- Centralized authorization

**Microservices**:
- Multiple security perimeters
- All communication over network (intercept risk)
- Authentication at each service
- Distributed authorization
- More attack surface

---

## Defense in Depth

**Layered security approach** - multiple defensive layers.

\`\`\`
┌─────────────────────────────────────┐
│ Network Security (Firewalls, VPC)  │
├─────────────────────────────────────┤
│ API Gateway (Auth, Rate Limiting)   │
├─────────────────────────────────────┤
│ Service Mesh (mTLS, AuthZ Policies) │
├─────────────────────────────────────┤
│ Service-Level Security              │
├─────────────────────────────────────┤
│ Data Encryption (at rest & transit) │
└─────────────────────────────────────┘
\`\`\`

**Principle**: If one layer fails, others protect the system.

---

## Authentication & Authorization

### 1. API Gateway Authentication

**Gateway authenticates users**, services trust the gateway.

\`\`\`javascript
// API Gateway
async function authenticate(req, res, next) {
    const token = req.headers['authorization']?.replace('Bearer ', '');
    
    if (!token) {
        return res.status(401).json({ error: 'No token provided' });
    }
    
    try {
        // Verify JWT
        const decoded = jwt.verify(token, JWT_SECRET);
        
        // Add user context to headers for downstream services
        req.headers['X-User-Id'] = decoded.userId;
        req.headers['X-User-Email'] = decoded.email;
        req.headers['X-User-Roles'] = JSON.stringify(decoded.roles);
        
        next();
    } catch (error) {
        return res.status(401).json({ error: 'Invalid token' });
    }
}

app.use(authenticate);
\`\`\`

**Downstream service** trusts headers:
\`\`\`javascript
// Order Service
app.post('/orders', async (req, res) => {
    const userId = req.headers['X-User-Id']; // Trusts gateway
    const roles = JSON.parse(req.headers['X-User-Roles'] || '[]');
    
    // Authorization
    if (!roles.includes('customer')) {
        return res.status(403).json({ error: 'Forbidden' });
    }
    
    const order = await createOrder({ ...req.body, userId });
    res.json(order);
});
\`\`\`

**Security concern**: What if attacker bypasses gateway?

**Solution**: Service Mesh with mTLS + Authorization Policies.

### 2. Service-to-Service Authentication

**Problem**: How does Order Service verify Payment Service is really Payment Service?

**Solution**: Mutual TLS (mTLS)

#### Mutual TLS (mTLS)

**Both client and server** authenticate each other with certificates.

**Without mTLS**:
\`\`\`
Order Service → Payment Service
"I'm Order Service" (no proof)
\`\`\`

**With mTLS**:
\`\`\`
Order Service → Payment Service
Order Service presents certificate signed by trusted CA
Payment Service verifies certificate
Payment Service presents its certificate
Order Service verifies certificate
✅ Both authenticated
\`\`\`

**Istio automatically** handles mTLS:
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

**Certificates**:
- Istio provisions certificates to sidecar proxies
- Certificates rotate automatically (every 24 hours)
- Services talk to localhost (proxy handles mTLS)

### 3. OAuth 2.0 for Third-Party Access

**Use case**: Mobile app, partner APIs.

**Flow (Authorization Code)**:
\`\`\`
1. User clicks "Login with Google"
2. Redirect to Google authorization server
3. User grants permission
4. Google redirects back with authorization code
5. App exchanges code for access token
6. App uses access token to call API
\`\`\`

**Example**:
\`\`\`javascript
// API Gateway
app.get('/auth/google', (req, res) => {
    const authUrl = \`https://accounts.google.com/o/oauth2/v2/auth?
        client_id=\${GOOGLE_CLIENT_ID}
        &redirect_uri=\${REDIRECT_URI}
        &response_type=code
        &scope=openid email profile\`;
    res.redirect(authUrl);
});

app.get('/auth/google/callback', async (req, res) => {
    const { code } = req.query;
    
    // Exchange code for tokens
    const response = await axios.post('https://oauth2.googleapis.com/token', {
        code,
        client_id: GOOGLE_CLIENT_ID,
        client_secret: GOOGLE_CLIENT_SECRET,
        redirect_uri: REDIRECT_URI,
        grant_type: 'authorization_code'
    });
    
    const { access_token, id_token } = response.data;
    
    // Verify ID token and create session
    const userInfo = jwt.decode(id_token);
    const sessionToken = createJWT({ userId: userInfo.sub, email: userInfo.email });
    
    res.cookie('session', sessionToken);
    res.redirect('/dashboard');
});
\`\`\`

### 4. Authorization Patterns

#### Role-Based Access Control (RBAC)

**Users have roles**, roles have permissions.

\`\`\`javascript
const roles = {
    customer: ['read:products', 'create:order', 'read:own-orders'],
    admin: ['read:products', 'create:product', 'read:all-orders', 'delete:order'],
    support: ['read:products', 'read:all-orders', 'update:order-status']
};

function authorize(requiredPermission) {
    return (req, res, next) => {
        const userRoles = JSON.parse(req.headers['X-User-Roles'] || '[]');
        const userPermissions = userRoles.flatMap(role => roles[role] || []);
        
        if (!userPermissions.includes(requiredPermission)) {
            return res.status(403).json({ error: 'Forbidden' });
        }
        
        next();
    };
}

// Usage
app.get('/orders', authorize('read:all-orders'), async (req, res) => {
    // Only admin and support can access
    const orders = await getAllOrders();
    res.json(orders);
});
\`\`\`

#### Attribute-Based Access Control (ABAC)

**More flexible** - considers attributes (user, resource, environment).

\`\`\`javascript
function canAccessOrder(user, order, environment) {
    // User is order owner
    if (order.userId === user.id) return true;
    
    // Admin can access all
    if (user.roles.includes('admin')) return true;
    
    // Support can access during business hours
    if (user.roles.includes('support')) {
        const hour = new Date().getHours();
        return hour >= 9 && hour < 17; // 9 AM - 5 PM
    }
    
    return false;
}

app.get('/orders/:id', async (req, res) => {
    const order = await getOrder(req.params.id);
    const user = { id: req.headers['X-User-Id'], roles: JSON.parse(req.headers['X-User-Roles']) };
    
    if (!canAccessOrder(user, order, {})) {
        return res.status(403).json({ error: 'Forbidden' });
    }
    
    res.json(order);
});
\`\`\`

---

## Network Security

### 1. Network Segmentation

**Isolate services** in different network zones.

\`\`\`
┌─────────────────────────────────┐
│ Public Zone (Internet-facing)   │
│ - API Gateway                   │
│ - Web Server                    │
└──────────┬──────────────────────┘
           │ Firewall
┌──────────▼──────────────────────┐
│ Application Zone (Internal)     │
│ - Order Service                 │
│ - User Service                  │
│ - Product Service               │
└──────────┬──────────────────────┘
           │ Firewall
┌──────────▼──────────────────────┐
│ Data Zone (Restricted)          │
│ - Databases                     │
│ - Cache                         │
└─────────────────────────────────┘
\`\`\`

**Kubernetes NetworkPolicy**:
\`\`\`yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: payment-service-policy
spec:
  podSelector:
    matchLabels:
      app: payment-service
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: order-service  # Only Order Service can call Payment
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres  # Payment can only talk to its database
    ports:
    - protocol: TCP
      port: 5432
\`\`\`

**Result**: Even if attacker compromises one service, they can't access others.

### 2. API Gateway Rate Limiting

**Prevent DDoS attacks** and abuse.

\`\`\`javascript
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // 100 requests per window
    message: 'Too many requests, please try again later',
    standardHeaders: true,
    legacyHeaders: false,
    // Rate limit by user ID (if authenticated) or IP
    keyGenerator: (req) => {
        return req.headers['X-User-Id'] || req.ip;
    }
});

app.use('/api/', limiter);
\`\`\`

**Tiered rate limiting**:
\`\`\`javascript
const limits = {
    free: { windowMs: 60000, max: 10 },      // 10 req/min
    pro: { windowMs: 60000, max: 100 },      // 100 req/min
    enterprise: { windowMs: 60000, max: 1000 } // 1000 req/min
};

function getRateLimiter(req) {
    const tier = req.user?.tier || 'free';
    return rateLimit(limits[tier]);
}
\`\`\`

---

## Data Security

### 1. Encryption in Transit

**Always use TLS** for service communication.

**API Gateway to Client**: HTTPS (TLS 1.2+)
**Service to Service**: mTLS (via service mesh)

\`\`\`javascript
// HTTPS server
const https = require('https');
const fs = require('fs');

const options = {
    key: fs.readFileSync('server-key.pem'),
    cert: fs.readFileSync('server-cert.pem')
};

https.createServer(options, app).listen(443);
\`\`\`

### 2. Encryption at Rest

**Encrypt sensitive data** in database.

\`\`\`javascript
const crypto = require('crypto');

// Encryption
function encrypt(text, key) {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv('aes-256-gcm', Buffer.from(key, 'hex'), iv);
    
    let encrypted = cipher.update(text, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    const authTag = cipher.getAuthTag();
    
    return {
        encrypted,
        iv: iv.toString('hex'),
        authTag: authTag.toString('hex')
    };
}

// Decryption
function decrypt(encrypted, iv, authTag, key) {
    const decipher = crypto.createDecipheriv(
        'aes-256-gcm',
        Buffer.from(key, 'hex'),
        Buffer.from(iv, 'hex')
    );
    
    decipher.setAuthTag(Buffer.from(authTag, 'hex'));
    
    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return decrypted;
}

// Store credit card
async function saveCreditCard(userId, cardNumber) {
    const encrypted = encrypt(cardNumber, ENCRYPTION_KEY);
    
    await db.creditCards.insert({
        userId,
        cardNumber: encrypted.encrypted,
        iv: encrypted.iv,
        authTag: encrypted.authTag
    });
}
\`\`\`

**Better**: Use **vault services** (HashiCorp Vault, AWS KMS) for key management.

### 3. Secrets Management

**Never hardcode secrets**. Use secret management tools.

**Kubernetes Secrets**:
\`\`\`yaml
apiVersion: v1
kind: Secret
metadata:
  name: payment-service-secrets
type: Opaque
data:
  stripe-api-key: c2stdGVzdF8xMjM0NTY3ODkw  # base64
  database-password: cGFzc3dvcmQxMjM=        # base64
\`\`\`

**HashiCorp Vault**:
\`\`\`javascript
const vault = require('node-vault')({
    endpoint: 'http://vault:8200',
    token: process.env.VAULT_TOKEN
});

// Read secret
const secret = await vault.read('secret/data/payment-service');
const stripeApiKey = secret.data.data['stripe-api-key'];
\`\`\`

---

## Input Validation & Sanitization

**Validate all inputs** to prevent injection attacks.

### SQL Injection

❌ **Vulnerable**:
\`\`\`javascript
const query = \`SELECT * FROM users WHERE email = '\${req.body.email}'\`;
// Attacker sends: email = "' OR '1'='1"
// Query becomes: SELECT * FROM users WHERE email = '' OR '1'='1'
// Returns all users!
\`\`\`

✅ **Safe** (Parameterized queries):
\`\`\`javascript
const query = 'SELECT * FROM users WHERE email = $1';
const result = await db.query(query, [req.body.email]);
\`\`\`

### NoSQL Injection

❌ **Vulnerable**:
\`\`\`javascript
const user = await db.users.findOne({ email: req.body.email });
// Attacker sends: { "email": { "$ne": null } }
// Returns first user in database!
\`\`\`

✅ **Safe** (Validate input):
\`\`\`javascript
const Joi = require('joi');

const schema = Joi.object({
    email: Joi.string().email().required()
});

const { error, value } = schema.validate(req.body);
if (error) {
    return res.status(400).json({ error: error.details[0].message });
}

const user = await db.users.findOne({ email: value.email });
\`\`\`

### Cross-Site Scripting (XSS)

**Sanitize output**:
\`\`\`javascript
const escapeHtml = require('escape-html');

app.get('/user/:id', async (req, res) => {
    const user = await getUser(req.params.id);
    
    res.send(\`
        <h1>Welcome, \${escapeHtml(user.name)}</h1>
    \`);
});
\`\`\`

---

## Audit Logging

**Log all security-relevant events**.

\`\`\`javascript
function auditLog(event) {
    logger.info({
        event: event.type,
        userId: event.userId,
        resource: event.resource,
        action: event.action,
        result: event.result,
        timestamp: new Date().toISOString(),
        ip: event.ip,
        userAgent: event.userAgent
    });
}

app.post('/orders', async (req, res) => {
    try {
        const order = await createOrder(req.body);
        
        auditLog({
            type: 'ORDER_CREATED',
            userId: req.headers['X-User-Id'],
            resource: 'order',
            action: 'create',
            result: 'success',
            ip: req.ip,
            userAgent: req.headers['user-agent']
        });
        
        res.json(order);
    } catch (error) {
        auditLog({
            type: 'ORDER_CREATE_FAILED',
            userId: req.headers['X-User-Id'],
            resource: 'order',
            action: 'create',
            result: 'failure',
            error: error.message,
            ip: req.ip
        });
        
        throw error;
    }
});
\`\`\`

**What to log**:
- Authentication attempts (success/failure)
- Authorization failures
- Data access (sensitive data)
- Configuration changes
- Service-to-service calls

**What NOT to log**:
- Passwords
- API keys
- Credit card numbers
- PII without masking

---

## Security Headers

**Add security headers** to all responses.

\`\`\`javascript
app.use((req, res, next) => {
    // Prevent clickjacking
    res.setHeader('X-Frame-Options', 'DENY');
    
    // Prevent MIME sniffing
    res.setHeader('X-Content-Type-Options', 'nosniff');
    
    // XSS Protection
    res.setHeader('X-XSS-Protection', '1; mode=block');
    
    // Content Security Policy
    res.setHeader('Content-Security-Policy', "default-src 'self'");
    
    // HSTS (force HTTPS)
    res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
    
    next();
});
\`\`\`

---

## Dependency Scanning

**Scan for vulnerabilities** in dependencies.

\`\`\`bash
# npm audit
npm audit

# Fix vulnerabilities
npm audit fix

# Snyk
snyk test
snyk monitor

# OWASP Dependency Check
dependency-check --project my-service --scan .
\`\`\`

**Automate** in CI/CD pipeline:
\`\`\`yaml
# .gitlab-ci.yml
security-scan:
  stage: test
  script:
    - npm audit
    - snyk test --severity-threshold=high
  allow_failure: false  # Fail build if vulnerabilities found
\`\`\`

---

## Container Security

### 1. Use Minimal Base Images

\`\`\`dockerfile
# ❌ Large attack surface
FROM node:18

# ✅ Minimal image
FROM node:18-alpine
\`\`\`

### 2. Run as Non-Root User

\`\`\`dockerfile
FROM node:18-alpine

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nodejs -u 1001

WORKDIR /app
COPY --chown=nodejs:nodejs . .

# Switch to non-root user
USER nodejs

CMD ["node", "server.js"]
\`\`\`

### 3. Scan Images for Vulnerabilities

\`\`\`bash
# Trivy
trivy image order-service:v1.2.3

# Clair
clairctl analyze order-service:v1.2.3
\`\`\`

---

## Best Practices

1. **Defense in depth**: Multiple security layers
2. **Zero trust**: Verify everything, trust nothing
3. **mTLS**: Encrypt all service-to-service communication
4. **Least privilege**: Minimum permissions needed
5. **Input validation**: Validate all inputs
6. **Audit logging**: Log security events
7. **Secrets management**: Use vaults, never hardcode
8. **Dependency scanning**: Regular vulnerability checks
9. **Network segmentation**: Isolate services
10. **Security headers**: Add to all responses

---

## Key Takeaways

1. **Distributed** systems have larger attack surface
2. **mTLS** authenticates services to each other
3. **API Gateway** centralizes authentication
4. **Network policies** restrict service communication
5. **Encrypt** data in transit (TLS) and at rest
6. **Validate** all inputs to prevent injection
7. **Audit log** all security events
8. **Scan** dependencies and containers for vulnerabilities`,
      quiz: [
        {
          id: 'q1-security',
          question:
            'Why is security more challenging in microservices compared to monoliths? What additional measures are needed?',
          sampleAnswer:
            'Microservices increase attack surface: (1) All communication over network (vs in-process function calls) - can be intercepted, (2) Multiple entry points (vs single perimeter), (3) Authentication needed at each service, (4) More services to secure and monitor. Additional measures: (1) mTLS for service-to-service authentication, (2) API Gateway for centralized auth, (3) Network policies to restrict communication, (4) Service mesh for automatic encryption, (5) Distributed audit logging, (6) Secrets management (Vault) for credentials. Defense in depth is critical - multiple layers of security so if one fails, others protect the system.',
          keyPoints: [
            'More attack surface (network communication, multiple services)',
            'Need mTLS for service-to-service authentication',
            'API Gateway centralizes user authentication',
            'Network policies restrict which services can communicate',
            'Defense in depth: multiple security layers',
          ],
        },
        {
          id: 'q2-security',
          question:
            'Explain mTLS (Mutual TLS). How does it differ from regular TLS? Why is it important for microservices?',
          sampleAnswer:
            "Regular TLS: Only server authenticates itself (client verifies server certificate). Example: Your browser verifies bank.com is really the bank. mTLS (Mutual TLS): BOTH client and server authenticate each other with certificates. Both parties present and verify certificates. Important for microservices because: (1) Services need to verify each other (Order Service verifies it's really talking to Payment Service, not an attacker), (2) Prevents spoofing attacks, (3) Encrypts all communication, (4) Works automatically with service mesh (Istio). Without mTLS, attacker could pretend to be a service. Service mesh makes mTLS automatic - provisions certificates, rotates them, handles all crypto without code changes.",
          keyPoints: [
            'mTLS: both parties authenticate (vs TLS: only server)',
            'Both present and verify certificates',
            'Prevents service spoofing attacks',
            'Service mesh (Istio) makes it automatic',
            'Certificates rotate automatically',
          ],
        },
        {
          id: 'q3-security',
          question:
            'What security vulnerabilities should you watch for when accepting user input? How do you prevent them?',
          sampleAnswer:
            'Vulnerabilities: (1) SQL Injection - attacker injects SQL code (email="\' OR \'1\'=\'1"). Prevention: parameterized queries/prepared statements. (2) NoSQL Injection - attacker sends {email: {"$ne": null}}. Prevention: validate input types with schema validator (Joi, Zod). (3) XSS (Cross-Site Scripting) - attacker injects JavaScript. Prevention: escape HTML output, use Content-Security-Policy header. (4) Command Injection - attacker injects shell commands. Prevention: avoid exec(), validate inputs. General rule: NEVER trust user input. Always validate (whitelist approach, not blacklist), sanitize output, use parameterized queries, and implement input length limits. Use validation libraries (Joi, Zod) rather than custom regex.',
          keyPoints: [
            'SQL Injection: use parameterized queries',
            'NoSQL Injection: validate input types with schema',
            'XSS: escape HTML output, CSP headers',
            'Never trust user input - always validate',
            'Use validation libraries (Joi, Zod)',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-security-1',
          question:
            'What is the primary purpose of mTLS (Mutual TLS) in microservices?',
          options: [
            'To make services faster',
            'To authenticate both client and server, encrypting all communication',
            'To reduce network latency',
            'To store passwords securely',
          ],
          correctAnswer: 1,
          explanation:
            "mTLS (Mutual TLS) ensures both the client and server authenticate each other using certificates, and encrypts all communication. This prevents service spoofing (attacker pretending to be Payment Service) and eavesdropping. Regular TLS only authenticates the server. In microservices, services need to verify they're talking to legitimate services, not attackers. Service mesh (Istio) can automate mTLS setup. Option 1 is wrong (mTLS adds slight overhead). Option 3 is wrong (mTLS adds latency). Option 4 is unrelated (that's hashing/encryption).",
        },
        {
          id: 'mc-security-2',
          question:
            'Why should you use parameterized queries instead of string concatenation for database queries?',
          options: [
            'Parameterized queries are faster',
            'To prevent SQL injection attacks',
            'To reduce database load',
            'For better error messages',
          ],
          correctAnswer: 1,
          explanation:
            "Parameterized queries (prepared statements) prevent SQL injection by separating SQL code from data. With string concatenation, attacker can inject malicious SQL: email=\"' OR '1'='1\" becomes SELECT * FROM users WHERE email='' OR '1'='1' (returns all users!). Parameterized queries treat user input as data, not code: db.query(\"SELECT * FROM users WHERE email=$1\", [email]). The database escapes input automatically. This is the #1 defense against SQL injection. Option 1 is sometimes true but not the main reason. Options 3 and 4 are not primary benefits.",
        },
        {
          id: 'mc-security-3',
          question:
            'What is the principle of "defense in depth" in microservices security?',
          options: [
            'Use only one very strong security measure',
            'Multiple layers of security so if one fails, others protect the system',
            'Deep encryption algorithms',
            'Hiding services from the internet',
          ],
          correctAnswer: 1,
          explanation:
            'Defense in depth means multiple security layers: network firewalls, API Gateway auth, service mesh mTLS, service-level authorization, encryption, audit logging. If one layer is breached, others still protect. Example: Attacker bypasses API Gateway → still blocked by network policies → still can\'t decrypt mTLS → still blocked by service-level auth. Single security measure (Option 1) is risky - single point of failure. Option 3 misunderstands "depth". Option 4 is security by obscurity (bad practice).',
        },
        {
          id: 'mc-security-4',
          question:
            'Where should you store sensitive secrets (API keys, passwords) in Kubernetes?',
          options: [
            'Hardcoded in application code',
            'In Kubernetes Secrets or external secret management (HashiCorp Vault)',
            'In environment variables in Dockerfile',
            'In source control (Git)',
          ],
          correctAnswer: 1,
          explanation:
            'Use Kubernetes Secrets (base64 encoded, access-controlled) or external secret management systems (HashiCorp Vault, AWS Secrets Manager) for sensitive data. These provide encryption, access control, audit logging, and secret rotation. Option 1 is extremely dangerous (secrets in code). Option 3 exposes secrets in container images. Option 4 is worst (secrets in Git history forever). Never commit secrets to source control. Use secret management tools with proper encryption and access controls.',
        },
        {
          id: 'mc-security-5',
          question:
            'What is the purpose of Network Policies in Kubernetes for microservices security?',
          options: [
            'To make network faster',
            'To restrict which pods can communicate with each other',
            'To encrypt data at rest',
            'To load balance traffic',
          ],
          correctAnswer: 1,
          explanation:
            "Network Policies restrict pod-to-pod communication at network level. Example: Only Order Service can call Payment Service, Payment Service can only access its database. This implements principle of least privilege and limits blast radius if a service is compromised. Without network policies, any pod can talk to any pod (dangerous). Option 1 is wrong (no performance benefit). Option 3 is wrong (that's encryption at rest). Option 4 is wrong (that's Services/Ingress).",
        },
      ],
    },
    {
      id: 'microservices-monitoring',
      title: 'Microservices Monitoring',
      content: `Monitoring microservices is fundamentally different from monoliths. With distributed services, you need comprehensive observability across the entire system.

## The Three Pillars of Observability

### 1. Metrics

**What**: Numeric measurements over time (CPU, memory, request rate, latency).

**Tools**: Prometheus, Grafana, Datadog

**RED Metrics** (for every service):
- **Rate**: Requests per second
- **Errors**: Error rate (%)
- **Duration**: Latency (p50, p95, p99)

**Example metrics**:
\`\`\`
http_requests_total{service="order-service", method="POST", status="200"} 1547
http_request_duration_seconds{service="order-service", quantile="0.95"} 0.245
\`\`\`

**Implementation** (Prometheus):
\`\`\`javascript
const promClient = require('prom-client');
const register = new promClient.Registry();

// Counter: total requests
const httpRequestsTotal = new promClient.Counter({
    name: 'http_requests_total',
    help: 'Total HTTP requests',
    labelNames: ['method', 'route', 'status_code'],
    registers: [register]
});

// Histogram: request duration
const httpRequestDuration = new promClient.Histogram({
    name: 'http_request_duration_seconds',
    help: 'HTTP request duration in seconds',
    labelNames: ['method', 'route', 'status_code'],
    buckets: [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10],
    registers: [register]
});

// Middleware
app.use((req, res, next) => {
    const end = httpRequestDuration.startTimer();
    
    res.on('finish', () => {
        httpRequestsTotal.inc({
            method: req.method,
            route: req.route?.path || 'unknown',
            status_code: res.statusCode
        });
        
        end({
            method: req.method,
            route: req.route?.path || 'unknown',
            status_code: res.statusCode
        });
    });
    
    next();
});

// Expose metrics endpoint
app.get('/metrics', async (req, res) => {
    res.set('Content-Type', register.contentType);
    res.end(await register.metrics());
});
\`\`\`

**Prometheus scrapes** the /metrics endpoint every 15s.

**Grafana Dashboard**:
\`\`\`
Order Service Dashboard
━━━━━━━━━━━━━━━━━━━━━━━
Request Rate: 245 req/s  📈
Error Rate: 0.5%         ✅
P95 Latency: 245ms       ⚠️
P99 Latency: 1.2s        ❌

[Graph showing request rate over time]
[Graph showing latency percentiles]
\`\`\`

### 2. Logs

**What**: Text records of events (structured or unstructured).

**Tools**: ELK Stack (Elasticsearch, Logstash, Kibana), Loki, Splunk

**Structured logging**:
\`\`\`javascript
const logger = require('pino')();

// Bad: Unstructured
logger.info('User 123 created order 456 for $99.99');

// Good: Structured
logger.info({
    event: 'order_created',
    userId: '123',
    orderId: '456',
    amount: 99.99,
    currency: 'USD',
    timestamp: new Date().toISOString()
});
\`\`\`

**Correlation ID** for tracing across services:
\`\`\`javascript
app.use((req, res, next) => {
    req.correlationId = req.headers['x-correlation-id'] || uuidv4();
    res.setHeader('x-correlation-id', req.correlationId);
    next();
});

// Log with correlation ID
logger.info({
    correlationId: req.correlationId,
    event: 'processing_order',
    orderId: order.id
});

// Pass to downstream services
await axios.post('http://payment-service/charge', payment, {
    headers: {
        'x-correlation-id': req.correlationId
    }
});
\`\`\`

**Centralized logging** (ELK):
\`\`\`
All services → Logstash → Elasticsearch → Kibana

Query: correlationId:"abc-123"
Results:
[Order Service] 10:00:01 - Order created
[Inventory Service] 10:00:02 - Inventory reserved
[Payment Service] 10:00:03 - Payment charged
[Shipping Service] 10:00:04 - Shipment created
\`\`\`

### 3. Distributed Tracing

**What**: Track requests across multiple services.

**Tools**: Jaeger, Zipkin, AWS X-Ray

**Problem**: Request touches 5 services, where's the bottleneck?

**Solution**: Distributed tracing shows entire request flow.

**Implementation** (OpenTelemetry):
\`\`\`javascript
const { NodeTracerProvider } = require('@opentelemetry/sdk-trace-node');
const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');

const provider = new NodeTracerProvider();
provider.addSpanProcessor(new SimpleSpanProcessor(new JaegerExporter({
    endpoint: 'http://jaeger:14268/api/traces'
})));
provider.register();

const tracer = provider.getTracer('order-service');

// Create span
app.post('/orders', async (req, res) => {
    const span = tracer.startSpan('create_order');
    
    try {
        const order = await createOrder(req.body);
        span.setAttributes({
            'order.id': order.id,
            'user.id': req.body.userId
        });
        
        res.json(order);
    } catch (error) {
        span.recordException(error);
        span.setStatus({ code: SpanStatusCode.ERROR });
        throw error;
    } finally {
        span.end();
    }
});
\`\`\`

**Jaeger UI**:
\`\`\`
Trace ID: abc-123-def-456
Duration: 1.2s

Timeline:
├─ API Gateway (50ms)
│  └─ Order Service (1150ms)
│     ├─ Payment Service (800ms) ⬅ SLOW!
│     │  └─ Fraud Check (750ms) ⬅ BOTTLENECK!
│     └─ Inventory Service (150ms)
│        └─ Database Query (120ms)

Root cause: Fraud check is slow
\`\`\`

---

## Alerting

**Don't** alert on everything. Alert on **actionable** symptoms.

### Good Alerts

**Symptom-based** (user-facing):
\`\`\`yaml
# High error rate
- alert: HighErrorRate
  expr: |
    rate(http_requests_total{status_code=~"5.."}[5m]) 
    / 
    rate(http_requests_total[5m]) > 0.05
  for: 5m
  annotations:
    summary: "Error rate > 5% for 5 minutes"
    
# High latency
- alert: HighLatency
  expr: |
    histogram_quantile(0.95, 
      rate(http_request_duration_seconds_bucket[5m])
    ) > 1
  for: 10m
  annotations:
    summary: "P95 latency > 1s for 10 minutes"

# Service down
- alert: ServiceDown
  expr: up{job="order-service"} == 0
  for: 1m
  annotations:
    summary: "Order service is down"
\`\`\`

### Bad Alerts

**Cause-based** (not actionable):
\`\`\`
❌ Alert: CPU > 80%
   Problem: So what? Is it affecting users?
   Better: Alert on high latency or errors
   
❌ Alert: Disk > 90%
   Problem: When will it fill up? 
   Better: Predict when it will reach 100%
   
❌ Alert: A single request failed
   Problem: Too noisy, not actionable
   Better: Alert on error rate > 5% for 5 minutes
\`\`\`

---

## Service-Level Objectives (SLOs)

**Define** acceptable reliability targets.

### SLI (Service Level Indicator)

**Measurement** of service health:
- Availability: % of requests that succeed
- Latency: % of requests below threshold
- Throughput: Requests per second

### SLO (Service Level Objective)

**Target** for SLI:
- 99.9% of requests succeed (availability)
- 95% of requests complete < 500ms (latency)

### SLA (Service Level Agreement)

**Contract** with users (with penalties):
- If availability < 99.9%, refund 10% of monthly fee

**Example SLO**:
\`\`\`yaml
Service: order-service
SLO:
  - availability: 99.9% (43 minutes downtime/month)
  - latency_p95: < 500ms
  - latency_p99: < 1000ms
  
Error Budget:
  - 0.1% of requests can fail (99.9% SLO)
  - At 1M requests/month: 1,000 errors allowed
  - Current: 500 errors used (50% of budget)
  - Remaining: 500 errors
\`\`\`

**Error Budget Policy**:
- Budget remaining: Keep shipping features
- Budget exhausted: Focus on reliability

---

## Dashboards

### Service Dashboard

**For each service**:
\`\`\`
Order Service Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔷 Health: Healthy ✅

📊 RED Metrics (last 1 hour)
├─ Request Rate: 245 req/s
├─ Error Rate: 0.5%
└─ Latency: P50=120ms, P95=250ms, P99=500ms

💾 Resources
├─ CPU: 45% (3 pods)
├─ Memory: 60% (768MB / 1.5GB)
└─ Disk: 30%

🔗 Dependencies
├─ Payment Service: Healthy ✅
├─ Inventory Service: Healthy ✅
└─ Database: Healthy ✅

⚠️ Recent Alerts
└─ None

📈 [Request Rate Graph]
📈 [Latency Distribution Graph]
📈 [Error Rate Graph]
\`\`\`

### System Dashboard

**Overall health**:
\`\`\`
System Overview
━━━━━━━━━━━━━━━━━━━━━━━━━
🌍 Total Request Rate: 2,500 req/s
📊 Overall Error Rate: 0.3%
⚡ P95 Latency: 300ms

Services (10 total):
✅ 9 Healthy
⚠️ 1 Degraded (Recommendation Service)

[Service Map showing interconnections]
[Request flow waterfall]
\`\`\`

---

## Anomaly Detection

**Automatically** detect unusual patterns.

**Example**: Request rate normally 100 req/s ± 10%

Sudden spike to 1000 req/s → Alert!
Could be: Attack, viral content, or legitimate traffic spike

**Tools**: Datadog anomaly detection, Prometheus predict_linear

\`\`\`yaml
- alert: AnomalousRequestRate
  expr: |
    abs(rate(http_requests_total[5m]) - avg_over_time(rate(http_requests_total[5m])[1h:5m]))
    /
    stddev_over_time(rate(http_requests_total[5m])[1h:5m])
    > 3
  annotations:
    summary: "Request rate is 3 standard deviations from normal"
\`\`\`

---

## Health Checks

**Different levels** of health:

### Liveness

**Is service alive?**
\`\`\`javascript
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});
\`\`\`

Kubernetes restarts if fails.

### Readiness

**Is service ready for traffic?**
\`\`\`javascript
app.get('/ready', async (req, res) => {
    try {
        // Check dependencies
        await database.ping();
        await paymentServiceClient.healthCheck();
        
        res.json({ status: 'ready' });
    } catch (error) {
        res.status(503).json({ 
            status: 'not ready',
            error: error.message
        });
    }
});
\`\`\`

Kubernetes removes from load balancer if fails.

### Deep Health

**Detailed health of all components**:
\`\`\`javascript
app.get('/health/deep', async (req, res) => {
    const health = {
        status: 'healthy',
        checks: {
            database: await checkDatabase(),
            redis: await checkRedis(),
            paymentService: await checkPaymentService(),
            inventoryService: await checkInventoryService()
        }
    };
    
    const hasFailure = Object.values(health.checks)
        .some(check => check.status === 'unhealthy');
    
    if (hasFailure) {
        health.status = 'degraded';
    }
    
    res.json(health);
});

async function checkDatabase() {
    try {
        await database.ping();
        return { status: 'healthy', latency: 5 };
    } catch (error) {
        return { status: 'unhealthy', error: error.message };
    }
}
\`\`\`

---

## Cost Monitoring

**Track infrastructure costs** per service.

**Example**:
\`\`\`
Monthly Costs by Service:
├─ Order Service: $500
│  ├─ Compute: $300 (3 pods @ $100/pod)
│  ├─ Database: $150
│  └─ Networking: $50
├─ Payment Service: $800
│  ├─ Compute: $400
│  ├─ Database: $300
│  └─ External API: $100
└─ Inventory Service: $400

Total: $1,700/month

Cost per Request: $0.0005
\`\`\`

**Optimize**: 
- Autoscaling (reduce idle pods)
- Right-sizing (don't over-provision)
- Reserved instances (save 30-50%)

---

## Monitoring Best Practices

1. **Monitor symptoms, not causes** (alert on errors, not CPU)
2. **Set SLOs** and track error budgets
3. **Use correlation IDs** to trace requests across services
4. **Structured logging** for easy querying
5. **Dashboard per service** + system overview
6. **Alert only on actionable issues** (no noise)
7. **Distributed tracing** for debugging
8. **Monitor costs** per service
9. **Health checks** at multiple levels (liveness, readiness)
10. **Anomaly detection** for unusual patterns

---

## Interview Tips

**Red Flags**:
❌ Only monitoring server metrics (CPU, memory)
❌ No distributed tracing
❌ Alerts on everything (alert fatigue)

**Good Responses**:
✅ Explain three pillars: metrics, logs, traces
✅ Mention RED metrics (Rate, Errors, Duration)
✅ Discuss SLOs and error budgets
✅ Talk about correlation IDs
✅ Mention specific tools (Prometheus, Jaeger, ELK)

**Sample Answer**:
*"For microservices monitoring, I'd implement the three pillars of observability: (1) Metrics - Prometheus scraping RED metrics (Rate, Errors, Duration) from each service with Grafana dashboards, (2) Logs - Structured logging with correlation IDs sent to ELK stack for centralized querying, (3) Distributed Tracing - Jaeger/OpenTelemetry to track requests across services and identify bottlenecks. I'd define SLOs (99.9% availability, P95 < 500ms) and alert on SLO violations, not symptoms. Each service would have liveness and readiness probes for Kubernetes auto-healing."*

---

## Key Takeaways

1. **Three pillars**: Metrics, Logs, Distributed Tracing
2. **RED metrics**: Rate, Errors, Duration (for every service)
3. **Structured logging** with correlation IDs
4. **Distributed tracing** tracks requests across services
5. **SLOs** define reliability targets, error budgets guide priorities
6. **Alert on symptoms** (errors, latency) not causes (CPU)
7. **Health checks** enable automatic recovery
8. **Dashboards** per service + system overview
9. **Tools**: Prometheus, Grafana, Jaeger, ELK`,
      quiz: [
        {
          id: 'q1-monitoring',
          question:
            'Explain the three pillars of observability and how they complement each other in microservices.',
          sampleAnswer:
            "Three pillars: (1) Metrics - numeric measurements over time (request rate, latency, errors). Good for: alerting, dashboards, trends. Tools: Prometheus/Grafana. (2) Logs - text records of events. Good for: debugging specific issues, audit trails. Tools: ELK stack. (3) Distributed Tracing - tracks requests across services showing full call path and timing. Good for: finding bottlenecks, understanding dependencies. Tools: Jaeger. They complement: Metrics alert you THAT there's a problem (P95 latency spiked), Logs tell you WHAT happened (specific errors), Traces show you WHERE the problem is (which service is slow). Example: Alert fires for high latency → check traces to find slow service → check logs for that service to see errors. Correlation IDs tie them together.",
          keyPoints: [
            'Metrics: numeric measurements, alerting, trends (Prometheus)',
            'Logs: text records, debugging, audit (ELK)',
            'Traces: request flow across services, bottlenecks (Jaeger)',
            'Complement: Metrics=THAT, Logs=WHAT, Traces=WHERE',
            'Correlation IDs tie them together across services',
          ],
        },
        {
          id: 'q2-monitoring',
          question:
            'What are RED metrics? Why are they important for microservices monitoring?',
          sampleAnswer:
            "RED metrics: (1) Rate - requests per second, (2) Errors - error rate percentage, (3) Duration - latency percentiles (P50, P95, P99). Important because they measure user-facing impact directly. Rate shows traffic patterns and helps capacity planning. Errors show user-facing failures (alert if > 1%). Duration shows user experience (alert if P95 > 500ms). These are SYMPTOMS that users experience, not causes. Better than monitoring CPU/memory (causes) which don't always correlate with user impact. Example: CPU at 80% but P95 latency is 100ms → no problem. CPU at 50% but P95 latency is 2s → big problem. Monitor every service with RED metrics, aggregate for system view.",
          keyPoints: [
            'Rate: requests/second (traffic patterns, capacity)',
            'Errors: error rate % (user-facing failures)',
            'Duration: latency P50/P95/P99 (user experience)',
            'Measure user-facing symptoms, not causes',
            "Better than CPU/memory (causes that don't always affect users)",
          ],
        },
        {
          id: 'q3-monitoring',
          question:
            'How do correlation IDs help with debugging in microservices? How do you implement them?',
          sampleAnswer:
            'Correlation ID is unique identifier attached to every request as it flows through multiple services. Helps debug by: (1) Trace single request across all services, (2) Correlate logs from different services, (3) Find which service failed. Implementation: (1) API Gateway generates UUID when request enters system, (2) Add to HTTP header X-Correlation-ID, (3) Each service extracts correlation ID, includes in all logs, passes to downstream services in headers, (4) Store in thread-local storage or request context. Querying: Search logs/traces by correlation ID to see full request flow. Example: User reports "order failed" → get correlation ID from their request → query logs for that ID → see Order Service succeeded, Payment Service failed with "insufficient funds".',
          keyPoints: [
            'Correlation ID: unique identifier per request',
            'Traces request flow across all services',
            'Generated at API Gateway, passed via headers',
            'Included in all logs and traces',
            'Query by correlation ID to debug specific request',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-monitoring-1',
          question: 'What are the three pillars of observability?',
          options: [
            'CPU, Memory, Disk',
            'Metrics, Logs, Distributed Tracing',
            'Frontend, Backend, Database',
            'Unit tests, Integration tests, E2E tests',
          ],
          correctAnswer: 1,
          explanation:
            'The three pillars of observability are: (1) Metrics - numeric measurements over time (request rate, latency, errors), (2) Logs - text records of events for debugging, (3) Distributed Tracing - tracking requests across multiple services. Together they provide complete visibility into distributed systems. Option 1 lists infrastructure metrics (part of observability but not the pillars). Options 3 and 4 are unrelated concepts.',
        },
        {
          id: 'mc-monitoring-2',
          question: 'What are RED metrics and why are they important?',
          options: [
            'Redis, Elasticsearch, Docker',
            'Rate, Errors, Duration - measure user-facing service health',
            'Read, Execute, Debug',
            'Replica, Endpoint, Deployment',
          ],
          correctAnswer: 1,
          explanation:
            'RED metrics measure user-facing health: Rate (requests/second), Errors (error percentage), Duration (latency percentiles p50/p95/p99). These are symptoms that directly impact users, making them better than cause-based metrics like CPU/memory. Monitor RED metrics for every microservice. Options 1, 3, and 4 are not related to monitoring metrics.',
        },
        {
          id: 'mc-monitoring-3',
          question: 'What is the purpose of a correlation ID in microservices?',
          options: [
            'To generate unique database IDs',
            'To track a single request as it flows through multiple services',
            'To correlate CPU and memory usage',
            'To link services together',
          ],
          correctAnswer: 1,
          explanation:
            'Correlation ID is a unique identifier (UUID) attached to each request, passed through all services via HTTP headers. It allows you to trace a single request across multiple services in logs and traces, making debugging much easier. Example: query logs by correlation ID to see full request flow. Option 1 confuses correlation ID with database ID. Option 3 is nonsensical. Option 4 misunderstands the concept.',
        },
        {
          id: 'mc-monitoring-4',
          question:
            'When should you trigger an alert in microservices monitoring?',
          options: [
            'When CPU usage exceeds 80%',
            'When user-facing symptoms occur (high error rate > 5%, P95 latency > 1s)',
            'For every single request failure',
            'When disk space reaches 50%',
          ],
          correctAnswer: 1,
          explanation:
            "Alert on user-facing symptoms: error rate > threshold (e.g., 5% for 5 minutes), latency exceeding SLO (P95 > 1s), or service completely down. Don't alert on causes (CPU, disk) unless they directly impact users. Don't alert on single failures (too noisy). Option 1 - high CPU doesn't always affect users. Option 3 - single failures create alert fatigue. Option 4 - 50% disk isn't urgent (alert at 90% and predict when it will fill).",
        },
        {
          id: 'mc-monitoring-5',
          question: 'What is an SLO (Service Level Objective)?',
          options: [
            'A slow service that needs optimization',
            'A target for reliability metrics (e.g., 99.9% availability, P95 < 500ms)',
            'A service level agreement with legal penalties',
            'A type of Kubernetes object',
          ],
          correctAnswer: 1,
          explanation:
            'SLO is a target for service reliability: "99.9% of requests succeed" (availability SLO) or "95% of requests < 500ms" (latency SLO). SLOs define acceptable service performance and error budgets (0.1% of requests can fail). When budget exhausted, focus on reliability instead of features. Option 3 confuses SLO with SLA (SLA is contract with penalties, SLO is internal target). Options 1 and 4 are unrelated.',
        },
      ],
    },
    {
      id: 'event-driven-microservices',
      title: 'Event-Driven Microservices',
      content: `Event-driven architecture enables loose coupling, scalability, and resilience in microservices. Services communicate through events rather than direct calls.

## What is Event-Driven Architecture?

**Synchronous** (request-response):
\`\`\`
Order Service → calls → Payment Service (waits for response)
                         ↓
                      Returns result
\`\`\`

**Asynchronous** (event-driven):
\`\`\`
Order Service → publishes OrderCreated event → Message Bus
                                                    ↓
                                          Payment Service subscribes
\`\`\`

**Key difference**: Order Service doesn't wait for Payment Service.

---

## Events vs Commands

### Commands

**Intent**: Tell service to do something

**Example**: \`ChargePayment\`, \`ReserveInventory\`

**Sent to**: Specific service

**Response**: Success or failure

### Events

**Intent**: Announce something happened

**Example**: \`OrderCreated\`, \`PaymentCompleted\`

**Sent to**: Anyone interested (pub/sub)

**Response**: None (fire and forget)

**Comparison**:
\`\`\`javascript
// Command (synchronous)
const result = await paymentService.chargePayment({
    orderId: '123',
    amount: 99.99
});

if (result.success) {
    // Continue
} else {
    // Handle failure
}

// Event (asynchronous)
await eventBus.publish('OrderCreated', {
    orderId: '123',
    userId: 'user-456',
    total: 99.99
});
// Don't wait for response, continue immediately
\`\`\`

---

## Message Brokers

**Tools**: RabbitMQ, Apache Kafka, AWS SQS/SNS, Google Pub/Sub

### RabbitMQ

**Good for**: Task queues, RPC, routing patterns

**Pattern**: Exchanges + Queues

\`\`\`javascript
const amqp = require('amqplib');

// Publisher
async function publishOrderCreated(order) {
    const connection = await amqp.connect('amqp://localhost');
    const channel = await connection.createChannel();
    
    const exchange = 'orders';
    await channel.assertExchange(exchange, 'topic', { durable: true });
    
    channel.publish(
        exchange,
        'order.created',
        Buffer.from(JSON.stringify(order)),
        { persistent: true }
    );
    
    console.log('Published OrderCreated event');
}

// Subscriber
async function subscribeToOrderEvents() {
    const connection = await amqp.connect('amqp://localhost');
    const channel = await connection.createChannel();
    
    const exchange = 'orders';
    const queue = 'payment-service-orders';
    
    await channel.assertExchange(exchange, 'topic', { durable: true });
    await channel.assertQueue(queue, { durable: true });
    await channel.bindQueue(queue, exchange, 'order.*');
    
    channel.consume(queue, (msg) => {
        const event = JSON.parse(msg.content.toString());
        console.log('Received:', event);
        
        // Process event
        processOrderCreated(event);
        
        // Acknowledge
        channel.ack(msg);
    });
}
\`\`\`

### Apache Kafka

**Good for**: High throughput, event streaming, event sourcing

**Pattern**: Topics + Partitions

\`\`\`javascript
const { Kafka } = require('kafkajs');

const kafka = new Kafka({
    clientId: 'order-service',
    brokers: ['localhost:9092']
});

// Producer
async function publishOrderCreated(order) {
    const producer = kafka.producer();
    await producer.connect();
    
    await producer.send({
        topic: 'orders',
        messages: [
            {
                key: order.id,  // Partition key
                value: JSON.stringify({
                    eventType: 'OrderCreated',
                    data: order,
                    timestamp: new Date().toISOString()
                })
            }
        ]
    });
    
    await producer.disconnect();
}

// Consumer
async function subscribeToOrderEvents() {
    const consumer = kafka.consumer({ groupId: 'payment-service' });
    await consumer.connect();
    await consumer.subscribe({ topic: 'orders', fromBeginning: false });
    
    await consumer.run({
        eachMessage: async ({ topic, partition, message }) => {
            const event = JSON.parse(message.value.toString());
            
            if (event.eventType === 'OrderCreated') {
                await processOrderCreated(event.data);
            }
        }
    });
}
\`\`\`

---

## Event Design

### Event Structure

**Good event**:
\`\`\`javascript
{
    eventId: "evt_abc123",           // Unique ID
    eventType: "OrderCreated",        // Event type
    eventVersion: "1.0",              // Schema version
    timestamp: "2024-01-15T10:30:00Z", // When it happened
    source: "order-service",          // Who published it
    data: {                           // Event payload
        orderId: "order-456",
        userId: "user-789",
        items: [
            { productId: "prod-1", quantity: 2, price: 29.99 }
        ],
        total: 59.98,
        currency: "USD"
    },
    metadata: {                       // Context
        correlationId: "req_xyz",
        causationId: "evt_previous",
        userId: "user-789"
    }
}
\`\`\`

### Event Versioning

**Problem**: Event schema changes

**Solution**: Version events

\`\`\`javascript
// Version 1.0
{
    eventType: "OrderCreated",
    eventVersion: "1.0",
    data: {
        orderId: "123",
        total: 99.99
    }
}

// Version 2.0 (added currency)
{
    eventType: "OrderCreated",
    eventVersion: "2.0",
    data: {
        orderId: "123",
        total: 99.99,
        currency: "USD"  // NEW
    }
}

// Consumer handles both versions
function handleOrderCreated(event) {
    const { data, eventVersion } = event;
    
    let currency = 'USD';  // Default
    
    if (eventVersion === '2.0') {
        currency = data.currency;
    }
    
    // Process with currency
}
\`\`\`

---

## Event Patterns

### 1. Event Notification

**Notify** other services something happened.

**Example**: Order created → notify email service

\`\`\`javascript
// Order Service
await eventBus.publish('OrderCreated', { orderId, userId, total });

// Email Service
eventBus.on('OrderCreated', async (event) => {
    await sendOrderConfirmationEmail(event.userId, event.orderId);
});

// Analytics Service
eventBus.on('OrderCreated', async (event) => {
    await trackOrderMetric(event);
});
\`\`\`

**Benefits**:
✅ Services decoupled
✅ Easy to add new subscribers
✅ Publisher doesn't know subscribers

### 2. Event-Carried State Transfer

**Include** all necessary data in event (reduce queries).

\`\`\`javascript
// ❌ Bad: Minimal data
{
    eventType: "OrderCreated",
    data: {
        orderId: "123"
    }
}
// Subscriber must call Order Service to get details

// ✅ Good: Include all data
{
    eventType: "OrderCreated",
    data: {
        orderId: "123",
        userId: "user-456",
        items: [{ productId: "prod-1", quantity: 2 }],
        total: 59.98,
        shippingAddress: {...}
    }
}
// Subscriber has everything it needs
\`\`\`

### 3. Event Sourcing

**Store** all changes as events (event log is source of truth).

**Traditional**:
\`\`\`sql
-- Current state only
SELECT * FROM orders WHERE id = 123;
-- Result: { id: 123, status: "SHIPPED", total: 99.99 }
-- Lost history: When was it created? When shipped?
\`\`\`

**Event Sourcing**:
\`\`\`javascript
// Event log
[
    { eventType: "OrderCreated", timestamp: "10:00:00", data: {...} },
    { eventType: "PaymentReceived", timestamp: "10:00:05", data: {...} },
    { eventType: "OrderShipped", timestamp: "10:30:00", data: {...} }
]

// Reconstruct current state by replaying events
function getOrderState(orderId) {
    const events = getEvents(orderId);
    let state = {};
    
    for (const event of events) {
        state = applyEvent(state, event);
    }
    
    return state;
}

function applyEvent(state, event) {
    switch (event.eventType) {
        case 'OrderCreated':
            return { ...event.data, status: 'PENDING' };
        case 'PaymentReceived':
            return { ...state, status: 'PAID' };
        case 'OrderShipped':
            return { ...state, status: 'SHIPPED', shippedAt: event.timestamp };
        default:
            return state;
    }
}
\`\`\`

**Benefits**:
✅ Full audit trail
✅ Time travel (state at any point)
✅ Debugging (replay events)
✅ Can add new projections

**Drawbacks**:
❌ Complexity
❌ Query performance (must replay)
❌ Event versioning

### 4. CQRS (Command Query Responsibility Segregation)

**Separate** read and write models.

**Architecture**:
\`\`\`
Write Side (Commands):
  Order Service → Event Store

Events:
  OrderCreated, PaymentReceived, OrderShipped

Read Side (Queries):
  Event Handler → Read Database (denormalized views)
  
Query:
  API → Read Database (fast queries)
\`\`\`

**Implementation**:
\`\`\`javascript
// Write side (commands)
async function createOrder(orderData) {
    const order = { id: generateId(), ...orderData, status: 'PENDING' };
    
    // Store event
    await eventStore.append('OrderCreated', order);
    
    // Publish event
    await eventBus.publish('OrderCreated', order);
    
    return order;
}

// Read side (projections)
eventBus.on('OrderCreated', async (event) => {
    // Update read model
    await orderReadDB.insert({
        orderId: event.data.id,
        userId: event.data.userId,
        total: event.data.total,
        status: 'PENDING',
        createdAt: event.timestamp
    });
});

eventBus.on('OrderShipped', async (event) => {
    // Update read model
    await orderReadDB.update(event.data.orderId, {
        status: 'SHIPPED',
        shippedAt: event.timestamp
    });
});

// Query (fast!)
async function getOrdersByUser(userId) {
    return await orderReadDB.query({ userId });
}
\`\`\`

---

## Handling Failures

### Idempotency

**Problem**: Event processed twice

**Solution**: Idempotent event handlers

\`\`\`javascript
// ❌ Not idempotent
eventBus.on('OrderCreated', async (event) => {
    await database.query(
        'UPDATE inventory SET quantity = quantity - $1 WHERE productId = $2',
        [event.quantity, event.productId]
    );
});
// If event processed twice, quantity decremented twice!

// ✅ Idempotent
eventBus.on('OrderCreated', async (event) => {
    // Check if already processed
    const existing = await database.query(
        'SELECT * FROM processed_events WHERE eventId = $1',
        [event.eventId]
    );
    
    if (existing.length > 0) {
        return; // Already processed
    }
    
    // Process (in transaction)
    await database.transaction(async (tx) => {
        // Update inventory
        await tx.query(
            'UPDATE inventory SET quantity = quantity - $1 WHERE productId = $2',
            [event.quantity, event.productId]
        );
        
        // Mark as processed
        await tx.query(
            'INSERT INTO processed_events (eventId, processedAt) VALUES ($1, $2)',
            [event.eventId, new Date()]
        );
    });
});
\`\`\`

### Dead Letter Queue

**Problem**: Event processing fails repeatedly

**Solution**: Move to dead letter queue after N retries

\`\`\`javascript
eventBus.on('OrderCreated', async (event) => {
    try {
        await processOrderCreated(event);
    } catch (error) {
        const retryCount = event.retryCount || 0;
        
        if (retryCount < 3) {
            // Retry with exponential backoff
            const delay = Math.pow(2, retryCount) * 1000;
            setTimeout(() => {
                eventBus.publish('OrderCreated', {
                    ...event,
                    retryCount: retryCount + 1
                });
            }, delay);
        } else {
            // Move to dead letter queue
            await deadLetterQueue.send(event);
            await alerting.notify('Event processing failed after 3 retries', event);
        }
    }
});
\`\`\`

### Eventual Consistency

**Accept** that data is temporarily inconsistent.

**Example**:
\`\`\`
Time 10:00:00 - Order created
Time 10:00:01 - Email service receives event
Time 10:00:02 - Email sent

Between 10:00:00 and 10:00:02, user hasn't received email yet (eventually consistent)
\`\`\`

**How to handle**:
1. **Don't show** intermediate states to users
2. **Status fields**: "Processing...", "Pending..."
3. **Eventual consistency UI**: "Email will be sent shortly"

---

## Event-Driven vs Request-Response

| Aspect | Request-Response | Event-Driven |
|--------|-----------------|--------------|
| **Coupling** | Tight (caller knows callee) | Loose (publisher doesn't know subscribers) |
| **Failure** | Immediate failure | Eventual processing (resilient) |
| **Performance** | Blocking (synchronous) | Non-blocking (asynchronous) |
| **Consistency** | Immediate | Eventual |
| **Scalability** | Limited by slowest service | Highly scalable |
| **Debugging** | Easier (call stack) | Harder (distributed) |
| **Use case** | Real-time responses needed | Background processing, notifications |

**When to use each**:

**Request-Response**:
- User needs immediate response (login, search)
- Simple workflows
- Strong consistency required

**Event-Driven**:
- Background tasks (emails, analytics)
- Multiple services interested
- High scalability needed
- Loose coupling preferred

---

## Best Practices

1. **Events are immutable** (never change published event)
2. **Include all data** (event-carried state transfer)
3. **Version events** for schema evolution
4. **Idempotent handlers** (safe to process twice)
5. **Dead letter queues** for failed events
6. **Correlation IDs** for tracing
7. **Event store** for audit trail
8. **Monitor event lag** (time between publish and process)

---

## Interview Tips

**Red Flags**:
❌ Using events for everything (including synchronous operations)
❌ Not handling failures
❌ Ignoring eventual consistency

**Good Responses**:
✅ Explain events vs commands
✅ Discuss trade-offs (loose coupling vs complexity)
✅ Mention idempotency
✅ Talk about specific tools (Kafka, RabbitMQ)
✅ Discuss eventual consistency

**Sample Answer**:
*"For event-driven microservices, I'd use Kafka for high-throughput event streaming or RabbitMQ for simpler pub/sub. Services publish events (OrderCreated, PaymentCompleted) to a message broker, and interested services subscribe. This provides loose coupling - services don't know about each other. Trade-offs: eventual consistency (Order Service doesn't wait for Email Service), complexity (distributed debugging), need for idempotent handlers. I'd use events for background tasks (emails, analytics) and request-response for operations needing immediate feedback (login, search). Include correlation IDs for tracing, implement dead letter queues for failures, and monitor event processing lag."*

---

## Key Takeaways

1. **Events** announce something happened; **Commands** tell service to do something
2. **Message brokers**: Kafka (high throughput), RabbitMQ (flexible routing)
3. **Event-carried state transfer** includes all data in event
4. **Event sourcing** stores all changes as events (full audit trail)
5. **CQRS** separates read and write models
6. **Idempotency** makes handlers safe to execute multiple times
7. **Eventual consistency** is acceptable trade-off for loose coupling
8. **Use events** for background tasks, **request-response** for immediate feedback`,
      quiz: [
        {
          id: 'q1-events',
          question:
            'What is the difference between events and commands in microservices? When would you use each?',
          sampleAnswer:
            'Commands tell service to DO something (ChargePayment, ReserveInventory). Sent to specific service. Expect success/failure response. Synchronous. Events announce something HAPPENED (OrderCreated, PaymentCompleted). Published to anyone interested (pub/sub). No response expected. Asynchronous. Use commands for: operations needing immediate feedback (user clicks "checkout" → charge payment → show confirmation). Use events for: background tasks (send email, update analytics), multiple subscribers (OrderCreated → email service, analytics service, warehouse service), loose coupling (new services can subscribe without changing publisher). Example: Use command to charge payment (need to know if it succeeded), use event to notify email service (fire and forget).',
          keyPoints: [
            'Commands: tell service to do something, specific recipient, expect response',
            'Events: announce something happened, pub/sub, no response expected',
            'Commands: synchronous, immediate feedback needed',
            'Events: asynchronous, background tasks, multiple subscribers',
            'Trade-off: Commands=tight coupling but immediate, Events=loose coupling but eventual',
          ],
        },
        {
          id: 'q2-events',
          question:
            'What is idempotency in event-driven systems? Why is it important and how do you implement it?',
          sampleAnswer:
            'Idempotency: processing same event multiple times has same effect as processing once. Important because: message brokers may deliver event multiple times (at-least-once delivery), retries on failures, network issues. Without idempotency: inventory decremented twice, email sent twice, payment charged twice. Implementation: (1) Track processed events in database with event ID, (2) Before processing, check if event ID already processed, (3) Use database transaction to update state AND mark as processed atomically. Example: INSERT INTO processed_events (eventId) before decrementing inventory. If event processed twice, second time fails on unique constraint. Alternative: use database unique constraints (orderId+status) to make operations naturally idempotent.',
          keyPoints: [
            'Idempotency: same event processed multiple times = same result',
            'Important because: at-least-once delivery, retries, network issues',
            'Without it: double charges, duplicate emails, incorrect state',
            'Implementation: track processed event IDs in database',
            'Use transactions to update state AND mark as processed atomically',
          ],
        },
        {
          id: 'q3-events',
          question:
            'Compare Kafka vs RabbitMQ for event-driven microservices. When would you choose each?',
          sampleAnswer:
            'Kafka: Distributed commit log, high throughput (millions msg/sec), persistence (events stored permanently), partitioning for parallelism, event replay (consumers can replay from any offset), consumer groups. Best for: event streaming, event sourcing, high throughput, need event history. RabbitMQ: Traditional message queue, flexible routing (exchanges, bindings), multiple protocols (AMQP, MQTT), simpler, lower throughput. Best for: task queues, RPC, complex routing, need message acknowledgment. Choose Kafka for: analytics pipeline, event sourcing, need to replay events, very high scale. Choose RabbitMQ for: simpler requirements, need flexible routing, task distribution, request-reply pattern. Both support pub/sub, but Kafka better for streaming, RabbitMQ better for routing.',
          keyPoints: [
            'Kafka: distributed log, high throughput, persistence, replay events',
            'RabbitMQ: message queue, flexible routing, simpler, lower throughput',
            'Kafka best for: event streaming, event sourcing, high scale',
            'RabbitMQ best for: task queues, routing, RPC',
            'Choose based on: throughput needs, need for event replay, complexity tolerance',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-events-1',
          question:
            'What is the main advantage of event-driven architecture over request-response?',
          options: [
            "It's always faster",
            "Loose coupling - services don't need to know about each other",
            'It eliminates the need for databases',
            "It's easier to debug",
          ],
          correctAnswer: 1,
          explanation:
            'Event-driven architecture provides loose coupling: services publish events without knowing who subscribes. New subscribers can be added without changing publisher. Services can fail independently. Option 1 is wrong (not always faster - eventual consistency means delay). Option 3 is nonsense (still need databases). Option 4 is wrong (event-driven is actually harder to debug due to distributed nature). The main benefit is loose coupling and independent scaling.',
        },
        {
          id: 'mc-events-2',
          question: 'What is idempotency in event processing?',
          options: [
            'Processing events as fast as possible',
            'Processing the same event multiple times produces the same result',
            'Processing events in order',
            'Processing events without errors',
          ],
          correctAnswer: 1,
          explanation:
            "Idempotency means processing the same event multiple times has the same effect as processing it once. This is crucial because message brokers often deliver events at-least-once (may deliver multiple times). Without idempotency, you'd have duplicate charges, double decrements, etc. Implementation: track processed event IDs, check before processing, use transactions. Options 1, 3, and 4 describe different concepts (performance, ordering, reliability) but not idempotency.",
        },
        {
          id: 'mc-events-3',
          question: 'What is event sourcing?',
          options: [
            'Getting events from external sources',
            'Storing all changes as a sequence of events; event log is source of truth',
            'Finding the source of bugs in event handlers',
            'A way to compress events',
          ],
          correctAnswer: 1,
          explanation:
            'Event sourcing stores all state changes as a sequence of immutable events. Instead of storing current state in database (status="SHIPPED"), store events (OrderCreated, PaymentReceived, OrderShipped). Current state is derived by replaying events. Benefits: full audit trail, time travel debugging, can create new views by replaying events. Drawback: complexity, query performance. Options 1, 3, and 4 misunderstand the concept.',
        },
        {
          id: 'mc-events-4',
          question: 'What is a dead letter queue (DLQ)?',
          options: [
            'A queue for events from terminated services',
            'A queue for events that failed to process after multiple retries',
            'A queue for events that are no longer needed',
            'A backup queue',
          ],
          correctAnswer: 1,
          explanation:
            'Dead Letter Queue stores events that failed to process after multiple retries (e.g., 3 attempts with exponential backoff). This prevents poison messages from blocking the queue forever. Events in DLQ can be analyzed, fixed, and reprocessed manually. Alerts should fire when events land in DLQ. Option 1 confuses terminology. Option 3 is wrong (events aren\'t "deleted"). Option 4 is wrong (it\'s for failures, not backup).',
        },
        {
          id: 'mc-events-5',
          question:
            'When should you use event-driven architecture vs request-response?',
          options: [
            "Always use events (they're better)",
            'Events for background tasks/notifications; request-response when immediate feedback needed',
            'Always use request-response (events are too complex)',
            "They're interchangeable",
          ],
          correctAnswer: 1,
          explanation:
            'Use events for: background tasks (emails, analytics), multiple interested parties, loose coupling needed, eventual consistency acceptable. Use request-response for: operations needing immediate feedback (login, search, checkout confirmation), simple workflows, strong consistency required. Example: Charge payment with request-response (need to know if succeeded), notify email service with event (fire and forget). Options 1 and 3 are extreme positions. Option 4 is wrong (they have different characteristics and trade-offs).',
        },
      ],
    },
    {
      id: 'microservices-anti-patterns',
      title: 'Microservices Anti-Patterns',
      content: `Learning from mistakes is crucial. These anti-patterns represent common pitfalls when adopting microservices.

## 1. Distributed Monolith

**Problem**: Microservices that are tightly coupled, defeating the purpose.

**Symptoms**:
- Services must be deployed together
- Changes in one service break others
- Shared database
- Synchronous chains of calls
- No independent deployment

**Example**:
\`\`\`
Order Service → Payment Service → Fraud Service → Bank Service
(All must be deployed together, changes ripple through all services)
\`\`\`

**Why it's bad**: All complexity of microservices (distributed, network calls) without benefits (independent deployment, scaling).

**Solution**:
- Database per service
- Asynchronous communication
- Backward-compatible APIs
- Independent deployment pipelines

**Real example**: Teams split monolith by technical layers (UI layer service, business logic service, data access service) instead of business domains → distributed monolith.

---

## 2. Nanoservices

**Problem**: Services that are too small (excessive granularity).

**Symptoms**:
- Service has 1-2 endpoints
- Service called by only one other service
- Too much network overhead
- Debugging nightmare (request touches 20+ services)

**Example**:
\`\`\`
User Service
├─ GetUser Service
├─ UpdateUser Service
├─ DeleteUser Service
└─ CreateUser Service

Why? These should be ONE service!
\`\`\`

**Why it's bad**: Network latency, operational overhead (20+ services to deploy/monitor), no clear boundaries.

**Solution**:
- Follow business capabilities (DDD)
- Service should have cohesive responsibility
- Rule of thumb: 5-15 services for most systems (not 100+)

**Quote**: *"If services are too fine-grained, you'll end up with a distributed mess." - Sam Newman*

---

## 3. Shared Database

**Problem**: Multiple services accessing same database.

**Example**:
\`\`\`
Order Service  ─┐
User Service   ─┼─→ Shared Database
Product Service─┘
\`\`\`

**Why it's bad**:
- Tight coupling (schema change breaks multiple services)
- Can't scale independently
- Can't choose different database types
- No clear ownership
- Transaction boundaries unclear

**Solution**:
- Database per service
- Services communicate via APIs or events
- Data duplication is okay (trade-off for independence)

**Exception**: Migration from monolith (Strangler Fig pattern starts with shared DB)

---

## 4. Lack of API Versioning

**Problem**: Breaking API changes without versioning.

**Bad**:
\`\`\`javascript
// Version 1
POST /orders
{
    "userId": "123",
    "productId": "456"
}

// Version 2 (BREAKING CHANGE)
POST /orders
{
    "userId": "123",
    "items": [{ "productId": "456", "quantity": 1 }]  // Changed format!
}
\`\`\`

Old clients break when deployed.

**Solution**:
\`\`\`javascript
// Version 1 (still works)
POST /v1/orders

// Version 2 (new format)
POST /v2/orders

// OR use header versioning
POST /orders
Headers: Accept: application/vnd.api.v2+json
\`\`\`

**Best practice**: Support old version for 6-12 months while clients migrate.

---

## 5. Synchronous Coupling Chains

**Problem**: Long chains of synchronous calls.

**Example**:
\`\`\`
API Gateway → Order Service → Payment Service → Fraud Service → Bank API
              ↓
         Inventory Service → Warehouse API
              ↓
         Shipping Service → Carrier API

Total latency: 50ms + 100ms + 200ms + 150ms + 80ms + 120ms = 700ms
\`\`\`

**Why it's bad**:
- High latency (sum of all latencies)
- Low availability (product of all availabilities)
- Cascading failures

**Availability math**:
\`\`\`
If each service has 99.9% uptime:
6 services in chain = 0.999^6 = 99.4% uptime

43 minutes downtime/month → 4.3 hours downtime/month!
\`\`\`

**Solution**:
- Asynchronous communication where possible
- Circuit breakers
- Cache responses
- Parallelize calls

**Better**:
\`\`\`javascript
// Parallel calls
const [payment, inventory, shipping] = await Promise.all([
    paymentService.authorize(order),
    inventoryService.reserve(order.items),
    shippingService.calculate(order.address)
]);
\`\`\`

---

## 6. No Monitoring/Observability

**Problem**: Deploying microservices without proper monitoring.

**Symptoms**:
- Don't know which service is slow
- Can't trace requests across services
- No alerts when services fail
- Debugging takes hours

**Solution**:
- Metrics (Prometheus + Grafana)
- Distributed tracing (Jaeger)
- Centralized logging (ELK)
- Correlation IDs
- Health checks
- Dashboards

**Real story**: Company deployed 20 microservices, production issue took 8 hours to debug because no distributed tracing. After adding Jaeger, similar issues resolved in minutes.

---

## 7. Trying to Do ACID Transactions

**Problem**: Distributed transactions across services.

**Bad**:
\`\`\`javascript
BEGIN TRANSACTION;
  await orderService.createOrder(order);
  await paymentService.charge(payment);
  await inventoryService.reserve(items);
COMMIT;
\`\`\`

**Why it's bad**: Doesn't work! Each service has its own database. 2PC (Two-Phase Commit) is problematic in microservices.

**Solution**: Saga pattern with eventual consistency

\`\`\`javascript
// Choreography or Orchestration
1. Create order (local transaction)
2. Publish OrderCreated event
3. Payment service charges (local transaction)
4. Publish PaymentCompleted event
5. Inventory service reserves (local transaction)

// If payment fails, compensating transactions rollback
\`\`\`

---

## 8. Microservices First

**Problem**: Starting greenfield project with microservices.

**Why it's bad**:
- Don't know boundaries yet
- Over-engineering
- More complexity upfront
- Team may lack expertise

**Better**: Start with modular monolith

**Evolution**:
\`\`\`
1. Modular Monolith (clear module boundaries)
   ↓
2. Extract most painful service (e.g., report generation)
   ↓
3. Gradually extract more services (Strangler Fig)
   ↓
4. Mature microservices architecture
\`\`\`

**Quote**: *"Almost every successful microservices story has started with a monolith that became too big and was broken up." - Martin Fowler*

---

## 9. Ignoring Conway's Law

**Problem**: Microservices architecture doesn't match team structure.

**Conway's Law**: *"Organizations design systems that mirror their communication structure."*

**Bad**:
\`\`\`
Architecture: Order Service, Payment Service, Shipping Service
Team structure: Frontend Team, Backend Team, DBA Team

Result: Each service touched by all teams → coordination nightmare
\`\`\`

**Good**:
\`\`\`
Team 1: Owns Order Service (frontend, backend, database)
Team 2: Owns Payment Service (frontend, backend, database)
Team 3: Owns Shipping Service (frontend, backend, database)

Result: Teams can deploy independently
\`\`\`

**Solution**: Align team ownership with service boundaries

---

## 10. Hardcoded Config

**Problem**: Configuration in code or environment variables in Dockerfile.

**Bad**:
\`\`\`javascript
// Hardcoded
const PAYMENT_SERVICE_URL = 'http://payment-service:8080';

// Or in Dockerfile
ENV PAYMENT_SERVICE_URL=http://payment-service:8080
\`\`\`

**Why it's bad**: Can't change config without rebuilding image, different per environment.

**Solution**: ConfigMaps, Secrets, or config service

\`\`\`yaml
# Kubernetes ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: order-service-config
data:
  PAYMENT_SERVICE_URL: "http://payment-service"
  DATABASE_HOST: "postgres.prod.example.com"
\`\`\`

---

## 11. Smart Gateway, Dumb Services

**Problem**: Business logic in API Gateway.

**Bad**:
\`\`\`javascript
// API Gateway
app.post('/orders', async (req, res) => {
    // Business logic in gateway!
    const tax = calculateTax(req.body.items);
    const discount = applyPromotions(req.body.items);
    const total = calculateTotal(req.body.items, tax, discount);
    
    // Then call service
    await orderService.create({ ...req.body, total });
});
\`\`\`

**Why it's bad**: Logic duplicated if you add another gateway, hard to test, gateway becomes bottleneck.

**Solution**: Gateway only routes/authenticates

\`\`\`javascript
// API Gateway (routing only)
app.post('/orders', async (req, res) => {
    const result = await orderService.create(req.body);
    res.json(result);
});

// Order Service (business logic)
async function create(orderData) {
    const tax = calculateTax(orderData.items);
    const discount = applyPromotions(orderData.items);
    const total = calculateTotal(orderData.items, tax, discount);
    // ...
}
\`\`\`

---

## 12. Not Automating Deployment

**Problem**: Manual deployment of microservices.

**Why it's bad**: With 10+ services, manual deployment is error-prone, slow, doesn't scale.

**Solution**: CI/CD pipeline

\`\`\`yaml
# Example
git push → Run tests → Build container → Deploy to staging → Deploy to prod
\`\`\`

**Essential**:
- Automated tests
- Containerization (Docker)
- Orchestration (Kubernetes)
- Monitoring
- Rollback capability

---

## How to Avoid Anti-Patterns

**Checklist**:
- ✅ **Start with monolith** (or modular monolith)
- ✅ **Database per service** (no shared databases)
- ✅ **Clear service boundaries** (business capabilities, not technical layers)
- ✅ **API versioning** (backward compatibility)
- ✅ **Async communication** where possible
- ✅ **Comprehensive monitoring** (metrics, logs, traces)
- ✅ **Saga pattern** (not distributed transactions)
- ✅ **Team ownership** aligned with services (Conway's Law)
- ✅ **Automated CI/CD** (cannot do microservices manually)
- ✅ **Service mesh** for cross-cutting concerns (at scale)

---

## Interview Tips

**Red Flags**:
❌ Jumping straight to microservices
❌ Not mentioning trade-offs
❌ Ignoring operational complexity

**Good Responses**:
✅ Acknowledge anti-patterns
✅ Explain trade-offs (microservices aren't always better)
✅ Start with monolith recommendation
✅ Mention specific anti-patterns (distributed monolith, nanoservices)

**Sample Answer**:
*"Common microservices anti-patterns: (1) Distributed monolith - services tightly coupled, must deploy together, negating benefits, (2) Nanoservices - too granular, excessive network overhead, (3) Shared database - defeats purpose of independence, (4) Synchronous coupling chains - high latency and cascading failures, (5) No monitoring - can't debug distributed systems. To avoid: start with modular monolith, decompose by business domains (DDD), database per service, async communication, comprehensive observability, automated CI/CD. Microservices add complexity - only worth it at scale with mature team."*

---

## Key Takeaways

1. **Distributed monolith**: Worst of both worlds (complexity without benefits)
2. **Nanoservices**: Too small is as bad as too big
3. **Start with monolith**: Extract services gradually
4. **Database per service**: Mandatory for true independence
5. **Async communication**: Reduce coupling and cascading failures
6. **Monitor everything**: Can't operate microservices blind
7. **No ACID across services**: Use Saga pattern
8. **Align teams with services**: Conway's Law
9. **Automate deployment**: Essential at scale
10. **Microservices aren't always better**: Trade-offs matter`,
      quiz: [
        {
          id: 'q1-anti',
          question:
            'What is a distributed monolith? Why is it considered the worst of both worlds?',
          sampleAnswer:
            "Distributed monolith: microservices that are tightly coupled, must be deployed together, share database, or have synchronous call chains. Worst of both worlds because: (1) All complexity of microservices - distributed system, network calls, eventual consistency, multiple deployments, (2) None of benefits - can't deploy independently, can't scale independently, can't choose different tech stacks, changes ripple through all services. Example: Services that share database or must deploy together to avoid breaking changes. Better to have actual monolith (simpler) or proper microservices (independent). How to fix: database per service, async communication, API versioning, backward-compatible changes.",
          keyPoints: [
            'Distributed monolith: microservices that are tightly coupled',
            'All complexity of microservices (distributed, network)',
            'None of benefits (independent deployment, scaling)',
            'Often caused by: shared database, synchronous chains, no API versioning',
            'Fix: database per service, async communication, independence',
          ],
        },
        {
          id: 'q2-anti',
          question:
            'Why should you NOT start a greenfield project with microservices? What should you do instead?',
          sampleAnswer:
            "Don't start greenfield with microservices because: (1) Don't know correct boundaries yet - will get wrong initially, costly to refactor, (2) Over-engineering - premature optimization, (3) Team may lack expertise - microservices require mature operations, (4) Operational complexity - distributed systems are hard. Instead: Start with modular monolith - clear module boundaries within monolith, easy to split later. Then: Extract most painful part first (e.g., report generation), gradually extract more (Strangler Fig pattern), eventually reach mature microservices. Almost every successful microservices story started as monolith. Example: Amazon, Netflix, Uber all started as monoliths. Only split when monolith became too large.",
          keyPoints: [
            "Don't start greenfield with microservices (don't know boundaries)",
            'Start with modular monolith (clear boundaries, easy to split)',
            'Extract gradually when pain points emerge (Strangler Fig)',
            'Microservices require operational maturity',
            'Most successful microservices stories started as monoliths',
          ],
        },
        {
          id: 'q3-anti',
          question:
            "What is Conway's Law and why does it matter for microservices architecture?",
          sampleAnswer:
            'Conway\'s Law: "Organizations design systems that mirror their communication structure." Matters because: Architecture and team structure must align. Bad: Order Service, Payment Service, Shipping Service BUT teams organized as Frontend Team, Backend Team, DBA Team → every service touched by all teams → coordination nightmare, slow releases. Good: Team 1 owns Order Service end-to-end (frontend, backend, DB), Team 2 owns Payment Service end-to-end → teams can deploy independently, no coordination needed. Solution: Align team ownership with service boundaries. Each team should own their services completely. This enables independent deployment and velocity. Inverse Conway: Design architecture first, then structure teams to match.',
          keyPoints: [
            "Conway's Law: systems mirror communication structure",
            'Architecture and team structure must align',
            'Bad: functional teams (frontend/backend) touching all services',
            'Good: each team owns services end-to-end',
            'Enables independent deployment and fast iteration',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc-anti-1',
          question: 'What is a distributed monolith?',
          options: [
            'A monolith deployed on multiple servers',
            'Microservices that are tightly coupled and must be deployed together',
            'A large microservice',
            'A monolith with distributed caching',
          ],
          correctAnswer: 1,
          explanation:
            'Distributed monolith refers to microservices that are so tightly coupled they must be deployed together, negating microservices benefits. Symptoms: shared database, synchronous chains, breaking changes ripple across services. This is worst of both worlds - all complexity of microservices without benefits (independent deployment, scaling). Fix: database per service, async communication, API versioning. Option 1 is just horizontal scaling. Option 3 is a nanoservice concern. Option 4 is unrelated.',
        },
        {
          id: 'mc-anti-2',
          question:
            'Why should you NOT start a greenfield project with microservices?',
          options: [
            'Microservices are outdated',
            "You don't know the correct service boundaries yet; start with modular monolith",
            'Microservices are only for large companies',
            'Microservices are slower',
          ],
          correctAnswer: 1,
          explanation:
            "Don't start greenfield with microservices because you don't know correct boundaries yet. You'll guess wrong and face costly refactoring. Start with modular monolith (clear boundaries, easy to split later), extract services gradually when pain points emerge (Strangler Fig). Almost every successful microservices story (Amazon, Netflix, Uber) started as monolith. Option 1 is wrong (microservices are current). Option 3 is wrong (size matters but not the only factor). Option 4 is wrong (microservices can be faster at scale).",
        },
        {
          id: 'mc-anti-3',
          question:
            'What is the problem with long synchronous call chains in microservices?',
          options: [
            'They are too fast',
            'High latency (sum of all) and low availability (product of all)',
            'They use too much memory',
            'They are easy to debug',
          ],
          correctAnswer: 1,
          explanation:
            'Long synchronous chains: A → B → C → D have high latency (50ms + 100ms + 150ms = 300ms) and low availability (99.9% × 99.9% × 99.9% = 99.7%). One service failure breaks entire chain. Solution: async communication where possible, parallelize calls, circuit breakers, caching. Example: 6 services with 99.9% uptime each = 99.4% combined (43 min/month downtime → 4.3 hrs/month). Option 1 is wrong (slower, not faster). Option 3 is not the main issue. Option 4 is wrong (hard to debug, not easy).',
        },
        {
          id: 'mc-anti-4',
          question: "What is Conway's Law?",
          options: [
            'A law about API design',
            'Organizations design systems that mirror their communication structure',
            'A law about database schemas',
            'A law about deployment frequency',
          ],
          correctAnswer: 1,
          explanation:
            "Conway's Law: \"Organizations design systems that mirror their communication structure.\" For microservices: team structure must align with architecture. Each team should own their services end-to-end (frontend, backend, DB) to enable independent deployment. Don't have functional teams (Frontend Team, Backend Team) each touching all services (coordination nightmare). Options 1, 3, and 4 are unrelated concepts. Conway's Law is about team structure affecting system design.",
        },
        {
          id: 'mc-anti-5',
          question:
            'Why is a shared database an anti-pattern in microservices?',
          options: [
            'Databases are expensive',
            "It creates tight coupling; services can't evolve or scale independently",
            "It's too slow",
            "It's a security risk",
          ],
          correctAnswer: 1,
          explanation:
            "Shared database creates tight coupling: schema change breaks multiple services, can't scale databases independently, can't choose different database types (SQL vs NoSQL), unclear ownership, transaction boundaries unclear. Defeats the purpose of microservices (independence). Solution: database per service pattern. Services communicate via APIs or events. Data duplication is acceptable trade-off for independence. Options 1, 3, and 4 may be concerns but aren't the main reason shared database is an anti-pattern.",
        },
      ],
    },
  ],
  keyTakeaways: [
    'Microservices vs Monolith: Choose based on team size, scale, and operational maturity',
    'Service Decomposition: Follow domain boundaries (DDD), not technical layers',
    'Inter-Service Communication: Balance sync (simple) vs async (resilient)',
    'Data Management: Database per service to enable true independence',
    'Circuit Breakers: Prevent cascading failures in distributed systems',
    'Service Mesh: Provides observability, security, and traffic management',
    'Saga Pattern: Handle distributed transactions with eventual consistency',
    'Start Simple: Monolith first, extract services strategically using Strangler Fig',
  ],
  learningObjectives: [
    'Understand trade-offs between monolithic and microservices architectures',
    'Master service decomposition strategies using business capabilities and DDD',
    'Design effective inter-service communication patterns',
    'Implement resilience patterns (circuit breakers, retries, timeouts)',
    'Handle data consistency in distributed systems using Saga pattern',
    'Build observable microservices with distributed tracing',
    'Design for failure with fault isolation and graceful degradation',
    'Apply microservices patterns to real-world system design problems',
  ],
};
