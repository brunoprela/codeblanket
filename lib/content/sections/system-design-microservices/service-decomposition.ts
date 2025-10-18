/**
 * Service Decomposition Strategies Section
 */

export const servicedecompositionSection = {
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
};
