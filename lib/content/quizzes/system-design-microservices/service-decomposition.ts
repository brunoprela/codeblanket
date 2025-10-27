/**
 * Quiz questions for Service Decomposition Strategies section
 */

export const servicedecompositionQuiz = [
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
    sampleAnswer: `Domain-Driven Design\'s "bounded context" is one of the most important concepts for microservices decomposition. Let me explain with a concrete example.

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

**Process:**1. **Identify Domains**: What business capabilities exist? (Scheduling, Clinical, Billing, Emergency)

2. **Find Ubiquitous Language**: What terms does business use? (Patient, Appointment, Diagnosis, Invoice)

3. **Discover Multiple Meanings**: Does "Patient" mean same thing everywhere? (No!)

4. **Draw Boundaries**: Where does meaning change? → That\'s a bounded context boundary

5. **Map to Services**: Each bounded context → One microservice

**Interview Approach:**

When asked to design a complex system:

1. "I'll use Domain-Driven Design to identify bounded contexts"
2. "The term 'Customer' likely means different things in different parts of the system"
3. "In the Sales context, Customer is a lead. In Support context, Customer is a ticket owner. In Billing context, Customer is a payer."
4. "Each bounded context becomes a microservice with its own data model"
5. "Services communicate via events or API calls, with eventual consistency"

**Key Takeaways:**1. Bounded contexts define where domain models are consistent
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
- What\'s slow to deploy?
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

**Phase 1: Preparation (Week 1)**1. **Build platform infrastructure:**
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

**Phase 2: Parallel Run (Weeks 2-3)**1. **Implement Notification Service:**
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

**Phase 3: Cutover (Week 4)**1. **Feature flag to route % of traffic:**
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

**Approach:**1. Set up Elasticsearch
2. Index product data
3. Keep SQL search as fallback
4. Route 10% of searches to Elasticsearch
5. Gradually increase to 100%
6. Remove SQL search

**Decision Framework Summary:**

**Extract First When Service Has:**1. ✅ Low coupling (few dependencies)
2. ✅ Clear boundary (well-defined interface)
3. ✅ Low risk (failure doesn't break core business)
4. ✅ Async friendly (or read-only)
5. ✅ Quick extraction (2-4 weeks)
6. ✅ Learning value (teaches patterns for next services)

**Don't Extract First When Service:**1. ❌ Highly coupled (touches everything)
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
];
