/**
 * Uber Architecture Section
 */

export const uberarchitectureSection = {
  id: 'uber-architecture',
  title: 'Uber Architecture',
  content: `Uber is a global ride-hailing platform connecting riders with drivers in real-time across 10,000+ cities worldwide. With millions of rides per day, Uber's architecture must handle complex challenges including real-time location tracking, dynamic pricing, efficient matching, and high availability. This section explores the technical systems that power Uber's platform.

## Overview

Uber's architecture handles massive scale and unique challenges:
- **150+ million active users** worldwide
- **25+ million trips** per day
- **6 million drivers** globally
- **Real-time operations**: Location updates every 4 seconds
- **Microservices**: 2,000+ services
- **Multi-region deployment**: 50+ geographic regions

### Key Challenges

1. **Real-time**: Matching riders to drivers in seconds
2. **Geospatial**: Efficient queries for nearby drivers
3. **Scalability**: Peak demand (Friday nights, bad weather)
4. **Reliability**: High availability (downtime = lost revenue + angry users)
5. **Global**: Different regulations, currencies, payment methods per region

### Architectural Principles

1. **Microservices**: Independent, deployable services
2. **Event-driven**: Asynchronous communication via Kafka
3. **Regional isolation**: Each region can operate independently
4. **Horizontal scalability**: Add capacity by adding instances
5. **Observability**: Comprehensive monitoring and tracing

---

## High-Level Architecture

\`\`\`
┌────────────┐
│   Mobile   │ (Rider/Driver Apps)
│   Clients  │
└─────┬──────┘
      │ HTTPS / WebSocket
      ▼
┌─────────────────────────────────────────────────────┐
│            API Gateway / Edge Services              │
│    (Authentication, Rate Limiting, Routing)         │
└─────────────────┬───────────────────────────────────┘
                  │
      ┌───────────┴───────────┐
      │                       │
      ▼                       ▼
┌──────────────┐    ┌──────────────────┐
│  Rider       │    │  Driver          │
│  Services    │    │  Services        │
└──────────────┘    └──────────────────┘
      │                       │
      └───────────┬───────────┘
                  ▼
    ┌──────────────────────────┐
    │   Matching Engine        │
    │  (DISCO - Dispatch       │
    │   Optimization)          │
    └──────────────────────────┘
                  │
      ┌───────────┴───────────┐
      │                       │
      ▼                       ▼
┌──────────────┐    ┌──────────────────┐
│  Pricing     │    │  Geospatial      │
│  Service     │    │  Index           │
│  (Surge)     │    │  (Nearest        │
│              │    │   Drivers)       │
└──────────────┘    └──────────────────┘
      │                       │
      └───────────┬───────────┘
                  ▼
    ┌──────────────────────────┐
    │     Payment Service      │
    └──────────────────────────┘
                  │
                  ▼
    ┌──────────────────────────┐
    │    Data Stores           │
    │  - PostgreSQL            │
    │  - MySQL (Schemaless)    │
    │  - Redis                 │
    │  - Cassandra             │
    └──────────────────────────┘
\`\`\`

---

## Core Components

### 1. Real-Time Location Tracking

Uber tracks millions of driver locations in real-time to match with nearby riders.

**Requirements**:
- Drivers send location every 4 seconds
- Low latency (<100ms to process update)
- Store recent location history (last hour)
- Query nearby drivers efficiently

**Architecture**:

**Location Service**:

1. **Ingestion**:
   - Driver app sends GPS coordinates (lat, lon)
   - WebSocket connection for low latency
   - Protobuf for compact message format

2. **Storage**:
   - **Redis** for current location (in-memory, fast)
   - **Time-series DB** for history (InfluxDB/Cassandra)

**Data Model (Redis)**:
\`\`\`
Key: driver:location:driver123
Value: {lat: 37.7749, lon: -122.4194, timestamp: 1698158400}
TTL: 60 seconds (if no update, driver considered offline)
\`\`\`

**Geospatial Indexing**:
- Problem: Find drivers within 5km of rider
- Naive approach: Calculate distance to all drivers (slow!)
- Solution: **Geospatial index** (covered next section)

**Location Update Flow**:
\`\`\`
Driver App → WebSocket → Location Service
                              ↓
                         Update Redis (current location)
                              ↓
                         Update Geospatial Index
                              ↓
                         Publish event to Kafka (for analytics)
\`\`\`

**Scale**:
- 25 million location updates per minute
- Sub-100ms latency requirement
- Multi-region deployment (data locality)

---

### 2. Geospatial Indexing (H3)

Finding nearby drivers is critical for ride matching. Uber uses **H3**, a hexagonal hierarchical geospatial indexing system developed by Uber.

**Problem with Naive Approach**:
- 1 million drivers online in a city
- Rider requests ride → Calculate distance to all 1M drivers → Too slow!

**Geospatial Index Solutions**:

**Traditional Approaches**:
1. **Grid-based**: Divide world into grid cells, query nearby cells
2. **Quadtree**: Hierarchical tree structure, recursively divide space
3. **Geohashing**: Encode location as hash, similar hashes = nearby

**Uber's Solution: H3 (Hexagonal Hierarchical Index)**

**Why Hexagons?**:
- Equal distance from center to all edges (unlike squares)
- Better approximation of circular range
- No distortion near poles

**How H3 Works**:

1. **Divide Earth into hexagons** at multiple resolutions:
   - Resolution 0: 122 hexagons (very large)
   - Resolution 5: ~2.5 million hexagons (~1 km² each)
   - Resolution 10: ~600 billion hexagons (~1 m² each)

2. **Each location gets hex ID** at each resolution
3. **Hierarchical**: Each hex contains smaller hexes at finer resolution

**Finding Nearby Drivers**:

\`\`\`
Rider location: (37.7749, -122.4194)

1. Convert to H3 hex ID at resolution 8 (hex_A)
2. Get k-ring (neighboring hexes): hex_A, hex_B, hex_C, ... (19 hexes for k=2)
3. Query drivers in these hexes (Redis: O(19) lookups)
4. Calculate exact distance to drivers in these hexes
5. Return closest drivers

Instead of checking 1M drivers, check ~100-500 drivers in nearby hexes!
\`\`\`

**Index Structure (Redis)**:
\`\`\`
Key: hex:8a2a1072b59ffff (hex ID at resolution 8)
Value: [driver123, driver456, driver789]
\`\`\`

**Benefits**:
- **Fast queries**: O(k) hex lookups instead of O(n) driver checks
- **Scalable**: Works globally, handles millions of drivers
- **Flexible**: Adjust resolution based on density (finer in cities, coarser in rural areas)

**Open Source**: Uber open-sourced H3 (h3geo.org), used by other companies.

---

### 3. Matching Engine (DISCO - Dispatch Optimization)

DISCO is Uber's dispatch system that matches riders to drivers optimally.

**Goals**:
- **Fast**: Match within 5 seconds
- **Efficient**: Minimize driver pickup time
- **Fair**: Balance earnings across drivers
- **Global optimization**: Consider entire system, not just individual ride

**Matching Algorithm**:

**Simple Approach (Greedy)**:
- Find nearest available driver → Assign ride
- Problems:
  - Suboptimal globally (driver might be better for different ride)
  - Unfair (drivers near busy areas get more rides)

**Uber's Approach (Batch Matching with Optimization)**:

**Every 5 seconds**:

1. **Collect pending requests**:
   - Riders who requested in last 5 seconds
   - Available drivers in area

2. **Formulate as optimization problem**:
   - Minimize total pickup time + wait time
   - Subject to constraints (driver availability, capacity)
   - Consider future demand (don't assign driver if better ride likely coming)

3. **Solve optimization**:
   - Integer linear programming (ILP) or greedy heuristics
   - Balance speed (5 sec deadline) vs optimality

4. **Assign rides**:
   - Send dispatch to drivers
   - Update driver status to "on trip"

**Example**:
\`\`\`
Riders: [R1, R2, R3]
Drivers: [D1, D2, D3]

Distances:
  R1: D1=1km, D2=2km, D3=5km
  R2: D1=3km, D2=1km, D3=2km
  R3: D1=4km, D2=3km, D3=1km

Greedy: R1→D1, R2→D2, R3→D3 (Total: 1+1+1=3km)
Optimal: R1→D2, R2→D3, R3→D1 (Total: 2+2+4=8km, worse!)

But if R4 arrives needing D3 urgently:
Optimal considering R4: R1→D1, R2→D2, R3→D3, R4→? (no driver!)
Better: Hold D3 for R4

Uber's algorithm considers future demand probabilities!
\`\`\`

**Advanced Features**:

**1. Pooling (UberPool/UberX Share)**:
- Match multiple riders going similar direction to same driver
- More complex: Route optimization, time window constraints
- Trade-off: Lower cost vs longer trip time

**2. Destination Prediction**:
- Predict where rider wants to go (before they enter it)
- Based on: Time of day, day of week, pickup location, historical trips
- Pre-match driver heading that direction

**3. Driver Repositioning**:
- Suggest drivers move to high-demand areas
- Balance supply and demand across city

---

### 4. Dynamic Pricing (Surge Pricing)

Surge pricing increases prices during high demand to incentivize more drivers.

**Goals**:
- **Balance supply and demand**: Higher price → More drivers come online
- **Reduce wait times**: Ensure enough drivers available
- **Revenue optimization**: Maximize marketplace efficiency

**How Surge Works**:

1. **Demand Forecasting**:
   - Predict ride demand in each area for next 15-30 minutes
   - Based on: Historical data, events, weather, time of day

2. **Supply Tracking**:
   - Count available drivers in each area
   - Consider drivers on trip (will be available soon)

3. **Calculate Surge Multiplier**:
   - If demand > supply → Surge
   - Multiplier: 1.2x, 1.5x, 2x, up to 5x+ (rare)
   - Formula considers demand/supply ratio, rider sensitivity (elasticity)

\`\`\`
demand_ratio = predicted_demand / available_supply

if demand_ratio > 1.5:
    surge_multiplier = min(1 + (demand_ratio - 1) * 0.5, 5.0)
else:
    surge_multiplier = 1.0

Notify riders: "Fares are slightly higher due to increased demand"
\`\`\`

4. **Update Prices Dynamically**:
   - Recalculate every 1-5 minutes
   - Display to riders before requesting
   - Drivers see surge heat map (encourages movement to high-surge areas)

**Data Model**:
\`\`\`
Table: surge_pricing
- zone_id (hexagon or neighborhood)
- timestamp
- surge_multiplier
- demand_count
- supply_count
\`\`\`

**Challenges**:

**1. Surge Avoidance**:
- Riders wait for surge to end
- Solution: Show estimated end time, offer fixed upfront price

**2. Fairness**:
- Riders complain about "price gouging"
- Solution: Cap surge at 5x, explain rationale (more drivers online)

**3. Prediction Accuracy**:
- False surge → Riders don't request, demand drops
- Solution: Machine learning models, feedback loops

---

### 5. ETA Calculation

Accurate ETA (Estimated Time of Arrival) is critical for user experience.

**ETA Types**:

**1. Driver Arrival ETA**:
- How long until driver picks up rider?
- Shown before requesting ride

**2. Trip ETA**:
- How long until rider reaches destination?
- Shown during trip

**Challenges**:
- Traffic conditions (rush hour, accidents)
- Road types (highway vs city streets)
- Driver behavior (speed, route choice)
- Special events (concerts, sports games)

**Uber's Approach**:

**Routing Engine**:
- Use road network graph (nodes = intersections, edges = roads)
- Apply routing algorithms (A*, Dijkstra with optimizations)
- Incorporate real-time traffic data

**Traffic Data Sources**:
1. **Uber's own data**: Historical trips, current driver speeds
2. **Third-party**: Google Maps API, HERE, TomTom
3. **Live events**: Accidents reported by drivers or news

**ETA Prediction Model**:

Traditional approach:
\`\`\`
ETA = distance / average_speed
\`\`\`

Uber's approach (Machine Learning):
\`\`\`
Features:
- Distance (straight-line and route)
- Time of day, day of week
- Historical speed on route segments
- Weather conditions
- Live traffic data
- Driver characteristics (aggressive vs cautious)

Model: Gradient Boosted Trees (XGBoost)
Output: Predicted travel time (seconds)
\`\`\`

**Accuracy**:
- Target: 90% of ETAs within ±2 minutes
- Continuous improvement via A/B testing

**Real-Time Updates**:
- Recalculate ETA every 30 seconds during trip
- Update rider and recipient (if shared)

---

### 6. Payment Processing

Uber processes millions of payments daily across multiple payment methods and currencies.

**Payment Methods**:
- Credit/debit cards
- PayPal, Venmo
- Cash (in some regions)
- In-app wallets (Uber Cash)
- Split payments (multiple riders)

**Payment Flow**:

1. **Trip Completion**:
   - Driver marks trip as complete
   - System calculates fare (base + distance + time + surge)

2. **Charge Customer**:
   - Retrieve payment method
   - Call payment gateway (Braintree, Stripe, Adyen)
   - Handle retries if payment fails

3. **Payout Driver**:
   - Driver earns ~75-80% of fare
   - Payouts aggregated weekly
   - Transfer to driver's bank account

**Challenges**:

**1. Fraud Prevention**:
- Stolen credit cards
- Solution: Machine learning fraud detection, 3D Secure, CVV verification

**2. Currency Conversion**:
- Rider pays in USD, driver gets paid in INR
- Solution: Real-time exchange rates, minimize forex fees

**3. Failed Payments**:
- Card declined, insufficient funds
- Solution: Retry with exponential backoff, email notification, lock account until resolved

**Data Consistency**:
- Ensure exactly-once payment (no double charging)
- Idempotency keys for payment API calls
- Distributed transactions (saga pattern)

**Architecture**:
\`\`\`
Trip Completion → Payment Service
                       ↓
                  Validate fare calculation
                       ↓
                  Check fraud score
                       ↓
                  Call payment gateway (Stripe, Braintree)
                       ↓
                  Record transaction in DB
                       ↓
                  Publish event (trip_paid)
                       ↓
                  Payout Service (handle driver payout)
\`\`\`

---

## Data Storage Layer

Uber uses polyglot persistence for different data needs.

### 1. MySQL (Schemaless)

Uber initially used PostgreSQL but built **Schemaless**, a sharding layer on top of MySQL.

**Why Schemaless?**:
- Application-level sharding (horizontal scalability)
- Schema flexibility (JSON documents in BLOB column)
- Migration away from PostgreSQL for operational reasons

**Data Model**:
\`\`\`sql
Table: trips
- trip_id (shard key)
- data (JSON blob)
- updated_at
\`\`\`

**Sharding Strategy**:
- Hash trip_id to determine shard
- 100s of MySQL shards
- Each shard is a MySQL instance

**Challenges**:
- No joins across shards
- Foreign key constraints don't work globally
- Rebalancing shards (add capacity)

**Use Cases**:
- Trips (ride details)
- Users (rider/driver profiles)
- Payments (transaction records)

---

### 2. Redis

Used for caching and real-time data.

**Use Cases**:

**1. Session Storage**:
- Active sessions (logged-in users)
- Token validation

**2. Location Cache**:
- Current driver locations
- TTL of 60 seconds (stale locations auto-expire)

**3. Geospatial Index**:
- H3 hex → Driver list
- Supports GEORADIUS queries

**4. Rate Limiting**:
- Limit API requests per user
- Token bucket algorithm

**5. Real-Time Features**:
- Driver availability status
- Trip state (in-progress, completed)

**Deployment**:
- Redis Cluster for high availability
- Multi-region replication

---

### 3. Cassandra

Used for high-throughput, append-heavy workloads.

**Use Cases**:

**1. Event Logs**:
- Trip events (requested, accepted, started, completed)
- Location history
- Payment events

**2. Metrics and Analytics**:
- Driver performance metrics
- Rider behavior analytics

**Data Model**:
\`\`\`
Table: trip_events
Partition Key: trip_id
Clustering Key: timestamp
Columns: event_type, data
\`\`\`

**Why Cassandra?**:
- High write throughput (millions of events/second)
- Time-series data (append-only, time-ordered)
- Scalable (linear scalability)

**Consistency**:
- Quorum writes (W=2, N=3)
- Accept eventual consistency

---

### 4. Kafka (Event Streaming)

Kafka is Uber's central nervous system for event-driven architecture.

**Use Cases**:

**1. Event Sourcing**:
- All state changes published as events
- Example: trip_requested, trip_accepted, trip_started, trip_completed

**2. Data Pipelines**:
- Stream events to analytics systems (Hadoop, Spark)
- Real-time dashboards (driver performance, system health)

**3. Service Communication**:
- Asynchronous communication between microservices
- Decouple producers and consumers

**Example Flow**:
\`\`\`
Trip completed → Payment Service publishes "trip_paid" event to Kafka
                       ↓
    ┌──────────────────┼──────────────────┐
    ▼                  ▼                  ▼
Payout Service   Analytics Pipeline   Notification Service
(pay driver)     (update metrics)     (send receipt)
\`\`\`

**Scale**:
- 1 trillion messages per day
- 100+ petabytes of data
- Multi-region clusters

---

## Microservices Architecture

Uber has 2,000+ microservices, organized by domain.

**Service Examples**:

**1. Rider Services**:
- Trip management (request, cancel)
- Fare estimation
- Ride history

**2. Driver Services**:
- Driver onboarding
- Availability management
- Earnings tracking

**3. Matching Services**:
- DISCO (dispatch optimization)
- ETA calculation
- Route optimization

**4. Pricing Services**:
- Surge pricing
- Promotions and discounts
- Fare calculation

**5. Payment Services**:
- Payment processing
- Refunds
- Driver payouts

**Communication**:
- **Synchronous**: gRPC for low-latency, request-response
- **Asynchronous**: Kafka for event-driven, decoupled communication

**Service Mesh (Ringpop)**:
- Uber built Ringpop for service discovery and consistent hashing
- Nodes form a ring, use gossip protocol for membership
- Application-level sharding (route requests to correct instance)

---

## Regional Architecture

Uber operates in 10,000+ cities across 50+ geographic regions.

**Regional Isolation**:
- Each region is self-contained
- Can operate independently if connection to other regions lost
- Reduces latency (data locality)

**Regional Components**:
- API Gateway
- Microservices (deployed per region)
- Databases (sharded by region)
- Kafka (regional clusters)

**Cross-Region**:
- Aggregate analytics (send data to central data warehouse)
- Global configuration (rollout features globally)
- User roaming (rider travels to different region)

**Data Residency**:
- GDPR and local regulations require data stored locally
- EU region keeps EU user data in EU

---

## Observability and Monitoring

Uber's distributed system requires comprehensive observability.

**Metrics**:
- System metrics (CPU, memory, network)
- Business metrics (rides/second, revenue, driver utilization)
- SLIs (availability, latency, error rate)

**Distributed Tracing**:
- Jaeger for tracing requests across microservices
- Trace ID propagated through all service calls
- Identify bottlenecks and latency issues

**Logging**:
- Centralized logging (ELK stack)
- Structured logs (JSON format)
- Correlation via trace ID

**Alerting**:
- Threshold-based (error rate >1%)
- Anomaly detection (ML models detect unusual patterns)
- On-call rotation for incident response

---

## Key Lessons

### 1. Geospatial Indexing is Critical

Efficient queries for nearby drivers enabled by H3. Don't underestimate the importance of spatial data structures.

### 2. Real-Time Requires Optimization

Location updates every 4 seconds, matching within 5 seconds. Use in-memory caches (Redis), geospatial indexes, and optimized algorithms.

### 3. Global Optimization Over Local

Matching isn't just finding nearest driver—it's optimizing entire system considering future demand.

### 4. Dynamic Pricing Balances Marketplace

Surge pricing is controversial but effective at balancing supply and demand, reducing wait times.

### 5. Polyglot Persistence

Different data stores for different needs: MySQL for transactional, Redis for caching, Cassandra for events, Kafka for streaming.

### 6. Regional Isolation for Scale

Operating globally requires regional deployment, data locality, and independent operation.

---

## Interview Tips

**Q: How would you find the nearest drivers to a rider?**

A: Use geospatial indexing. Convert rider location to H3 hexagon ID at resolution 8 (~1 km²). Query neighboring hexagons (k-ring, e.g., k=2 gives 19 hexagons). Retrieve drivers in these hexagons from Redis (O(19) lookups). Calculate exact distance to these drivers (~100-500 drivers typically). Return closest N drivers sorted by distance. This avoids checking all 1M drivers in the city. H3 is preferred over simple grid or geohashing because hexagons have equal distance from center to edges, better approximating circular range.

**Q: How does Uber handle surge pricing?**

A: Surge pricing dynamically adjusts prices based on supply/demand ratio. Every 1-5 minutes: (1) Predict demand in each zone (H3 hexagons) for next 15-30 minutes using historical data, events, weather. (2) Count available drivers in each zone. (3) Calculate demand_ratio = predicted_demand / available_supply. (4) If demand_ratio >1.5, apply surge multiplier: surge = min(1 + (demand_ratio - 1) * 0.5, 5.0). (5) Display to riders before request, show to drivers as heat map (encourages repositioning). Store in database (zone_id, timestamp, surge_multiplier). Recalculate frequently to adapt to changing conditions.

**Q: How would you design Uber's matching system?**

A: Use batch matching with optimization. Every 5 seconds: (1) Collect pending ride requests and available drivers. (2) Formulate as optimization problem: minimize total pickup time + wait time, subject to driver availability constraints. (3) Consider future demand probabilities (don't assign driver if better match likely soon). (4) Solve using integer linear programming or greedy heuristics (balance 5-second deadline vs optimality). (5) Assign rides to drivers. (6) Send dispatch via push notification. Handle edge cases: driver rejects (reassign), rider cancels (release driver), pooling (route optimization). Monitor metrics: average wait time, pickup time, driver utilization.

---

## Summary

Uber's architecture demonstrates building a real-time, geospatial marketplace at global scale:

**Key Takeaways**:

1. **H3 geospatial indexing**: Efficient nearest-driver queries using hexagonal grid
2. **Real-time location tracking**: WebSocket, Redis, 4-second update frequency
3. **Matching optimization**: Batch matching with global optimization (DISCO)
4. **Dynamic pricing**: Surge pricing balances supply and demand
5. **Microservices**: 2,000+ services communicating via gRPC and Kafka
6. **Polyglot persistence**: MySQL (Schemaless), Redis, Cassandra, Kafka
7. **Regional isolation**: Independent regions for scale and compliance
8. **Observability**: Comprehensive metrics, tracing, and logging

Uber's architecture showcases the importance of geospatial algorithms, real-time systems, and marketplace optimization for a global ridesharing platform.
`,
};
