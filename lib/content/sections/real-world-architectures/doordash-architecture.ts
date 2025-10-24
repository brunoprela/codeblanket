/**
 * DoorDash Architecture Section
 */

export const doordasharchitectureSection = {
  id: 'doordash-architecture',
  title: 'DoorDash Architecture',
  content: `DoorDash is the largest food delivery platform in the United States with 30+ million customers and 500,000+ merchants. DoorDash's architecture must handle real-time logistics, dynamic dispatch, location tracking, and marketplace optimization at massive scale. This section explores the technical systems powering DoorDash's three-sided marketplace.

## Overview

DoorDash's scale and challenges:
- **30+ million customers** (consumers ordering food)
- **500,000+ merchants** (restaurants)
- **3+ million Dashers** (delivery drivers)
- **1+ billion deliveries** annually
- **Real-time operations**: Location tracking, dispatch, ETA prediction
- **Logistics optimization**: Route planning, driver assignment

### Key Architectural Challenges

1. **Real-time dispatch**: Match orders to optimal Dasher within seconds
2. **Location tracking**: Track millions of Dashers in real-time
3. **ETA prediction**: Accurate delivery time estimates
4. **Dynamic pricing**: Surge pricing, promotions, Dasher pay
5. **Marketplace optimization**: Balance supply (Dashers) and demand (orders)
6. **Scale**: Handle dinner rush (5-9 PM), millions of concurrent users

---

## High-Level Architecture

\`\`\`
┌─────────────────┐
│   Consumer App  │ (Place Order)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   API Gateway                       │
│   (Authentication, Rate Limiting)   │
└────────┬────────────────────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────────┐
│ Order  │ │   Dispatch   │
│Service │ │   Service    │
│        │ │  (Matching)  │
└────┬───┘ └──────┬───────┘
     │            │
     └─────┬──────┘
           ▼
┌─────────────────────────┐
│   Dasher Service        │
│   (Location, Status)    │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Location Service      │
│   (Real-time Tracking)  │
└─────────────────────────┘
\`\`\`

---

## Core Components

### 1. Order Placement and Lifecycle

**Order Flow**:

\`\`\`
1. Customer browses restaurants (Search/Discovery Service)
2. Customer adds items to cart
3. Customer places order (Order Service)
4. Payment captured (Payment Service via Stripe)
5. Order sent to restaurant (Merchant App via push notification)
6. Restaurant confirms order (prep time: 15 minutes)
7. Dispatch algorithm finds optimal Dasher (Dispatch Service)
8. Dasher accepts delivery
9. Dasher arrives at restaurant (GPS tracking)
10. Restaurant hands off food
11. Dasher delivers to customer
12. Order completed, payment released to restaurant and Dasher
\`\`\`

**Order States**:
\`\`\`
created → confirmed (restaurant) → dasher_assigned → dasher_at_restaurant
→ picked_up → dasher_en_route → delivered → completed

Or:
created → canceled (customer/restaurant/dasher cancels)
\`\`\`

**Data Model** (PostgreSQL):

\`\`\`sql
Table: orders
- order_id (primary key, UUID)
- customer_id (FK to users)
- merchant_id (FK to merchants)
- dasher_id (nullable, FK to dashers)
- status (enum: created, confirmed, dispatched, picked_up, delivered, completed, canceled)
- items (JSON array)
  - item_id, name, quantity, price, customizations
- subtotal (decimal)
- delivery_fee (decimal)
- service_fee (decimal)
- tip (decimal)
- total_amount (decimal)
- delivery_address (JSON: street, city, zip, lat, lon)
- restaurant_address (JSON)
- customer_phone (encrypted)
- special_instructions (text)
- scheduled_delivery_time (nullable timestamp)
- estimated_prep_time (minutes)
- estimated_delivery_time (timestamp)
- actual_delivery_time (nullable timestamp)
- created_at (timestamp)
- updated_at (timestamp)

Table: order_events
- event_id
- order_id (FK)
- event_type (created, confirmed, dispatched, picked_up, delivered, etc.)
- metadata (JSON)
- timestamp

Index: (order_id, timestamp) for order history
\`\`\`

---

### 2. Real-Time Location Tracking

DoorDash tracks millions of Dashers in real-time to dispatch efficiently.

**Location Updates**:

\`\`\`
Dasher app:
- Send GPS coordinates every 4 seconds (while on delivery)
- Send via WebSocket (persistent connection)
- Include: latitude, longitude, heading, speed, timestamp
\`\`\`

**Location Service Architecture**:

\`\`\`
Dasher App → WebSocket → Location Gateway
                               ↓
                         Kafka (location_updates topic)
                               ↓
                         Stream Processor (Flink)
                               ↓
                    ┌──────────┴───────────┐
                    ▼                      ▼
              Redis (current)        Cassandra (history)
         (dasher_id → location)    (dasher_id, timestamp → location)
\`\`\`

**Redis Storage** (Current Location):
\`\`\`
Key: dasher:location:{dasher_id}
Value: {lat: 37.7749, lon: -122.4194, timestamp: 1698158400, heading: 45, speed: 25}
TTL: 60 seconds (if no update, dasher considered offline)
\`\`\`

**Geospatial Indexing** (H3, same as Uber):
- Convert location to H3 hexagon ID
- Index Dashers by hex ID (Redis)
- Fast queries: "Find Dashers within 2 miles of restaurant"

**Data Model** (Redis):
\`\`\`
Key: hex:{h3_hex_id}
Value: Set of dasher_ids in this hex
\`\`\`

**Query**:
\`\`\`python
def find_nearby_dashers(restaurant_lat, restaurant_lon, radius_km=2):
    # Convert to H3 hex at resolution 9 (~500m per hex)
    restaurant_hex = h3.geo_to_h3(restaurant_lat, restaurant_lon, 9)
    
    # Get k-ring (neighboring hexes)
    # k=4 covers approximately 2km radius
    hex_ring = h3.k_ring(restaurant_hex, 4)
    
    # Query Redis for Dashers in these hexes
    dasher_ids = set()
    for hex_id in hex_ring:
        dashers = redis.smembers(f"hex:{hex_id}")
        dasher_ids.update(dashers)
    
    # Get current locations for these Dashers
    locations = redis.mget([f"dasher:location:{d}" for d in dasher_ids])
    
    # Filter by exact distance
    nearby_dashers = []
    for dasher_id, location in zip(dasher_ids, locations):
        if location:
            distance = haversine(restaurant_lat, restaurant_lon, location['lat'], location['lon'])
            if distance <= radius_km:
                nearby_dashers.append({
                    'dasher_id': dasher_id,
                    'distance': distance,
                    'location': location
                })
    
    return nearby_dashers
\`\`\`

**Historical Location** (Cassandra):
- Store location history for analytics
- Use cases: Route reconstruction, ETA accuracy analysis, dispute resolution

\`\`\`
Table: location_history
Partition Key: dasher_id
Clustering Key: timestamp DESC
Columns: lat, lon, heading, speed, accuracy

Query: "Get Dasher X's location at time T"
       → Single partition, time-ordered
\`\`\`

---

### 3. Dispatch and Matching

Dispatch is DoorDash's core algorithm: assign orders to optimal Dasher.

**Dispatch Challenges**:
- **Multiple objectives**: Minimize delivery time, maximize Dasher earnings, ensure merchant satisfaction
- **Constraints**: Dasher capacity (can't carry 10 orders), restaurant prep time, customer wait time tolerance
- **Real-time**: Must decide within 5-10 seconds
- **Scale**: Thousands of orders per minute during dinner rush

**Dispatch Algorithm**:

**Simple Approach (Greedy)**:
\`\`\`
Find nearest available Dasher to restaurant → Assign order
\`\`\`

**Problems**:
- Suboptimal globally (Dasher might be better for different order)
- Doesn't consider: Prep time, Dasher's current location trajectory, batching opportunities

**DoorDash's Approach (Optimization)**:

**Batch Matching** (every 30 seconds):

\`\`\`
1. Collect pending orders (last 30 seconds)
2. Collect available Dashers (online, no current delivery or finishing soon)
3. Formulate as optimization problem:
   - Minimize: Total delivery time + wait time
   - Maximize: Dasher utilization + earnings
   - Constraints: 
     - Dasher can carry max 2 orders (stacked deliveries)
     - Orders must be delivered within promised time
     - Dasher must accept (predicted acceptance rate)
4. Solve optimization (Linear Programming or greedy heuristics)
5. Assign orders to Dashers
6. Send dispatch notifications to Dashers
7. Dashers have 60 seconds to accept or decline
\`\`\`

**Optimization Formulation**:

\`\`\`
Variables:
- x_{i,j} = 1 if order i assigned to Dasher j, 0 otherwise

Objective:
Minimize: Σ (delivery_time_{i,j} * x_{i,j}) + Σ (wait_time_i * x_{i,j})

Constraints:
- Σ_j x_{i,j} = 1  (each order assigned to exactly one Dasher)
- Σ_i x_{i,j} <= 2  (each Dasher max 2 orders)
- delivery_time_{i,j} <= promised_time_i  (on-time delivery)
- distance_{j, restaurant_i} <= max_distance  (reasonable pickup distance)
\`\`\`

**Factors Considered**:

**1. Distance**:
- Dasher to restaurant (pickup distance)
- Restaurant to customer (delivery distance)
- Current location to restaurant (if Dasher in transit)

**2. Estimated Time**:
- Pickup time: Distance / average_speed + restaurant_prep_time
- Delivery time: Pickup time + delivery_distance / average_speed
- Traffic conditions (real-time data from Google Maps API)

**3. Dasher State**:
- Available (online, no delivery)
- On delivery (but finishing soon, can stack)
- Busy (skip)

**4. Dasher Quality**:
- Acceptance rate (likelihood to accept offer)
- Completion rate (unlikely to cancel)
- Customer rating (high-quality service)
- On-time rate (reliable)

**5. Order Characteristics**:
- Order value (higher value → more experienced Dasher)
- Special requirements (alcohol → ID check capable)
- Customer preferences (favorite Dasher if repeat customer)

**6. Stacked Deliveries** (Batching):
- Can Dasher pick up 2 orders from nearby restaurants?
- Deliver both en route (same direction)?
- Ensures both delivered on time?

**Example**:
\`\`\`
Order A: Restaurant R1 → Customer C1 (2 miles)
Order B: Restaurant R2 → Customer C2 (1.5 miles)
Dasher D: Currently at location L

Option 1: Assign A to D
- D → R1 (1 mile, 3 min)
- R1 → C1 (2 miles, 6 min)
- Total: 9 min for Order A

Option 2: Assign B to D
- D → R2 (0.5 miles, 2 min)
- R2 → C2 (1.5 miles, 5 min)
- Total: 7 min for Order B

Option 3: Stack both (if R1 and R2 close, C1 and C2 same direction)
- D → R1 → R2 → C1 → C2
- Total: 15 min for both
- But: Parallel delivery (another Dasher takes 9 min + 7 min = 16 min total)
- Stacking saves 1 min, keeps one Dasher free for next order
\`\`\`

**Dispatch Service Architecture**:

\`\`\`python
class DispatchService:
    def dispatch_batch(self):
        # Run every 30 seconds
        orders = self.get_pending_orders()
        dashers = self.get_available_dashers()
        
        # Generate candidates (order, dasher pairs)
        candidates = []
        for order in orders:
            nearby_dashers = self.find_nearby_dashers(
                order.restaurant_location, 
                radius_km=5
            )
            for dasher in nearby_dashers:
                score = self.calculate_score(order, dasher)
                if score > threshold:
                    candidates.append((order, dasher, score))
        
        # Solve optimization (greedy or LP solver)
        assignments = self.optimize(candidates)
        
        # Send dispatch notifications
        for order, dasher in assignments:
            self.send_dispatch(dasher, order)
            self.update_order_status(order, 'dispatched')
    
    def calculate_score(self, order, dasher):
        # Predict delivery time
        pickup_time = self.estimate_pickup_time(dasher.location, order.restaurant, order.prep_time)
        delivery_time = self.estimate_delivery_time(order.restaurant, order.customer_location)
        total_time = pickup_time + delivery_time
        
        # Predict acceptance probability
        acceptance_prob = self.predict_acceptance(dasher, order)
        
        # Calculate score (higher = better)
        # Penalize long delivery times, reward high acceptance prob
        score = acceptance_prob / (total_time + 1)
        
        return score
\`\`\`

---

### 4. ETA Prediction

Accurate ETA (Estimated Time of Arrival) is critical for customer experience.

**ETA Components**:

**1. Prep Time Prediction**:
- Historical data: Restaurant X takes average 15 minutes for orders
- Time of day: Dinner rush (longer prep time)
- Order complexity: Large orders take longer
- Restaurant staffing (detected via prep time patterns)

**ML Model**:
\`\`\`
Features:
- Restaurant ID, time of day, day of week
- Order size (item count, subtotal)
- Restaurant's recent prep times (last hour average)
- Special events (local events, weather)

Model: Gradient Boosted Trees (XGBoost)
Predict: Prep time in minutes
\`\`\`

**2. Pickup Time Prediction**:
- Distance from Dasher to restaurant
- Travel time (Google Maps API or internal model)
- Traffic conditions (real-time)
- Dasher's speed patterns (some faster than others)

**3. Delivery Time Prediction**:
- Distance from restaurant to customer
- Travel time
- Traffic
- Delivery difficulty (apartment building takes longer than house)

**4. Total ETA**:
\`\`\`
ETA = prep_time + pickup_time + delivery_time + buffer

Example:
Prep time: 15 minutes (restaurant cooking)
Pickup time: 5 minutes (Dasher drives to restaurant)
Delivery time: 10 minutes (Dasher delivers to customer)
Buffer: 3 minutes (safety margin)
Total ETA: 33 minutes
\`\`\`

**Real-Time ETA Updates**:

As order progresses, refine ETA:
\`\`\`
Order confirmed → Update prep_time (restaurant gave estimate)
Dasher assigned → Update pickup_time (know Dasher's location)
Dasher en route to restaurant → Continuously update pickup_time (GPS tracking)
Dasher picked up food → Update delivery_time (know route, traffic)
Dasher en route to customer → Continuously update delivery_time (GPS tracking)
\`\`\`

**Communication**:
- Push notifications to customer: "Your order is arriving in 10 minutes"
- Live map showing Dasher's location
- Update ETA every minute (recalculate based on current location)

---

### 5. Dynamic Pricing and Surge

DoorDash adjusts pricing based on supply and demand.

**Pricing Components**:

**Delivery Fee** (Customer pays):
\`\`\`
Base delivery fee: $2-$5 (depends on distance)
+ Surge (high demand): $0-$10
- DashPass (subscription): $0 (free delivery)
+ Small order fee: $2 (if subtotal < $10)
= Total delivery fee
\`\`\`

**Dasher Pay**:
\`\`\`
Base pay: $2-$10 (per delivery, based on distance, time, desirability)
+ Peak pay (surge): $1-$5 per delivery
+ Tips (from customer)
= Total earnings
\`\`\`

**Surge Pricing**:

When demand > supply (more orders than Dashers):
- Increase delivery fee (customer pays more)
- Increase peak pay (incentivize Dashers to come online)

**Surge Algorithm**:

\`\`\`python
def calculate_surge(zone_id, time):
    # Count pending orders in zone
    pending_orders = count_pending_orders(zone_id)
    
    # Count available Dashers in zone
    available_dashers = count_available_dashers(zone_id)
    
    # Calculate demand/supply ratio
    ratio = pending_orders / (available_dashers + 1)
    
    # Surge multiplier based on ratio
    if ratio > 3:
        surge = 5  # High surge
    elif ratio > 2:
        surge = 3  # Medium surge
    elif ratio > 1.5:
        surge = 1  # Low surge
    else:
        surge = 0  # No surge
    
    # Adjust for time of day (dinner rush higher surge)
    if is_dinner_rush(time):
        surge *= 1.5
    
    return surge
\`\`\`

**Promotions**:
- Customer: "$5 off first order", "Free delivery"
- Dasher: "Complete 5 deliveries, earn $25 bonus"
- Restaurant: "DoorDash covers delivery fee" (promotional campaigns)

---

### 6. Merchant (Restaurant) Integration

Restaurants integrate with DoorDash via tablet or POS integration.

**Order Notification**:
\`\`\`
Customer places order → DoorDash sends to restaurant:
- Push notification to DoorDash tablet
- Audible alert (ding)
- Order printed on receipt printer (optional)
- POS integration (order appears in restaurant's system)
\`\`\`

**Merchant Actions**:
- **Confirm**: Accept order, provide prep time (10-20 minutes)
- **Reject**: Out of items, too busy, outside delivery radius
- **Modify**: Change prep time (taking longer than expected)

**Menu Management**:
- Restaurants manage menus via Merchant Portal (web app)
- Add/remove items, update prices, mark items out of stock
- Sync menu to DoorDash platform (near real-time)

**Data Model** (PostgreSQL):
\`\`\`sql
Table: merchants
- merchant_id
- name
- address
- cuisine_type
- rating (average of customer reviews)
- delivery_radius (miles)
- is_active (online/offline)
- operating_hours (JSON)

Table: menu_items
- item_id
- merchant_id (FK)
- name
- description
- price
- category (appetizer, entree, dessert)
- customizations (JSON array)
  - option_name, choices, prices
- is_available (in stock)
- image_url

Table: merchant_stats
- merchant_id
- date
- orders_count
- average_prep_time
- acceptance_rate
- rating
\`\`\`

---

### 7. Customer Experience

**Search and Discovery**:

\`\`\`
User opens app → Show nearby restaurants (Elasticsearch):
- Geospatial query (restaurants within 5 miles)
- Filters: Cuisine, price range, rating, delivery time
- Sorting: Distance, rating, popularity, delivery time
- Personalization: Past orders, preferences
\`\`\`

**Search Index** (Elasticsearch):
\`\`\`json
{
  "merchant_id": "12345",
  "name": "Pizza Place",
  "cuisine": ["Italian", "Pizza"],
  "location": {"lat": 37.7749, "lon": -122.4194},
  "rating": 4.5,
  "delivery_time": 30,  // minutes (estimated)
  "delivery_fee": 2.99,
  "minimum_order": 10,
  "is_open": true,
  "accepts_doordash": true,
  "popular_items": ["Margherita Pizza", "Pepperoni Pizza"]
}
\`\`\`

**Cart and Checkout**:
- Add items to cart (stored in Redis or client-side)
- Apply promo codes
- Select delivery time (ASAP or scheduled)
- Tip Dasher (percentage or custom amount)
- Payment (credit card via Stripe)

**Order Tracking**:
- Real-time status updates (confirmed, picked up, en route)
- Live map showing Dasher's location
- ETA countdown
- Contact Dasher (call/text via proxy number)

---

### 8. Payment Processing

DoorDash handles complex three-way payments: Customer → DoorDash → Restaurant + Dasher.

**Payment Flow**:

\`\`\`
1. Customer places order:
   - Charge customer's card (via Stripe)
   - Amount: Food + delivery fee + tip
   - Funds held by DoorDash (escrow)

2. Order completed:
   - Transfer to restaurant: Food cost - DoorDash commission (20-30%)
   - Transfer to Dasher: Base pay + tip
   - DoorDash keeps: Commission + delivery fee - base pay

Example:
Food: $20
Delivery fee: $3
Tip: $5
Total charged to customer: $28

Payouts:
- Restaurant: $20 - $5 (25% commission) = $15
- Dasher: $3 (base pay) + $5 (tip) = $8
- DoorDash: $5 (commission) + $3 (delivery fee) - $3 (base pay) = $5
\`\`\`

**Payment Challenges**:

**1. Refunds**:
- Wrong order, missing items, late delivery
- Partial or full refund
- Restaurant responsible vs DoorDash responsible (impacts who pays)

**2. Chargebacks**:
- Customer disputes charge
- DoorDash provides evidence (delivery proof, photos)

**3. Multi-Currency** (International):
- Different currencies in different countries
- Exchange rate handling

**4. Payout Timing**:
- Dashers: Weekly payout (direct deposit)
- Fast Pay: Instant payout for $1.99 fee (via debit card)
- Restaurants: Weekly or bi-weekly

---

## Technology Stack

### Backend

- **Python**: Primary backend language (Django/Flask initially, microservices now)
- **Go**: Performance-critical services (dispatch, location tracking)
- **Java**: Some services
- **Kotlin**: Android app

### Data Storage

- **PostgreSQL**: Primary database (orders, users, merchants, Dashers)
- **Redis**: Caching, real-time data (Dasher locations, surge pricing)
- **Cassandra**: Time-series data (location history, events)
- **Elasticsearch**: Search (restaurants, items)

### Data Processing

- **Apache Kafka**: Event streaming (orders, location updates, payments)
- **Apache Flink**: Stream processing (real-time analytics, surge detection)
- **Apache Spark**: Batch processing (ML training, reporting)
- **Airflow**: Workflow orchestration (data pipelines)

### Machine Learning

- **TensorFlow**: Deep learning (ETA prediction, demand forecasting)
- **XGBoost**: Gradient boosted trees (prep time prediction, surge pricing)
- **Scikit-learn**: Classical ML

### Infrastructure

- **AWS**: Primary cloud provider
- **Kubernetes**: Container orchestration
- **Terraform**: Infrastructure as code

### Monitoring

- **Datadog**: Metrics, logs, traces
- **PagerDuty**: On-call, alerting
- **Sentry**: Error tracking

---

## Key Lessons from DoorDash Architecture

### 1. Real-Time Logistics is Complex

Dispatch algorithm must optimize multiple objectives (delivery time, Dasher earnings, costs) with real-time constraints. Batch matching with optimization outperforms greedy.

### 2. Location Tracking at Scale

H3 geospatial indexing enables fast "nearby Dashers" queries. Redis for current location (low latency), Cassandra for history.

### 3. ETA Prediction Drives Experience

Accurate ETAs require ML models considering prep time, traffic, Dasher behavior. Continuous refinement as order progresses.

### 4. Dynamic Pricing Balances Marketplace

Surge pricing increases supply (incentivize Dashers) and reduces demand (higher prices) during peak times.

### 5. Three-Sided Marketplace is Complex

Balancing customer, merchant, and Dasher needs requires careful optimization. Each side has different objectives and constraints.

---

## Interview Tips

**Q: How would you design DoorDash's dispatch system?**

A: Use batch matching with optimization. Every 30 seconds: (1) Collect pending orders and available Dashers. (2) For each order, find nearby Dashers (H3 geospatial index, radius 5km). (3) Calculate score for each (order, Dasher) pair: predicted delivery time, acceptance probability, Dasher quality. (4) Formulate as optimization: minimize total delivery time subject to constraints (Dasher capacity ≤2 orders, on-time delivery, acceptance probability >threshold). (5) Solve with Linear Programming or greedy algorithm (balance optimality vs speed). (6) Assign orders, send dispatch notifications. (7) Dashers accept/decline within 60 seconds. Handle declines: reassign to next-best Dasher. Consider stacked deliveries: Can Dasher handle 2 orders from nearby restaurants delivering same direction?

**Q: How does DoorDash track Dasher locations in real-time?**

A: Dasher app sends GPS coordinates every 4 seconds via WebSocket. Location Gateway receives updates, publishes to Kafka topic (location_updates). Flink processes stream: (1) Update Redis (current location): dasher:location:{id} with TTL 60s. (2) Convert to H3 hex ID (resolution 9), update Redis set hex:{id} → dasher_ids. (3) Write to Cassandra (historical): location_history partitioned by dasher_id, clustered by timestamp. Query nearby Dashers: Convert restaurant location to H3 hex, get k-ring (neighboring hexes), query Redis sets for dasher_ids in these hexes, filter by exact distance. Result: Sub-100ms queries for "Dashers within 2km."

**Q: How would you implement surge pricing for DoorDash?**

A: Divide city into zones (H3 hexagons). For each zone, every minute: (1) Count pending orders (orders not yet assigned to Dasher). (2) Count available Dashers (online, no delivery or finishing soon). (3) Calculate demand/supply ratio = pending_orders / available_dashers. (4) If ratio >3: surge=$5, ratio >2: surge=$3, ratio >1.5: surge=$1, else: surge=$0. (5) Adjust for time (dinner rush 5-9 PM: multiply surge by 1.5). (6) Store in Redis: surge:{zone_id} = surge_amount. (7) Customer delivery fee += surge. (8) Dasher peak pay += surge (incentive to go online). Monitor supply response: If Dashers come online, surge decreases. Display to Dashers: "Earn $5 extra per delivery in downtown zone."

---

## Summary

DoorDash's architecture handles real-time logistics for three-sided marketplace at scale:

**Key Takeaways**:

1. **Batch dispatch**: Optimize order-Dasher matching every 30 seconds, consider delivery time, acceptance probability, stacking
2. **H3 geospatial indexing**: Fast "nearby Dashers" queries, Redis for current location, Cassandra for history
3. **ETA prediction**: ML models for prep time, pickup time, delivery time, continuous refinement
4. **Dynamic pricing**: Surge pricing based on demand/supply ratio, balances marketplace
5. **Real-time tracking**: WebSocket for GPS updates (4-second frequency), Kafka + Flink for processing
6. **Three-way payments**: Customer → DoorDash → Restaurant + Dasher, handle commissions, tips, refunds
7. **Search and discovery**: Elasticsearch for restaurant search, geospatial + filters + personalization

DoorDash's success comes from sophisticated dispatch optimization, real-time location tracking, accurate ETA prediction, and dynamic marketplace balancing.
`,
};
