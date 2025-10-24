/**
 * Design Uber Section
 */

export const uberSection = {
    id: 'uber',
    title: 'Design Uber',
    content: `Uber is a ride-hailing platform connecting riders and drivers in real-time. The core challenges include: real-time location tracking, efficient matching algorithms, ETA calculations, surge pricing, and handling millions of concurrent connections globally.

## Problem Statement

Design Uber with:
- **Request Ride**: Riders request rides specifying pickup/dropoff locations
- **Driver Matching**: Match rider with nearby available driver (< 30 seconds)
- **Real-time Tracking**: Track driver location, show ETA to rider
- **Navigation**: Provide turn-by-turn directions to driver
- **Pricing**: Dynamic pricing with surge multipliers
- **Payment**: Process payments at trip completion
- **Ratings**: Riders rate drivers, drivers rate riders
- **Trip History**: View past trips

**Scale**: 100M riders, 5M drivers, 15M trips/day

---

## Step 1: Requirements Gathering

### Functional Requirements

1. **Request Ride**: Rider enters pickup/dropoff, requests ride
2. **Match Driver**: Find nearby available driver and send request
3. **Accept/Decline**: Driver accepts or declines ride request
4. **Real-time Tracking**: Show driver location on map (update every 3-5 seconds)
5. **Navigation**: Provide optimal route to driver
6. **Trip Lifecycle**: Start trip → En route → Complete → Payment
7. **Pricing**: Calculate fare (base + distance + time + surge)
8. **Payment**: Process payment via credit card/wallet
9. **Ratings & Reviews**: 5-star rating system
10. **Trip History**: View past rides

### Non-Functional Requirements

1. **Low Latency**: < 30sec to match driver, < 100ms location updates
2. **High Availability**: 99.99% uptime (critical for safety)
3. **Scalable**: Handle 15M trips/day globally
4. **Real-time**: Location updates within 5 seconds
5. **Accurate ETA**: Within ±2 minutes
6. **Global**: Support 600+ cities worldwide

---

## Step 2: Capacity Estimation

### Traffic

**Trips**:
- 15M trips/day = ~175 trips/sec
- Peak (Friday 6 PM): ~500 trips/sec

**Active Users**:
- At any moment: 1M riders + 500K drivers = 1.5M active users
- Each sending location every 5 seconds = 300K location updates/sec

**WebSocket Connections**:
- 1.5M concurrent WebSocket connections (for real-time updates)
- ~7.5 GB memory for connection state (5 KB per connection)

### Storage

**Users**:
- 100M riders × 1 KB = 100 GB
- 5M drivers × 1 KB = 5 GB

**Trips**:
- 15M trips/day × 2 KB metadata = 30 GB/day
- 1 year: 30 GB × 365 = 11 TB

**Location History**:
- 500K active drivers × 1 location/5sec × 86,400 sec/day = 8.6B location points/day
- Each point: 20 bytes (driver_id, lat, lon, timestamp) = 172 GB/day
- 1 year: 172 GB × 365 = 62 TB

---

## Step 3: System API Design

### Request Ride

\`\`\`json
POST /api/v1/rides/request
{
  "rider_id": 123,
  "pickup_location": {"lat": 37.7749, "lon": -122.4194},
  "dropoff_location": {"lat": 37.8044, "lon": -122.2712},
  "ride_type": "uberX"
}

Response (201):
{
  "ride_request_id": "req_abc123",
  "status": "searching",
  "estimated_wait_time": "3 min"
}
\`\`\`

### Update Driver Location

\`\`\`json
POST /api/v1/drivers/location
{
  "driver_id": 456,
  "location": {"lat": 37.7749, "lon": -122.4194},
  "heading": 45,
  "speed": 30,
  "timestamp": 1698160000
}

Response (200):
{
  "status": "received"
}
\`\`\`

### Accept Ride

\`\`\`json
POST /api/v1/rides/{ride_id}/accept
{
  "driver_id": 456
}

Response (200):
{
  "ride_id": "ride_xyz789",
  "rider": {
    "name": "Alice",
    "rating": 4.9,
    "pickup_location": {"lat": 37.7749, "lon": -122.4194}
  },
  "estimated_pickup_time": "4 min"
}
\`\`\`

---

## Step 4: Database Schema

### Users Table (PostgreSQL)

\`\`\`sql
CREATE TABLE users (
    user_id BIGINT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(20),
    user_type VARCHAR(10),  -- 'rider' or 'driver'
    rating DECIMAL(3,2),
    created_at TIMESTAMP
);
\`\`\`

### Drivers Table

\`\`\`sql
CREATE TABLE drivers (
    driver_id BIGINT PRIMARY KEY REFERENCES users(user_id),
    vehicle_type VARCHAR(20),  -- uberX, uberXL, black
    license_plate VARCHAR(10),
    current_location GEOMETRY(Point, 4326),  -- PostGIS
    is_available BOOLEAN,
    last_location_update TIMESTAMP,
    INDEX idx_location (current_location) USING GIST
);
\`\`\`

### Trips Table (Cassandra - Time-series)

\`\`\`cql
CREATE TABLE trips (
    trip_id UUID,
    rider_id BIGINT,
    driver_id BIGINT,
    pickup_location TEXT,  -- JSON: {lat, lon}
    dropoff_location TEXT,
    pickup_time TIMESTAMP,
    dropoff_time TIMESTAMP,
    distance_km DECIMAL,
    duration_minutes INT,
    fare_amount DECIMAL,
    status VARCHAR(20),  -- requested, accepted, started, completed
    created_at TIMESTAMP,
    PRIMARY KEY (trip_id)
);
\`\`\`

### Location History (Cassandra)

\`\`\`cql
CREATE TABLE location_history (
    driver_id BIGINT,
    timestamp TIMESTAMP,
    location TEXT,  -- {lat, lon}
    heading INT,
    speed INT,
    PRIMARY KEY (driver_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);
\`\`\`

---

## Step 5: High-Level Architecture

\`\`\`
┌────────────────┐         ┌────────────────┐
│  Rider App     │         │  Driver App    │
└───────┬────────┘         └───────┬────────┘
        │ WebSocket                │ WebSocket
        │                          │
        ▼                          ▼
┌──────────────────────────────────────────┐
│       WebSocket Gateway (Stateful)       │
│     Manages 1.5M concurrent connections  │
└───────────────────┬──────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌──────────┐  ┌──────────┐  ┌──────────────┐
│  Ride    │  │ Location │  │  Matching    │
│ Service  │  │ Service  │  │  Service     │
└────┬─────┘  └────┬─────┘  └──────┬───────┘
     │             │                │
     │             ▼                │
     │      ┌─────────────┐         │
     │      │   Redis     │         │
     │      │(Geospatial) │         │
     │      └─────────────┘         │
     │                              │
     ▼                              ▼
┌─────────────────────────────────────────┐
│        PostgreSQL (Users, Drivers)      │
└─────────────────────────────────────────┘

     ┌──────────────────┐
     │  Cassandra       │
     │ (Trips, Locations)│
     └──────────────────┘

     ┌──────────────────┐
     │   Kafka          │
     │ (Event Streaming)│
     └──────────────────┘

     ┌──────────────────┐
     │  Maps API        │
     │ (Google/Mapbox)  │
     └──────────────────┘
\`\`\`

---

## Step 6: Geospatial Indexing

**Core Challenge**: Find nearby available drivers within 5 km of rider.

### Approach 1: QuadTree

**Concept**: Recursively divide map into 4 quadrants until each contains manageable number of drivers.

\`\`\`
World Map
├─ NW Quadrant
│  ├─ NW-NW (San Francisco)
│  │  ├─ 50 drivers
│  │  └─ ...
│  └─ NW-SE
├─ NE Quadrant
└─ ...
\`\`\`

**Query**: 
1. Find quadrant containing rider location
2. Get all drivers in that quadrant
3. Filter by distance

**Pros**: Dynamic, adjusts to driver density
**Cons**: Requires in-memory data structure, complex updates

### Approach 2: Geohash

**Concept**: Encode lat/lon into base32 string. Nearby locations have common prefixes.

\`\`\`
San Francisco: 9q8yy
Nearby driver 1: 9q8yz (common prefix: 9q8y)
Nearby driver 2: 9q8yt (common prefix: 9q8y)
Los Angeles: 9q5c (different prefix)
\`\`\`

**Implementation**:
\`\`\`python
import geohash

# Encode location
driver_hash = geohash.encode(37.7749, -122.4194, precision=6)  # "9q8yyz"

# Store in Redis Sorted Set
redis.geoadd("drivers:available", -122.4194, 37.7749, driver_id)

# Find nearby drivers within 5 km
nearby = redis.georadius("drivers:available", -122.4194, 37.7749, 5, "km")
# Returns: [driver_123, driver_456, driver_789]
\`\`\`

**Pros**: Simple, built into Redis, efficient
**Cons**: Precision issues at borders

### Approach 3: Redis Geospatial (Recommended)

**Redis GEOADD/GEORADIUS** uses internally Sorted Sets with geohash.

\`\`\`
GEOADD drivers:available -122.4194 37.7749 driver_123
GEOADD drivers:available -122.4100 37.7800 driver_456

GEORADIUS drivers:available -122.4194 37.7749 5 km WITHDIST
# Returns:
# 1) driver_123, distance: 0.5 km
# 2) driver_456, distance: 2.3 km
\`\`\`

**Benefits**:
- Sub-millisecond queries
- Built-in distance calculation
- Automatically sorted by proximity
- Handles millions of drivers

---

## Step 7: Driver Matching Algorithm

**Flow**:

\`\`\`
1. RIDER REQUESTS RIDE
   - POST /api/v1/rides/request
   - Ride Service creates ride_request record
   - Status: "searching"

2. FIND NEARBY DRIVERS
   - Query Redis: GEORADIUS drivers:available {lat} {lon} 5 km
   - Returns: [driver_123, driver_456, driver_789]
   - Filter: Only available drivers (status = available)
   - Sort by: Distance (closest first)

3. SEND REQUEST TO CLOSEST DRIVER
   - Send push notification to driver_123 via WebSocket
   - Message: "New ride request: $10 fare, 0.5 km away"
   - Start 15-second timeout

4. DRIVER RESPONSE
   - Option A: Driver accepts → Match complete, notify rider
   - Option B: Driver declines → Try next driver (driver_456)
   - Option C: Timeout (15 sec) → Try next driver

5. MATCH COMPLETE
   - Update ride status: "accepted"
   - Publish "ride_accepted" event to Kafka
   - Send confirmation to rider via WebSocket:
     {
       "driver": {
         "name": "John",
         "rating": 4.8,
         "vehicle": "Toyota Camry (ABC123)",
         "location": {lat, lon},
         "eta": "4 min"
       }
     }

6. FALLBACK (No drivers accept)
   - After trying 5 drivers, expand radius: GEORADIUS 10 km
   - Increase fare multiplier (surge pricing)
   - Send notification: "No drivers nearby, expanding search..."
   - If still no match after 2 minutes: "No drivers available, try again"
\`\`\`

### Optimization: Pre-filtering

Before GEORADIUS, filter by:
- Vehicle type (uberX, uberXL, black)
- Driver rating (> 4.5)
- Acceptance rate (> 80%)

Store in separate Redis sets:
\`\`\`
drivers:available:uberX
drivers:available:uberXL
drivers:available:black
\`\`\`

---

## Step 8: Real-time Location Tracking

**Challenge**: Track 500K active drivers, update every 3-5 seconds.

### Driver Location Update Flow

\`\`\`
1. DRIVER APP SENDS LOCATION
   - Every 5 seconds (when driving) or 30 seconds (when idle)
   - POST /api/v1/drivers/location
   - Payload: {driver_id, lat, lon, heading, speed, timestamp}

2. LOCATION SERVICE PROCESSES UPDATE
   - Validate timestamp (not stale, not future)
   - Update Redis geospatial index:
     GEOADD drivers:available {lon} {lat} {driver_id}
   - Store in Cassandra for history:
     INSERT INTO location_history (driver_id, timestamp, location)
   - Publish to Kafka topic "driver_locations"

3. RIDER TRACKING (If on active trip)
   - Query: "Which riders are tracking this driver?"
   - Push location update via WebSocket to rider
   - Rider's map updates driver marker
   - Recalculate ETA based on new location

4. ETA CALCULATION
   - Use distance + current traffic conditions
   - Query Maps API: Get driving time from driver location to pickup
   - Update every 30 seconds
   - Push to rider: "John is 3 minutes away"
\`\`\`

### WebSocket Architecture

**Rider Connection**:
\`\`\`typescript
// Rider connects
const ws = new WebSocket('wss://uber.com/rider/123');

ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    if (update.type === 'driver_location') {
        // Update map with driver's new location
        map.updateMarker(update.driver_id, update.location);
    }
    if (update.type === 'eta_update') {
        // Update ETA display
        updateETA(update.eta_minutes);
    }
};
\`\`\`

**Server Broadcasting**:
\`\`\`python
# When driver location received
def on_driver_location_update(driver_id, location):
    # Find active trip for this driver
    trip = get_active_trip(driver_id)
    if trip:
        rider_id = trip.rider_id
        # Send to rider's WebSocket
        websocket_send(rider_id, {
            'type': 'driver_location',
            'location': location,
            'eta': calculate_eta(location, trip.pickup_location)
        })
\`\`\`

---

## Step 9: ETA Calculation

**Components**:

1. **Distance**: Haversine formula for straight-line distance
2. **Route**: Actual driving route (not straight line)
3. **Traffic**: Real-time traffic conditions
4. **Historical Data**: Average speed on route at this time

**Implementation**:

\`\`\`python
def calculate_eta(driver_location, destination):
    # Option 1: Haversine (fast approximation)
    distance_km = haversine(driver_location, destination)
    avg_speed_kmh = 40  # City driving
    eta_minutes = (distance_km / avg_speed_kmh) * 60
    
    # Option 2: Maps API (accurate with traffic)
    result = google_maps.directions(
        origin=driver_location,
        destination=destination,
        mode='driving',
        departure_time='now'  # Includes current traffic
    )
    eta_seconds = result['routes'][0]['legs'][0]['duration']['value']
    eta_minutes = eta_seconds / 60
    
    return eta_minutes
\`\`\`

**Optimization**:
- Cache common routes (airport → downtown)
- Update ETA every 30-60 seconds (not every second)
- Use historical data when Maps API unavailable

---

## Step 10: Surge Pricing

**Dynamic Pricing Algorithm**:

\`\`\`
surge_multiplier = demand / supply

Where:
- demand = number of ride requests in area (last 5 minutes)
- supply = number of available drivers in area
\`\`\`

**Implementation**:

\`\`\`python
def calculate_surge(lat, lon, radius_km=2):
    # Count ride requests in area (last 5 min)
    requests = redis.zcount(
        'ride_requests:geo',
        geohash_min(lat, lon, radius_km),
        geohash_max(lat, lon, radius_km)
    )
    
    # Count available drivers in area
    drivers = redis.georadius_count(
        'drivers:available',
        lon, lat, radius_km, 'km'
    )
    
    if drivers == 0:
        return 3.0  # Max surge
    
    ratio = requests / drivers
    
    if ratio < 1:
        return 1.0  # No surge
    elif ratio < 2:
        return 1.5  # 1.5x
    elif ratio < 3:
        return 2.0  # 2x
    else:
        return 3.0  # 3x (max)
\`\`\`

**Surge Zones**:
- Divide city into grid cells (H3 hexagons)
- Calculate surge per cell
- Display heat map in rider app

**Benefits**:
- Incentivizes drivers to go to high-demand areas
- Reduces wait times
- Balances supply and demand

---

## Step 11: Payment Processing

**Flow**:

\`\`\`
1. TRIP COMPLETES
   - Driver taps "End Trip"
   - Calculate final fare:
     base_fare + (distance × per_km_rate) + (time × per_min_rate) × surge_multiplier
   - Example: $2 + (10 km × $1.5/km) + (20 min × $0.3/min) × 1.5 = $30

2. CHARGE RIDER
   - Use stored payment method (credit card on file)
   - Call Stripe API:
     stripe.charges.create({
       amount: 3000,  // $30.00 in cents
       currency: 'usd',
       customer: rider_stripe_id,
       description: 'Uber trip from A to B'
     })
   
3. HANDLE RESULT
   - Success → Mark trip as paid, send receipt
   - Failure → Retry 3 times, then mark as unpaid
   - Send push notification: "Your card was declined"

4. PAY DRIVER
   - Uber takes 25% commission
   - Driver receives 75%: $30 × 0.75 = $22.50
   - Weekly payout to driver's bank account (ACH transfer)

5. RECEIPT
   - Email/SMS receipt to rider
   - Store in trip history
\`\`\`

### Idempotency

**Problem**: Network failure during payment → Retry charges rider twice

**Solution**: Use idempotency key
\`\`\`python
stripe.charges.create(
    amount=3000,
    currency='usd',
    customer=rider_id,
    idempotency_key=f'trip_{trip_id}'  # Unique per trip
)
# If retried with same key, Stripe returns original charge (no duplicate)
\`\`\```

---

## Step 12: Ratings System

**Two-way Ratings**:

\`\`\`
After trip completes:
1. Rider rates driver (1-5 stars) + optional comment
2. Driver rates rider (1-5 stars)

Rating stored:
- Trip-level: trip_ratings table (specific trip)
- User-level: users table (rolling average)
\`\`\`

**Algorithm**:

\`\`\`python
def update_user_rating(user_id, new_rating):
    user = db.get_user(user_id)
    
    # Weighted average (recent ratings weighted more)
    total_trips = user.total_trips + 1
    current_rating = user.rating
    
    # Simple average
    new_avg = ((current_rating * user.total_trips) + new_rating) / total_trips
    
    # Weighted (last 100 trips more important)
    if total_trips > 100:
        weight_old = 0.7
        weight_new = 0.3
        new_avg = (current_rating * weight_old) + (new_rating * weight_new)
    
    db.update_user(user_id, rating=new_avg, total_trips=total_trips)
\`\`\`

**Consequences**:
- Driver rating < 4.6 → Warning, retraining required
- Driver rating < 4.3 → Deactivated
- Rider rating < 4.5 → May have longer wait times (drivers can see rating)

---

## Step 13: Database Sharding

### Sharding Users/Drivers by Geographic Region

**Strategy**: Shard by \`region_id\` (city/metro area)

\`\`\`
Shard 1: San Francisco, Oakland (Region 1)
Shard 2: New York City (Region 2)
Shard 3: Los Angeles (Region 3)
...
\`\`\`

**Benefits**:
- Most queries are within same city
- Trips rarely cross regions (99% within city)
- Easy to add new cities (new shard)

**Cross-region trips**:
- Store in both shards
- Rare enough to handle as exception

### Sharding Trips by trip_id

**Cassandra Partitioning**:
\`\`\`cql
PRIMARY KEY (trip_id)
-- trips distributed across cluster by trip_id hash
\`\`\```

**Query patterns**:
- Get trip by ID: Single partition query (fast)
- Get user's trips: Secondary index or maintain separate table

\`\`\`cql
CREATE TABLE trips_by_user (
    user_id BIGINT,
    trip_id UUID,
    created_at TIMESTAMP,
    PRIMARY KEY (user_id, created_at)
) WITH CLUSTERING ORDER BY (created_at DESC);
\`\`\`

---

## Step 14: Optimizations

### 1. Predictive Positioning

**ML Model**: Predict where demand will be high

\`\`\`
Features:
- Time of day
- Day of week
- Historical demand patterns
- Events (concerts, sports)
- Weather

Output: Probability of demand per grid cell

Action: Incentivize drivers to position in high-probability areas
\`\`\```

### 2. Route Optimization

**Challenge**: Driver picks up multiple riders (Uber Pool)

**Algorithm**: Traveling Salesman Problem (TSP)
- Use approximation algorithms (genetic algorithm, simulated annealing)
- Constraint: Max 10-minute detour per rider

### 3. Batching Location Updates

**Problem**: 300K location updates/sec overwhelming

**Solution**:
- Client buffers 10 updates (50 seconds worth)
- Sends batch: [location1, location2, ..., location10]
- Server processes batch (reduces requests by 10x)

### 4. Connection Pooling

**Problem**: Opening new DB connection for every request is slow

**Solution**:
- Maintain connection pool (100 connections per app server)
- Reuse connections
- PgBouncer for PostgreSQL connection pooling

---

## Step 15: Monitoring & Metrics

**Key Metrics**:

1. **Match Rate**: % of ride requests successfully matched (target: > 95%)
2. **Match Time**: Time to match driver (target: < 30 seconds)
3. **ETA Accuracy**: Actual vs predicted arrival time (target: ±2 minutes)
4. **Acceptance Rate**: % of drivers who accept requests (target: > 80%)
5. **Completion Rate**: % of accepted trips completed (target: > 98%)
6. **Active Drivers**: Number of drivers online (track hourly trends)
7. **WebSocket Connection Health**: Connection drops, reconnects

**Alerting**:
- Match rate < 90% → Surge pricing not working, need more drivers
- Average match time > 60 sec → Matching service overloaded
- WebSocket connection drops > 5% → Gateway issues

---

## Step 16: Safety & Compliance

**Features**:

1. **Emergency Button**: SOS button calls 911, shares location with police
2. **Trip Sharing**: Share trip details with friends/family in real-time
3. **Driver Verification**: Background checks, vehicle inspection
4. **Two-way Authentication**: PIN verification for rider/driver
5. **Trip Recording**: GPS trace stored for disputes
6. **Insurance**: Trip insurance coverage

---

## Trade-offs

### Accuracy vs Latency (ETA Calculation)

**Option 1**: Haversine formula (fast, ±5 min error)
**Option 2**: Maps API (accurate, 100-500ms latency)
**Production**: Haversine for initial estimate, Maps API for final ETA

### Strong Consistency vs Availability (Driver Availability)

**Scenario**: Driver accepts ride, but status not updated before second request comes

**Option 1**: Use distributed locks (strong consistency, slower)
**Option 2**: Eventual consistency (rare double-booking, compensate rider)
**Production**: Eventual consistency + compensation

---

## Interview Tips

### What to Clarify

1. **Scale**: How many users? Trips per day?
2. **Features**: Pool rides? Multiple stops? Driver ratings?
3. **Geographic Scope**: Single city or global?
4. **Accuracy**: ETA accuracy requirements?

### What to Emphasize

1. **Geospatial indexing**: Redis GEORADIUS for nearby driver search
2. **WebSocket**: Real-time location tracking and notifications
3. **Matching algorithm**: Iterative matching with timeout/fallback
4. **ETA calculation**: Maps API with traffic data
5. **Surge pricing**: Dynamic supply/demand algorithm

### Common Mistakes

1. ❌ Using SQL JOIN for nearby driver search (too slow, no geospatial index)
2. ❌ Polling for location updates (use WebSocket push)
3. ❌ Synchronous payment processing (use async with idempotency)
4. ❌ Not considering surge pricing (supply/demand balancing)

### Follow-up Questions

- "How do you handle driver going offline mid-trip?"
- "What if rider cancels after driver started driving?"
- "How do you detect fraud (fake GPS locations)?"
- "How would you implement Uber Pool (multiple riders)?"

---

## Summary

**Core Components**:
1. **Geospatial Index (Redis)**: Find nearby drivers (< 10ms)
2. **Matching Service**: Iterative matching with 15-sec timeout per driver
3. **WebSocket Gateway**: 1.5M concurrent connections for real-time updates
4. **Location Service**: Process 300K location updates/sec, store in Cassandra
5. **Maps API**: Route calculation, ETA prediction with traffic
6. **Surge Pricing Engine**: Dynamic pricing based on supply/demand ratio
7. **Payment Service**: Stripe integration with idempotency
8. **PostgreSQL**: Users, drivers, real-time availability
9. **Cassandra**: Trips, location history (time-series data)
10. **Kafka**: Event streaming (ride_requested, ride_accepted, trip_completed)

**Key Design Decisions**:
- ✅ **Redis Geospatial**: GEORADIUS for sub-millisecond nearby driver queries
- ✅ **WebSocket**: Real-time bidirectional communication (not HTTP polling)
- ✅ **Iterative matching**: Try drivers sequentially with 15-sec timeout
- ✅ **Maps API + Traffic**: Accurate ETA calculation
- ✅ **Surge pricing**: Ratio-based dynamic pricing (demand/supply)
- ✅ **Cassandra for trips**: Write-optimized for high-volume trip storage
- ✅ **Eventual consistency**: Accept rare double-booking, compensate riders
- ✅ **Sharding by region**: Geo-sharding for locality (SF, NYC, LA separate shards)

**Capacity**:
- 15M trips/day (175 trips/sec, 500 peak)
- 1.5M concurrent active users (1M riders, 500K drivers)
- 300K location updates/sec
- Sub-30-second driver matching
- ±2 minute ETA accuracy

This design handles **Uber-scale operations** across **600+ cities globally** with **real-time location tracking**, **efficient geospatial matching**, and **sub-second response times** for location queries, while maintaining high availability and handling millions of concurrent connections.`,
};

