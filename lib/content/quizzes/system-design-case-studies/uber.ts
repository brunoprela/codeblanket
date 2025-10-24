/**
 * Quiz questions for Design Uber section
 */

export const uberQuiz = [
  {
    id: 'q1',
    question:
      'Explain how Uber uses Redis Geospatial indexing to find nearby drivers. Compare this approach to using SQL with lat/lon columns and distance calculations.',
    sampleAnswer:
      'REDIS GEOSPATIAL APPROACH: (1) Store driver locations: GEOADD drivers:available -122.4194 37.7749 driver_123. Redis internally stores as Sorted Set with geohash scores. (2) Query nearby: GEORADIUS drivers:available -122.4194 37.7749 5 km WITHDIST. Returns drivers within 5 km sorted by distance in <10ms. Uses geohash + Z-order curve for spatial indexing. SQL APPROACH: SELECT * FROM drivers WHERE ST_Distance_Sphere(location, POINT(-122.4194, 37.7749)) < 5000 AND is_available = true ORDER BY distance. Requires full table scan or R-tree spatial index (slower, 100-500ms for millions of drivers). COMPARISON: Redis: O(log N + K) where K = results, sub-10ms, in-memory. SQL: O(N) without spatial index, 100-500ms even with index (disk I/O). For 500K active drivers, Redis: 5-10ms. SQL: 200-500ms. PRODUCTION: Redis for real-time matching (<30 sec requirement). SQL as backup/audit trail. KEY INSIGHT: Geospatial queries at scale require specialized data structures (geohash, quadtree, R-tree) - standard SQL B-trees insufficient.',
    keyPoints: [
      'Redis GEORADIUS: O(log N) lookups, sub-10ms for millions of drivers',
      'Uses geohash internally for spatial indexing',
      'SQL distance calculations: Full table scan or slow R-tree index (100-500ms)',
      'Real-time matching requires <30 sec, Redis meets requirement',
      'Geospatial indexing is mandatory for location-based services at scale',
    ],
  },
  {
    id: 'q2',
    question:
      'Design the driver matching algorithm. A rider requests a ride, and 50 drivers are within 5 km. Walk through the matching process, handling driver declines and timeouts.',
    sampleAnswer:
      'MATCHING ALGORITHM: (1) FIND CANDIDATES: GEORADIUS returns 50 drivers sorted by distance. Filter: Only available status, correct vehicle type (uberX), rating >4.5. Result: 20 eligible drivers. (2) RANK DRIVERS: Sort by: Distance (weight 0.5), acceptance rate (weight 0.3), rating (weight 0.2). Top driver: driver_123 (0.5 km, 90% acceptance, 4.9 rating). (3) SEND REQUEST: Push notification to driver_123 via WebSocket: "New ride: $15 fare, 0.5 km away, pickup in 3 min". Start 15-second countdown timer. (4) AWAIT RESPONSE: Case A - ACCEPT: Driver taps accept within 15 sec. Match complete, notify rider: "John accepted, arriving in 4 min". Case B - DECLINE: Driver taps decline. Immediately try next driver (driver_456), repeat step 3. Case C - TIMEOUT: No response after 15 sec. Assume driver unavailable (phone off, bad network). Try next driver. (5) FALLBACK: After trying 5 drivers with no acceptance: Expand radius to 10 km (more drivers), increase surge multiplier (1.0x → 1.5x to incentivize), broadcast to multiple drivers simultaneously (first to accept wins). (6) NO MATCH: After 2 minutes and 20 drivers: "No drivers available, try again". Log event for supply analysis. CONCURRENCY HANDLING: Use distributed lock (Redis): SETNX ride_request:123:locked 1 EX 15. Prevents same driver receiving multiple requests. After accept/decline, release lock. OPTIMIZATION: Predictive accept probability model. If driver has 95% acceptance rate and is 0.3 km away, high confidence - send request first. Skip drivers with <50% acceptance (likely to decline). KEY INSIGHT: Matching is iterative with timeouts, not batch auction. Timeouts critical for responsiveness (<30 sec total matching time).',
    keyPoints: [
      'Find 50 candidates, filter to 20 eligible, rank by distance+acceptance+rating',
      'Iterative matching: Try closest driver, 15-sec timeout, try next on decline',
      'Fallback: Expand radius, increase surge, broadcast to multiple',
      'Distributed lock prevents double-booking',
      'Optimize: Skip low-acceptance-rate drivers',
    ],
  },
  {
    id: 'q3',
    question:
      "Explain Uber's surge pricing algorithm. How does it balance supply and demand? Discuss the calculation, implementation with geospatial zones, and impact on matching.",
    sampleAnswer:
      'SURGE PRICING ALGORITHM: (1) CALCULATION: surge_multiplier = max(1.0, demand / supply). Demand = ride requests in area (last 5 min). Supply = available drivers in area. Example: 100 requests, 50 drivers → 100/50 = 2.0x surge. 20 requests, 100 drivers → 20/100 = 0.2 → 1.0x (no surge). (2) GEOSPATIAL ZONES: Divide city into H3 hexagon grid (each hex ~2 km across). Calculate surge independently per hex. Downtown Friday 6 PM: High demand, few drivers → 2.5x surge. Suburb midday: Low demand, many drivers → 1.0x (no surge). (3) IMPLEMENTATION: Background job runs every 1 minute: For each hex: Count ride_requests (Redis: ZCOUNT with geohash range), count available_drivers (Redis: GEORADIUS_COUNT). Calculate ratio, update surge_zones:hex_id. (4) DISPLAY TO USERS: Rider app shows surge heat map (red = high surge). Driver app shows surge zones: "Earn 2x in downtown area". (5) IMPACT ON MATCHING: Higher fare incentivizes drivers to move to high-surge areas. More supply → Lower surge. Riders see "Surge pricing in effect: 2.0x" before requesting. Can choose to wait or pay surge. (6) DYNAMIC ADJUSTMENT: Surge increases → More drivers go online (opportunity to earn more) + Some riders cancel (too expensive). This balances supply/demand within 10-15 minutes. (7) MAX CAP: Uber caps surge at 3.0x to avoid price gouging perception. Hurricane/disaster: Higher caps may apply. BENEFITS: Without surge: 100 requests, 50 drivers → 50 requests unmatched (long wait times). With surge: Price increases → 20 riders cancel + 30 new drivers come online → 80 requests, 80 drivers → All matched. Surge reduces wait time for willing-to-pay riders. CONTROVERSY: Criticized as "price gouging" during emergencies. Uber PR: "Surge brings more drivers, reduces wait times". KEY INSIGHT: Dynamic pricing solves marketplace imbalance - same principle as airline ticket pricing, hotel pricing.',
    keyPoints: [
      'Surge = demand / supply ratio per geospatial hex zone',
      'Calculated every minute based on requests and available drivers',
      'Incentivizes drivers to high-demand areas (balances marketplace)',
      'Dynamic: Higher surge → more supply + less demand → surge drops',
      'Caps at 3.0x to avoid perception of price gouging',
    ],
  },
];
