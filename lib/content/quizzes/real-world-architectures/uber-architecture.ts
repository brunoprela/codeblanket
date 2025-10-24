/**
 * Quiz questions for Uber Architecture section
 */

export const uberarchitectureQuiz = [
    {
        id: 'q1',
        question: 'Explain how Uber uses H3 geospatial indexing to find nearby drivers efficiently. Why is this better than calculating distance to all drivers?',
        sampleAnswer: 'H3 divides Earth into hexagonal cells at multiple resolutions. For "find drivers near restaurant": (1) Convert restaurant location to H3 hex ID at resolution 8 (~500m per hex). (2) Get k-ring of neighboring hexes (k=4 covers ~2km radius = 61 hexes). (3) Query Redis for drivers in these hexes (O(61) lookups). (4) Calculate exact distance only to these drivers (~100-500 typically). Instead of checking 1M drivers in city (O(n)), check only drivers in nearby hexes (O(k)). Benefits: Fast queries (<100ms), scalable to millions of drivers, works globally. Why hexagons: Equal distance from center to all edges (unlike squares), better approximation of circular range. Alternative (naive): Calculate distance to all drivers = O(n) = too slow. H3 reduces search space by 1000x.',
        keyPoints: [
            'H3 hexagonal grid: Convert location to hex ID, query neighboring hexes',
            'Query only drivers in nearby hexes (~100-500) instead of all drivers (1M+)',
            'O(k) complexity instead of O(n), sub-100ms latency',
            'Hexagons better than squares: equal edge distances, circular approximation',
        ],
    },
    {
        id: 'q2',
        question: 'Describe Uber\'s dispatch algorithm (DISCO). How does it optimize driver-order matching beyond simple "nearest driver" assignment?',
        sampleAnswer: 'DISCO (Dispatch Optimization) uses batch matching every 30 seconds. Collect pending orders and available drivers. For each order, find nearby drivers (H3 index). Calculate score for each (order, driver) pair considering: (1) Estimated pickup time and delivery time. (2) Driver acceptance probability (some drivers decline far orders). (3) Driver quality (rating, completion rate). (4) Future demand prediction (don\'t assign driver if better order coming). (5) Stacked delivery opportunities (can driver handle 2 orders?). Formulate as optimization problem: minimize total delivery time subject to constraints (driver capacity ≤2, on-time delivery). Solve with Linear Programming or greedy heuristics. Simple nearest-driver is suboptimal globally - driver might be better for different order, doesn\'t consider future demand. DISCO balances immediate assignment with global optimization.',
        keyPoints: [
            'Batch matching every 30 seconds (not instant greedy assignment)',
            'Considers: pickup time, acceptance probability, driver quality, future demand',
            'Optimization problem: minimize total delivery time with constraints',
            'Enables stacked deliveries (one driver, two orders)',
        ],
    },
    {
        id: 'q3',
        question: 'How does Uber track driver locations in real-time at scale, and how is this data stored and queried?',
        sampleAnswer: 'Real-time location tracking: (1) Dasher app sends GPS coordinates every 4 seconds via WebSocket. (2) Location Gateway receives updates, publishes to Kafka (location_updates topic). (3) Flink processes stream in real-time. (4) Update Redis for current location: dasher:location:{id} with TTL 60s (auto-expires if no heartbeat). (5) Convert to H3 hex ID, update Redis set: hex:{id} → dasher_ids. (6) Write to Cassandra for history (partition by dasher_id, cluster by timestamp). Query "nearby drivers": Convert restaurant location to H3 hex, get k-ring neighbors, query Redis sets for dasher_ids, filter by exact distance. Result: Sub-100ms queries despite millions of drivers. Storage: Redis (current, low latency), Cassandra (history, analytics). Scale: Billions of location updates daily.',
        keyPoints: [
            'GPS updates every 4 seconds via WebSocket to Location Gateway',
            'Kafka + Flink for stream processing',
            'Redis for current location (TTL 60s) and H3 hex indexing',
            'Cassandra for historical tracking (analytics, dispute resolution)',
        ],
    },
];

