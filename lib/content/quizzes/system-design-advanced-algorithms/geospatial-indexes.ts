/**
 * Quiz questions for Geospatial Indexes section
 */

export const geospatialindexesQuiz = [
  {
    id: 'q1',
    question:
      'Explain how geohashing works: how does it encode latitude and longitude into a string, and why do nearby locations share prefixes?',
    sampleAnswer:
      'GEOHASHING PROCESS: (1) BINARY ENCODING: Recursively narrow down lat/lon ranges. Start with lat range [-90,90]. If target ≥ midpoint: append 1, take upper half. If target < midpoint: append 0, take lower half. Repeat for N bits. Example: SF lat 37.7749°. Is 37.7749 ≥ 0? YES→1, range [0,90]. Is 37.7749 ≥ 45? NO→0, range [0,45]. Continue: lat bits "10110...". Do same for longitude: lon bits "01001...". (2) INTERLEAVING: Alternate lat and lon bits. Lat:1 0 1 1 0. Lon:0 1 0 0 1. Interleaved: 01 10 01 10 01. (3) BASE-32 ENCODING: Convert bit pairs to base-32 characters. Result: "9q8yyk" for SF. WHY NEARBY LOCATIONS SHARE PREFIXES: Geohashing divides world into grid. First character: coarse (continent level). Each additional character: 32x refinement (subdivide into 32 smaller squares). "9q8yy" = specific ~2km neighborhood in SF. All locations in that neighborhood share "9q8yy" prefix. This enables efficient database queries: WHERE geohash LIKE "9q8yy%" uses B-tree index, returns all locations in that square. CAVEAT: At square boundaries, nearby locations have different prefixes (must check adjacent cells). This is why Uber checks adjacent geohashes when expanding search radius.',
    keyPoints: [
      'Binary encoding: recursively narrow lat/lon ranges (0/1 bits)',
      'Interleave lat and lon bits alternately',
      'Convert to base-32 string for compact representation',
      'Nearby locations in same grid square share prefixes',
      'Enables prefix matching with B-tree indexes (fast DB queries)',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare Quadtree, R-Tree, and Geohash for proximity queries. When would you choose each approach?',
    sampleAnswer:
      'QUADTREE: Recursively divides 2D space into 4 quadrants. Dense areas→more subdivisions. Sparse areas→fewer levels. Query: traverse tree, prune quadrants outside search radius. PROS: Dynamic, good for in-memory, adapts to density. CONS: Imbalanced for skewed data, not great for disk storage. USE WHEN: In-memory index, point queries, dynamic dataset (frequent inserts/deletes). Example: Game engines (objects in map regions). R-TREE: Hierarchical bounding boxes (rectangles). Minimizes overlap between siblings. Query: traverse tree, check if bbox intersects search area. PROS: Disk-friendly, handles ranges/polygons, used by PostGIS. CONS: More complex, rebalancing on inserts. USE WHEN: Disk-based database, range queries, polygon searches, irregular shapes. Example: PostGIS (spatial queries in PostgreSQL). GEOHASH: Encode lat/lon as string, use B-tree index. Query: prefix match + check adjacent cells. PROS: Simple, works with existing B-tree indexes, no custom data structure. CONS: Grid squares ≠ circles (must check neighbors), less optimal at boundaries. USE WHEN: Database with B-tree support, point queries, uniform grid sufficient. Example: MongoDB 2dsphere, Redis GEOADD, Uber driver matching. DECISION TREE: In-memory + dynamic→Quadtree. Database + rectangles→R-tree (PostGIS). Database + simple→Geohash (MongoDB/Redis). For Uber: Geohash is perfect. Fast lookups, works with Redis, acceptable boundary errors (just check 1-2 adjacent cells).',
    keyPoints: [
      'Quadtree: in-memory, dynamic, point queries, adapts to density',
      'R-Tree: disk-based, ranges/polygons, PostGIS, optimal for complex shapes',
      'Geohash: B-tree compatible, simple, uniform grid, Uber/MongoDB',
      'Choose based on: in-memory vs disk, points vs ranges, DB support',
      'Production: PostGIS→R-tree, MongoDB/Redis→Geohash, Games→Quadtree',
    ],
  },
  {
    id: 'q3',
    question:
      'Design the geospatial architecture for Uber driver matching. How do you efficiently match riders with nearby drivers at scale?',
    sampleAnswer:
      'UBER ARCHITECTURE: (1) DATA STORAGE: Redis with geospatial commands. Store driver locations: GEOADD drivers -122.4194 37.7749 driver:123. Redis uses geohash internally with sorted set (fast range queries). (2) RIDER REQUESTS RIDE: Compute rider geohash (precision=6, ~600m squares). geohash(37.7749, -122.4194, 6) = "9q8yyk". (3) QUERY DRIVERS: Query drivers with matching 6-char prefix: GEORADIUS drivers -122.4194 37.7749 2 km WITHDIST. Redis returns drivers in 2km radius with distances. (4) FALLBACK: If <10 drivers found, expand to adjacent geohash cells (8 neighbors). Check neighbors: "9q8yyj", "9q8yyn", etc. (5) MATCHING: Sort drivers by distance, check availability. Assign closest available driver. Send notification via push. (6) UPDATES: Driver locations updated every 5 seconds: GEOADD drivers -122.4200 37.7750 driver:123. SCALE: 1M drivers in SF. Geohash precision=6 creates ~millions of cells. Average ~10 drivers per cell. Query: O(log N) via sorted set, returns ~10-50 drivers in milliseconds. WHY THIS WORKS: Most matches within 600m square (same geohash). Rare cases expand to neighbors (8 more cells). Avoids scanning all 1M drivers. Alternative (naive): Check distance to all 1M drivers (too slow). Alternative (Quadtree): In-memory only, harder to distribute. Geohash + Redis: Fast, distributed, production-proven.',
    keyPoints: [
      'Redis GEOADD/GEORADIUS with geohash-based sorted sets',
      'Rider geohash precision=6 (~600m squares)',
      'Query matching geohash prefix + adjacent cells if needed',
      'O(log N) via sorted set, returns 10-50 candidates in milliseconds',
      'Scales to millions of drivers, avoids scanning all drivers',
    ],
  },
];
