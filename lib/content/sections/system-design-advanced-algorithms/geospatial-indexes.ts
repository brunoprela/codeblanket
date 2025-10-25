/**
 * Geospatial Indexes Section - Comprehensive content
 */

export const geospatialindexesSection = {
  id: 'geospatial-indexes',
  title: 'Geospatial Indexes',
  content: `Geospatial indexes solve a fundamental problem: **"How do we efficiently find nearby locations in massive datasets?"**

This powers:
- "Find restaurants within 2 miles" (Yelp, Google Maps)
- "Match riders with nearby drivers" (Uber, Lyft)
- "Find friends nearby" (Facebook, Snapchat)
- "Show properties in this map view" (Airbnb, Zillow)

## The Problem with Naive Search

\`\`\`python
# Naive: Check distance to every location
def find_nearby (target_lat, target_lon, radius, all_locations):
    nearby = []
    for loc in all_locations:
        distance = calculate_distance (target_lat, target_lon, 
                                     loc.lat, loc.lon)
        if distance <= radius:
            nearby.append (loc)
    return nearby

# For 1 million locations: 1 million distance calculations!
\`\`\`

**Problems**:
- ❌ O(N) time complexity (slow for large datasets)
- ❌ Computes distance for every location (expensive)
- ❌ Doesn't scale (1 billion locations = 1 billion calculations)

**Geospatial index solution**: O(log N) queries using spatial partitioning.

---

## Key Geospatial Data Structures

### 1. Quadtree

**Recursively divides 2D space into 4 quadrants.**

\`\`\`
World map:
+-------------+
|  NW  |  NE  |
|------|------|
|  SW  |  SE  |
+-------------+

If a quadrant has too many points, subdivide:
    NW quadrant:
    +-------+
    |NW|NE |
    |--|---|
    |SW|SE |
    +-------+
\`\`\`

**Properties**:
- Each node has 4 children (or 0 if leaf)
- Dynamically subdivides dense areas
- Sparse areas: fewer subdivisions
- Height: O(log N) on average

**Use cases**: Point queries (find exact location)

### 2. R-Tree

**Hierarchical bounding boxes containing locations.**

\`\`\`
Root: Bounding box covering entire map
  ├─ Child 1: Box covering California
  │   ├─ Box covering LA
  │   └─ Box covering SF
  └─ Child 2: Box covering Texas
      ├─ Box covering Houston
      └─ Box covering Dallas
\`\`\`

**Properties**:
- Each node: bounding box (min_lat, max_lat, min_lon, max_lon)
- Leaf nodes: actual locations
- Minimizes overlap between siblings
- Height: O(log N)

**Use cases**: Range queries (rectangles), spatial joins

### 3. Geohash

**Encodes lat/lon as a string by interleaving bits.**

\`\`\`
Location: San Francisco (37.7749° N, 122.4194° W)
Geohash: "9q8yyk"

Precision:
- 1 char: ±2500 km
- 5 chars: ±2.4 km
- 9 chars: ±4.8 m
\`\`\`

**Properties**:
- Nearby locations share prefixes
- Can use B-tree index (string prefix matching)
- Simple to implement

**Use cases**: Databases with B-tree indexes (PostgreSQL, MongoDB)

---

## Geohash: Deep Dive

### How Geohash Works

**Step 1: Binary encoding of latitude**

\`\`\`
Latitude range: [-90, 90]
Target: 37.7749°

Iteration 1: Is 37.7749 >= 0 (midpoint)? YES → 1
  New range: [0, 90]

Iteration 2: Is 37.7749 >= 45 (midpoint)? NO → 0
  New range: [0, 45]

Iteration 3: Is 37.7749 >= 22.5 (midpoint)? YES → 1
  New range: [22.5, 45]

Continue for N bits...
Latitude bits: 10110...
\`\`\`

**Step 2: Binary encoding of longitude** (same process)

\`\`\`
Longitude range: [-180, 180]
Target: -122.4194°

Longitude bits: 01001...
\`\`\`

**Step 3: Interleave bits**

\`\`\`
Lat bits:  1 0 1 1 0...
Lon bits:  0 1 0 0 1...

Interleaved: 01 10 01 10 01...
\`\`\`

**Step 4: Convert to base-32**

\`\`\`
01101010 01110001 10110...
→ Base-32: "9q8yyk"
\`\`\`

### Geohash Precision

| Chars | Lat Precision | Lon Precision | Example Use Case |
|-------|---------------|---------------|------------------|
| 1 | ±23° | ±23° | Continent |
| 3 | ±0.7° | ±0.7° | City |
| 5 | ±0.02° | ±0.02° | Neighborhood (~2 km) |
| 7 | ±0.0007° | ±0.0007° | Street (~70 m) |
| 9 | ±0.00002° | ±0.00002° | House (~5 m) |

### Implementation

\`\`\`python
BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"

def encode_geohash (lat, lon, precision=9):
    lat_range, lon_range = [-90, 90], [-180, 180]
    geohash = []
    bits = []
    bit = 0
    ch = 0
    
    while len (geohash) < precision:
        if bit % 2 == 0:  # Even bits: longitude
            mid = (lon_range[0] + lon_range[1]) / 2
            if lon > mid:
                ch |= (1 << (4 - (bit // 2) % 5))
                lon_range[0] = mid
            else:
                lon_range[1] = mid
        else:  # Odd bits: latitude
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat > mid:
                ch |= (1 << (4 - (bit // 2) % 5))
                lat_range[0] = mid
            else:
                lat_range[1] = mid
        
        bit += 1
        if bit % 10 == 0:
            geohash.append(BASE32[ch])
            ch = 0
    
    return ''.join (geohash)

# Usage
geohash = encode_geohash(37.7749, -122.4194, precision=7)
print(geohash)  # "9q8yyk8"
\`\`\`

### Querying with Geohash

\`\`\`sql
-- PostgreSQL with B-tree index on geohash column
CREATE INDEX idx_geohash ON locations (geohash);

-- Find locations within 2km of San Francisco
-- SF geohash (5 chars): "9q8yy"

SELECT * FROM locations
WHERE geohash LIKE '9q8yy%';

-- Scans only matching prefix (very fast!)
\`\`\`

**Caveat**: Geohash squares don't perfectly match circles. Need to check adjacent cells at boundaries.

---

## R-Tree: Deep Dive

### Structure

\`\`\`
Internal node:
{
  bbox: {min_lat, max_lat, min_lon, max_lon},
  children: [child1, child2, ...]
}

Leaf node:
{
  location: {lat, lon, data}
}
\`\`\`

### Insertion Algorithm

\`\`\`python
def insert (root, location):
    # 1. Find leaf node with minimal bbox expansion
    node = choose_leaf (root, location)
    
    # 2. Add location to leaf
    node.entries.append (location)
    
    # 3. If overflow, split node
    if len (node.entries) > MAX_ENTRIES:
        split_node (node)
\`\`\`

### Range Query

\`\`\`python
def range_query (node, search_bbox):
    results = []
    
    if is_leaf (node):
        # Check each location
        for loc in node.entries:
            if bbox_contains (search_bbox, loc):
                results.append (loc)
    else:
        # Recursively search children
        for child in node.children:
            if bbox_intersects (search_bbox, child.bbox):
                results.extend (range_query (child, search_bbox))
    
    return results
\`\`\`

**Time complexity**: O(log N) average, O(N) worst case

---

## Quadtree: Deep Dive

### Structure

\`\`\`python
class QuadTreeNode:
    def __init__(self, bbox, capacity=4):
        self.bbox = bbox  # (min_lat, max_lat, min_lon, max_lon)
        self.capacity = capacity
        self.points = []
        self.divided = False
        self.nw = self.ne = self.sw = self.se = None
\`\`\`

### Insertion

\`\`\`python
def insert (self, point):
    # Outside boundary
    if not self.bbox.contains (point):
        return False
    
    # Space available
    if len (self.points) < self.capacity and not self.divided:
        self.points.append (point)
        return True
    
    # Need to subdivide
    if not self.divided:
        self.subdivide()
    
    # Recursively insert into quadrants
    return (self.nw.insert (point) or self.ne.insert (point) or
            self.sw.insert (point) or self.se.insert (point))

def subdivide (self):
    # Create 4 children (NW, NE, SW, SE quadrants)
    mid_lat = (self.bbox.min_lat + self.bbox.max_lat) / 2
    mid_lon = (self.bbox.min_lon + self.bbox.max_lon) / 2
    
    self.nw = QuadTreeNode(BBox (mid_lat, self.bbox.max_lat,
                                 self.bbox.min_lon, mid_lon))
    # ... create NE, SW, SE similarly
    
    self.divided = True
\`\`\`

### Range Query

\`\`\`python
def query_range (self, search_bbox):
    results = []
    
    # No intersection
    if not self.bbox.intersects (search_bbox):
        return results
    
    # Check points in this node
    for point in self.points:
        if search_bbox.contains (point):
            results.append (point)
    
    # Recursively check children
    if self.divided:
        results.extend (self.nw.query_range (search_bbox))
        results.extend (self.ne.query_range (search_bbox))
        results.extend (self.sw.query_range (search_bbox))
        results.extend (self.se.query_range (search_bbox))
    
    return results
\`\`\`

---

## Real-World Systems

### 1. Uber (Geohash + Quadtree)

**Driver matching**:
1. Rider requests ride at location (lat, lon)
2. Compute rider geohash (precision=6, ~600m squares)
3. Query drivers with matching geohash prefix
4. If < 10 drivers, expand to adjacent geohashes
5. Calculate exact distances, sort, assign closest driver

**Why this works**: Most matches within same geohash square. Rare expansion to neighbors.

### 2. MongoDB (Geospatial Queries)

\`\`\`javascript
// 2dsphere index (geospatial)
db.places.createIndex({ location: "2dsphere" })

// Find within radius
db.places.find({
  location: {
    $near: {
      $geometry: { type: "Point", coordinates: [-122.4194, 37.7749] },
      $maxDistance: 2000  // 2000 meters
    }
  }
})
\`\`\`

**Internally**: Uses geohash-like encoding with B-tree index.

### 3. PostgreSQL (PostGIS with R-Tree)

\`\`\`sql
-- Create spatial index (R-Tree variant: GiST)
CREATE INDEX idx_location ON places USING GIST(location);

-- Find within 2km
SELECT * FROM places
WHERE ST_DWithin(
  location,
  ST_MakePoint(-122.4194, 37.7749)::geography,
  2000  -- meters
);
\`\`\`

**GiST index**: Generalized Search Tree, implements R-Tree for geospatial.

### 4. Redis (Geospatial Commands with Sorted Sets)

\`\`\`redis
GEOADD drivers -122.4194 37.7749 driver:123
GEOADD drivers -122.4100 37.7800 driver:456

GEORADIUS drivers -122.4194 37.7749 2 km WITHDIST
# Returns drivers within 2km with distances
\`\`\`

**Internally**: Geohash stored in sorted set, enables range queries.

### 5. Google S2 (Advanced Geospatial Library)

**Used by**: Google Maps, Uber (H3)

**Key innovation**: Maps sphere to cube faces, hierarchical cells.

**Benefits**:
- Handles spherical geometry correctly
- Uniform cell sizes
- Fast neighbor finding

---

## Comparison

| Structure | Query Time | Insert Time | Space | Best For |
|-----------|------------|-------------|-------|----------|
| **Quadtree** | O(log N) avg | O(log N) | O(N) | In-memory, dynamic |
| **R-Tree** | O(log N) avg | O(log N) | O(N) | Ranges, polygons |
| **Geohash** | O(log N) | O(log N) | O(N) | DB with B-tree |
| **Grid** | O(1) | O(1) | O(cells) | Uniform density |

---

## Interview Tips

### Key Talking Points

1. **Problem**: Find nearby locations efficiently (O(log N) vs O(N))
2. **Options**: Quadtree (in-memory), R-tree (ranges), Geohash (DB index)
3. **Geohash**: String prefix matching, works with B-tree
4. **Real-world**: Uber (geohash), MongoDB (2dsphere), PostGIS (R-tree)
5. **Trade-offs**: Exact distance vs approximate grid

### Common Questions

**"How does Uber match riders with drivers?"**
- Rider requests ride at (lat, lon)
- Compute geohash (precision=6, ~600m)
- Query drivers in same geohash
- Expand to adjacent cells if needed
- Sort by exact distance, assign closest

**"Why use geohash instead of lat/lon range queries?"**
- Geohash: Single B-tree index scan (fast)
- Lat/lon: Two range conditions (slower, multiple indexes)
- Prefix matching natural for nearby locations

**"What are limitations of geohash?"**
- Square grid ≠ circular radius (edge cases)
- Must check adjacent cells at boundaries
- Not optimal for very large or very small areas

### Design Exercise: Design Yelp "Find Nearby Restaurants"

\`\`\`
Requirement: Find restaurants within 2 miles of user location

Solution:
1. Database: PostgreSQL with PostGIS
2. Schema: restaurants (id, name, location GEOGRAPHY)
3. Index: CREATE INDEX USING GIST(location)
4. Query: ST_DWithin (location, user_point, 2 miles)

Scale: 1M restaurants
- R-tree depth: log(1M) ≈ 20
- Query time: 20 node checks → milliseconds

Alternative (Geohash):
1. Add geohash column (precision=7, ~70m)
2. Index: CREATE INDEX ON restaurants (geohash)
3. Query: geohash LIKE 'prefix%'
4. Check adjacent geohashes at boundaries
\`\`\`

---

## Summary

**Geospatial indexes** enable efficient proximity queries on location data.

**Key structures**:
- ✅ **Quadtree**: In-memory, dynamic, point queries
- ✅ **R-Tree**: Disk-based, ranges, polygons
- ✅ **Geohash**: String encoding, B-tree compatible

**Industry adoption**: Uber, MongoDB, PostgreSQL/PostGIS, Redis, Google Maps

Understanding geospatial indexes is **essential** for location-based services and proximity matching.`,
};
