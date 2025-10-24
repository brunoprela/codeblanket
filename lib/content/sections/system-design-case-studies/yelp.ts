/**
 * Design Yelp Section
 */

export const yelpSection = {
    id: 'yelp',
    title: 'Design Yelp (Nearby Places)',
    content: `Yelp is a local business discovery platform where users search for nearby restaurants, shops, and services based on location, ratings, and preferences. The core challenges are: efficient geospatial queries (find businesses within radius), handling massive read traffic (100M+ searches/day), aggregating millions of reviews in real-time, ranking results by relevance/quality/distance, and maintaining data freshness as businesses update hours/menus.

## Problem Statement

Design Yelp with:
- **Proximity Search**: Find restaurants/businesses within X miles of current location
- **Advanced Filters**: Cuisine type, price range ($-$$$$), rating (4+ stars), "open now", delivery/takeout
- **Reviews & Ratings**: Users write detailed reviews with photos, 1-5 star ratings
- **Business Pages**: Display info (photos, hours, menu, contact, parking, amenities)
- **Check-ins**: Track user visits, award badges, verify reviewers
- **Collections**: Save favorite places to lists ("Date Night Spots", "Best Pizza")
- **Search Ranking**: Balance relevance, rating, popularity, distance, and freshness
- **Real-Time Updates**: Business hours change, new reviews appear immediately

**Scale**: 200 million businesses globally, 500 million DAU, 100 million searches/day, 500 million reviews, 50 million photos

---

## Step 1: Requirements Gathering

### Functional Requirements

1. **Location-Based Search**: "Find pizza within 5 miles of me"
2. **Text Search**: Match business name, cuisine, category
3. **Filters**: Distance, rating, price, open now, delivery, parking
4. **Business Details**: Photos, menu, hours, contact, reviews
5. **Write Reviews**: Text + photos + 1-5 star rating
6. **User Actions**: Check-in, save to collection, share
7. **Business Owner Features**: Update hours, respond to reviews, add photos
8. **Trending/Popular**: "Trending restaurants near you"

### Non-Functional Requirements

1. **Low Latency**: Search results < 200ms (p95)
2. **High Availability**: 99.95% uptime
3. **Scalability**: Handle 100M searches/day, 10M QPS peak
4. **Accuracy**: Geospatial queries accurate within 0.1% error
5. **Freshness**: New reviews appear within 1 minute
6. **Fault Tolerance**: Graceful degradation if search service down

---

## Step 2: Capacity Estimation

### Traffic

**Daily Active Users (DAU)**: 500 million

**Searches**:
- 100 million searches/day
- Average: 1,157 searches/second
- Peak (lunch/dinner): 10K searches/second

**Writes**:
- 10 million reviews/day = 115 reviews/second
- 5 million photo uploads/day = 58 photos/second
- 1 million business updates/day = 12 updates/second

### Storage

**Businesses**:
- 200 million businesses × 5 KB metadata = 1 TB

**Reviews**:
- 500 million reviews × 2 KB (text + metadata) = 1 TB
- 50 million photos × 500 KB = 25 TB

**Total**: ~27 TB (structured data) + photos (growing 9 TB/year)

### Bandwidth

**Search Results**:
- 10K searches/sec × 50 KB (20 results × 2.5 KB) = 500 MB/sec = 43 TB/day

**Photo Serving**:
- 500M DAU × 10 photo views/day × 100 KB (thumbnail) = 500 TB/day

---

## Step 3: Geospatial Indexing (Core Challenge)

### Problem: Efficient Proximity Queries

**Naive Approach** (Too Slow):

\`\`\`sql
-- Find restaurants within 5 miles of (37.7749, -122.4194)
SELECT * FROM businesses
WHERE category = 'restaurant'
  AND SQRT(POW(latitude - 37.7749, 2) + POW(longitude - 122.4194, 2)) * 69 < 5
ORDER BY distance
LIMIT 20;

-- Problem: Full table scan (200M rows), calculate distance for each
-- Time: 10+ seconds (unacceptable)
\`\`\`

**Haversine Distance Formula** (Accurate):

\`\`\`python
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    distance = R * c  # Distance in km
    return distance

# Example: San Francisco to Palo Alto
distance = haversine_distance(37.7749, -122.4194, 37.4419, -122.1430)
# Result: ~44 km
\`\`\`

---

## Step 4: Solution 1 - Geohash

**Concept**: Encode lat/lon into a base32 string. Nearby locations share prefixes.

**How Geohash Works**:

\`\`\`
1. Start with world map: latitude [-90, 90], longitude [-180, 180]
2. Divide longitude into 2 halves: West [-180, 0] = 0, East [0, 180] = 1
3. Divide latitude into 2 halves: South [-90, 0] = 0, North [0, 90] = 1
4. Interleave bits: lon, lat, lon, lat, ...
5. Convert binary to base32: 00000 = '0', 00001 = '1', ...

Example: (37.7749, -122.4194) San Francisco
Binary: 01001110101000100110...
Base32: 9q8yy9mf2s7w

Precision:
- 4 chars (9q8y): ±20 km
- 5 chars (9q8yy): ±2.4 km
- 6 chars (9q8yy9): ±0.61 km
- 7 chars (9q8yy9m): ±0.076 km (76 meters)
\`\`\`

**Database Schema with Geohash**:

\`\`\`sql
CREATE TABLE businesses (
    business_id BIGINT PRIMARY KEY,
    name VARCHAR(200),
    category VARCHAR(50),  -- restaurant, cafe, shop
    cuisine VARCHAR(50),   -- Italian, Chinese, Mexican
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    geohash_4 CHAR(4),     -- ±20 km precision
    geohash_5 CHAR(5),     -- ±2.4 km precision
    geohash_6 CHAR(6),     -- ±0.61 km precision
    geohash_7 CHAR(7),     -- ±76 m precision
    rating DECIMAL(2,1),
    review_count INT,
    price_range INT,       -- 1-4 ($, $$, $$$, $$$$)
    is_open_now BOOLEAN,
    INDEX idx_geohash_6_category (geohash_6, category),
    INDEX idx_geohash_5_rating (geohash_5, rating)
);
\`\`\`

**Query with Geohash**:

\`\`\`python
def search_nearby(lat, lon, radius_km, category):
    # Calculate geohash precision based on radius
    if radius_km <= 1:
        precision = 7  # 76m
    elif radius_km <= 5:
        precision = 6  # 610m
    elif radius_km <= 20:
        precision = 5  # 2.4 km
    else:
        precision = 4  # 20 km
    
    # Encode user location
    user_geohash = encode_geohash(lat, lon, precision)  # e.g., "9q8yy9"
    
    # Get neighboring geohash cells (to cover entire radius)
    neighbors = get_neighbors(user_geohash)  # Returns ["9q8yy9", "9q8yy8", "9q8yyd", ...]
    
    # Query database
    results = db.query("""
        SELECT * FROM businesses
        WHERE geohash_{precision} IN ({neighbors})
          AND category = '{category}'
        ORDER BY rating DESC, review_count DESC
        LIMIT 100
    """.format(precision=precision, neighbors=','.join(neighbors), category=category))
    
    # Filter by exact distance (some results outside radius due to geohash approximation)
    filtered = []
    for business in results:
        distance = haversine_distance(lat, lon, business.latitude, business.longitude)
        if distance <= radius_km:
            business.distance = distance
            filtered.append(business)
    
    # Sort by distance (or ranking score)
    filtered.sort(key=lambda b: b.distance)
    return filtered[:20]

# Example: Find pizza within 5 km of San Francisco
results = search_nearby(37.7749, -122.4194, radius_km=5, category='restaurant')
\`\`\`

**Geohash Neighbors** (Edge Handling):

\`\`\`
If user is at edge of geohash cell, businesses in adjacent cells might be closer.
Solution: Query current cell + 8 neighbors (North, South, East, West, NE, NW, SE, SW).

       ┌─────┬─────┬─────┐
       │ NW  │  N  │ NE  │
       ├─────┼─────┼─────┤
       │  W  │ YOU │  E  │
       ├─────┼─────┼─────┤
       │ SW  │  S  │ SE  │
       └─────┴─────┴─────┘

Geohash library provides: geohash.neighbors('9q8yy9')
\`\`\`

**Pros**:
- ✅ Simple to implement (standard libraries exist)
- ✅ Database-agnostic (works with PostgreSQL, MySQL)
- ✅ Fast queries (indexed prefix search)

**Cons**:
- ❌ Edge cases require querying neighbors (9 cells)
- ❌ Geohash cells are rectangular (not circular search radius)
- ❌ Some false positives require distance filtering

---

## Step 5: Solution 2 - PostgreSQL PostGIS

**PostGIS**: PostgreSQL extension for geospatial data.

**Schema**:

\`\`\`sql
CREATE EXTENSION postgis;

CREATE TABLE businesses (
    business_id BIGINT PRIMARY KEY,
    name VARCHAR(200),
    category VARCHAR(50),
    location GEOGRAPHY(POINT, 4326),  -- lat/lon with Earth projection
    rating DECIMAL(2,1),
    review_count INT
);

-- Spatial index (R-tree)
CREATE INDEX idx_location ON businesses USING GIST(location);
\`\`\`

**Query**:

\`\`\`sql
-- Find restaurants within 5 km
SELECT
    business_id,
    name,
    ST_Distance(location, ST_MakePoint(-122.4194, 37.7749)::geography) AS distance
FROM businesses
WHERE category = 'restaurant'
  AND ST_DWithin(
    location,
    ST_MakePoint(-122.4194, 37.7749)::geography,
    5000  -- 5 km = 5000 meters
  )
ORDER BY distance
LIMIT 20;

-- Query time: 20-50ms (with spatial index)
\`\`\`

**Pros**:
- ✅ Built-in distance calculations (accurate Haversine)
- ✅ No edge cases (true radius search)
- ✅ Spatial indexing (R-tree, fast)

**Cons**:
- ❌ PostgreSQL-specific (vendor lock-in)
- ❌ Complex setup (extension, spatial index)

---

## Step 6: Solution 3 - Redis Geo (Recommended for Hot Data)

**Redis Geo**: In-memory geospatial index using sorted sets.

**Implementation**:

\`\`\`python
import redis

redis_client = redis.Redis(host='localhost', port=6379)

# Add businesses to Redis Geo index
def add_business(category, business_id, lat, lon):
    redis_client.geoadd(
        f"businesses:{category}",
        lon, lat, business_id  # Note: Redis uses (lon, lat) order
    )

# Add 1000 restaurants
add_business('restaurant', 123, 37.7749, -122.4194)
add_business('restaurant', 456, 37.7849, -122.4094)
# ... 998 more

# Search nearby
def search_nearby_redis(lat, lon, radius_km, category):
    results = redis_client.georadius(
        f"businesses:{category}",
        lon, lat,
        radius_km, unit='km',
        withdist=True,  # Return distances
        withcoord=True,  # Return coordinates
        sort='ASC',      # Sort by distance
        count=20
    )
    
    # Results: [(business_id, distance, (lon, lat)), ...]
    return results

# Example
results = search_nearby_redis(37.7749, -122.4194, radius_km=5, category='restaurant')
# Query time: < 10ms (in-memory)
\`\`\`

**When to Use Redis Geo**:
- Hot data: Popular cities (NYC, SF, LA) cached in Redis
- Fast reads: < 10ms response time
- Limited dataset: 10K-100K businesses per category per city (fits in memory)

**Architecture**:

\`\`\`
1. Cache popular cities/categories in Redis Geo
2. Database (PostgreSQL PostGIS) as source of truth
3. Cache miss → Query database → Populate Redis
4. TTL: 1 hour (businesses rarely move)
\`\`\`

---

## Step 7: Full-Text Search (Elasticsearch)

**Problem**: Geospatial alone isn't enough. Need text matching for queries like "best vegan pizza."

**Elasticsearch Index**:

\`\`\`json
PUT /businesses
{
  "mappings": {
    "properties": {
      "business_id": {"type": "long"},
      "name": {
        "type": "text",
        "fields": {"keyword": {"type": "keyword"}}
      },
      "category": {"type": "keyword"},
      "cuisine": {"type": "keyword"},
      "description": {"type": "text"},
      "location": {"type": "geo_point"},  // Lat/lon for geo queries
      "rating": {"type": "float"},
      "review_count": {"type": "integer"},
      "price_range": {"type": "integer"},
      "tags": {"type": "keyword"}  // ["vegan", "gluten-free", "outdoor seating"]
    }
  }
}
\`\`\`

**Complex Query** (Text + Geo + Filters):

\`\`\`json
POST /businesses/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "multi_match": {
            "query": "vegan pizza",
            "fields": ["name^3", "cuisine^2", "description", "tags^2"],
            "type": "best_fields"
          }
        }
      ],
      "filter": [
        {
          "geo_distance": {
            "distance": "5km",
            "location": {"lat": 37.7749, "lon": -122.4194}
          }
        },
        {"term": {"category": "restaurant"}},
        {"range": {"rating": {"gte": 4.0}}},
        {"term": {"price_range": 2}}  // $$
      ]
    }
  },
  "sort": [
    {
      "_geo_distance": {
        "location": {"lat": 37.7749, "lon": -122.4194},
        "order": "asc",
        "unit": "km"
      }
    }
  ],
  "size": 20
}
\`\`\`

**Explanation**:
- **must**: Text relevance score (BM25 algorithm)
- **filter**: Geo + rating + price (no scoring, just yes/no)
- **fields boost**: Name matches (^3) ranked higher than description
- **sort**: By distance (closest first)

**Why Elasticsearch?**:
- ✅ Combined text + geo search
- ✅ Fast full-text search (inverted index)
- ✅ Flexible ranking (custom scoring functions)
- ✅ Faceted search (count by price range, rating buckets)

---

## Step 8: Ranking Algorithm

**Ranking Formula**:

\`\`\`python
def calculate_ranking_score(business, user_location, query):
    # 1. Text Relevance (0-1)
    relevance = elasticsearch_score(business, query)  # BM25 score
    
    # 2. Rating Score (0-1)
    rating_score = business.rating / 5.0  # 4.5 stars → 0.9
    
    # 3. Popularity Score (0-1)
    # Log scale: 10 reviews = 0.5, 100 reviews = 0.67, 1000 reviews = 0.75
    popularity_score = math.log10(business.review_count + 1) / 4.0
    popularity_score = min(popularity_score, 1.0)
    
    # 4. Distance Score (0-1)
    distance_km = haversine_distance(user_location, business.location)
    if distance_km < 1:
        distance_score = 1.0
    elif distance_km < 5:
        distance_score = 1.0 - (distance_km - 1) / 4 * 0.3  # Linear decay
    else:
        distance_score = 0.7 * math.exp(-(distance_km - 5) / 10)  # Exponential decay
    
    # 5. Freshness Score (0-1)
    days_since_update = (now - business.last_updated).days
    freshness_score = math.exp(-days_since_update / 30)  # 30-day half-life
    
    # 6. User Preference (ML Model)
    user_preference = predict_user_affinity(user, business)  # Collaborative filtering: 0-1
    
    # Weighted combination
    score = (
        0.25 * relevance +
        0.20 * rating_score +
        0.15 * popularity_score +
        0.15 * distance_score +
        0.10 * freshness_score +
        0.15 * user_preference
    )
    
    return score

# Sort businesses by score
businesses = search_nearby(user_location, query)
ranked = sorted(businesses, key=lambda b: calculate_ranking_score(b, user_location, query), reverse=True)
return ranked[:20]
\`\`\`

**Boosting Factors**:

\`\`\`python
# Boost businesses with certain attributes
if business.has_photos:
    score *= 1.1  # 10% boost
if business.verified:
    score *= 1.05  # 5% boost
if business.claimed_by_owner:
    score *= 1.05
if business.responded_to_reviews:
    score *= 1.03
\`\`\`

---

## Step 9: Reviews & Ratings Aggregation

### Schema

\`\`\`sql
CREATE TABLE reviews (
    review_id BIGINT PRIMARY KEY,
    business_id BIGINT,
    user_id BIGINT,
    rating INT CHECK (rating BETWEEN 1 AND 5),
    text TEXT,
    photos JSON,  -- ["https://s3.../photo1.jpg", ...]
    created_at TIMESTAMP,
    helpful_count INT DEFAULT 0,
    funny_count INT DEFAULT 0,
    cool_count INT DEFAULT 0,
    INDEX idx_business (business_id, created_at),
    INDEX idx_user (user_id, created_at)
);

CREATE TABLE review_votes (
    user_id BIGINT,
    review_id BIGINT,
    vote_type VARCHAR(10),  -- "helpful", "funny", "cool"
    PRIMARY KEY (user_id, review_id, vote_type)
);
\`\`\`

### Real-Time Rating Aggregation

**Problem**: Calculating average rating on every query is expensive.

**Solution**: Pre-compute and cache in Redis.

\`\`\`python
# On new review submission
def add_review(business_id, user_id, rating, text):
    # Store review in database
    review_id = db.insert("reviews", {
        business_id: business_id,
        user_id: user_id,
        rating: rating,
        text: text,
        created_at: now()
    })
    
    # Update Redis aggregation
    redis_client.zadd(f"ratings:{business_id}", {review_id: rating})
    redis_client.incr(f"review_count:{business_id}")
    
    # Calculate new average (async)
    update_business_rating_async(business_id)
    
    return review_id

# Async worker (Celery task)
def update_business_rating_async(business_id):
    # Get all ratings from Redis
    ratings = redis_client.zrange(f"ratings:{business_id}", 0, -1, withscores=True)
    
    if not ratings:
        return
    
    # Calculate average
    total = sum(rating for _, rating in ratings)
    count = len(ratings)
    avg_rating = total / count
    
    # Update database
    db.execute("""
        UPDATE businesses
        SET rating = ?, review_count = ?
        WHERE business_id = ?
    """, avg_rating, count, business_id)
    
    # Update Elasticsearch
    es.update('businesses', business_id, {'rating': avg_rating, 'review_count': count})
\`\`\`

**Why Async?**:
- Don't block review submission (user waits)
- Batch updates (every 5 minutes, not on every review)
- Eventual consistency acceptable (rating lags by minutes)

---

## Step 10: "Open Now" Feature

**Business Hours Schema**:

\`\`\`sql
CREATE TABLE business_hours (
    business_id BIGINT,
    day_of_week INT,  -- 0=Sunday, 1=Monday, ..., 6=Saturday
    open_time TIME,
    close_time TIME,
    PRIMARY KEY (business_id, day_of_week)
);

-- Example: Restaurant open Mon-Fri 11:00-22:00, Sat-Sun 10:00-23:00
INSERT INTO business_hours VALUES
(123, 1, '11:00', '22:00'),  -- Monday
(123, 2, '11:00', '22:00'),  -- Tuesday
...
(123, 6, '10:00', '23:00');  -- Saturday
\`\`\`

**Query Logic**:

\`\`\`python
def is_open_now(business_id):
    now = datetime.now()
    day_of_week = now.weekday()  # 0=Monday, 6=Sunday
    current_time = now.time()
    
    hours = db.query("""
        SELECT open_time, close_time
        FROM business_hours
        WHERE business_id = ? AND day_of_week = ?
    """, business_id, day_of_week)
    
    if not hours:
        return False  # Closed today
    
    open_time, close_time = hours
    
    # Handle overnight hours (e.g., 22:00 - 02:00)
    if close_time < open_time:
        return current_time >= open_time or current_time <= close_time
    else:
        return open_time <= current_time <= close_time

# Filter search results
businesses = search_nearby(location, query)
if filter_open_now:
    businesses = [b for b in businesses if is_open_now(b.business_id)]
\`\`\`

**Optimization**: Pre-compute "is_open_now" boolean flag (updated every 15 minutes).

\`\`\`python
# Cron job: Every 15 minutes
def update_open_now_flags():
    now = datetime.now()
    day_of_week = now.weekday()
    current_time = now.time()
    
    # Bulk update
    db.execute("""
        UPDATE businesses b
        SET is_open_now = EXISTS (
            SELECT 1 FROM business_hours h
            WHERE h.business_id = b.business_id
              AND h.day_of_week = ?
              AND (
                (h.close_time >= h.open_time AND ? BETWEEN h.open_time AND h.close_time)
                OR (h.close_time < h.open_time AND (? >= h.open_time OR ? <= h.close_time))
              )
        )
    """, day_of_week, current_time, current_time, current_time)
\`\`\`

---

## Step 11: High-Level Architecture

\`\`\`
┌──────────────┐
│   Browser    │
│   Mobile App │
└──────┬───────┘
       │ HTTPS
       ▼
┌──────────────┐
│  CloudFlare  │
│  (CDN, DDoS) │
└──────┬───────┘
       │
┌──────▼───────┐
│Load Balancer │
└──────┬───────┘
       │
       ├────────────────────────┬────────────────────────┐
       │                        │                        │
       ▼                        ▼                        ▼
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│Search Service│        │Business      │        │Review Service│
│              │        │Service       │        │              │
└──────┬───────┘        └──────┬───────┘        └──────┬───────┘
       │                       │                        │
       ▼                       ▼                        ▼
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│Elasticsearch │        │ PostgreSQL   │        │  PostgreSQL  │
│(Search Index)│        │(Business DB) │        │  (Reviews DB)│
└──────────────┘        └──────┬───────┘        └──────────────┘
                               │
                        ┌──────▼───────┐
                        │Read Replicas │
                        │(5 replicas)  │
                        └──────────────┘

┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│    Redis     │        │      S3      │        │    Kafka     │
│(Geo Cache,   │        │   (Photos)   │        │(Event Stream)│
│ Ratings)     │        └──────────────┘        └──────────────┘
└──────────────┘
\`\`\`

---

## Step 12: Database Sharding Strategy

**Shard by Geographic Region**:

\`\`\`
Rationale: Most searches are local (users search near their location).

Sharding:
- Shard 1: US West Coast (CA, OR, WA)
- Shard 2: US East Coast (NY, MA, FL)
- Shard 3: US Central (TX, IL, OH)
- Shard 4: Europe
- Shard 5: Asia
- ...

Routing:
- User location: (37.7749, -122.4194) → San Francisco → Shard 1
- Query: "pizza near me" → Routes to Shard 1 only (fast, local)
\`\`\`

**Cross-Shard Queries** (Rare):

\`\`\`
Scenario: User traveling, searches "pizza near airport" while in flight.
- Location: Mid-air between SF and NYC
- Solution: Scatter-gather (query both Shard 1 and Shard 2), merge results
- Performance: 2× slower (acceptable for rare case)
\`\`\`

**Shard Key**:

\`\`\`sql
-- Map business to shard using geohash prefix
shard_id = hash(geohash_2) % num_shards

Example:
- San Francisco: geohash_2 = "9q" → shard_id = hash("9q") % 10 = 3
- New York: geohash_2 = "dr" → shard_id = hash("dr") % 10 = 7
\`\`\`

---

## Step 13: Caching Strategy

### Multi-Level Cache

**Level 1: CDN (CloudFlare)**
- Static assets: CSS, JS, images
- Business photos (immutable, long TTL)

**Level 2: Redis (Application Cache)**
- Search results: Key = \`search:{geohash}:{query}:{filters}\`, TTL = 5 minutes
- Business details: Key = \`business:{id}\`, TTL = 1 hour
- Ratings aggregation: Key = \`ratings:{business_id}\`, no TTL (updated on write)
- Geo index: Redis Geo (popular cities)

**Level 3: Application Memory**
- Hot businesses (top 1000 in each city)
- Category mappings, config

**Cache Invalidation**:

\`\`\`python
# When business updates hours
def update_business_hours(business_id, new_hours):
    db.update("business_hours", business_id, new_hours)
    
    # Invalidate caches
    redis_client.delete(f"business:{business_id}")
    
    # Invalidate search results containing this business (expensive, skip)
    # Instead: Accept stale cache for 5 minutes (TTL)
    
    # Update Elasticsearch
    es.update('businesses', business_id, {'is_open_now': calculate_open_now(new_hours)})
\`\`\`

---

## Step 14: Data Freshness

**Challenge**: Businesses update info (hours, menu, photos). How quickly do changes appear?

**Update Flow**:

\`\`\`
1. BUSINESS OWNER UPDATES HOURS (via Yelp for Business)
   - POST /api/business/123/hours
   - Update database (PostgreSQL)
   - Publish event to Kafka: "business_updated"

2. ASYNC WORKERS CONSUME EVENTS
   - Update Elasticsearch index (1-5 seconds lag)
   - Invalidate Redis cache (immediate)
   - Recompute "is_open_now" flag

3. USER SEARCHES "pizza open now"
   - Elasticsearch query (fresh data, <5 sec lag)
   - Return updated results

Consistency:
- Cache: 5-minute stale data acceptable
- Search: 5-second lag acceptable
- Database: Immediate (source of truth)
\`\`\`

---

## Step 15: Handling High Traffic (Optimizations)

### 1. Query Optimization

**Pagination** (Cursor-Based):

\`\`\`
GET /search?lat=37.7749&lon=-122.4194&query=pizza&cursor=eyJkaXN0YW5jZSI6Mi4zLCJpZCI6MTIzfQ==

Cursor encodes: {"distance": 2.3, "id": 123}
Next page: WHERE distance > 2.3 OR (distance = 2.3 AND id > 123)
\`\`\`

### 2. Autocomplete (Typeahead)

**Trie for Business Names**:

\`\`\`python
# Pre-build Trie of business names
trie = Trie()
trie.insert("Tony's Pizza", business_id=123, score=4.5)
trie.insert("Pizza Hut", business_id=456, score=3.8)

# User types "piz"
suggestions = trie.autocomplete("piz")
# Returns: ["Pizza Hut", "Tony's Pizza"] (sorted by score)
\`\`\`

**Elasticsearch Alternative**:

\`\`\`json
POST /businesses/_search
{
  "suggest": {
    "name-suggest": {
      "prefix": "piz",
      "completion": {
        "field": "name.suggest",
        "size": 10,
        "contexts": {
          "location": {"lat": 37.7749, "lon": -122.4194, "precision": 5}
        }
      }
    }
  }
}
\`\`\`

### 3. Denormalization

**Problem**: Joining business + reviews + photos on every query is slow.

**Solution**: Denormalize common data.

\`\`\`sql
-- Businesses table includes denormalized fields
CREATE TABLE businesses (
    business_id BIGINT PRIMARY KEY,
    name VARCHAR(200),
    rating DECIMAL(2,1),        -- Denormalized from reviews
    review_count INT,           -- Denormalized
    top_3_photos JSON,          -- ["url1", "url2", "url3"]
    popular_dishes JSON,        -- ["Margherita Pizza", "Pasta"]
    average_wait_time INT,      -- Minutes
    ...
);
\`\`\`

---

## Step 16: Monitoring & Observability

**Key Metrics**:

1. **Search Latency**: p50, p95, p99 (target < 200ms)
2. **Search QPS**: Queries per second (track peaks)
3. **Cache Hit Rate**: Redis hit rate (target > 80%)
4. **Geospatial Accuracy**: % of results within specified radius
5. **Elasticsearch Query Time**: Index performance
6. **Database Query Time**: Slow query log (> 100ms)

**Alerts**:
- Search latency p95 > 300ms → Scale Elasticsearch cluster
- Cache hit rate < 70% → Increase Redis capacity, adjust TTLs
- Error rate > 1% → Investigate (DB down, ES timeout)
- Review submission lag > 1 min → Check Kafka consumer lag

---

## Step 17: Fraud Detection (Fake Reviews)

**Signals**:
- Multiple reviews from same IP in short time
- Account age < 7 days with 10+ reviews
- Generic review text (ML model detects)
- No check-in history (can't review without visiting)

**Prevention**:

\`\`\`python
def validate_review(user_id, business_id, text):
    # 1. Check-in required
    has_checkin = db.query("""
        SELECT 1 FROM checkins
        WHERE user_id = ? AND business_id = ?
        LIMIT 1
    """, user_id, business_id)
    
    if not has_checkin:
        return {"error": "Must check in before reviewing"}
    
    # 2. One review per user per business
    existing = db.query("""
        SELECT 1 FROM reviews
        WHERE user_id = ? AND business_id = ?
    """, user_id, business_id)
    
    if existing:
        return {"error": "Already reviewed this business"}
    
    # 3. ML fraud detection
    fraud_score = ml_model.predict_fraud(user_id, business_id, text)
    if fraud_score > 0.8:
        flag_for_manual_review(user_id, business_id)
    
    return {"valid": True}
\`\`\`

---

## Step 18: Business Owner Features

**Yelp for Business Dashboard**:

\`\`\`
Features:
- Update hours, photos, menu
- Respond to reviews (public replies)
- View analytics (page views, search appearances)
- Promote business (paid ads)

API:
POST /api/business/123/reply
{
  "review_id": 456,
  "text": "Thank you for your feedback! We've addressed the issue."
}
\`\`\`

**Review Response**:

\`\`\`sql
CREATE TABLE review_responses (
    review_id BIGINT PRIMARY KEY,
    business_id BIGINT,
    response_text TEXT,
    responded_at TIMESTAMP,
    FOREIGN KEY (review_id) REFERENCES reviews(review_id)
);
\`\`\`

---

## Trade-offs

**Geohash vs PostGIS**:
- Geohash: Simple, portable, requires neighbor queries
- PostGIS: Accurate, powerful, PostgreSQL-specific
- **Choice**: Geohash for sharded databases, PostGIS for single-region

**Redis Geo vs Database**:
- Redis: Fast (< 10ms), limited capacity (in-memory)
- Database: Slower (50ms), unlimited capacity
- **Choice**: Redis for hot data (popular cities), database for cold data

**Real-Time vs Batch**:
- Real-Time ratings: Every review updates average (expensive)
- Batch updates: Update every 5 minutes (cheaper, slight lag)
- **Choice**: Batch (eventual consistency acceptable)

**Consistency vs Availability**:
- Strong consistency: All users see same rating (slow)
- Eventual consistency: Rating lags by minutes (fast)
- **Choice**: Eventual (AP system)

---

## Interview Tips

**Clarify**:
1. Scale: City-level (1M businesses) or global (200M)?
2. Features: Search only, or reviews + photos + check-ins?
3. Real-time: Instant updates or eventual consistency?
4. Read/write ratio: 100:1 (optimize reads) or 10:1?

**Emphasize**:
1. **Geospatial Indexing**: Geohash or PostGIS for proximity queries
2. **Elasticsearch**: Combined text + geo + filter search
3. **Ranking Algorithm**: Multi-factor scoring (relevance, rating, distance)
4. **Caching**: Redis for search results, geo index, ratings
5. **Sharding**: Geographic sharding for locality

**Common Mistakes**:
- Using naive distance calculation (full table scan)
- No text search (only geo queries)
- Ignoring ranking (chronological results, not relevant)
- Not caching (database overwhelmed)
- No fraud detection (fake reviews)

**Follow-up Questions**:
- "How to handle businesses with multiple locations? (Each location separate entry)"
- "How to detect spam reviews? (ML model, check-in verification)"
- "How to scale to 1 billion searches/day? (Shard Elasticsearch, add caching)"
- "How to support offline mode? (Cache recent searches, nearby businesses)"

---

## Summary

**Core Components**:
1. **Geospatial Index**: Geohash (PostgreSQL) + Redis Geo (hot data)
2. **Search Engine**: Elasticsearch (text + geo + filters)
3. **Ranking Service**: Multi-factor scoring algorithm
4. **Reviews System**: PostgreSQL + Redis aggregation
5. **Caching Layer**: Redis (search results, geo, ratings)
6. **Database**: Sharded PostgreSQL (by geographic region)
7. **Event Stream**: Kafka (review events, business updates)

**Key Decisions**:
- ✅ Geohash for efficient radius queries (indexed prefix search)
- ✅ Elasticsearch for combined text + geospatial search
- ✅ Redis Geo for hot data (popular cities, < 10ms latency)
- ✅ Geographic sharding (most queries local, no cross-shard)
- ✅ Async rating aggregation (batch updates every 5 min)
- ✅ Multi-level caching (CDN, Redis, app memory)
- ✅ Eventual consistency (AP system, rating lags acceptable)

**Capacity**:
- 200M businesses globally
- 500M DAU
- 100M searches/day (10K QPS peak)
- < 200ms search latency (p95)
- 500M reviews
- 80%+ cache hit rate

Yelp's architecture prioritizes **low-latency search** (< 200ms), **accurate geospatial queries** (within radius), and **relevant ranking** (best results first) to help millions of users discover local businesses worldwide.`,
};

