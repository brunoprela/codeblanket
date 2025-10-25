/**
 * Quiz questions for Design Yelp section
 */

export const yelpQuiz = [
  {
    id: 'q1',
    question:
      'Explain how geohashing enables efficient nearby business queries. Why is it better than calculating distances to all businesses in the database?',
    sampleAnswer:
      'GEOHASHING: Encode lat/lon (37.7749, -122.4194) to base32 string "9q8yy9mf". Nearby locations share common prefixes: Business A: 9q8yy9mf, Business B: 9q8yy9mx (prefix 9q8yy = ~5km radius). QUERY EFFICIENCY: SELECT * FROM businesses WHERE geohash LIKE \'9q8yy%\' hits index on geohash column, returns ~1000 businesses in area. O(log N) index lookup vs O(N) full table scan. NAIVE APPROACH: Calculate Haversine distance to ALL businesses: SELECT *, SQRT(POW(lat-?, 2) + POW(lon-?, 2)) as dist FROM businesses WHERE dist < 5 ORDER BY dist. Must scan entire table (200M businesses), calculate distance for each, sort. Time: 10+ seconds. GEOHASH APPROACH: Filter by prefix first (narrows to 1000), then calculate exact distance. Time: 10-50ms. KEY INSIGHT: Geospatial index (geohash, quadtree, R-tree) transforms 2D location query into 1D string prefix query (fast with B-tree index).',
    keyPoints: [
      'Geohash encodes lat/lon to string, nearby locations share prefix',
      "Query: WHERE geohash LIKE '9q8yy%' uses B-tree index (fast)",
      'Narrows 200M businesses to ~1000 candidates before distance calc',
      'Naive approach: Calculate distance to all 200M (too slow)',
      'Result: 10-50ms query time vs 10+ seconds',
    ],
  },
  {
    id: 'q2',
    question:
      'Design the search ranking algorithm for Yelp. User searches "best pizza". 100 pizzerias found within 5 km. How do you rank them?',
    sampleAnswer:
      'RANKING FACTORS: (1) Relevance (40%): TF-IDF match score. "Tony\'s Pizza" (contains "pizza" in name) scores higher than "Italian Restaurant" (pizza in description). Boost: Name match 3x, category 2x, description 1x. (2) Rating (30%): 4.8 stars scores higher than 4.2. Normalize: (rating - 3.0) / 2.0 = 0-1 scale. (3) Popularity (20%): More reviews = more trustworthy. Score = log(review_count + 1). Logarithmic prevents popular places dominating. (4) Distance (10%): Closer = slightly higher. Score = max(0, 1 - distance/radius). FORMULA: score = 0.4*relevance + 0.3*(rating-3)/2 + 0.2*log(reviews+1)/10 + 0.1*(1-dist/5). EXAMPLE: Pizza A: 5km, 4.9 rating, 500 reviews, perfect match. Score = 0.4*1.0 + 0.3*0.95 + 0.2*0.62 + 0.1*0.0 = 0.809. Pizza B: 1km, 4.3 rating, 50 reviews, partial match. Score = 0.4*0.6 + 0.3*0.65 + 0.2*0.39 + 0.1*0.8 = 0.553. Rank A higher despite B being closer. PERSONALIZATION: User history: Prefers Italian → Boost Italian restaurants. Previous visits: Hide already reviewed places. Time of day: Lunch time → Boost quick service. KEY INSIGHT: Multi-factor ranking balances multiple signals - no single "best" answer.',
    keyPoints: [
      'Weighted score: 40% relevance, 30% rating, 20% popularity, 10% distance',
      'Relevance: TF-IDF with boost for name matches',
      'Popularity: Logarithmic (prevents dominance by chain restaurants)',
      'Distance: Minor factor (users willing to travel for quality)',
      'Personalization: User history, time of day adjustments',
    ],
  },
  {
    id: 'q3',
    question:
      'How do you aggregate ratings efficiently? Business has 10,000 reviews. User requests business page. Explain real-time aggregation strategy.',
    sampleAnswer:
      'REAL-TIME AGGREGATION PROBLEM: Naive: SELECT AVG(rating) FROM reviews WHERE business_id=123 on every page load. 10K reviews = slow query (100-200ms), high DB load. SOLUTION - REDIS AGGREGATION: (1) WRITE PATH (New review): User submits rating=5 for business 123. Store review in database (async). Update Redis: ZADD business:123:ratings 5 {timestamp}, INCR business:123:count. Calculate new average: sum / count. Store: SET business:123:avg_rating 4.7. (2) READ PATH (Page load): Query Redis: GET business:123:avg_rating (sub-ms). Display: 4.7 stars (10,234 reviews). No database query needed. (3) SYNC TO DB (Batch job hourly): UPDATE businesses SET rating=4.7, review_count=10234 WHERE id=123. Database has eventual consistency (1-hour lag acceptable for display). (4) DISTRIBUTED RATINGS: Histogram: HSET business:123:histogram "5" 6000, "4" 2500, "3" 1000, "2" 300, "1" 200. Display rating distribution bar chart. FAILURE HANDLING: Redis crashes: Fallback to database (slower but works). Rebuild Redis from database on restart. ALTERNATIVE - PRECOMPUTED: Materialized view updated on every review insert (heavier write cost). Dropbox approach: Periodic recalculation (eventual consistency). KEY INSIGHT: Move aggregation from query time to write time (amortize cost across all users).',
    keyPoints: [
      'Redis stores pre-aggregated stats (avg_rating, review_count)',
      'Update Redis on each new review (ZADD, INCR, calculate avg)',
      'Page load: GET from Redis (sub-ms) vs DB query (100ms)',
      'Batch sync to database hourly (eventual consistency)',
      'Fallback to DB if Redis unavailable',
    ],
  },
];
