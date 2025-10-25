/**
 * Multiple choice questions for Design Yelp section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const yelpMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Yelp needs to find all restaurants within 5 km of a user. Which data structure/algorithm is most efficient?',
    options: [
      'Scan all 200M businesses, calculate distance for each',
      'Geohash index: Filter by prefix, then calculate exact distance (10-50ms)',
      'Sort by distance using SQL ORDER BY',
      'Use linear search with caching',
    ],
    correctAnswer: 1,
    explanation:
      "GEOHASH APPROACH: Index businesses by geohash. Query: WHERE geohash LIKE '9q8yy%' returns ~1000 candidates in 5km area (uses B-tree index, O(log N)). Calculate exact Haversine distance for 1000 candidates. Total time: 10-50ms. FULL SCAN (Option A): Calculate distance to all 200M businesses. Even with indexed lat/lon, must compute distance for each. Time: 10+ seconds (unusable). SORTING (Option C): ORDER BY distance requires computing distance first (back to full scan problem). CACHING (Option D): Cannot cache every possible user location (infinite combinations). Geospatial indexing is the only practical solution at scale.",
  },
  {
    id: 'mc2',
    question:
      'Why does Yelp use Elasticsearch for search instead of SQL LIKE queries?',
    options: [
      'Elasticsearch is cheaper',
      'SQL LIKE is slow and limited; Elasticsearch provides full-text search, relevance ranking, and geospatial queries',
      'SQL databases cannot store text',
      'Elasticsearch is required by law',
    ],
    correctAnswer: 1,
    explanation:
      'SQL LIKE LIMITATIONS: WHERE name LIKE \'%pizza%\' requires full table scan (no index for middle wildcard), no relevance ranking, no stemming ("pizzeria" != "pizza"), no typo tolerance. Time: Seconds for 200M businesses. ELASTICSEARCH BENEFITS: Inverted index: Instant lookup of documents containing "pizza". Relevance: TF-IDF scoring ranks best matches first. Fuzzy matching: "piza" → "pizza". Geospatial: Built-in radius queries. Faceted search: Filter by rating, price simultaneously. Time: 10-100ms for complex queries. PRODUCTION: Use Elasticsearch for search, SQL for transactional data (reviews, orders). Best tool for each job.',
  },
  {
    id: 'mc3',
    question:
      'Yelp caches popular search results in Redis with 5-minute TTL. Why not cache for 24 hours to reduce load further?',
    options: [
      'Redis cannot store data for 24 hours',
      'Business data changes frequently (new reviews, rating updates, hours); 24-hour cache shows stale data',
      '5 minutes is a legal requirement',
      'Redis runs out of memory',
    ],
    correctAnswer: 1,
    explanation:
      'STALENESS PROBLEM: Business gets new review (5 stars) → Rating improves 4.2 → 4.5. With 24-hour cache: Users see old 4.2 rating for up to 24 hours (poor UX). Restaurant closes early → Hours cached as "open" for 24 hours (users arrive to closed door). FRESHNESS VS LOAD: 5-minute TTL: Data max 5 minutes stale (acceptable). Cache hit rate still high (search results don\'t change much in 5 min). Reduces DB load by 80-90%. 24-hour TTL: Higher cache hit rate but unacceptable staleness. RECOMMENDATION: 5-10 minutes for search results, 1-2 hours for business details (change less frequently). TTL tuned per data type.',
  },
  {
    id: 'mc4',
    question:
      'A pizza restaurant has 10,000 reviews with average rating 4.7. A new 1-star review is submitted. What is the impact on the average?',
    options: [
      'Average drops to 2.85 (significant impact)',
      'Average drops to ~4.696 (minimal impact due to large sample)',
      'Average unchanged (1 review ignored)',
      'Average recalculated daily only',
    ],
    correctAnswer: 1,
    explanation:
      'CALCULATION: Old average: (4.7 × 10,000) = 47,000 total stars. New review: +1 star. New average: (47,000 + 1) / 10,001 = 4.6995 ≈ 4.70. IMPACT: Negligible (0.0005 decrease). LAW OF LARGE NUMBERS: With many reviews, single review has minimal impact. This prevents manipulation (fake reviews less effective at scale). CONTRAST: New business (10 reviews, 4.7 avg): One 1-star review → (4.7×10 + 1)/11 = 4.36 (significant drop). This is why Yelp highlights "new" vs "established" businesses - confidence in rating.',
  },
  {
    id: 'mc5',
    question:
      'Yelp shards businesses by geographic region (SF, NYC, LA on separate shards). A user in SF searches for "pizza". Which shard (s) are queried?',
    options: [
      'All shards (need global results)',
      'SF shard only (user location in SF)',
      'NYC shard (where most pizza is)',
      'Random shard',
    ],
    correctAnswer: 1,
    explanation:
      'GEO-SHARDING: User location (SF) → Route to SF shard. Query only SF shard (restaurants in SF). Return results. BENEFITS: Single-shard query (fast, no scatter-gather), data locality (most searches are local). EDGE CASE: User near border (SF/Oakland) → May need to query 2 shards. User searching different city ("pizza in NYC" while in SF) → Route to NYC shard. TRADE-OFF: Optimizes for common case (local search 95% of queries), handles cross-region as exception. Alternative: Shard by business_id (must scatter-gather ALL shards for every query - slow). Geo-sharding is optimal for location-based services.',
  },
];
