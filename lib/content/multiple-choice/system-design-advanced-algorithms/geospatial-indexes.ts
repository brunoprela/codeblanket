/**
 * Multiple choice questions for Geospatial Indexes section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const geospatialindexesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the time complexity of a naive proximity search (checking distance to every location)?',
    options: ['O(1)', 'O(log N)', 'O(N)', 'O(N²)'],
    correctAnswer: 2,
    explanation:
      'Naive proximity search requires checking distance from target to every location: O(N) where N = number of locations. For 1 million locations, this means 1 million distance calculations per query. Geospatial indexes (Quadtree, R-Tree, Geohash) reduce this to O(log N) using spatial partitioning.',
  },
  {
    id: 'mc2',
    question:
      'How does geohash encoding enable fast proximity queries in databases?',
    options: [
      'It stores exact distances',
      'It calculates distances faster',
      'Nearby locations share common prefixes, enabling B-tree prefix matching',
      'It uses less storage space',
    ],
    correctAnswer: 2,
    explanation:
      'Geohash encodes lat/lon as strings where nearby locations share common prefixes (e.g., all locations in a neighborhood start with "9q8yy"). This enables fast queries with B-tree indexes: WHERE geohash LIKE "9q8yy%" uses index scan instead of full table scan. Works great with MongoDB, PostgreSQL, Redis.',
  },
  {
    id: 'mc3',
    question:
      'Which geospatial structure is best for disk-based databases handling rectangle and polygon queries?',
    options: ['Quadtree', 'R-Tree', 'Geohash', 'Hash table'],
    correctAnswer: 1,
    explanation:
      'R-Trees are optimal for disk-based databases with rectangle/polygon queries. They use hierarchical bounding boxes, minimize overlap, and are disk-friendly. PostGIS uses R-Tree variant (GiST index) for spatial queries. Quadtrees are better in-memory, Geohash works for point queries with B-trees.',
  },
  {
    id: 'mc4',
    question: 'What is a major limitation of geohash for proximity queries?',
    options: [
      'It is too slow',
      'Grid squares do not perfectly match circular search areas at boundaries',
      'It cannot encode latitude and longitude',
      'It requires too much memory',
    ],
    correctAnswer: 1,
    explanation:
      'Geohash divides space into square grid cells. At cell boundaries, nearby locations may have different prefixes. A circular 2km search radius might span multiple geohash squares. Solution: check adjacent geohash cells (8 neighbors). This is acceptable overhead—Uber does this when expanding search radius.',
  },
  {
    id: 'mc5',
    question: 'Which production systems use geospatial indexes?',
    options: [
      'Only academic research',
      'Uber, MongoDB, and PostgreSQL PostGIS',
      'Only mapping applications',
      'Only single-server applications',
    ],
    correctAnswer: 1,
    explanation:
      'Geospatial indexes are production-standard: Uber (driver matching with geohash+Redis), MongoDB (2dsphere index), PostgreSQL PostGIS (R-Tree GiST), Google Maps (S2), Redis (GEOADD/GEORADIUS). Any location-based service at scale uses geospatial indexes. This is core infrastructure, not theory.',
  },
];
