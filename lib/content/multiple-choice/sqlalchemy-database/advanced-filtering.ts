import { MultipleChoiceQuestion } from '@/lib/types';

export const advancedFilteringMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sql-filter-mc-1',
    question:
      'What is the difference between JSON and JSONB column types in PostgreSQL?',
    options: [
      'No difference, they are aliases',
      'JSONB is binary format, supports indexing and faster queries; JSON stores as text',
      'JSON is faster for all operations',
      'JSONB only works with arrays, JSON works with objects',
    ],
    correctAnswer: 1,
    explanation:
      'JSONB (JSON Binary) stores data in decomposed binary format, enabling: (1) GIN indexes for fast queries, (2) faster processing (no reparsing), (3) automatic deduplication of keys, (4) consistent key ordering. JSON stores as plain text: faster writes (no binary conversion), preserves whitespace and key order, but no indexing support and slower queries. For production applications that query JSON data, always use JSONB. Only use JSON if you need exact text preservation (rare).',
  },
  {
    id: 'sql-filter-mc-2',
    question: 'What does the @> operator do in PostgreSQL JSONB queries?',
    options: [
      'Greater than comparison',
      'Array contains element',
      'JSON containment: left side contains right side',
      'Concatenation',
    ],
    correctAnswer: 2,
    explanation:
      'The @> operator checks if the left JSONB value contains the right JSONB value. Example: {\"a\": 1, \"b\": 2} @> {\"a\": 1} returns true. {\"tags\": [\"python\", \"sql\"]} @> {\"tags\": [\"python\"]} returns true. It works on both objects and arrays. Requires GIN index for performance. Common use: WHERE metadata @> \'{\"brand\": \"Apple\"}\' finds all products with brand Apple. The inverse operator <@ checks if left is contained by right.',
  },
  {
    id: 'sql-filter-mc-3',
    question:
      'What index type should you create on a tsvector column for full-text search?',
    options: ['B-tree', 'Hash', 'GIN (Generalized Inverted Index)', 'GiST'],
    correctAnswer: 2,
    explanation:
      'GIN (Generalized Inverted Index) is the standard index for full-text search on tsvector columns. It creates an inverted index mapping each lexeme (word stem) to the documents containing it, enabling fast full-text queries. Without GIN index, full-text search requires sequential scan (10-100x slower). GiST is also possible but generally slower for lookups (though faster for updates). B-tree and Hash indexes do not support full-text search operations. Command: CREATE INDEX idx_search ON articles USING gin (search_vector);',
  },
  {
    id: 'sql-filter-mc-4',
    question:
      'In full-text search, what does the & operator mean in to_tsquery()?',
    options: [
      'Bitwise AND',
      'Pattern matching',
      'Boolean AND: both terms must be present',
      'Proximity operator',
    ],
    correctAnswer: 2,
    explanation:
      'The & operator in to_tsquery() means boolean AND: both terms must be present in the document. Example: to_tsquery(\'python & database\') matches documents containing both "python" AND "database". Other operators: | (OR, matches either term), ! (NOT, excludes term), <-> (phrase, adjacent words), <N> (proximity, within N words). Example query: SELECT * FROM articles WHERE search_vector @@ to_tsquery(\'python & database & !java\') - finds articles with python AND database but NOT java.',
  },
  {
    id: 'sql-filter-mc-5',
    question: 'What does the PostgreSQL ARRAY contains operator @> check for?',
    options: [
      'Array length',
      'Element exists at any position',
      'Left array contains all elements from right array',
      'Array concatenation',
    ],
    correctAnswer: 2,
    explanation:
      "The @> operator for arrays checks if the left array contains all elements from the right array. Example: ARRAY[1,2,3,4] @> ARRAY[2,3] returns true (contains both 2 and 3). ARRAY['python', 'sql', 'java'] @> ARRAY['python'] returns true. Order does not matter, and left array can have additional elements. For checking single element: use ANY(array_column) = value. For overlap (at least one common element): use &&. For exact match: use =. Requires GIN index for performance: CREATE INDEX ON table USING gin (array_column);",
  },
];
