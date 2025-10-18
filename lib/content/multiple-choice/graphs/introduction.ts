/**
 * Multiple choice questions for Introduction to Graphs section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is a graph?',
    options: [
      'A tree',
      'Vertices (nodes) connected by edges modeling relationships/networks',
      'An array',
      'A sorted list',
    ],
    correctAnswer: 1,
    explanation:
      'A graph consists of vertices (nodes) connected by edges. Graphs model networks like social connections, maps, dependencies, etc. Unlike trees, graphs can have cycles.',
  },
  {
    id: 'mc2',
    question: 'What is the most common graph representation and why?',
    options: [
      'Adjacency matrix',
      'Adjacency list - space efficient O(V+E), fast neighbor iteration',
      'Edge list',
      'Array',
    ],
    correctAnswer: 1,
    explanation:
      'Adjacency list is most common: space O(V+E) vs matrix O(V²), fast to iterate neighbors. Most real graphs are sparse, making list more efficient.',
  },
  {
    id: 'mc3',
    question: 'What is a DAG?',
    options: [
      'A weighted graph',
      'Directed Acyclic Graph - directed with no cycles',
      'A tree',
      'Dense graph',
    ],
    correctAnswer: 1,
    explanation:
      'DAG = Directed Acyclic Graph. Edges have direction and no cycles exist. Used for dependency graphs, scheduling, compilation order. Enables topological sorting.',
  },
  {
    id: 'mc4',
    question: 'When should you use adjacency matrix over adjacency list?',
    options: [
      'Always',
      'Dense graphs where checking edge existence is frequent (O(1) lookup)',
      'Sparse graphs',
      'Never',
    ],
    correctAnswer: 1,
    explanation:
      'Use matrix when graph is dense (many edges) and need fast O(1) edge lookup. Matrix uses O(V²) space - inefficient for sparse graphs where list is better.',
  },
  {
    id: 'mc5',
    question: 'What is the difference between directed and undirected graphs?',
    options: [
      'No difference',
      'Directed: edges have direction (A→B), Undirected: edges are bidirectional (A↔B)',
      'Directed is faster',
      'Undirected has more edges',
    ],
    correctAnswer: 1,
    explanation:
      "Directed: edges go one way (A→B doesn't mean B→A). Undirected: edges go both ways (A-B means both A→B and B→A). Social networks often undirected, web pages directed.",
  },
];
