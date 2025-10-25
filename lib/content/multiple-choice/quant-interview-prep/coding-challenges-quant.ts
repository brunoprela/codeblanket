import { MultipleChoiceQuestion } from '@/lib/types';

export const codingChallengesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'ccq-mc-1',
    question:
      'For an order book with n orders, what is the time complexity to get the best bid using a max-heap?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)'],
    correctAnswer: 0,
    explanation:
      'Getting the best bid (top of max-heap) is O(1). The maximum element is always at the root. Insertion and deletion are O(log n), but peek is constant time.',
  },
  {
    id: 'ccq-mc-2',
    question:
      'Newton-Raphson method for implied volatility typically converges in how many iterations?',
    options: ['1-2', '3-5', '10-15', '50-100'],
    correctAnswer: 1,
    explanation:
      'Newton-Raphson has quadratic convergence, typically needing 3-5 iterations to reach machine precision for implied volatility. Binary search would need 20-30 iterations for similar accuracy.',
  },
  {
    id: 'ccq-mc-3',
    question:
      'Which algorithm detects arbitrage cycles in currency exchange rate graphs?',
    options: ['Dijkstra', 'Bellman-Ford', 'Kruskal', 'Floyd-Warshall'],
    correctAnswer: 1,
    explanation:
      'Bellman-Ford detects negative cycles, which correspond to arbitrage opportunities when exchange rates are converted to log scale. Dijkstra cannot handle negative weights. Floyd-Warshall works but is O(n³) always, while Bellman-Ford is O(VE).',
  },
  {
    id: 'ccq-mc-4',
    question:
      'For portfolio optimization with n assets, computing covariance matrix takes:',
    options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(n³)'],
    correctAnswer: 2,
    explanation:
      'Covariance matrix is n×n, requiring O(n²) pairwise covariance calculations. Computing optimal weights via quadratic programming is O(n³) worst case, but specialized methods exist for large portfolios.',
  },
  {
    id: 'ccq-mc-5',
    question:
      'Binary search to find implied volatility with tolerance 0.0001 needs approximately how many iterations?',
    options: ['7', '14', '20', '40'],
    correctAnswer: 1,
    explanation:
      'Binary search: k iterations gives precision ~(range/2^k). For vol range [0,2] and tolerance 0.0001: 2/2^k = 0.0001 → 2^k = 20,000 → k ≈ 14.3. So approximately 14-15 iterations needed.',
  },
];
