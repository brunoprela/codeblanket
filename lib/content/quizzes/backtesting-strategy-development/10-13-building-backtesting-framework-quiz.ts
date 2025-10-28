import { MultipleChoiceQuestion } from '@/lib/types';

const buildingBacktestingFrameworkQuiz: MultipleChoiceQuestion[] = [
  {
    id: 'framework-1',
    question:
      'What is the primary advantage of event-driven backtesting architecture over vectorized backtesting?',
    options: [
      'Event-driven is faster and uses less memory',
      'Event-driven processes data chronologically, preventing look-ahead bias by design',
      'Event-driven is easier to implement and requires less code',
      'Event-driven produces more optimistic results',
    ],
    correctAnswer: 1,
    explanation:
      'Event-driven architecture processes market data chronologically, bar by bar, ensuring the strategy only has access to information available at each point in time—making look-ahead bias structurally impossible. Vectorized backtesting loads all data at once into arrays/DataFrames, making it easy to accidentally peek at future data. Option A is backwards (event-driven is slower but more accurate). Option C is wrong (event-driven requires more code and complexity). Option D is incorrect and undesirable. The key trade-off: event-driven sacrifices speed for correctness, which is essential for production trading systems.',
    difficulty: 'intermediate',
  },
  {
    id: 'framework-2',
    question:
      'In a properly designed backtesting framework, which component should be responsible for calculating Sharpe ratio and other performance metrics?',
    options: [
      'The Strategy class (integrated with trading logic)',
      'The ExecutionHandler class (calculated during order execution)',
      'A separate Portfolio/Analytics class (separation of concerns)',
      'The DataHandler class (calculated with market data)',
    ],
    correctAnswer: 2,
    explanation:
      'Performance metrics should be calculated by a separate Portfolio or Analytics component following the separation of concerns principle. The Strategy class should ONLY contain trading logic (signals generation). The ExecutionHandler simulates order fills. The DataHandler provides market data. Mixing responsibilities makes code harder to test, maintain, and reuse. For example, you should be able to swap in a different strategy without changing metrics calculation, or change slippage models without touching strategy code. This modular design is standard in production systems at firms like Two Sigma and WorldQuant.',
    difficulty: 'easy',
  },
  {
    id: 'framework-3',
    question:
      "Your backtesting framework's ExecutionHandler simulates market orders with 0.05% slippage and 0.10% commission. A strategy generates 1000 trades per year. What is the annual cost drag from these frictions alone (ignoring returns)?",
    options: [
      '0.15% (0.05% + 0.10%)',
      '1.5% (0.15% × 10 round trips)',
      '150% (0.15% × 1000 trades)',
      '3.0% (0.15% × 2 × 1000 trades, accounting for entry and exit)',
    ],
    correctAnswer: 3,
    explanation:
      'Each trade incurs BOTH entry and exit costs. With 1000 trades/year: Each trade has entry (0.05% slippage + 0.10% commission = 0.15%) and exit (another 0.15%), totaling 0.30% per complete round trip. Total annual cost = 0.30% × 1000 = 300% of portfolio value—clearly unsustainable. This illustrates why high-frequency strategies need institutional-grade execution (sub-penny spreads, maker rebates). Option A forgets these costs apply to every trade. Option B miscalculates. Option C forgets exit costs. This calculation is critical—many strategies look profitable until realistic transaction costs are applied.',
    difficulty: 'advanced',
  },
  {
    id: 'framework-4',
    question:
      'What is the main reason professional backtesting frameworks use abstract base classes (ABC) or interfaces for Strategy, DataHandler, and ExecutionHandler?',
    options: [
      'Abstract classes make the code run faster',
      "It's required by Python type checking systems",
      'It enables polymorphism: easily swap components without changing other code',
      'Abstract classes use less memory than regular classes',
    ],
    correctAnswer: 2,
    explanation:
      "Abstract base classes enable polymorphism—you can swap different implementations of strategies, data sources, or execution models without modifying the backtesting engine. For example, you can test the same strategy with HistoricalCSVDataHandler, DatabaseDataHandler, or LiveDataHandler just by passing a different implementation. The Backtest class doesn't need to know implementation details, only that the component conforms to the interface. Option A and D are wrong (performance/memory aren't affected). Option B is partially true but not the main reason. This design pattern is fundamental to maintainable production systems.",
    difficulty: 'intermediate',
  },
  {
    id: 'framework-5',
    question:
      "In production backtesting frameworks, why should the Portfolio class maintain separate 'cash' and 'positions' state rather than just tracking total equity?",
    options: [
      'Separate tracking is more memory efficient',
      'It enables proper leverage calculation, margin requirements, and prevents over-trading',
      'It makes the code look more professional',
      'Cash and positions are automatically synchronized so tracking is unnecessary',
    ],
    correctAnswer: 1,
    explanation:
      "Tracking cash separately is essential for realistic simulation. A strategy might generate a buy signal, but if cash is insufficient, the order must be rejected or sized down—this happens in live trading. Without cash tracking, a backtest might assume infinite leverage, showing unrealistic returns. Additionally, short positions and margin requirements require explicit cash tracking. For example, if you have $100K and are 50% in positions, you have $50K buying power—but with margin, you might have $150K. Option A is irrelevant. Option C is superficial. Option D is wrong (they must be explicitly managed). Proper cash management prevents 'fantasy trading' in backtests.",
    difficulty: 'advanced',
  },
];

export default buildingBacktestingFrameworkQuiz;
