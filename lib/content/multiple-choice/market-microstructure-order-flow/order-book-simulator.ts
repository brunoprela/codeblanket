import { MultipleChoiceQuestion } from '@/lib/types';

export const orderBookSimulatorMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'order-book-simulator-mc-1',
        question: 'In a price-time priority matching engine, three limit buy orders arrive in sequence: Order A: 100 shares @ $100.00 (t=1000ns), Order B: 100 shares @ $100.01 (t=1001ns), Order C: 100 shares @ $100.00 (t=1002ns). A sell market order for 150 shares arrives at t=1003ns. Which orders get filled and in what sequence?',
        options: [
            'Order B (100 @ $100.01), then Order A (50 @ $100.00)',
            'Order A (100 @ $100.00), then Order C (50 @ $100.00)',
            'Order B (100 @ $100.01), then Order C (50 @ $100.00)',
            'All three orders partially filled at $100.00 average'
        ],
        correctAnswer: 0,
        explanation: `**Order B first, then Order A (Option 0)** is correct due to price-time priority. Matching engines prioritize: (1) PRICE first (best price wins), (2) TIME second (earlier timestamp wins at same price). Order B has the best price ($100.01 > $100.00 for buys), so it fills first (100 shares). The remaining 50 shares match against the next best price ($100.00), where Order A has time priority (t=1000ns < t=1002ns), so Order A gets the remaining 50 shares. Order C receives no fill despite being at the same price as Order A because it arrived later. This demonstrates the fundamental matching algorithm used by all major exchanges.`
    },
    {
        id: 'order-book-simulator-mc-2',
        question: 'A Fill-Or-Kill (FOK) order to buy 5,000 shares @ $150.00 arrives when the order book has: $150.00 ask (3,000 shares), $150.01 ask (1,500 shares), $150.02 ask (2,000 shares). What happens to the FOK order?',
        options: [
            'Fills completely: 3,000 @ $150.00 + 1,500 @ $150.01 + 500 @ $150.02',
            'Partially fills 3,000 shares at $150.00, remainder canceled',
            'Rejected entirely because insufficient liquidity at limit price',
            'Converts to limit order and rests on book for remaining quantity'
        ],
        correctAnswer: 2,
        explanation: `**Rejected entirely (Option 2)** because FOK orders require complete fill at the limit price or better, and only 4,500 shares are available at acceptable prices (3,000 @ $150.00 + 1,500 @ $150.01). The FOK order cannot fill all 5,000 shares without going to $150.02, which exceeds the $150.01 effective limit for matching. FOK semantics are all-or-nothing: if the full quantity cannot be executed immediately, the entire order is canceled. This differs from IOC (Immediate-Or-Cancel) which would partial fill 4,500 shares and cancel the remainder. FOK is used by institutional traders who need guaranteed complete execution or no execution at all, avoiding partial fills that leave unwanted positions.`
    },
    {
        id: 'order-book-simulator-mc-3',
        question: 'The matching engine processes 100,000 orders in 2.5 seconds with the following latency distribution: median 15μs, 95th percentile 45μs, 99th percentile 350μs, maximum 12ms. The 12ms outlier is MOST likely caused by:',
        options: [
            'Network congestion affecting order arrival times',
            'Garbage collection pause in the programming language runtime',
            'CPU thermal throttling due to overheating',
            'Lock contention from concurrent order processing'
        ],
        correctAnswer: 1,
        explanation: `**Garbage collection pause (Option 1)** is the most likely cause of a 12ms latency spike. The telltale signs are: (1) median and P95 are excellent (15μs, 45μs), suggesting the core algorithm is fast, (2) P99 jumps to 350μs but is still reasonable, (3) maximum spikes to 12ms - a classic GC pause pattern. Languages like Java, C#, Python have stop-the-world GC pauses that can freeze execution for 5-50ms. In production HFT systems, this is solved by: using low-GC languages (C++, Rust), tuning GC (generational GC, concurrent collectors), or eliminating allocations entirely (object pools, pre-allocation). Network congestion would affect multiple consecutive orders, CPU throttling would show gradual degradation, and lock contention would elevate P50/P95, not just the max.`
    },
    {
        id: 'order-book-simulator-mc-4',
        question: 'A market maker bot maintains quotes of $100.00 bid / $100.10 ask with 100 shares on each side. After buying 50 shares (now +50 position), how should the bot adjust quotes to manage inventory risk?',
        options: [
            'Widen both bid and ask by $0.05 to $99.95 / $100.15',
            'Shift quotes down: $99.95 bid / $100.05 ask (skew toward selling)',
            'Shift quotes up: $100.05 bid / $100.15 ask (skew toward buying more)',
            'Keep quotes unchanged - position is too small to matter'
        ],
        correctAnswer: 1,
        explanation: `**Shift quotes down to $99.95 / $100.05 (Option 1)** is correct for inventory management. After buying 50 shares (+50 long position), the market maker wants to sell inventory to return to neutral. By lowering both bid and ask by $0.05, the bot: (1) makes the ask more competitive ($100.05 vs $100.10), increasing probability of selling, (2) makes the bid less competitive ($99.95 vs $100.00), decreasing probability of buying more, (3) maintains the same spread width (still $0.10). This is called "inventory skewing" or "quote shading" - a core market making technique. The Avellaneda-Stoikov model mathematically derives optimal skew based on position size, risk aversion, and volatility. Without inventory management, market makers accumulate dangerous one-sided positions that risk large losses during adverse price moves.`
    },
    {
        id: 'order-book-simulator-mc-5',
        question: 'You are building a production order book simulator and must choose a data structure for storing orders at each price level. The workload is: 80% order additions, 15% cancellations, 5% modifications. Orders at the same price must be served FIFO. Which data structure provides the best performance?',
        options: [
            'Balanced binary search tree (Red-Black tree) for O(log n) operations',
            'Hash map with doubly-linked list for O(1) average case',
            'Priority queue (heap) for O(log n) insertions',
            'Array with binary search for O(log n) lookups'
        ],
        correctAnswer: 1,
        explanation: `**Hash map with doubly-linked list (Option 1)** provides optimal performance for this workload. Architecture: hash map maps price → linked list of orders at that price. Benefits: (1) O(1) order addition (append to linked list tail), (2) O(1) order cancellation (linked list removal with order ID lookup), (3) FIFO guarantee (linked list preserves order), (4) O(1) best price access (maintain separate pointer to best bid/ask price). The 80% addition workload heavily favors O(1) append over tree O(log n) insertion. This is the data structure used by real exchanges like NASDAQ (INET) and CME (Globex). Priority queue doesn't maintain FIFO at same price. Arrays require O(n) insertion. BST has O(log n) overhead unnecessary when hash map gives O(1). For production systems handling millions of orders/day, the difference between O(1) and O(log n) is significant at scale.`
    }
];

