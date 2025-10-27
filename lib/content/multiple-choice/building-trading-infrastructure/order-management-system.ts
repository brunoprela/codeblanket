export const orderManagementSystemMC = [
  {
    id: 'order-management-system-mc-1',
    question:
      'What is the PRIMARY purpose of maintaining an order state machine (NEW → PENDING_RISK → ACCEPTED → FILLED) in an OMS?',
    options: [
      'To make the code more complex and object-oriented',
      'To ensure valid state transitions, prevent invalid operations (like filling a cancelled order), and provide audit trail',
      'To slow down order processing for safety',
      'To make orders easier to cancel',
    ],
    correctAnswer: 1,
    explanation:
      "Order State Machine Benefits: (1) VALIDATION: Ensures valid transitions only (e.g., can't fill CANCELLED order, can't cancel FILLED order). Prevents data corruption. (2) BUSINESS LOGIC: Encodes trading rules (order must pass risk before execution). Enforces workflow. (3) AUDIT TRAIL: Every state change logged (NEW → ACCEPTED at 10:00:01, ACCEPTED → FILLED at 10:00:05). Required for regulatory compliance (SEC, FINRA). (4) DEBUGGING: Clear history of what happened to order. Example: Order rejected? Check state transitions to find where/why. (5) CONSISTENCY: All systems agree on order state (risk system, execution system, reporting). Without state machine: Ad-hoc state management → bugs (fill cancelled order, double-fill, orphaned orders). Real-world: All professional OMS systems (Bloomberg, Interactive Brokers, Citadel) use state machines for order management.",
  },
  {
    id: 'order-management-system-mc-2',
    question:
      'When processing order fills, why is it critical to use database locking (SELECT FOR UPDATE) or atomic operations?',
    options: [
      'To make the database slower and more reliable',
      'To prevent concurrent fills from causing race conditions where filled_quantity is incorrectly calculated',
      'To ensure fills are processed in alphabetical order',
      'To allow multiple OMS instances to process the same fill',
    ],
    correctAnswer: 1,
    explanation:
      'Concurrent Fill Problem: Order: 100 shares, currently filled: 0. Fill A (50 shares) and Fill B (50 shares) arrive simultaneously. WITHOUT LOCKING: Process A: Read filled_qty = 0, Calculate new_filled = 0 + 50 = 50, Process B: Read filled_qty = 0 (before A commits!), Calculate new_filled = 0 + 50 = 50, Both write filled_qty = 50 → WRONG (should be 100). Lost update problem. WITH LOCKING (SELECT FOR UPDATE): Process A: Acquires lock, reads filled_qty = 0, calculates 50, commits (releases lock), Process B: Waits for lock, reads filled_qty = 50, calculates 100, commits → CORRECT. ALTERNATIVES: Atomic operations (Redis INCR), Message queue (serial processing), Optimistic locking (version numbers). Real-world: All high-frequency trading systems use locking or serialization. Without it: Position tracking breaks → massive P&L errors → regulatory fines. Example: Knight Capital lost $440M in 2012 partly due to order state bugs.',
  },
  {
    id: 'order-management-system-mc-3',
    question:
      'Why do most OMS systems use the "cancel-replace" pattern for order amendments instead of modifying orders in-place?',
    options: [
      'In-place modification is faster but cancel-replace is easier to code',
      'Most exchanges do not support in-place modification; cancel-replace provides consistent behavior across venues and clear audit trail',
      'Cancel-replace generates more fees for brokers',
      'In-place modification violates SEC regulations',
    ],
    correctAnswer: 1,
    explanation:
      'Cancel-Replace vs In-Place Modification: EXCHANGE SUPPORT: Most exchanges (NYSE, NASDAQ): Cancel-replace only (no modify support). Few exchanges (some ECNs): Support modify, but behavior inconsistent. Using cancel-replace everywhere: Consistent behavior across all venues. AUDIT TRAIL: Cancel-replace: Two distinct orders (original + replacement) in database. Clear history: Order A cancelled (reason: replaced by Order B), Order B created (parent: Order A). In-place modify: Same order ID, harder to audit (which fields changed when?). RISK MANAGEMENT: Cancel-replace: New order goes through risk checks again (quantity increased? check limits). In-place modify: Risk check bypass risk (quantity change may not be validated). FAILURE HANDLING: Cancel-replace: If replace fails, original order remains (safe). In-place modify: If modify fails mid-update, order in unknown state (dangerous). EXAMPLE: Original: BUY 100 AAPL @ $150 LIMIT, Modify to: BUY 200 AAPL @ $151 LIMIT. Cancel-replace: Cancel original, Create new order (parent_order_id = original). In-place: Update same order (quantity: 100→200, price: $150→$151). Industry standard: Cancel-replace for safety and consistency.',
  },
  {
    id: 'order-management-system-mc-4',
    question:
      'What is the correct formula for calculating average fill price when an order receives multiple fills at different prices?',
    options: [
      'Average of all fill prices: (price1 + price2 + price3) / 3',
      'Weighted average by quantity: Σ(fill_quantity × fill_price) / total_filled_quantity',
      'Last fill price becomes the average price',
      'First fill price becomes the average price',
    ],
    correctAnswer: 1,
    explanation:
      'Average Fill Price Calculation: WRONG: Simple average of prices: (Fill 1: $100, Fill 2: $101, Fill 3: $99) / 3 = $100. Problem: Ignores quantity (what if Fill 1 was 1000 shares, Fill 3 was 10 shares?). CORRECT: Weighted average by quantity: Formula: avg_price = Σ(fill_qty × fill_price) / total_filled_qty. Example: Fill 1: 50 shares @ $100 = $5,000, Fill 2: 30 shares @ $101 = $3,030, Fill 3: 20 shares @ $99 = $1,980. Total: 100 shares, $10,010 → avg = $10,010 / 100 = $100.10. IMPLEMENTATION (Running Total): Start: filled_qty = 0, avg_price = 0. Fill 1 (50 @ $100): total_cost = 0 + (50 × $100) = $5,000, avg_price = $5,000 / 50 = $100.00. Fill 2 (30 @ $101): total_cost = $5,000 + (30 × $101) = $8,030, avg_price = $8,030 / 80 = $100.375. Fill 3 (20 @ $99): total_cost = $8,030 + (20 × $99) = $10,010, avg_price = $10,010 / 100 = $100.10. WHY IT MATTERS: P&L calculation depends on average price. Wrong average → wrong P&L → regulatory issues. Commission calculation may depend on notional value. Regulatory reporting requires accurate fill prices.',
  },
  {
    id: 'order-management-system-mc-5',
    question:
      'What is the purpose of maintaining an execution_id (exchange-provided unique identifier) in the fills table with a UNIQUE constraint?',
    options: [
      'To make database queries faster',
      'To prevent duplicate fill processing when exchanges send the same fill multiple times due to network retries',
      'To track which exchange executed the fill',
      'To generate unique IDs for reporting purposes',
    ],
    correctAnswer: 1,
    explanation:
      'Execution ID for Fill Deduplication: PROBLEM: Exchanges may send same fill multiple times: Network retry: Fill message lost → exchange resends. Failover: Primary server crashes → backup resends fills. Without dedup: Process same fill twice → double-count filled quantity → overfill → position error. EXAMPLE: Order: 100 shares AAPL, Fill A: 50 shares (execution_id="EXEC-12345") received twice. Without dedup: Process fill 1: filled_qty = 0 + 50 = 50, Process fill 2: filled_qty = 50 + 50 = 100 (WRONG! Order should still be 50 filled). Position now thinks fully filled when only 50 executed. SOLUTION: execution_id UNIQUE constraint: On receive fill, check: SELECT * FROM fills WHERE execution_id = "EXEC-12345". If exists: Ignore (duplicate). If not exists: Process fill, INSERT with execution_id. Database prevents duplicates automatically (UNIQUE violation on retry). IMPLEMENTATION: try: db.insert_fill(fill), process_fill(fill) except UniqueViolationError: log("Duplicate fill detected: {fill.execution_id}"), return "ALREADY_PROCESSED". IMPORTANCE: Prevents position tracking errors (catastrophic in production). All professional trading systems implement fill deduplication. Alternative: Idempotency keys for API calls (same concept).',
  },
];
