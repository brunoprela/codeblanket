export const howTradingWorksQuiz = [
    {
        id: 'htw-q-1',
        question:
            'Design an order management system (OMS) that handles multiple order types (market, limit, stop, stop-limit). Cover: (1) order validation (sufficient funds, market hours, trading halts), (2) order book management (price-time priority), (3) execution logic for each order type, (4) partial fills handling, (5) order status tracking (pending, filled, canceled, rejected). Include edge cases: What if limit order at exact mid-price? What if stop triggered but no liquidity? How to handle modified orders? Provide complete code architecture.',
        sampleAnswer: `Order Management System implementation covering validation pipeline checking cash balance, buying power (margin), market status via API, order book using heaps for O(log n) insertion and O(1) best bid/ask retrieval, execution engine with different strategies per order type (market executes immediately against best quotes, limit posts to book if no match, stop monitors price and triggers market order, stop-limit triggers limit order), partial fill handling tracking filled_quantity separate from order_quantity allowing multiple executions per order, status state machine (PENDING → OPEN → PARTIALLY_FILLED → FILLED or CANCELED/REJECTED), edge cases include limit order at mid-price posts to buy side if buying (acts as aggressive), stop triggered with insufficient liquidity results in partial fill with remaining quantity canceled, order modifications treated as cancel-replace to avoid race conditions.`,
        keyPoints: [
            'Validation: Check cash >= order value, margin buying power if applicable, market open 9:30am-4pm ET, stock not halted',
            'Order book: Use heaps (priority queues) for O(log n) insert, O(1) best bid/ask, maintain price-time priority',
            'Execution: Market order walks book until filled, limit order posts if no match, stop monitors price and triggers',
            'Partial fills: Track filled_quantity vs total_quantity, allow multiple executions, calculate average fill price',
            'Edge cases: Mid-price limit acts aggressive, stop with no liquidity = partial fill, modifications = cancel-replace atomic',
        ],
    },
    {
        id: 'htw-q-2',
        question:
            'Analyze payment for order flow (PFOF): How does Citadel Securities profit from paying Robinhood $0.002/share for retail order flow? Calculate: (1) market maker profit per trade (bid-ask capture minus execution costs), (2) break-even order flow volume, (3) risk analysis (inventory risk, adverse selection), (4) price improvement requirements (SEC Rule 605), (5) comparison to exchange economics. Is PFOF good or bad for retail investors? Build quantitative model with realistic assumptions.',
        sampleAnswer: `PFOF economics model: Citadel profits by capturing bid-ask spread minus costs: Revenue per 100-share trade = $0.01 spread × 100 shares = $1.00, Costs = PFOF $0.20 (100 × $0.002) + execution $0.05 + technology $0.05 = $0.30, Profit = $1.00 - $0.30 = $0.70 per trade (70 bps). At 1M retail orders daily = $700K daily profit = $175M annually. Risks include: inventory risk (holding positions overnight, hedged with delta-neutral strategies costing ~10 bps), adverse selection minimal for retail (uninformed order flow), SEC Rule 605 requires price improvement (must beat NBBO by ≥$0.01, which Citadel does 80%+ of time). Exchange comparison: Sending to NYSE costs broker $0.003/share rebates but worse execution (+$0.02 wider spread) = net worse for customer. Conclusion: PFOF is net positive for retail (saves $1-2 per trade via price improvement) but controversial due to conflicts of interest and opacity. Analysis should consider: order flow quality (retail vs institutional), spread capture rates (varies by volatility), regulatory compliance costs.`,
        keyPoints: [
            'Market maker profit: Capture $0.01 bid-ask spread, pay $0.002 PFOF, net $0.008/share = $0.80 per 100 shares',
            'Break-even: Need spread capture rate >20% (PFOF cost / gross spread) to be profitable, actual is 80%+',
            'Risk: Inventory risk hedged with delta-neutral strategies, adverse selection low for retail (uninformed flow)',
            'Price improvement: SEC Rule 605 requires beating NBBO, Citadel typically gives $0.01 improvement (customer saves $1 per 100 shares)',
            'Verdict: Net positive for retail (free trading + price improvement > cost of PFOF), but lacks transparency',
        ],
    },
    {
        id: 'htw-q-3',
        question:
            'Design a settlement and reconciliation system for a brokerage. Cover: (1) T+2 settlement tracking (trade date vs settlement date), (2) pre-settlement trading (buying power calculation), (3) end-of-day reconciliation (trades vs positions vs cash), (4) corporate actions handling (dividends, splits, mergers), (5) error detection and resolution (breaks, DK trades). How do you handle: customer sells before settlement? Multiple trades same day? Failed settlements (buyer insufficient funds)? Include state diagrams and failure recovery.',
        sampleAnswer: `Settlement system design: T+2 tracking maintains trade_date, expected_settlement_date (T+2 business days excluding holidays), actual_settlement_date when DTCC confirms, pre-settlement trading allows selling before settlement BUT total sales must not exceed settled cash + buying_power, buying power calculation uses unsettled_cash × margin_rate (typically 50% for Reg T), end-of-day reconciliation compares: broker records vs DTCC records, expected positions (prior + trades) vs actual positions, expected cash (prior + sales - buys - fees) vs actual cash, discrepancies flagged as breaks requiring investigation, corporate actions subscribed from DTCC: dividends credited on pay_date, splits adjusted as stock_split_ratio (2:1 = double shares, half price), mergers handle as forced sale + acquisition, error detection includes: DK trades (don't know - counterparty doesn't recognize), quantity breaks (expected 100 shares, received 90), price breaks (expected $50.00, received $50.10), resolution workflow: auto-reconcile if <$10 discrepancy, manual review if >$10, escalate to DTCC if unresolved after 2 days. Edge cases: customer sells before T+2 settlement allowed if within buying power (otherwise good faith violation), multiple same-day trades netted for settlement (buy 100, sell 50 = settle +50 shares), failed settlement handled by buy-in (broker buys shares in market, charges customer + fees).`,
        keyPoints: [
            'T+2 tracking: Trade Monday settles Wednesday, maintain trade_date and settlement_date separately, allow pre-settlement trading within limits',
            'Buying power: Settled_cash + (unsettled_cash × 50%) + margin_loan - current_holdings = available buying power',
            'EOD reconciliation: Compare broker books vs DTCC, expected positions vs actual, flag breaks >$10 for manual review',
            'Corporate actions: Subscribe DTCC feed, credit dividends on pay_date, adjust splits automatically (2:1 = double shares, half price)',
            'Error handling: Auto-reconcile <$10 breaks, manual review >$10, buy-in if settlement fails (customer pays market price + fees)',
        ],
    },
];

