import { MultipleChoiceQuestion } from '@/lib/types';

export const howTradingWorksMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'htw-mc-1',
        question:
            'You place a market buy order for 100 shares when the market shows $50.00 bid / $50.05 ask. At what price will you most likely be filled?',
        options: [
            '$50.00 (best bid price)',
            '$50.025 (mid-price)',
            '$50.05 (best ask price)',
            '$50.10 (worse than ask)',
        ],
        correctAnswer: 2,
        explanation:
            'Market buy orders execute at the ASK price ($50.05) because you\'re taking liquidity from sellers. Market sell orders execute at BID price. The spread ($0.05) is your transaction cost. With PFOF, you might get price improvement (e.g., $50.04), saving $1 on 100 shares. Key: Market orders guarantee execution but NOT price. In volatile markets, you might get filled significantly worse than quoted price (slippage).',
    },
    {
        id: 'htw-mc-2',
        question:
            'What does T+2 settlement mean?',
        options: [
            'Trade executes in 2 seconds',
            'Trade settles 2 business days after trade date',
            'Trade requires 2 confirmations',
            'Trade costs 2% commission',
        ],
        correctAnswer: 1,
        explanation:
            'T+2 = Trade date + 2 business days. Buy Monday → settles Wednesday. Cash and shares officially transfer at settlement. You can trade before settlement (but must have funds to settle). Pattern day trader rule: <$25K accounts limited to 3 day trades per 5 days. Good faith violation: Selling before settlement with insufficient settled cash. Crypto settles instantly (no T+2), which is one reason it\'s appealing for traders.',
    },
    {
        id: 'htw-mc-3',
        question:
            'Robinhood routes your 100-share order to Citadel Securities. Citadel pays Robinhood $0.002/share. How much does Robinhood earn?',
        options: [
            '$0.02 (2 cents total)',
            '$0.20 (20 cents total)',
            '$2.00 ($0.02 per share)',
            '$20.00 ($0.20 per share)',
        ],
        correctAnswer: 1,
        explanation:
            '$0.002/share × 100 shares = $0.20. Robinhood earns $0.20 from PFOF on this trade. At 1M orders/day averaging 100 shares: $0.20 × 1M = $200K daily = $50M annually from PFOF. This funds zero-commission trading. Citadel profits by capturing bid-ask spread ($0.01-0.02/share) minus PFOF cost. Controversial because: (1) Conflict of interest (broker incentivized to route for payment not best execution), (2) Opacity (users don\'t see the economics), (3) Banned in UK, Canada.',
    },
    {
        id: 'htw-mc-4',
        question:
            'You place limit buy order at $100 when stock is trading at $101. What happens?',
        options: [
            'Order fills immediately at $100',
            'Order fills immediately at $101 (market price)',
            'Order added to book, waits for $100 or better',
            'Order rejected (price too low)',
        ],
        correctAnswer: 2,
        explanation:
            'Limit buy at $100 when market at $101: Order posted to book, waits until price falls to $100 or lower. If price never reaches $100, order never fills. Advantage: Price control (won\'t pay > $100). Disadvantage: May miss the trade. Use limit orders when: (1) Building position over time (not urgent), (2) Illiquid stocks (wide spreads), (3) Want specific entry price. Use market orders when: (1) Need immediate execution, (2) Liquid stocks (tight spreads), (3) Exiting positions urgently.',
    },
    {
        id: 'htw-mc-5',
        question:
            'Order book shows: Buy orders at $100 (9:30:00am), $100 (9:30:05am), $99.99. Sell order at $100 arrives. Which buy order matches?',
        options: [
            '$99.99 (highest price priority)',
            'Both $100 orders (split equally)',
            '$100 (9:30:00am) - time priority',
            '$100 (9:30:05am) - most recent',
        ],
        correctAnswer: 2,
        explanation:
            'Price-time priority: (1) Best price wins (both $100 bids beat $99.99), (2) Earliest timestamp wins (9:30:00am before 9:30:05am). Order at 9:30:00am gets filled. This is how all exchanges match orders. If you want to ensure execution, you must either: (1) Offer better price, or (2) Get your order in first. High-frequency traders invest millions in speed (co-location, microwave towers) to win time priority. For retail investors: Doesn\'t matter much (milliseconds), but understanding helps debug why orders don\'t fill.',
    },
];

