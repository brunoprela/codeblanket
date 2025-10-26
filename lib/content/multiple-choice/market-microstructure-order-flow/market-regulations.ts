import { MultipleChoiceQuestion } from '@/lib/types';

export const marketRegulationsMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'market-regulations-mc-1',
        question: 'Under Reg NMS Rule 611 (Order Protection Rule), a broker receives a buy order for 1,000 shares. The NBBO is $100.00 bid (NYSE) × $100.10 ask (NASDAQ). The broker executes the order at $100.15 on Cboe. Is this a trade-through violation?',
        options: [
            'Yes, violated by buying at $100.15 when NASDAQ offered at $100.10',
            'No, executions above NBBO ask are always allowed',
            'Yes, but only if NASDAQ had sufficient size to fill the order',
            'No, if the broker simultaneously routed an ISO to NASDAQ'
        ],
        correctAnswer: 0,
        explanation: `**Yes, this is a trade-through violation (Option 0)**. Reg NMS Rule 611 prohibits trading through protected quotes. The broker bought at $100.15 when NASDAQ was offering at $100.10—a clear violation of the Order Protection Rule. The broker should have either: (1) routed to NASDAQ first to execute at $100.10, or (2) used an Intermarket Sweep Order (ISO) to simultaneously sweep NASDAQ before executing elsewhere. Without using an ISO, this execution violated the rule and could result in fines. Protected quotes are the NBBO, and all trading centers must either route to the best price or use ISOs to comply.`
    },
    {
        id: 'market-regulations-mc-2',
        question: 'A trading algorithm places 10,000 sell orders over a 30-minute period, resulting in 150 actual executions. The order-to-trade ratio is 67:1. Which regulatory concern does this pattern raise?',
        options: [
            'Violation of sub-penny rule (Reg NMS Rule 612)',
            'Potential spoofing/layering market manipulation',
            'Exceeding maximum access fees (Reg NMS Rule 610)',
            'Circuit breaker trigger under LULD rules'
        ],
        correctAnswer: 1,
        explanation: `**Potential spoofing/layering (Option 1)** is the primary regulatory concern. A 67:1 order-to-trade ratio is extremely high and suggests the algorithm is placing orders without genuine intent to execute—a hallmark of spoofing. While there's no bright-line threshold, ratios above 10:1 typically trigger surveillance alerts, and 67:1 would warrant serious investigation. Regulators (SEC, CFTC, exchanges) monitor order-to-trade ratios as a key indicator of manipulative behavior. The firm would need to demonstrate legitimate reasons for this pattern (e.g., market-making with dynamic requoting) or face potential enforcement action.`
    },
    {
        id: 'market-regulations-mc-3',
        question: 'Under Reg NMS Rule 613 (Consolidated Audit Trail), what is the required timestamp precision for reporting high-frequency trading activity?',
        options: [
            '1 second precision is sufficient for all reporting',
            '1 millisecond for all equity trading',
            '50 nanoseconds for co-located high-frequency activity',
            'Timestamp precision requirements vary by trading venue'
        ],
        correctAnswer: 2,
        explanation: `**50 nanoseconds for HFT (Option 2)** is correct. CAT requirements specify clock synchronization to within 50 nanoseconds of the National Institute of Standards and Technology (NIST) atomic clock for Industry Members executing orders. This extreme precision is necessary to accurately reconstruct market events, identify market manipulation, and understand order flow in high-frequency environments where microseconds matter. Firms must use hardware timestamping and GPS/PTP synchronization to achieve this accuracy. Less stringent requirements apply to non-HFT activity, but HFT firms face the 50ns standard due to the speed of their operations.`
    },
    {
        id: 'market-regulations-mc-4',
        question: 'Under MiFID II, a dark pool in the EU exceeds 8% of total trading volume for a particular stock. What action is required?',
        options: [
            'Pay a fine but continue operating normally',
            'Suspend dark trading for that stock at that venue',
            'Reduce fees to incentivize lit trading',
            'Report to ESMA but no immediate action required'
        ],
        correctAnswer: 1,
        explanation: `**Suspend dark trading for that stock (Option 1)** is the required action under MiFID II's double volume cap mechanism. When a venue exceeds 8% of total volume for a stock in dark trading, that venue must suspend dark trading in that stock for 6 months. This is an automatic, mandatory suspension—not a discretionary penalty. Additionally, if dark trading across ALL venues exceeds 4% of total volume for a stock, dark trading is suspended at all venues EU-wide. These strict caps aim to preserve lit market transparency and prevent excessive fragmentation into dark pools. Venues must monitor volume percentages daily and comply immediately upon breach.`
    },
    {
        id: 'market-regulations-mc-5',
        question: 'A trader places a large sell order at $100.00, then immediately places 20 buy orders at $99.95, $99.94, $99.93, ... down to $99.76, and cancels them all 200 milliseconds later without execution. This pattern is repeated 50 times in one hour. This is MOST likely an example of:',
        options: [
            'Legitimate market making activity',
            'Layering to manipulate the price downward',
            'Arbitrage between correlated securities',
            'Algorithmic error requiring a kill switch'
        ],
        correctAnswer: 1,
        explanation: `**Layering (Option 1)** is the manipulation technique being used here. The trader is creating a false impression of buy-side demand by placing numerous layered buy orders below the market, intending to push the market price down toward their sell order at $100.00. Once the sell order executes (or market moves favorably), all the buy layers are canceled. This is illegal market manipulation under SEC Rule 10b-5 and Dodd-Frank anti-manipulation provisions. The pattern indicators are: (1) large number of orders on opposite side from actual intent, (2) orders at multiple price levels ("layers"), (3) quick cancellation without execution, (4) repetitive pattern. This would trigger immediate surveillance alerts and likely result in enforcement action. Notable cases include fines of $10M+ for layering violations.`
    }
];

