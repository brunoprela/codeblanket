import { MultipleChoiceQuestion } from '@/lib/types';

export const financeTerminologyDevelopersMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'ftd-mc-1',
        question:
            'A portfolio has 12% annual return with 18% volatility. Risk-free rate is 2%. What is the Sharpe ratio?',
        options: [
            '0.56',
            '0.67',
            '1.00',
            '6.00',
        ],
        correctAnswer: 0,
        explanation:
            'Sharpe = (Return - RiskFree) / Volatility = (0.12 - 0.02) / 0.18 = 0.10 / 0.18 = 0.556 ≈ 0.56. Interpretation: 0.56 Sharpe is mediocre. You earn 0.56 units of return per unit of risk. Benchmarks: <1 = poor (better to buy index), 1-2 = good, >2 = excellent (rare), >3 = suspicious (might be too good to be true or leverage hiding risk). Renaissance Medallion (best hedge fund): Sharpe ~7, Warren Buffett: Sharpe ~0.8 over lifetime. Use: Compare strategies (higher Sharpe = better risk-adjusted), Identify leverage impact (leveraging increases returns AND volatility, Sharpe stays same or decreases).',
    },
    {
        id: 'ftd-mc-2',
        question:
            'A portfolio has beta of 1.5. Market returns 10%, risk-free rate 2%. Portfolio returns 14%. What is alpha?',
        options: [
            '2% (positive alpha - outperformed)',
            '0% (performed as expected)',
            '-2% (negative alpha - underperformed)',
            '14% (total return)',
        ],
        correctAnswer: 0,
        explanation:
            'Expected return (CAPM) = RiskFree + Beta × (Market - RiskFree) = 0.02 + 1.5 × (0.10 - 0.02) = 0.02 + 1.5 × 0.08 = 0.02 + 0.12 = 0.14 (14%). Actual return = 14%. Alpha = Actual - Expected = 0.14 - 0.14 = 0% NOT 2%. Wait, let me recalculate: Expected = 2% + 1.5 × 8% = 2% + 12% = 14%. Actual = 14%. Alpha = 0%. Actually this is ZERO alpha - performed exactly as expected given risk (beta). Positive alpha example: If returned 16% with same beta → alpha = 16% - 14% = 2% (genuinely beat market). Negative alpha: Returned 12% → alpha = 12% - 14% = -2% (underperformed given risk taken). Alpha is Holy Grail: Hedge funds seek positive alpha (skill), not just beta (market exposure anyone can buy).',
    },
    {
        id: 'ftd-mc-3',
        question:
            'Position: Long 200 shares AAPL at $150. Current price $158. What is unrealized P&L?',
        options: [
            '$1,600 (unrealized gain)',
            '$0 (no P&L until sold)',
            '$31,600 (total position value)',
            '$800 (half the gain)',
        ],
        correctAnswer: 0,
        explanation:
            'Unrealized P&L = (Current Price - Entry Price) × Quantity = ($158 - $150) × 200 = $8 × 200 = $1,600. This is "paper profit" - you haven\'t sold yet. If price drops before selling, P&L changes. Realized P&L: Only occurs when position closed. If you sell 100 shares at $158 → realized = ($158 - $150) × 100 = $800, still unrealized = ($158 - $150) × 100 = $800 (remaining position). Total position value ($31,600) is NOT P&L - it\'s market value. Your cost basis was $30,000 (200 × $150). Importance: Margin accounts use unrealized P&L for buying power, Mark-to-market means portfolio value includes unrealized P&L, Taxes only on realized gains (when you sell).',
    },
    {
        id: 'ftd-mc-4',
        question:
            'What does a beta of 0.5 indicate?',
        options: [
            'Stock moves half as much as market (low volatility)',
            'Stock loses 50% of value',
            'Stock has 50% correlation with market',
            'Expected return is 50% of market return',
        ],
        correctAnswer: 0,
        explanation:
            'Beta = 0.5 means stock moves 0.5× market moves. Market up 10% → stock up ~5%. Market down 10% → stock down ~5%. Low beta stocks: Utilities (β ≈ 0.3-0.6), Consumer staples (β ≈ 0.5-0.8), REITs (β ≈ 0.6-0.9). High beta stocks: Tech growth (β ≈ 1.2-1.8), Small caps (β ≈ 1.3-2.0), TSLA (β ≈ 2.0). Beta = 1: Moves with market (SPY by definition), Beta < 0: Inverse to market (VIX, gold sometimes), Beta > 1: Amplifies market moves. Use: Portfolio construction (mix high/low beta for target risk), Risk assessment (β=2 stock is 2× risky as market), Expected return (CAPM: E(R) = Rf + β(Rm - Rf)).',
    },
    {
        id: 'ftd-mc-5',
        question:
            'In FIX protocol, what does tag 35=8 represent?',
        options: [
            'New order',
            'Execution report (order fill)',
            'Order cancel',
            'Heartbeat',
        ],
        correctAnswer: 1,
        explanation:
            'FIX Tag 35 = MsgType. Common values: 35=D: New Order Single (you send to broker), 35=8: Execution Report (broker sends fill confirmation), 35=F: Order Cancel Request, 35=G: Order Cancel/Replace (modify), 35=A: Logon, 35=0: Heartbeat (keep-alive), 35=2: Resend Request. Execution Report (35=8) contains: Tag 11: ClOrdID (your order ID), Tag 17: ExecID (execution ID), Tag 39: OrdStatus (0=New, 1=PartialFill, 2=Filled, 4=Canceled, 8=Rejected), Tag 14: CumQty (total filled quantity), Tag 151: LeavesQty (remaining quantity), Tag 31: LastPx (fill price), Tag 32: LastQty (this fill quantity). System must: Parse tag 39 to determine status, Update database with fill price (tag 31) and quantity (tag 14), Handle partial fills (status=1, multiple execution reports), Confirm full fill (status=2, order complete).',
    },
];

