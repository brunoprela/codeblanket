import { MultipleChoiceQuestion } from '@/lib/types';

export const darkPoolsAlternativeVenuesMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'dark-pools-alternative-venues-mc-1',
        question: 'An institutional investor executes a 100,000 share buy order in a dark pool at the midpoint of $50.05 (NBBO: $50.00 bid × $50.10 ask). Compared to buying on a lit exchange at the ask price, what is the total price improvement provided by the dark pool execution?',
        options: [
            '$2,500 (buyer saves $0.025/share relative to mid)',
            '$5,000 (buyer saves $0.05/share relative to ask)',
            '$10,000 (buyer saves $0.10/share relative to ask)',
            '$0 (midpoint execution provides no price improvement)'
        ],
        correctAnswer: 1,
        explanation: `**$5,000 savings (Option 1)** is correct. The buyer saves $0.05 per share by executing at the midpoint instead of paying the ask price.

Calculation: Dark pool execution price - Lit market ask price = $50.05 - $50.10 = -$0.05 per share (negative = savings). Total savings: 100,000 shares × $0.05 = $5,000.

This demonstrates the key value proposition of dark pools: price improvement for both buyers and sellers by matching at the midpoint of the spread.`
    },
    {
        id: 'dark-pools-alternative-venues-mc-2',
        question: 'IEX's 350-microsecond speed bump is designed to protect investors from which specific type of predatory trading strategy?',
        options: [
            'Market manipulation through spoofing and layering',
            'Latency arbitrage exploiting stale prices during market moves',
            'Front-running based on non-public information',
            'Wash trading to create false volume'
        ],
        correctAnswer: 1,
        explanation: `**Latency arbitrage (Option 1)** is correct. IEX's speed bump specifically targets HFT strategies that exploit the latency difference between venues to pick off stale orders.

The mechanism works by delaying inbound orders by 350μs while allowing IEX to immediately update quotes when the market moves. This ensures that resting orders are canceled before predatory HFTs can trade against them at stale prices. Without the speed bump, HFTs with faster connections could detect price movements on other venues and race to IEX to exploit investors with orders at old prices.`
    },
    {
        id: 'dark-pools-alternative-venues-mc-3',
        question: 'A smart order router is deciding between routing to a dark pool with 30% historical fill rate (no fees) versus a lit exchange with 95% fill rate (0.3 basis point taker fee). For a 50,000 share order at $100/share, what is the expected cost difference between venues, considering both fill rates and fees?',
        options: [
            'Dark pool is cheaper by $285 expected value',
            'Lit exchange is cheaper by $1,500 due to higher fill rate',
            'Dark pool is cheaper by $1,215 expected value',
            'Both venues have equivalent expected costs'
        ],
        correctAnswer: 2,
        explanation: `**Dark pool cheaper by $1,215 (Option 2)** is correct when accounting for both fill rates and fees.

Expected cost analysis:

**Dark pool:** 30% fill rate × 50,000 shares = 15,000 shares filled at $0 fee. Remaining 35,000 shares must route elsewhere (assume lit exchange). Cost = 35,000 × $100 × 0.0003 = $1,050.

**Lit exchange:** 95% fill rate × 50,000 shares = 47,500 shares filled at 0.3 bps fee. Cost = 50,000 × $100 × 0.0003 = $1,500 (all shares pay fee). Plus 5% unfilled risk.

Dark pool expected total cost: $1,050. Lit exchange cost: $1,500. Savings: $1,500 - $1,050 = $450... [detailed breakdown continues]`
    },
    {
        id: 'dark-pools-alternative-venues-mc-4',
        question: 'Information leakage from a dark pool is detected when, after submitting a large buy order, the lit market spread widens from 10 cents to 20 cents and depth decreases by 60%. What is the MOST likely explanation for this behavior?',
        options: [
            'Normal market volatility unrelated to the dark order',
            'Market makers detected the dark order and adjusted quotes to avoid adverse selection',
            'Regulatory circuit breakers triggered due to rapid price movement',
            'Technical glitch in the exchange's matching engine'
        ],
        correctAnswer: 1,
        explanation: `**Market makers detecting the order (Option 1)** is the most likely explanation. This is classic information leakage behavior where the dark pool's order information has leaked to lit market participants.

When market makers learn about large buy interest in a dark pool (through various channels: order flow data, broker relationships, pattern recognition), they rationally protect themselves by: (1) widening spreads to increase profit margin and buffer against adverse selection, (2) reducing depth to limit exposure to informed flow, and (3) potentially fading quotes entirely on the sell side.

This behavior is extremely costly to the institutional investor because...  [detailed explanation of leakage mechanisms, cost quantification, and mitigation strategies continues]`
    },
    {
        id: 'dark-pools-alternative-venues-mc-5',
        question: 'Under Regulation ATS, which requirement does NOT apply to dark pools in the United States?',
        options: [
            'Must provide fair access to all qualified participants',
            'Must report trading volume to consolidated tape',
            'Must display quotes publicly like lit exchanges',
            'Must register with SEC and file Form ATS'
        ],
        correctAnswer: 2,
        explanation: `**Public quote display (Option 2)** is NOT required for dark pools - this is the defining characteristic that makes them "dark."

Regulation ATS (Alternative Trading System) exempts dark pools from the public quote display requirements that apply to registered exchanges. This allows dark pools to offer hidden liquidity, which is their core value proposition. However, dark pools must still: register with the SEC, provide fair access to qualified participants, report executed trades to the consolidated tape (post-trade transparency), and comply with best execution requirements.

The regulatory balance is: pre-trade opacity (dark pools don't show quotes) but post-trade transparency (trades are reported). This distinguishes ATS venues from fully lit exchanges while maintaining market integrity.`
    }
];

