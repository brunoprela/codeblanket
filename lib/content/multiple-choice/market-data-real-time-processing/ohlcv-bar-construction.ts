import { MultipleChoiceQuestion } from '@/lib/types';

export const ohlcvBarConstructionMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'ohlcv-bar-construction-mc-1',
        question:
            'You are building 1-minute bars from ticks. A tick arrives at 9:30:45.123 with price $150.25. Which bar does this tick belong to?',
        options: [
            '9:30:00 bar (floor to minute start)',
            '9:31:00 bar (ceiling to next minute)',
            '9:30:45 bar (exact timestamp)',
            '9:30:30 bar (round to nearest 30 seconds)',
        ],
        correctAnswer: 0,
        explanation:
            'Ticks are assigned to bars by flooring the timestamp to the bar interval start. Tick at 9:30:45 belongs to the 9:30:00-9:31:00 bar. Formula: bar_start = floor(timestamp / interval) × interval. For 1-minute bars: floor(9:30:45 / 60) × 60 = floor(570.75 / 60) × 60 = 9 × 60 = 540 seconds = 9:30:00. This ensures consistent bar assignment and prevents gaps. Never ceiling (creates overlap) or use exact timestamp (creates too many bars). Standard convention: Bar timestamp = start time, bar contains [start, start+interval) ticks. Example: 9:30:00 bar contains ticks from 9:30:00.000 to 9:30:59.999. The close price is the last tick before 9:31:00. Production systems must be consistent - if backtesting uses floor, live trading must too, otherwise strategy behavior differs.',
    },
    {
        id: 'ohlcv-bar-construction-mc-2',
        question:
            'A 1-minute bar has OHLC = ($150.00, $150.50, $149.50, $150.25). A late tick arrives at $151.00. How should you update the bar?',
        options: [
            'Update: O=$150.00, H=$151.00, L=$149.50, C=$151.00 (update high and close)',
            'Update: O=$151.00, H=$151.00, L=$149.50, C=$151.00 (update all)',
            'Reject the late tick (bar already closed)',
            'Create new bar with single tick',
        ],
        correctAnswer: 0,
        explanation:
            'Late ticks should update high/low if they exceed current values, and update close if they are the most recent tick chronologically. Since late tick price is $151.00 > $150.50 (current high), update high to $151.00. If the late tick\'s timestamp is after the previous close tick, also update close to $151.00. Never update open (first tick defines open). Don\'t reject late ticks unless extremely late (> 5 seconds), as they contain valid market information. However, maintain a "late tick window" - after this window, stop accepting late ticks for a bar (e.g., accept late ticks for 5 seconds after bar close). Production consideration: If updating historical bars, notify downstream systems (bars are typically considered immutable after publication). Some systems maintain two versions: preliminary bars (accept late ticks) and final bars (immutable). Bloomberg accepts late ticks for ~1 minute after bar close.',
    },
    {
        id: 'ohlcv-bar-construction-mc-3',
        question:
            'Volume-based bars with 10K share threshold: AAPL has 50K ticks/day at 500 shares/tick. How many bars are created per day?',
        options: [
            '2,500 bars (50K ticks × 500 shares / 10K threshold)',
            '5 bars (50K / 10K)',
            '50 bars (one per 1K ticks)',
            '390 bars (one per minute)',
        ],
        correctAnswer: 0,
        explanation:
            'Volume bars complete when cumulative volume reaches threshold. Daily volume: 50K ticks × 500 shares/tick = 25M shares. Number of bars: 25M shares / 10K shares/bar = 2,500 bars/day. This is much higher than time-based bars (390 per day = 6.5 hours × 60 minutes). Volume bars adapt to activity: More bars during high-volume periods (market open/close), fewer during lunch. Compare to time bars: Time bars = fixed 390/day. Volume bars = variable, activity-driven. For low-volume stocks (100 shares/tick): 50K × 100 = 5M shares → 5M / 10K = 500 bars/day (fewer than AAPL). Volume bars normalize activity across stocks and time periods. Research shows volume bars have better statistical properties (lower autocorrelation) than time bars. However, irregular time intervals make them harder to visualize and combine with time-based indicators (RSI expects equal time intervals).',
    },
    {
        id: 'ohlcv-bar-construction-mc-4',
        question:
            'What is VWAP (Volume-Weighted Average Price) for a bar with these ticks: ($150 × 100 shares), ($151 × 200 shares), ($149 × 300 shares)?',
        options: [
            '$149.83 (weighted by volume)',
            '$150.00 (simple average)',
            '$149.50 (median price)',
            '$150.17 (time-weighted)',
        ],
        correctAnswer: 0,
        explanation:
            'VWAP = (Σ price × volume) / Σ volume. Calculation: Numerator: ($150 × 100) + ($151 × 200) + ($149 × 300) = $15,000 + $30,200 + $44,700 = $89,900. Denominator: 100 + 200 + 300 = 600 shares. VWAP = $89,900 / 600 = $149.833. VWAP weights each price by its volume, giving more importance to large trades. This differs from simple average: ($150 + $151 + $149) / 3 = $150.00. VWAP is superior for bar representation because it reflects actual executed value. Use cases: (1) Benchmark for execution quality (traders aim to beat VWAP), (2) Support/resistance in technical analysis, (3) Fair value estimate for large orders. VWAP limitations: Sensitive to outliers, not applicable to bars with zero volume, biased toward high-volume ticks. Alternative: TWAP (Time-Weighted Average Price) gives equal weight to each time period. Production: Always calculate VWAP for bars, store alongside OHLC.',
    },
    {
        id: 'ohlcv-bar-construction-mc-5',
        question:
            'You build 5-minute bars by aggregating five 1-minute bars. Which formula is correct for the 5-minute bar OHLC?',
        options: [
            'O=first.open, H=max(all.high), L=min(all.low), C=last.close, V=sum(all.volume)',
            'O=avg(all.open), H=avg(all.high), L=avg(all.low), C=avg(all.close), V=avg(all.volume)',
            'O=max(all.open), H=max(all.high), L=min(all.low), C=min(all.close), V=sum(all.volume)',
            'O=first.open, H=first.high, L=first.low, C=last.close, V=sum(all.volume)',
        ],
        correctAnswer: 0,
        explanation:
            'Correct aggregation: Open = first bar\'s open (first price of period), High = maximum high across all bars (highest price reached), Low = minimum low across all bars (lowest price reached), Close = last bar\'s close (final price of period), Volume = sum of all volumes (total traded). Example: Five 1-min bars: Bar 1: O=150, H=151, L=149, C=150.5, V=1000. Bar 2: O=150.5, H=152, L=150, C=151.5, V=1200. Bar 3: O=151.5, H=151.5, L=150.5, C=151, V=800. Bar 4: O=151, H=153, L=151, C=152.5, V=1500. Bar 5: O=152.5, H=152.5, L=151.5, C=152, V=900. Aggregated 5-min bar: O=150 (Bar 1 open), H=153 (max of all highs), L=149 (min of all lows), C=152 (Bar 5 close), V=5400 (sum). Never average OHLC values (meaningless). Never use first bar\'s high/low only (ignores movement in later bars). This hierarchical construction ensures consistency between timeframes and reduces computational load.',
    },
];

