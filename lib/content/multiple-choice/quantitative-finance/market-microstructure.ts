import type { MultipleChoiceQuestion } from '@/lib/types';

export const marketMicrostructureMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'market-microstructure-mc-1',
    question:
      'A stock has a bid-ask spread of $0.10. The bid price is $100.00. If the spread is composed of 30% order processing costs, 40% inventory risk, and 30% adverse selection, and volatility doubles while all else remains constant, what is the expected new spread (approximately)?',
    options: [
      '$0.10 (unchanged)',
      '$0.13 (30% increase)',
      '$0.16 (60% increase)',
      '$0.20 (100% increase)',
    ],
    correctAnswer: 1,
    explanation:
      'When volatility doubles, inventory risk scales with volatility (spread ∝ σ), so the inventory component doubles from 0.04 to 0.08. Adverse selection also increases (information asymmetry rises with volatility), conservatively increasing from 0.03 to 0.045 (50% increase). Processing costs remain constant at 0.03. New spread = 0.03 + 0.08 + 0.045 = $0.155, approximately $0.16, representing a 60% increase. This demonstrates that spread widening during volatility spikes is driven primarily by inventory risk and adverse selection, not processing costs.',
  },
  {
    id: 'market-microstructure-mc-2',
    question:
      'You execute a market order to buy 50,000 shares (0.5% of average daily volume) in a stock with 2% daily volatility. Using the Almgren-Chriss square-root model with eta=0.5, what is the expected market impact?',
    options: [
      '0.7 basis points',
      '2.2 basis points',
      '7.1 basis points',
      '22.4 basis points',
    ],
    correctAnswer: 2,
    explanation:
      'Using Almgren-Chriss: Impact = eta × volatility × √(Q/V). With eta=0.5, vol=0.02, Q/V=0.005, we get Impact = 0.5 × 0.02 × √0.005 = 0.01 × 0.0707 = 0.000707 = 7.1 basis points. This square-root relationship shows that doubling order size only increases impact by √2 = 1.41×, not 2×. For this size (0.5% ADV), 7 bps is material and justifies using an algo execution strategy rather than a simple market order.',
  },
  {
    id: 'market-microstructure-mc-3',
    question:
      'An order book shows 5,000 shares bid within 3 ticks and 8,000 shares offered within 3 ticks. What is the order imbalance ratio, and what does it predict?',
    options: [
      '+0.23, predicts upward price movement',
      '-0.23, predicts downward price movement',
      '+0.60, predicts upward price movement',
      '-0.60, predicts downward price movement',
    ],
    correctAnswer: 1,
    explanation:
      'Imbalance = (Bid - Ask) / (Bid + Ask) = (5000 - 8000) / (5000 + 8000) = -3000 / 13000 = -0.23. A negative imbalance indicates ask-heavy order book (more selling pressure), which predicts near-term downward price movement with 60-70% accuracy. This signal is exploited by HFT market makers who adjust quotes downward (lower both bid and ask) in anticipation of price decline, avoiding adverse selection risk of buying at stale prices before the decline.',
  },
  {
    id: 'market-microstructure-mc-4',
    question:
      'A trader observes microstructure noise causing observed high-frequency returns to have 60% higher variance than fundamental returns. What is the primary source of this noise, and how should it be filtered?',
    options: [
      'Information asymmetry; filter using volume-weighted prices',
      'Bid-ask bounce; filter using mid-quote prices instead of last trade',
      'Market impact; filter using arrival prices',
      'Order processing delays; filter using time-weighted averages',
    ],
    correctAnswer: 1,
    explanation:
      'Microstructure noise at high frequencies is primarily caused by bid-ask bounce-prices alternate between bid and ask as trades switch between buyer-initiated (at ask) and seller-initiated (at bid), creating artificial volatility even when fundamental value is constant. The solution is to use mid-quote prices ((bid + ask)/2) instead of last trade price, which filters out the bounce and better represents fundamental value. Volume-weighted or time-weighted prices still include bounce effects, and while market impact contributes to noise, bid-ask bounce is the dominant source at high frequencies.',
  },
  {
    id: 'market-microstructure-mc-5',
    question:
      'During the 2010 Flash Crash, the Dow dropped 1,000 points in minutes before recovering. What microstructure phenomenon primarily caused this extreme volatility?',
    options: [
      'Adverse selection by informed traders exploiting private information',
      'HFT market makers withdrawing liquidity (reducing quotes), creating illiquidity spiral',
      'Order processing delays causing stale quotes and latency arbitrage',
      'Widening bid-ask spreads due to increased inventory risk from volatility',
    ],
    correctAnswer: 1,
    explanation:
      'The Flash Crash was primarily caused by HFT market makers withdrawing liquidity when volatility spiked. As prices fell rapidly, HFTs cancelled their quotes to avoid adverse selection and inventory risk, removing displayed liquidity from the order book. This created an illiquidity spiral: reduced liquidity → wider spreads → more price impact from incoming orders → higher volatility → more HFT withdrawals. The "ghost liquidity" phenomenon (liquidity that disappears when needed most) demonstrated the fragility of HFT-provided liquidity during stress. While spread widening occurred, it was a consequence rather than cause of the HFT withdrawal.',
  },
];
