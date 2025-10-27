import { MultipleChoiceQuestion } from '@/lib/types';

export const dividendDiscountModelMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'ddm-mc-1',
    question:
      'A stock pays $2 dividend currently. Dividends grow 5% annually forever. Required return is 10%. Using Gordon Growth Model, what is the stock price?',
    options: ['$20', '$40', '$42', '$50'],
    correctAnswer: 2,
    explanation:
      'Option 3 is correct: $42. Gordon Growth: P0 = D1 / (r - g). D1 = $2 × 1.05 = $2.10 (next year dividend). P0 = $2.10 / (0.10 - 0.05) = $2.10 / 0.05 = $42. Common mistakes: Option 1 ($20) uses D0 instead of D1: $2 / 0.10 = $20 (wrong—must grow dividend 1 year). Option 2 ($40) forgets to grow: $2 / 0.05 = $40 (used D0 = $2, not D1 = $2.10). Option 4 ($50) uses wrong denominator. Key: Gordon Growth needs NEXT YEAR dividend (D1), not current (D0).',
  },
  {
    id: 'ddm-mc-2',
    question:
      'In Gordon Growth Model P = D/(r-g), what happens to stock price if required return (r) increases from 10% to 12%, holding all else equal?',
    options: [
      'Price increases (lower discount rate effect)',
      'Price decreases (higher discount rate)',
      'Price unchanged (r and g offset)',
      'Cannot determine without knowing g',
    ],
    correctAnswer: 1,
    explanation:
      'Option 2 is correct: Price decreases. Formula: P = D / (r - g). If r increases, denominator (r - g) increases → P decreases. Example: D = $2, g = 5%. At r = 10%: P = $2 / (0.10 - 0.05) = $40. At r = 12%: P = $2 / (0.12 - 0.05) = $28.57 (price drops 29%). Intuition: Higher required return = investors demand higher return = willing to pay less for same dividend stream. This is inverse relationship: r ↑ → P ↓.',
  },
  {
    id: 'ddm-mc-3',
    question:
      'Which company is MOST appropriately valued using Dividend Discount Model?',
    options: [
      'Amazon (pays no dividend, reinvests in growth)',
      'Utility company (pays 5% dividend yield, stable)',
      'Tesla (high growth, no dividend)',
      'Biotech startup (no revenue, no dividend)',
    ],
    correctAnswer: 1,
    explanation:
      'Option 2 is correct: Utility company. DDM works best for: (1) Mature companies with stable, predictable dividends. (2) High dividend payout ratios (60%+). (3) Low growth (3-5% annually). Utilities fit perfectly: Regulated, stable cash flows, pay 60-80% of earnings as dividends, grow slowly (GDP-ish). Options 1, 3, 4 (Amazon, Tesla, biotech) pay NO dividends—DDM gives value of $0 (nonsensical). Must use DCF for growth/tech companies.',
  },
  {
    id: 'ddm-mc-4',
    question:
      'In two-stage DDM, terminal value typically represents what percentage of total stock value?',
    options: ['10-20%', '30-40%', '50-70%', '80-90%'],
    correctAnswer: 2,
    explanation:
      'Option 3 is correct: 50-70%. Two-stage DDM: Phase 1 (high growth, years 1-5): Contributes 30-50% of value. Phase 2 (stable growth, perpetuity): Contributes 50-70% of value (terminal value). Similar to DCF where terminal value = 60-80% of EV. Why so high? Perpetuity captures infinite years vs only 5 years of high growth. If terminal value >80%, over-reliant on perpetuity assumptions (risky). If <40%, short high-growth period or very low stable growth.',
  },
  {
    id: 'ddm-mc-5',
    question:
      'A stock is valued at $50 using DDM (4% dividend yield). The company announces 50% dividend cut but uses savings for buybacks. How should valuation change?',
    options: [
      'Drop to $25 (dividends halved)',
      'Stay at $50 (total payout unchanged)',
      'Increase (buybacks are tax-efficient)',
      'Depends on tax rates of investors',
    ],
    correctAnswer: 3,
    explanation:
      'Option 3 is correct: Increase (buybacks are tax-efficient). Analysis: Dividends: $2/share (4% yield on $50 stock), taxed as ordinary income (~37% max rate). Buybacks: $2/share returned via repurchases, taxed as capital gains (~20% max rate) only when sold. After-tax value: Dividends: $2 × (1 - 0.37) = $1.26/share. Buybacks: $2 × (1 - 0.20) = $1.60/share (when realized). Buybacks deliver 27% more after-tax value! But: DDM may undervalue initially because models dividends only (need to adjust for total shareholder yield). Correct approach: Value total payout (div + buyback), adjust for tax efficiency → value should increase, not stay flat.',
  },
];
