import { MultipleChoiceQuestion } from '@/lib/types';

export const threeStatementModelMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'tsm-mc-1',
    question:
      'In a three-statement model, Depreciation & Amortization (D&A) of $50M appears on the income statement. How does this flow through to the other statements?',
    options: [
      'Cash Flow: -$50M in Operating CF; Balance Sheet: -$50M Accumulated Depreciation',
      'Cash Flow: +$50M in Operating CF; Balance Sheet: +$50M Accumulated Depreciation',
      'Cash Flow: +$50M in Operating CF; Balance Sheet: -$50M Cash, +$50M Accumulated Depreciation',
      'Cash Flow: No impact (non-cash item); Balance Sheet: +$50M Accumulated Depreciation',
    ],
    correctAnswer: 1,
    explanation:
      'Option 2 is correct. D&A is a non-cash expense that reduces net income on the income statement. In the cash flow statement, we ADD BACK the $50M in Operating Cash Flow because no actual cash left the company (non-cash expense). On the balance sheet, Accumulated Depreciation increases by $50M (contra-asset account), which reduces net PP&E. The full flow: Income Statement: -$50M D&A expense (reduces net income). Cash Flow Statement: +$50M add-back in Operating CF section (to reconcile net income to cash). Balance Sheet: +$50M Accumulated Depreciation (reduces PP&E net value). Cash on balance sheet does NOT decrease because D&A doesn\'t use cash. This is why profitable companies with high D&A (capital-intensive industries) can have strong cash flow despite lower net income.',
  },
  {
    id: 'tsm-mc-2',
    question:
      'A company reports Net Income of $100M. Accounts Receivable increased from $80M to $95M during the year. In the Cash Flow Statement, this change in AR should appear as:',
    options: [
      '+$15M in Operating Cash Flow (cash source)',
      '-$15M in Operating Cash Flow (cash use)',
      '+$15M in Investing Cash Flow (cash source)',
      'No impact on Cash Flow Statement',
    ],
    correctAnswer: 1,
    explanation:
      'Option 2 is correct: -$15M in Operating Cash Flow (cash use). When Accounts Receivable increases, it means the company made sales (recorded as revenue) but hasn\'t collected cash yet—customers owe more. This is a use of cash because the company provided goods/services but hasn\'t received payment. In the cash flow statement Operating section: Start with Net Income ($100M), then adjust for changes in working capital: "Increase in Accounts Receivable" = -$15M (reduces operating CF). Intuition: If AR goes up, cash goes down (relative to net income). If AR goes down, customers paid, cash goes up. Option 1 is wrong (wrong sign), Option 3 is wrong category (AR is working capital, part of operations, not investing), Option 4 misses the working capital adjustment that\'s critical to reconciling accrual income to cash flow.',
  },
  {
    id: 'tsm-mc-3',
    question:
      'In Year 1 of your model, the Balance Sheet shows Total Assets = $500M and Liabilities + Equity = $498M (doesn\'t balance). Which is the LEAST likely explanation?',
    options: [
      'Retained Earnings formula doesn\'t include Net Income from Income Statement',
      'Cash on Balance Sheet is hard-coded instead of linked to Cash Flow Statement',
      'Capital Expenditures were recorded on the Balance Sheet incorrectly',
      'The Depreciation rate on PP&E is too high compared to industry average',
    ],
    correctAnswer: 3,
    explanation:
      'Option 4 (depreciation rate too high) is LEAST likely to cause balance sheet not balancing. The depreciation rate affects the level of Accumulated Depreciation and therefore PP&E net value, but it doesn\'t break the fundamental accounting identity (Assets = L + E). Even if depreciation is 20% instead of 10%, assets decrease but the balance sheet still balances. Options 1, 2, and 3 are common causes of balance sheet not balancing: (1) If Retained Earnings formula is "=Opening_RE" instead of "=Opening_RE + Net Income - Dividends", equity is understated by the cumulative net income, creating a gap. (2) If Cash is hard-coded (e.g., =$100M) instead of "=CashFlow_Sheet_EndingCash", changes in cash flow don\'t update balance sheet, breaking the tie. (3) CapEx should increase PP&E (Gross) on balance sheet. If CapEx is $50M but PP&E only increases $30M, assets are understated. The balance sheet identity is MUST HOLD. Depreciation rates affect magnitude of accounts but not whether equation balances.',
  },
  {
    id: 'tsm-mc-4',
    question:
      'A company has projected EBITDA of $200M, CapEx of $80M, increase in Net Working Capital of $30M, and taxes of $25M. What is the Free Cash Flow?',
    options: ['$200M', '$120M', '$90M', '$65M'],
    correctAnswer: 3,
    explanation:
      'Option 4 is correct: $65M. Free Cash Flow (FCF) represents cash available to investors (debt and equity holders) after necessary investments. Formula: FCF = EBITDA - CapEx - ∆NWC - Taxes. (Note: D&A is already added back in EBITDA, don\'t double-count). Calculation: EBITDA: $200M. - CapEx: -$80M (cash outflow for PP&E). - ∆NWC: -$30M (working capital increase uses cash). - Taxes: -$25M. = FCF: $65M. Common mistakes: Option 1 ($200M) ignores all cash uses—just EBITDA. Option 2 ($120M) only subtracts CapEx, forgets NWC and taxes. Option 3 ($90M) subtracts CapEx and NWC but forgets taxes. FCF is the"gold standard" metric for valuation because it shows actual cash generated that can be: (1) Paid to debt holders (interest/principal), (2) Returned to equity holders (dividends/buybacks), (3) Kept as cash buffer. Low/negative FCF despite positive EBITDA (as in this case) is common for high-growth companies making heavy investments.',
  },
  {
    id: 'tsm-mc-5',
    question:
      'In a three-statement model with a circular reference (cash earns interest, interest affects net income, net income affects cash), which solution is most professional?',
    options: [
      'Delete interest income to break the circular reference',
      'Enable Excel iterative calculation to solve the circular reference automatically',
      'Hard-code cash balances for all projection years',
      'Build the model in Python instead to avoid Excel circular reference issues',
    ],
    correctAnswer: 1,
    explanation:
      'Option 2 (enable iterative calculation) is the most professional solution. Circular references in financial models are often unavoidable and legitimate—cash balance affects interest income, which flows through to net income and back to cash. This is real economics, not a modeling error. Excel\'s iterative calculation feature (File → Options → Formulas → Enable iterative calculation) solves this by repeatedly calculating until values converge (typically 100-1000 iterations, converges in <1 second). Set maximum iterations to 100 and convergence to 0.01. Option 1 (delete interest) loses realism—cash does earn interest in reality. Option 3 (hard-code) defeats purpose of dynamic model and requires manual recalculation if any assumption changes. Option 4 (Python) solves circular references well (via iterative solvers), but switching platforms just to avoid circular references is overkill when Excel handles it natively. Alternative professional solution: Restructure model to make cash a"plug" or balancing item calculated as Assets - (Liabilities + Equity), which breaks the circular dependency. But iterative calculation is simpler and standard practice in investment banking/private equity models.',
  },
];

