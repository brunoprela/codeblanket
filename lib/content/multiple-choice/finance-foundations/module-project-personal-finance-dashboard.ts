import { MultipleChoiceQuestion } from '@/lib/types';

export const moduleProjectPersonalFinanceDashboardMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mpfd-mc-1',
      question:
        'Which Python library is best for building an interactive finance dashboard with charts?',
      options: [
        'Flask (web framework)',
        'Streamlit (interactive dashboards)',
        'Django (full web framework)',
        'requests (HTTP library)',
      ],
      correctAnswer: 1,
      explanation:
        'Streamlit is PURPOSE-BUILT for data dashboards. Code example: `import streamlit as st; st.title("Portfolio"); st.line_chart(data)`. Advantages: Pure Python (no HTML/CSS/JavaScript needed), Interactive widgets (sliders, dropdowns, date pickers), Auto-rerun on code change (instant feedback), Built-in charts (Plotly, Matplotlib), Free deployment (Streamlit Cloud). Alternatives: Dash (Plotly, similar but more complex), Panel (HoloViz, more flexible), Gradio (ML models), Jupyter notebooks (research, not production). Flask: General web framework (need to write frontend), Django: Full-stack framework (overkill for dashboard), Voilà: Convert Jupyter to web app. Production: Streamlit for internal dashboards (quick), React + FastAPI for customer-facing (scalable).',
    },
    {
      id: 'mpfd-mc-2',
      question:
        'For tracking portfolio performance, which return calculation method is most accurate when making deposits/withdrawals?',
      options: [
        'Simple return ((End - Start) / Start)',
        'Time-weighted return (TWRR)',
        'Dollar-weighted return (IRR)',
        'Arithmetic average of monthly returns',
      ],
      correctAnswer: 1,
      explanation:
        'Time-weighted return (TWRR) is GOLD STANDARD for portfolio performance because it eliminates effect of deposits/withdrawals. Example: Start $100K, deposit $50K mid-year, end $160K. Simple return: ($160K - $100K) / $100K = 60% (WRONG - inflated by deposit). TWRR: Break into periods: (1) $100K → $105K before deposit = 5% return, (2) $155K → $160K after deposit = 3.2% return. TWRR = (1.05 × 1.032) - 1 = 8.4% (correct - excludes deposit impact). IRR (dollar-weighted): Accounts for timing of cash flows, answers "what return did I personally earn?" (depends when you invested). TWRR answers "how well did strategy perform?" (independent of timing). Use TWRR for: comparing to benchmarks (S&P 500), evaluating money managers, fund performance. Use IRR for: personal return (your actual dollars), real estate projects, private equity.',
    },
    {
      id: 'mpfd-mc-3',
      question: 'What is the "4% rule" in retirement planning?',
      options: [
        'Save 4% of income each year',
        'Withdraw 4% of portfolio annually in retirement',
        'Earn 4% return in retirement',
        'Pay 4% fees to financial advisor',
      ],
      correctAnswer: 1,
      explanation:
        '4% rule: Withdraw 4% of initial portfolio value each year, adjust for inflation. Example: Retire with $1M → withdraw $40K year 1, $41.2K year 2 (inflation), etc. Success rate: 95% over 30 years (Trinity Study, 1998). Assumptions: 60/40 stocks/bonds, 30-year retirement, historical US returns. Failure modes: Retire into bear market (sequence of returns risk), Lower future returns (4% may be too high if returns <7%), Longer retirement (live to 100 = 40 years, not 30). Conservative: Use 3.5% ($35K per $1M), Aggressive: 5% ($50K per $1M but higher failure risk). Calculation: savings_needed = annual_spending / 0.04. If need $80K/year → need $80K / 0.04 = $2M. Alternatives: Dynamic spending (reduce spending in down years), Guardrails (cut spending if portfolio drops 20%), Annuities (guaranteed income for life).',
    },
    {
      id: 'mpfd-mc-4',
      question:
        'For tax efficiency, where should you hold bonds (taxable account or IRA)?',
      options: [
        'Taxable (bonds are tax-efficient)',
        'IRA (bonds are tax-inefficient)',
        "Doesn't matter (same result)",
        'Depends on bond type only',
      ],
      correctAnswer: 1,
      explanation:
        'Bonds in IRA because bond interest taxed as ORDINARY INCOME (up to 37%). Example: $10K bond yields 4% = $400 interest. Taxable account: $400 × 37% tax = $148 tax (keep $252 = 2.52% after-tax return). IRA: $400 × 0% tax now = $400 (4% return, pay tax on withdrawal later). Asset location hierarchy: IRA (tax-deferred): Bonds (ordinary income tax), REITs (ordinary income), High-dividend stocks, Taxable: Growth stocks (capital gains 20% < ordinary 37%), Index funds (low turnover, deferred gains), Municipal bonds (interest tax-free, ONLY in taxable). Roth IRA: Highest-growth assets (never taxed again). Exception: Municipal bonds in taxable (interest tax-free), not IRA (no benefit). Math: After-tax return = pre-tax return × (1 - tax_rate). Bonds: 4% × (1 - 0.37) = 2.52% in taxable. Stocks: 10% × (1 - 0.20) = 8% in taxable (long-term cap gains). Bonds suffer more in taxable → put in IRA.',
    },
    {
      id: 'mpfd-mc-5',
      question: 'What is a "wash sale" and why does it matter?',
      options: [
        'Selling stock underwater (at a loss)',
        'Selling then repurchasing within 30 days (loss disallowed)',
        'Selling stock after 1 year (long-term gains)',
        'Selling all positions (wash out portfolio)',
      ],
      correctAnswer: 1,
      explanation:
        'Wash sale rule (IRS): If you sell stock at a loss and repurchase "substantially identical" security within 30 days (before or after), loss is DISALLOWED for taxes. Example: Jan 1: Buy 100 AAPL @ $150 ($15K). Mar 1: Sell 100 AAPL @ $120 ($12K, $3K loss). Mar 15: Buy 100 AAPL @ $125 ($12.5K). Result: $3K loss disallowed (repurchased within 30 days). Loss added to cost basis of new position: new_cost_basis = $125 + ($150 - $120) = $155. Why: Prevents "selling for tax purposes" without actually exiting position. How to avoid: Wait 31 days before repurchasing, Buy similar (not identical) security (sell VTI, buy ITOT), Use different account (sell in taxable, buy in IRA - technically allowed but controversial). Substantially identical: AAPL = AAPL (same stock = yes), VTI ≠ ITOT (different funds = no), AAPL call option = similar to AAPL stock (IRS says yes). Tax-loss harvesting: Intentionally sell losers, buy similar (harvest losses, maintain exposure), claim $3K loss against income annually.',
    },
  ];
