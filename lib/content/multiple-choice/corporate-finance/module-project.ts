import { MultipleChoiceQuestion } from '@/lib/types';

export const moduleProjectMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'module-project-mc-1',
        question:
            'In the TechDistributor project, which valuation method typically yields the HIGHEST valuation?',
        options: [
            'DCF analysis (standalone, no synergies)',
            'Comparable company analysis (trading multiples)',
            'LBO analysis (financial buyer)',
            'Strategic M&A analysis (with synergies)',
        ],
        correctAnswer: 3,
        explanation:
            'Strategic M&A analysis (with synergies) typically yields highest valuation because: Strategic buyers can realize operational synergies (revenue + cost). Synergies can be $500M-1B+ for transformative deals. Justify premium prices (10-12× EBITDA vs. 8-9× standalone). Ranking (low to high): LBO < DCF < Comps < Strategic M&A. LBO: Most conservative (leverage constraints, no synergies, 8-9×). DCF: Intrinsic value (no synergies, but forward-looking, 9-10×). Comps: Market-based (reflects minority stakes, 9-10×). Strategic M&A: Premium prices (includes synergies + control, 10-13×). In TechDistributor: LBO: 9× ($450M), DCF: 9.6× ($480M), Comps: 9-10× ($450-500M), Strategic: 10× ($500M).',
    },
    {
        id: 'module-project-mc-2',
        question:
            'You discover TechDistributor can improve working capital (free $50M cash) OR invest in a new warehouse ($50M CapEx, 18% IRR). Company WACC is 10%. Which should you prioritize?',
        options: [
            'Working capital improvement (free $50M cash)',
            'New warehouse investment ($50M, 18% IRR)',
            'Both equally valuable',
            'Neither—return $50M to shareholders',
        ],
        correctAnswer: 1,
        explanation:
            'Prioritize new warehouse investment (18% IRR > 10% WACC). NPV framework: Working capital: Frees $50M cash (PV = $50M). Opportunity: Invest at WACC (10%). NPV = $50M (one-time, no return). Warehouse: Invest $50M, earn 18% IRR. NPV = -$50M + PV(future cash flows at 10% discount). If IRR (18%) > WACC (10%), NPV > 0. NPV ≈ $50M × (0.18/0.10 - 1) = $50M × 0.80 = $40M (simplified). Optimal strategy: (1) Invest $50M in warehouse (highest NPV, +$40M). (2) Improve working capital to fund warehouse ($50M freed → $50M invested). (3) Don\'t return cash to shareholders if positive-NPV projects available. Key lesson: Working capital optimization unlocks capital for high-return investments. Prioritize NPV: Warehouse ($40M NPV) > WC improvement ($0 NPV, but enables warehouse).',
    },
    {
        id: 'module-project-mc-3',
        question:
            'In your TechDistributor LBO, EBITDA grows from $50M to $120M over 5 years. You buy and sell at 9× EBITDA. What is the primary source of equity value creation?',
        options: [
            'Multiple expansion (9× entry, higher exit)',
            'EBITDA growth ($50M → $120M)',
            'Deleveraging (debt paydown)',
            'Working capital optimization',
        ],
        correctAnswer: 1,
        explanation:
            'EBITDA growth ($50M → $120M, +140%) is the primary value driver. Value creation bridge: Entry EV = $50M × 9 = $450M. Exit EV = $120M × 9 = $1,080M. Increase = $630M. Sources: EBITDA growth: ($120M - $50M) × 9 = $630M (100% of EV increase!). Multiple expansion: 0 (bought and sold at 9×). Deleveraging: Converts debt to equity (doesn\'t change EV, but increases equity value). WC optimization: Minimal impact on EV (one-time cash, not recurring). For equity returns: EBITDA growth creates enterprise value. Deleveraging converts it to equity value (debt → equity). Example: Entry: $450M EV, $270M debt → $180M equity. Exit: $1,080M EV, $100M debt → $980M equity. Equity MOIC = $980M / $180M = 5.4×. Decompose: EBITDA growth → $630M EV increase. Deleveraging → $170M debt paid down (converts to equity). Total equity increase = $630M + $170M = $800M. But this is a simplification—both matter. Correct answer: EBITDA growth drives enterprise value creation (primary driver).',
    },
    {
        id: 'module-project-mc-4',
        question:
            'Your base case DCF values TechDistributor at $480M. Sensitivity analysis shows: Best case: $600M (WACC 8%, Terminal growth 4%). Worst case: $380M (WACC 12%, Terminal growth 2%). What does this tell you?',
        options: [
            'The valuation is highly uncertain; proceed with caution',
            'Use the midpoint ($490M) as fair value',
            'The company is undervalued at any price below $600M',
            'Sensitivity analysis is irrelevant; trust the base case',
        ],
        correctAnswer: 0,
        explanation:
            'High valuation range ($380-600M, ±25% from base) indicates high uncertainty. Proceed with caution. Interpretation: Terminal assumptions (WACC, growth) dominate valuation. Small changes (WACC ±2%, growth ±1%) → large valuation swings (±$100M+). This is NORMAL for DCF (terminal value often 60-80% of total). But highlights model sensitivity. Implications: Don\'t anchor on point estimate ($480M). Report range: $400-550M (exclude extremes). Triangulate with other methods (comps, LBO, precedents). Perform scenario analysis (bull/base/bear cases). Stress test downside (what if recession, margin compression?). Recommendation: Offer $420-450M (below base case, buffer for uncertainty). Walk if price > $500M (upper bound of reasonable range). Do NOT use midpoint ($490M) blindly (ignores distribution of outcomes). Do NOT trust best case ($600M) as fair value (too optimistic). Key lesson: Valuation is art + science. Always show sensitivity/range, not just point estimate. Triangulate multiple methods. Margin of safety crucial in uncertain situations.',
    },
    {
        id: 'module-project-mc-5',
        question:
            'After completing the TechDistributor project, which skill is MOST valuable for corporate finance roles?',
        options: [
            'Excel modeling (building complex formulas)',
            'Financial theory (MM propositions, CAPM)',
            'Judgment (interpreting results, making recommendations)',
            'Python programming (automating analysis)',
        ],
        correctAnswer: 2,
        explanation:
            'Judgment (interpreting results, making recommendations) is most valuable. Why: Technical skills (Excel, Python, theory) are necessary but not sufficient. Anyone can build a DCF model (formula is straightforward). Hard part: Choosing assumptions (revenue growth, margins, WACC, terminal growth). Interpreting results (is $480M valuation reasonable? Compare to comps, precedents.). Making trade-offs (prioritize synergies vs. working capital? Match strategic offer or pass?). Communicating to senior leadership (tell story, not just numbers). In TechDistributor project: Technical: Built DCF, LBO, WACC models (important foundation). Judgment: Decided to pass on $500M offer despite positive NPV (better capital allocation elsewhere). Recommended 40% leverage (balanced tax shield vs. distress risk). Prioritized EBITDA synergies over working capital (88% vs. 10% of value). Corporate finance is decision-making under uncertainty. Judgment > technical skills. Excel/Python: Commoditized (everyone can learn). Theory: Important foundation (but doesn\'t tell you what to do). Judgment: Comes from experience, pattern recognition, wisdom (hardest to develop). In practice: Junior roles: Excel/Python skills valued (analysts execute models). Senior roles: Judgment valued (principals/partners make decisions, deploy capital). Recommendation: Master technical skills (table stakes). Develop judgment through: Case studies, Real deals (internships, jobs), Mentorship (learn from experienced practitioners). Module 4 taught both: Tools (DCF, LBO, WACC) + Judgment (when to use, how to interpret, trade-offs). This is the path to corporate finance mastery.',
    },
];

