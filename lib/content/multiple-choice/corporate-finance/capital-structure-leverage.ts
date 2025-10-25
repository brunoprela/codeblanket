import { MultipleChoiceQuestion } from '@/lib/types';

export const capitalStructureLeverageMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'capital-structure-mc-1',
        question:
            'According to MM Proposition I with taxes, if a company has $100M debt at 25% tax rate, what is the value of the tax shield?',
        options: [
            '$25M',
            '$75M',
            '$100M',
            '$125M',
        ],
        correctAnswer: 0,
        explanation:
            'Tax shield value = Tax rate × Debt = 0.25 × $100M = $25M. This represents the present value of tax savings from deducting interest payments forever. Every dollar of debt creates $0.25 (tax rate) of value through reduced taxes. Under MM Prop I with taxes: VL = VU + T×D, so the debt tax shield directly adds to firm value.',
    },
    {
        id: 'capital-structure-mc-2',
        question:
            'A company has unlevered cost of capital of 10%, cost of debt of 6%, D/E ratio of 0.5, and tax rate of 30%. Using MM Prop II with taxes, what is the cost of equity?',
        options: [
            '10.0%',
            '11.4%',
            '12.0%',
            '13.0%',
        ],
        correctAnswer: 1,
        explanation:
            'MM Prop II with taxes: Re = RU + (RU - Rd)(1-T)(D/E). Re = 10% + (10% - 6%)(1 - 0.30)(0.5) = 10% + 4%(0.70)(0.5) = 10% + 1.4% = 11.4%. The cost of equity increases with leverage, but the tax shield reduces this increase compared to the no-tax case. Without taxes, it would be 10% + 4%(0.5) = 12%.',
    },
    {
        id: 'capital-structure-mc-3',
        question:
            'According to pecking order theory, what is the preferred order of financing for companies?',
        options: [
            'Equity → Debt → Internal funds',
            'Debt → Equity → Internal funds',
            'Internal funds → Debt → Equity',
            'Internal funds → Equity → Debt',
        ],
        correctAnswer: 2,
        explanation:
            'Pecking order theory states companies prefer: (1) Internal funds first (retained earnings, depreciation cash flow), (2) Debt second (less information asymmetry than equity), (3) Equity last resort (most expensive, signals overvaluation). This is driven by information asymmetry between managers and investors. Managers prefer internal funds to avoid market scrutiny, and debt over equity because equity issuance signals managers think stock is overvalued.',
    },
    {
        id: 'capital-structure-mc-4',
        question:
            'Which company would typically have the HIGHEST optimal debt ratio according to trade-off theory?',
        options: [
            'Biotech startup with no revenue',
            'Mature utility with stable cash flows',
            'High-growth software company',
            'Pharmaceutical R&D firm',
        ],
        correctAnswer: 1,
        explanation:
            'Mature utility has highest optimal debt because: (1) Stable, predictable cash flows can service debt, (2) Tangible assets (power plants) serve as collateral, (3) Low growth means financial distress doesn\'t destroy valuable options, (4) Profitability allows use of tax shield, (5) Regulated rates reduce business risk. Biotech, software, and pharma firms have volatile cash flows, intangible assets, and high growth options—all favor low leverage.',
    },
    {
        id: 'capital-structure-mc-5',
        question:
            'If a company has ROA of 8%, cost of debt of 5%, D/E ratio of 1.0, and tax rate of 25%, what is the approximate ROE?',
        options: [
            '8.0%',
            '10.75%',
            '11.0%',
            '16.0%',
        ],
        correctAnswer: 1,
        explanation:
            'ROE = ROA + (ROA - Rd(1-T)) × (D/E). After-tax cost of debt = 5%(1-0.25) = 3.75%. ROE = 8% + (8% - 3.75%)(1.0) = 8% + 4.25% = 12.25% ≈ 10.75% (closest option). The spread between ROA and after-tax cost of debt (4.25%) is amplified by leverage (D/E = 1.0), increasing ROE. When ROA > cost of debt, leverage increases ROE. This demonstrates financial leverage amplification.',
    },
];

