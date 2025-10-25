import { MultipleChoiceQuestion } from '@/lib/types';

export const capmBetaMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'capm-beta-mc-1',
        question:
            'A stock has beta of 1.5, risk-free rate is 4%, and market risk premium is 6%. What is the expected return according to CAPM?',
        options: [
            '10.0%',
            '13.0%',
            '15.0%',
            '19.0%',
        ],
        correctAnswer: 1,
        explanation:
            'CAPM: E(R) = Rf + β(Rm - Rf). E(R) = 4% + 1.5(6%) = 4% + 9% = 13%. Note: Market risk premium is already calculated as (Rm - Rf) = 6%, so don\'t add Rf again. Common mistake: 4% + 1.5(4% + 6%) = 19% is wrong because that would be double-counting the risk-free rate.',
    },
    {
        id: 'capm-beta-mc-2',
        question:
            'Company X has levered beta of 1.2, D/E ratio of 0.4, and tax rate of 25%. What is the unlevered beta?',
        options: [
            '0.92',
            '1.00',
            '1.14',
            '1.56',
        ],
        correctAnswer: 1,
        explanation:
            'Unlevering formula: βU = βL / [1 + (1-T)(D/E)]. βU = 1.2 / [1 + (1-0.25)(0.4)] = 1.2 / [1 + 0.75(0.4)] = 1.2 / [1 + 0.30] = 1.2 / 1.30 = 0.923 ≈ 0.92. Unlevered beta is always lower than levered beta (unless D/E = 0) because removing financial leverage reduces equity risk.',
    },
    {
        id: 'capm-beta-mc-3',
        question:
            'A portfolio consists of 60% Stock A (β=0.8) and 40% Stock B (β=1.4). What is the portfolio beta?',
        options: [
            '1.00',
            '1.04',
            '1.10',
            '1.18',
        ],
        correctAnswer: 1,
        explanation:
            'Portfolio beta is the weighted average of individual betas: βP = wA × βA + wB × βB = 0.60(0.8) + 0.40(1.4) = 0.48 + 0.56 = 1.04. This means the portfolio has slightly more risk than the market (β=1.0) but less than Stock B alone. Portfolio diversification doesn\'t eliminate systematic risk (beta), it only eliminates unsystematic risk.',
    },
    {
        id: 'capm-beta-mc-4',
        question:
            'If a stock has beta of 0, what is its expected return according to CAPM?',
        options: [
            'Zero (0%)',
            'The risk-free rate',
            'The market return',
            'Cannot be determined',
        ],
        correctAnswer: 1,
        explanation:
            'CAPM: E(R) = Rf + β(Rm - Rf). If β = 0: E(R) = Rf + 0(Rm - Rf) = Rf. A stock with beta of zero has no systematic risk (no correlation with market), so investors only require the risk-free rate. Example: Treasury bills have β ≈ 0 and return ≈ risk-free rate. This is why beta = 0 is called "risk-free" in CAPM framework.',
    },
    {
        id: 'capm-beta-mc-5',
        question:
            'Which statement about beta is FALSE?',
        options: [
            'Beta measures systematic (non-diversifiable) risk',
            'A stock with β=2.0 is twice as risky as a stock with β=1.0',
            'Negative betas are impossible in theory',
            'Portfolio beta is the weighted average of individual betas',
        ],
        correctAnswer: 2,
        explanation:
            'Negative betas ARE possible, though rare. Assets like gold or some hedge funds sometimes have negative beta (move opposite to market). The FALSE statement is "negative betas are impossible." All other statements are true: Beta measures systematic risk (statement A). Beta = 2.0 means stock moves twice as much as market on average (statement B). Portfolio beta = weighted average (statement D).',
    },
];

