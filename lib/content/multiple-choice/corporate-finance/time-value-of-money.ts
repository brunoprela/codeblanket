import { MultipleChoiceQuestion } from '@/lib/types';

export const timeValueOfMoneyMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'tvm-mc-1',
        question:
            'You need $100,000 in 8 years for a down payment. You can earn 6% annually. Using quarterly compounding, how much must you invest today?',
        options: [
            '$62,741',
            '$62,317',
            '$61,763',
            '$61,391',
        ],
        correctAnswer: 3,
        explanation:
            'With quarterly compounding: PV = FV / (1 + r/m)^(m×n) = $100,000 / (1 + 0.06/4)^(4×8) = $100,000 / (1.015)^32 = $61,391. Note: Annual compounding would give $62,741. More frequent compounding reduces required initial investment because money grows faster.',
    },
    {
        id: 'tvm-mc-2',
        question:
            'A bond pays $50 every 6 months forever (perpetuity). If your required semi-annual return is 4%, what is the bond worth?',
        options: [
            '$625',
            '$1,250',
            '$2,500',
            '$5,000',
        ],
        correctAnswer: 1,
        explanation:
            'Perpetuity formula: PV = PMT / r. Here, PMT = $50 (semi-annual) and r = 4% (semi-annual). PV = $50 / 0.04 = $1,250. Common mistake: Using annual rate (8%) would give $625, but we must match payment frequency to rate frequency. Semi-annual payments require semi-annual rate.',
    },
    {
        id: 'tvm-mc-3',
        question:
            'You save $500/month for 30 years. Your account earns 7% APR compounded monthly. What will you have at retirement?',
        options: [
            '$180,000',
            '$400,000',
            '$566,764',
            '$612,204',
        ],
        correctAnswer: 3,
        explanation:
            'Future value of annuity: FV = PMT × [(1+r)^n - 1] / r. PMT = $500, r = 0.07/12 = 0.005833, n = 360 months. FV = $500 × [(1.005833)^360 - 1] / 0.005833 = $612,204. Note: You contributed only $180,000 (option A), but compound interest added $432,204! This demonstrates the power of long-term compounding.',
    },
    {
        id: 'tvm-mc-4',
        question:
            'A credit card advertises "18% APR with daily compounding." What is the Effective Annual Rate (EAR)?',
        options: [
            '18.00%',
            '18.81%',
            '19.56%',
            '19.72%',
        ],
        correctAnswer: 3,
        explanation:
            'EAR = (1 + APR/m)^m - 1. With daily compounding (m=365): EAR = (1 + 0.18/365)^365 - 1 = 0.1972 = 19.72%. The "true" cost is 19.72%, not the advertised 18%. This is why credit card debt is so expensive—daily compounding makes the effective rate much higher than the stated APR.',
    },
    {
        id: 'tvm-mc-5',
        question:
            'You have a $200,000 mortgage at 5% for 30 years with monthly payments. After 10 years of payments, approximately how much principal do you still owe?',
        options: [
            '$133,333 (2/3 of original)',
            '$157,000',
            '$172,000',
            '$185,000',
        ],
        correctAnswer: 2,
        explanation:
            'Monthly payment = $1,073.64. After 120 payments (10 years), remaining balance ≈ $172,000. Common misconception: Linear amortization would leave $133,333 (2/3 × $200K). But early payments are mostly interest! In years 1-10, you pay ~$70K interest and only ~$28K principal. Use formula: Remaining = PV × (1+r)^n - PMT × [(1+r)^n - 1]/r. Or create amortization schedule. After 10 years, you\'ve paid $128,837 total but only $28K went to principal.',
    },
];

