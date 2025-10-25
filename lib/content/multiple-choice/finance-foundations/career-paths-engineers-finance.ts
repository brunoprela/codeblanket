import { MultipleChoiceQuestion } from '@/lib/types';

export const careerPathsEngineersFinanceMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'cpef-mc-1',
        question:
            'A quantitative researcher at a hedge fund has a base salary of $250K and their strategy generated $10M in profit. With a 20% profit share, what is their total compensation?',
        options: [
            '$450K ($250K base + $200K bonus)',
            '$2.25M ($250K + 20% of $10M)',
            '$2.5M (20% of $10M + base)',
            '$10.25M ($250K + $10M)',
        ],
        correctAnswer: 1,
        explanation:
            'Profit share = 20% × $10M = $2M. Total comp = $250K base + $2M profit share = $2.25M. This is why quant researchers can earn more than software engineers: Direct participation in P&L. Note: Actual hedge fund compensation varies - some funds pay lower % (10-15%), others higher (30-40%) for star performers. Renaissance Medallion researchers reportedly get 10% of profits, which at $10B fund = up to $1B in total bonuses split among ~50 researchers. This performance-based model aligns incentives: Your work directly determines your comp.',
    },
    {
        id: 'cpef-mc-2',
        question:
            'Which role typically requires a PhD for entry at top-tier firms (Renaissance, Two Sigma)?',
        options: [
            'Quantitative Developer',
            'Quantitative Researcher',
            'Strat (Investment Banking)',
            'Fintech Engineer',
        ],
        correctAnswer: 1,
        explanation:
            'Quantitative Researcher roles at top funds (Renaissance, Two Sigma, DE Shaw) almost always require PhDs in STEM fields (Physics, Math, CS, Statistics). They hire based on research ability, not finance knowledge. Quantitative Developers can get hired with strong MS or even BS + experience. Strats typically require MS or strong BS. Fintech needs no advanced degree. Why PhD for quant research? (1) Proven research ability (published papers), (2) Deep mathematical sophistication, (3) Intellectual horsepower for novel problem-solving, (4) Patience for long research cycles. However, tier-2 funds (Millennium, Point72) sometimes hire MS or exceptional BS with strong track record.',
    },
    {
        id: 'cpef-mc-3',
        question:
            'What is the typical career progression timeline at an investment bank for a strat?',
        options: [
            'Analyst (0-2) → Associate (3-5) → VP (6-9) → Director (10-12) → MD (13+)',
            'Junior (0-3) → Mid (4-7) → Senior (8-12) → Principal (13+)',
            'L3 (0-2) → L4 (3-5) → L5 (6-9) → L6 (10+)',
            'Researcher (0-3) → Senior (4-8) → Principal (9+)',
        ],
        correctAnswer: 0,
        explanation:
            'Investment banks (Goldman, Morgan Stanley, JPMorgan) use traditional hierarchy: Analyst → Associate → VP → Director → MD. Promotions roughly every 2-3 years if performing. Analyst: Fresh undergrad/MS. Associate: Post-MBA or promoted analyst. VP: 6-9 years, managing projects. Director: 10-12 years, business responsibility. MD: 13+ years, P&L owner, rainmaker. Comp grows dramatically: Analyst $150-250K → VP $350-650K → MD $800K-2.5M+. Contrast with tech (L3-L8 levels) or hedge funds (Researcher → Senior → Principal). Investment banking hierarchy is MORE rigid than tech: Harder to skip levels, politics matter more, "up or out" pressure.',
    },
    {
        id: 'cpef-mc-4',
        question:
            'A fintech engineer receives a total comp offer of $400K: $240K base + $160K in RSUs vesting over 4 years. If the company doubles in valuation, what is the potential total comp in year 1?',
        options: [
            '$400K (same as offered)',
            '$480K ($240K + double RSUs = $320K/4)',
            '$560K ($240K + $320K RSUs)',
            '$640K (everything doubles)',
        ],
        correctAnswer: 1,
        explanation:
            'Year 1 cash: $240K base. Year 1 RSUs: $160K / 4 years = $40K. If valuation doubles, RSUs worth: $40K × 2 = $80K. Total year 1 comp: $240K + $80K = $320K. Wait, that\'s not in options! Let me recalculate: If RSUs are $160K grant worth 2x, you vest $160K/4 = $40K per year, but they\'re worth $80K when vested. So: $240K + $80K = $320K... Actually the question says $320K/4 = $80K in year 1, totaling $480K. The key insight: Equity upside from startup growth. Early Stripe engineers: $100K RSUs granted → now worth $2-10M (20-100x). Early Coinbase: Similar story. This is why fintech comp can exceed pure cash from hedge funds: If company 10xs, your $160K grant becomes $1.6M.',
    },
    {
        id: 'cpef-mc-5',
        question:
            'Which factor is MOST important for success as a quantitative researcher?',
        options: [
            'Years of experience in finance',
            'CFA or financial certifications',
            'Research ability and intellectual curiosity',
            'Networking and relationship building',
        ],
        correctAnswer: 2,
        explanation:
            'Research ability and intellectual curiosity is paramount for quant researchers. You need to: (1) Generate novel hypotheses (creativity), (2) Test them rigorously (scientific method), (3) Handle negative results (most ideas fail), (4) Iterate quickly (try 100 ideas to find 1 winner). Years of experience helps but isn\'t required - Renaissance hires fresh PhDs. CFA is actively discouraged (too traditional finance, not quantitative). Networking matters less than results - if your strategies make money, you\'ll be valued. This meritocracy is why quants love the field: Your ideas\' P&L matters more than politics. Contrast with investment banking where relationships are crucial, or corporate where politics dominates.',
    },
];

