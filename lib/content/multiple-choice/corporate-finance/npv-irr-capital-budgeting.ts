import { MultipleChoiceQuestion } from '@/lib/types';

export const npvIrrCapitalBudgetingMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'npv-irr-mc-1',
        question:
            'A project has cash flows of [-$100, $50, $70, -$30]. At a 10% discount rate, what should you conclude?',
        options: [
            'Accept the project (NPV = $-5.20)',
            'Reject the project (NPV = $-5.20)',
            'Use IRR instead due to non-conventional cash flows',
            'Calculate MIRR to make the decision',
        ],
        correctAnswer: 1,
        explanation:
            'Calculate NPV: NPV = -100 + 50/1.1 + 70/(1.1)² - 30/(1.1)³ = -100 + 45.45 + 57.85 - 22.54 = -$19.24 (negative). Decision rule: NPV < 0 → Reject project, regardless of cash flow pattern. While non-conventional cash flows may create multiple IRRs, NPV always works and gives clear answer. A negative NPV means project destroys value.',
    },
    {
        id: 'npv-irr-mc-2',
        question:
            'Project A requires $100 and returns $130 in one year (IRR = 30%). Project B requires $1,000 and returns $1,200 in one year (IRR = 20%). Required return is 15%. Which statement is correct?',
        options: [
            'Choose A because higher IRR (30% > 20%)',
            'Choose B because higher NPV despite lower IRR',
            'Choose A because better Profitability Index',
            'Indifferent—both exceed 15% hurdle rate',
        ],
        correctAnswer: 1,
        explanation:
            'Calculate NPVs: NPV_A = -100 + 130/1.15 = $13.04. NPV_B = -1000 + 1200/1.15 = $43.48. While Project A has higher IRR (30%), Project B creates more absolute value ($43.48 vs $13.04). For mutually exclusive projects, choose based on NPV, not IRR. The IRR only tells you return percentage; NPV tells you actual dollars of value created. Project B\'s larger scale generates more wealth for shareholders, even at lower percentage return.',
    },
    {
        id: 'npv-irr-mc-3',
        question:
            'A project has IRR of 18%. The company\'s WACC is 12%. What can you conclude?',
        options: [
            'Definitely accept the project',
            'Definitely reject the project',
            'Calculate NPV before deciding',
            'The project creates 6% excess return',
        ],
        correctAnswer: 2,
        explanation:
            'While IRR (18%) > WACC (12%) suggests acceptance, you should calculate NPV before deciding. Why? (1) If this is one of multiple mutually exclusive projects, you must compare NPVs. (2) Non-conventional cash flows may have multiple IRRs—the 18% might not be the relevant one. (3) Very long duration projects with distant cash flows can have IRR > WACC but NPV < 0 due to scale effects. Always calculate NPV as primary decision metric, use IRR as supplementary information.',
    },
    {
        id: 'npv-irr-mc-4',
        question:
            'Two projects have identical NPVs of $1M but different IRRs (Project X: 22%, Project Y: 18%). Discount rate is 12%. How should you choose?',
        options: [
            'Choose X due to higher IRR',
            'Either project—same NPV means same value creation',
            'Calculate Profitability Index to break the tie',
            'Choose Y because lower IRR means less risk',
        ],
        correctAnswer: 1,
        explanation:
            'If two projects have identical NPVs ($1M each), they create the same absolute value for shareholders. You should be indifferent between them from a pure financial perspective. The IRR difference (22% vs 18%) reflects different project characteristics (likely timing and scale of cash flows), but if NPVs are identical at your required return (12%), both add exactly $1M to firm value. Other factors (strategic fit, risk, implementation difficulty, resource requirements) should guide the final decision.',
    },
    {
        id: 'npv-irr-mc-5',
        question:
            'A manufacturing project requires $5M investment and generates $1.2M annually for 6 years. At 10% discount rate, the Profitability Index (PI) is:',
        options: [
            '0.048',
            '1.048',
            '1.44',
            '2.40',
        ],
        correctAnswer: 1,
        explanation:
            'PI = PV of future cash flows / Initial investment. PV of cash flows = $1.2M × [(1 - 1.1^-6) / 0.10] = $1.2M × 4.3553 = $5.226M. PI = $5.226M / $5M = 1.045 ≈ 1.048. Alternatively: PI = (NPV + Initial investment) / Initial investment. NPV = $5.226M - $5M = $0.226M. PI = ($0.226M + $5M) / $5M = 1.045. A PI > 1 means project creates value (accept). PI < 1 means value destruction (reject). PI is useful for ranking projects under capital rationing.',
    },
];

