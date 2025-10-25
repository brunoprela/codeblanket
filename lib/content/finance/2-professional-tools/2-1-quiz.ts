export const quizQuestions = [
    {
        id: '2-1-q1',
        question: 'A financial analyst needs to calculate the monthly payment on a $750,000 loan with a 6.5% annual interest rate over 25 years. Which Excel formula is correct?',
        options: [
            '=PMT(6.5%, 25, 750000)',
            '=PMT(6.5%/12, 25*12, -750000)',
            '=PMT(0.065/12, 300, 750000)',
            '=FV(6.5%/12, 25*12, 750000)',
            '=PV(6.5%, 300, -750000)'
        ],
        correctAnswer: 1,
        explanation: 'The correct formula is =PMT(6.5%/12, 25*12, -750000). You must divide the annual rate by 12 for monthly periods, multiply years by 12 for total periods, and the principal should be negative (representing an outflow). This returns approximately $5,109.46 per month.'
    },
    {
        id: '2-1-q2',
        question: 'When building a three-statement financial model, which color coding convention is standard in investment banking for formula cells?',
        options: [
            'Blue text - formulas can be modified',
            'Green text - key outputs and formulas',
            'Black text - formulas should not be changed',
            'Red text - formulas link to other workbooks',
            'Purple text - calculated formulas'
        ],
        correctAnswer: 2,
        explanation: 'Black text is the standard convention for formula cells that should not be modified. Blue text is for hard-coded inputs (assumptions), green for key outputs/summaries, and red for external links. This color coding helps prevent errors and makes models easier to audit.'
    },
    {
        id: '2-1-q3',
        question: 'You have a dataset with 50,000 stock prices and need to calculate portfolio-level Value at Risk (VaR) using Monte Carlo simulation with 10,000 iterations. Which tool is most appropriate?',
        options: [
            'Excel with standard formulas',
            'Excel with VBA macros',
            'Python with pandas and numpy',
            'Excel pivot tables',
            'Excel Goal Seek'
        ],
        correctAnswer: 2,
        explanation: 'Python with pandas and numpy is most appropriate. Excel struggles with large datasets (50K+ rows) and complex simulations. Monte Carlo with 10,000 iterations on a portfolio would create 500 million calculations, which Excel handles poorly. Python executes this in seconds with proper memory management.'
    },
    {
        id: '2-1-q4',
        question: 'In a DCF model, you want to reference the WACC assumption (cell B15 on the Assumptions sheet) in a way that it always points to that exact cell when copied across rows and columns. Which reference style is correct?',
        options: [
            'Assumptions!B15',
            'Assumptions!$B15',
            'Assumptions!B$15',
            'Assumptions!$B$15',
            'Assumptions![B15]'
        ],
        correctAnswer: 3,
        explanation: 'Assumptions!$B$15 uses absolute referencing ($ before both column and row), ensuring the reference never changes when copied. This is critical in financial models where assumptions must be consistently referenced across multiple projections. Without absolute references, formulas would incorrectly adjust to different cells.'
    },
    {
        id: '2-1-q5',
        question: 'A private equity analyst is building an LBO model with hidden rows for detailed calculations. Which function should replace SUM() to ensure subtotals only include visible rows?',
        options: [
            '=SUM(A1:A100)',
            '=SUBTOTAL(109, A1:A100)',
            '=SUMIF(A1:A100, ">0")',
            '=AGGREGATE(1, A1:A100)',
            '=SUMVISIBLE(A1:A100)'
        ],
        correctAnswer: 1,
        explanation: '=SUBTOTAL(109, A1:A100) is correct. The 109 function number represents SUM while ignoring hidden rows. SUBTOTAL is specifically designed for this purpose. Regular SUM() includes hidden rows, which would produce incorrect totals. AGGREGATE could also work but SUBTOTAL is the standard for this use case in financial models.'
    }
];

