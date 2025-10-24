import { MultipleChoiceQuestion } from '../../../types';

export const buildingAiPoweredExcelEditorMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'bcap-bapee-mc-1',
      question:
        'How should an AI Excel formula generator handle ambiguous requests like "calculate growth"?',
      options: [
        'Generate any growth formula',
        'Ask clarifying questions: "Year-over-year (YoY) or month-over-month (MoM)?"',
        'Reject the request',
        'Always assume YoY growth',
      ],
      correctAnswer: 1,
      explanation:
        'Handle ambiguity with clarification: (1) Detect ambiguous intent (LLM classifies: "growth" could mean YoY, MoM, QoQ, CAGR), (2) Ask specific question with options, (3) Generate formula after user selects. Alternative: Generate multiple candidates, let user choose. Never guess - wrong formula wastes user time and reduces trust. Clarification takes 5s but ensures correctness.',
    },
    {
      id: 'bcap-bapee-mc-2',
      question:
        'What is the best approach for understanding interconnected financial models across multiple sheets?',
      options: [
        'Only analyze one sheet at a time',
        'Hybrid: AST parsing for structure/dependencies + LLM for semantic understanding',
        'Pure LLM analysis',
        'Ignore sheet relationships',
      ],
      correctAnswer: 1,
      explanation:
        'Hybrid approach: (1) AST parsing: Extract all formulas, build dependency graph (A1 depends on B1, C1), detect circular references, topological sort for calculation order - 100% accurate, free, (2) LLM: Analyze semantics ("This calculates revenue", "This is cost structure"), explain business logic - provides context. Pure LLM can misunderstand complex formulas or hallucinate dependencies. Together: precision + insight.',
    },
    {
      id: 'bcap-bapee-mc-3',
      question: 'How should Excel automation workflows be secured?',
      options: [
        'Store API keys in plain text',
        'Encrypt API keys (AES-256), OAuth2 for email, sandbox execution, audit logs',
        'No security needed',
        'Let users handle security',
      ],
      correctAnswer: 1,
      explanation:
        "Multi-layer security: (1) API keys: Encrypted in database (AES-256), never shown in logs, user provides via secure form, (2) Email: OAuth2 (Google/Microsoft), request only send permission, user can revoke, (3) Sandbox: Workflows run in isolated environment, rate limited, can't access other users' data, (4) Audit log: Track all executions, data accessed, emails sent (compliance). This prevents key leakage and unauthorized actions.",
    },
    {
      id: 'bcap-bapee-mc-4',
      question: 'When should Excel formulas be validated after AI generation?',
      options: [
        'Never validate',
        'Always: Check syntax (balanced parens), test with sample data, verify result type matches intent',
        'Only check syntax',
        'Trust AI completely',
      ],
      correctAnswer: 1,
      explanation:
        'Multi-stage validation: (1) Syntax check: Balanced parentheses, valid function names, correct argument count, (2) Test execution: Run formula on sample data, check for errors (#REF, #VALUE, #DIV/0), (3) Type check: If intent is "sum", result should be number not text, (4) Edge cases: Test with empty cells, zeros (division), (5) Explain back: Generate plain English explanation, user verifies understanding. Catches 95% of errors before user encounters them.',
    },
    {
      id: 'bcap-bapee-mc-5',
      question:
        'How should cell references be determined in generated formulas?',
      options: [
        'Always use absolute references ($A$1)',
        'Auto-detect: Relative (A1) for flexibility, absolute ($A$1) for constants/headers based on context',
        'Always use relative references',
        'Random selection',
      ],
      correctAnswer: 1,
      explanation:
        'Context-aware references: (1) Analyze nearby cells and intent, (2) Relative (A1) when: Formula should copy down (e.g., sum each row), cell is data not constant, (3) Absolute ($A$1) when: Referencing header/label, constant value, lookup table, (4) Mixed ($A1, A$1) for partial locking. Ask LLM: "Based on context, should this be relative or absolute?" with examples. Correct references prevent errors when formulas are copied.',
    },
  ];
