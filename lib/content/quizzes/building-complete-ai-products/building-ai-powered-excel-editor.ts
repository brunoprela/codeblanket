export const buildingAiPoweredExcelEditorQuiz = [
  {
    id: 'bcap-bapee-q-1',
    question:
      'Design an AI formula generator that converts natural language to Excel formulas. Challenges: (1) Ambiguous requests ("calculate growth"), (2) Multiple valid approaches, (3) Cell reference vs named ranges, (4) Complex formulas (nested IFs, VLOOKUPs). How do you: parse intent, generate accurate formulas, validate correctness, and explain the formula back to users?',
    sampleAnswer:
      'Multi-stage pipeline: (1) Intent parsing: LLM extracts: operation (sum, average, growth), range (which cells), conditions (if any). (2) Context gathering: Analyze nearby cells, detect data types (numbers, dates, text), identify table structure, check for named ranges. (3) Formula generation: Use Claude with rich context: "Generate Excel formula for: {intent}. Context: {nearby_data}. Available: {functions}." Include examples: "Growth = (new - old) / old". (4) Validation: Check syntax (balanced parens), test with sample data, verify result type matches intent. (5) Multi-candidate: Generate 3 formula variants, rank by: simplicity, performance, accuracy. Present top choice with alternatives. Ambiguity handling: If ambiguous ("growth"), ask clarification: "Year-over-year (YoY) or month-over-month (MoM)?" Cell references: Prefer relative (A1) for flexibility, absolute ($A$1) when referencing constants. Auto-detect based on context: if reference is header/constant → absolute. Named ranges: If table detected, suggest: "Create named range \'SalesData\' for clearer formulas." Complex formulas: Break into steps, explain each part: "This formula: (1) Looks up customer, (2) Checks if premium tier, (3) Applies discount." Error handling: If formula produces #REF or #VALUE, diagnose: "Error: Column B was deleted. Update formula to use Column C?" Explanation: Generate plain English: "This formula sums Column B values where Column A > 100."',
    keyPoints: [
      'Multi-stage: parse intent → gather context → generate → validate → explain',
      'Context is critical: nearby data types, table structure, named ranges',
      'Generate multiple candidates, rank by simplicity/performance/accuracy',
      'Handle ambiguity: ask clarifying questions before generating',
      'Explain back in plain English for user understanding and trust',
    ],
  },
  {
    id: 'bcap-bapee-q-2',
    question:
      'Your Excel AI assistant needs to understand and analyze financial models. How do you: (1) Parse interconnected sheets with formulas, (2) Build dependency graph, (3) Detect circular references, (4) Suggest optimizations, (5) Explain model to non-technical users? Compare: full AST parsing vs LLM understanding. Which is better for financial models?',
    sampleAnswer:
      'Hybrid approach: AST parsing (structure) + LLM (semantics). (1) Parse model: Use openpyxl to extract all formulas, build cell dependency graph: {A1: [B1, C1], B1: [D1]}. (2) Dependency graph: Construct directed graph, topological sort for calculation order. Detect circular refs (cycles in graph). (3) AST for precision: Parse each formula into AST, extract: referenced cells, functions used, constants. (4) LLM for semantics: Feed graph + sample values to Claude: "Explain what this financial model calculates." LLM identifies: revenue model, cost structure, profitability metrics. (5) Optimization: AST detects: volatile functions (NOW, RAND), redundant calculations (same formula repeated), inefficient lookups (VLOOKUP in loops). Suggest: "Replace VLOOKUP with INDEX/MATCH for 10x speed." (6) Circular refs: If detected, use LLM to suggest: "This creates circular reference. Did you mean to reference previous month (offset)?" (7) Explain model: Generate narrative: "This model forecasts revenue based on user growth (B5) × ARPU (B6), subtracts costs (Sheet2), calculates profit margin." Include diagram of data flow. Why hybrid: Pure LLM can misunderstand complex formulas, hallucinate dependencies. Pure AST lacks semantic understanding (doesn\'t know "revenue" vs "cost"). Together: AST for precision (100% accurate dependencies), LLM for insights. Cost: AST parsing (free), LLM analysis ($0.50-2 per model), acceptable for high-value financial models.',
    keyPoints: [
      'Hybrid: AST for structure/precision, LLM for semantic understanding',
      'Build dependency graph: topological sort, detect circular references',
      'AST detects: volatile functions, redundant calcs, inefficient lookups',
      'LLM explains: what model calculates, revenue/cost structure, insights',
      'AST 100% accurate for dependencies, LLM provides business context',
    ],
  },
  {
    id: 'bcap-bapee-q-3',
    question:
      'Design an automation system for Excel workflows. Examples: (1) "Every Monday, pull sales data from API, update this sheet, email summary to team", (2) "When revenue > $10k, flag row in green", (3) "Generate monthly report from template". How do you: define automations, trigger them, handle failures, and maintain security (API keys, email access)?',
    sampleAnswer:
      'Automation DSL + Workflow engine: (1) Definition: User describes in natural language, LLM converts to structured workflow: {trigger: {type: "schedule", cron: "0 9 * * 1"}, actions: [{type: "fetch_api", url, target_range}, {type: "generate_report"}, {type: "send_email"}]}. (2) Triggers: Schedule (cron), cell value change, manual, webhook. (3) Actions: API call, update cells, generate report, send email, run formula. Implementation: Store workflows in database, background worker (Celery) polls for due triggers. Execution: (1) Trigger fires → Load workflow → Execute actions sequentially → Log results. (2) Handle failures: Retry 3x with backoff, alert user if persistent failure, partial execution (some actions succeed). (3) Cell change trigger: Use Office.js event listeners (web) or VBA (desktop), webhook to backend. Security: (1) API keys: Encrypted in database (AES-256), user provides via secure form, never shown in logs. (2) Email: OAuth2 (Google/Microsoft), request only send permission, revokable. (3) Sandbox: Actions run in isolated environment, rate limited, can\'t access other users\' data. (4) Audit log: Track all executions, data accessed, emails sent. UI: Visual workflow builder (drag-drop nodes) + natural language interface. Show execution history: timestamp, status, errors. Allow: pause/resume, edit workflow, delete. Example: "Every Monday pull sales" → System generates: Schedule trigger + API action + Update cells + Email action. User reviews, tests, activates.',
    keyPoints: [
      'LLM converts natural language to structured workflow (triggers + actions)',
      'Triggers: schedule, cell change, manual, webhook',
      'Handle failures: retry with backoff, partial execution, user alerts',
      'Security: encrypted API keys, OAuth2 for email, sandboxed execution, audit logs',
      'UI: visual builder + natural language, execution history, test mode',
    ],
  },
];
