/**
 * Quiz questions for Code Review & Bug Detection section
 */

export const codereviewbugdetectionQuiz = [
  {
    id: 'bcgs-codereview-q-1',
    question:
      'Design a comprehensive code review system that prioritizes issues by severity and actionability. What criteria determine if a detected issue is critical vs. minor, and how would you prevent alert fatigue?',
    hint: 'Consider security impact, user impact, fix difficulty, and false positive rates.',
    sampleAnswer:
      '**Issue Prioritization System:** **Critical (Fix Immediately):** 1) **Security Vulnerabilities** - SQL injection, XSS, hardcoded secrets. Direct user/data risk. 2) **Data Loss Bugs** - Code that could delete/corrupt data. 3) **Crash-Causing Errors** - Null pointer dereference, uncaught exceptions in critical paths. Criteria: High user impact + High probability + Exploitable. **High (Fix Before Release):** 1) **Logic Errors** - Wrong calculations, off-by-one errors. Produces wrong results but doesn\'t crash. 2) **Resource Leaks** - Memory leaks, unclosed files. Degrades performance over time. 3) **Authentication Bypasses** - Security but maybe not directly exploitable. Criteria: Moderate user impact + High probability. **Medium (Fix in Next Sprint):** 1) **Performance Issues** - O(n²) algorithms, unnecessary loops. Affects UX but works. 2) **Code Smells** - God classes, long methods. Maintainability issues. 3) **Missing Error Handling** - Not all exceptions caught, missing validation. Criteria: Low immediate impact + Technical debt. **Low (Nice to Have):** 1) **Style Violations** - Naming conventions, formatting. 2) **Minor Optimizations** - Could be faster but acceptable. **Preventing Alert Fatigue:** 1) **Confidence Scoring** - Only show high-confidence issues (>80% certain). 2) **Context-Aware** - Don\'t flag test code for performance issues. 3) **Learn from Dismissals** - If user marks issue as "not a problem" multiple times, stop flagging similar. 4) **Actionable Only** - Each issue must have: Clear description, Exact location, Concrete fix suggestion. 5) **Aggregate Similar** - "Found 15 similar issues" not 15 separate alerts. **Example:** SQL Injection (Critical, fix now) vs Variable naming (Low, fix maybe never) vs O(n²) loop (Medium, optimize in refactor sprint).',
    keyPoints: [
      'Critical = security/data loss/crashes (immediate fix)',
      'Prioritize by: user impact × probability × exploitability',
      'Prevent fatigue: high confidence only, context-aware, learn from dismissals',
      'Every alert must be actionable with clear fix',
    ],
  },
  {
    id: 'bcgs-codereview-q-2',
    question:
      "Explain how to detect security vulnerabilities that aren't caught by simple pattern matching (e.g., SQL injection, XSS). What analysis techniques would catch complex attack vectors?",
    hint: 'Consider taint analysis, data flow tracking, and context-sensitive analysis.',
    sampleAnswer:
      '**Advanced Security Analysis Techniques:** **1) Taint Analysis** - Track "tainted" data (user input) through the program: ```python\\n# Simple pattern matching MISSES this:\\nuser_input = request.get("query")  # Tainted\\ntemp = user_input  # Still tainted\\nprocessed = transform(temp)  # Still tainted!\\nquery = f"SELECT * FROM users WHERE name=\'{processed}\'"  # SQL injection!``` Taint analysis tracks data flow: Mark sources (user input, files, network), Track through assignments/transformations, Flag if tainted data reaches sink (SQL query) without sanitization. **2) Data Flow Analysis** - Build graph of how data moves: ```python\\ndef vulnerable_endpoint(user_id):\\n    user_id = validate(user_id)  # Sanitized here\\n    if some_condition:\\n        user_id = request.get("backup_id")  # Tainted again!\\n    query = f"SELECT * FROM users WHERE id={user_id}"  # Vulnerable path!``` Track all possible paths from source to sink, Flag any path where sanitization is bypassed. **3) Context-Sensitive Analysis** - Same string is safe/unsafe depending on context: ```python\\nuser_input = request.get("name")\\nhtml = f"<div>{user_input}</div>"  # XSS vulnerable!\\nlog_entry = f"User logged in: {user_input}"  # Safe in logs\\nsql = f"SELECT * FROM users WHERE name=\'{user_input}\'"  # SQL injection!``` Analyze context of usage: HTML context → need HTML escaping, SQL context → need parameterized queries, Shell context → need shell escaping. **4) Semantic Understanding** - Understand sanitization intent: ```python\\ndef sanitize(input):\\n    return input.replace("\'", "\'")  # Incomplete! MySQL needs more\\n\\nquery = f"SELECT * FROM users WHERE name=\'{sanitize(user_input)}\'"  # Still vulnerable!``` Check if sanitization is appropriate for context: Basic escaping insufficient for SQL → flag, HTML sanitization used for SQL → flag. **5) Call Graph Analysis** - Track through function calls: ```python\\ndef get_user_query():\\n    return request.get("q")\\n\\ndef search():\\n    query = get_user_query()  # Indirectly tainted\\n    execute_sql(query)  # Vulnerable!\\n``` Build inter-procedural call graph, Track taints across function boundaries.',
    keyPoints: [
      'Taint analysis tracks user input through data flow',
      'Context-sensitive analysis checks if sanitization matches usage',
      'Data flow analysis finds all paths from source to sink',
      'Semantic understanding validates sanitization appropriateness',
    ],
  },
  {
    id: 'bcgs-codereview-q-3',
    question:
      "You've built a bug detector that has high false positive rate (flags many non-bugs). How would you improve precision without sacrificing recall? What machine learning or heuristic approaches could help?",
    hint: 'Consider confidence scoring, historical data, context learning, and user feedback.',
    sampleAnswer:
      '**Improving Bug Detection Precision:** **1) Confidence Scoring** - Don\'t just report bug/no-bug, score confidence: ```python\\ndef detect_bugs(code):\\n    potential_bugs = []\\n    \\n    for issue in all_checks(code):\\n        confidence = calculate_confidence(issue)\\n        if confidence > 0.8:  # Only high confidence\\n            potential_bugs.append((issue, confidence))``` Factors for confidence: Pattern strength (exact match vs fuzzy), Context appropriateness (is this test code?), Historical accuracy (has this pattern been wrong before?). **2) Context-Aware Filtering** - Same pattern is bug/not-bug depending on context: ```python\\n# In production code: BUG\\nassert user.is_admin()  # Production shouldn\'t use assert\\n\\n# In test code: OK\\nassert user.is_admin()  # Tests should use assert``` Track context: File location (tests/, scripts/), Function purpose (helper vs core logic), Comment hints (#noqa, #intentional). **3) Learn from User Feedback** - Build dataset from dismissals: ```python\\nclass BugDetectorWithLearning:\\n    def __init__(self):\\n        self.false_positives = []  # User said "not a bug"\\n    \\n    def detect(self, code):\\n        issues = run_detectors(code)\\n        \\n        # Filter using learned patterns\\n        return [i for i in issues if not self.looks_like_false_positive(i)]\\n    \\n    def looks_like_false_positive(self, issue):\\n        for fp in self.false_positives:\\n            if similar(issue, fp) > 0.9:\\n                return True``` **4) Ensemble Methods** - Combine multiple detectors, flag only if majority agree: ```python\\ndetectors = [static_analysis, LLM_detector, pattern_matcher]\\nvotes = [d.is_bug(code) for d in detectors]\\nif sum(votes) >= 2:  # Majority\\n    report_bug()``` **5) Explainability** - Require detector to explain WHY: ```python\\nissue = {\\n    "type": "null_pointer",\\n    "line": 42,\\n    "explanation": "user can be None on line 38, then dereferenced on line 42",\\n    "evidence": ["line 38: user = get_user() returns None", "line 42: user.name accessed",]\\n}``` Reject if explanation is weak. **6) Historical Accuracy Tracking** - Track which patterns are accurate: ```python\\npattern_accuracy = {\\n    "divide_by_zero": 0.95,  # Usually correct\\n    "long_function": 0.30,  # Often false positive\\n}``` Weight confidence by pattern accuracy. **Example:** Initially flag all "==None" as bugs. After learning that "== None" is intentional in some contexts (config checking), stop flagging those specific patterns.',
    keyPoints: [
      'Use confidence scoring, only report high-confidence issues',
      'Context-aware filtering (test vs production code)',
      'Learn from user feedback to avoid repeated false positives',
      'Ensemble methods - require multiple detectors to agree',
    ],
  },
];
