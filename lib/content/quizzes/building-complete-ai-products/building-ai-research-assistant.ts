export const buildingAiResearchAssistantQuiz = [
  {
    id: 'bcap-bara-q-1',
    question:
      "Design a multi-agent research system where agents collaborate to research a topic. Include: (1) Web search agent, (2) Document analysis agent, (3) Synthesis agent, (4) Fact-checking agent. How do they communicate? How do you prevent infinite loops? How do you handle conflicting information from different sources? What's the orchestration strategy?",
    sampleAnswer:
      'Architecture: Central orchestrator + 4 specialized agents. Orchestrator: Receives query, creates research plan, delegates to agents, synthesizes final report. Communication: Message queue (Redis) with structured messages {from, to, type, content, metadata}. Agents: (1) Search agent: Queries Google/Bing, extracts URLs, scores relevance, passes to Document agent. (2) Document agent: Fetches pages, extracts text, chunks, summarizes key points. (3) Synthesis agent: Receives summaries from multiple sources, identifies themes, creates coherent narrative. (4) Fact-check agent: Cross-references claims, scores confidence, flags inconsistencies. Workflow: Query → Orchestrator creates plan → Search agent (5 URLs) → Document agent (parallel processing) → Synthesis agent (draft report) → Fact-check agent (verify claims) → Orchestrator (final report). Prevent loops: Max depth (3 levels), visited URLs tracking, timeout (5min total). Conflicting info: Fact-check agent assigns confidence scores, prefers primary sources over blogs, shows multiple viewpoints when consensus lacking. Orchestration: Async task graph with dependencies, agents poll queue, orchestrator tracks progress, can cancel/retry failed tasks.',
    keyPoints: [
      'Central orchestrator delegates to specialized agents via message queue',
      'Async workflow with dependency graph (search → analyze → synthesize → verify)',
      'Prevent loops: max depth, visited tracking, timeout',
      'Handle conflicts: confidence scores, prefer primary sources, show multiple viewpoints',
      'Parallel processing for document analysis (speed optimization)',
    ],
  },
  {
    id: 'bcap-bara-q-2',
    question:
      'Your research assistant needs to process 100-page PDFs, analyze academic papers, and extract citations. How would you: (1) Extract structured data (title, authors, abstract, sections, citations), (2) Build a citation graph, (3) Find related papers, (4) Summarize while preserving accuracy? Compare vision models (GPT-4V) vs traditional PDF parsing (PyPDF2). Which is better for academic papers?',
    sampleAnswer:
      'Hybrid approach: (1) Traditional parsing (PyPDF2 + pdfplumber) for text extraction - fast, reliable, free. (2) LLM for structure understanding. Extraction: Parse PDF → Extract text blocks → Use Claude to identify sections (abstract, methods, results, etc.) with prompt: "Identify section boundaries in this academic paper." (3) Citations: Regex for common formats (APA, MLA, Chicago), extract DOIs, use Crossref API to resolve metadata. (4) Citation graph: Store in Neo4j or PostgreSQL with graph extension, links between papers. (5) Related papers: Embed abstracts (Voyage), vector search for similarity + citation links (papers citing same sources). Vision models: Use GPT-4V ONLY for: (1) Papers with complex layouts, (2) Figures/charts analysis, (3) Scanned PDFs (OCR). Cost: Vision $10-20 per 100-page paper vs $0.50 with text parsing. Speed: Vision 2-5min vs parsing 10s. Accuracy: Vision better for complex layouts, parsing better for standard papers. Recommendation: Parse first, use vision only if parsing fails or need figure analysis. Summarization: Use Claude with academic prompt: "Summarize this paper preserving key findings, methodology, and limitations. Include citations."',
    keyPoints: [
      'Hybrid: traditional parsing (fast, cheap) + LLM for understanding',
      'Extract citations with regex + Crossref API for metadata',
      'Citation graph in Neo4j, enables related paper discovery',
      'Vision models only for complex layouts/figures (10-20x more expensive)',
      'Summarize with specific prompts preserving accuracy and citations',
    ],
  },
  {
    id: 'bcap-bara-q-3',
    question:
      'Design a real-time research dashboard that updates as agents work. Requirements: show current task, sources found, key findings, and progress. How do you: (1) Stream updates from backend, (2) Handle long-running research (10-30min), (3) Allow user to guide research mid-process ("focus more on X"), (4) Save/resume research sessions? What\'s the UX for showing AI research in progress?',
    sampleAnswer:
      'Real-time architecture: (1) WebSocket connection between frontend and orchestrator. (2) Backend emits events: {type: "task_started", agent: "search", query}, {type: "source_found", url, title}, {type: "insight", content}. (3) Frontend displays in timeline view. Progress tracking: (1) Research plan with estimated steps (e.g., 5 searches, 15 documents, 1 synthesis). (2) Progress bar based on completed steps. (3) Time estimate based on historical data. User guidance: (1) "Focus" button on insights → sends message to orchestrator → adjusts research plan (search more related terms). (2) "Skip" source if irrelevant. (3) "Deep dive" on specific source. Save/resume: (1) Store research state in DB: {query, plan, completed_tasks, intermediate_results}. (2) On resume, reload state, continue from last step. (3) Generate session ID, shareable link. UX: (1) Hero card showing current task with animated progress. (2) Timeline of completed tasks (collapsible). (3) Live insights panel (updates as synthesis happens). (4) Source cards with relevance scores. (5) Allow export at any time (PDF report of current findings). Timeout: After 30min, auto-finalize even if incomplete. Show "Research timed out, here\'s what we found."',
    keyPoints: [
      'WebSocket for real-time updates (task started, source found, insight)',
      'Progress tracking with estimated steps and time based on history',
      'User guidance: focus, skip, deep dive (adjusts research plan dynamically)',
      'Save state to DB, shareable session URLs, resume from any point',
      'UX: current task hero card, timeline, live insights, source cards',
    ],
  },
];
