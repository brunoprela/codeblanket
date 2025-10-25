export const buildingAiResearchAssistant = {
  title: 'Building an AI Research Assistant',
  id: 'building-ai-research-assistant',
  content: `
# Building an AI Research Assistant

## Introduction

An AI Research Assistant is a sophisticated multi-agent system that can autonomously research topics, gather information from multiple sources, synthesize findings, and generate comprehensive reports. Think of it as having a tireless research team that can read hundreds of articles, extract key insights, and produce publication-ready reports.

This section covers building a production-ready research assistant that:
- Searches the web for relevant information
- Processes and analyzes documents (PDFs, articles, papers)
- Synthesizes information from multiple sources
- Generates structured reports with citations
- Handles multi-step research workflows
- Validates facts and checks sources

### Why Build a Research Assistant?

**Use Cases:**
- **Academic Research**: Literature reviews, paper summaries, citation analysis
- **Market Research**: Competitor analysis, market sizing, trend identification
- **Due Diligence**: Company research, background checks, risk assessment
- **Content Creation**: Blog research, fact-checking, source gathering
- **Legal Research**: Case law analysis, precedent search, regulation review

**Market Opportunity**: The research automation market is growing rapidly. Tools like Perplexity AI, Elicit, and Consensus show strong product-market fit.

### Architecture Overview

\`\`\`
┌──────────────────────────────────────────────────────────────┐
│              AI Research Assistant System                     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────┐      ┌─────────────┐      ┌──────────────┐ │
│  │   Query    │─────▶│  Planner    │─────▶│   Executor   │ │
│  │  Interface │      │   Agent     │      │    Agents    │ │
│  └────────────┘      └─────────────┘      └──────┬───────┘ │
│                                                    │          │
│                      ┌─────────────────────────────┼────────┐│
│                      │                             │        ││
│           ┌──────────▼──────┐           ┌─────────▼──────┐ ││
│           │  Web Search     │           │  Document      │ ││
│           │  Agent          │           │  Processor     │ ││
│           │  (Serper/Bing)  │           │  Agent         │ ││
│           └──────────┬──────┘           └─────────┬──────┘ ││
│                      │                             │        ││
│           ┌──────────▼──────┐           ┌─────────▼──────┐ ││
│           │  Summarizer     │           │  Fact          │ ││
│           │  Agent          │           │  Checker       │ ││
│           └──────────┬──────┘           └─────────┬──────┘ ││
│                      │                             │        ││
│                      └─────────────┬───────────────┘        ││
│                                    │                        ││
│                            ┌───────▼────────┐               ││
│                            │  Synthesizer   │               ││
│                            │  Agent         │               ││
│                            └───────┬────────┘               ││
│                                    │                        ││
│                            ┌───────▼────────┐               ││
│                            │  Report        │               ││
│                            │  Generator     │               ││
│                            └────────────────┘               ││
└──────────────────────────────────────────────────────────────┘
\`\`\`

---

## Multi-Agent Architecture

### Planner Agent

The Planner breaks down research queries into actionable steps:

\`\`\`python
"""
Planner Agent - Decomposes research tasks
"""

from typing import List, Dict
from pydantic import BaseModel
from openai import AsyncOpenAI

class ResearchStep(BaseModel):
    """Single research step"""
    step_number: int
    action: str  # "search", "read_document", "synthesize", "validate"
    query: str
    dependencies: List[int] = []
    priority: int = 1

class ResearchPlan(BaseModel):
    """Complete research plan"""
    steps: List[ResearchStep]
    estimated_time_minutes: int
    required_sources: int

class PlannerAgent:
    """
    Decomposes research queries into step-by-step plans
    """
    
    def __init__(self, llm_client: AsyncOpenAI):
        self.llm = llm_client
        
    async def create_plan (self, query: str) -> ResearchPlan:
        """
        Create a research plan for the query
        """
        prompt = f"""You are a research planner. Break down this research query into specific steps.

Research Query: {query}

Create a step-by-step plan. Each step should be one of:
- search: Search the web for information
- read_document: Read and analyze a specific document
- synthesize: Combine information from multiple sources
- validate: Fact-check claims

Format as JSON with this structure:
{{
  "steps": [
    {{
      "step_number": 1,
      "action": "search",
      "query": "specific search query",
      "dependencies": [],
      "priority": 1
    }}
  ],
  "estimated_time_minutes": 10,
  "required_sources": 5
}}

Generate the plan:"""

        response = await self.llm.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        plan_json = json.loads (response.choices[0].message.content)
        return ResearchPlan(**plan_json)

# Example usage
planner = PlannerAgent (llm_client)
plan = await planner.create_plan(
    "What are the latest developments in quantum computing and their commercial applications?"
)

print(f"Research plan with {len (plan.steps)} steps")
for step in plan.steps:
    print(f"{step.step_number}. {step.action}: {step.query}")

# Output:
# 1. search: latest quantum computing breakthroughs 2024
# 2. search: commercial quantum computing applications
# 3. search: quantum computing companies and startups
# 4. synthesize: compile findings on developments and applications
# 5. validate: verify claims about quantum computing capabilities
\`\`\`

### Web Search Agent

Searches the web and extracts relevant information:

\`\`\`python
"""
Web Search Agent using Serper API
"""

import aiohttp
from typing import List, Dict
from bs4 import BeautifulSoup
import asyncio

class SearchResult(BaseModel):
    """Search result with metadata"""
    title: str
    url: str
    snippet: str
    date: Optional[str] = None
    relevance_score: float = 0.0

class WebSearchAgent:
    """
    Searches web and extracts content
    """
    
    def __init__(self, serper_api_key: str, llm_client: AsyncOpenAI):
        self.api_key = serper_api_key
        self.llm = llm_client
        
    async def search (self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        Search web using Serper API
        """
        url = "https://google.serper.dev/search"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={"q": query, "num": num_results},
                headers={"X-API-KEY": self.api_key}
            ) as response:
                data = await response.json()
        
        results = []
        for item in data.get("organic", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                date=item.get("date"),
                relevance_score=1.0  # Can be refined
            ))
        
        return results
    
    async def extract_content (self, url: str) -> str:
        """
        Extract main content from URL
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get (url, timeout=10) as response:
                    html = await response.text()
            
            soup = BeautifulSoup (html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            
            # Get main content (heuristic)
            main_content = soup.find('main') or soup.find('article') or soup.body
            
            if main_content:
                text = main_content.get_text (separator='\\n', strip=True)
                # Limit to reasonable size
                return text[:10000]
            
            return ""
        
        except Exception as e:
            print(f"Error extracting {url}: {e}")
            return ""
    
    async def rank_by_relevance(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Use LLM to rank search results by relevance
        """
        prompt = f"""Rate the relevance of these search results to the query.

Query: {query}

Search Results:
{json.dumps([{
    'title': r.title,
    'snippet': r.snippet,
    'url': r.url
} for r in results], indent=2)}

Rate each result from 0.0 to 1.0 based on relevance.
Return JSON array of scores: [0.9, 0.7, ...]"""

        response = await self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        scores = json.loads (response.choices[0].message.content).get("scores", [])
        
        for i, score in enumerate (scores[:len (results)]):
            results[i].relevance_score = score
        
        return sorted (results, key=lambda x: x.relevance_score, reverse=True)

# Usage
search_agent = WebSearchAgent (serper_api_key, llm_client)

results = await search_agent.search("quantum computing applications")
ranked_results = await search_agent.rank_by_relevance(
    "commercial quantum computing applications",
    results
)

print(f"Top result: {ranked_results[0].title} (score: {ranked_results[0].relevance_score})")
\`\`\`

### Document Processor Agent

Processes and analyzes documents:

\`\`\`python
"""
Document Processor Agent
"""

from pypdf import PdfReader
from docx import Document as DocxDocument
import tiktoken

class DocumentChunk(BaseModel):
    """Document chunk with metadata"""
    content: str
    page: Optional[int] = None
    chunk_index: int
    tokens: int
    metadata: Dict[str, any] = {}

class DocumentProcessorAgent:
    """
    Processes documents (PDF, DOCX, TXT)
    """
    
    def __init__(self, llm_client: AsyncOpenAI):
        self.llm = llm_client
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
    async def process_document(
        self,
        file_path: str,
        chunk_size: int = 1000
    ) -> List[DocumentChunk]:
        """
        Process document and chunk it
        """
        # Extract text based on file type
        if file_path.endswith('.pdf'):
            text = self._extract_pdf (file_path)
        elif file_path.endswith('.docx'):
            text = self._extract_docx (file_path)
        else:
            text = Path (file_path).read_text (encoding='utf-8')
        
        # Chunk text
        chunks = self._chunk_text (text, chunk_size)
        
        return chunks
    
    def _extract_pdf (self, file_path: str) -> str:
        """Extract text from PDF"""
        reader = PdfReader (file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\\n\\n"
        return text
    
    def _extract_docx (self, file_path: str) -> str:
        """Extract text from DOCX"""
        doc = DocxDocument (file_path)
        return "\\n\\n".join([para.text for para in doc.paragraphs])
    
    def _chunk_text (self, text: str, chunk_size: int) -> List[DocumentChunk]:
        """
        Chunk text by token count with overlap
        """
        tokens = self.encoder.encode (text)
        chunks = []
        overlap = 200  # Token overlap between chunks
        
        for i in range(0, len (tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.encoder.decode (chunk_tokens)
            
            chunks.append(DocumentChunk(
                content=chunk_text,
                chunk_index=len (chunks),
                tokens=len (chunk_tokens)
            ))
        
        return chunks
    
    async def summarize_document (self, chunks: List[DocumentChunk]) -> str:
        """
        Summarize entire document from chunks
        """
        # For short documents, summarize directly
        if len (chunks) <= 5:
            full_text = "\\n\\n".join([c.content for c in chunks])
            return await self._summarize_text (full_text)
        
        # For long documents, hierarchical summarization
        # 1. Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = await self._summarize_text (chunk.content)
            chunk_summaries.append (summary)
        
        # 2. Combine and summarize summaries
        combined = "\\n\\n".join (chunk_summaries)
        final_summary = await self._summarize_text (combined)
        
        return final_summary
    
    async def _summarize_text (self, text: str) -> str:
        """Summarize single text"""
        response = await self.llm.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the following text, preserving key facts and insights."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    async def extract_key_points (self, text: str) -> List[str]:
        """
        Extract key points from text
        """
        prompt = f"""Extract 5-7 key points from this text.

Text:
{text}

Return as JSON array: ["point 1", "point 2", ...]"""

        response = await self.llm.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        return json.loads (response.choices[0].message.content).get("points", [])

# Usage
doc_processor = DocumentProcessorAgent (llm_client)

chunks = await doc_processor.process_document("research_paper.pdf")
summary = await doc_processor.summarize_document (chunks)
key_points = await doc_processor.extract_key_points (summary)

print(f"Document summary:\\n{summary}")
print(f"\\nKey points: {key_points}")
\`\`\`

### Synthesizer Agent

Combines information from multiple sources:

\`\`\`python
"""
Synthesizer Agent - Combines multi-source information
"""

class SourcedFact(BaseModel):
    """Fact with source attribution"""
    fact: str
    sources: List[str]
    confidence: float

class SynthesizerAgent:
    """
    Synthesizes information from multiple sources
    """
    
    def __init__(self, llm_client: AsyncOpenAI):
        self.llm = llm_client
    
    async def synthesize(
        self,
        query: str,
        sources: List[Dict[str, str]]  # [{"content": ..., "url": ...}]
    ) -> Dict[str, any]:
        """
        Synthesize information from multiple sources
        """
        # Combine sources with attribution
        source_text = ""
        for i, source in enumerate (sources):
            source_text += f"\\n\\n[Source {i+1}: {source['url']}]\\n{source['content']}"
        
        prompt = f"""Synthesize information from these sources to answer the query.

Query: {query}

Sources:
{source_text}

Provide:
1. Main findings (comprehensive answer)
2. Key facts with source citations
3. Conflicting information (if any)
4. Gaps in information

Format as JSON:
{{
  "answer": "comprehensive answer",
  "key_facts": [
    {{
      "fact": "statement",
      "sources": ["Source 1", "Source 3"],
      "confidence": 0.9
    }}
  ],
  "conflicts": ["any conflicting claims"],
  "gaps": ["information not found"]
}}"""

        response = await self.llm.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        return json.loads (response.choices[0].message.content)
    
    async def cross_reference(
        self,
        fact: str,
        sources: List[Dict[str, str]]
    ) -> Dict[str, any]:
        """
        Cross-reference a fact across sources
        """
        prompt = f"""Verify this fact across the provided sources:

Fact to verify: {fact}

Sources:
{json.dumps([{"url": s["url"], "excerpt": s["content"][:500]} for s in sources], indent=2)}

Determine:
1. How many sources support this fact?
2. Are there any contradictions?
3. Confidence level (0.0 to 1.0)

Return JSON:
{{
  "supported": true/false,
  "supporting_sources": ["Source 1", ...],
  "contradicting_sources": [],
  "confidence": 0.9,
  "explanation": "reasoning"
}}"""

        response = await self.llm.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        return json.loads (response.choices[0].message.content)

# Usage
synthesizer = SynthesizerAgent (llm_client)

sources = [
    {"url": "https://source1.com", "content": "Quantum computers use qubits..."},
    {"url": "https://source2.com", "content": "IBM quantum computer achieved..."},
    {"url": "https://source3.com", "content": "Google\'s quantum supremacy..."}
]

synthesis = await synthesizer.synthesize(
    "What are the latest quantum computing breakthroughs?",
    sources
)

print(f"Answer: {synthesis['answer']}")
print(f"Key facts: {len (synthesis['key_facts'])}")
\`\`\`

---

## Report Generation

### Structured Report Generator

\`\`\`python
"""
Report Generator - Creates formatted research reports
"""

from jinja2 import Template
from datetime import datetime

class ReportSection(BaseModel):
    """Report section"""
    title: str
    content: str
    level: int = 1  # Heading level
    subsections: List['ReportSection'] = []

class ResearchReport(BaseModel):
    """Complete research report"""
    title: str
    query: str
    executive_summary: str
    sections: List[ReportSection]
    sources: List[Dict[str, str]]
    generated_at: datetime
    word_count: int

class ReportGenerator:
    """
    Generates formatted research reports
    """
    
    def __init__(self, llm_client: AsyncOpenAI):
        self.llm = llm_client
    
    async def generate_report(
        self,
        query: str,
        synthesis: Dict[str, any],
        sources: List[Dict[str, str]]
    ) -> ResearchReport:
        """
        Generate complete research report
        """
        # Create executive summary
        exec_summary = await self._generate_executive_summary(
            query,
            synthesis
        )
        
        # Create sections
        sections = await self._generate_sections (query, synthesis)
        
        # Count words
        word_count = self._count_words (exec_summary, sections)
        
        return ResearchReport(
            title=await self._generate_title (query),
            query=query,
            executive_summary=exec_summary,
            sections=sections,
            sources=sources,
            generated_at=datetime.now(),
            word_count=word_count
        )
    
    async def _generate_title (self, query: str) -> str:
        """Generate report title"""
        response = await self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"Generate a professional report title for this research query: {query}"
            }],
            temperature=0.7,
            max_tokens=50
        )
        
        return response.choices[0].message.content.strip()
    
    async def _generate_executive_summary(
        self,
        query: str,
        synthesis: Dict[str, any]
    ) -> str:
        """Generate executive summary"""
        prompt = f"""Write an executive summary (2-3 paragraphs) for a research report.

Research Query: {query}

Key Findings:
{synthesis.get('answer', ')}

Executive Summary:"""

        response = await self.llm.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    async def _generate_sections(
        self,
        query: str,
        synthesis: Dict[str, any]
    ) -> List[ReportSection]:
        """Generate report sections"""
        sections = []
        
        # Introduction
        sections.append(ReportSection(
            title="Introduction",
            content=await self._generate_introduction (query),
            level=1
        ))
        
        # Main Findings
        sections.append(ReportSection(
            title="Main Findings",
            content=synthesis.get('answer', '),
            level=1
        ))
        
        # Key Facts
        key_facts_content = "\\n\\n".join([
            f"**{fact['fact']}**\\nSources: {', '.join (fact['sources'])}"
            for fact in synthesis.get('key_facts', [])
        ])
        
        sections.append(ReportSection(
            title="Key Facts",
            content=key_facts_content,
            level=1
        ))
        
        # Limitations
        if synthesis.get('gaps'):
            sections.append(ReportSection(
                title="Limitations & Gaps",
                content="\\n".join([f"- {gap}" for gap in synthesis['gaps']]),
                level=1
            ))
        
        # Conclusion
        sections.append(ReportSection(
            title="Conclusion",
            content=await self._generate_conclusion (query, synthesis),
            level=1
        ))
        
        return sections
    
    async def _generate_introduction (self, query: str) -> str:
        """Generate introduction section"""
        prompt = f"""Write an introduction paragraph for a research report on: {query}"""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    async def _generate_conclusion(
        self,
        query: str,
        synthesis: Dict[str, any]
    ) -> str:
        """Generate conclusion section"""
        prompt = f"""Write a conclusion paragraph summarizing the research findings.

Query: {query}
Findings: {synthesis.get('answer', ')[:500]}

Conclusion:"""

        response = await self.llm.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def _count_words (self, summary: str, sections: List[ReportSection]) -> int:
        """Count total words in report"""
        text = summary + " " + " ".join([s.content for s in sections])
        return len (text.split())
    
    def export_markdown (self, report: ResearchReport) -> str:
        """Export report as Markdown"""
        md = f"""# {report.title}

**Research Query:** {report.query}  
**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M')}  
**Word Count:** {report.word_count}

## Executive Summary

{report.executive_summary}

"""
        
        for section in report.sections:
            md += f"\\n{'#' * (section.level + 1)} {section.title}\\n\\n{section.content}\\n"
        
        # Add sources
        md += "\\n## Sources\\n\\n"
        for i, source in enumerate (report.sources):
            md += f"{i+1}. [{source.get('title', 'Source')}]({source['url']})\\n"
        
        return md
    
    def export_html (self, report: ResearchReport) -> str:
        """Export report as HTML"""
        template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>{{ report.title }}</title>
    <style>
        body { font-family: Georgia, serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; margin-top: 30px; }
        .metadata { color: #7f8c8d; font-size: 0.9em; }
        .source { margin-bottom: 10px; }
    </style>
</head>
<body>
    <h1>{{ report.title }}</h1>
    <div class="metadata">
        <p><strong>Research Query:</strong> {{ report.query }}</p>
        <p><strong>Generated:</strong> {{ report.generated_at.strftime('%Y-%m-%d %H:%M') }}</p>
        <p><strong>Word Count:</strong> {{ report.word_count }}</p>
    </div>
    
    <h2>Executive Summary</h2>
    <p>{{ report.executive_summary }}</p>
    
    {% for section in report.sections %}
    <h{{ section.level + 1 }}>{{ section.title }}</h{{ section.level + 1 }}>
    <p>{{ section.content | replace('\\n', '<br>') }}</p>
    {% endfor %}
    
    <h2>Sources</h2>
    {% for source in report.sources %}
    <div class="source">
        <a href="{{ source.url }}">{{ source.get('title', 'Source ' + loop.index|string) }}</a>
    </div>
    {% endfor %}
</body>
</html>
        """)
        
        return template.render (report=report)

# Usage
report_gen = ReportGenerator (llm_client)

report = await report_gen.generate_report (query, synthesis, sources)

# Export
markdown = report_gen.export_markdown (report)
html = report_gen.export_html (report)

print(f"Generated report: {report.title}")
print(f"Word count: {report.word_count}")
\`\`\`

---

## Orchestration & Workflow

### Complete Research Orchestrator

\`\`\`python
"""
Research Orchestrator - Coordinates all agents
"""

class ResearchOrchestrator:
    """
    Orchestrates entire research workflow
    """
    
    def __init__(
        self,
        llm_client: AsyncOpenAI,
        serper_api_key: str
    ):
        self.planner = PlannerAgent (llm_client)
        self.search_agent = WebSearchAgent (serper_api_key, llm_client)
        self.doc_processor = DocumentProcessorAgent (llm_client)
        self.synthesizer = SynthesizerAgent (llm_client)
        self.report_gen = ReportGenerator (llm_client)
    
    async def research(
        self,
        query: str,
        max_sources: int = 10,
        include_documents: bool = False
    ) -> ResearchReport:
        """
        Execute complete research workflow
        """
        print(f"Starting research: {query}")
        
        # 1. Create research plan
        print("Creating research plan...")
        plan = await self.planner.create_plan (query)
        print(f"Plan created with {len (plan.steps)} steps")
        
        # 2. Execute search steps
        all_sources = []
        
        for step in plan.steps:
            if step.action == "search":
                print(f"Searching: {step.query}")
                results = await self.search_agent.search (step.query, num_results=5)
                
                # Extract content from top results
                for result in results[:3]:
                    content = await self.search_agent.extract_content (result.url)
                    if content:
                        all_sources.append({
                            "title": result.title,
                            "url": result.url,
                            "content": content[:5000]  # Limit size
                        })
        
        print(f"Gathered {len (all_sources)} sources")
        
        # 3. Synthesize information
        print("Synthesizing information...")
        synthesis = await self.synthesizer.synthesize (query, all_sources[:max_sources])
        
        # 4. Generate report
        print("Generating report...")
        report = await self.report_gen.generate_report (query, synthesis, all_sources)
        
        print(f"Research complete! Report: {report.title}")
        return report
    
    async def research_with_documents(
        self,
        query: str,
        document_paths: List[str]
    ) -> ResearchReport:
        """
        Research with provided documents
        """
        # Process documents
        all_sources = []
        
        for doc_path in document_paths:
            print(f"Processing document: {doc_path}")
            chunks = await self.doc_processor.process_document (doc_path)
            summary = await self.doc_processor.summarize_document (chunks)
            
            all_sources.append({
                "title": Path (doc_path).name,
                "url": f"file://{doc_path}",
                "content": summary
            })
        
        # Also search web
        search_results = await self.search_agent.search (query, num_results=5)
        for result in search_results[:3]:
            content = await self.search_agent.extract_content (result.url)
            if content:
                all_sources.append({
                    "title": result.title,
                    "url": result.url,
                    "content": content[:5000]
                })
        
        # Synthesize and generate report
        synthesis = await self.synthesizer.synthesize (query, all_sources)
        report = await self.report_gen.generate_report (query, synthesis, all_sources)
        
        return report

# Complete example
async def main():
    orchestrator = ResearchOrchestrator(
        llm_client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        serper_api_key=os.getenv("SERPER_API_KEY")
    )
    
    # Execute research
    report = await orchestrator.research(
        "What are the latest breakthroughs in quantum computing and their commercial applications?"
    )
    
    # Export report
    markdown = orchestrator.report_gen.export_markdown (report)
    Path("research_report.md").write_text (markdown)
    
    print(f"Report saved to research_report.md")
    print(f"Word count: {report.word_count}")

if __name__ == "__main__":
    asyncio.run (main())
\`\`\`

---

## Production Considerations

### Caching & Cost Optimization

\`\`\`python
"""
Caching layer for research assistant
"""

import hashlib
from datetime import datetime, timedelta

class ResearchCache:
    """
    Cache research results to reduce costs
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 86400  # 24 hours
    
    def _get_cache_key (self, query: str) -> str:
        """Generate cache key from query"""
        return f"research:{hashlib.sha256(query.encode()).hexdigest()}"
    
    async def get (self, query: str) -> Optional[ResearchReport]:
        """Get cached research result"""
        key = self._get_cache_key (query)
        cached = await self.redis.get (key)
        
        if cached:
            return ResearchReport(**json.loads (cached))
        
        return None
    
    async def set (self, query: str, report: ResearchReport):
        """Cache research result"""
        key = self._get_cache_key (query)
        await self.redis.setex(
            key,
            self.ttl,
            report.model_dump_json()
        )
    
    async def invalidate (self, query: str):
        """Invalidate cached result"""
        key = self._get_cache_key (query)
        await self.redis.delete (key)
\`\`\`

### Error Handling & Retries

\`\`\`python
"""
Robust error handling for research pipeline
"""

from tenacity import retry, stop_after_attempt, wait_exponential

class RobustResearchOrchestrator(ResearchOrchestrator):
    """
    Research orchestrator with error handling
    """
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential (multiplier=1, min=4, max=10)
    )
    async def research (self, query: str, **kwargs) -> ResearchReport:
        """Research with automatic retries"""
        try:
            return await super().research (query, **kwargs)
        except Exception as e:
            logger.error (f"Research failed: {e}")
            raise
    
    async def safe_search (self, query: str) -> List[SearchResult]:
        """Search with fallback"""
        try:
            return await self.search_agent.search (query)
        except Exception as e:
            logger.warning (f"Search failed, using fallback: {e}")
            # Fallback to alternative search API
            return []
\`\`\`

---

## Conclusion

Building a production AI Research Assistant requires:

1. **Multi-Agent Architecture**: Specialized agents for planning, searching, processing, synthesizing
2. **Web Search Integration**: APIs like Serper, Bing for information gathering
3. **Document Processing**: Handle PDFs, DOCX, web content
4. **Synthesis & Fact-Checking**: Combine sources, verify claims
5. **Report Generation**: Structured, formatted, professional reports
6. **Orchestration**: Coordinate agents in complex workflows
7. **Caching & Optimization**: Reduce costs, improve speed
8. **Error Handling**: Robust retry logic, fallbacks

**Key Technologies**:
- **LangChain**: Multi-agent orchestration
- **Serper/Bing API**: Web search
- **OpenAI/Claude**: LLM reasoning
- **BeautifulSoup**: Web scraping
- **PyPDF/python-docx**: Document processing

The result is an autonomous research assistant that can conduct comprehensive research faster and more thoroughly than manual human research.
`,
};
