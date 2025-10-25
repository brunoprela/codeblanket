export const earningsCallAnalysis = {
  title: 'Earnings Call Analysis',
  id: 'earnings-call-analysis',
  content: `
# Earnings Call Analysis

## Introduction

Earnings calls are quarterly events where company executives discuss financial results, strategy, and outlook with analysts and investors. These calls contain rich, timely information that can drive significant market movements. However, they're delivered as audio/video with accompanying transcripts, making systematic analysis challenging without automation.

LLMs enable sophisticated analysis of earnings calls including transcript processing, sentiment analysis, management tone detection, Q&A analysis, and comparison across quarters. This section covers how to build production systems for real-time earnings call analysis to generate trading signals.

### Why Earnings Calls Matter

**Timeliness**: Calls happen within hours of earnings release, before full 10-Q filing
**Forward-Looking**: Management provides guidance and outlook
**Tone & Confidence**: Voice tone and word choice reveal confidence levels
**Analyst Questions**: Reveal concerns and focus areas
**Market Impact**: Calls can cause immediate 5-10%+ price movements

---

## Obtaining Earnings Call Transcripts

### Sources for Earnings Transcripts

\`\`\`python
"""
Multiple sources for earnings call transcripts
"""

import requests
from typing import List, Dict, Optional
from datetime import datetime

class EarningsTranscriptRetriever:
    """
    Retrieve earnings call transcripts from various sources
    """
    
    def __init__(self):
        self.sources = {
            'seeking_alpha': 'Free (limited), extensive coverage',
            'fool': 'Free, good coverage',
            'earnings_call_api': 'Paid, comprehensive',
            'company_ir': 'Free, direct from company'
        }
    
    def get_from_seeking_alpha (self, ticker: str, year: int, quarter: int) -> Optional[str]:
        """
        Scrape transcript from Seeking Alpha (educational example)
        Note: Check ToS before scraping
        
        Args:
            ticker: Stock ticker
            year: Year (e.g., 2024)
            quarter: Quarter (1-4)
            
        Returns:
            Transcript text or None
        """
        # Format: /article/{id}-{ticker}-q{quarter}-{year}-earnings-call-transcript
        # In practice, you'd need to:
        # 1. Search for the specific earnings call article
        # 2. Parse the article page
        # 3. Extract transcript text
        
        # This is a simplified example - actual implementation would be more complex
        search_query = f"{ticker} Q{quarter} {year} earnings call transcript"
        
        # Seeking Alpha requires authentication and has rate limits
        # Better approach: Use their API or partner services
        
        print(f"Search query: {search_query}")
        print("Note: Consider using Seeking Alpha API or partners like AlphaVantage")
        
        return None
    
    def get_from_company_ir (self, ticker: str, ir_url: str) -> List[Dict]:
        """
        Get earnings call information from company IR website
        
        Args:
            ticker: Stock ticker
            ir_url: Company investor relations URL
            
        Returns:
            List of earnings call metadata
        """
        # Most companies provide:
        # - Webcast links (often archived)
        # - PDF transcripts (sometimes)
        # - Audio recordings
        
        # Example for a typical IR site structure
        print(f"Checking {ticker} IR site: {ir_url}")
        
        # In practice, this requires custom parsing per company
        # as IR site structures vary significantly
        
        return []
    
    def get_from_api (self, ticker: str, api_key: str, 
                    count: int = 4) -> List[Dict]:
        """
        Get transcripts from paid API service
        
        Example using a hypothetical API
        """
        # Example services:
        # - Financial Modeling Prep API
        # - AlphaVantage
        # - Earnings Call API providers
        
        url = f"https://api.example.com/v1/transcripts/{ticker}"
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {"limit": count}
        
        try:
            response = requests.get (url, headers=headers, params=params)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"API request failed: {e}")
        
        return []

# Sample transcript structure
SAMPLE_TRANSCRIPT = """
Apple Inc. Q4 2023 Earnings Call
October 26, 2023

OPERATOR: Good day, and welcome to the Apple Inc. Fourth Quarter Fiscal Year 2023 Earnings Conference Call. Today\'s call is being recorded.

At this time, for opening remarks and introductions, I would like to turn the call over to Suhasini Chandramouli, Director of Investor Relations. Please go ahead.

SUHASINI CHANDRAMOULI: Thank you. Good afternoon, and thank you for joining us. Speaking today is Tim Cook, Apple's CEO; and Luca Maestri, Apple's CFO. After that, we'll open the call to questions from analysts.

Please note that some of the information you'll hear during our discussion today will consist of forward-looking statements...

TIM COOK: Thank you, Suhasini. Good afternoon, everyone, and thanks for joining the call. 

Today, Apple is reporting revenue of $89.5 billion, down 1% from a year ago. We set an all-time revenue record in Services and a September quarter revenue record in iPhone...

Our installed base of active devices reached an all-time high across all products and geographic segments, a testament to the unparalleled loyalty and satisfaction of our customers...

[Continues with detailed results and outlook]

LUCA MAESTRI: Thank you, Tim. Revenue for the September quarter was $89.5 billion, down 1% year over year. Foreign exchange headwinds had a negative impact of over 200 basis points...

[Detailed financial breakdown]

OPERATOR: We will now begin the question-and-answer session. Our first question comes from Katy Huberty from Morgan Stanley.

KATY HUBERTY: Yes, thanks. Good afternoon. Tim, you mentioned the iPhone installed base grew... [Question continues]

TIM COOK: Thanks for the question, Katy... [Answer continues]

[Q&A continues]

OPERATOR: That concludes today's question-and-answer session. I would now like to turn the conference back over to Suhasini Chandramouli for closing remarks.

SUHASINI CHANDRAMOULI: Thank you for joining us today. A replay of today's call will be available...
"""
\`\`\`

---

## Parsing Earnings Call Structure

### Extracting Sections from Transcripts

\`\`\`python
"""
Parse earnings call transcripts into structured sections
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class Speaker:
    """Represent a speaker in the call"""
    name: str
    title: str
    affiliation: str  # Company or analyst firm

@dataclass
class Statement:
    """Represent a statement in the call"""
    speaker: Speaker
    text: str
    section: str  # 'prepared_remarks' or 'qa'
    timestamp: Optional[str] = None

class EarningsCallParser:
    """
    Parse earnings call transcripts into structured format
    """
    
    def __init__(self):
        # Common section markers
        self.section_markers = {
            'qa_start': [
                r'question.and.answer',
                r'Q&A',
                r'open.+for.questions',
                r'begin.+question'
            ],
            'closing': [
                r'concludes.+call',
                r'closing.remarks',
                r'thank.+you.+for.+joining'
            ]
        }
    
    def identify_speakers (self, transcript: str) -> List[Speaker]:
        """
        Identify all speakers mentioned in transcript
        
        Returns:
            List of Speaker objects
        """
        speakers = []
        
        # Pattern: NAME, TITLE - often appears in introduction
        # Example: "Tim Cook, Chief Executive Officer"
        pattern = r'([A-Z][a-z]+\\s+[A-Z][a-z]+),\\s+([^:\\n]+)'
        
        matches = re.findall (pattern, transcript)
        
        for name, title in matches:
            # Determine if company exec or analyst
            if any (keyword in title.lower() for keyword in 
                  ['ceo', 'cfo', 'chief', 'president', 'director']):
                affiliation = 'Company'
            else:
                affiliation = 'Analyst'
            
            speakers.append(Speaker (name=name, title=title, affiliation=affiliation))
        
        return speakers
    
    def split_into_statements (self, transcript: str) -> List[Statement]:
        """
        Split transcript into individual statements
        
        Returns:
            List of Statement objects
        """
        statements = []
        
        # Common format: "SPEAKER NAME: statement text"
        # or "SPEAKER NAME (TITLE): statement text"
        pattern = r'([A-Z\\s]+?)(?:\\s+\\([^)]+\\))?:\\s+(.+?)(?=\\n[A-Z\\s]+?:|$)'
        
        matches = re.findall (pattern, transcript, re.DOTALL)
        
        current_section = 'prepared_remarks'
        
        for speaker_name, text in matches:
            speaker_name = speaker_name.strip()
            text = text.strip()
            
            # Check if we've entered Q&A
            if any (re.search (pattern, text.lower()) 
                  for pattern in self.section_markers['qa_start']):
                current_section = 'qa'
                continue
            
            # Skip operator statements (usually procedural)
            if speaker_name == 'OPERATOR':
                continue
            
            # Create Speaker object (simplified - in practice, lookup from identified speakers)
            speaker = Speaker (name=speaker_name, title='', affiliation='')
            
            statements.append(Statement(
                speaker=speaker,
                text=text,
                section=current_section
            ))
        
        return statements
    
    def extract_sections (self, transcript: str) -> Dict[str, str]:
        """
        Extract major sections from transcript
        
        Returns:
            Dictionary with section names and content
        """
        sections = {}
        
        # Find Q&A start
        qa_start = None
        for pattern in self.section_markers['qa_start']:
            match = re.search (pattern, transcript, re.IGNORECASE)
            if match:
                qa_start = match.start()
                break
        
        if qa_start:
            sections['prepared_remarks'] = transcript[:qa_start].strip()
            sections['qa'] = transcript[qa_start:].strip()
        else:
            # If can't find Q&A marker, it's all prepared remarks
            sections['prepared_remarks'] = transcript
            sections['qa'] = ''
        
        return sections
    
    def extract_qa_pairs (self, qa_section: str) -> List[Dict[str, str]]:
        """
        Extract question-answer pairs from Q&A section
        
        Returns:
            List of dictionaries with 'analyst', 'question', 'executive', 'answer'
        """
        qa_pairs = []
        
        # Pattern for analyst questions (simplified)
        # Usually: Analyst name from Firm: question text
        # Followed by: Executive name: answer text
        
        # This is a simplified version - production systems need more robust parsing
        statements = self.split_into_statements (qa_section)
        
        i = 0
        while i < len (statements) - 1:
            current = statements[i]
            next_stmt = statements[i + 1]
            
            # If current is from analyst and next from company, it's a Q&A pair
            if (current.speaker.affiliation == 'Analyst' and 
                next_stmt.speaker.affiliation == 'Company'):
                
                qa_pairs.append({
                    'analyst': current.speaker.name,
                    'question': current.text,
                    'executive': next_stmt.speaker.name,
                    'answer': next_stmt.text
                })
                
                i += 2  # Skip both statements
            else:
                i += 1
        
        return qa_pairs

# Example usage
parser = EarningsCallParser()

# Parse sample transcript
sections = parser.extract_sections(SAMPLE_TRANSCRIPT)
print(f"Prepared Remarks: {len (sections['prepared_remarks'])} chars")
print(f"Q&A: {len (sections['qa'])} chars")

# Extract statements
statements = parser.split_into_statements(SAMPLE_TRANSCRIPT)
print(f"\\nTotal statements: {len (statements)}")

# Show first few statements
for i, stmt in enumerate (statements[:3]):
    print(f"\\n{i+1}. {stmt.speaker.name} ({stmt.section}):")
    print(f"   {stmt.text[:100]}...")
\`\`\`

---

## LLM-Based Sentiment Analysis

### Analyzing Management Tone and Sentiment

\`\`\`python
"""
Analyze sentiment and tone in earnings calls
"""

import anthropic
from typing import Dict, List

class EarningsCallAnalyzer:
    """
    Analyze earnings calls using LLMs
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic (api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def analyze_management_tone (self, prepared_remarks: str) -> Dict:
        """
        Analyze management's tone and sentiment in prepared remarks
        
        Args:
            prepared_remarks: Text of prepared remarks section
            
        Returns:
            Structured analysis of tone and sentiment
        """
        prompt = f"""Analyze the tone and sentiment of this earnings call prepared remarks.

Provide analysis in JSON format with:
1. overall_sentiment: "Positive", "Neutral", or "Negative"
2. confidence_level: "High", "Medium", or "Low" (based on language used)
3. key_themes: List of 3-5 main themes discussed
4. forward_looking_statements: Summary of guidance and outlook
5. concerns_mentioned: Any challenges or headwinds discussed
6. optimistic_signals: Positive indicators or achievements highlighted
7. tone_shift: Compared to typical earnings call tone, is this more positive/negative?
8. red_flags: Any concerning language or evasiveness (if present)

Prepared Remarks:
{prepared_remarks[:8000]}

Return analysis as JSON."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def analyze_qa_session (self, qa_pairs: List[Dict[str, str]]) -> Dict:
        """
        Analyze Q&A session for insights
        
        Args:
            qa_pairs: List of question-answer pairs
            
        Returns:
            Analysis of Q&A session
        """
        # Format Q&A pairs for analysis
        qa_text = "\\n\\n".join([
            f"Q ({qa['analyst']}): {qa['question']}\\n"
            f"A ({qa['executive']}): {qa['answer']}"
            for qa in qa_pairs[:10]  # Limit to first 10 Q&As
        ])
        
        prompt = f"""Analyze this earnings call Q&A session.

Focus on:
1. Most frequently asked topics (what are analysts concerned about?)
2. Management\'s responsiveness (do they answer directly or deflect?)
3. Defensive vs confident responses
4. New information revealed (that wasn't in prepared remarks)
5. Questions that were dodged or answered vaguely
6. Analyst sentiment (are analysts skeptical or optimistic?)
7. Overall impression (did management handle questions well?)

Q&A Session:
{qa_text}

Provide analysis in structured format."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def compare_to_guidance (self, current_remarks: str, 
                           previous_guidance: str) -> str:
        """
        Compare current results to previous guidance
        
        Args:
            current_remarks: Current quarter's remarks
            previous_guidance: Previous quarter's guidance
            
        Returns:
            Analysis of how results compare to guidance
        """
        prompt = f"""Compare the current quarter's results to previous guidance.

Previous Quarter\'s Guidance:
{previous_guidance}

Current Quarter's Results:
{current_remarks[:6000]}

Analyze:
1. Did they meet, beat, or miss guidance?
2. What metrics exceeded expectations?
3. What metrics fell short?
4. How did management explain any misses?
5. Has guidance for next quarter changed (increased/decreased)?
6. Overall: Beat, Meet, or Miss?

Provide detailed comparison."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def identify_key_metrics (self, transcript: str) -> Dict:
        """
        Extract specific metrics and numbers mentioned
        
        Args:
            transcript: Full transcript or prepared remarks
            
        Returns:
            Extracted metrics and KPIs
        """
        prompt = f"""Extract all key financial metrics and KPIs mentioned in this earnings call.

For each metric, provide:
- Metric name
- Actual value
- Year-over-year change (if mentioned)
- Management's commentary on the metric

Focus on:
- Revenue (total and by segment)
- Earnings per share (EPS)
- Margins (gross, operating, net)
- User/customer metrics
- Guidance for next quarter
- Cash flow and cash position
- Any other material metrics

Transcript excerpt:
{transcript[:10000]}

Return as structured JSON with metric categories."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def generate_investment_summary (self, full_analysis: Dict) -> str:
        """
        Generate investment-focused summary from all analyses
        
        Args:
            full_analysis: Dict containing all previous analyses
            
        Returns:
            Investment summary with recommendation
        """
        combined_analysis = f"""
Management Tone Analysis:
{full_analysis['tone_analysis']}

Q&A Analysis:
{full_analysis['qa_analysis']}

Key Metrics:
{full_analysis['metrics']}
"""

        prompt = f"""Based on this comprehensive earnings call analysis, provide an investment-focused summary.

Include:
1. Key Takeaways (3-5 bullet points)
2. Strengths demonstrated in the call
3. Concerns or weaknesses identified
4. Surprise factors (positive or negative)
5. Guidance and outlook assessment
6. Potential market reaction (likely positive/negative/neutral)
7. Comparison to consensus expectations (if you can infer)
8. Investment implications (what should investors consider?)

Analysis:
{combined_analysis}

Provide a comprehensive but concise investment summary (400-500 words)."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

# Example usage
analyzer = EarningsCallAnalyzer (api_key="your-key")

# Analyze prepared remarks
sections = parser.extract_sections(SAMPLE_TRANSCRIPT)
tone_analysis = analyzer.analyze_management_tone (sections['prepared_remarks'])
print("Management Tone Analysis:")
print(tone_analysis)

# Analyze Q&A
qa_pairs = parser.extract_qa_pairs (sections['qa'])
qa_analysis = analyzer.analyze_qa_session (qa_pairs)
print("\\nQ&A Analysis:")
print(qa_analysis)

# Extract metrics
metrics = analyzer.identify_key_metrics(SAMPLE_TRANSCRIPT)
print("\\nKey Metrics:")
print(metrics)

# Generate investment summary
full_analysis = {
    'tone_analysis': tone_analysis,
    'qa_analysis': qa_analysis,
    'metrics': metrics
}
summary = analyzer.generate_investment_summary (full_analysis)
print("\\nInvestment Summary:")
print(summary)
\`\`\`

---

## Real-Time Earnings Analysis

### Processing Calls as They Happen

\`\`\`python
"""
Real-time earnings call analysis system
"""

import threading
import queue
from datetime import datetime
import json

class RealTimeEarningsAnalyzer:
    """
    Analyze earnings calls in real-time as transcript becomes available
    """
    
    def __init__(self, api_key: str):
        self.analyzer = EarningsCallAnalyzer (api_key)
        self.parser = EarningsCallParser()
        
        # Queue for transcript chunks
        self.transcript_queue = queue.Queue()
        self.analysis_results = []
        
        # State tracking
        self.current_section = 'prepared_remarks'
        self.accumulated_text = ''
        
    def process_transcript_chunk (self, chunk: str):
        """
        Process incoming transcript chunk
        
        Args:
            chunk: New text from live transcript
        """
        self.transcript_queue.put (chunk)
    
    def start_analysis (self):
        """
        Start real-time analysis thread
        """
        analysis_thread = threading.Thread (target=self._analysis_worker)
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def _analysis_worker (self):
        """
        Worker thread that processes transcript chunks
        """
        while True:
            try:
                # Get chunk from queue (blocks until available)
                chunk = self.transcript_queue.get (timeout=1)
                
                self.accumulated_text += chunk
                
                # Analyze when we have enough text
                if len (self.accumulated_text) > 1000:
                    self._perform_incremental_analysis()
                
                self.transcript_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Analysis error: {e}")
    
    def _perform_incremental_analysis (self):
        """
        Perform analysis on accumulated text
        """
        # Quick sentiment check
        prompt = f"""Provide a quick sentiment assessment of this earnings call excerpt.

Is the tone: Positive, Neutral, or Negative?
Key point mentioned (one sentence)?

Excerpt:
{self.accumulated_text[-2000:]}

Response format:
Sentiment: [Positive/Neutral/Negative]
Key Point: [one sentence]"""

        try:
            response = self.analyzer.client.messages.create(
                model=self.analyzer.model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'text_length': len (self.accumulated_text),
                'analysis': response.content[0].text
            }
            
            self.analysis_results.append (result)
            
            # Emit real-time signal
            self._emit_signal (result)
            
        except Exception as e:
            print(f"LLM analysis error: {e}")
    
    def _emit_signal (self, result: Dict):
        """
        Emit trading signal based on analysis
        
        Args:
            result: Analysis result
        """
        # Extract sentiment
        analysis_text = result['analysis'].lower()
        
        if 'positive' in analysis_text:
            signal = 'BUY'
            confidence = 0.6  # Initial confidence
        elif 'negative' in analysis_text:
            signal = 'SELL'
            confidence = 0.6
        else:
            signal = 'HOLD'
            confidence = 0.4
        
        trading_signal = {
            'timestamp': result['timestamp'],
            'signal': signal,
            'confidence': confidence,
            'reason': result['analysis']
        }
        
        print(f"\\n{'='*50}")
        print(f"REAL-TIME SIGNAL: {signal} (confidence: {confidence})")
        print(f"Reason: {trading_signal['reason']}")
        print(f"{'='*50}\\n")
        
        # In production, this would:
        # 1. Send to trading system
        # 2. Update dashboard
        # 3. Send alerts
        # 4. Log to database
    
    def get_final_analysis (self) -> Dict:
        """
        Generate comprehensive analysis after call ends
        
        Returns:
            Complete analysis
        """
        sections = self.parser.extract_sections (self.accumulated_text)
        
        # Comprehensive analysis now that we have full transcript
        full_analysis = {
            'tone_analysis': self.analyzer.analyze_management_tone(
                sections['prepared_remarks']
            ),
            'qa_analysis': self.analyzer.analyze_qa_session(
                self.parser.extract_qa_pairs (sections['qa'])
            ),
            'metrics': self.analyzer.identify_key_metrics (self.accumulated_text),
            'incremental_results': self.analysis_results
        }
        
        # Generate final summary
        summary = self.analyzer.generate_investment_summary (full_analysis)
        full_analysis['investment_summary'] = summary
        
        return full_analysis

# Example: Simulating real-time analysis
def simulate_real_time_call():
    """
    Simulate receiving transcript in real-time
    """
    analyzer = RealTimeEarningsAnalyzer (api_key="your-key")
    analyzer.start_analysis()
    
    # Simulate transcript chunks arriving
    transcript_chunks = SAMPLE_TRANSCRIPT.split('\\n\\n')
    
    for i, chunk in enumerate (transcript_chunks):
        print(f"Receiving chunk {i+1}/{len (transcript_chunks)}...")
        analyzer.process_transcript_chunk (chunk)
        
        # Simulate delay between chunks (real calls unfold over 60 min)
        import time
        time.sleep(2)
    
    # Wait for queue to be processed
    analyzer.transcript_queue.join()
    
    # Get final comprehensive analysis
    print("\\n\\nGenerating final analysis...")
    final_analysis = analyzer.get_final_analysis()
    
    print("\\nFinal Investment Summary:")
    print(final_analysis['investment_summary'])
    
    return final_analysis

# Run simulation
# final_results = simulate_real_time_call()
\`\`\`

---

## Comparative Analysis

### Tracking Changes Quarter-over-Quarter

\`\`\`python
"""
Compare earnings calls across quarters
"""

from typing import List
import pandas as pd

class EarningsComparator:
    """
    Compare earnings calls across multiple quarters
    """
    
    def __init__(self, api_key: str):
        self.analyzer = EarningsCallAnalyzer (api_key)
    
    def compare_tone_evolution (self, transcripts: List[Dict]) -> pd.DataFrame:
        """
        Track tone evolution across quarters
        
        Args:
            transcripts: List of dicts with 'quarter', 'year', 'text'
            
        Returns:
            DataFrame with tone metrics over time
        """
        results = []
        
        for transcript in transcripts:
            # Analyze tone
            tone_analysis = self.analyzer.analyze_management_tone(
                transcript['text']
            )
            
            # Parse JSON response (simplified)
            import json
            try:
                tone_data = json.loads (tone_analysis)
                results.append({
                    'quarter': f"Q{transcript['quarter']} {transcript['year']}",
                    'sentiment': tone_data.get('overall_sentiment'),
                    'confidence': tone_data.get('confidence_level'),
                    'key_themes': ', '.join (tone_data.get('key_themes', [])),
                })
            except:
                pass
        
        return pd.DataFrame (results)
    
    def identify_narrative_changes (self, current: str, previous: str) -> str:
        """
        Identify changes in narrative between quarters
        
        Args:
            current: Current quarter transcript
            previous: Previous quarter transcript
            
        Returns:
            Analysis of narrative changes
        """
        prompt = f"""Compare these two earnings call transcripts from consecutive quarters.

Identify:
1. Topics that are now emphasized (mentioned more or with more detail)
2. Topics that are de-emphasized or no longer mentioned
3. Changes in tone about specific business segments
4. New strategic initiatives mentioned
5. Changes in language about challenges/headwinds
6. Evolution of guidance and outlook
7. Overall: Is the narrative improving or deteriorating?

Previous Quarter:
{previous[:8000]}

Current Quarter:
{current[:8000]}

Provide detailed comparison of narrative evolution."""

        response = self.analyzer.client.messages.create(
            model=self.analyzer.model,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def track_management_credibility (self, 
                                    historical_guidances: List[Dict],
                                    actual_results: List[Dict]) -> str:
        """
        Assess management's credibility based on guidance accuracy
        
        Args:
            historical_guidances: Past guidance from earnings calls
            actual_results: Actual results from subsequent quarters
            
        Returns:
            Credibility assessment
        """
        # Format data for analysis
        comparison_text = ""
        for guidance, actual in zip (historical_guidances, actual_results):
            comparison_text += f"""
Quarter: {guidance['quarter']}
Guided Revenue: {guidance.get('revenue_guidance')}
Actual Revenue: {actual.get('actual_revenue')}
Guided EPS: {guidance.get('eps_guidance')}
Actual EPS: {actual.get('actual_eps')}
---
"""

        prompt = f"""Analyze management's guidance credibility based on historical accuracy.

{comparison_text}

Provide:
1. Overall guidance accuracy (Generally Accurate / Mixed / Often Misses)
2. Bias direction (Conservative / Aggressive / Balanced)
3. Credibility score (1-10)
4. Reliability of guidance going forward
5. Any patterns in guidance accuracy

Deliver credibility assessment."""

        response = self.analyzer.client.messages.create(
            model=self.analyzer.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

# Example usage
comparator = EarningsComparator (api_key="your-key")

# Compare narrative changes
narrative_analysis = comparator.identify_narrative_changes(
    current=current_quarter_transcript,
    previous=previous_quarter_transcript
)

print("Narrative Evolution:")
print(narrative_analysis)
\`\`\`

---

## Production Pipeline

### End-to-End Earnings Analysis System

\`\`\`python
"""
Complete production system for earnings call analysis
"""

import schedule
import time
from datetime import datetime, timedelta

class EarningsAnalysisPipeline:
    """
    Production pipeline for automated earnings analysis
    """
    
    def __init__(self, anthropic_key: str, db_path: str = "earnings.db"):
        self.analyzer = EarningsCallAnalyzer (anthropic_key)
        self.parser = EarningsCallParser()
        self.comparator = EarningsComparator (anthropic_key)
        
        # Initialize database
        self._init_database (db_path)
    
    def _init_database (self, db_path: str):
        """Create database for storing analyses"""
        import sqlite3
        
        conn = sqlite3.connect (db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS earnings_calls (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                quarter INTEGER,
                year INTEGER,
                call_date TEXT,
                transcript_url TEXT,
                overall_sentiment TEXT,
                confidence_level TEXT,
                key_metrics TEXT,
                investment_summary TEXT,
                processed_date TEXT,
                UNIQUE(ticker, quarter, year)
            )
        """)
        
        conn.commit()
        conn.close()
        
        self.db_path = db_path
    
    def monitor_earnings_calendar (self):
        """
        Monitor for upcoming earnings calls
        """
        # In production, integrate with:
        # - Earnings calendar API
        # - Company IR calendars
        # - Financial data providers
        
        today = datetime.now().date()
        
        # Example: Check if companies are reporting today
        earnings_today = self._get_earnings_calendar (today)
        
        for ticker in earnings_today:
            print(f"Processing {ticker} earnings call...")
            self.process_earnings_call (ticker)
    
    def process_earnings_call (self, ticker: str):
        """
        Complete processing of an earnings call
        
        Args:
            ticker: Stock ticker
        """
        try:
            # 1. Retrieve transcript
            transcript = self._retrieve_transcript (ticker)
            
            if not transcript:
                print(f"Could not retrieve transcript for {ticker}")
                return
            
            # 2. Parse sections
            sections = self.parser.extract_sections (transcript)
            
            # 3. Analyze with LLM
            tone_analysis = self.analyzer.analyze_management_tone(
                sections['prepared_remarks']
            )
            
            qa_pairs = self.parser.extract_qa_pairs (sections['qa'])
            qa_analysis = self.analyzer.analyze_qa_session (qa_pairs)
            
            metrics = self.analyzer.identify_key_metrics (transcript)
            
            # 4. Generate summary
            full_analysis = {
                'tone_analysis': tone_analysis,
                'qa_analysis': qa_analysis,
                'metrics': metrics
            }
            
            summary = self.analyzer.generate_investment_summary (full_analysis)
            
            # 5. Store in database
            self._store_analysis (ticker, {
                'transcript': transcript,
                'tone_analysis': tone_analysis,
                'summary': summary,
                'metrics': metrics
            })
            
            # 6. Generate alerts if needed
            self._generate_alerts (ticker, full_analysis)
            
            print(f"Successfully processed {ticker} earnings call")
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    def _retrieve_transcript (self, ticker: str) -> Optional[str]:
        """Retrieve transcript for ticker"""
        # Implementation depends on data source
        # Could be: API call, web scraping, etc.
        pass
    
    def _store_analysis (self, ticker: str, analysis: Dict):
        """Store analysis in database"""
        import sqlite3
        import json
        
        conn = sqlite3.connect (self.db_path)
        cursor = conn.cursor()
        
        # Extract key fields (simplified)
        cursor.execute("""
            INSERT OR REPLACE INTO earnings_calls
            (ticker, quarter, year, call_date, investment_summary, 
             key_metrics, processed_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker,
            datetime.now().month // 3 + 1,  # Quarter
            datetime.now().year,
            datetime.now().date().isoformat(),
            analysis['summary'],
            json.dumps (analysis['metrics']),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _generate_alerts (self, ticker: str, analysis: Dict):
        """Generate alerts for significant findings"""
        # Parse sentiment from analysis
        # If strongly positive or negative, send alert
        
        # Example alert triggers:
        # - Significant sentiment change vs previous quarter
        # - Guidance miss
        # - Major risk factors mentioned
        # - Defensive responses in Q&A
        
        print(f"Alert check complete for {ticker}")
    
    def run_continuous (self):
        """
        Run continuously, checking for earnings calls
        """
        # Schedule checks
        schedule.every().day.at("09:00").do (self.monitor_earnings_calendar)
        schedule.every().day.at("16:00").do (self.monitor_earnings_calendar)
        
        print("Earnings analysis pipeline started...")
        
        while True:
            schedule.run_pending()
            time.sleep(60)

# Initialize pipeline
# pipeline = EarningsAnalysisPipeline (anthropic_key="your-key")
# pipeline.run_continuous()
\`\`\`

---

## Best Practices

### Key Takeaways

1. **Real-Time Processing**: Process calls as they happen for fastest signals
2. **Sentiment + Facts**: Combine sentiment analysis with factual metric extraction
3. **Comparative Analysis**: Always compare to previous quarters
4. **Q&A is Critical**: Analyst questions reveal true concerns
5. **Management Tone**: Confidence levels matter as much as content
6. **Track Credibility**: Monitor guidance accuracy over time
7. **Automate Alerts**: Set up alerts for significant sentiment shifts
8. **Validate with Audio**: Consider voice tone analysis from actual audio
9. **Context Matters**: Compare to consensus expectations and street guidance
10. **Fast Action**: Earnings-driven moves happen quickly; speed is essential

---

## Summary

We've covered:
- Obtaining earnings call transcripts from various sources
- Parsing transcripts into structured sections
- LLM-based sentiment and tone analysis
- Q&A session analysis for deeper insights
- Real-time analysis as calls unfold
- Comparative analysis across quarters
- Building production pipelines for automated analysis
- Trading signal generation from earnings analysis

In the next section, we'll cover financial news analysis at scale for generating trading signals from breaking news.
`,
};
