export const section11 = {
  title: 'NLP for Financial Documents',
  content: `
# NLP for Financial Documents

Apply natural language processing to extract insights from earnings calls, MD&A sections, and analyst reports - automate document analysis at scale.

## Section 1: Financial Sentiment Analysis with FinBERT

\`\`\`python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
from typing import Dict, List
import numpy as np

class AdvancedFinancialSentimentAnalyzer:
    """
    Production sentiment analysis using FinBERT.
    
    Features:
    - Sentence-level and document-level sentiment
    - Temporal sentiment tracking
    - Sentiment correlation with returns
    - Confidence-weighted scoring
    """
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def analyze_sentence(self, text: str) -> Dict:
        """Analyze sentiment of single sentence/paragraph."""
        
        inputs = self.tokenizer(text, 
                               return_tensors="pt", 
                               truncation=True, 
                               max_length=512,
                               padding=True)
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_idx = torch.argmax(probabilities).item()
        
        labels = ["negative", "neutral", "positive"]
        
        return {
            'sentiment': labels[sentiment_idx],
            'confidence': probabilities[0][sentiment_idx].item(),
            'scores': {
                'negative': probabilities[0][0].item(),
                'neutral': probabilities[0][1].item(),
                'positive': probabilities[0][2].item()
            },
            'sentiment_score': probabilities[0][2].item() - probabilities[0][0].item()  # -1 to +1
        }
    
    def analyze_document(self, text: str) -> Dict:
        """
        Analyze entire document by aggregating sentence-level sentiment.
        
        Uses confidence-weighted averaging for more accurate results.
        """
        import nltk
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Analyze each sentence
        sentence_results = []
        for sentence in sentences:
            if len(sentence.split()) < 3:  # Skip very short sentences
                continue
            
            result = self.analyze_sentence(sentence)
            sentence_results.append({
                'text': sentence[:100],  # First 100 chars for reference
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'sentiment_score': result['sentiment_score']
            })
        
        if not sentence_results:
            return {'error': 'No valid sentences found'}
        
        # Aggregate with confidence weighting
        total_weight = sum(r['confidence'] for r in sentence_results)
        weighted_score = sum(r['sentiment_score'] * r['confidence'] 
                           for r in sentence_results) / total_weight
        
        # Count sentiment distribution
        sentiment_counts = {
            'positive': sum(1 for r in sentence_results if r['sentiment'] == 'positive'),
            'neutral': sum(1 for r in sentence_results if r['sentiment'] == 'neutral'),
            'negative': sum(1 for r in sentence_results if r['sentiment'] == 'negative')
        }
        
        # Overall document sentiment
        if weighted_score > 0.2:
            overall = 'positive'
        elif weighted_score < -0.2:
            overall = 'negative'
        else:
            overall = 'neutral'
        
        return {
            'overall_sentiment': overall,
            'sentiment_score': weighted_score,
            'sentence_count': len(sentence_results),
            'sentiment_distribution': sentiment_counts,
            'average_confidence': sum(r['confidence'] for r in sentence_results) / len(sentence_results),
            'sentence_details': sentence_results[:10]  # Top 10 for inspection
        }
    
    def analyze_earnings_call_transcript(self, transcript: str) -> Dict:
        """
        Analyze earnings call transcript with section separation.
        
        Separates:
        - Prepared remarks (CEO/CFO statements)
        - Q&A section (analyst questions + management answers)
        """
        
        # Simple heuristic: Q&A usually starts with "Question", "Analyst", or "Q:"
        qa_start_patterns = [
            r'(?i)question.*answer',
            r'(?i)q&a',
            r'(?i)^analyst:',
            r'(?i)^q:'
        ]
        
        qa_start = len(transcript)
        for pattern in qa_start_patterns:
            match = re.search(pattern, transcript)
            if match:
                qa_start = min(qa_start, match.start())
        
        prepared_remarks = transcript[:qa_start]
        qa_section = transcript[qa_start:]
        
        # Analyze each section
        prepared_sentiment = self.analyze_document(prepared_remarks)
        qa_sentiment = self.analyze_document(qa_section)
        
        return {
            'prepared_remarks': prepared_sentiment,
            'qa_section': qa_sentiment,
            'divergence': abs(prepared_sentiment['sentiment_score'] - 
                            qa_sentiment['sentiment_score']),
            'red_flag': prepared_sentiment['sentiment_score'] > 0.3 and 
                       qa_sentiment['sentiment_score'] < -0.1  # Positive prepared, negative Q&A
        }

# Example usage
analyzer = AdvancedFinancialSentimentAnalyzer()

text = """
We achieved record revenue growth of 35% driven by strong demand across all segments.
Our margin expansion reflects operational excellence and pricing power.
However, we're monitoring supply chain headwinds that may impact Q4.
"""

result = analyzer.analyze_document(text)
print(f"Overall Sentiment: {result['overall_sentiment']}")
print(f"Sentiment Score: {result['sentiment_score']:.2f}")
print(f"Distribution: {result['sentiment_distribution']}")
\`\`\`

## Section 2: Named Entity Recognition & Information Extraction

\`\`\`python
import spacy
import re
from typing import List, Dict, Tuple
import pandas as pd

class FinancialInformationExtractor:
    """
    Extract structured information from financial documents.
    
    Extracts:
    - Financial metrics (revenue, earnings, margins)
    - Guidance figures
    - Company names and entities
    - Dates and time periods
    - Forward-looking statements
    """
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        # Comprehensive regex patterns
        self.patterns = {
            'revenue': [
                r'revenue[s]?\s+(?:of|was|grew|increased)?\s*(?:to|by)?\s*\$?([\d,\.]+)\s*(million|billion|M|B)',
                r'\$?([\d,\.]+)\s*(million|billion|M|B)\s+in\s+revenue'
            ],
            'earnings': [
                r'(?:earnings|EPS|income)\s+(?:of|was|grew)?\s*\$?([\d,\.]+)',
                r'net\s+income\s+of\s+\$?([\d,\.]+)\s*(million|billion|M|B)'
            ],
            'margin': [
                r'(?:gross|operating|net|EBITDA)\s+margin\s+(?:of|was)?\s*([\d,\.]+)%',
                r'margin\s+(?:expanded|contracted|improved)\s+(?:to|by)\s*([\d,\.]+)%'
            ],
            'growth': [
                r'(?:grew|increased|growth)\s+(?:of|by)?\s*([\d,\.]+)%',
                r'([\d,\.]+)%\s+(?:increase|growth|improvement)'
            ],
            'guidance': [
                r'(?:expect|forecast|guide|project|anticipate)\s+(?:revenue|earnings|EPS)\s+(?:of|to be)?\s*\$?([\d,\.]+)',
                r'guidance\s+(?:of|for)?\s*\$?([\d,\.]+)\s+(?:to|-)?\s*\$?([\d,\.]+)?'
            ]
        }
    
    def extract_all_metrics(self, text: str) -> Dict[str, List[Dict]]:
        """Extract all financial metrics from text."""
        
        results = {}
        
        for metric_type, pattern_list in self.patterns.items():
            matches = []
            
            for pattern in pattern_list:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Extract value and convert to float
                    value_str = match.group(1).replace(',', '')
                    
                    try:
                        value = float(value_str)
                        
                        # Handle units (million/billion)
                        if len(match.groups()) > 1 and match.group(2):
                            unit = match.group(2).upper()
                            if unit in ['BILLION', 'B']:
                                value *= 1_000_000_000
                            elif unit in ['MILLION', 'M']:
                                value *= 1_000_000
                        
                        # Extract context (50 chars before and after)
                        context_start = max(0, match.start() - 50)
                        context_end = min(len(text), match.end() + 50)
                        context = text[context_start:context_end].strip()
                        
                        matches.append({
                            'value': value,
                            'raw_text': match.group(0),
                            'context': context,
                            'position': match.start()
                        })
                    
                    except ValueError:
                        continue
            
            results[metric_type] = matches
        
        return results
    
    def extract_companies(self, text: str) -> List[str]:
        """Extract company names using NER."""
        
        doc = self.nlp(text)
        
        companies = []
        for ent in doc.ents:
            if ent.label_ == "ORG":
                companies.append(ent.text)
        
        return list(set(companies))  # Remove duplicates
    
    def extract_dates(self, text: str) -> List[Dict]:
        """Extract dates and time references."""
        
        doc = self.nlp(text)
        
        dates = []
        for ent in doc.ents:
            if ent.label_ == "DATE":
                dates.append({
                    'text': ent.text,
                    'position': ent.start_char
                })
        
        return dates
    
    def identify_forward_looking_statements(self, text: str) -> List[Dict]:
        """
        Identify forward-looking statements.
        
        These have legal significance and indicate future projections.
        """
        
        # Forward-looking indicator words
        fls_keywords = [
            'expect', 'anticipate', 'believe', 'estimate', 'intend',
            'plan', 'project', 'forecast', 'goal', 'target', 'outlook',
            'guidance', 'may', 'will', 'should', 'could', 'would'
        ]
        
        sentences = text.split('.')
        fls = []
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Check if sentence contains forward-looking keywords
            contains_fls = any(keyword in sentence_lower for keyword in fls_keywords)
            
            # Check for future tense
            has_future_tense = re.search(r'\bwill\b|\bshall\b', sentence_lower)
            
            if contains_fls or has_future_tense:
                fls.append({
                    'sentence': sentence.strip(),
                    'position': i,
                    'keywords': [kw for kw in fls_keywords if kw in sentence_lower]
                })
        
        return fls
    
    def extract_comparative_statements(self, text: str) -> List[Dict]:
        """Extract year-over-year or quarter-over-quarter comparisons."""
        
        comparison_patterns = [
            r'(?:increased|decreased|grew|declined)\s+(\d+)%\s+(?:year-over-year|YoY|y/y)',
            r'(?:up|down)\s+(\d+)%\s+(?:from|vs|versus)\s+(?:last|prior)\s+(?:year|quarter)',
            r'compared\s+to\s+(?:last|prior)\s+(?:year|quarter),?\s+(\d+)%'
        ]
        
        comparisons = []
        for pattern in comparison_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                comparisons.append({
                    'change_pct': float(match.group(1)),
                    'text': match.group(0),
                    'context': text[max(0, match.start()-50):min(len(text), match.end()+50)]
                })
        
        return comparisons
    
    def create_structured_summary(self, text: str) -> Dict:
        """Create comprehensive structured summary of financial document."""
        
        return {
            'metrics': self.extract_all_metrics(text),
            'companies_mentioned': self.extract_companies(text),
            'dates': self.extract_dates(text),
            'forward_looking': self.identify_forward_looking_statements(text),
            'comparisons': self.extract_comparative_statements(text),
            'document_length': len(text),
            'sentence_count': len(text.split('.'))
        }

# Example
extractor = FinancialInformationExtractor()

text = """
Apple Inc. reported revenue of $89.5 billion in Q4 2023, up 8% year-over-year.
Gross margin expanded to 44.1%, driven by favorable mix.
We expect revenue to grow 10-12% in Q1 2024, with EPS of $1.85 to $1.95.
"""

summary = extractor.create_structured_summary(text)
print("Extracted Metrics:", summary['metrics'])
print("Forward-Looking:", len(summary['forward_looking']), "statements")
\`\`\`

## Section 3: Topic Modeling for MD&A Analysis

\`\`\`python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
from typing import List, Dict

class FinancialTopicModeler:
    """
    Topic modeling for MD&A sections using LDA.
    
    Identifies:
    - Key themes discussed
    - Topic weight changes over time
    - Emerging risks or opportunities
    """
    
    def __init__(self, n_topics: int = 10):
        self.n_topics = n_topics
        self.vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            max_df=0.7,  # Ignore terms appearing in >70% of docs
            min_df=2     # Ignore terms appearing in <2 docs
        )
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=50,
            learning_method='online',
            random_state=42
        )
    
    def fit(self, documents: List[str]) -> 'FinancialTopicModeler':
        """Fit LDA model on document corpus."""
        
        # Create document-term matrix
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        
        # Fit LDA
        self.lda_model.fit(doc_term_matrix)
        
        return self
    
    def get_topic_keywords(self, n_words: int = 10) -> Dict[int, List[str]]:
        """Get top keywords for each topic."""
        
        feature_names = self.vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            topics[topic_idx] = [feature_names[i] for i in top_indices]
        
        return topics
    
    def get_document_topics(self, document: str) -> np.ndarray:
        """Get topic distribution for a document."""
        
        doc_term_matrix = self.vectorizer.transform([document])
        topic_dist = self.lda_model.transform(doc_term_matrix)
        
        return topic_dist[0]
    
    def analyze_topic_evolution(self, documents_by_year: Dict[int, str]) -> pd.DataFrame:
        """
        Analyze how topics evolve over time.
        
        Useful for detecting emerging themes like:
        - Increasing "China" mentions (geopolitical risk)
        - Rising "supply chain" discussion (operational challenges)
        - Growing "sustainability" focus (ESG initiatives)
        """
        
        # Get topic distribution for each year
        results = []
        
        for year, document in sorted(documents_by_year.items()):
            topic_dist = self.get_document_topics(document)
            
            result = {'year': year}
            for i, weight in enumerate(topic_dist):
                result[f'topic_{i}'] = weight
            
            results.append(result)
        
        df = pd.DataFrame(results)
        
        return df
    
    def detect_topic_shifts(self, 
                           documents_by_year: Dict[int, str],
                           threshold: float = 0.10) -> List[Dict]:
        """
        Detect significant topic weight changes (>10% shift).
        
        These often signal important business changes:
        - New markets (increased international focus)
        - Challenges (rising regulatory/litigation mentions)
        - Strategic pivots (new product categories)
        """
        
        df = self.analyze_topic_evolution(documents_by_year)
        shifts = []
        
        years = df['year'].values
        if len(years) < 2:
            return shifts
        
        first_year = df.iloc[0]
        latest_year = df.iloc[-1]
        
        topic_keywords = self.get_topic_keywords(n_words=5)
        
        for col in df.columns:
            if col.startswith('topic_'):
                topic_idx = int(col.split('_')[1])
                change = latest_year[col] - first_year[col]
                
                if abs(change) > threshold:
                    shifts.append({
                        'topic_id': topic_idx,
                        'keywords': topic_keywords[topic_idx],
                        'weight_change': change,
                        'initial_weight': first_year[col],
                        'final_weight': latest_year[col],
                        'direction': 'increasing' if change > 0 else 'decreasing'
                    })
        
        return sorted(shifts, key=lambda x: abs(x['weight_change']), reverse=True)

# Example usage
modeler = FinancialTopicModeler(n_topics=5)

# Simulate 5 years of MD&A sections
mda_documents = {
    2019: "Strong growth in domestic markets. New product launches successful...",
    2020: "Pandemic impact on operations. Supply chain disruptions. Remote work transition...",
    2021: "Recovery in demand. Digital transformation accelerating. E-commerce growth...",
    2022: "Inflation pressures. Rising costs. Pricing actions taken. Supply chain improving...",
    2023: "AI investments. International expansion, especially China. Regulatory challenges..."
}

# Fit model on all documents
modeler.fit(list(mda_documents.values()))

# Analyze topic evolution
topic_evolution = modeler.analyze_topic_evolution(mda_documents)
print("Topic Evolution:")
print(topic_evolution)

# Detect major shifts
shifts = modeler.detect_topic_shifts(mda_documents, threshold=0.15)
print(f"\\nDetected {len(shifts)} major topic shifts")
for shift in shifts:
    print(f"  Topic {shift['topic_id']}: {shift['keywords'][:3]}")
    print(f"  Changed {shift['weight_change']:.1%} ({shift['direction']})")
\`\`\`

## Section 4: Document Readability & Obfuscation Detection

\`\`\`python
import textstat
import re
from typing import Dict

class DocumentComplexityAnalyzer:
    """
    Analyze document readability and detect obfuscation.
    
    Research shows companies increase complexity when:
    - Hiding bad news
    - Facing litigation
    - Manipulating earnings
    """
    
    @staticmethod
    def calculate_fog_index(text: str) -> float:
        """
        Gunning Fog Index: measures readability.
        
        < 12: High school level (readable)
        12-16: College level  
        > 16: Graduate level (complex)
        > 18: Very difficult to read
        """
        return textstat.gunning_fog(text)
    
    @staticmethod
    def calculate_flesch_reading_ease(text: str) -> float:
        """
        Flesch Reading Ease: 0-100 scale.
        
        90-100: Very easy (5th grade)
        60-70: Standard (8-9th grade)
        30-50: Difficult (college)
        0-30: Very difficult (graduate)
        """
        return textstat.flesch_reading_ease(text)
    
    @staticmethod
    def count_passive_voice(text: str) -> Dict:
        """
        Count passive voice usage.
        
        High passive voice often indicates evasiveness:
        - "Mistakes were made" (passive) vs
        - "We made mistakes" (active)
        """
        
        # Simple heuristic: "was/were/been" + past participle
        passive_patterns = [
            r'\b(was|were|been)\s+\w+ed\b',
            r'\b(was|were|been)\s+\w+en\b'
        ]
        
        total_matches = 0
        for pattern in passive_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            total_matches += len(matches)
        
        sentence_count = len(text.split('.'))
        passive_per_sentence = total_matches / sentence_count if sentence_count > 0 else 0
        
        return {
            'passive_count': total_matches,
            'passive_per_sentence': passive_per_sentence,
            'is_high': passive_per_sentence > 0.3  # >30% is high
        }
    
    @staticmethod
    def count_hedge_words(text: str) -> Dict:
        """
        Count hedging/qualifying words.
        
        Examples: "possibly", "approximately", "substantially"
        High hedging may indicate uncertainty or evasiveness.
        """
        
        hedge_words = [
            'approximately', 'substantially', 'generally', 'largely',
            'possibly', 'probably', 'perhaps', 'somewhat', 'relatively',
            'certain', 'various', 'some', 'several', 'significant',
            'material', 'considerable', 'potential', 'apparent'
        ]
        
        text_lower = text.lower()
        word_count = len(text.split())
        
        hedge_count = sum(text_lower.count(word) for word in hedge_words)
        hedge_ratio = hedge_count / word_count if word_count > 0 else 0
        
        return {
            'hedge_count': hedge_count,
            'hedge_ratio': hedge_ratio,
            'is_high': hedge_ratio > 0.03  # >3% is high
        }
    
    @staticmethod
    def analyze_complexity_changes(current_text: str, prior_text: str) -> Dict:
        """
        Compare complexity between periods.
        
        Increasing complexity is red flag for obfuscation.
        """
        
        current_fog = DocumentComplexityAnalyzer.calculate_fog_index(current_text)
        prior_fog = DocumentComplexityAnalyzer.calculate_fog_index(prior_text)
        
        current_passive = DocumentComplexityAnalyzer.count_passive_voice(current_text)
        prior_passive = DocumentComplexityAnalyzer.count_passive_voice(prior_text)
        
        current_hedge = DocumentComplexityAnalyzer.count_hedge_words(current_text)
        prior_hedge = DocumentComplexityAnalyzer.count_hedge_words(prior_text)
        
        # Calculate changes
        fog_change = current_fog - prior_fog
        passive_change = current_passive['passive_per_sentence'] - prior_passive['passive_per_sentence']
        hedge_change = current_hedge['hedge_ratio'] - prior_hedge['hedge_ratio']
        
        # Red flags
        red_flags = []
        
        if fog_change > 2.0:
            red_flags.append(f"Fog Index increased {fog_change:.1f} points - document became more complex")
        
        if passive_change > 0.15:
            red_flags.append(f"Passive voice increased {passive_change:.0%} - potential evasiveness")
        
        if hedge_change > 0.01:
            red_flags.append(f"Hedge words increased {hedge_change:.0%} - increased uncertainty language")
        
        return {
            'fog_index': {'current': current_fog, 'prior': prior_fog, 'change': fog_change},
            'passive_voice': {'current': current_passive['passive_per_sentence'],
                            'prior': prior_passive['passive_per_sentence'],
                            'change': passive_change},
            'hedge_words': {'current': current_hedge['hedge_ratio'],
                          'prior': prior_hedge['hedge_ratio'],
                          'change': hedge_change},
            'red_flags': red_flags,
            'overall_assessment': 'OBFUSCATION DETECTED' if len(red_flags) >= 2 else 'NORMAL'
        }

# Example
analyzer = DocumentComplexityAnalyzer()

prior_mda = "We grew revenue 20%. Margins improved. We invested in new products."

current_mda = """
Substantially favorable macroeconomic circumstances, in conjunction with certain
operational efficiencies that were implemented during the fiscal period, contributed
to approximately enhanced performance metrics across various segments of our
diversified business portfolio, notwithstanding certain challenges that may have
been encountered in specific market conditions.
"""

analysis = analyzer.analyze_complexity_changes(current_mda, prior_mda)

print(f"Fog Index Change: {analysis['fog_index']['change']:.1f}")
print(f"Assessment: {analysis['overall_assessment']}")
print(f"Red Flags: {len(analysis['red_flags'])}")
for flag in analysis['red_flags']:
    print(f"  - {flag}")
\`\`\`

## Section 5: Sentiment-Return Correlation Analysis

\`\`\`python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class SentimentReturnCorrelation:
    """
    Analyze correlation between document sentiment and stock returns.
    
    Validates if NLP sentiment has predictive power.
    """
    
    @staticmethod
    def calculate_returns(prices: pd.Series, periods: List[int] = [1, 5, 30]) -> pd.DataFrame:
        """Calculate forward returns for multiple periods."""
        
        returns = pd.DataFrame(index=prices.index)
        
        for period in periods:
            returns[f't+{period}'] = prices.pct_change(period).shift(-period)
        
        return returns
    
    @staticmethod
    def correlate_sentiment_returns(
        sentiment_scores: pd.Series,
        stock_returns: pd.DataFrame
    ) -> Dict:
        """Calculate correlation between sentiment and forward returns."""
        
        correlations = {}
        
        for col in stock_returns.columns:
            # Remove NaN values
            mask = ~(sentiment_scores.isna() | stock_returns[col].isna())
            
            if mask.sum() < 10:  # Need at least 10 observations
                continue
            
            corr, pvalue = stats.pearsonr(
                sentiment_scores[mask],
                stock_returns[col][mask]
            )
            
            correlations[col] = {
                'correlation': corr,
                'p_value': pvalue,
                'significant': pvalue < 0.05,
                'sample_size': mask.sum()
            }
        
        return correlations
    
    @staticmethod
    def backtest_sentiment_strategy(
        sentiment_scores: pd.Series,
        returns: pd.Series,
        threshold: float = 0.3
    ) -> Dict:
        """
        Backtest simple sentiment strategy:
        - Buy when sentiment > threshold
        - Sell when sentiment < -threshold
        - Hold otherwise
        """
        
        # Generate signals
        signals = pd.Series(0, index=sentiment_scores.index)
        signals[sentiment_scores > threshold] = 1  # Buy
        signals[sentiment_scores < -threshold] = -1  # Sell
        
        # Calculate strategy returns
        strategy_returns = signals.shift(1) * returns
        
        # Performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()
        
        # Win rate
        trades = strategy_returns[strategy_returns != 0]
        win_rate = (trades > 0).sum() / len(trades) if len(trades) > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'avg_return_per_trade': trades.mean() if len(trades) > 0 else 0
        }

# Example
# correlator = SentimentReturnCorrelation()

# Simulate data
# dates = pd.date_range('2020-01-01', periods=100, freq='Q')
# sentiment = pd.Series(np.random.randn(100) * 0.5, index=dates)
# returns = pd.Series(np.random.randn(100) * 0.03 + sentiment * 0.02, index=dates)

# results = correlator.backtest_sentiment_strategy(sentiment, returns)
# print(f"Total Return: {results['total_return']:.1%}")
# print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
# print(f"Win Rate: {results['win_rate']:.1%}")
\`\`\`

## Key Takeaways

1. **FinBERT is essential** - 15% better accuracy than generic BERT on financial text
2. **Sentence-level > Document-level** - Aggregate carefully with confidence weighting
3. **Context matters** - Same word different meaning in different contexts
4. **Complexity increases signal problems** - Rising Fog Index = potential obfuscation
5. **Topic modeling reveals themes** - LDA identifies emerging risks/opportunities
6. **Sentiment predicts returns** - Positive earnings calls → 2-5% outperformance over 30 days
7. **Forward-looking statements** - Legal significance, indicates management confidence
8. **Passive voice = evasiveness** - "Mistakes were made" vs "We made mistakes"

Master financial NLP and you can analyze thousands of documents in minutes, extracting insights that would take analysts weeks!
`,
  discussionQuestions: [],
  multipleChoiceQuestions: [],
};
