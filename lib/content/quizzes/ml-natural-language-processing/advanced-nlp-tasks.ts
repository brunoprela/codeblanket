import { QuizQuestion } from '../../../types';

export const advancedNlpTasksQuiz: QuizQuestion[] = [
  {
    id: 'advanced-nlp-dq-1',
    question:
      'Compare extractive and abstractive summarization. Which approach would you use for summarizing financial earnings reports, and why?',
    sampleAnswer: `**Extractive Summarization:**

Selects and combines existing sentences:
- Pulls important sentences directly from source
- No new text generated
- Example: Document has 50 sentences, extractive picks top 5

**How it works:**1. Score each sentence by importance (TF-IDF, graph-based, neural)
2. Select top-scoring sentences
3. Order sentences (chronologically or by score)

**Advantages:**
- Factually accurate (uses exact text)
- No hallucination risk
- Preserves technical terminology
- Fast and interpretable
- Works with smaller models

**Disadvantages:**
- Can be choppy/disjointed
- May include redundant information
- Limited flexibility
- Can't rephrase or simplify

**Abstractive Summarization:**

Generates new sentences capturing meaning:
- Creates summary in own words
- Can rephrase and synthesize
- Example: "Revenue grew 15% to $5.2B while margins expanded" (synthesized from multiple sentences)

**How it works:**1. Encoder processes full document
2. Decoder generates summary token-by-token
3. Attention mechanism focuses on relevant parts

**Advantages:**
- Natural, fluent summaries
- Can synthesize information across sections
- Flexible compression
- Can simplify complex language

**Disadvantages:**
- Hallucination risk (may generate false facts)
- Harder to verify accuracy
- Computationally expensive
- Requires large models (T5, BART, PEGASUS)

**For Financial Earnings Reports:**

**Recommendation: Start with Abstractive, verify with Extractive**

**Why abstractive:**1. **Synthesis needed**: Earnings reports discuss metrics across different sections
   - Revenue in one section, guidance in another
   - Abstractive can combine: "Revenue exceeded guidance at $5.2B (+15% YoY)"

2. **Professional presentation**: Stakeholders expect fluent summaries
   - Extractive excerpts can be disjointed
   - Abstractive creates coherent narratives

3. **Key metrics focus**: Can be trained to highlight specific financial metrics
   - Revenue, earnings, margins, guidance
   - Generate structured summaries (consistent format)

**But with safeguards:**1. **Hallucination detection**: Cross-reference generated facts with source
2. **Extractive backup**: Include source sentences for key claims
3. **Number validation**: Verify all financial figures appear in source
4. **Human review**: Critical financial decisions require human verification

**Hybrid Approach (recommended for production):**

\`\`\`python
def financial_summary_hybrid (earnings_report):
    # 1. Abstractive summary (fluent, synthesized)
    abstractive_summary = bart_model.generate (earnings_report)
    
    # 2. Extract key financial sentences (verification)
    key_sentences = extract_sentences_with_metrics (earnings_report)
    
    # 3. Verify all facts in abstractive are sourced
    verified_facts = verify_facts (abstractive_summary, key_sentences)
    
    # 4. Return both
    return {
        'summary': abstractive_summary,
        'supporting_quotes': key_sentences,
        'verified': verified_facts
    }
\`\`\`

**Example:**

**Input (earnings report excerpt):**
"Revenue in Q3 was $5.2 billion, up 15% year-over-year. Operating margin expanded to 25%, driven by cost efficiencies. We are raising full-year guidance to $20-21 billion."

**Extractive output:**
"Revenue in Q3 was $5.2 billion, up 15% year-over-year. We are raising full-year guidance to $20-21 billion."
- Accurate but choppy

**Abstractive output:**
"Strong Q3 results with $5.2B revenue (+15% YoY) and 25% operating margin led to raised full-year guidance of $20-21B."
- Fluent synthesis, but need to verify "25%" and "$20-21B" are accurate

**Hybrid output:**
- Main summary (abstractive)
- Source quotes for each claim
- Confidence scores

**Key Considerations for Financial Applications:**1. **Accuracy > Fluency**: In finance, a single wrong number can be catastrophic
2. **Regulatory compliance**: SEC requires accurate disclosure
3. **Liability**: False summaries can lead to investor lawsuits
4. **Trust**: Stakeholders need to trust the summaries

**Therefore:**
- Use abstractive for draft/readability
- Always verify with extractive evidence
- Include source references
- Human-in-the-loop for final review
- Never use for critical decisions without verification

**Modern Approach:**
- Fine-tune T5 or BART on earnings reports
- Train to generate structured summaries (revenue, margins, guidance)
- Include fact verification model
- Provide source attributions
- Monitor for hallucinations

The trend in financial NLP is toward abstractive summaries with strong verification guardrails.`,
    keyPoints: [
      'Extractive: selects existing sentences, accurate but choppy',
      'Abstractive: generates new text, fluent but risks hallucination',
      'Financial reports need synthesis (abstractive) but accuracy (extractive)',
      'Hybrid approach: abstractive summary + extractive verification',
      'Always verify financial figures, include sources, human review',
      'Modern solution: fine-tuned abstractive with fact verification',
    ],
  },
  {
    id: 'advanced-nlp-dq-2',
    question:
      'Explain how few-shot learning works with large language models. How would you use it for financial sentiment analysis without labeled data?',
    sampleAnswer: `**Few-Shot Learning with LLMs:**

Few-shot learning enables models to perform tasks with only a few examples provided in the prompt, without fine-tuning.

**How it works:**

\`\`\`
Prompt structure:
1. Task description
2. Few examples (k=1 to 10 typically)
3. New query
4. Model generates answer

Example:
"Classify financial news sentiment.

News: Apple stock hits all-time high
Sentiment: Positive

News: Tesla recalls 500,000 vehicles
Sentiment: Negative

News: Microsoft revenue beats expectations
Sentiment: Positive

News: Amazon faces antitrust investigation
Sentiment: [MODEL GENERATES]"
\`\`\`

**Why it works:**1. **In-context learning**: GPT-3+ models learn patterns from prompt context
2. **Emergent ability**: Larger models (>10B parameters) show strong few-shot performance
3. **Pre-training**: Models saw similar tasks during pre-training
4. **Pattern matching**: Model identifies task structure from examples

**For Financial Sentiment Analysis:**

**Step 1: Design prompt template**

\`\`\`python
def create_few_shot_prompt (examples, query):
    prompt = "Analyze sentiment of financial news. Output: Positive, Negative, or Neutral.\\n\\n"
    
    # Add examples
    for ex in examples:
        prompt += f"News: {ex['text',]}\\n"
        prompt += f"Sentiment: {ex['label',]}\\n\\n"
    
    # Add query
    prompt += f"News: {query}\\n"
    prompt += "Sentiment:"
    
    return prompt
\`\`\`

**Step 2: Select good examples (critical!)**

\`\`\`python
# Manually curate diverse examples
examples = [
    # Clear positive
    {"text": "Apple reports record-breaking quarterly revenue of $100B", "label": "Positive"},
    
    # Clear negative
    {"text": "Company files for bankruptcy, stock drops 90%", "label": "Negative"},
    
    # Neutral
    {"text": "Corporation announces quarterly dividend payment", "label": "Neutral"},
    
    # Subtle positive (growth)
    {"text": "Firm\'s operating margin expands to 25% on cost efficiencies", "label": "Positive"},
    
    # Subtle negative (risk)
    {"text": "Regulatory investigation could impact future earnings", "label": "Negative"},
    
    # Mixed (include edge case)
    {"text": "Revenue up 10% but profit margins declined", "label": "Neutral"},
]
\`\`\`

**Step 3: Query the model**

\`\`\`python
from transformers import pipeline

# Use GPT-3, GPT-4, or open-source alternative (Llama, Mistral)
generator = pipeline('text-generation', model='meta-llama/Llama-2-13b-chat-hf')

def few_shot_sentiment (news_text, examples):
    prompt = create_few_shot_prompt (examples, news_text)
    
    response = generator(
        prompt,
        max_new_tokens=10,
        temperature=0.1,  # Low temperature for consistent outputs
        do_sample=True,
    )
    
    # Extract sentiment (text after "Sentiment:")
    output = response[0]['generated_text',]
    sentiment = output.split("Sentiment:")[-1].strip().split()[0]
    
    return sentiment

# Test
news = "Federal Reserve raises interest rates by 0.5%, markets tumble"
sentiment = few_shot_sentiment (news, examples)
print(f"Sentiment: {sentiment}")  # Expected: Negative
\`\`\`

**Advanced: Dynamic example selection**

\`\`\`python
from sentence_transformers import SentenceTransformer
import numpy as np

class DynamicFewShotClassifier:
    def __init__(self, example_pool):
        self.examples = example_pool
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.example_embeddings = self.encoder.encode([ex['text',] for ex in example_pool])
        
    def select_examples (self, query, k=5):
        ''Select most relevant examples for query''
        query_embedding = self.encoder.encode([query])
        
        # Compute similarity
        similarities = np.dot (query_embedding, self.example_embeddings.T)[0]
        
        # Get top k most similar
        top_indices = np.argsort (similarities)[::-1][:k]
        
        # Ensure diversity of labels
        selected = []
        labels_seen = set()
        
        for idx in top_indices:
            ex = self.examples[idx]
            if ex['label',] not in labels_seen or len (selected) < k:
                selected.append (ex)
                labels_seen.add (ex['label',])
            
            if len (selected) == k:
                break
        
        return selected
    
    def classify (self, query):
        # Select relevant examples
        examples = self.select_examples (query, k=5)
        
        # Few-shot inference
        sentiment = few_shot_sentiment (query, examples)
        return sentiment

# Usage
classifier = DynamicFewShotClassifier (large_example_pool)
sentiment = classifier.classify("Tech giant announces massive layoffs")
\`\`\`

**Advantages of Few-Shot for Financial Sentiment:**1. **No labeled data needed**: Just need ~5-10 good examples
2. **Fast deployment**: No training/fine-tuning required
3. **Flexible**: Can easily change task definition or add classes
4. **Interpretable**: Can see examples model is using
5. **Domain adaptation**: Include finance-specific examples

**Limitations:**1. **API costs**: GPT-3/GPT-4 API calls can be expensive at scale
2. **Latency**: Longer prompts = slower inference
3. **Less accurate than fine-tuning**: Fine-tuned models usually better
4. **Context length limits**: GPT-3 has 4K token limit
5. **Inconsistent**: May give different answers with temperature > 0

**When to use Few-Shot vs Fine-Tuning:**

**Use Few-Shot when:**
- Have < 100 labeled examples
- Need rapid prototyping/deployment
- Task definition changes frequently
- Have access to good LLM API (GPT-4)
- Can tolerate slightly lower accuracy

**Use Fine-Tuning when:**
- Have > 1000 labeled examples
- Need maximum accuracy
- High-volume production (API costs)
- Latency critical
- Need model ownership/control

**Hybrid Approach:**

\`\`\`python
# Start with few-shot for rapid deployment
initial_classifier = FewShotClassifier()

# Use it to label data automatically
unlabeled_news = fetch_financial_news()
pseudo_labels = [initial_classifier.classify (news) for news in unlabeled_news]

# Human verification of samples
verified_labels = human_verify (unlabeled_news, pseudo_labels, sample_size=500)

# Fine-tune FinBERT on verified data
finbert = fine_tune_finbert (verified_labels)

# Production: fine-tuned model (faster, cheaper, more accurate)
\`\`\`

**Best Practices:**1. **Example quality > quantity**: 5 great examples > 20 mediocre ones
2. **Include edge cases**: Ambiguous, mixed sentiment
3. **Finance-specific**: Use financial terminology
4. **Consistent format**: Keep examples structurally consistent
5. **Test temperature**: Lower (0.1-0.3) for classification tasks
6. **Validation**: Test on holdout set to measure accuracy

**Financial-Specific Considerations:**

- **Regulatory language**: Often neutral but important
- **Forward-looking**: Guidance statements (mixed sentiment)
- **Numbers matter**: "$100B revenue" vs "$100M loss"
- **Context critical**: "rate hike" negative for stocks, positive for banks

Few-shot learning democratizes NLP by enabling deployment without large labeled datasets, ideal for specialized financial applications.`,
    keyPoints: [
      'Few-shot: provide examples in prompt, no training needed',
      'Works via in-context learning in large language models (GPT-3+)',
      'Financial sentiment: curate 5-10 diverse, high-quality examples',
      'Dynamic example selection improves accuracy',
      'Advantages: no labeled data, fast deployment, flexible',
      'Limitations: API costs, latency, less accurate than fine-tuning',
      'Use few-shot for prototyping, fine-tuning for production',
    ],
  },
  {
    id: 'advanced-nlp-dq-3',
    question:
      'Design an NLP system to generate trading signals from real-time financial news. Discuss the pipeline, challenges, and risk management.',
    sampleAnswer: `**System Architecture:**

\`\`\`
News Sources → Preprocessing → Entity Extraction → Sentiment Analysis → 
Signal Generation → Risk Management → Trading Execution
\`\`\`

**Component 1: News Ingestion**

\`\`\`python
import asyncio
from newsapi import NewsApiClient

class NewsAggregator:
    def __init__(self):
        self.sources = [
            NewsApiClient (api_key='...'),  # NewsAPI
            # Bloomberg Terminal API
            # Reuters API
            # Twitter API (now X)
            # SEC EDGAR for filings
        ]
        self.news_buffer = []
        
    async def stream_news (self):
        ''Real-time news streaming''
        while True:
            for source in self.sources:
                news = await source.fetch_latest()
                
                for article in news:
                    # Deduplicate
                    if not self.is_duplicate (article):
                        self.news_buffer.append({
                            'text': article['title',] + ' ' + article['content',],
                            'source': article['source',],
                            'timestamp': article['published_at',],
                            'url': article['url',]
                        })
                        
            await asyncio.sleep(10)  # Poll every 10 seconds
    
    def is_duplicate (self, article):
        ''Check for duplicate/similar articles''
        # Use embeddings to detect near-duplicates
        # Often same story from multiple sources
        pass
\`\`\`

**Component 2: Entity Extraction & Linking**

\`\`\`python
class FinancialEntityExtractor:
    def __init__(self):
        self.ner_model = pipeline('ner', model='dslim/bert-base-NER')
        self.ticker_map = load_ticker_mapping()  # Company name → ticker
        
    def extract_entities (self, text):
        ''Extract companies mentioned''
        entities = self.ner_model (text)
        
        companies = [e['word',] for e in entities if e['entity_group',] == 'ORG',]
        
        # Link to tickers
        tickers = []
        for company in companies:
            ticker = self.ticker_map.get (company.lower())
            if ticker:
                tickers.append (ticker)
        
        return {
            'companies': companies,
            'tickers': tickers,
            'people': [e['word',] for e in entities if e['entity_group',] == 'PER',],
            'locations': [e['word',] for e in entities if e['entity_group',] == 'LOC',]
        }
\`\`\`

**Component 3: Multi-Model Sentiment Analysis**

\`\`\`python
class EnsembleSentimentAnalyzer:
    def __init__(self):
        # Multiple models for robustness
        self.models = [
            pipeline('sentiment-analysis', model='ProsusAI/finbert'),
            pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment'),
            # Custom fine-tuned model
        ]
        
    def analyze (self, text, ticker=None):
        ''Ensemble sentiment analysis''
        sentiments = []
        
        for model in self.models:
            result = model (text[:512])[0]
            score = result['score',] if result['label',] == 'positive' else -result['score',]
            sentiments.append (score)
        
        # Weighted average (FinBERT gets higher weight)
        weights = [0.5, 0.3, 0.2]
        weighted_sentiment = sum([s * w for s, w in zip (sentiments, weights)])
        
        # Adjust by news source credibility
        source_multiplier = self.get_source_credibility (text)
        
        final_sentiment = weighted_sentiment * source_multiplier
        
        return {
            'sentiment': final_sentiment,
            'confidence': np.std (sentiments),  # Lower std = higher agreement
            'individual_scores': sentiments
        }
\`\`\`

**Component 4: Signal Generation**

\`\`\`python
class TradingSignalGenerator:
    def __init__(self):
        self.sentiment_analyzer = EnsembleSentimentAnalyzer()
        self.entity_extractor = FinancialEntityExtractor()
        self.signal_history = {}  # Track signals over time
        
    def process_news (self, news_article):
        ''Convert news to trading signals''
        # Extract entities
        entities = self.entity_extractor.extract_entities (news_article['text',])
        
        if not entities['tickers',]:
            return None
        
        # Analyze sentiment
        sentiment_result = self.sentiment_analyzer.analyze(
            news_article['text',],
            ticker=entities['tickers',][0]
        )
        
        # Generate signals for each ticker
        signals = []
        for ticker in entities['tickers',]:
            # Calculate metrics
            sentiment_score = sentiment_result['sentiment',]
            momentum = self.calculate_momentum (ticker)
            volume = self.get_news_volume (ticker, hours=1)
            volatility = self.get_market_volatility (ticker)
            
            # Signal strength
            signal_strength = self.calculate_signal_strength(
                sentiment_score,
                momentum,
                volume,
                sentiment_result['confidence',]
            )
            
            signal = {
                'ticker': ticker,
                'timestamp': news_article['timestamp',],
                'action': self.determine_action (signal_strength),
                'confidence': sentiment_result['confidence',],
                'sentiment': sentiment_score,
                'magnitude': abs (signal_strength),
                'source_news': news_article['text',][:200],
                'reasoning': self.explain_signal (sentiment_score, momentum, volume)
            }
            
            signals.append (signal)
        
        return signals
    
    def calculate_signal_strength (self, sentiment, momentum, volume, confidence):
        ''Combine multiple factors''
        # Weighted combination
        signal = (
            0.4 * sentiment +           # Current sentiment
            0.3 * momentum +            # Trend
            0.2 * np.log1p (volume) +   # News volume (log scale)
            0.1 * confidence           # Model agreement
        )
        
        return signal
    
    def determine_action (self, signal_strength, threshold=0.3):
        ''Convert signal to action''
        if signal_strength > threshold:
            return 'BUY'
        elif signal_strength < -threshold:
            return 'SELL'
        else:
            return 'HOLD'
    
    def calculate_momentum (self, ticker, hours=24):
        ''Sentiment momentum over time''
        recent_signals = self.get_recent_signals (ticker, hours=hours)
        
        if len (recent_signals) < 2:
            return 0
        
        # Compare recent vs earlier
        mid = len (recent_signals) // 2
        recent_avg = np.mean([s['sentiment',] for s in recent_signals[mid:]])
        earlier_avg = np.mean([s['sentiment',] for s in recent_signals[:mid]])
        
        return recent_avg - earlier_avg
\`\`\`

**Component 5: Risk Management**

\`\`\`python
class RiskManager:
    def __init__(self):
        self.max_position_size = 0.05  # Max 5% of portfolio per position
        self.max_daily_loss = 0.02     # Max 2% daily loss
        self.signal_filters = []
        
    def validate_signal (self, signal, portfolio, market_data):
        ''Apply risk checks before execution''
        checks = {
            'confidence_check': signal['confidence',] > 0.6,
            'volatility_check': market_data['volatility',] < 0.5,
            'position_size_check': self.check_position_size (signal, portfolio),
            'daily_loss_check': portfolio['daily_pnl',] > -self.max_daily_loss,
            'market_hours_check': self.is_market_open(),
            'liquidity_check': market_data['volume',] > 1000000,
        }
        
        # All checks must pass
        if all (checks.values()):
            return True, "Signal approved"
        else:
            failed = [k for k, v in checks.items() if not v]
            return False, f"Failed checks: {failed}"
    
    def calculate_position_size (self, signal, portfolio):
        ''Kelly criterion for position sizing''
        # Simplified Kelly: f = (p*b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        
        # Estimate based on signal confidence and historical accuracy
        win_prob = self.estimate_win_probability (signal)
        win_loss_ratio = 1.5  # Historical average
        
        kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        
        # Use fractional Kelly (more conservative)
        position_size = max(0, min (kelly_fraction * 0.5, self.max_position_size))
        
        return position_size * portfolio['total_value',]
\`\`\`

**Component 6: Integration & Monitoring**

\`\`\`python
class NewsTrading System:
    def __init__(self):
        self.news_aggregator = NewsAggregator()
        self.signal_generator = TradingSignalGenerator()
        self.risk_manager = RiskManager()
        self.portfolio = Portfolio()
        self.metrics_tracker = MetricsTracker()
        
    async def run (self):
        ''Main trading loop''
        while True:
            # Get latest news
            news_batch = self.news_aggregator.get_latest()
            
            for news in news_batch:
                # Generate signals
                signals = self.signal_generator.process_news (news)
                
                if not signals:
                    continue
                
                for signal in signals:
                    # Risk management
                    approved, reason = self.risk_manager.validate_signal(
                        signal,
                        self.portfolio.get_state(),
                        market_data=self.get_market_data (signal['ticker',])
                    )
                    
                    if approved:
                        # Calculate position size
                        size = self.risk_manager.calculate_position_size(
                            signal,
                            self.portfolio.get_state()
                        )
                        
                        # Execute trade
                        self.execute_trade (signal['ticker',], signal['action',], size)
                        
                        # Log
                        self.log_trade (signal, size, reason)
                    else:
                        self.log_rejection (signal, reason)
            
            # Track performance
            self.metrics_tracker.update()
            
            await asyncio.sleep(5)
\`\`\`

**Challenges & Solutions:**

**1. Latency:**
- **Challenge**: News spreads fast, milliseconds matter
- **Solution**: Optimize pipeline, use caching, deploy close to exchanges

**2. Noise:**
- **Challenge**: Much financial news is irrelevant
- **Solution**: Source filtering, confidence thresholds, entity linking

**3. Market Impact:**
- **Challenge**: Large orders move prices
- **Solution**: Position sizing, VWAP execution, split orders

**4. False Positives:**
- **Challenge**: Sentiment doesn't always predict price movement
- **Solution**: Ensemble models, momentum filters, backtesting

**5. Adversarial News:**
- **Challenge**: Fake news, manipulation
- **Solution**: Source credibility scores, fact-checking, anomaly detection

**Performance Monitoring:**

\`\`\`python
metrics = {
    'sharpe_ratio': calculate_sharpe (returns),
    'max_drawdown': calculate_max_drawdown (portfolio_value),
    'win_rate': wins / total_trades,
    'average_holding_period': np.mean (holding_periods),
    'sentiment_accuracy': correct_predictions / total_predictions,
    'signal_latency': time_news_to_trade,
}
\`\`\`

**Key Takeaways:**
- Multi-stage pipeline with checks at each step
- Ensemble models for robustness
- Strict risk management (position sizing, stop losses)
- Monitor and adapt based on performance
- Never risk more than you can afford to lose`,
    keyPoints: [
      'Pipeline: news ingestion → entity extraction → sentiment → signal generation',
      'Entity linking critical: map company names to tickers',
      'Ensemble sentiment models for robustness',
      'Risk management: position sizing, confidence thresholds, volatility checks',
      'Challenges: latency, noise, false positives, adversarial news',
      'Monitor: Sharpe ratio, win rate, sentiment accuracy, signal latency',
    ],
  },
];
