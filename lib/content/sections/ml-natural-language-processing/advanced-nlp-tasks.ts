/**
 * Section: Advanced NLP Tasks
 * Module: Natural Language Processing
 *
 * Covers machine translation, summarization, text generation, dialogue systems,
 * few-shot learning, and financial NLP applications
 */

export const advancedNlpTasksSection = {
  id: 'advanced-nlp-tasks',
  title: 'Advanced NLP Tasks',
  content: `
# Advanced NLP Tasks

## Introduction

Advanced NLP encompasses complex tasks like translation, summarization, and text generation. We'll also explore financial NLP applications including SEC filings analysis, earnings call transcripts, and news sentiment for trading.

## Machine Translation

### Transformer-based Translation

\`\`\`python
from transformers import MarianMTModel, MarianTokenizer

# Load translation model (English to French)
model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained (model_name)
model = MarianMTModel.from_pretrained (model_name)

def translate (text):
    """Translate English to French"""
    inputs = tokenizer (text, return_tensors='pt', padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode (translated[0], skip_special_tokens=True)
    return translated_text

# Example
english = "Machine learning is revolutionizing technology"
french = translate (english)
print(f"EN: {english}")
print(f"FR: {french}")
# Output: L'apprentissage automatique révolutionne la technologie
\`\`\`

### Fine-tuning for Domain-specific Translation

\`\`\`python
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

# Load parallel corpus
dataset = load_dataset('wmt14', 'de-en')

def preprocess_function (examples):
    """Prepare source and target"""
    inputs = [ex['en'] for ex in examples['translation']]
    targets = [ex['de'] for ex in examples['translation']]
    
    model_inputs = tokenizer (inputs, max_length=128, truncation=True)
    
    # Setup target tokenization
    with tokenizer.as_target_tokenizer():
        labels = tokenizer (targets, max_length=128, truncation=True)
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = dataset.map (preprocess_function, batched=True)

# Data collator handles padding
data_collator = DataCollatorForSeq2Seq (tokenizer, model=model)

# Training
training_args = TrainingArguments(
    output_dir='./translation_model',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
)

trainer.train()
\`\`\`

### Evaluation: BLEU Score

\`\`\`python
from datasets import load_metric

bleu = load_metric('bleu')

predictions = ["the cat is on the mat"]
references = [["the cat is on the mat"]]

result = bleu.compute (predictions=predictions, references=references)
print(f"BLEU score: {result['bleu']:.4f}")

# Multiple references
predictions = ["the cat is on the mat"]
references = [[
    "the cat is on the mat",
    "there is a cat on the mat"
]]

result = bleu.compute (predictions=predictions, references=references)
print(f"BLEU score with multiple refs: {result['bleu']:.4f}")
\`\`\`

## Text Summarization

### Extractive Summarization

\`\`\`python
from transformers import pipeline

# Using extractive summarization
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

article = """
Artificial intelligence has made remarkable progress in recent years. Deep learning models
can now perform tasks that were once thought impossible for machines. Natural language
processing has enabled computers to understand and generate human language with impressive
accuracy. Computer vision systems can identify objects and faces better than humans in
many cases. These advances have led to practical applications in healthcare, finance,
transportation, and many other fields. However, challenges remain in making AI systems
more robust, interpretable, and aligned with human values.
"""

summary = summarizer (article, max_length=50, min_length=25, do_sample=False)
print(summary[0]['summary_text'])

# Output: Artificial intelligence has made remarkable progress in recent years.
# Deep learning models can now perform tasks that were once thought impossible for machines.
\`\`\`

### Abstractive Summarization with T5

\`\`\`python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def summarize (text, max_length=100):
    """Generate abstractive summary"""
    # T5 requires task prefix
    input_text = f"summarize: {text}"
    
    inputs = tokenizer.encode (input_text, return_tensors='pt', max_length=512, truncation=True)
    
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    summary = tokenizer.decode (summary_ids[0], skip_special_tokens=True)
    return summary

# Example
article = """
The Federal Reserve announced today that it will raise interest rates by 0.25 percentage
points to combat inflation. This marks the third rate increase this year. The Fed chair
stated that the economy remains strong but inflation is running above the target rate.
Markets reacted positively to the news, with major indices closing higher.
"""

summary = summarize (article)
print(summary)
# Output: fed raises interest rates by 0.25% to combat inflation. markets react positively.
\`\`\`

### Fine-tuning for Custom Summarization

\`\`\`python
from datasets import load_dataset

# Load summarization dataset (CNN/DailyMail)
dataset = load_dataset('cnn_dailymail', '3.0.0')

def preprocess_function (examples):
    """Prepare articles and summaries"""
    inputs = ["summarize: " + doc for doc in examples['article']]
    model_inputs = tokenizer (inputs, max_length=512, truncation=True, padding='max_length')
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer (examples['highlights'], max_length=128, truncation=True, padding='max_length')
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = dataset.map (preprocess_function, batched=True)

# Training
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir='./summarization_model',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
)

trainer.train()
\`\`\`

## Text Generation

### Controlled Generation with GPT

\`\`\`python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def generate_text (prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.95):
    """Generate text with various strategies"""
    inputs = tokenizer.encode (prompt, return_tensors='pt')
    
    # Generate with different strategies
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,      # Randomness (lower = more focused)
        top_k=top_k,                  # Sample from top k tokens
        top_p=top_p,                  # Nucleus sampling
        do_sample=True,
        num_return_sequences=3,       # Generate multiple
        pad_token_id=tokenizer.eos_token_id
    )
    
    texts = [tokenizer.decode (output, skip_special_tokens=True) for output in outputs]
    return texts

# Example
prompt = "Machine learning is"
generations = generate_text (prompt, max_length=50, temperature=0.7)

for i, text in enumerate (generations):
    print(f"\\nGeneration {i+1}:")
    print(text)
\`\`\`

### Few-shot Learning with GPT

\`\`\`python
def few_shot_generation (task_description, examples, query):
    """Few-shot learning via prompting"""
    # Build prompt with examples
    prompt = f"{task_description}\\n\\n"
    
    for example in examples:
        prompt += f"Input: {example['input']}\\n"
        prompt += f"Output: {example['output']}\\n\\n"
    
    prompt += f"Input: {query}\\n"
    prompt += "Output:"
    
    # Generate
    inputs = tokenizer.encode (prompt, return_tensors='pt')
    outputs = model.generate (inputs, max_length=len (inputs[0]) + 50)
    result = tokenizer.decode (outputs[0], skip_special_tokens=True)
    
    # Extract just the answer
    answer = result.split("Output:")[-1].strip()
    return answer

# Example: Sentiment classification via few-shot
task = "Classify the sentiment of movie reviews."
examples = [
    {"input": "This movie was fantastic!", "output": "Positive"},
    {"input": "Waste of time and money.", "output": "Negative"},
    {"input": "It was okay, nothing special.", "output": "Neutral"},
]

query = "Best film I've seen this year!"
result = few_shot_generation (task, examples, query)
print(f"Sentiment: {result}")
\`\`\`

## Dialogue Systems

### Conversational AI

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load conversational model
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')

class ChatBot:
    def __init__(self):
        self.chat_history_ids = None
        
    def respond (self, user_input):
        """Generate response maintaining context"""
        # Encode user input
        new_input_ids = tokenizer.encode (user_input + tokenizer.eos_token, return_tensors='pt')
        
        # Append to chat history
        bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1) if self.chat_history_ids is not None else new_input_ids
        
        # Generate response
        self.chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
        )
        
        # Decode response
        response = tokenizer.decode (self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response
    
    def reset (self):
        """Clear chat history"""
        self.chat_history_ids = None

# Example conversation
chatbot = ChatBot()

print("User: Hello!")
print(f"Bot: {chatbot.respond('Hello!')}")

print("\\nUser: What\'s the weather like?")
print(f"Bot: {chatbot.respond('What's the weather like?')}")

print("\\nUser: Thanks!")
print(f"Bot: {chatbot.respond('Thanks!')}")
\`\`\`

## Financial NLP Applications

### Analyzing SEC Filings (10-K, 10-Q)

\`\`\`python
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

def fetch_sec_filing (ticker, form_type='10-K'):
    """Fetch SEC filing from EDGAR"""
    # This is simplified - use sec-api or edgar in practice
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type={form_type}&count=1"
    # ... fetch and parse filing
    return filing_text

def analyze_risk_factors (filing_text):
    """Extract and analyze risk factors section"""
    # Extract risk section (usually "Item 1A. Risk Factors")
    # Simplified example
    risk_section = extract_section (filing_text, "Risk Factors")
    
    # Sentiment analysis
    sentiment_analyzer = pipeline('sentiment-analysis', model='ProsusAI/finbert')
    
    # Split into chunks
    chunks = [risk_section[i:i+512] for i in range(0, len (risk_section), 512)]
    
    sentiments = []
    for chunk in chunks:
        result = sentiment_analyzer (chunk)[0]
        sentiments.append (result)
    
    # Aggregate
    avg_sentiment = sum([s['score'] if s['label'] == 'positive' else -s['score'] for s in sentiments]) / len (sentiments)
    
    return {
        'overall_sentiment': avg_sentiment,
        'sentiments': sentiments
    }

# Example
filing = fetch_sec_filing('AAPL', '10-K')
risks = analyze_risk_factors (filing)
print(f"Risk sentiment: {risks['overall_sentiment']:.3f}")
\`\`\`

### Earnings Call Transcript Analysis

\`\`\`python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Financial sentiment model (FinBERT)
finbert = pipeline('sentiment-analysis', model='ProsusAI/finbert')

def analyze_earnings_call (transcript):
    """Analyze earnings call transcript"""
    
    # 1. Sentiment analysis
    paragraphs = transcript.split('\\n\\n')
    sentiments = []
    
    for para in paragraphs:
        if len (para.strip()) < 50:
            continue
        
        sentiment = finbert (para[:512])[0]  # FinBERT max length
        sentiments.append({
            'text': para[:100] + '...',
            'label': sentiment['label'],
            'score': sentiment['score']
        })
    
    # 2. Extract key financial metrics mentioned
    financial_terms = ['revenue', 'earnings', 'profit', 'guidance', 'growth', 'margin']
    
    key_mentions = []
    for para in paragraphs:
        for term in financial_terms:
            if term in para.lower():
                # Extract sentence with term
                sentences = para.split('.')
                for sent in sentences:
                    if term in sent.lower():
                        key_mentions.append({
                            'term': term,
                            'context': sent.strip()
                        })
    
    # 3. Q&A sentiment (often more revealing)
    qa_section = extract_qa_section (transcript)
    qa_sentiment = finbert (qa_section[:512])[0] if qa_section else None
    
    # 4. Management tone analysis
    management_paragraphs = extract_management_discussion (transcript)
    management_sentiment = [finbert (p[:512])[0] for p in management_paragraphs]
    
    return {
        'overall_sentiments': sentiments,
        'key_financial_mentions': key_mentions,
        'qa_sentiment': qa_sentiment,
        'management_tone': management_sentiment,
        'overall_score': sum([s['score'] if s['label'] == 'positive' else -s['score'] 
                             for s in sentiments]) / len (sentiments)
    }

# Example
transcript = """
Thank you for joining our Q3 earnings call. We're pleased to report strong revenue
growth of 15% year-over-year, exceeding our guidance. Our operating margins expanded
to 25%, driven by operational efficiencies and favorable product mix.

Looking ahead, we're raising our full-year guidance to $5.2-5.4 billion in revenue,
up from our previous range of $5.0-5.2 billion. We remain confident in our growth
trajectory and market position.
"""

analysis = analyze_earnings_call (transcript)
print(f"Overall sentiment score: {analysis['overall_score']:.3f}")
print(f"\\nKey mentions:")
for mention in analysis['key_financial_mentions'][:5]:
    print(f"  - {mention['term']}: {mention['context']}")
\`\`\`

### News Sentiment for Trading Signals

\`\`\`python
import pandas as pd
from datetime import datetime, timedelta

class NewsSentimentTrader:
    """Trading signals from news sentiment"""
    
    def __init__(self):
        self.sentiment_analyzer = pipeline('sentiment-analysis', model='ProsusAI/finbert')
        
    def fetch_news (self, ticker, days=1):
        """Fetch news articles (use news API in practice)"""
        # Simplified - use NewsAPI, Bloomberg, etc. in practice
        articles = []
        # ... fetch news
        return articles
    
    def analyze_sentiment_batch (self, articles):
        """Analyze sentiment of multiple articles"""
        sentiments = []
        
        for article in articles:
            # Analyze headline and body
            headline_sentiment = self.sentiment_analyzer (article['headline'])[0]
            body_sentiment = self.sentiment_analyzer (article['body'][:512])[0]
            
            # Weight headline higher (more impact)
            combined_score = (
                0.6 * (headline_sentiment['score'] if headline_sentiment['label'] == 'positive' else -headline_sentiment['score']) +
                0.4 * (body_sentiment['score'] if body_sentiment['label'] == 'positive' else -body_sentiment['score'])
            )
            
            sentiments.append({
                'timestamp': article['timestamp'],
                'headline': article['headline'],
                'sentiment': combined_score,
                'headline_label': headline_sentiment['label'],
                'body_label': body_sentiment['label']
            })
        
        return sentiments
    
    def generate_signal (self, ticker, lookback_hours=24):
        """Generate trading signal from recent news sentiment"""
        # Fetch recent news
        articles = self.fetch_news (ticker, days=1)
        
        # Analyze
        sentiments = self.analyze_sentiment_batch (articles)
        
        # Calculate metrics
        avg_sentiment = sum([s['sentiment'] for s in sentiments]) / len (sentiments) if sentiments else 0
        sentiment_momentum = self.calculate_momentum (sentiments)
        volume = len (sentiments)  # News volume can be a signal
        
        # Generate signal
        signal = {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'average_sentiment': avg_sentiment,
            'momentum': sentiment_momentum,
            'news_volume': volume,
            'recommendation': self.make_recommendation (avg_sentiment, sentiment_momentum, volume)
        }
        
        return signal
    
    def calculate_momentum (self, sentiments):
        """Calculate sentiment momentum (trend)"""
        if len (sentiments) < 2:
            return 0
        
        # Sort by time
        sorted_sentiments = sorted (sentiments, key=lambda x: x['timestamp'])
        
        # Recent vs earlier sentiment
        mid_point = len (sorted_sentiments) // 2
        recent_avg = sum([s['sentiment'] for s in sorted_sentiments[mid_point:]]) / (len (sorted_sentiments) - mid_point)
        earlier_avg = sum([s['sentiment'] for s in sorted_sentiments[:mid_point]]) / mid_point
        
        momentum = recent_avg - earlier_avg
        return momentum
    
    def make_recommendation (self, sentiment, momentum, volume):
        """Generate buy/sell/hold recommendation"""
        # Simple rule-based strategy
        if sentiment > 0.3 and momentum > 0.1 and volume > 5:
            return "BUY"
        elif sentiment < -0.3 and momentum < -0.1 and volume > 5:
            return "SELL"
        else:
            return "HOLD"

# Example usage
trader = NewsSentimentTrader()
signal = trader.generate_signal('AAPL')

print(f"Ticker: {signal['ticker']}")
print(f"Sentiment: {signal['average_sentiment']:.3f}")
print(f"Momentum: {signal['momentum']:.3f}")
print(f"News Volume: {signal['news_volume']}")
print(f"Recommendation: {signal['recommendation']}")
\`\`\`

### Named Entity Recognition for Financial Documents

\`\`\`python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Financial NER model
ner_model = pipeline('ner', model='dslim/bert-base-NER-uncased', aggregation_strategy='simple')

def extract_financial_entities (text):
    """Extract companies, people, locations from financial text"""
    entities = ner_model (text)
    
    # Organize by type
    organized = {
        'companies': [],
        'people': [],
        'locations': [],
        'other': []
    }
    
    for entity in entities:
        if entity['entity_group'] == 'ORG':
            organized['companies'].append (entity['word'])
        elif entity['entity_group'] == 'PER':
            organized['people'].append (entity['word'])
        elif entity['entity_group'] == 'LOC':
            organized['locations'].append (entity['word'])
        else:
            organized['other'].append (entity['word'])
    
    return organized

# Example
text = """
Apple Inc. CEO Tim Cook announced a new partnership with Microsoft Corporation.
The deal, valued at $10 billion, will expand Apple\'s presence in Europe and Asia.
The announcement was made at Apple's headquarters in Cupertino, California.
"""

entities = extract_financial_entities (text)
print("Companies:", entities['companies'])
print("People:", entities['people'])
print("Locations:", entities['locations'])
\`\`\`

### ESG Sentiment Analysis

\`\`\`python
def analyze_esg_factors (company_report):
    """Analyze Environmental, Social, Governance factors from reports"""
    
    # Define ESG keywords
    esg_keywords = {
        'environmental': ['carbon', 'emissions', 'renewable', 'sustainability', 'climate'],
        'social': ['diversity', 'inclusion', 'community', 'safety', 'labor'],
        'governance': ['board', 'compliance', 'ethics', 'transparency', 'audit']
    }
    
    sentiment_analyzer = pipeline('sentiment-analysis', model='ProsusAI/finbert')
    
    esg_scores = {}
    
    for category, keywords in esg_keywords.items():
        # Extract paragraphs mentioning keywords
        relevant_paragraphs = []
        for para in company_report.split('\\n\\n'):
            if any (keyword in para.lower() for keyword in keywords):
                relevant_paragraphs.append (para)
        
        # Analyze sentiment
        if relevant_paragraphs:
            sentiments = [sentiment_analyzer (p[:512])[0] for p in relevant_paragraphs]
            avg_score = sum([s['score'] if s['label'] == 'positive' else -s['score'] 
                           for s in sentiments]) / len (sentiments)
            
            esg_scores[category] = {
                'score': avg_score,
                'mentions': len (relevant_paragraphs),
                'examples': [p[:100] + '...' for p in relevant_paragraphs[:3]]
            }
        else:
            esg_scores[category] = {
                'score': 0,
                'mentions': 0,
                'examples': []
            }
    
    # Overall ESG score
    overall = sum([v['score'] for v in esg_scores.values()]) / 3
    
    return {
        'overall_esg_score': overall,
        'category_scores': esg_scores
    }

# Example
report = """
Our company is committed to sustainability. We reduced carbon emissions by 25% this year
through renewable energy adoption. Our solar installations now power 60% of operations.

We prioritize diversity and inclusion, with 45% of our workforce from underrepresented
groups. Our board of directors reflects diverse backgrounds and perspectives.

Corporate governance remains a top priority. We maintain transparent reporting practices
and have strengthened our audit committee.
"""

esg_analysis = analyze_esg_factors (report)
print(f"Overall ESG Score: {esg_analysis['overall_esg_score']:.3f}")
for category, data in esg_analysis['category_scores'].items():
    print(f"\\n{category.upper()}: {data['score']:.3f} ({data['mentions']} mentions)")
\`\`\`

## Evaluation Metrics for Generation Tasks

\`\`\`python
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# ROUGE for summarization
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

generated = "The cat sat on the mat."
reference = "A cat was sitting on the mat."

scores = scorer.score (reference, generated)
print("ROUGE scores:")
for key, value in scores.items():
    print(f"{key}: Precision={value.precision:.3f}, Recall={value.recall:.3f}, F1={value.fmeasure:.3f}")

# BERTScore for semantic similarity
P, R, F1 = bert_score([generated], [reference], lang='en')
print(f"\\nBERTScore F1: {F1.item():.3f}")
\`\`\`

## Summary

Advanced NLP tasks covered:
- **Machine Translation**: Seq2seq models, BLEU scores
- **Summarization**: Extractive vs abstractive, ROUGE metrics
- **Text Generation**: GPT, controlled generation, few-shot learning
- **Dialogue Systems**: Conversational AI, context maintenance
- **Financial NLP**: SEC filings, earnings calls, news sentiment, ESG analysis

These advanced techniques power real-world applications from chatbots to trading systems.
`,
};
