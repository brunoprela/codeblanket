/**
 * Few-Shot Learning & Examples Section
 * Module 2: Prompt Engineering & Optimization
 */

export const fewshotlearningexamplesSection = {
  id: 'few-shot-learning-examples',
  title: 'Few-Shot Learning & Examples',
  content: `# Few-Shot Learning & Examples

Master the art of teaching LLMs through examples to achieve reliable, consistent outputs in production.

## Overview: The Power of Examples

Few-shot learning is one of the most powerful techniques in prompt engineering. Instead of just telling the AI what to do, you **show** it with examples.

### Why Examples Work

**Human learning**: "Show me how, then I'll do it"
**Few-shot learning**: Same concept for AI

\`\`\`python
# Without examples (zero-shot)
zero_shot = "Extract entities from: 'Apple released iPhone 15 in California'"
# Unpredictable: might extract "Apple", "iPhone", "California" or format differently

# With examples (few-shot)
few_shot = """
Extract entities in format: {{"company": [], "product": [], "location": []}}

Examples:
Text: "Google launched Pixel in Mountain View"
Output: {{"company": ["Google"], "product": ["Pixel"], "location": ["Mountain View"]}}

Text: "Tesla unveiled Cybertruck in Austin, Texas"  
Output: {{"company": ["Tesla"], "product": ["Cybertruck"], "location": ["Austin", "Texas"]}}

Text: "Apple released iPhone 15 in California"
Output:"""
# Much more reliable and consistent!
\`\`\`

## Few-Shot Learning Fundamentals

### What is Few-Shot Learning?

**Few-shot learning** is providing the model with a small number of input-output examples to demonstrate the desired behavior before asking it to perform a new task.

\`\`\`python
from openai import OpenAI

client = OpenAI()

def zero_shot_example():
    """Task with no examples."""
    prompt = """Classify the sentiment: "The movie was okay, not great but watchable."
    
Sentiment:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def few_shot_example():
    """Same task with examples."""
    prompt = """Classify the sentiment as positive, negative, or neutral.

Examples:

Text: "Absolutely loved it! Best experience ever!"
Sentiment: positive

Text: "Terrible service. Complete waste of time."
Sentiment: negative

Text: "It was fine. Nothing special but okay."
Sentiment: neutral

Now classify:
Text: "The movie was okay, not great but watchable."
Sentiment:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

# Compare outputs
print("Zero-shot:", zero_shot_example())
print("Few-shot:", few_shot_example())

# Few-shot will be more consistent with format and classification
\`\`\`

### How Many Shots (Examples)?

\`\`\`python
def optimal_example_count(task_complexity: str, model_capability: str) -> int:
    """
    Determine optimal number of examples.
    
    General guidelines:
    - Simple tasks: 1-3 examples
    - Medium complexity: 3-5 examples  
    - Complex tasks: 5-10 examples
    - Very complex: 10-20 examples (rarely more)
    
    More examples = more tokens = higher cost
    """
    
    complexity_map = {
        ('simple', 'high'): 1,      # Strong model, easy task
        ('simple', 'medium'): 2,
        ('medium', 'high'): 3,
        ('medium', 'medium'): 5,
        ('complex', 'high'): 5,
        ('complex', 'medium'): 8,
        ('very_complex', 'high'): 10,
        ('very_complex', 'medium'): 15,
    }
    
    return complexity_map.get((task_complexity, model_capability), 5)

# Examples
print(optimal_example_count('simple', 'high'))      # 1 example enough
print(optimal_example_count('complex', 'medium'))   # 8 examples better
\`\`\`

## Choosing Representative Examples

### Example Selection Principles

\`\`\`python
from typing import List, Dict
import random

class ExampleSelector:
    """
    Select the best examples for few-shot learning.
    Critical for consistent performance.
    """
    
    def __init__(self, example_pool: List[Dict]):
        """
        Args:
            example_pool: List of dicts with 'input', 'output', 'metadata'
        """
        self.example_pool = example_pool
    
    def select_diverse_examples(self, n: int = 3) -> List[Dict]:
        """
        Select diverse examples covering different patterns.
        
        Diversity criteria:
        - Different input lengths
        - Different patterns/edge cases
        - Different output types
        - Representative of common cases
        """
        
        if n >= len(self.example_pool):
            return self.example_pool
        
        selected = []
        
        # 1. Include shortest example
        shortest = min(self.example_pool, key=lambda x: len(x['input']))
        selected.append(shortest)
        
        # 2. Include longest example  
        longest = max(self.example_pool, key=lambda x: len(x['input']))
        if longest != shortest:
            selected.append(longest)
        
        # 3. Fill remaining with random diverse examples
        remaining = [e for e in self.example_pool if e not in selected]
        remaining_needed = n - len(selected)
        
        if remaining_needed > 0:
            selected.extend(random.sample(remaining, min(remaining_needed, len(remaining))))
        
        return selected[:n]
    
    def select_similar_examples(
        self,
        query: str,
        n: int = 3,
        similarity_func = None
    ) -> List[Dict]:
        """
        Select examples most similar to the current query.
        Dynamic few-shot learning.
        """
        
        if similarity_func is None:
            # Simple similarity: shared words
            def default_similarity(a: str, b: str) -> float:
                words_a = set(a.lower().split())
                words_b = set(b.lower().split())
                intersection = words_a & words_b
                union = words_a | words_b
                return len(intersection) / len(union) if union else 0
            
            similarity_func = default_similarity
        
        # Score each example
        scored = [
            (example, similarity_func(query, example['input']))
            for example in self.example_pool
        ]
        
        # Sort by similarity and take top n
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [example for example, score in scored[:n]]
    
    def select_by_difficulty(self, difficulty: str, n: int = 3) -> List[Dict]:
        """
        Select examples by difficulty level.
        Start easy, show progressively harder examples.
        """
        
        # Filter by difficulty if metadata exists
        filtered = [
            ex for ex in self.example_pool
            if ex.get('metadata', {}).get('difficulty') == difficulty
        ]
        
        if len(filtered) >= n:
            return random.sample(filtered, n)
        
        return filtered + random.sample(
            [e for e in self.example_pool if e not in filtered],
            n - len(filtered)
        )

# Usage example
examples_pool = [
    {
        'input': 'hi',
        'output': 'Hello! How can I help?',
        'metadata': {'difficulty': 'easy', 'type': 'greeting'}
    },
    {
        'input': 'What is machine learning?',
        'output': 'Machine learning is a subset of AI...',
        'metadata': {'difficulty': 'medium', 'type': 'definition'}
    },
    {
        'input': 'Explain backpropagation in neural networks with calculus',
        'output': 'Backpropagation uses the chain rule...',
        'metadata': {'difficulty': 'hard', 'type': 'technical'}
    },
    {
        'input': 'How are you?',
        'output': 'I\'m doing well, thanks for asking!',
        'metadata': {'difficulty': 'easy', 'type': 'greeting'}
    }
]

selector = ExampleSelector(examples_pool)

# Get diverse examples
diverse = selector.select_diverse_examples(n=2)
print("Diverse examples:", [e['input'] for e in diverse])

# Get similar examples for a query
similar = selector.select_similar_examples("What is AI?", n=2)
print("Similar examples:", [e['input'] for e in similar])
\`\`\`

## Example Ordering and Placement

### Order Matters!

\`\`\`python
def test_example_ordering():
    """
    Demonstrate that example order affects output quality.
    Best practice: simple → complex
    """
    
    from openai import OpenAI
    client = OpenAI()
    
    # Same examples, different orders
    examples = [
        ("2 + 2", "4"),
        ("15 + 27", "42"),
        ("123 + 456", "579"),
    ]
    
    # Order 1: Simple to complex (recommended)
    simple_to_complex = """
Examples:
2 + 2 = 4
15 + 27 = 42
123 + 456 = 579

Now solve: 89 + 76 ="""
    
    # Order 2: Complex to simple (not recommended)
    complex_to_simple = """
Examples:
123 + 456 = 579
15 + 27 = 42
2 + 2 = 4

Now solve: 89 + 76 ="""
    
    # Order 3: Random (worst)
    random_order = """
Examples:
15 + 27 = 42
2 + 2 = 4
123 + 456 = 579

Now solve: 89 + 76 ="""
    
    # Simple to complex typically performs best
    # Model learns pattern progression
    
    return simple_to_complex

class ExampleOrganizer:
    """
    Organize examples for optimal few-shot learning.
    """
    
    @staticmethod
    def order_by_complexity(examples: List[Dict], complexity_key: str = 'length') -> List[Dict]:
        """
        Order examples from simple to complex.
        
        Args:
            examples: List of example dicts
            complexity_key: How to measure complexity ('length', 'difficulty', etc.)
        """
        
        if complexity_key == 'length':
            return sorted(examples, key=lambda x: len(x['input']))
        elif complexity_key == 'difficulty':
            difficulty_order = {'easy': 1, 'medium': 2, 'hard': 3}
            return sorted(
                examples,
                key=lambda x: difficulty_order.get(
                    x.get('metadata', {}).get('difficulty', 'medium'),
                    2
                )
            )
        else:
            return examples
    
    @staticmethod
    def order_by_recency(examples: List[Dict]) -> List[Dict]:
        """
        Most recent examples first (for evolving tasks).
        """
        return sorted(
            examples,
            key=lambda x: x.get('metadata', {}).get('timestamp', 0),
            reverse=True
        )
    
    @staticmethod
    def order_by_success_rate(examples: List[Dict]) -> List[Dict]:
        """
        Put examples that historically led to best results first.
        """
        return sorted(
            examples,
            key=lambda x: x.get('metadata', {}).get('success_rate', 0.5),
            reverse=True
        )

# Example usage
examples = [
    {'input': 'Complex multi-step problem with edge cases...', 'output': '...'},
    {'input': 'Simple hello world', 'output': 'Hello!'},
    {'input': 'Medium difficulty task', 'output': '...'}
]

organizer = ExampleOrganizer()
ordered = organizer.order_by_complexity(examples)
print("Ordered examples:", [e['input'][:30] for e in ordered])
\`\`\`

### Placement Strategies

\`\`\`python
class ExamplePlacementStrategy:
    """Different ways to place examples in prompts."""
    
    @staticmethod
    def prefix_examples(task: str, examples: List[Dict], query: str) -> str:
        """
        Examples before task (most common).
        """
        examples_text = "\\n\\n".join([
            f"Input: {ex['input']}\\nOutput: {ex['output']}"
            for ex in examples
        ])
        
        return f"""{task}

Examples:
{examples_text}

Now complete:
Input: {query}
Output:"""
    
    @staticmethod
    def sandwich_examples(task: str, examples: List[Dict], query: str) -> str:
        """
        Task - Examples - Task pattern (very effective).
        """
        examples_text = "\\n\\n".join([
            f"Input: {ex['input']}\\nOutput: {ex['output']}"
            for ex in examples
        ])
        
        return f"""{task}

Here are some examples:
{examples_text}

Now apply the same pattern to:
Input: {query}
Output:"""
    
    @staticmethod
    def inline_examples(task: str, examples: List[Dict], query: str) -> str:
        """
        Examples interwoven with instructions.
        """
        prompt = f"{task}\\n\\n"
        
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\\n"
            prompt += f"  Input: {ex['input']}\\n"
            prompt += f"  Output: {ex['output']}\\n"
            prompt += f"  (Note: {ex.get('note', 'Follow this pattern')}\\n\\n"
        
        prompt += f"Now your turn:\\nInput: {query}\\nOutput:"
        
        return prompt

# Test different placements
task = "Extract the person's name and age from the text."
examples = [
    {'input': 'John Smith is 25 years old', 'output': '{"name": "John Smith", "age": 25}'},
    {'input': 'Meet Sarah, age 30', 'output': '{"name": "Sarah", "age": 30}'}
]
query = "My name is Alice and I am 28"

strategy = ExamplePlacementStrategy()

print("PREFIX:")
print(strategy.prefix_examples(task, examples, query))
print("\\n" + "="*50 + "\\n")

print("SANDWICH:")
print(strategy.sandwich_examples(task, examples, query))
\`\`\`

## Example Diversity

### Coverage Across Variations

\`\`\`python
from typing import Set

class DiversityChecker:
    """
    Ensure examples cover diverse cases.
    Critical for robust few-shot learning.
    """
    
    @staticmethod
    def check_input_diversity(examples: List[Dict]) -> Dict[str, any]:
        """
        Analyze diversity of examples.
        """
        
        lengths = [len(ex['input']) for ex in examples]
        
        # Check length variance
        avg_length = sum(lengths) / len(lengths)
        length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        # Check vocabulary diversity
        all_words = set()
        for ex in examples:
            all_words.update(ex['input'].lower().split())
        
        # Check pattern diversity (simple heuristic)
        patterns: Set[str] = set()
        for ex in examples:
            # Simple pattern: length category + has numbers
            length_cat = 'short' if len(ex['input']) < 20 else 'medium' if len(ex['input']) < 50 else 'long'
            has_numbers = 'nums' if any(c.isdigit() for c in ex['input']) else 'no_nums'
            patterns.add(f"{length_cat}_{has_numbers}")
        
        return {
            'num_examples': len(examples),
            'avg_length': avg_length,
            'length_variance': length_variance,
            'unique_words': len(all_words),
            'unique_patterns': len(patterns),
            'diversity_score': len(patterns) / len(examples),  # 0-1 score
            'recommendation': 'Good diversity' if len(patterns) / len(examples) > 0.5 else 'Add more diverse examples'
        }
    
    @staticmethod
    def suggest_missing_cases(examples: List[Dict], task_type: str) -> List[str]:
        """
        Suggest what types of examples are missing.
        """
        
        suggestions = []
        
        # Check for edge cases
        has_empty = any(len(ex['input'].strip()) == 0 for ex in examples)
        has_long = any(len(ex['input']) > 100 for ex in examples)
        has_short = any(len(ex['input']) < 10 for ex in examples)
        has_numbers = any(any(c.isdigit() for c in ex['input']) for ex in examples)
        has_special_chars = any(any(not c.isalnum() and not c.isspace() for c in ex['input']) for ex in examples)
        
        if not has_empty and task_type in ['classification', 'extraction']:
            suggestions.append("Add example with empty/null input")
        
        if not has_long:
            suggestions.append("Add example with long input (>100 chars)")
        
        if not has_short:
            suggestions.append("Add example with very short input (<10 chars)")
        
        if not has_numbers and task_type in ['extraction', 'parsing']:
            suggestions.append("Add example with numbers")
        
        if not has_special_chars:
            suggestions.append("Add example with special characters")
        
        return suggestions

# Example usage
examples = [
    {'input': 'Simple short text', 'output': 'result1'},
    {'input': 'Another short one', 'output': 'result2'},
    {'input': 'Yet another short text', 'output': 'result3'}
]

checker = DiversityChecker()
diversity_report = checker.check_input_diversity(examples)

print("Diversity Analysis:")
for key, value in diversity_report.items():
    print(f"  {key}: {value}")

print("\\nMissing cases:")
suggestions = checker.suggest_missing_cases(examples, 'extraction')
for suggestion in suggestions:
    print(f"  - {suggestion}")
\`\`\`

## Dynamic Example Selection (RAG for Examples)

### Retrieve Relevant Examples on the Fly

\`\`\`python
from typing import List, Dict, Optional
import numpy as np

class DynamicExampleRetriever:
    """
    Retrieve most relevant examples for each query.
    Like RAG (Retrieval Augmented Generation) but for examples.
    Used in production systems like Cursor.
    """
    
    def __init__(self, example_database: List[Dict]):
        """
        Args:
            example_database: Large pool of examples with embeddings
        """
        self.examples = example_database
        self.embeddings = None
    
    def embed_examples(self, embed_func=None):
        """
        Create embeddings for all examples.
        In production, use OpenAI embeddings or sentence-transformers.
        """
        
        if embed_func is None:
            # Placeholder: simple word-based embedding
            def simple_embed(text: str) -> np.ndarray:
                words = text.lower().split()
                # Create a simple bag-of-words style embedding
                vocab = set()
                for ex in self.examples:
                    vocab.update(ex['input'].lower().split())
                vocab = sorted(vocab)
                
                vec = np.zeros(len(vocab))
                for i, word in enumerate(vocab):
                    if word in words:
                        vec[i] = 1
                
                return vec
            
            embed_func = simple_embed
        
        self.embeddings = [
            embed_func(ex['input'])
            for ex in self.examples
        ]
    
    def retrieve_similar(
        self,
        query: str,
        n: int = 3,
        embed_func=None
    ) -> List[Dict]:
        """
        Retrieve n most similar examples to query.
        """
        
        if self.embeddings is None:
            self.embed_examples(embed_func)
        
        # Embed query
        if embed_func is None:
            # Use same simple embedding
            words = query.lower().split()
            vocab = set()
            for ex in self.examples:
                vocab.update(ex['input'].lower().split())
            vocab = sorted(vocab)
            
            query_vec = np.zeros(len(vocab))
            for i, word in enumerate(vocab):
                if word in words:
                    query_vec[i] = 1
        else:
            query_vec = embed_func(query)
        
        # Calculate similarities
        similarities = []
        for i, ex_vec in enumerate(self.embeddings):
            # Cosine similarity
            dot_product = np.dot(query_vec, ex_vec)
            norm_product = (np.linalg.norm(query_vec) * np.linalg.norm(ex_vec))
            similarity = dot_product / norm_product if norm_product > 0 else 0
            similarities.append((self.examples[i], similarity))
        
        # Sort by similarity and return top n
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [ex for ex, sim in similarities[:n]]
    
    def create_dynamic_prompt(
        self,
        task: str,
        query: str,
        n_examples: int = 3
    ) -> str:
        """
        Create a prompt with dynamically selected examples.
        """
        
        relevant_examples = self.retrieve_similar(query, n=n_examples)
        
        examples_text = "\\n\\n".join([
            f"Input: {ex['input']}\\nOutput: {ex['output']}"
            for ex in relevant_examples
        ])
        
        return f"""{task}

Here are similar examples:
{examples_text}

Now apply to:
Input: {query}
Output:"""

# Usage example
example_database = [
    {
        'input': 'What is Python?',
        'output': 'Python is a high-level programming language...'
    },
    {
        'input': 'Explain machine learning',
        'output': 'Machine learning is a subset of AI...'
    },
    {
        'input': 'What is JavaScript?',
        'output': 'JavaScript is a programming language...'
    },
    {
        'input': 'Define neural networks',
        'output': 'Neural networks are computational models...'
    },
    {
        'input': 'What is TypeScript?',
        'output': 'TypeScript is a typed superset of JavaScript...'
    }
]

retriever = DynamicExampleRetriever(example_database)

# For a query about Java, it should retrieve programming language examples
query = "What is Java?"
relevant = retriever.retrieve_similar(query, n=2)

print("Query:", query)
print("\\nMost relevant examples:")
for ex in relevant:
    print(f"  - {ex['input']}")

# Create dynamic prompt
prompt = retriever.create_dynamic_prompt(
    task="Answer the question concisely",
    query=query,
    n_examples=2
)
print("\\nGenerated prompt:")
print(prompt)
\`\`\`

## Example Management System

### Building a Production Example Database

\`\`\`python
import json
from datetime import datetime
from typing import List, Dict, Optional

class ExampleDatabase:
    """
    Manage examples for few-shot prompting at scale.
    Used in production AI systems.
    """
    
    def __init__(self, filepath: Optional[str] = None):
        self.examples: Dict[str, List[Dict]] = {}  # task_id -> examples
        self.filepath = filepath
        
        if filepath:
            self.load(filepath)
    
    def add_example(
        self,
        task_id: str,
        input_text: str,
        output_text: str,
        metadata: Optional[Dict] = None
    ):
        """Add a new example to the database."""
        
        if task_id not in self.examples:
            self.examples[task_id] = []
        
        example = {
            'input': input_text,
            'output': output_text,
            'metadata': metadata or {},
            'added_at': datetime.now().isoformat(),
            'usage_count': 0,
            'success_rate': 1.0  # Start optimistic
        }
        
        self.examples[task_id].append(example)
    
    def get_examples(
        self,
        task_id: str,
        n: int = 3,
        strategy: str = 'best_performing'
    ) -> List[Dict]:
        """
        Get n examples for a task.
        
        Strategies:
        - 'random': Random selection
        - 'best_performing': Highest success rate
        - 'most_used': Most frequently used
        - 'newest': Most recent
        """
        
        if task_id not in self.examples:
            return []
        
        examples = self.examples[task_id]
        
        if strategy == 'random':
            import random
            return random.sample(examples, min(n, len(examples)))
        
        elif strategy == 'best_performing':
            sorted_examples = sorted(
                examples,
                key=lambda x: x['success_rate'],
                reverse=True
            )
            return sorted_examples[:n]
        
        elif strategy == 'most_used':
            sorted_examples = sorted(
                examples,
                key=lambda x: x['usage_count'],
                reverse=True
            )
            return sorted_examples[:n]
        
        elif strategy == 'newest':
            sorted_examples = sorted(
                examples,
                key=lambda x: x['added_at'],
                reverse=True
            )
            return sorted_examples[:n]
        
        return examples[:n]
    
    def record_usage(
        self,
        task_id: str,
        example_index: int,
        success: bool
    ):
        """Record that an example was used and whether it led to success."""
        
        if task_id in self.examples and example_index < len(self.examples[task_id]):
            example = self.examples[task_id][example_index]
            
            # Update usage count
            example['usage_count'] += 1
            
            # Update success rate (moving average)
            n = example['usage_count']
            old_rate = example['success_rate']
            new_rate = (old_rate * (n - 1) + (1 if success else 0)) / n
            example['success_rate'] = new_rate
    
    def get_statistics(self, task_id: str) -> Dict:
        """Get statistics about examples for a task."""
        
        if task_id not in self.examples:
            return {}
        
        examples = self.examples[task_id]
        
        return {
            'total_examples': len(examples),
            'avg_success_rate': sum(ex['success_rate'] for ex in examples) / len(examples),
            'total_uses': sum(ex['usage_count'] for ex in examples),
            'best_example': max(examples, key=lambda x: x['success_rate'])['input'][:50],
            'most_used': max(examples, key=lambda x: x['usage_count'])['input'][:50]
        }
    
    def save(self, filepath: Optional[str] = None):
        """Save database to file."""
        filepath = filepath or self.filepath
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(self.examples, f, indent=2)
    
    def load(self, filepath: str):
        """Load database from file."""
        try:
            with open(filepath, 'r') as f:
                self.examples = json.load(f)
        except FileNotFoundError:
            self.examples = {}

# Usage example
db = ExampleDatabase('examples.json')

# Add examples for sentiment analysis
db.add_example(
    task_id='sentiment_analysis',
    input_text='Great product!',
    output_text='positive',
    metadata={'confidence': 'high', 'difficulty': 'easy'}
)

db.add_example(
    task_id='sentiment_analysis',
    input_text='It was okay, nothing special',
    output_text='neutral',
    metadata={'confidence': 'medium', 'difficulty': 'medium'}
)

# Get best examples
examples = db.get_examples('sentiment_analysis', n=2, strategy='best_performing')
print("Best performing examples:")
for ex in examples:
    print(f"  {ex['input']} -> {ex['output']} (success rate: {ex['success_rate']:.2f})")

# Record usage
db.record_usage('sentiment_analysis', 0, success=True)

# Get statistics
stats = db.get_statistics('sentiment_analysis')
print("\\nTask statistics:", stats)

# Save
db.save()
\`\`\`

## Production Checklist

✅ **Example Quality**
- Representative of actual use cases
- Correct and verified outputs
- Cover edge cases and variations
- Clear and unambiguous

✅ **Example Diversity**
- Different input types/lengths
- Various difficulty levels
- Common and rare cases
- Edge cases included

✅ **Example Selection**
- Choose optimal number (3-5 typically)
- Order from simple to complex
- Use dynamic selection when possible
- Track performance metrics

✅ **Example Management**
- Store in database
- Version control
- Track usage and success rates
- Update based on performance

✅ **Testing**
- Test with and without examples
- Compare different example sets
- Measure consistency
- A/B test example strategies

## Key Takeaways

1. **Examples dramatically improve consistency** - Show, don't just tell
2. **3-5 examples is usually optimal** - More isn't always better (costs tokens)
3. **Order matters** - Simple to complex works best
4. **Diversity is critical** - Cover different patterns and edge cases
5. **Dynamic selection works best** - Choose examples relevant to each query
6. **Track performance** - Know which examples lead to success
7. **Manage examples like data** - Database, versioning, metrics
8. **Representative examples** - Must reflect real-world usage
9. **Format consistency** - Keep input/output format uniform
10. **Production systems use RAG** - Retrieve relevant examples dynamically

## Next Steps

Now that you understand few-shot learning, you're ready to explore **Chain-of-Thought Prompting** - teaching LLMs to show their reasoning for complex problem-solving.`,
};
