export const promptEngineering = {
  title: 'Prompt Engineering',
  id: 'prompt-engineering',
  content: `
# Prompt Engineering

## Introduction

Prompt engineering is the art and science of crafting inputs that elicit desired outputs from LLMs. Unlike traditional programming where you write explicit instructions, prompt engineering involves designing natural language prompts that guide models toward correct answers. From zero-shot to chain-of-thought, prompt engineering techniques can dramatically improve model performance without any fine-tuning.

### Why Prompt Engineering Matters

**No Training Needed**: Get better results instantly
**Cost-Effective**: Avoid fine-tuning expenses
**Iterative**: Rapidly test and improve
**Transferable**: Techniques work across models
**Accessible**: Anyone can learn and apply

---

## Fundamentals of Prompting

### Basic Prompt Structure

\`\`\`python
"""
Understanding prompt components
"""

class PromptComponents:
    """
    Anatomy of an effective prompt
    
    Components:
    1. Role/Context: Who the model should be
    2. Task: What to do
    3. Input: The data to process
    4. Output Format: How to structure response
    5. Constraints: Limitations and requirements
    6. Examples: Few-shot demonstrations
    """
    
    def basic_prompt(self, text):
        """
        Simple prompt
        """
        return f"Summarize: {text}"
    
    def structured_prompt(self, text):
        """
        Well-structured prompt with all components
        """
        return f"""You are an expert summarizer specialized in technical content.

Task: Create a concise summary of the following text.

Input:
{text}

Requirements:
- Maximum 3 sentences
- Focus on key technical points
- Use bullet points if listing multiple items

Summary:"""
    
    def compare_approaches(self):
        """
        Demonstrate impact of structure
        """
        text = "Transformers use attention mechanisms..."
        
        # Basic prompt
        basic = self.basic_prompt(text)
        # Result: Variable quality, unpredictable format
        
        # Structured prompt
        structured = self.structured_prompt(text)
        # Result: Consistent quality, predictable format
        
        return basic, structured

# Example usage
import anthropic
client = anthropic.Anthropic()

def test_prompts(text):
    """
    Compare different prompt structures
    """
    prompts = [
        # Bad: Too vague
        f"Summarize: {text}",
        
        # Better: Add context
        f"As a technical writer, summarize: {text}",
        
        # Best: Complete structure
        f"""You are a technical writer creating documentation.

Task: Summarize the following text for software engineers.

Text: {text}

Format your summary as:
- Main concept: [one sentence]
- Key details: [bullet points]
- Practical application: [one sentence]

Summary:"""
    ]
    
    results = []
    for prompt in prompts:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        results.append(response.content[0].text)
    
    return results
\`\`\`

---

## Zero-Shot Prompting

### Direct Instructions

\`\`\`python
"""
Zero-shot: No examples, just the task
"""

class ZeroShotPrompting:
    """
    Techniques for zero-shot prompts
    """
    
    def clear_instruction(self, task):
        """
        Be explicit and specific
        """
        # Bad: Vague
        bad = "Write about AI"
        
        # Good: Specific
        good = """Write a 200-word explanation of transformer architecture 
for someone with basic programming knowledge but no ML background. 
Include an analogy to help understanding."""
        
        return bad, good
    
    def classification_prompt(self, text, labels):
        """
        Zero-shot classification
        """
        prompt = f"""Classify the following text into one of these categories: {', '.join(labels)}

Text: "{text}"

Category:"""
        
        return prompt
    
    def extraction_prompt(self, text, fields):
        """
        Zero-shot information extraction
        """
        prompt = f"""Extract the following information from the text below.

Fields to extract: {', '.join(fields)}

Text: {text}

Provide output as JSON with keys: {', '.join(fields)}

JSON:"""
        
        return prompt
    
    def generation_prompt(self, spec):
        """
        Zero-shot generation with constraints
        """
        prompt = f"""Generate content with these specifications:

Type: {spec['type']}
Topic: {spec['topic']}
Length: {spec['length']}
Style: {spec['style']}
Audience: {spec['audience']}

Additional requirements:
{chr(10).join('- ' + req for req in spec['requirements'])}

Content:"""
        
        return prompt

# Practical examples
def zero_shot_examples():
    """
    Real-world zero-shot use cases
    """
    import openai
    client = openai.OpenAI()
    
    # 1. Sentiment analysis
    def sentiment(text):
        prompt = f"""Analyze the sentiment of this text. 
Respond with: Positive, Negative, or Neutral.

Text: "{text}"

Sentiment:"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    
    # 2. Translation
    def translate(text, target_language):
        prompt = f"""Translate the following text to {target_language}. 
Maintain the original tone and meaning.

Text: {text}

Translation:"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    # 3. Code generation
    def generate_code(description):
        prompt = f"""Write Python code to accomplish this task:

Task: {description}

Requirements:
- Include docstrings
- Handle edge cases
- Add type hints
- Include example usage

Code:"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    
    # Examples
    print(sentiment("I absolutely love this product!"))
    print(translate("Hello, world!", "French"))
    print(generate_code("Calculate fibonacci sequence"))
\`\`\`

---

## Few-Shot Prompting

### Learning from Examples

\`\`\`python
"""
Few-shot: Provide examples to guide the model
"""

class FewShotPrompting:
    """
    Few-shot learning techniques
    """
    
    def format_examples(self, examples):
        """
        Format few-shot examples
        """
        formatted = ""
        for ex in examples:
            formatted += f"Input: {ex['input']}\\n"
            formatted += f"Output: {ex['output']}\\n\\n"
        return formatted
    
    def few_shot_prompt(self, examples, new_input):
        """
        Create few-shot prompt
        """
        prompt = self.format_examples(examples)
        prompt += f"Input: {new_input}\\nOutput:"
        return prompt
    
    def classification_few_shot(self):
        """
        Few-shot classification
        """
        examples = [
            {"input": "I love this product!", "output": "Positive"},
            {"input": "Terrible experience", "output": "Negative"},
            {"input": "It's okay", "output": "Neutral"}
        ]
        
        new_text = "This is amazing!"
        prompt = self.few_shot_prompt(examples, new_text)
        
        return prompt
    
    def extraction_few_shot(self):
        """
        Few-shot information extraction
        """
        examples = [
            {
                "input": "John Smith works at Google in Mountain View",
                "output": "Name: John Smith, Company: Google, Location: Mountain View"
            },
            {
                "input": "Sarah Johnson is CEO of Microsoft in Seattle",
                "output": "Name: Sarah Johnson, Company: Microsoft, Location: Seattle"
            }
        ]
        
        new_text = "Mike Brown is a developer at Apple in Cupertino"
        prompt = self.few_shot_prompt(examples, new_text)
        
        return prompt

# Advanced few-shot techniques
class AdvancedFewShot:
    """
    Advanced few-shot strategies
    """
    
    def select_diverse_examples(self, examples, n=5):
        """
        Choose diverse examples for better coverage
        """
        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import SentenceTransformer
        
        # Embed examples
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode([ex['input'] for ex in examples])
        
        # Select diverse set
        selected = []
        selected_indices = []
        
        # Start with random example
        idx = 0
        selected.append(examples[idx])
        selected_indices.append(idx)
        
        # Iteratively select most different examples
        while len(selected) < n:
            similarities = cosine_similarity(
                embeddings[selected_indices],
                embeddings
            )
            avg_similarity = similarities.mean(axis=0)
            
            # Select least similar (most diverse)
            remaining = [i for i in range(len(examples)) if i not in selected_indices]
            next_idx = remaining[avg_similarity[remaining].argmin()]
            
            selected.append(examples[next_idx])
            selected_indices.append(next_idx)
        
        return selected
    
    def dynamic_few_shot(self, examples, query, k=3):
        """
        Select most relevant examples for query
        """
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Embed query and examples
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_emb = model.encode([query])
        example_embs = model.encode([ex['input'] for ex in examples])
        
        # Find most similar examples
        similarities = cosine_similarity(query_emb, example_embs)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        selected = [examples[i] for i in top_k_indices]
        
        return selected
    
    def format_with_reasoning(self, examples):
        """
        Include reasoning in examples
        """
        formatted = ""
        for ex in examples:
            formatted += f"Input: {ex['input']}\\n"
            formatted += f"Reasoning: {ex['reasoning']}\\n"
            formatted += f"Output: {ex['output']}\\n\\n"
        return formatted

# Example: Math word problems with few-shot
def math_problem_few_shot():
    """
    Few-shot learning for math
    """
    examples = [
        {
            "input": "If John has 5 apples and buys 3 more, how many does he have?",
            "output": "8"
        },
        {
            "input": "Sarah had 10 cookies and ate 4. How many are left?",
            "output": "6"
        },
        {
            "input": "A box contains 15 items. If you take out 7, how many remain?",
            "output": "8"
        }
    ]
    
    prompt = f"""Solve these math problems:

{chr(10).join(f"Q: {ex['input']}\\nA: {ex['output']}\\n" for ex in examples)}
Q: Mike has 20 marbles and loses 8. How many does he have left?
A:"""
    
    return prompt

# How many examples to use?
few_shot_guidelines = {
    "1-2 examples": "Simple tasks, clear patterns",
    "3-5 examples": "Most common, good balance",
    "5-10 examples": "Complex tasks, subtle distinctions",
    "10+ examples": "Rare, may hurt due to context limits"
}
\`\`\`

---

## Chain-of-Thought (CoT) Prompting

### Step-by-Step Reasoning

\`\`\`python
"""
Chain-of-Thought: Guide model through reasoning steps
"""

class ChainOfThoughtPrompting:
    """
    CoT techniques for complex reasoning
    """
    
    def standard_cot(self, problem):
        """
        Standard CoT with "Let's think step by step"
        """
        prompt = f"""{problem}

Let's think step by step."""
        
        return prompt
    
    def few_shot_cot(self, examples, problem):
        """
        Few-shot CoT with reasoning examples
        """
        prompt = ""
        for ex in examples:
            prompt += f"Q: {ex['question']}\\n"
            prompt += f"A: Let's think step by step.\\n{ex['reasoning']}\\n"
            prompt += f"Therefore, the answer is {ex['answer']}.\\n\\n"
        
        prompt += f"Q: {problem}\\n"
        prompt += f"A: Let's think step by step.\\n"
        
        return prompt
    
    def zero_shot_cot(self):
        """
        Zero-shot CoT (surprisingly effective!)
        """
        problem = """Roger has 5 tennis balls. He buys 2 more cans of 
tennis balls. Each can has 3 tennis balls. How many tennis balls 
does he have now?"""
        
        # Without CoT
        simple_prompt = f"{problem}\\n\\nAnswer:"
        
        # With CoT (just add "Let's think step by step")
        cot_prompt = f"{problem}\\n\\nLet's think step by step."
        
        return simple_prompt, cot_prompt
    
    def self_consistency(self, problem, n_samples=5):
        """
        Generate multiple reasoning paths, take majority vote
        """
        prompt = f"""{problem}

Let's think step by step."""
        
        # Generate multiple solutions
        solutions = []
        for _ in range(n_samples):
            response = generate(prompt, temperature=0.7)
            answer = extract_final_answer(response)
            solutions.append(answer)
        
        # Majority vote
        from collections import Counter
        final_answer = Counter(solutions).most_common(1)[0][0]
        
        return final_answer

# Complex reasoning examples
class ComplexReasoning:
    """
    CoT for challenging problems
    """
    
    def math_reasoning(self):
        """
        Multi-step math problem
        """
        prompt = """Q: A store sells apples for $2 each and oranges for $3 each. 
If you buy 4 apples and 5 oranges, and then get a 10% discount on 
the total, how much do you pay?

A: Let's solve this step by step.

Step 1: Calculate cost of apples
4 apples × $2/apple = $8

Step 2: Calculate cost of oranges  
5 oranges × $3/orange = $15

Step 3: Calculate total before discount
$8 + $15 = $23

Step 4: Calculate discount amount
10% of $23 = 0.10 × $23 = $2.30

Step 5: Calculate final price
$23 - $2.30 = $20.70

Therefore, the answer is $20.70.

Q: A restaurant bill is $85. You want to leave a 20% tip, and there's 
an 8% tax. If tip is calculated on the pre-tax amount, what's the 
total you pay?

A: Let's solve this step by step."""
        
        return prompt
    
    def logical_reasoning(self):
        """
        Logic puzzles with CoT
        """
        prompt = """Q: All roses are flowers. Some flowers fade quickly. 
Therefore, can we conclude that some roses fade quickly?

A: Let's reason through this step by step.

Step 1: Identify what we know
- Premise 1: All roses are flowers
- Premise 2: Some flowers fade quickly

Step 2: Analyze the logical structure
- "All roses are flowers" means roses ⊂ flowers
- "Some flowers fade quickly" means there exists a subset of flowers that fade

Step 3: Can we connect these?
- We know roses are within flowers
- We know some subset of flowers fades quickly
- But we don't know if roses overlap with the fading subset

Step 4: Conclusion
No, we cannot conclude that some roses fade quickly. The fading flowers 
might not include any roses.

Therefore, the answer is No - the conclusion doesn't follow logically.

Q: If it's raining, the ground is wet. The ground is wet. Is it raining?

A: Let's reason through this step by step."""
        
        return prompt
    
    def code_reasoning(self):
        """
        Code tracing with CoT
        """
        prompt = """Q: What does this code output?

def f(n):
    if n <= 1:
        return 1
    return n * f(n-1)

print(f(4))

A: Let's trace the execution step by step.

Step 1: Call f(4)
- n = 4, n > 1, so return 4 * f(3)

Step 2: Need to compute f(3)
- n = 3, n > 1, so return 3 * f(2)

Step 3: Need to compute f(2)
- n = 2, n > 1, so return 2 * f(1)

Step 4: Need to compute f(1)
- n = 1, n <= 1, so return 1

Step 5: Unwind the recursion
- f(1) = 1
- f(2) = 2 * f(1) = 2 * 1 = 2
- f(3) = 3 * f(2) = 3 * 2 = 6
- f(4) = 4 * f(3) = 4 * 6 = 24

Therefore, the code outputs 24.

Q: What does this code output?

def g(x):
    return x + 1

def h(x):
    return g(x) * 2

print(h(h(3)))

A: Let's trace the execution step by step."""
        
        return prompt

# Tree of Thoughts (ToT)
class TreeOfThoughts:
    """
    Explore multiple reasoning paths (more advanced than CoT)
    """
    
    def generate_thoughts(self, problem, n=3):
        """
        Generate multiple next steps
        """
        prompt = f"""{problem}

Generate {n} different possible next steps:"""
        
        # Returns multiple reasoning directions
        return prompt
    
    def evaluate_thoughts(self, problem, thought):
        """
        Evaluate quality of reasoning path
        """
        prompt = f"""Problem: {problem}

Reasoning step: {thought}

Rate this step from 1-10 on:
1. Correctness
2. Helpfulness
3. Clarity

Rating:"""
        
        return prompt
    
    def search_tree(self, problem, depth=3, breadth=3):
        """
        Search tree of possible reasoning paths
        """
        # Simplified tree search
        # 1. Generate breadth thoughts at each step
        # 2. Evaluate each
        # 3. Keep top k
        # 4. Repeat for depth levels
        # 5. Return best complete path
        pass

# When to use CoT?
cot_guidelines = {
    "Use CoT when": [
        "Multi-step reasoning required",
        "Math problems",
        "Logic puzzles",
        "Code tracing",
        "Complex decision-making"
    ],
    "Skip CoT when": [
        "Simple factual queries",
        "Single-step tasks",
        "Creative writing",
        "When tokens are expensive"
    ]
}
\`\`\`

---

## Advanced Prompting Techniques

### System Messages and Roles

\`\`\`python
"""
Advanced prompting strategies
"""

class AdvancedPrompting:
    """
    Sophisticated prompting techniques
    """
    
    def system_message_design(self):
        """
        Effective system messages
        """
        # Basic: Generic assistant
        basic = "You are a helpful assistant."
        
        # Better: Specific role
        better = "You are an expert Python programmer with 10 years of experience."
        
        # Best: Detailed persona
        best = """You are a senior software engineer specializing in Python and machine learning.

Your responses should:
- Be technically accurate and detailed
- Include code examples when relevant  
- Explain trade-offs and best practices
- Reference relevant documentation
- Warn about common pitfalls

Your tone is professional but approachable."""
        
        return basic, better, best
    
    def multi_role_prompting(self):
        """
        Simulate multiple experts
        """
        prompt = """Analyze this code from three perspectives:

1. Security Expert: Identify security vulnerabilities
2. Performance Engineer: Suggest optimizations
3. Code Reviewer: Check for best practices

Code:
def process_user_input(data):
    return eval(data)

Analysis:"""
        
        return prompt
    
    def constitutional_ai_principles(self):
        """
        Add behavioral constraints
        """
        prompt = """You are an AI assistant that follows these principles:

1. Harmlessness: Never provide harmful information
2. Helpfulness: Provide useful, accurate responses
3. Honesty: Admit when you don't know something
4. Respect: Be respectful to all users

When answering:
- If question is harmful, explain why and offer alternative
- If uncertain, express uncertainty
- If question is outside expertise, recommend resources

Now, answer the user's question:"""
        
        return prompt

# Prompt templates
class PromptTemplates:
    """
    Reusable prompt patterns
    """
    
    def classification_template(self, text, categories):
        """
        Classification with confidence
        """
        return f"""Classify the following text into one of these categories: {', '.join(categories)}

Text: "{text}"

Provide your answer in this format:
Category: [chosen category]
Confidence: [high/medium/low]
Reasoning: [brief explanation]

Classification:"""
    
    def extraction_template(self, text, schema):
        """
        Structured extraction
        """
        return f"""Extract information from the text according to this schema:

Schema:
{json.dumps(schema, indent=2)}

Text:
{text}

Return valid JSON matching the schema. If a field is not present, use null.

JSON:"""
    
    def transformation_template(self, text, transformation):
        """
        Text transformation
        """
        return f"""Transform the following text:

Original text:
{text}

Transformation: {transformation}

Transformed text:"""
    
    def comparison_template(self, item1, item2, criteria):
        """
        Structured comparison
        """
        return f"""Compare these two items based on the given criteria:

Item 1: {item1}
Item 2: {item2}

Criteria: {', '.join(criteria)}

Provide comparison in this format:
For each criterion:
- [Criterion]: Winner: [item1/item2], Reason: [explanation]

Comparison:"""

# Prompt chaining
class PromptChaining:
    """
    Break complex tasks into steps
    """
    
    def research_paper_summary(self, paper_text):
        """
        Multi-step summarization
        """
        # Step 1: Extract key information
        extraction_prompt = f"""Extract the following from this research paper:
- Main hypothesis
- Methodology
- Key findings
- Limitations

Paper: {paper_text}

Extracted information:"""
        
        extracted = generate(extraction_prompt)
        
        # Step 2: Generate summary
        summary_prompt = f"""Based on this extracted information, write a 
3-paragraph summary for a general audience:

{extracted}

Summary:"""
        
        summary = generate(summary_prompt)
        
        # Step 3: Generate key takeaways
        takeaways_prompt = f"""Based on this summary, list 3 key takeaways:

{summary}

Key takeaways:"""
        
        takeaways = generate(takeaways_prompt)
        
        return summary, takeaways
    
    def code_generation_pipeline(self, spec):
        """
        Chained code generation
        """
        # Step 1: Design
        design_prompt = f"""Design the architecture for this system:

Specification: {spec}

Provide:
1. High-level architecture
2. Key components
3. Data flow

Design:"""
        
        design = generate(design_prompt)
        
        # Step 2: Generate code for each component
        components = extract_components(design)
        code_parts = []
        
        for component in components:
            code_prompt = f"""Implement this component:

Component: {component}
Context: {design}

Code:"""
            code = generate(code_prompt)
            code_parts.append(code)
        
        # Step 3: Integration
        integration_prompt = f"""Integrate these components:

Components:
{chr(10).join(code_parts)}

Generate the main integration code:"""
        
        final_code = generate(integration_prompt)
        
        return final_code
\`\`\`

---

## Prompt Optimization

### Iterative Improvement

\`\`\`python
"""
Systematic prompt optimization
"""

class PromptOptimization:
    """
    Methods to improve prompts
    """
    
    def ab_test_prompts(self, prompt_a, prompt_b, test_cases):
        """
        Compare prompt performance
        """
        results_a = []
        results_b = []
        
        for test in test_cases:
            response_a = generate(prompt_a + test['input'])
            response_b = generate(prompt_b + test['input'])
            
            score_a = evaluate(response_a, test['expected'])
            score_b = evaluate(response_b, test['expected'])
            
            results_a.append(score_a)
            results_b.append(score_b)
        
        # Statistical comparison
        import numpy as np
        from scipy import stats
        
        mean_a = np.mean(results_a)
        mean_b = np.mean(results_b)
        
        # T-test
        t_stat, p_value = stats.ttest_ind(results_a, results_b)
        
        return {
            'prompt_a_mean': mean_a,
            'prompt_b_mean': mean_b,
            'significant': p_value < 0.05,
            'winner': 'A' if mean_a > mean_b else 'B'
        }
    
    def iterative_refinement(self, initial_prompt, test_cases, iterations=5):
        """
        Automatically improve prompts
        """
        current_prompt = initial_prompt
        best_score = 0
        
        for i in range(iterations):
            # Test current prompt
            score = self.evaluate_prompt(current_prompt, test_cases)
            
            if score > best_score:
                best_score = score
                best_prompt = current_prompt
            
            # Generate variations
            variations = self.generate_variations(current_prompt)
            
            # Test variations
            variation_scores = [
                self.evaluate_prompt(v, test_cases) 
                for v in variations
            ]
            
            # Select best
            best_variation_idx = np.argmax(variation_scores)
            current_prompt = variations[best_variation_idx]
        
        return best_prompt, best_score
    
    def evaluate_prompt(self, prompt, test_cases):
        """
        Score prompt on test cases
        """
        scores = []
        
        for test in test_cases:
            response = generate(prompt + test['input'])
            score = compute_similarity(response, test['expected'])
            scores.append(score)
        
        return np.mean(scores)
    
    def generate_variations(self, prompt):
        """
        Create prompt variations
        """
        # Use LLM to generate variations
        meta_prompt = f"""Generate 3 variations of this prompt 
that might improve performance:

Original: {prompt}

Variations:"""
        
        response = generate(meta_prompt)
        variations = parse_variations(response)
        
        return variations

# Common prompt problems and fixes
prompt_debugging = {
    "Problem: Inconsistent outputs": {
        "solution": "Lower temperature, add output format constraints"
    },
    "Problem: Too verbose": {
        "solution": "Add length constraints, request concise responses"
    },
    "Problem: Hallucinations": {
        "solution": "Ask for citations, use retrieval-augmented generation"
    },
    "Problem: Refuses valid requests": {
        "solution": "Rephrase task, provide context, use system message"
    },
    "Problem: Ignores instructions": {
        "solution": "Repeat key instructions, use formatting, add examples"
    }
}

# Prompt engineering best practices
best_practices = [
    "Be specific and explicit",
    "Provide context and examples",
    "Structure prompts clearly",
    "Use consistent formatting",
    "Test on diverse inputs",
    "Iterate based on results",
    "Version control your prompts",
    "Document what works",
    "Consider token costs",
    "Monitor for drift over time"
]
\`\`\`

---

## Conclusion

Prompt engineering techniques:

1. **Zero-Shot**: Direct instructions, no examples
2. **Few-Shot**: Provide 3-5 examples
3. **Chain-of-Thought**: "Let's think step by step"
4. **System Messages**: Define role and constraints
5. **Prompt Chaining**: Break complex tasks into steps

**Key Principles**:
- Be clear and specific
- Provide context and examples
- Use consistent formatting
- Iterate and test
- Consider output format

**Performance Gains**:
- Zero-shot to few-shot: 10-30% improvement
- Few-shot to CoT: 20-50% improvement on reasoning
- Optimized prompts: 2-5x better consistency

**Practical Tips**:
- Start simple, add complexity as needed
- Test on diverse inputs
- Version control prompts like code
- Use temperature=0 for consistency
- Monitor costs (longer prompts = more tokens)

Prompt engineering is the highest-leverage skill for working with LLMs—master it before considering fine-tuning.
`,
};
