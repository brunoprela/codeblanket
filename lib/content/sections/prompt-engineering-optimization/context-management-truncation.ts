/**
 * Context Management & Truncation Section
 * Module 2: Prompt Engineering & Optimization
 */

export const contextmanagementtruncationSection = {
  id: 'context-management-truncation',
  title: 'Context Management & Truncation',
  content: `# Context Management & Truncation

Master managing context windows and handling long documents when building production AI applications.

## Overview: The Context Window Challenge

LLMs have limited context windows. Managing context effectively is critical for production apps like Cursor that work with large codebases.

### Context Window Limits (2024)

\`\`\`python
context_limits = {
    'gpt-3.5-turbo': 16_385,        # 16K tokens
    'gpt-4': 8_192,                  # 8K tokens
    'gpt-4-turbo': 128_000,          # 128K tokens
    'claude-3-opus': 200_000,        # 200K tokens
    'claude-3-sonnet': 200_000,      # 200K tokens
    'claude-3-haiku': 200_000,       # 200K tokens
    'gemini-1.5-pro': 1_000_000,     # 1M tokens!
}

# But: More context = higher cost and slower responses
# Need strategies to manage context efficiently
\`\`\`

## Token Counting and Management

\`\`\`python
import tiktoken

class TokenManager:
    """
    Manage tokens to stay within context limits.
    Essential for production applications.
    """
    
    def __init__(self, model: str = "gpt-4", max_tokens: int = 8000):
        self.model = model
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def fits_in_context(self, text: str, reserve_tokens: int = 500) -> bool:
        """Check if text fits in context window."""
        tokens = self.count_tokens(text)
        return tokens + reserve_tokens <= self.max_tokens
    
    def truncate_to_fit(
        self,
        text: str,
        max_tokens: int = None,
        from_start: bool = True
    ) -> str:
        """Truncate text to fit in token budget."""
        
        max_tokens = max_tokens or self.max_tokens
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate tokens
        if from_start:
            truncated_tokens = tokens[:max_tokens]
        else:
            truncated_tokens = tokens[-max_tokens:]
        
        # Decode back to text
        return self.encoding.decode(truncated_tokens)
    
    def smart_truncate(
        self,
        text: str,
        max_tokens: int,
        preserve_start: int = 500,
        preserve_end: int = 500
    ) -> str:
        """
        Smart truncation preserving start and end.
        Used by Cursor for large files.
        """
        
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Keep start and end, truncate middle
        start_tokens = tokens[:preserve_start]
        end_tokens = tokens[-preserve_end:]
        
        truncated_length = preserve_start + preserve_end
        if truncated_length > max_tokens:
            # If still too long, split budget
            keep_start = max_tokens // 2
            keep_end = max_tokens - keep_start
            start_tokens = tokens[:keep_start]
            end_tokens = tokens[-keep_end:]
        
        # Add truncation marker
        middle_marker = self.encoding.encode("\\n\\n[... content truncated ...]\\n\\n")
        combined_tokens = start_tokens + middle_marker + end_tokens
        
        return self.encoding.decode(combined_tokens)

# Usage
manager = TokenManager(model="gpt-4", max_tokens=8000)

long_text = "Your code here..." * 1000

print(f"Original tokens: {manager.count_tokens(long_text)}")
print(f"Fits in context: {manager.fits_in_context(long_text)}")

truncated = manager.smart_truncate(long_text, max_tokens=2000)
print(f"Truncated tokens: {manager.count_tokens(truncated)}")
\`\`\`

## Truncation Methods

### Different Strategies for Different Use Cases

\`\`\`python
from enum import Enum
from typing import List

class TruncationStrategy(Enum):
    """Different truncation approaches."""
    BEGINNING = "keep_beginning"
    END = "keep_end"
    MIDDLE = "keep_middle"
    SMART = "smart_sliding"
    SUMMARIZE = "summarize_first"

class ContextTruncator:
    """
    Multiple truncation strategies for different scenarios.
    """
    
    def __init__(self, token_manager: TokenManager):
        self.tm = token_manager
    
    def truncate(
        self,
        text: str,
        max_tokens: int,
        strategy: TruncationStrategy = TruncationStrategy.SMART
    ) -> str:
        """Truncate using specified strategy."""
        
        if strategy == TruncationStrategy.BEGINNING:
            return self._keep_beginning(text, max_tokens)
        elif strategy == TruncationStrategy.END:
            return self._keep_end(text, max_tokens)
        elif strategy == TruncationStrategy.MIDDLE:
            return self._keep_middle(text, max_tokens)
        elif strategy == TruncationStrategy.SMART:
            return self._smart_sliding(text, max_tokens)
        elif strategy == TruncationStrategy.SUMMARIZE:
            return self._summarize_then_use(text, max_tokens)
        
        return text
    
    def _keep_beginning(self, text: str, max_tokens: int) -> str:
        """Keep beginning, truncate end."""
        return self.tm.truncate_to_fit(text, max_tokens, from_start=True)
    
    def _keep_end(self, text: str, max_tokens: int) -> str:
        """Keep end, truncate beginning."""
        return self.tm.truncate_to_fit(text, max_tokens, from_start=False)
    
    def _keep_middle(self, text: str, max_tokens: int) -> str:
        """
        Keep middle section.
        Useful when you know relevant content is in the middle.
        """
        tokens = self.tm.encoding.encode(text)
        total_tokens = len(tokens)
        
        if total_tokens <= max_tokens:
            return text
        
        # Calculate middle section
        start_idx = (total_tokens - max_tokens) // 2
        end_idx = start_idx + max_tokens
        
        middle_tokens = tokens[start_idx:end_idx]
        return self.tm.encoding.decode(middle_tokens)
    
    def _smart_sliding(self, text: str, max_tokens: int) -> str:
        """
        Preserve important parts (beginning and end).
        How Cursor handles large files.
        """
        return self.tm.smart_truncate(
            text,
            max_tokens,
            preserve_start=max_tokens // 3,
            preserve_end=max_tokens // 3
        )
    
    def _summarize_then_use(self, text: str, max_tokens: int) -> str:
        """
        Summarize content first, then provide full details.
        Best for analysis tasks.
        """
        # This would call LLM to summarize
        # For demo, just show the concept
        if self.tm.count_tokens(text) <= max_tokens:
            return text
        
        # In production: call LLM to summarize
        summary = "[Summary of content would go here]"
        
        # Then include as much original as fits
        remaining_budget = max_tokens - self.tm.count_tokens(summary)
        partial_text = self.tm.truncate_to_fit(text, remaining_budget)
        
        return f"{summary}\\n\\n[Original content (truncated)]:\\n{partial_text}"

# Example usage
truncator = ContextTruncator(manager)

long_doc = "Your long document..." * 500

# Different strategies for different needs
beginning = truncator.truncate(long_doc, 1000, TruncationStrategy.BEGINNING)
smart = truncator.truncate(long_doc, 1000, TruncationStrategy.SMART)

print("Beginning strategy tokens:", manager.count_tokens(beginning))
print("Smart strategy tokens:", manager.count_tokens(smart))
\`\`\`

## Hierarchical Context Management

### Managing Multiple Documents

\`\`\`python
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class ContextItem:
    """A piece of context with metadata."""
    content: str
    priority: float  # 0-1, higher = more important
    category: str  # 'system', 'recent', 'relevant', 'background'
    tokens: int

class HierarchicalContextManager:
    """
    Manage multiple context items with priorities.
    Used in production AI apps like Cursor.
    """
    
    def __init__(self, token_manager: TokenManager, max_tokens: int = 8000):
        self.tm = token_manager
        self.max_tokens = max_tokens
        self.reserve_tokens = 500  # For response
    
    def build_context(
        self,
        items: List[ContextItem],
        query: str
    ) -> str:
        """
        Build optimal context from multiple items.
        Prioritizes important items to fit in budget.
        """
        
        # Calculate available budget
        query_tokens = self.tm.count_tokens(query)
        available_tokens = self.max_tokens - query_tokens - self.reserve_tokens
        
        # Sort by priority (higher first)
        sorted_items = sorted(items, key=lambda x: x.priority, reverse=True)
        
        # Add items until budget exhausted
        selected_items = []
        tokens_used = 0
        
        for item in sorted_items:
            if tokens_used + item.tokens <= available_tokens:
                selected_items.append(item)
                tokens_used += item.tokens
            else:
                # Try to fit truncated version
                remaining_budget = available_tokens - tokens_used
                if remaining_budget > 100:  # Minimum useful size
                    truncated_content = self.tm.truncate_to_fit(
                        item.content,
                        remaining_budget
                    )
                    selected_items.append(ContextItem(
                        content=truncated_content,
                        priority=item.priority,
                        category=item.category,
                        tokens=remaining_budget
                    ))
                    tokens_used += remaining_budget
                break
        
        # Build context by category
        context_parts = []
        
        for category in ['system', 'recent', 'relevant', 'background']:
            category_items = [
                item for item in selected_items
                if item.category == category
            ]
            
            if category_items:
                context_parts.append(f"# {category.upper()}\\n")
                for item in category_items:
                    context_parts.append(item.content)
                context_parts.append("\\n")
        
        return "\\n".join(context_parts)

# Example usage
tm = TokenManager()
manager = HierarchicalContextManager(tm, max_tokens=8000)

# Multiple context items with priorities
items = [
    ContextItem(
        content="You are a helpful coding assistant.",
        priority=1.0,  # Highest priority
        category='system',
        tokens=tm.count_tokens("You are a helpful coding assistant.")
    ),
    ContextItem(
        content="Current file: user.py\\nclass User: ...",
        priority=0.9,
        category='recent',
        tokens=tm.count_tokens("Current file: user.py\\nclass User: ...")
    ),
    ContextItem(
        content="Related file: database.py\\nclass Database: ...",
        priority=0.6,
        category='relevant',
        tokens=tm.count_tokens("Related file: database.py\\nclass Database: ...")
    ),
    ContextItem(
        content="Documentation: ...",
        priority=0.3,
        category='background',
        tokens=tm.count_tokens("Documentation: ...")
    ),
]

query = "Add error handling to User class"
context = manager.build_context(items, query)

print("Built context:")
print(context)
print(f"\\nTotal tokens: {tm.count_tokens(context)}")
\`\`\`

## Sliding Window Techniques

### For Processing Long Documents

\`\`\`python
from typing import Iterator, List

class SlidingWindowProcessor:
    """
    Process long documents using sliding window.
    Useful for summarization, analysis of long texts.
    """
    
    def __init__(self, token_manager: TokenManager, window_size: int = 2000):
        self.tm = token_manager
        self.window_size = window_size
        self.overlap = window_size // 4  # 25% overlap
    
    def create_windows(self, text: str) -> List[str]:
        """Split text into overlapping windows."""
        
        tokens = self.tm.encoding.encode(text)
        windows = []
        
        position = 0
        while position < len(tokens):
            # Extract window
            window_tokens = tokens[position:position + self.window_size]
            window_text = self.tm.encoding.decode(window_tokens)
            windows.append(window_text)
            
            # Move position with overlap
            position += self.window_size - self.overlap
        
        return windows
    
    def process_windowed(
        self,
        text: str,
        process_func: callable,
        aggregate_func: callable = None
    ) -> any:
        """
        Process long text in windows and aggregate results.
        
        Args:
            text: Long text to process
            process_func: Function to process each window
            aggregate_func: Function to combine window results
        """
        
        windows = self.create_windows(text)
        results = []
        
        print(f"Processing {len(windows)} windows...")
        
        for i, window in enumerate(windows):
            print(f"  Window {i+1}/{len(windows)}")
            result = process_func(window)
            results.append(result)
        
        if aggregate_func:
            return aggregate_func(results)
        
        return results

# Example: Summarize long document
from openai import OpenAI

client = OpenAI()
tm = TokenManager()
processor = SlidingWindowProcessor(tm, window_size=2000)

def summarize_window(text: str) -> str:
    """Summarize a window of text."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Summarize in 2-3 sentences:\\n\\n{text}"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def combine_summaries(summaries: List[str]) -> str:
    """Combine window summaries into final summary."""
    combined = "\\n\\n".join(summaries)
    
    # If combined still too long, summarize the summaries
    if tm.count_tokens(combined) > 2000:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"Create a comprehensive summary from these section summaries:\\n\\n{combined}"
                }
            ]
        )
        return response.choices[0].message.content
    
    return combined

# Process long document
long_document = "Your long document text here..." * 1000

# This would actually call the LLM for each window
# final_summary = processor.process_windowed(
#     long_document,
#     process_func=summarize_window,
#     aggregate_func=combine_summaries
# )
\`\`\`

## How Cursor Manages File Contexts

\`\`\`python
class CursorStyleContextManager:
    """
    Approximate how Cursor manages context for large codebases.
    """
    
    def __init__(self, token_manager: TokenManager):
        self.tm = token_manager
        self.max_context = 8000
    
    def build_codebase_context(
        self,
        current_file: str,
        current_selection: str,
        related_files: List[Dict[str, str]],
        user_query: str
    ) -> str:
        """
        Build context like Cursor does.
        Prioritizes current file and selection.
        """
        
        # Budget allocation
        query_tokens = self.tm.count_tokens(user_query)
        available = self.max_context - query_tokens - 500
        
        # Priority allocation
        budgets = {
            'current_selection': int(available * 0.3),  # 30% for selection
            'current_file': int(available * 0.4),       # 40% for current file
            'related_files': int(available * 0.3),      # 30% for related files
        }
        
        context_parts = []
        
        # 1. Current selection (highest priority)
        if current_selection:
            selection = self.tm.truncate_to_fit(
                current_selection,
                budgets['current_selection']
            )
            context_parts.append(f"# CURRENT SELECTION\\n{selection}\\n")
        
        # 2. Current file context
        if current_file:
            # Use smart truncation to keep beginning and end
            file_context = self.tm.smart_truncate(
                current_file,
                budgets['current_file']
            )
            context_parts.append(f"# CURRENT FILE\\n{file_context}\\n")
        
        # 3. Related files (most relevant first)
        related_budget = budgets['related_files']
        related_budget_per_file = related_budget // len(related_files) if related_files else 0
        
        if related_files and related_budget_per_file > 100:
            context_parts.append("# RELATED FILES\\n")
            for file_info in related_files[:3]:  # Max 3 related files
                file_summary = self.tm.truncate_to_fit(
                    f"## {file_info['path']}\\n{file_info['content']}",
                    related_budget_per_file
                )
                context_parts.append(file_summary + "\\n")
        
        return "\\n".join(context_parts)

# Example
cursor_manager = CursorStyleContextManager(tm)

context = cursor_manager.build_codebase_context(
    current_file="class User:\\n    def __init__(self):...",
    current_selection="def login(self, username, password):\\n    ...",
    related_files=[
        {'path': 'auth.py', 'content': 'class Auth:...'},
        {'path': 'database.py', 'content': 'class DB:...'}
    ],
    user_query="Add error handling to login method"
)

print(context)
\`\`\`

## Production Checklist

✅ **Token Management**
- Count tokens accurately
- Stay within limits
- Reserve tokens for response
- Monitor token usage
- Optimize token efficiency

✅ **Truncation Strategy**
- Choose appropriate strategy
- Preserve critical information
- Test different approaches
- Handle edge cases
- Log truncation events

✅ **Context Prioritization**
- Rank by importance
- System prompts highest priority
- Recent context next
- Background info last
- Dynamic allocation

✅ **Performance**
- Cache token counts
- Efficient encoding/decoding
- Parallel processing when possible
- Monitor latency impact
- Optimize for common cases

✅ **User Experience**
- Transparent about truncation
- Show what was included/excluded
- Allow user control
- Provide feedback
- Handle errors gracefully

## Key Takeaways

1. **Context windows are limited** - Must manage carefully
2. **Token counting is essential** - Use tiktoken for accuracy
3. **Smart truncation preserves value** - Keep beginning and end
4. **Prioritize context items** - Most important first
5. **Sliding windows for long documents** - Process in chunks
6. **Cursor's approach works** - Current file + selection + related
7. **Hierarchical context helps** - System > Recent > Relevant > Background
8. **Overlap prevents information loss** - Use in sliding windows
9. **Monitor token usage** - Critical for cost control
10. **Test truncation strategies** - Different tasks need different approaches

## Next Steps

Now that you understand context management, you're ready to explore **Negative Prompting & Constraints** - learning how to tell LLMs what NOT to do and set boundaries for safe, reliable outputs.`,
};
