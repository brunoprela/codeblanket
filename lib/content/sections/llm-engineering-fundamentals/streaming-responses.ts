/**
 * Streaming Responses Section
 * Module 1: LLM Engineering Fundamentals
 */

export const streamingresponsesSection = {
  id: 'streaming-responses',
  title: 'Streaming Responses',
  content: `# Streaming Responses

Master streaming to build responsive, real-time LLM applications like ChatGPT's interface.

## Why Streaming Matters

Without streaming, users wait for the entire response to complete. With streaming, they see

 tokens as they're generated.

### The User Experience Difference

\`\`\`python
"""
WITHOUT STREAMING:
User sends prompt
[5-10 second wait with loading spinner]
Full response appears at once

WITH STREAMING:
User sends prompt
[0.5 seconds]
Tokens start appearing: "The..."
[continues appearing] "The best way to..."
[streaming] "The best way to learn Python is..."
User can read while generating!

Benefits:
1. PERCEIVED SPEED - Feels instant
2. ENGAGEMENT - User can start reading immediately
3. CANCELLATION - Can stop if going wrong direction
4. BETTER UX - Matches ChatGPT experience users expect
"""
\`\`\`

## Server-Sent Events (SSE)

LLM APIs use Server-Sent Events for streaming.

### How SSE Works

\`\`\`python
"""
Server-Sent Events (SSE):

HTTP Connection Flow:
1. Client opens connection
2. Server keeps connection open
3. Server sends events as available
4. Each event is a chunk of data
5. Connection closes when done

SSE Format:
data: {"chunk": "Hello"}

data: {"chunk": " world"}

data: {"chunk": "!"}

data: [DONE]

Each "data:" line is an event
"""
\`\`\`

## Basic Streaming with OpenAI

### Simple Streaming Example

\`\`\`python
from openai import OpenAI

client = OpenAI()

def simple_stream (prompt: str):
    """Basic streaming example."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True  # ← Enable streaming!
    )

    # Iterate over chunks
    for chunk in response:
        # Check if chunk has content
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end=', flush=True)

    print()  # New line at end

# Usage
simple_stream("Explain what Python is in 3 sentences.")
# Output appears token by token: "Python" "is" "a" ...
\`\`\`

### Understanding Stream Chunks

\`\`\`python
from openai import OpenAI

client = OpenAI()

def inspect_stream_chunks (prompt: str):
    """Examine streaming chunk structure."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for i, chunk in enumerate (response):
        print(f"\\nChunk {i}:")
        print(f"  ID: {chunk.id}")
        print(f"  Model: {chunk.model}")
        print(f"  Choices: {len (chunk.choices)}")

        if chunk.choices:
            choice = chunk.choices[0]
            print(f"  Delta role: {choice.delta.role}")
            print(f"  Delta content: {choice.delta.content}")
            print(f"  Finish reason: {choice.finish_reason}")

        if i >= 5:  # Just show first few
            print("\\n... (truncated)")
            break

inspect_stream_chunks("Say hello!")
\`\`\`

### Stream Structure

\`\`\`python
"""
Streaming Response Structure:

Chunk 1:
{
    "id": "chatcmpl-123",
    "choices": [{
        "delta": {"role": "assistant"},
        "finish_reason": null
    }]
}

Chunk 2:
{
    "id": "chatcmpl-123",
    "choices": [{
        "delta": {"content": "Hello"},
        "finish_reason": null
    }]
}

Chunk 3:
{
    "id": "chatcmpl-123",
    "choices": [{
        "delta": {"content": "!"},
        "finish_reason": null
    }]
}

Final Chunk:
{
    "id": "chatcmpl-123",
    "choices": [{
        "delta": {},
        "finish_reason": "stop"
    }]
}

Key observations:
- First chunk has role
- Middle chunks have content
- Last chunk has finish_reason
- delta.content can be None
"""
\`\`\`

## Production Streaming Handler

\`\`\`python
from openai import OpenAI
from typing import Generator, Optional, Callable
from dataclasses import dataclass

@dataclass
class StreamChunk:
    """Structured streaming chunk."""
    content: Optional[str]
    finish_reason: Optional[str]
    is_final: bool

class StreamingHandler:
    """
    Production-ready streaming handler with callbacks.
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI()
        self.model = model

    def stream(
        self,
        messages: list,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        **kwargs
    ) -> Generator[StreamChunk, None, None]:
        """
        Stream with callbacks for handling chunks.

        Args:
            messages: Chat messages
            on_chunk: Called for each content chunk
            on_complete: Called with full text when done
            on_error: Called if error occurs
            **kwargs: Additional parameters (temperature, etc.)
        """

        full_content = []

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs
            )

            for chunk in response:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                content = choice.delta.content
                finish_reason = choice.finish_reason

                # Create structured chunk
                stream_chunk = StreamChunk(
                    content=content,
                    finish_reason=finish_reason,
                    is_final=finish_reason is not None
                )

                # Collect content
                if content:
                    full_content.append (content)

                    # Call chunk callback
                    if on_chunk:
                        on_chunk (content)

                yield stream_chunk

                # Check if done
                if finish_reason:
                    complete_text = '.join (full_content)

                    if on_complete:
                        on_complete (complete_text)

                    break

        except Exception as e:
            if on_error:
                on_error (e)
            raise

# Usage with callbacks
handler = StreamingHandler()

def on_chunk (content: str):
    """Called for each chunk."""
    print(content, end=', flush=True)

def on_complete (full_text: str):
    """Called when done."""
    print(f"\\n\\n[Completed: {len (full_text)} characters]")

def on_error (error: Exception):
    """Called on error."""
    print(f"\\n\\n[Error: {error}]")

messages = [{"role": "user", "content": "Count from 1 to 10"}]

# Stream with callbacks
for chunk in handler.stream(
    messages,
    on_chunk=on_chunk,
    on_complete=on_complete,
    on_error=on_error
):
    pass  # Callbacks handle everything
\`\`\`

## Collecting Streamed Content

### Stream Accumulator

\`\`\`python
from typing import List, Dict

class StreamAccumulator:
    """
    Accumulate streaming chunks into complete message.
    """

    def __init__(self):
        self.chunks: List[str] = []
        self.role: Optional[str] = None
        self.finish_reason: Optional[str] = None

    def add_chunk (self, chunk) -> Optional[str]:
        """
        Add chunk to accumulator.
        Returns content if present.
        """
        if not chunk.choices:
            return None

        choice = chunk.choices[0]
        delta = choice.delta

        # Capture role (first chunk)
        if delta.role:
            self.role = delta.role

        # Capture content
        if delta.content:
            self.chunks.append (delta.content)
            return delta.content

        # Capture finish reason
        if choice.finish_reason:
            self.finish_reason = choice.finish_reason

        return None

    def get_full_content (self) -> str:
        """Get complete accumulated content."""
        return '.join (self.chunks)

    def to_message (self) -> Dict[str, str]:
        """Convert to message format."""
        return {
            'role': self.role or 'assistant',
            'content': self.get_full_content()
        }

    def is_complete (self) -> bool:
        """Check if stream is complete."""
        return self.finish_reason is not None

    def reset (self):
        """Reset for reuse."""
        self.chunks = []
        self.role = None
        self.finish_reason = None

# Usage
from openai import OpenAI

client = OpenAI()
accumulator = StreamAccumulator()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Say hi!"}],
    stream=True
)

for chunk in response:
    content = accumulator.add_chunk (chunk)
    if content:
        print(content, end=', flush=True)

print(f"\\n\\nFull content: {accumulator.get_full_content()}")
print(f"Message format: {accumulator.to_message()}")
print(f"Finish reason: {accumulator.finish_reason}")
\`\`\`

## Error Handling in Streaming

Streaming can fail mid-generation. Handle gracefully!

### Stream Error Handling

\`\`\`python
from openai import OpenAI
import time

def safe_stream(
    messages: list,
    max_retries: int = 3,
    timeout: float = 30.0
) -> str:
    """
    Stream with error handling and retry logic.
    """
    client = OpenAI()

    for attempt in range (max_retries):
        try:
            content_parts = []
            start_time = time.time()

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                stream=True
            )

            for chunk in response:
                # Check timeout
                if time.time() - start_time > timeout:
                    raise TimeoutError (f"Stream exceeded {timeout}s")

                # Extract content
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    content_parts.append (content)
                    print(content, end=', flush=True)

                # Check if done
                if chunk.choices and chunk.choices[0].finish_reason:
                    print()  # New line
                    return '.join (content_parts)

            # If we get here, stream ended without finish_reason
            print("\\n[Warning: Stream ended unexpectedly]")
            return '.join (content_parts)

        except Exception as e:
            print(f"\\n[Error on attempt {attempt + 1}: {e}]")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"[Retrying in {wait_time}s...]")
                time.sleep (wait_time)
            else:
                print("[Max retries exceeded]")
                raise

# Usage
try:
    result = safe_stream(
        messages=[{"role": "user", "content": "Write a story"}],
        max_retries=3,
        timeout=30.0
    )
    print(f"Success! Got {len (result)} characters")
except Exception as e:
    print(f"Failed: {e}")
\`\`\`

## Streaming for Different Scenarios

### Scenario 1: Real-time Console Output

\`\`\`python
import sys

def stream_to_console (prompt: str):
    """Stream output directly to console."""
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    print("Assistant: ", end=')

    for chunk in response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end=', flush=True)
            sys.stdout.flush()  # Force output

    print()  # Newline at end

stream_to_console("Tell me a joke")
\`\`\`

### Scenario 2: Web API Streaming

\`\`\`python
"""
For FastAPI or Flask web applications.
Stream to client browser in real-time.
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI
import json

app = FastAPI()
client = OpenAI()

@app.post("/chat/stream")
async def chat_stream (prompt: str):
    """
    Streaming endpoint for web clients.
    """

    def generate():
        """Generator function for SSE."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content

                # Format as SSE
                yield f"data: {json.dumps({'content': content})}\\n\\n"

        # Send done event
        yield f"data: {json.dumps({'done': True})}\\n\\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

# Client JavaScript to consume:
"""
const eventSource = new EventSource('/chat/stream?prompt=Hello');

eventSource.onmessage = (event) => {
    const data = JSON.parse (event.data);

    if (data.done) {
        eventSource.close();
    } else {
        document.getElementById('output').textContent += data.content;
    }
};
"""
\`\`\`

### Scenario 3: Save to File While Streaming

\`\`\`python
def stream_to_file (prompt: str, filepath: str):
    """Stream to console AND save to file."""
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    with open (filepath, 'w') as f:
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content

                # Write to console
                print(content, end=', flush=True)

                # Write to file
                f.write (content)
                f.flush()  # Force write to disk

    print(f"\\n\\nSaved to {filepath}")

stream_to_file(
    "Write a short poem about coding",
    "poem.txt"
)
\`\`\`

## Streaming with Conversation History

\`\`\`python
class StreamingConversation:
    """
    Manage streaming conversations with history.
    """

    def __init__(self, system_prompt: str = "You are helpful."):
        self.client = OpenAI()
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]

    def stream_chat (self, user_input: str) -> str:
        """
        Stream a turn in the conversation.
        """
        # Add user message
        self.messages.append({
            "role": "user",
            "content": user_input
        })

        # Stream response
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
            stream=True
        )

        # Accumulate assistant response
        assistant_message = []

        print("Assistant: ", end=')

        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                assistant_message.append (content)
                print(content, end=', flush=True)

        print()  # Newline

        # Add to history
        full_response = '.join (assistant_message)
        self.messages.append({
            "role": "assistant",
            "content": full_response
        })

        return full_response

    def get_history (self) -> list:
        """Get conversation history."""
        return self.messages.copy()

# Usage
conv = StreamingConversation()

# Turn 1
response1 = conv.stream_chat("What is Python?")

# Turn 2 - has context from turn 1
response2 = conv.stream_chat("What are its main uses?")

# Turn 3
response3 = conv.stream_chat("Show me a code example")

print(f"\\nTotal messages: {len (conv.get_history())}")
\`\`\`

## Streaming Performance

### Measuring Stream Performance

\`\`\`python
import time
from dataclasses import dataclass

@dataclass
class StreamMetrics:
    """Metrics for streaming performance."""
    time_to_first_token: float
    total_time: float
    total_tokens: int
    tokens_per_second: float
    chunk_count: int

def measure_stream_performance (prompt: str) -> StreamMetrics:
    """
    Measure streaming performance metrics.
    """
    client = OpenAI()

    start_time = time.time()
    first_token_time = None
    chunk_count = 0
    token_count = 0

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in response:
        chunk_count += 1

        if chunk.choices[0].delta.content:
            # Mark first token time
            if first_token_time is None:
                first_token_time = time.time() - start_time

            # Count tokens (rough estimate)
            content = chunk.choices[0].delta.content
            token_count += len (content.split())

            print(content, end=', flush=True)

    total_time = time.time() - start_time
    tokens_per_second = token_count / total_time if total_time > 0 else 0

    print()  # Newline

    return StreamMetrics(
        time_to_first_token=first_token_time or 0,
        total_time=total_time,
        total_tokens=token_count,
        tokens_per_second=tokens_per_second,
        chunk_count=chunk_count
    )

# Test
metrics = measure_stream_performance("Explain quantum computing")

print(f"\\nPerformance Metrics:")
print(f"  Time to first token: {metrics.time_to_first_token:.3f}s")
print(f"  Total time: {metrics.total_time:.3f}s")
print(f"  Total tokens: {metrics.total_tokens}")
print(f"  Tokens/second: {metrics.tokens_per_second:.1f}")
print(f"  Chunks: {metrics.chunk_count}")
\`\`\`

## Canceling Streams

Stop generation mid-stream if needed.

\`\`\`python
import threading
import time

class CancelableStream:
    """
    Stream that can be canceled.
    """

    def __init__(self):
        self.client = OpenAI()
        self.should_cancel = False

    def stream_with_cancel(
        self,
        messages: list,
        cancel_event: threading.Event
    ) -> str:
        """
        Stream that checks cancel event.
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )

        content_parts = []

        for chunk in response:
            # Check if canceled
            if cancel_event.is_set():
                print("\\n[Stream canceled]")
                break

            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                content_parts.append (content)
                print(content, end=', flush=True)

        return '.join (content_parts)

# Usage
streamer = CancelableStream()
cancel_event = threading.Event()

def run_stream():
    """Run stream in thread."""
    result = streamer.stream_with_cancel(
        messages=[{"role": "user", "content": "Write a very long essay about Python"}],
        cancel_event=cancel_event
    )
    print(f"\\n\\nGot {len (result)} characters before cancel")

# Start stream in thread
thread = threading.Thread (target=run_stream)
thread.start()

# Cancel after 3 seconds
time.sleep(3)
cancel_event.set()

thread.join()
\`\`\`

## Streaming Best Practices

\`\`\`python
"""
STREAMING BEST PRACTICES:

1. ALWAYS FLUSH OUTPUT
   - Use flush=True in print()
   - Or sys.stdout.flush()
   - Otherwise output is buffered

2. HANDLE ERRORS GRACEFULLY
   - Streams can fail mid-generation
   - Implement retry logic
   - Show error messages to user

3. SHOW PROGRESS
   - Use spinner or dots while waiting for first token
   - Show typing indicator
   - Let user know something is happening

4. ALLOW CANCELLATION
   - Long streams should be cancelable
   - Check cancel flags in loop
   - Clean up resources

5. ACCUMULATE CONTENT
   - Save streamed content for history
   - Don't lose partial responses
   - Use accumulator pattern

6. MEASURE PERFORMANCE
   - Track time to first token
   - Monitor tokens per second
   - Optimize based on metrics

7. TEST THOROUGHLY
   - Test with slow connections
   - Test error scenarios
   - Test cancellation

8. SET TIMEOUTS
   - Don't let streams hang forever
   - Reasonable timeout (30-60s)
   - Retry on timeout
"""
\`\`\`

## Key Takeaways

1. **Streaming improves UX** - users see output immediately
2. **Set stream=True** to enable streaming
3. **Iterate over chunks** - each contains a small piece
4. **delta.content** contains the actual text
5. **First chunk** has role, last has finish_reason
6. **Accumulate chunks** to build complete response
7. **Handle errors** - streams can fail mid-generation
8. **Flush output** for real-time display
9. **Measure time to first token** - key UX metric
10. **Allow cancellation** for long responses

## Next Steps

Now you can stream responses for better UX. Next: **Error Handling & Retry Logic** - learning to build robust LLM applications that gracefully handle failures.`,
};
