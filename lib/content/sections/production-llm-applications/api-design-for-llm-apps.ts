export const apiDesignForLlmAppsContent = `
# API Design for LLM Apps

## Introduction

API design is crucial for LLM applications because it defines how clients interact with your AI capabilities, how you handle streaming responses, and how you manage the unique challenges of working with large language models. Unlike traditional REST APIs where responses are typically fast and predictable, LLM APIs must handle long-running requests, stream partial responses, manage token limits, and provide rich error information.

In this section, we'll explore the principles and patterns for designing robust, user-friendly APIs for LLM applications. We'll cover RESTful design, streaming responses via WebSockets and Server-Sent Events, versioning strategies, and best practices for documentation and error handling. Whether you're building an internal API or a public-facing service, these principles will help you create APIs that are intuitive, reliable, and scalable.

## RESTful API Design Principles

While LLM applications have unique requirements, the fundamentals of REST still apply and provide a solid foundation.

### Resource-Oriented Design

Think in terms of resources rather than actions. For an LLM application, your resources might include:

**Conversations**: Collections of messages between user and AI
**Messages**: Individual user or assistant messages
**Prompts**: Reusable prompt templates
**Generations**: Completed AI generations
**Models**: Available LLM models
**Documents**: Files for RAG or analysis

### HTTP Methods

Use HTTP methods semantically:

**GET**: Retrieve resources (conversations, generations, model info)
**POST**: Create new resources (new conversation, generate response)
**PUT/PATCH**: Update existing resources (edit message, update prompt)
**DELETE**: Remove resources (delete conversation, clear history)

### Status Codes

Use appropriate HTTP status codes:

**200 OK**: Successful GET request
**201 Created**: Successfully created resource
**202 Accepted**: Request accepted for async processing
**400 Bad Request**: Invalid input (malformed prompt, exceeded token limit)
**401 Unauthorized**: Missing or invalid API key
**429 Too Many Requests**: Rate limit exceeded
**500 Internal Server Error**: Unexpected error
**503 Service Unavailable**: LLM provider is down or overloaded

### API Structure Example

\`\`\`python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid

app = FastAPI(title="LLM API", version="1.0.0")

# Request/Response Models
class Message(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field (default_factory=datetime.utcnow)

class ConversationCreate(BaseModel):
    system_prompt: Optional[str] = Field(None, description="System prompt")
    metadata: Optional[dict] = Field (default_factory=dict)

class ConversationResponse(BaseModel):
    id: str
    created_at: datetime
    system_prompt: Optional[str]
    message_count: int

class GenerateRequest(BaseModel):
    message: str = Field(..., description="User message")
    model: str = Field (default="gpt-3.5-turbo", description="LLM model to use")
    temperature: float = Field (default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    stream: bool = Field (default=False, description="Enable streaming")

class GenerateResponse(BaseModel):
    message: str
    model: str
    tokens_used: int
    finish_reason: str


# Conversations Resource
@app.post("/v1/conversations", response_model=ConversationResponse, status_code=201)
async def create_conversation (conversation: ConversationCreate):
    """
    Create a new conversation.
    
    A conversation maintains context across multiple messages.
    """
    conversation_id = str (uuid.uuid4())
    
    # Store in database
    db_conversation = {
        'id': conversation_id,
        'created_at': datetime.utcnow(),
        'system_prompt': conversation.system_prompt,
        'messages': [],
        'metadata': conversation.metadata
    }
    
    save_conversation (db_conversation)
    
    return ConversationResponse(
        id=conversation_id,
        created_at=db_conversation['created_at'],
        system_prompt=conversation.system_prompt,
        message_count=0
    )


@app.get("/v1/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation (conversation_id: str):
    """Get conversation details."""
    conversation = get_conversation_from_db (conversation_id)
    
    if not conversation:
        raise HTTPException (status_code=404, detail="Conversation not found")
    
    return ConversationResponse(
        id=conversation['id'],
        created_at=conversation['created_at'],
        system_prompt=conversation.get('system_prompt'),
        message_count=len (conversation['messages'])
    )


@app.delete("/v1/conversations/{conversation_id}", status_code=204)
async def delete_conversation (conversation_id: str):
    """Delete a conversation and all its messages."""
    deleted = delete_conversation_from_db (conversation_id)
    
    if not deleted:
        raise HTTPException (status_code=404, detail="Conversation not found")
    
    return None


# Messages Resource
@app.get("/v1/conversations/{conversation_id}/messages", response_model=List[Message])
async def get_messages (conversation_id: str, limit: int = 50, offset: int = 0):
    """
    Get messages from a conversation.
    
    Supports pagination via limit and offset parameters.
    """
    conversation = get_conversation_from_db (conversation_id)
    
    if not conversation:
        raise HTTPException (status_code=404, detail="Conversation not found")
    
    messages = conversation['messages'][offset:offset + limit]
    
    return [Message(**msg) for msg in messages]


# Generation Resource
@app.post("/v1/conversations/{conversation_id}/generate", response_model=GenerateResponse)
async def generate_response(
    conversation_id: str,
    request: GenerateRequest
):
    """
    Generate an AI response in a conversation.
    
    Adds the user message to the conversation, generates a response,
    and adds the assistant message to the conversation.
    """
    conversation = get_conversation_from_db (conversation_id)
    
    if not conversation:
        raise HTTPException (status_code=404, detail="Conversation not found")
    
    # Add user message
    user_message = {
        'role': 'user',
        'content': request.message,
        'timestamp': datetime.utcnow()
    }
    conversation['messages'].append (user_message)
    
    # Build messages for LLM
    llm_messages = []
    if conversation.get('system_prompt'):
        llm_messages.append({
            'role': 'system',
            'content': conversation['system_prompt']
        })
    
    llm_messages.extend([
        {'role': msg['role'], 'content': msg['content']}
        for msg in conversation['messages']
    ])
    
    try:
        # Generate response
        response = openai.ChatCompletion.create(
            model=request.model,
            messages=llm_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add assistant message
        conversation['messages'].append({
            'role': 'assistant',
            'content': assistant_message,
            'timestamp': datetime.utcnow()
        })
        
        # Save updated conversation
        update_conversation (conversation)
        
        return GenerateResponse(
            message=assistant_message,
            model=request.model,
            tokens_used=response.usage.total_tokens,
            finish_reason=response.choices[0].finish_reason
        )
        
    except openai.error.InvalidRequestError as e:
        # Token limit exceeded or other validation error
        raise HTTPException (status_code=400, detail=str (e))
    
    except openai.error.RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    except Exception as e:
        logging.error (f"Error generating response: {str (e)}")
        raise HTTPException (status_code=500, detail="Internal server error")


# Models Resource
@app.get("/v1/models")
async def list_models():
    """List available LLM models."""
    return {
        "models": [
            {
                "id": "gpt-4",
                "name": "GPT-4",
                "description": "Most capable model, best for complex tasks",
                "max_tokens": 8192,
                "cost_per_1k_tokens": {"input": 0.03, "output": 0.06}
            },
            {
                "id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "description": "Fast and cost-effective for most tasks",
                "max_tokens": 4096,
                "cost_per_1k_tokens": {"input": 0.001, "output": 0.002}
            }
        ]
    }
\`\`\`

## Streaming Responses with Server-Sent Events

For LLM applications, streaming is essential for good user experience. Users can see responses as they're generated rather than waiting several seconds for a complete response.

### Why Streaming Matters

**Perceived Performance**: Users see output immediately, making the application feel faster even if total generation time is the same.

**Early Cancellation**: Users can stop generation if they see the response isn't what they need.

**Progressive Enhancement**: For long responses, users can start reading while generation continues.

**Feedback**: Real-time indication that the system is working, reducing anxiety during long generations.

### Server-Sent Events (SSE) Pattern

SSE is simpler than WebSockets and perfect for one-way streaming from server to client.

\`\`\`python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import openai
import json

app = FastAPI()

async def generate_stream(
    messages: list,
    model: str = "gpt-3.5-turbo"
) -> AsyncGenerator[str, None]:
    """
    Stream LLM responses using Server-Sent Events format.
    
    Yields chunks in SSE format: "data: {json}\\n\\n"
    """
    try:
        # Start streaming from OpenAI
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.7
        )
        
        # Track accumulated content for complete message
        full_content = ""
        
        for chunk in response:
            if chunk.choices[0].delta.get('content'):
                content = chunk.choices[0].delta.content
                full_content += content
                
                # Send chunk to client in SSE format
                data = {
                    'type': 'content',
                    'content': content
                }
                yield f"data: {json.dumps (data)}\\n\\n"
            
            # Handle finish reason
            if chunk.choices[0].finish_reason:
                data = {
                    'type': 'done',
                    'finish_reason': chunk.choices[0].finish_reason,
                    'full_content': full_content
                }
                yield f"data: {json.dumps (data)}\\n\\n"
        
    except openai.error.OpenAIError as e:
        # Send error to client
        error_data = {
            'type': 'error',
            'error': str (e)
        }
        yield f"data: {json.dumps (error_data)}\\n\\n"
    
    except Exception as e:
        error_data = {
            'type': 'error',
            'error': 'Internal server error'
        }
        yield f"data: {json.dumps (error_data)}\\n\\n"


@app.post("/v1/chat/stream")
async def stream_chat (message: str, conversation_id: Optional[str] = None):
    """
    Stream a chat response.
    
    Returns Server-Sent Events stream with incremental responses.
    """
    # Get conversation history if conversation_id provided
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message}
    ]
    
    if conversation_id:
        conversation = get_conversation (conversation_id)
        if conversation:
            messages = build_messages_from_conversation (conversation)
            messages.append({"role": "user", "content": message})
    
    return StreamingResponse(
        generate_stream (messages),
        media_type="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # Disable buffering in nginx
        }
    )


# Client-side JavaScript to consume the stream:
\"\""
const eventSource = new EventSource('/v1/chat/stream?message=Hello');

eventSource.onmessage = (event) => {
  const data = JSON.parse (event.data);
  
  if (data.type === 'content') {
    // Append content to UI
    appendToChat (data.content);
  } else if (data.type === 'done') {
    // Generation complete
    console.log('Finish reason:', data.finish_reason);
    eventSource.close();
  } else if (data.type === 'error') {
    // Handle error
    showError (data.error);
    eventSource.close();
  }
};

eventSource.onerror = (error) => {
  console.error('Stream error:', error);
  eventSource.close();
};
\"\""
\`\`\`

## WebSocket Pattern for Bidirectional Streaming

WebSockets provide full-duplex communication, useful for interactive applications where the client needs to send interruptions or additional context during generation.

\`\`\`python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict
import openai
import json
import asyncio

app = FastAPI()

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect (self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
    
    def disconnect (self, client_id: str):
        self.active_connections.pop (client_id, None)
    
    async def send_message (self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json (message)


manager = ConnectionManager()


@app.websocket("/ws/chat/{client_id}")
async def websocket_chat (websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time chat.
    
    Supports:
    - Streaming responses
    - Message cancellation
    - Real-time status updates
    """
    await manager.connect (websocket, client_id)
    
    conversation_history = []
    generation_task = None
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            message_type = data.get('type')
            
            if message_type == 'message':
                # User sent a message
                user_message = data.get('content')
                
                # Add to conversation history
                conversation_history.append({
                    'role': 'user',
                    'content': user_message
                })
                
                # Start generation
                generation_task = asyncio.create_task(
                    generate_and_stream (websocket, conversation_history)
                )
                
            elif message_type == 'cancel':
                # Cancel ongoing generation
                if generation_task and not generation_task.done():
                    generation_task.cancel()
                    await websocket.send_json({
                        'type': 'cancelled',
                        'message': 'Generation cancelled'
                    })
            
            elif message_type == 'clear':
                # Clear conversation history
                conversation_history = []
                await websocket.send_json({
                    'type': 'cleared',
                    'message': 'Conversation cleared'
                })
    
    except WebSocketDisconnect:
        manager.disconnect (client_id)
        if generation_task:
            generation_task.cancel()


async def generate_and_stream (websocket: WebSocket, messages: list):
    """Generate response and stream to WebSocket."""
    try:
        # Notify client that generation started
        await websocket.send_json({
            'type': 'status',
            'status': 'generating'
        })
        
        # Stream from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )
        
        full_response = ""
        
        for chunk in response:
            if chunk.choices[0].delta.get('content'):
                content = chunk.choices[0].delta.content
                full_response += content
                
                # Send chunk via WebSocket
                await websocket.send_json({
                    'type': 'content',
                    'content': content
                })
        
        # Add assistant response to history
        messages.append({
            'role': 'assistant',
            'content': full_response
        })
        
        # Notify completion
        await websocket.send_json({
            'type': 'done',
            'full_content': full_response
        })
        
    except asyncio.CancelledError:
        # Generation was cancelled
        raise
    
    except Exception as e:
        await websocket.send_json({
            'type': 'error',
            'error': str (e)
        })
\`\`\`

## API Versioning Strategies

As your LLM application evolves, you'll need to make changes to your API. Versioning ensures backward compatibility and smooth transitions.

### URL Path Versioning

The most common and explicit approach:

\`\`\`python
# Version 1
@app.post("/v1/generate")
async def generate_v1(prompt: str):
    return {"result": generate (prompt)}

# Version 2 with additional parameters
@app.post("/v2/generate")
async def generate_v2(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7
):
    return {
        "result": generate (prompt, model, temperature),
        "model": model,
        "temperature": temperature
    }
\`\`\`

**Pros**: Clear and explicit, easy to route, easy to deprecate old versions
**Cons**: Duplicates routes, can lead to code duplication

### Header-Based Versioning

Version specified in request headers:

\`\`\`python
from fastapi import Header, HTTPException

@app.post("/generate")
async def generate(
    prompt: str,
    api_version: str = Header (default="v1", alias="X-API-Version")
):
    if api_version == "v1":
        return generate_v1_logic (prompt)
    elif api_version == "v2":
        return generate_v2_logic (prompt)
    else:
        raise HTTPException (status_code=400, detail="Unsupported API version")
\`\`\`

**Pros**: Clean URLs, version not in path
**Cons**: Less discoverable, requires header manipulation

### Deprecation Strategy

\`\`\`python
from fastapi import FastAPI, Response
from datetime import datetime

app = FastAPI()

@app.post("/v1/generate")
async def generate_v1(prompt: str, response: Response):
    """
    Deprecated: Use /v2/generate instead.
    This endpoint will be removed on 2024-12-31.
    """
    # Add deprecation headers
    response.headers["X-API-Deprecated"] = "true"
    response.headers["X-API-Deprecation-Date"] = "2024-12-31"
    response.headers["X-API-Deprecation-Info"] = "Use /v2/generate instead"
    
    # Log usage of deprecated endpoint
    log_deprecated_usage("v1/generate", request.client.host)
    
    return {"result": generate (prompt)}
\`\`\`

## OpenAPI Documentation

FastAPI automatically generates OpenAPI (Swagger) documentation, but you should enhance it with detailed information.

\`\`\`python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="LLM API",
    description="""
    Production-ready API for Large Language Model applications.
    
    ## Features
    
    * **Conversations**: Maintain context across messages
    * **Streaming**: Real-time response streaming
    * **Multiple Models**: Support for GPT-4, GPT-3.5, and more
    * **Rate Limiting**: Per-user and per-API-key limits
    * **Cost Tracking**: Monitor your API usage and costs
    
    ## Authentication
    
    All endpoints require an API key in the \`X-API-Key\` header.
    
    ## Rate Limits
    
    - Free tier: 100 requests/hour
    - Pro tier: 1000 requests/hour
    - Enterprise: Custom limits
    """,
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "support@example.com"
    },
    license_info={
        "name": "MIT"
    }
)

class GenerateRequest(BaseModel):
    """Request model for text generation."""
    
    prompt: str = Field(
        ...,
        description="The text prompt for generation",
        example="Write a haiku about programming"
    )
    model: str = Field(
        default="gpt-3.5-turbo",
        description="LLM model to use",
        example="gpt-4"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Controls randomness. Lower is more deterministic.",
        example=0.7
    )
    max_tokens: Optional[int] = Field(
        None,
        description="Maximum tokens to generate",
        example=100
    )
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Write a haiku about programming",
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 100
            }
        }


@app.post(
    "/v1/generate",
    response_model=GenerateResponse,
    summary="Generate text",
    description="""
    Generate text using a large language model.
    
    This endpoint accepts a prompt and returns generated text. You can control
    the generation with parameters like temperature and max_tokens.
    
    **Cost**: Charges apply based on tokens used. See /v1/models for pricing.
    """,
    responses={
        200: {
            "description": "Successful generation",
            "content": {
                "application/json": {
                    "example": {
                        "result": "Code flows like water,\\nBugs emerge then disappear,\\nDebug until dawn.",
                        "tokens_used": 23,
                        "model": "gpt-3.5-turbo"
                    }
                }
            }
        },
        400: {"description": "Invalid request (e.g., token limit exceeded)"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    },
    tags=["Generation"]
)
async def generate (request: GenerateRequest):
    """Generate text from prompt."""
    # Implementation
    pass
\`\`\`

## Error Handling and Error Responses

Provide detailed, actionable error messages that help developers debug issues.

\`\`\`python
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional

class ErrorResponse(BaseModel):
    """Standard error response format."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID for support")


@app.exception_handler(Exception)
async def global_exception_handler (request: Request, exc: Exception):
    """Global exception handler with detailed error responses."""
    
    # Generate request ID for tracking
    request_id = str (uuid.uuid4())
    
    # Log error with request ID
    logging.error (f"Request {request_id} failed: {str (exc)}", exc_info=True)
    
    # Return detailed error response
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "request_id": request_id,
            "details": {
                "type": type (exc).__name__
            }
        }
    )


# Specific error handlers for LLM-related errors
@app.exception_handler (openai.error.InvalidRequestError)
async def openai_invalid_request_handler (request: Request, exc):
    """Handle OpenAI invalid request errors."""
    
    request_id = str (uuid.uuid4())
    
    # Parse OpenAI error to provide helpful message
    error_message = str (exc)
    
    if "maximum context length" in error_message:
        return JSONResponse(
            status_code=400,
            content={
                "error": "token_limit_exceeded",
                "message": "Your request exceeded the model's token limit",
                "details": {
                    "suggestion": "Try reducing the conversation history or prompt length",
                    "max_tokens": 4096  # Depends on model
                },
                "request_id": request_id
            }
        )
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "invalid_request",
            "message": error_message,
            "request_id": request_id
        }
    )


@app.exception_handler (openai.error.RateLimitError)
async def openai_rate_limit_handler (request: Request, exc):
    """Handle OpenAI rate limit errors."""
    
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": "You've exceeded your rate limit. Please try again later.",
            "details": {
                "retry_after": 60,  # seconds
                "suggestion": "Consider upgrading your plan for higher limits"
            }
        },
        headers={
            "Retry-After": "60"
        }
    )
\`\`\`

## Rate Limiting Implementation

Protect your API from abuse and manage costs with rate limiting.

\`\`\`python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import time
from collections import defaultdict
from datetime import datetime, timedelta

app = FastAPI()

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self):
        # Store buckets per API key
        self.buckets = defaultdict (lambda: {
            'tokens': 100,
            'last_update': time.time()
        })
        
        # Rate limit configs per tier
        self.configs = {
            'free': {'rate': 100, 'period': 3600},  # 100/hour
            'pro': {'rate': 1000, 'period': 3600},   # 1000/hour
            'enterprise': {'rate': 10000, 'period': 3600}  # 10000/hour
        }
    
    def check_rate_limit (self, api_key: str, tier: str = 'free') -> dict:
        """
        Check if request is within rate limit.
        
        Returns dict with:
        - allowed: bool
        - remaining: int
        - reset_at: datetime
        """
        config = self.configs[tier]
        bucket = self.buckets[api_key]
        
        now = time.time()
        time_passed = now - bucket['last_update']
        
        # Refill tokens based on time passed
        tokens_to_add = (time_passed / config['period']) * config['rate']
        bucket['tokens'] = min (config['rate'], bucket['tokens'] + tokens_to_add)
        bucket['last_update'] = now
        
        # Check if request allowed
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            remaining = int (bucket['tokens'])
            reset_at = datetime.utcnow() + timedelta(
                seconds=(config['rate'] - bucket['tokens']) / config['rate'] * config['period']
            )
            
            return {
                'allowed': True,
                'remaining': remaining,
                'reset_at': reset_at
            }
        else:
            reset_at = datetime.utcnow() + timedelta(
                seconds=(1 - bucket['tokens']) / config['rate'] * config['period']
            )
            
            return {
                'allowed': False,
                'remaining': 0,
                'reset_at': reset_at
            }


rate_limiter = RateLimiter()


@app.middleware("http")
async def rate_limit_middleware (request: Request, call_next):
    """Apply rate limiting to all requests."""
    
    # Skip rate limiting for docs endpoints
    if request.url.path in ['/docs', '/openapi.json']:
        return await call_next (request)
    
    # Get API key and user tier
    api_key = request.headers.get('X-API-Key')
    
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={"error": "missing_api_key", "message": "API key required"}
        )
    
    # Get user tier from database
    user = get_user_by_api_key (api_key)
    tier = user.get('tier', 'free') if user else 'free'
    
    # Check rate limit
    limit_result = rate_limiter.check_rate_limit (api_key, tier)
    
    if not limit_result['allowed']:
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": "Rate limit exceeded",
                "reset_at": limit_result['reset_at'].isoformat()
            },
            headers={
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str (int (limit_result['reset_at'].timestamp())),
                "Retry-After": str((limit_result['reset_at'] - datetime.utcnow()).seconds)
            }
        )
    
    # Process request
    response = await call_next (request)
    
    # Add rate limit headers
    response.headers["X-RateLimit-Remaining"] = str (limit_result['remaining'])
    response.headers["X-RateLimit-Reset"] = str (int (limit_result['reset_at'].timestamp()))
    
    return response
\`\`\`

## GraphQL Alternative

For complex LLM applications with many related resources, GraphQL can provide a more flexible API.

\`\`\`python
import strawberry
from strawberry.fastapi import GraphQLRouter
from typing import List, Optional

@strawberry.type
class Message:
    role: str
    content: str
    timestamp: str

@strawberry.type
class Conversation:
    id: str
    messages: List[Message]
    created_at: str
    
    @strawberry.field
    def message_count (self) -> int:
        return len (self.messages)

@strawberry.type
class GenerationResult:
    content: str
    tokens_used: int
    model: str

@strawberry.type
class Query:
    @strawberry.field
    def conversation (self, id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return get_conversation_from_db (id)
    
    @strawberry.field
    def conversations (self, limit: int = 10) -> List[Conversation]:
        """List conversations."""
        return get_conversations_from_db (limit=limit)

@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_conversation (self, system_prompt: Optional[str] = None) -> Conversation:
        """Create a new conversation."""
        return create_conversation_in_db (system_prompt)
    
    @strawberry.mutation
    def generate_response(
        self,
        conversation_id: str,
        message: str,
        model: str = "gpt-3.5-turbo"
    ) -> GenerationResult:
        """Generate an AI response."""
        # Add message to conversation
        # Generate response
        # Return result
        pass

schema = strawberry.Schema (query=Query, mutation=Mutation)

# Add to FastAPI app
graphql_app = GraphQLRouter (schema)
app.include_router (graphql_app, prefix="/graphql")
\`\`\`

## Best Practices Summary

1. **Use RESTful principles** but adapt for LLM-specific needs like streaming

2. **Always stream responses** when possible for better user experience

3. **Version your API** from the start to allow evolution without breaking clients

4. **Provide detailed documentation** with examples and error responses

5. **Implement rate limiting** to protect your service and manage costs

6. **Use semantic HTTP status codes** and detailed error messages

7. **Support both synchronous and asynchronous** patterns based on use case

8. **Include request IDs** in errors for support and debugging

9. **Add rate limit headers** to help clients manage their usage

10. **Design for cancellation** - users should be able to stop long-running requests

Your API is the primary interface to your LLM application. Invest time in designing it well, documenting it thoroughly, and handling edge cases gracefully.
`;
