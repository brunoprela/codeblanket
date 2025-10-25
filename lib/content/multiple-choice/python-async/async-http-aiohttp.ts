import { MultipleChoiceQuestion } from '@/lib/types';

export const asyncHttpAiohttpMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'aha-mc-1',
    question:
      'Why should you reuse aiohttp.ClientSession across multiple requests?',
    options: [
      'Sessions make requests faster',
      'Sessions manage connection pooling, reusing TCP connections to avoid handshake overhead',
      'You can only create one session per program',
      'Sessions are required by the HTTP protocol',
    ],
    correctAnswer: 1,
    explanation:
      'ClientSession maintains a connection pool that reuses TCP connections across requests. Creating a connection involves TCP handshake (~50-100ms) + TLS handshake (~100-200ms) = 150-300ms overhead. Reusing connections eliminates this. Example: Bad: for url in urls: async with ClientSession() as session: await session.get(url) creates new session per request (1000 requests × 200ms overhead = 200 seconds wasted!). Good: async with ClientSession() as session: await gather(*[session.get(url) for url in urls]) reuses connections (0ms overhead after first). Session also: manages DNS cache, handles cookies automatically, configures default headers/timeouts. Always create ONE session and reuse for all requests to same service.',
  },
  {
    id: 'aha-mc-2',
    question:
      'What is the difference between aiohttp.ClientError and aiohttp.ClientResponseError?',
    options: [
      'They are the same exception',
      'ClientError is base class for all client errors, ClientResponseError is specifically for HTTP error statuses (4xx, 5xx)',
      'ClientError is for timeouts, ClientResponseError is for connection errors',
      'ClientError is deprecated',
    ],
    correctAnswer: 1,
    explanation:
      'ClientError is the base exception class for all aiohttp client-side errors. ClientResponseError is a subclass for HTTP error response statuses (4xx/5xx). Hierarchy: ClientError (base) → ClientConnectorError (connection issues), ClientResponseError (HTTP errors), ClientSSLError (SSL issues), ClientPayloadError, etc. Example: try: await session.get(url); except ClientResponseError: # HTTP 404, 500, etc.; except ClientConnectorError: # DNS, network, connection refused; except ClientError: # Catches all above. Use specific exceptions for targeted handling, ClientError as catch-all.',
  },
  {
    id: 'aha-mc-3',
    question: 'What does response.raise_for_status() do in aiohttp?',
    options: [
      'It prints the HTTP status code',
      'It raises ClientResponseError if status is 4xx or 5xx',
      'It returns the status code',
      'It retries the request if it failed',
    ],
    correctAnswer: 1,
    explanation:
      "response.raise_for_status() checks if the HTTP status indicates an error (4xx client error, 5xx server error) and raises ClientResponseError if so. Example: async with session.get(url) as response: response.raise_for_status() # Raises if status >= 400; data = await response.json(). Useful for treating HTTP errors as exceptions rather than checking response.status manually. Without: if response.status >= 400: raise Exception() (manual check). With: response.raise_for_status() (automatic). Note: Some APIs use 200 for errors in JSON body—raise_for_status() won't catch these. Always check response structure when needed.",
  },
  {
    id: 'aha-mc-4',
    question:
      'How do you stream a large file download with aiohttp to avoid loading it all in memory?',
    options: [
      'Use await response.read()',
      'Use async for chunk in response.content.iter_chunked(size)',
      'Use response.stream()',
      'You cannot stream with aiohttp',
    ],
    correctAnswer: 1,
    explanation:
      'Use response.content.iter_chunked(chunk_size) to stream data in chunks without loading the entire response in memory. Example: async with session.get(url) as response: with open("file.zip", "wb") as f: async for chunk in response.content.iter_chunked(8192): f.write(chunk). Memory usage: Only 8KB chunk in memory at a time (vs gigabytes for full file). Compare to await response.read() which loads entire response in memory before returning (bad for large files). Streaming is essential for: large files, video/audio, server-sent events, any response that doesn\'t fit in memory. Default chunk size 8192 (8KB) is good balance between memory and I/O efficiency.',
  },
  {
    id: 'aha-mc-5',
    question: "What happens if you don't close an aiohttp ClientSession?",
    options: [
      'Nothing, Python garbage collector handles it',
      'Resource leak: open connections, file descriptors not released, potential "too many open files" error',
      'The program crashes immediately',
      'Subsequent requests fail',
    ],
    correctAnswer: 1,
    explanation:
      'Not closing ClientSession causes resource leaks: (1) TCP connections remain open (waste server resources), (2) File descriptors not released (OS limit typically ~1024, then "too many open files" error), (3) Memory not freed (buffers, internal state), (4) Pending data not flushed. Example: for i in range(10000): session = ClientSession(); await session.get(url); # Leak! After ~1000 iterations, hits file descriptor limit, program crashes. Correct: async with ClientSession() as session: ... (automatic cleanup) or try: session = ClientSession(); await session.get(url); finally: await session.close() (manual). Always use context manager (async with) for automatic cleanup—it\'s exception-safe.',
  },
];
