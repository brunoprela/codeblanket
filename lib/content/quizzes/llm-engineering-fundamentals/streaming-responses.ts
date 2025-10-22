/**
 * Quiz questions for Streaming Responses section
 */

export const streamingresponsesQuiz = [
    {
        id: 'q1',
        question:
            'Explain why streaming responses dramatically improve perceived performance even though the total time to generate the response is the same. How would you measure the user experience improvement quantitatively?',
        sampleAnswer:
            'Streaming improves perceived performance through psychological principles: (1) Time to first token is ~0.5s vs 5-10s for full response - users see activity immediately, reducing perceived wait time by 90%+, (2) Progressive disclosure keeps users engaged - they can start reading while generation continues, utilizing otherwise wasted time, (3) Reduces anxiety - blank loading spinners create uncertainty about whether the system is working; streaming provides constant feedback. Quantitative measurements: (1) Time to First Byte (TTFB) - measure time until first token appears (should be <1s), (2) Perceived wait time surveys - ask users "how long did this feel?" vs actual time; streaming typically feels 50-70% faster than actual duration, (3) Engagement metrics - do users cancel/abandon less with streaming? Measure cancellation rates, (4) Task completion - can users complete their task faster? For reading tasks, they start reading at ~2s with streaming vs 10s without, saving 8 seconds, (5) Scroll behavior - track when users start scrolling; with streaming they scroll as content appears, without streaming they wait then scroll all at once. Real data from ChatGPT: Users perceive streaming responses as nearly instant even though generation takes 10+ seconds, because the 0.5s first token time feels immediate. Without streaming, same 10s generation feels like "waiting", making the experience feel 10-20x slower despite identical total time.',
        keyPoints: [
            'Time to first token (<1s) is key to perceived speed',
            'Users can read while generation continues',
            'Streaming provides constant feedback vs uncertain waiting',
            'Measure TTFB, perceived wait time, and engagement',
            'Psychological improvement despite same total time'
        ]
    },
    {
        id: 'q2',
        question:
            'You are building a code generation feature that streams responses. A user reports that sometimes the code is cut off mid-generation. Describe the potential causes and how you would debug and fix this issue.',
        sampleAnswer:
            'Potential causes and solutions: (1) Timeout issues - streaming connections can timeout if individual chunks take too long. Debug: Check if cutoffs correlate with long pauses in generation. Log time between chunks. Fix: Implement connection keep-alive with periodic heartbeat chunks, increase timeout settings on both client and server, add reconnection logic if timeout occurs. (2) Client-side buffer overflow - browser or client may have buffer limits for streaming responses. Debug: Check if cutoffs happen at consistent byte counts (e.g., always at ~64KB). Test with different clients/browsers. Fix: Implement backpressure handling, consume chunks as they arrive rather than buffering all, use proper streaming APIs (SSE, WebSocket) with flow control. (3) Max tokens limit hit - LLM has max_tokens parameter that stops generation. Debug: Check if finish_reason is "length" vs "stop". Log token counts. Fix: Increase max_tokens parameter or make it dynamic based on prompt size, show warning to user when approaching limit. (4) Network interruption - user connection drops mid-stream. Debug: Implement connection monitoring, log disconnect events. Fix: Save partial responses, allow resume/retry from last chunk, show user notification of disconnect with retry option. (5) Server-side crash - application crashes mid-generation. Debug: Check server logs for exceptions during streaming. Fix: Wrap streaming in try-catch, save partial response before error, return error chunk to client with graceful handling. Implementation: Accumulate all chunks on client side, detect incomplete responses (no finish_reason or no closing marker), show "Generation interrupted" message with retry button, save partial response for debugging.',
        keyPoints: [
            'Timeouts are common cause - implement keep-alive',
            'Check max_tokens limit and finish_reason',
            'Network issues require retry and resume logic',
            'Save partial responses for debugging',
            'Log connection state and chunk timing'
        ]
    },
    {
        id: 'q3',
        question:
            'Compare the implementation complexity and user experience of streaming vs non-streaming responses for a web application. Under what circumstances would you choose NOT to use streaming?',
        sampleAnswer:
            'Implementation complexity comparison: Non-streaming is simpler - single request/response, standard HTTP, easy error handling, works with any infrastructure. Streaming is more complex - requires SSE or WebSockets, stateful connections, chunk buffering on client, more complex error handling (mid-stream failures), need to handle connection drops and reconnection. However, user experience strongly favors streaming for responses >2 seconds. When NOT to use streaming: (1) Very short responses (<100 tokens, <1 second generation) - overhead of streaming setup exceeds time saved; users do not perceive difference under 1 second anyway. (2) Responses requiring post-processing - if you need to validate, format, or modify the complete response before showing users (e.g., checking for PII, validating JSON structure), streaming forces either showing unvalidated content or buffering (negating streaming benefits). (3) Batch/background processing - if user is not waiting for response (async jobs, batch operations), streaming adds complexity without UX benefit. (4) Simple infrastructure - if you are on platforms that do not support long-lived connections (some serverless, simple hosting), non-streaming is easier. (5) Mobile apps with poor connectivity - streaming over unreliable connections can be worse than waiting for single response; connection drops mid-stream frustrate users. Best practice: Use streaming for interactive chat/generation features where users wait and watch (chat, code generation, writing assistance). Use non-streaming for API endpoints, background jobs, or very fast operations (<1s). Implement both and choose based on use case - many applications need both patterns.',
        keyPoints: [
            'Streaming is more complex but better UX for long responses',
            'Skip streaming for <1s responses - no perceived benefit',
            'Do not stream if post-processing required before display',
            'Infrastructure limitations may require non-streaming',
            'Choose based on use case - both patterns have valid uses'
        ]
    }
];

