/**
 * Multiple choice questions for Streaming Responses section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const streamingresponsesMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'mc1',
        question: 'What is the main UX benefit of streaming responses?',
        options: [
            'Responses are generated faster',
            'Users see output immediately (time to first token <1s)',
            'Responses are higher quality',
            'Costs are lower'
        ],
        correctAnswer: 1,
        explanation:
            'Streaming\'s main UX benefit is perceived speed - users see the first token in ~0.5s vs waiting 5-10s for the complete response. Total generation time is the same, but progressive disclosure dramatically improves perceived performance and engagement.'
    },
    {
        id: 'mc2',
        question: 'How do you enable streaming in OpenAI API calls?',
        options: [
            'Set temperature=0',
            'Add stream=True parameter',
            'Use a different model',
            'Streaming is always enabled'
        ],
        correctAnswer: 1,
        explanation:
            'Add stream=True to the API call parameters. This changes the response from a single complete object to an iterator/stream of chunks. You must then iterate over the chunks to collect the full response.'
    },
    {
        id: 'mc3',
        question: 'What information is in the final chunk of a streamed response?',
        options: [
            'The complete response text',
            'The finish_reason (e.g., "stop" or "length")',
            'Only token count',
            'Nothing, stream just closes'
        ],
        correctAnswer: 1,
        explanation:
            'The final chunk contains finish_reason indicating why generation stopped ("stop" for natural completion, "length" for hitting max_tokens, "content_filter" for policy violation). Earlier chunks contain content; the final chunk signals completion.'
    },
    {
        id: 'mc4',
        question: 'When should you NOT use streaming?',
        options: [
            'For chat applications',
            'For very short responses (<1 second)',
            'For code generation',
            'Streaming should always be used'
        ],
        correctAnswer: 1,
        explanation:
            'Skip streaming for very short responses (<1s total time) where the streaming overhead exceeds the benefit. Also skip if you need to post-process the complete response (validate, transform) before showing users. For most interactive use cases, streaming improves UX significantly.'
    },
    {
        id: 'mc5',
        question: 'What is Server-Sent Events (SSE)?',
        options: [
            'A security protocol',
            'The HTTP protocol used for streaming LLM responses',
            'A caching mechanism',
            'A token counting library'
        ],
        correctAnswer: 1,
        explanation:
            'Server-Sent Events (SSE) is the HTTP protocol used for streaming - the server keeps the connection open and sends data chunks as they become available. Each chunk is prefixed with "data:" and the stream ends with a "[DONE]" message.'
    }
];

