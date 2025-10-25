/**
 * Multiple choice questions for Output Parsing & Structured Data section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const outputparsingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the main advantage of using JSON mode over regex parsing?',
    options: [
      'JSON mode is faster',
      'JSON mode guarantees valid, parseable output',
      'JSON mode costs less',
      'JSON mode works with all models',
    ],
    correctAnswer: 1,
    explanation:
      'JSON mode forces the LLM to return valid JSON, eliminating parsing failures from malformed output. Regex parsing has 10-30% failure rates due to format variations, escaped characters, and nested structures. JSON mode reduces failures to <1%.',
  },
  {
    id: 'mc2',
    question: 'What does Pydantic provide for structured data extraction?',
    options: [
      'Faster API calls',
      'Type validation and automatic conversion',
      'Caching',
      'Streaming support',
    ],
    correctAnswer: 1,
    explanation:
      'Pydantic validates types (int vs string), formats (email, URL), and constraints (min/max values). It automatically converts types when possible (string "30" to int 30) and provides clear error messages for validation failures. This ensures data quality beyond just valid JSON.',
  },
  {
    id: 'mc3',
    question: 'How do you enable JSON mode in OpenAI API calls?',
    options: [
      'Set temperature=0',
      'Add response_format={"type": "json_object"}',
      'Use a different model',
      'JSON mode is always enabled',
    ],
    correctAnswer: 1,
    explanation:
      'Add response_format={"type": "json_object"} to the API call. You must also instruct the model to output JSON in the system message (e.g., "Return your response as JSON"). The API will then ensure syntactically valid JSON output.',
  },
  {
    id: 'mc4',
    question:
      'What should you do if JSON mode returns invalid JSON despite being enabled?',
    options: [
      'Give up immediately',
      'Try extracting JSON from markdown, then ask LLM to fix it',
      'Switch to regex parsing',
      'Report a bug',
    ],
    correctAnswer: 1,
    explanation:
      'First try common fixes: extract from ```json``` markdown blocks, find JSON between first { and last }. If still invalid, send the response back to the LLM asking it to fix the JSON with the parse error included. This recovers ~80% of failures.',
  },
  {
    id: 'mc5',
    question: 'What is the Instructor library used for?',
    options: [
      'Training models',
      'Simplified structured data extraction with automatic retries',
      'Prompt engineering',
      'Cost tracking',
    ],
    correctAnswer: 1,
    explanation:
      'Instructor simplifies structured extraction with one-line syntax: client.create (response_model=MyModel). It handles JSON mode, Pydantic validation, and automatic retries with feedback when validation fails. It abstracts away boilerplate but reduces control over retry logic.',
  },
];
