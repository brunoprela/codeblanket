/**
 * Multiple choice questions for Local LLM Deployment section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const localllmdeploymentMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'mc1',
        question: 'At what monthly API cost does local LLM deployment typically become cost-effective?',
        options: [
            '$100/month',
            '$500/month',
            '$2,000+/month',
            'Never cost-effective'
        ],
        correctAnswer: 2,
        explanation:
            'Local deployment becomes cost-effective at $2,000+/month API costs. At this scale, GPU hardware ($3K) and electricity ($150/month) pay for themselves quickly. At lower volumes ($100-500/month), API is cheaper when factoring in infrastructure and maintenance costs.'
    },
    {
        id: 'mc2',
        question: 'What does 4-bit quantization do?',
        options: [
            'Reduces model size and memory 4x with 90-95% quality retention',
            'Reduces quality by 4%',
            'Increases speed 4x',
            'Divides cost by 4'
        ],
        correctAnswer: 0,
        explanation:
            '4-bit quantization reduces model size from 16GB to 4GB (4x smaller) by compressing weights from 16-bit to 4-bit precision. This enables running models on consumer GPUs while retaining 90-95% of full precision quality - excellent trade-off for most applications.'
    },
    {
        id: 'mc3',
        question: 'Which tool is recommended for serving local LLMs in production?',
        options: [
            'Loading models directly in Python',
            'vLLM or Ollama for optimized serving',
            'Running in Jupyter notebooks',
            'Excel'
        ],
        correctAnswer: 1,
        explanation:
            'Use specialized serving tools like vLLM or Ollama in production. They provide: optimized inference (2-3x faster), request queuing, batching, model persistence, and APIs. Direct Python loading lacks these production features and is 2-3x slower.'
    },
    {
        id: 'mc4',
        question: 'What GPU memory is needed to run a quantized 8B parameter model?',
        options: [
            '2GB',
            '8-16GB',
            '32GB',
            '64GB+'
        ],
        correctAnswer: 1,
        explanation:
            'A 4-bit quantized 8B parameter model needs ~5GB VRAM, so 8-16GB GPUs (RTX 3080, RTX 4070) work well. Full precision needs ~16GB. Larger models (70B) need 32GB+ even quantized. Match your GPU to your model size requirements.'
    },
    {
        id: 'mc5',
        question: 'When should you choose local LLM deployment over API?',
        options: [
            'Always go local to save money',
            'When you have high volume, acceptable quality from open models, and technical capacity',
            'Never, APIs are always better',
            'Only for offline applications'
        ],
        correctAnswer: 1,
        explanation:
            'Choose local when: (1) High volume (>$2K/month API cost) justifies investment, (2) Open models (Llama 3, Mistral) meet quality requirements, (3) Team has technical capacity for infrastructure. Otherwise, APIs provide better ROI. Many applications use hybrid: local for simple, API for complex.'
    }
];

