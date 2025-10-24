import { Module } from '@/lib/types';

const aiSafetyGuardrailsModule: Module = {
  id: 'applied-ai-safety',
  title: 'AI Safety & Guardrails',
  description:
    'Master building safe and responsible AI applications with comprehensive guardrails, content moderation, and compliance measures',
  icon: '🛡️',
  difficulty: 'intermediate',
  duration: '2 weeks',
  sections: [
    {
      id: 'ai-safety-fundamentals',
      title: 'AI Safety Fundamentals',
      description:
        'Understanding the critical foundations of AI safety, responsible AI principles, and why safety matters in production systems',
    },
    {
      id: 'content-moderation',
      title: 'Content Moderation',
      description:
        "Implementing robust content moderation systems using OpenAI's moderation API, custom filters, and toxicity detection",
    },
    {
      id: 'prompt-injection-defense',
      title: 'Prompt Injection Defense',
      description:
        'Defending against prompt injection attacks with validation strategies, instruction hierarchies, and anomaly detection',
    },
    {
      id: 'pii-detection-removal',
      title: 'PII Detection & Removal',
      description:
        'Detecting and removing personally identifiable information with pattern-based detection, NER, and GDPR compliance',
    },
    {
      id: 'hallucination-detection',
      title: 'Hallucination Detection',
      description:
        'Identifying and mitigating LLM hallucinations through confidence scoring, fact-checking, and validation techniques',
    },
    {
      id: 'output-validation-guardrails',
      title: 'Output Validation & Guardrails',
      description:
        'Implementing comprehensive output validation with schema checking, quality thresholds, and the Guardrails library',
    },
    {
      id: 'rate-limiting-safety',
      title: 'Rate Limiting for Safety',
      description:
        'Preventing abuse through intelligent rate limiting, suspicious pattern detection, and account management',
    },
    {
      id: 'bias-detection-mitigation',
      title: 'Bias Detection & Mitigation',
      description:
        'Measuring and mitigating bias in AI systems with fairness metrics, diverse testing, and continuous monitoring',
    },
    {
      id: 'audit-logging-compliance',
      title: 'Audit Logging & Compliance',
      description:
        'Building comprehensive audit systems for GDPR, CCPA, and SOC 2 compliance with proper data retention and reporting',
    },
    {
      id: 'building-safety-layer',
      title: 'Building a Safety Layer',
      description:
        'Architecting a complete safety layer with pre-processing, post-processing, human review, and incident response',
    },
  ],
  keyTakeaways: [
    'Safety is not optional - every production AI system needs comprehensive guardrails',
    'Content moderation must handle multiple levels: toxicity, NSFW, harmful content, and PII',
    'Prompt injection attacks are real and require defense-in-depth strategies',
    'PII detection requires both pattern-based and ML-based approaches for GDPR compliance',
    'Hallucination detection combines confidence scoring, fact-checking, and consistency validation',
    'Output validation should use schema enforcement and quality thresholds',
    'Rate limiting prevents abuse and suspicious pattern detection identifies bad actors',
    'Bias detection and mitigation must be continuous, not one-time activities',
    'Audit logging is critical for compliance and incident investigation',
    'A comprehensive safety layer requires pre-processing, post-processing, and human oversight',
  ],
  learningObjectives: [
    'Understand the fundamental principles of AI safety and responsible AI',
    'Implement multi-level content moderation systems',
    'Defend against prompt injection and adversarial attacks',
    'Detect and remove PII for GDPR/CCPA compliance',
    'Build hallucination detection and mitigation systems',
    'Create comprehensive output validation with guardrails',
    'Implement intelligent rate limiting for abuse prevention',
    'Measure and mitigate bias in AI outputs',
    'Build audit logging systems for compliance',
    'Architect a complete, production-ready safety layer',
  ],
  prerequisites: [
    'llm-engineering-fundamentals',
    'prompt-engineering-optimization',
    'production-llm-applications',
  ],
  practicalProjects: [
    {
      title: 'Content Moderation System',
      description:
        'Build a multi-level content moderation system with toxicity detection, NSFW filtering, and custom rules',
    },
    {
      title: 'PII Detection & Redaction Tool',
      description:
        'Create a comprehensive PII detection system that identifies and redacts sensitive information',
    },
    {
      title: 'Prompt Injection Defense Layer',
      description:
        'Implement a defense system that detects and blocks prompt injection attempts',
    },
    {
      title: 'Hallucination Detector',
      description:
        'Build a system that identifies likely hallucinations through confidence scoring and fact-checking',
    },
    {
      title: 'Complete Safety Layer',
      description:
        'Design and implement a production-ready safety layer with pre/post-processing and human review workflows',
    },
  ],
  productionExamples: [
    'How OpenAI implements content moderation in ChatGPT',
    "Anthropic's Constitutional AI approach to safety",
    "Google's responsible AI guardrails in Gemini",
    'Enterprise AI safety layers for regulated industries',
    'Financial services compliance in AI systems',
  ],
};

export default aiSafetyGuardrailsModule;
