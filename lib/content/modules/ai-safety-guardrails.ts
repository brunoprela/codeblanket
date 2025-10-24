/**
 * AI Safety & Guardrails Module
 * Master building safe and responsible AI applications
 */

import { Module } from '@/lib/types';

// Import section content strings
import { aiSafetyFundamentalsSection } from '../sections/ai-safety-guardrails/ai-safety-fundamentals';
import { contentModerationSection } from '../sections/ai-safety-guardrails/content-moderation';
import { promptInjectionDefenseSection } from '../sections/ai-safety-guardrails/prompt-injection-defense';
import { piiDetectionRemovalSection } from '../sections/ai-safety-guardrails/pii-detection-removal';
import { hallucinationDetectionSection } from '../sections/ai-safety-guardrails/hallucination-detection';
import { outputValidationGuardrailsSection } from '../sections/ai-safety-guardrails/output-validation-guardrails';
import { rateLimitingSafetySection } from '../sections/ai-safety-guardrails/rate-limiting-safety';
import { biasDetectionMitigationSection } from '../sections/ai-safety-guardrails/bias-detection-mitigation';
import { auditLoggingComplianceSection } from '../sections/ai-safety-guardrails/audit-logging-compliance';
import { buildingSafetyLayerSection } from '../sections/ai-safety-guardrails/building-safety-layer';

// Import quizzes
import { aisafetyfundamentalsQuiz } from '../quizzes/ai-safety-guardrails/ai-safety-fundamentals';
import { contentmoderationQuiz } from '../quizzes/ai-safety-guardrails/content-moderation';
import { promptinjectiondefenseQuiz } from '../quizzes/ai-safety-guardrails/prompt-injection-defense';
import { piidetectionremovalQuiz } from '../quizzes/ai-safety-guardrails/pii-detection-removal';
import { hallucinationdetectionQuiz } from '../quizzes/ai-safety-guardrails/hallucination-detection';
import { outputvalidationguardrailsQuiz } from '../quizzes/ai-safety-guardrails/output-validation-guardrails';
import { ratelimitingsafetyQuiz } from '../quizzes/ai-safety-guardrails/rate-limiting-safety';
import { biasdetectionmitigationQuiz } from '../quizzes/ai-safety-guardrails/bias-detection-mitigation';
import { auditloggingcomplianceQuiz } from '../quizzes/ai-safety-guardrails/audit-logging-compliance';
import { buildingsafetylayerQuiz } from '../quizzes/ai-safety-guardrails/building-safety-layer';

// Import multiple choice
import { aisafetyfundamentalsMultipleChoice } from '../multiple-choice/ai-safety-guardrails/ai-safety-fundamentals';
import { contentmoderationMultipleChoice } from '../multiple-choice/ai-safety-guardrails/content-moderation';
import { promptinjectiondefenseMultipleChoice } from '../multiple-choice/ai-safety-guardrails/prompt-injection-defense';
import { piidetectionremovalMultipleChoice } from '../multiple-choice/ai-safety-guardrails/pii-detection-removal';
import { hallucinationdetectionMultipleChoice } from '../multiple-choice/ai-safety-guardrails/hallucination-detection';
import { outputvalidationguardrailsMultipleChoice } from '../multiple-choice/ai-safety-guardrails/output-validation-guardrails';
import { ratelimitingsafetyMultipleChoice } from '../multiple-choice/ai-safety-guardrails/rate-limiting-safety';
import { biasdetectionmitigationMultipleChoice } from '../multiple-choice/ai-safety-guardrails/bias-detection-mitigation';
import { auditloggingcomplianceMultipleChoice } from '../multiple-choice/ai-safety-guardrails/audit-logging-compliance';
import { buildingsafetylayerMultipleChoice } from '../multiple-choice/ai-safety-guardrails/building-safety-layer';

const aiSafetyGuardrailsModule: Module = {
  id: 'applied-ai-safety',
  title: 'AI Safety & Guardrails',
  description:
    'Master building safe and responsible AI applications with comprehensive guardrails, content moderation, and compliance measures',
  icon: '🛡️',
  sections: [
    {
      id: 'ai-safety-fundamentals',
      title: 'AI Safety Fundamentals',
      content: aiSafetyFundamentalsSection,
      quiz: aisafetyfundamentalsQuiz,
      multipleChoice: aisafetyfundamentalsMultipleChoice,
    },
    {
      id: 'content-moderation',
      title: 'Content Moderation',
      content: contentModerationSection,
      quiz: contentmoderationQuiz,
      multipleChoice: contentmoderationMultipleChoice,
    },
    {
      id: 'prompt-injection-defense',
      title: 'Prompt Injection Defense',
      content: promptInjectionDefenseSection,
      quiz: promptinjectiondefenseQuiz,
      multipleChoice: promptinjectiondefenseMultipleChoice,
    },
    {
      id: 'pii-detection-removal',
      title: 'PII Detection & Removal',
      content: piiDetectionRemovalSection,
      quiz: piidetectionremovalQuiz,
      multipleChoice: piidetectionremovalMultipleChoice,
    },
    {
      id: 'hallucination-detection',
      title: 'Hallucination Detection',
      content: hallucinationDetectionSection,
      quiz: hallucinationdetectionQuiz,
      multipleChoice: hallucinationdetectionMultipleChoice,
    },
    {
      id: 'output-validation-guardrails',
      title: 'Output Validation & Guardrails',
      content: outputValidationGuardrailsSection,
      quiz: outputvalidationguardrailsQuiz,
      multipleChoice: outputvalidationguardrailsMultipleChoice,
    },
    {
      id: 'rate-limiting-safety',
      title: 'Rate Limiting for Safety',
      content: rateLimitingSafetySection,
      quiz: ratelimitingsafetyQuiz,
      multipleChoice: ratelimitingsafetyMultipleChoice,
    },
    {
      id: 'bias-detection-mitigation',
      title: 'Bias Detection & Mitigation',
      content: biasDetectionMitigationSection,
      quiz: biasdetectionmitigationQuiz,
      multipleChoice: biasdetectionmitigationMultipleChoice,
    },
    {
      id: 'audit-logging-compliance',
      title: 'Audit Logging & Compliance',
      content: auditLoggingComplianceSection,
      quiz: auditloggingcomplianceQuiz,
      multipleChoice: auditloggingcomplianceMultipleChoice,
    },
    {
      id: 'building-safety-layer',
      title: 'Building a Safety Layer',
      content: buildingSafetyLayerSection,
      quiz: buildingsafetylayerQuiz,
      multipleChoice: buildingsafetylayerMultipleChoice,
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
    'Bias detection and mitigation are ethical and legal requirements',
    'Comprehensive audit logging enables compliance and incident response',
    'A safety layer should integrate all guardrails with monitoring and alerting',
  ],
};

export default aiSafetyGuardrailsModule;
