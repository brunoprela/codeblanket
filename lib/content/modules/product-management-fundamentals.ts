/**
 * Product Management Fundamentals Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { whatIsProductManagementSection } from '../sections/product-management-fundamentals/what-is-product-management';
import { pmMindsetSection } from '../sections/product-management-fundamentals/pm-mindset';
import { coreCompetenciesSection } from '../sections/product-management-fundamentals/core-competencies';
import { discoveryVsDeliverySection } from '../sections/product-management-fundamentals/discovery-vs-delivery';
import { pmToolkitSection } from '../sections/product-management-fundamentals/pm-toolkit';
import { workingWithEngineeringSection } from '../sections/product-management-fundamentals/working-with-engineering';
import { workingWithDesignSection } from '../sections/product-management-fundamentals/working-with-design';
import { workingWithSalesMarketingSection } from '../sections/product-management-fundamentals/working-with-sales-marketing';
import { writingForPMsSection } from '../sections/product-management-fundamentals/writing-for-pms';
import { pmCommunicationSection } from '../sections/product-management-fundamentals/pm-communication';
import { pmCareerDevelopmentSection } from '../sections/product-management-fundamentals/pm-career-development';
import { pmAtDifferentStagesSection } from '../sections/product-management-fundamentals/pm-at-different-stages';
import { pmInterviewPrepSection } from '../sections/product-management-fundamentals/pm-interview-prep';

// Import quizzes (discussion questions)
import { whatIsProductManagementQuiz } from '../quizzes/product-management-fundamentals/what-is-product-management';
import { pmMindsetQuiz } from '../quizzes/product-management-fundamentals/pm-mindset';
import { coreCompetenciesQuiz } from '../quizzes/product-management-fundamentals/core-competencies';
import { discoveryVsDeliveryQuiz } from '../quizzes/product-management-fundamentals/discovery-vs-delivery';
import { pmToolkitQuiz } from '../quizzes/product-management-fundamentals/pm-toolkit';
import { workingWithEngineeringQuiz } from '../quizzes/product-management-fundamentals/working-with-engineering';
import { workingWithDesignQuiz } from '../quizzes/product-management-fundamentals/working-with-design';
import { workingWithSalesMarketingQuiz } from '../quizzes/product-management-fundamentals/working-with-sales-marketing';
import { writingForPMsQuiz } from '../quizzes/product-management-fundamentals/writing-for-pms';
import { pmCommunicationQuiz } from '../quizzes/product-management-fundamentals/pm-communication';
import { pmCareerDevelopmentQuiz } from '../quizzes/product-management-fundamentals/pm-career-development';
import { pmAtDifferentStagesQuiz } from '../quizzes/product-management-fundamentals/pm-at-different-stages';
import { pmInterviewPrepQuiz } from '../quizzes/product-management-fundamentals/pm-interview-prep';

// Import multiple choice
import { whatIsProductManagementMultipleChoice } from '../multiple-choice/product-management-fundamentals/what-is-product-management';
import { pmMindsetMultipleChoice } from '../multiple-choice/product-management-fundamentals/pm-mindset';
import { coreCompetenciesMultipleChoice } from '../multiple-choice/product-management-fundamentals/core-competencies';
import { discoveryVsDeliveryMultipleChoice } from '../multiple-choice/product-management-fundamentals/discovery-vs-delivery';
import { pmToolkitMultipleChoice } from '../multiple-choice/product-management-fundamentals/pm-toolkit';
import { workingWithEngineeringMultipleChoice } from '../multiple-choice/product-management-fundamentals/working-with-engineering';
import { workingWithDesignMultipleChoice } from '../multiple-choice/product-management-fundamentals/working-with-design';
import { workingWithSalesMarketingMultipleChoice } from '../multiple-choice/product-management-fundamentals/working-with-sales-marketing';
import { writingForPMsMultipleChoice } from '../multiple-choice/product-management-fundamentals/writing-for-pms';
import { pmCommunicationMultipleChoice } from '../multiple-choice/product-management-fundamentals/pm-communication';
import { pmCareerDevelopmentMultipleChoice } from '../multiple-choice/product-management-fundamentals/pm-career-development';
import { pmAtDifferentStagesMultipleChoice } from '../multiple-choice/product-management-fundamentals/pm-at-different-stages';
import { pmInterviewPrepMultipleChoice } from '../multiple-choice/product-management-fundamentals/pm-interview-prep';

export const productManagementFundamentalsModule: Module = {
  id: 'product-management-fundamentals',
  title: 'Product Management Fundamentals',
  description:
    'Master the foundations of product management - the role, mindset, core competencies, and essential skills needed to become a successful product manager.',
  category: 'Product Management',
  difficulty: 'Beginner',
  estimatedTime: '2-3 weeks',
  prerequisites: [],
  icon: '🎯',
  keyTakeaways: [
    'Understand the product manager role and responsibilities',
    'Develop the product management mindset and mental models',
    'Master core PM competencies: strategic thinking, user empathy, technical fluency',
    'Navigate product discovery and delivery effectively',
    'Build strong cross-functional relationships with engineering, design, and GTM teams',
    'Communicate clearly and influence without authority',
    'Use the right tools to manage products efficiently',
    'Plan your product management career path',
    'Prepare for product management interviews',
  ],
  learningObjectives: [
    'Define what product managers do and how the role varies across companies',
    'Adopt the product manager mindset: customer obsession, data-driven decisions, bias for action',
    'Assess your PM competencies and create a development plan',
    'Balance product discovery (finding the right product) with product delivery (building it right)',
    'Select and use the essential PM toolkit',
    'Work effectively with engineering teams on technical decisions',
    'Collaborate with design on user experience and prototyping',
    'Partner with sales and marketing on go-to-market strategies',
    'Write clear product requirements, strategy documents, and updates',
    'Present to executives and stakeholders with confidence',
    'Navigate different company stages from startup to enterprise',
    'Excel in product management interviews',
  ],
  sections: [
    {
      ...whatIsProductManagementSection,
      quiz: whatIsProductManagementQuiz,
      multipleChoice: whatIsProductManagementMultipleChoice,
    },
    {
      ...pmMindsetSection,
      quiz: pmMindsetQuiz,
      multipleChoice: pmMindsetMultipleChoice,
    },
    {
      ...coreCompetenciesSection,
      quiz: coreCompetenciesQuiz,
      multipleChoice: coreCompetenciesMultipleChoice,
    },
    {
      ...discoveryVsDeliverySection,
      quiz: discoveryVsDeliveryQuiz,
      multipleChoice: discoveryVsDeliveryMultipleChoice,
    },
    {
      ...pmToolkitSection,
      quiz: pmToolkitQuiz,
      multipleChoice: pmToolkitMultipleChoice,
    },
    {
      ...workingWithEngineeringSection,
      quiz: workingWithEngineeringQuiz,
      multipleChoice: workingWithEngineeringMultipleChoice,
    },
    {
      ...workingWithDesignSection,
      quiz: workingWithDesignQuiz,
      multipleChoice: workingWithDesignMultipleChoice,
    },
    {
      ...workingWithSalesMarketingSection,
      quiz: workingWithSalesMarketingQuiz,
      multipleChoice: workingWithSalesMarketingMultipleChoice,
    },
    {
      ...writingForPMsSection,
      quiz: writingForPMsQuiz,
      multipleChoice: writingForPMsMultipleChoice,
    },
    {
      ...pmCommunicationSection,
      quiz: pmCommunicationQuiz,
      multipleChoice: pmCommunicationMultipleChoice,
    },
    {
      ...pmCareerDevelopmentSection,
      quiz: pmCareerDevelopmentQuiz,
      multipleChoice: pmCareerDevelopmentMultipleChoice,
    },
    {
      ...pmAtDifferentStagesSection,
      quiz: pmAtDifferentStagesQuiz,
      multipleChoice: pmAtDifferentStagesMultipleChoice,
    },
    {
      ...pmInterviewPrepSection,
      quiz: pmInterviewPrepQuiz,
      multipleChoice: pmInterviewPrepMultipleChoice,
    },
  ],
};
