import { Module } from '../../types';
import { backtestingStrategyDevelopment } from '../finance/10-backtesting-strategy-development/index';

// Helper to flatten Content sections into single content string
const flattenContent = (contentObj: any) => {
  if (contentObj.content) return contentObj.content;
  if (contentObj.sections && contentObj.sections.length > 0) {
    return contentObj.sections.map((s: any) => s.content).join('\n\n');
  }
  return '';
};

// Transform function for multiple choice
const transformMC = (mc: any) => ({
  id: mc.id,
  question: mc.question,
  options: mc.options,
  correctAnswer: mc.correctAnswer,
  explanation: mc.explanation,
  difficulty: mc.difficulty,
});

// Transform function for discussion questions
const transformDiscussion = (section: any) => {
  // Extract discussion content from sections
  if (section.sections && section.sections.length > 0) {
    // Discussion files have the questions embedded in the content
    return flattenContent(section);
  }
  return '';
};

export const backtestingStrategyDevelopmentModule: Module = {
  id: 'backtesting-strategy-development',
  title: 'Backtesting & Strategy Development',
  description:
    'Master the complete backtesting lifecycle from data management through production deployment, including walk-forward analysis, parameter optimization, and building production-grade backtesting infrastructure',
  icon: '📊',
  sections: backtestingStrategyDevelopment.sections.map((section: any) => ({
    id: section.id,
    title: section.title,
    content: flattenContent(section),
    quiz: section.quiz || [],
    multipleChoice: (section.quiz || []).map(transformMC),
    discussion: section.discussion ? flattenContent(section.discussion) : '',
  })),
  quizzes: backtestingStrategyDevelopment.sections.flatMap((section: any) =>
    (section.quiz || []).map(transformMC),
  ),
  multipleChoiceQuestions: backtestingStrategyDevelopment.sections.flatMap(
    (section: any) => (section.quiz || []).map(transformMC),
  ),
};
