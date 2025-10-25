import { Module } from '../../types';

// Import sections
import { fundamentals } from '@/lib/content/sections/financial-statements-analysis/fundamentals';
import { incomeStatement } from '@/lib/content/sections/financial-statements-analysis/income-statement';
import { balanceSheet } from '@/lib/content/sections/financial-statements-analysis/balance-sheet';
import { cashFlow } from '@/lib/content/sections/financial-statements-analysis/cash-flow';
import { ratios } from '@/lib/content/sections/financial-statements-analysis/ratios';
import { edgar } from '@/lib/content/sections/financial-statements-analysis/edgar';
import { quality } from '@/lib/content/sections/financial-statements-analysis/quality';
import { credit } from '@/lib/content/sections/financial-statements-analysis/credit';
import { peers } from '@/lib/content/sections/financial-statements-analysis/peers';
import { pipeline } from '@/lib/content/sections/financial-statements-analysis/pipeline';
import { nlp } from '@/lib/content/sections/financial-statements-analysis/nlp';
import { project } from '@/lib/content/sections/financial-statements-analysis/project';

// Import quizzes (discussion questions)
import { fundamentalsDiscussionQuestions } from '@/lib/content/quizzes/financial-statements-analysis/fundamentals';
import { incomeStatementDiscussionQuestions } from '@/lib/content/quizzes/financial-statements-analysis/income-statement';
import { balanceSheetDiscussionQuestions } from '@/lib/content/quizzes/financial-statements-analysis/balance-sheet';
import { cashFlowDiscussionQuestions } from '@/lib/content/quizzes/financial-statements-analysis/cash-flow';
import { ratiosDiscussionQuestions } from '@/lib/content/quizzes/financial-statements-analysis/ratios';
import { edgarDiscussionQuestions } from '@/lib/content/quizzes/financial-statements-analysis/edgar';
import { qualityDiscussionQuestions } from '@/lib/content/quizzes/financial-statements-analysis/quality';
import { creditDiscussionQuestions } from '@/lib/content/quizzes/financial-statements-analysis/credit';
import { peersDiscussionQuestions } from '@/lib/content/quizzes/financial-statements-analysis/peers';
import { pipelineDiscussionQuestions } from '@/lib/content/quizzes/financial-statements-analysis/pipeline';
import { nlpDiscussionQuestions } from '@/lib/content/quizzes/financial-statements-analysis/nlp';
import { projectDiscussionQuestions } from '@/lib/content/quizzes/financial-statements-analysis/project';

// Import multiple choice questions
import { fundamentalsMultipleChoiceQuestions } from '@/lib/content/multiple-choice/financial-statements-analysis/fundamentals';
import { incomeStatementMultipleChoiceQuestions } from '@/lib/content/multiple-choice/financial-statements-analysis/income-statement';
import { balanceSheetMultipleChoiceQuestions } from '@/lib/content/multiple-choice/financial-statements-analysis/balance-sheet';
import { cashFlowMultipleChoiceQuestions } from '@/lib/content/multiple-choice/financial-statements-analysis/cash-flow';
import { ratiosMultipleChoiceQuestions } from '@/lib/content/multiple-choice/financial-statements-analysis/ratios';
import { edgarMultipleChoiceQuestions } from '@/lib/content/multiple-choice/financial-statements-analysis/edgar';
import { qualityMultipleChoiceQuestions } from '@/lib/content/multiple-choice/financial-statements-analysis/quality';
import { creditMultipleChoiceQuestions } from '@/lib/content/multiple-choice/financial-statements-analysis/credit';
import { peersMultipleChoiceQuestions } from '@/lib/content/multiple-choice/financial-statements-analysis/peers';
import { pipelineMultipleChoiceQuestions } from '@/lib/content/multiple-choice/financial-statements-analysis/pipeline';
import { nlpMultipleChoiceQuestions } from '@/lib/content/multiple-choice/financial-statements-analysis/nlp';
import { projectMultipleChoiceQuestions } from '@/lib/content/multiple-choice/financial-statements-analysis/project';

export const financialStatementsAnalysisModule: Module = {
  id: 'financial-statements-analysis',
  title: 'Financial Statements & Analysis',
  description:
    'Master reading and analyzing financial statements programmatically - build automated systems to parse and analyze company financials',
  sections: [
    {
      id: fundamentals.slug,
      title: fundamentals.title,
      content: fundamentals.content,
      discussionQuestions: fundamentalsDiscussionQuestions,
      multipleChoiceQuestions: fundamentalsMultipleChoiceQuestions,
    },
    {
      id: incomeStatement.slug,
      title: incomeStatement.title,
      content: incomeStatement.content,
      discussionQuestions: incomeStatementDiscussionQuestions,
      multipleChoiceQuestions: incomeStatementMultipleChoiceQuestions,
    },
    {
      id: balanceSheet.slug,
      title: balanceSheet.title,
      content: balanceSheet.content,
      discussionQuestions: balanceSheetDiscussionQuestions,
      multipleChoiceQuestions: balanceSheetMultipleChoiceQuestions,
    },
    {
      id: cashFlow.slug,
      title: cashFlow.title,
      content: cashFlow.content,
      discussionQuestions: cashFlowDiscussionQuestions,
      multipleChoiceQuestions: cashFlowMultipleChoiceQuestions,
    },
    {
      id: ratios.slug,
      title: ratios.title,
      content: ratios.content,
      discussionQuestions: ratiosDiscussionQuestions,
      multipleChoiceQuestions: ratiosMultipleChoiceQuestions,
    },
    {
      id: edgar.slug,
      title: edgar.title,
      content: edgar.content,
      discussionQuestions: edgarDiscussionQuestions,
      multipleChoiceQuestions: edgarMultipleChoiceQuestions,
    },
    {
      id: quality.slug,
      title: quality.title,
      content: quality.content,
      discussionQuestions: qualityDiscussionQuestions,
      multipleChoiceQuestions: qualityMultipleChoiceQuestions,
    },
    {
      id: credit.slug,
      title: credit.title,
      content: credit.content,
      discussionQuestions: creditDiscussionQuestions,
      multipleChoiceQuestions: creditMultipleChoiceQuestions,
    },
    {
      id: peers.slug,
      title: peers.title,
      content: peers.content,
      discussionQuestions: peersDiscussionQuestions,
      multipleChoiceQuestions: peersMultipleChoiceQuestions,
    },
    {
      id: pipeline.slug,
      title: pipeline.title,
      content: pipeline.content,
      discussionQuestions: pipelineDiscussionQuestions,
      multipleChoiceQuestions: pipelineMultipleChoiceQuestions,
    },
    {
      id: nlp.slug,
      title: nlp.title,
      content: nlp.content,
      discussionQuestions: nlpDiscussionQuestions,
      multipleChoiceQuestions: nlpMultipleChoiceQuestions,
    },
    {
      id: project.slug,
      title: project.title,
      content: project.content,
      discussionQuestions: projectDiscussionQuestions,
      multipleChoiceQuestions: projectMultipleChoiceQuestions,
    },
  ],
};
