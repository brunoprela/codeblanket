import { Module } from '../../types';

// Import sections
import { excelPowerUser } from '@/lib/content/sections/professional-tools/2-1-excel-power-user';
import { bloombergTerminal } from '@/lib/content/sections/professional-tools/2-2-bloomberg-terminal';
import { financialDataPlatforms } from '@/lib/content/sections/professional-tools/2-3-financial-data-platforms';
import { freeDataSources } from '@/lib/content/sections/professional-tools/2-4-free-data-sources';
import { chartingTechnicalAnalysis } from '@/lib/content/sections/professional-tools/2-5-charting-technical-analysis';
import { jupyterLabQuant } from '@/lib/content/sections/professional-tools/2-6-jupyter-lab-quant';
import { gitTradingStrategies } from '@/lib/content/sections/professional-tools/2-7-git-trading-strategies';
import { databasesFinancialData } from '@/lib/content/sections/professional-tools/2-8-databases-financial-data';
import { quantWorkstation } from '@/lib/content/sections/professional-tools/2-9-quant-workstation';
import { moduleProject } from '@/lib/content/sections/professional-tools/2-10-module-project';

// Import quizzes (discussion questions)
import { excelPowerUserQuiz } from '@/lib/content/quizzes/professional-tools/2-1-excel-power-user';
import { bloombergTerminalQuiz } from '@/lib/content/quizzes/professional-tools/2-2-bloomberg-terminal';
import { financialDataPlatformsQuiz } from '@/lib/content/quizzes/professional-tools/2-3-financial-data-platforms';
import { freeDataSourcesQuiz } from '@/lib/content/quizzes/professional-tools/2-4-free-data-sources';
import { chartingTechnicalAnalysisQuiz } from '@/lib/content/quizzes/professional-tools/2-5-charting-technical-analysis';
import { jupyterLabQuantQuiz } from '@/lib/content/quizzes/professional-tools/2-6-jupyter-lab-quant';
import { gitTradingStrategiesQuiz } from '@/lib/content/quizzes/professional-tools/2-7-git-trading-strategies';
import { databasesFinancialDataQuiz } from '@/lib/content/quizzes/professional-tools/2-8-databases-financial-data';
import { quantWorkstationQuiz } from '@/lib/content/quizzes/professional-tools/2-9-quant-workstation';
import { moduleProjectQuiz } from '@/lib/content/quizzes/professional-tools/2-10-module-project';

// Import multiple choice questions
import { excelPowerUserMultipleChoice } from '@/lib/content/multiple-choice/professional-tools/2-1-excel-power-user';
import { bloombergTerminalMultipleChoice } from '@/lib/content/multiple-choice/professional-tools/2-2-bloomberg-terminal';
import { financialDataPlatformsMultipleChoice } from '@/lib/content/multiple-choice/professional-tools/2-3-financial-data-platforms';
import { freeDataSourcesMultipleChoice } from '@/lib/content/multiple-choice/professional-tools/2-4-free-data-sources';
import { chartingTechnicalAnalysisMultipleChoice } from '@/lib/content/multiple-choice/professional-tools/2-5-charting-technical-analysis';
import { jupyterLabQuantMultipleChoice } from '@/lib/content/multiple-choice/professional-tools/2-6-jupyter-lab-quant';
import { gitTradingStrategiesMultipleChoice } from '@/lib/content/multiple-choice/professional-tools/2-7-git-trading-strategies';
import { databasesFinancialDataMultipleChoice } from '@/lib/content/multiple-choice/professional-tools/2-8-databases-financial-data';
import { quantWorkstationMultipleChoice } from '@/lib/content/multiple-choice/professional-tools/2-9-quant-workstation';
import { moduleProjectMultipleChoice } from '@/lib/content/multiple-choice/professional-tools/2-10-module-project';

// Transform functions
const transformQuiz = (quiz: any) => ({
  id: quiz.id,
  question: quiz.question,
  sampleAnswer: quiz.sampleAnswer,
  keyPoints: quiz.keyPoints || [],
});

const transformMC = (mc: any) => ({
  id: mc.id,
  question: mc.question,
  options: mc.options,
  correctAnswer: mc.correctAnswer,
  explanation: mc.explanation,
});

export const professionalToolsModule: Module = {
  id: 'professional-tools',
  title: 'Professional Tools & Technologies',
  description:
    'Master Excel, Bloomberg, data platforms, and build a professional quant workstation',
  sections: [
    {
      id: excelPowerUser.id,
      title: excelPowerUser.title,
      content: excelPowerUser.content,
      quiz: excelPowerUserQuiz.map(transformQuiz),
      multipleChoice: excelPowerUserMultipleChoice.map(transformMC),
    },
    {
      id: bloombergTerminal.id,
      title: bloombergTerminal.title,
      content: bloombergTerminal.content,
      quiz: bloombergTerminalQuiz.map(transformQuiz),
      multipleChoice: bloombergTerminalMultipleChoice.map(transformMC),
    },
    {
      id: financialDataPlatforms.id,
      title: financialDataPlatforms.title,
      content: financialDataPlatforms.content,
      quiz: financialDataPlatformsQuiz.map(transformQuiz),
      multipleChoice: financialDataPlatformsMultipleChoice.map(transformMC),
    },
    {
      id: freeDataSources.id,
      title: freeDataSources.title,
      content: freeDataSources.content,
      quiz: freeDataSourcesQuiz.map(transformQuiz),
      multipleChoice: freeDataSourcesMultipleChoice.map(transformMC),
    },
    {
      id: chartingTechnicalAnalysis.id,
      title: chartingTechnicalAnalysis.title,
      content: chartingTechnicalAnalysis.content,
      quiz: chartingTechnicalAnalysisQuiz.map(transformQuiz),
      multipleChoice: chartingTechnicalAnalysisMultipleChoice.map(transformMC),
    },
    {
      id: jupyterLabQuant.id,
      title: jupyterLabQuant.title,
      content: jupyterLabQuant.content,
      quiz: jupyterLabQuantQuiz.map(transformQuiz),
      multipleChoice: jupyterLabQuantMultipleChoice.map(transformMC),
    },
    {
      id: gitTradingStrategies.id,
      title: gitTradingStrategies.title,
      content: gitTradingStrategies.content,
      quiz: gitTradingStrategiesQuiz.map(transformQuiz),
      multipleChoice: gitTradingStrategiesMultipleChoice.map(transformMC),
    },
    {
      id: databasesFinancialData.id,
      title: databasesFinancialData.title,
      content: databasesFinancialData.content,
      quiz: databasesFinancialDataQuiz.map(transformQuiz),
      multipleChoice: databasesFinancialDataMultipleChoice.map(transformMC),
    },
    {
      id: quantWorkstation.id,
      title: quantWorkstation.title,
      content: quantWorkstation.content,
      quiz: quantWorkstationQuiz.map(transformQuiz),
      multipleChoice: quantWorkstationMultipleChoice.map(transformMC),
    },
    {
      id: moduleProject.id,
      title: moduleProject.title,
      content: moduleProject.content,
      quiz: moduleProjectQuiz.map(transformQuiz),
      multipleChoice: moduleProjectMultipleChoice.map(transformMC),
    },
  ],
};
