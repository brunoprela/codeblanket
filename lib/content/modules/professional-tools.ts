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

// Helper to flatten Content sections into single content string
const flattenContent = (contentObj: any) => {
  if (contentObj.content) return contentObj.content;
  if (contentObj.sections && contentObj.sections.length > 0) {
    return contentObj.sections.map((s: any) => s.content).join('\n\n');
  }
  return '';
};

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
  icon: '🛠️',
  sections: [
    {
      id: excelPowerUser.id,
      title: excelPowerUser.title,
      content: flattenContent(excelPowerUser),
      quiz: excelPowerUserQuiz.map(transformQuiz),
      multipleChoice: excelPowerUserMultipleChoice.map(transformMC),
    },
    {
      id: bloombergTerminal.id,
      title: bloombergTerminal.title,
      content: flattenContent(bloombergTerminal),
      quiz: bloombergTerminalQuiz.map(transformQuiz),
      multipleChoice: bloombergTerminalMultipleChoice.map(transformMC),
    },
    {
      id: financialDataPlatforms.id,
      title: financialDataPlatforms.title,
      content: flattenContent(financialDataPlatforms),
      quiz: financialDataPlatformsQuiz.map(transformQuiz),
      multipleChoice: financialDataPlatformsMultipleChoice.map(transformMC),
    },
    {
      id: freeDataSources.id,
      title: freeDataSources.title,
      content: flattenContent(freeDataSources),
      quiz: freeDataSourcesQuiz.map(transformQuiz),
      multipleChoice: freeDataSourcesMultipleChoice.map(transformMC),
    },
    {
      id: chartingTechnicalAnalysis.id || '2-5-charting-technical-analysis',
      title: chartingTechnicalAnalysis.title,
      content: flattenContent(chartingTechnicalAnalysis),
      quiz: (chartingTechnicalAnalysisQuiz.questions || []).map(transformQuiz),
      multipleChoice: (
        chartingTechnicalAnalysisMultipleChoice.questions || []
      ).map(transformMC),
    },
    {
      id: jupyterLabQuant.id,
      title: jupyterLabQuant.title,
      content: flattenContent(jupyterLabQuant),
      quiz: (jupyterLabQuantQuiz.questions || []).map(transformQuiz),
      multipleChoice: (jupyterLabQuantMultipleChoice.questions || []).map(
        transformMC,
      ),
    },
    {
      id: gitTradingStrategies.id,
      title: gitTradingStrategies.title,
      content: flattenContent(gitTradingStrategies),
      quiz: (gitTradingStrategiesQuiz.questions || []).map(transformQuiz),
      multipleChoice: (gitTradingStrategiesMultipleChoice.questions || []).map(
        transformMC,
      ),
    },
    {
      id: databasesFinancialData.id,
      title: databasesFinancialData.title,
      content: flattenContent(databasesFinancialData),
      quiz: (databasesFinancialDataQuiz.questions || []).map(transformQuiz),
      multipleChoice: (
        databasesFinancialDataMultipleChoice.questions || []
      ).map(transformMC),
    },
    {
      id: quantWorkstation.id,
      title: quantWorkstation.title,
      content: flattenContent(quantWorkstation),
      quiz: (quantWorkstationQuiz.questions || []).map(transformQuiz),
      multipleChoice: (quantWorkstationMultipleChoice.questions || []).map(
        transformMC,
      ),
    },
    {
      id: moduleProject.id,
      title: moduleProject.title,
      content: flattenContent(moduleProject),
      quiz: (moduleProjectQuiz.questions || []).map(transformQuiz),
      multipleChoice: (moduleProjectMultipleChoice.questions || []).map(
        transformMC,
      ),
    },
  ],
};
