/**
 * File Processing & Document Understanding Module
 * Module 3: Applied AI Engineering Curriculum
 */

import { Module } from '../../types';

// Import sections
import { filesystemoperationspathhandlingSection } from '../sections/file-processing-document-understanding/file-system-operations-path-handling';
import { textfileprocessingSection } from '../sections/file-processing-document-understanding/text-file-processing';
import { excelfilemanipulationSection } from '../sections/file-processing-document-understanding/excel-file-manipulation';
import { exceladvancedoperationsSection } from '../sections/file-processing-document-understanding/excel-advanced-operations';
import { pdfprocessingextractionSection } from '../sections/file-processing-document-understanding/pdf-processing-extraction';
import { worddocumentprocessingSection } from '../sections/file-processing-document-understanding/word-document-processing';
import { imageprocessingllmsSection } from '../sections/file-processing-document-understanding/image-processing-llms';
import { csvstructureddataSection } from '../sections/file-processing-document-understanding/csv-structured-data';
import { binaryfilehandlingSection } from '../sections/file-processing-document-understanding/binary-file-handling';
import { markdownrichtextSection } from '../sections/file-processing-document-understanding/markdown-rich-text';
import { diffgenerationpatchapplicationSection } from '../sections/file-processing-document-understanding/diff-generation-patch-application';
import { fileembeddingsemanticsearchSection } from '../sections/file-processing-document-understanding/file-embedding-semantic-search';
import { unstructuredlibrarydeepdiveSection } from '../sections/file-processing-document-understanding/unstructured-library-deep-dive';
import { buildinguniversalfileeditorSection } from '../sections/file-processing-document-understanding/building-universal-file-editor';

// Import quizzes
import { filesystemoperationspathhandlingQuiz } from '../quizzes/file-processing-document-understanding/file-system-operations-path-handling';
import { textfileprocessingQuiz } from '../quizzes/file-processing-document-understanding/text-file-processing';
import { excelfilemanipulationQuiz } from '../quizzes/file-processing-document-understanding/excel-file-manipulation';
import { exceladvancedoperationsQuiz } from '../quizzes/file-processing-document-understanding/excel-advanced-operations';
import { pdfprocessingextractionQuiz } from '../quizzes/file-processing-document-understanding/pdf-processing-extraction';
import { worddocumentprocessingQuiz } from '../quizzes/file-processing-document-understanding/word-document-processing';
import { imageprocessingllmsQuiz } from '../quizzes/file-processing-document-understanding/image-processing-llms';
import { csvstructureddataQuiz } from '../quizzes/file-processing-document-understanding/csv-structured-data';
import { binaryfilehandlingQuiz } from '../quizzes/file-processing-document-understanding/binary-file-handling';
import { markdownrichtextQuiz } from '../quizzes/file-processing-document-understanding/markdown-rich-text';
import { diffgenerationpatchapplicationQuiz } from '../quizzes/file-processing-document-understanding/diff-generation-patch-application';
import { fileembeddingsemanticsearchQuiz } from '../quizzes/file-processing-document-understanding/file-embedding-semantic-search';
import { unstructuredlibrarydeepdiveQuiz } from '../quizzes/file-processing-document-understanding/unstructured-library-deep-dive';
import { buildinguniversalfileeditorQuiz } from '../quizzes/file-processing-document-understanding/building-universal-file-editor';

// Import multiple choice
import { filesystemoperationspathhandlingMultipleChoice } from '../multiple-choice/file-processing-document-understanding/file-system-operations-path-handling';
import { textfileprocessingMultipleChoice } from '../multiple-choice/file-processing-document-understanding/text-file-processing';
import { excelfilemanipulationMultipleChoice } from '../multiple-choice/file-processing-document-understanding/excel-file-manipulation';
import { exceladvancedoperationsMultipleChoice } from '../multiple-choice/file-processing-document-understanding/excel-advanced-operations';
import { pdfprocessingextractionMultipleChoice } from '../multiple-choice/file-processing-document-understanding/pdf-processing-extraction';
import { worddocumentprocessingMultipleChoice } from '../multiple-choice/file-processing-document-understanding/word-document-processing';
import { imageprocessingllmsMultipleChoice } from '../multiple-choice/file-processing-document-understanding/image-processing-llms';
import { csvstructureddataMultipleChoice } from '../multiple-choice/file-processing-document-understanding/csv-structured-data';
import { binaryfilehandlingMultipleChoice } from '../multiple-choice/file-processing-document-understanding/binary-file-handling';
import { markdownrichtextMultipleChoice } from '../multiple-choice/file-processing-document-understanding/markdown-rich-text';
import { diffgenerationpatchapplicationMultipleChoice } from '../multiple-choice/file-processing-document-understanding/diff-generation-patch-application';
import { fileembeddingsemanticsearchMultipleChoice } from '../multiple-choice/file-processing-document-understanding/file-embedding-semantic-search';
import { unstructuredlibrarydeepdiveMultipleChoice } from '../multiple-choice/file-processing-document-understanding/unstructured-library-deep-dive';
import { buildinguniversalfileeditorMultipleChoice } from '../multiple-choice/file-processing-document-understanding/building-universal-file-editor';

export const fileProcessingDocumentUnderstandingModule: Module = {
  id: 'applied-ai-file-processing',
  title: 'File Processing & Document Understanding',
  description:
    'Master parsing, understanding, and manipulating any file type with LLMs - from Excel to PDFs to code files. Build production document processing systems.',
  category: 'Applied AI',
  difficulty: 'Intermediate',
  estimatedTime: '3 weeks',
  prerequisites: ['python-fundamentals', 'python-intermediate'],
  icon: '📄',
  keyTakeaways: [
    'Process any file type with appropriate libraries',
    'Extract structured data from documents',
    'Generate and apply diffs for code editing',
    'Implement semantic search across files',
    'Build safe file modification systems',
    'Integrate LLMs with file processing',
    'Create production document pipelines',
    'Handle binary and complex formats',
  ],
  learningObjectives: [
    'Master file system operations and path handling',
    'Process text files efficiently with chunking and diffs',
    'Manipulate Excel files with formulas and formatting',
    'Extract data from PDFs with OCR and tables',
    'Generate and modify Word documents',
    'Process images with OCR and vision LLMs',
    'Handle CSV, JSON, and structured data',
    'Work with binary files and archives',
    'Generate markdown and rich text',
    'Implement diff generation and patch application',
    'Build semantic search with file embeddings',
    'Use Unstructured library for universal processing',
    'Create a complete universal file editor',
  ],
  sections: [
    {
      ...filesystemoperationspathhandlingSection,
      quiz: filesystemoperationspathhandlingQuiz,
      multipleChoice: filesystemoperationspathhandlingMultipleChoice,
    },
    {
      ...textfileprocessingSection,
      quiz: textfileprocessingQuiz,
      multipleChoice: textfileprocessingMultipleChoice,
    },
    {
      ...excelfilemanipulationSection,
      quiz: excelfilemanipulationQuiz,
      multipleChoice: excelfilemanipulationMultipleChoice,
    },
    {
      ...exceladvancedoperationsSection,
      quiz: exceladvancedoperationsQuiz,
      multipleChoice: exceladvancedoperationsMultipleChoice,
    },
    {
      ...pdfprocessingextractionSection,
      quiz: pdfprocessingextractionQuiz,
      multipleChoice: pdfprocessingextractionMultipleChoice,
    },
    {
      ...worddocumentprocessingSection,
      quiz: worddocumentprocessingQuiz,
      multipleChoice: worddocumentprocessingMultipleChoice,
    },
    {
      ...imageprocessingllmsSection,
      quiz: imageprocessingllmsQuiz,
      multipleChoice: imageprocessingllmsMultipleChoice,
    },
    {
      ...csvstructureddataSection,
      quiz: csvstructureddataQuiz,
      multipleChoice: csvstructureddataMultipleChoice,
    },
    {
      ...binaryfilehandlingSection,
      quiz: binaryfilehandlingQuiz,
      multipleChoice: binaryfilehandlingMultipleChoice,
    },
    {
      ...markdownrichtextSection,
      quiz: markdownrichtextQuiz,
      multipleChoice: markdownrichtextMultipleChoice,
    },
    {
      ...diffgenerationpatchapplicationSection,
      quiz: diffgenerationpatchapplicationQuiz,
      multipleChoice: diffgenerationpatchapplicationMultipleChoice,
    },
    {
      ...fileembeddingsemanticsearchSection,
      quiz: fileembeddingsemanticsearchQuiz,
      multipleChoice: fileembeddingsemanticsearchMultipleChoice,
    },
    {
      ...unstructuredlibrarydeepdiveSection,
      quiz: unstructuredlibrarydeepdiveQuiz,
      multipleChoice: unstructuredlibrarydeepdiveMultipleChoice,
    },
    {
      ...buildinguniversalfileeditorSection,
      quiz: buildinguniversalfileeditorQuiz,
      multipleChoice: buildinguniversalfileeditorMultipleChoice,
    },
  ],
};
