/**
 * Code Understanding & AST Manipulation Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import astfundamentalsSection from '../sections/code-understanding-ast-manipulation/ast-fundamentals';
import pythoncodeanalysisSection from '../sections/code-understanding-ast-manipulation/python-code-analysis';
import treesitterparsingSection from '../sections/code-understanding-ast-manipulation/tree-sitter-parsing';
import codestructureanalysisSection from '../sections/code-understanding-ast-manipulation/code-structure-analysis';
import symbolresolutionSection from '../sections/code-understanding-ast-manipulation/symbol-resolution';
import codemodificationastSection from '../sections/code-understanding-ast-manipulation/code-modification-ast';
import staticanalysisSection from '../sections/code-understanding-ast-manipulation/static-analysis';
import typesystemunderstandingSection from '../sections/code-understanding-ast-manipulation/type-system-understanding';
import documentationextractionSection from '../sections/code-understanding-ast-manipulation/documentation-extraction';
import codesimilaritydetectionSection from '../sections/code-understanding-ast-manipulation/code-similarity-detection';
import languageserverprotocolSection from '../sections/code-understanding-ast-manipulation/language-server-protocol';
import buildingcodeunderstandingengineSection from '../sections/code-understanding-ast-manipulation/building-code-understanding-engine';

// Import quizzes
import { astfundamentalsQuiz } from '../quizzes/code-understanding-ast-manipulation/ast-fundamentals';
import { pythoncodeanalysisQuiz } from '../quizzes/code-understanding-ast-manipulation/python-code-analysis';
import { treesitterparsingQuiz } from '../quizzes/code-understanding-ast-manipulation/tree-sitter-parsing';
import { codestructureanalysisQuiz } from '../quizzes/code-understanding-ast-manipulation/code-structure-analysis';
import { symbolresolutionQuiz } from '../quizzes/code-understanding-ast-manipulation/symbol-resolution';
import { codemodificationastQuiz } from '../quizzes/code-understanding-ast-manipulation/code-modification-ast';
import { staticanalysisQuiz } from '../quizzes/code-understanding-ast-manipulation/static-analysis';
import { typesystemunderstandingQuiz } from '../quizzes/code-understanding-ast-manipulation/type-system-understanding';
import { documentationextractionQuiz } from '../quizzes/code-understanding-ast-manipulation/documentation-extraction';
import { codesimilaritydetectionQuiz } from '../quizzes/code-understanding-ast-manipulation/code-similarity-detection';
import { languageserverprotocolQuiz } from '../quizzes/code-understanding-ast-manipulation/language-server-protocol';
import { buildingcodeunderstandingengineQuiz } from '../quizzes/code-understanding-ast-manipulation/building-code-understanding-engine';

// Import multiple choice
import { astfundamentalsMultipleChoice } from '../multiple-choice/code-understanding-ast-manipulation/ast-fundamentals';
import { pythoncodeanalysisMultipleChoice } from '../multiple-choice/code-understanding-ast-manipulation/python-code-analysis';
import { treesitterparsingMultipleChoice } from '../multiple-choice/code-understanding-ast-manipulation/tree-sitter-parsing';
import { codestructureanalysisMultipleChoice } from '../multiple-choice/code-understanding-ast-manipulation/code-structure-analysis';
import { symbolresolutionMultipleChoice } from '../multiple-choice/code-understanding-ast-manipulation/symbol-resolution';
import { codemodificationastMultipleChoice } from '../multiple-choice/code-understanding-ast-manipulation/code-modification-ast';
import { staticanalysisMultipleChoice } from '../multiple-choice/code-understanding-ast-manipulation/static-analysis';
import { typesystemunderstandingMultipleChoice } from '../multiple-choice/code-understanding-ast-manipulation/type-system-understanding';
import { documentationextractionMultipleChoice } from '../multiple-choice/code-understanding-ast-manipulation/documentation-extraction';
import { codesimilaritydetectionMultipleChoice } from '../multiple-choice/code-understanding-ast-manipulation/code-similarity-detection';
import { languageserverprotocolMultipleChoice } from '../multiple-choice/code-understanding-ast-manipulation/language-server-protocol';
import { buildingcodeunderstandingengineMultipleChoice } from '../multiple-choice/code-understanding-ast-manipulation/building-code-understanding-engine';

export const codeUnderstandingAstManipulationModule: Module = {
    id: 'applied-ai-code-understanding',
    title: 'Code Understanding & AST Manipulation',
    description:
        'Master Abstract Syntax Trees (AST) and build powerful code analysis tools: from parsing and traversal to building language servers and complete code understanding engines.',
    category: 'Applied AI',
    difficulty: 'Advanced',
    estimatedTime: '14 hours',
    prerequisites: ['Python fundamentals', 'Data structures', 'Basic compiler concepts'],
    icon: '🌳',
    keyTakeaways: [
        'Understand AST structure and traversal patterns',
        'Parse and analyze Python code programmatically',
        'Use tree-sitter for multi-language parsing',
        'Analyze code structure and extract metrics',
        'Resolve symbols and track references across files',
        'Modify code safely using AST transformations',
        'Perform static analysis and detect code quality issues',
        'Work with type systems and type inference',
        'Extract documentation from code automatically',
        'Detect code clones and measure similarity',
        'Understand LSP architecture and implementation',
        'Build production-ready code understanding engines',
    ],
    learningObjectives: [
        'Parse code into Abstract Syntax Trees using ast module',
        'Traverse and analyze AST nodes with visitors',
        'Use tree-sitter for error-tolerant, incremental parsing',
        'Extract code metrics like cyclomatic complexity and call graphs',
        'Build symbol tables and resolve references',
        'Perform scope analysis and handle imports',
        'Transform ASTs to modify code programmatically',
        'Implement codemods for large-scale refactoring',
        'Build static analyzers to detect bugs and code smells',
        'Perform data flow and control flow analysis',
        'Understand type annotations and type inference',
        'Extract function signatures and docstrings',
        'Detect code duplication using AST comparison',
        'Implement LSP servers for language support',
        'Build semantic indexes for fast code queries',
        'Design complete code understanding engines',
    ],
    sections: [
        {
            ...astfundamentalsSection,
            quiz: astfundamentalsQuiz,
            multipleChoice: astfundamentalsMultipleChoice,
        },
        {
            ...pythoncodeanalysisSection,
            quiz: pythoncodeanalysisQuiz,
            multipleChoice: pythoncodeanalysisMultipleChoice,
        },
        {
            ...treesitterparsingSection,
            quiz: treesitterparsingQuiz,
            multipleChoice: treesitterparsingMultipleChoice,
        },
        {
            ...codestructureanalysisSection,
            quiz: codestructureanalysisQuiz,
            multipleChoice: codestructureanalysisMultipleChoice,
        },
        {
            ...symbolresolutionSection,
            quiz: symbolresolutionQuiz,
            multipleChoice: symbolresolutionMultipleChoice,
        },
        {
            ...codemodificationastSection,
            quiz: codemodificationastQuiz,
            multipleChoice: codemodificationastMultipleChoice,
        },
        {
            ...staticanalysisSection,
            quiz: staticanalysisQuiz,
            multipleChoice: staticanalysisMultipleChoice,
        },
        {
            ...typesystemunderstandingSection,
            quiz: typesystemunderstandingQuiz,
            multipleChoice: typesystemunderstandingMultipleChoice,
        },
        {
            ...documentationextractionSection,
            quiz: documentationextractionQuiz,
            multipleChoice: documentationextractionMultipleChoice,
        },
        {
            ...codesimilaritydetectionSection,
            quiz: codesimilaritydetectionQuiz,
            multipleChoice: codesimilaritydetectionMultipleChoice,
        },
        {
            ...languageserverprotocolSection,
            quiz: languageserverprotocolQuiz,
            multipleChoice: languageserverprotocolMultipleChoice,
        },
        {
            ...buildingcodeunderstandingengineSection,
            quiz: buildingcodeunderstandingengineQuiz,
            multipleChoice: buildingcodeunderstandingengineMultipleChoice,
        },
    ],
};

