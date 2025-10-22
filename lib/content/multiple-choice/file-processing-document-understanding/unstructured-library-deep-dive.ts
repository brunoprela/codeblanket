/**
 * Multiple choice questions for Unstructured Library Deep Dive section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const unstructuredlibrarydeepdiveMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'fpdu-unstructured-mc-1',
        question: 'What is the main advantage of the Unstructured library?',
        options: ['It is faster than other libraries', 'It provides unified API for processing any document type', 'It is free and open source', 'It has better table extraction'],
        correctAnswer: 1,
        explanation: 'Unstructured provides a unified API that can automatically detect and process any file type (PDF, Word, HTML, etc.) without switching libraries.',
    },
    {
        id: 'fpdu-unstructured-mc-2',
        question: 'What does partition_auto() do?',
        options: ['Splits documents into parts', 'Automatically detects file type and processes it', 'Creates table partitions', 'Partitions disk space'],
        correctAnswer: 1,
        explanation: 'partition_auto() automatically detects the file type and applies the appropriate processing strategy without manual configuration.',
    },
    {
        id: 'fpdu-unstructured-mc-3',
        question: 'What is the "hi_res" strategy in Unstructured?',
        options: ['High resolution image processing', 'High-accuracy layout analysis with OCR and visual detection', 'Faster processing mode', 'High-quality output format'],
        correctAnswer: 1,
        explanation: 'The "hi_res" strategy uses advanced techniques including OCR and layout analysis for more accurate extraction, especially for PDFs with complex layouts.',
    },
    {
        id: 'fpdu-unstructured-mc-4',
        question: 'How does Unstructured represent document structure?',
        options: ['As a single text string', 'As elements with types (Title, Text, Table, etc.)', 'As JSON only', 'As HTML'],
        correctAnswer: 1,
        explanation: 'Unstructured returns elements with types (Title, NarrativeText, Table, etc.) that represent the document structure and hierarchy.',
    },
    {
        id: 'fpdu-unstructured-mc-5',
        question: 'What parameter enables table structure extraction in Unstructured?',
        options: ['extract_tables=True', 'infer_table_structure=True', 'parse_tables=True', 'table_mode="extract"'],
        correctAnswer: 1,
        explanation: 'Setting infer_table_structure=True enables Unstructured to detect and extract table structure, including cells and relationships.',
    },
];

