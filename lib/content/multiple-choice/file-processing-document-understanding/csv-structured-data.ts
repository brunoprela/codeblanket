/**
 * Multiple choice questions for CSV & Structured Data section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const csvstructureddataMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fpdu-csv-mc-1',
    question: 'What is the most memory-efficient way to read a 5GB CSV file?',
    options: [
      'df = pd.read_csv("file.csv")',
      'df = pd.read_csv("file.csv", chunksize=10000)',
      'df = pd.read_csv("file.csv", low_memory=True)',
      'with open("file.csv") as f: data = f.read()',
    ],
    correctAnswer: 1,
    explanation:
      'Using chunksize parameter reads the file in chunks, processing one chunk at a time without loading the entire file into memory.',
  },
  {
    id: 'fpdu-csv-mc-2',
    question: 'How do you handle encoding errors when reading CSV files?',
    options: [
      'Ignore them',
      'Try UTF-8, then fallback to latin-1 or use chardet',
      'Convert file to UTF-8 first',
      'Use binary mode',
    ],
    correctAnswer: 1,
    explanation:
      'Best practice is to try UTF-8 first (most common), then fallback to latin-1 (permissive), or use chardet library for automatic detection.',
  },
  {
    id: 'fpdu-csv-mc-3',
    question: 'What does csv.Sniffer do?',
    options: [
      'Validates CSV data',
      'Auto-detects CSV delimiter and format',
      'Compresses CSV files',
      'Encrypts CSV data',
    ],
    correctAnswer: 1,
    explanation:
      'csv.Sniffer analyzes a sample of CSV data to automatically detect the delimiter, quote character, and other format parameters.',
  },
  {
    id: 'fpdu-csv-mc-4',
    question: 'What is JSONL?',
    options: [
      'A compressed JSON format',
      'Newline-delimited JSON (one JSON object per line)',
      'A JSON linting tool',
      'Large JSON files',
    ],
    correctAnswer: 1,
    explanation:
      'JSONL (JSON Lines) is a format where each line is a valid JSON object. It is ideal for streaming and processing large datasets line-by-line.',
  },
  {
    id: 'fpdu-csv-mc-5',
    question: 'How do you read a TSV (tab-separated values) file with pandas?',
    options: [
      'pd.read_tsv("file.tsv")',
      'pd.read_csv("file.tsv", sep="\\t")',
      'pd.read_table("file.tsv")',
      'Both B and C',
    ],
    correctAnswer: 3,
    explanation:
      'Both pd.read_csv with sep="\\t" and pd.read_table() work for TSV files. pd.read_table() is essentially pd.read_csv with sep="\\t" by default.',
  },
];
