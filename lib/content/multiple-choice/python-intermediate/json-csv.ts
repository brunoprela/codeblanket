/**
 * Multiple choice questions for Working with JSON and CSV section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const jsoncsvMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pi-jsoncsv-mc-1',
    question: 'Why use newline="" when writing CSV files?',
    options: [
      'Makes file smaller',
      'Prevents extra blank lines on Windows',
      'Required for UTF-8',
      'Speeds up writing',
    ],
    correctAnswer: 1,
    explanation:
      'newline="" prevents the csv module from adding extra blank lines on Windows.',
  },
  {
    id: 'pi-jsoncsv-mc-2',
    question: 'What does json.dump() vs json.dumps() do?',
    options: [
      'dump() is faster',
      'dump() writes to file, dumps() returns string',
      'dumps() is newer',
      'They are identical',
    ],
    correctAnswer: 1,
    explanation:
      'json.dump() writes directly to a file object, while json.dumps() returns a JSON string.',
  },
  {
    id: 'pi-jsoncsv-mc-3',
    question: 'What Python value becomes null in JSON?',
    options: ['0', 'False', 'None', '""'],
    correctAnswer: 2,
    explanation: 'Python None is converted to JSON null.',
  },
  {
    id: 'pi-jsoncsv-mc-4',
    question: 'How do you pretty-print JSON with indentation?',
    options: [
      'json.dumps(data, pretty=True)',
      'json.dumps(data, indent=2)',
      'json.dumps(data, format="pretty")',
      'json.pretty(data)',
    ],
    correctAnswer: 1,
    explanation:
      'Use the indent parameter: json.dumps(data, indent=2) for readable JSON.',
  },
  {
    id: 'pi-jsoncsv-mc-5',
    question: 'What does csv.DictReader do?',
    options: [
      'Reads CSV as list of lists',
      'Reads CSV as list of dictionaries',
      'Converts CSV to JSON',
      'Validates CSV data',
    ],
    correctAnswer: 1,
    explanation:
      'DictReader reads each CSV row as a dictionary with column names as keys.',
  },
];
