/**
 * Multiple choice questions for Excel Advanced Operations section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const exceladvancedoperationsMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'fpdu-excel-adv-mc-1',
        question:
            'How do you write an Excel formula (not a calculated value) using openpyxl?',
        options: [
            'cell.value = "SUM(A1:A10)"',
            'cell.value = "=SUM(A1:A10)"',
            'cell.formula = "SUM(A1:A10)"',
            'cell.set_formula("SUM(A1:A10)")',
        ],
        correctAnswer: 1,
        explanation:
            'Excel formulas must start with "=" when setting cell.value in openpyxl. The formula is stored as a string starting with "=".',
    },
    {
        id: 'fpdu-excel-adv-mc-2',
        question:
            'What is the difference between loading with data_only=True vs data_only=False?',
        options: [
            'data_only=True loads faster',
            'data_only=True reads formula results, False reads formula strings',
            'data_only=True skips empty cells',
            'data_only=True only loads the first sheet',
        ],
        correctAnswer: 1,
        explanation:
            'data_only=True reads the cached values of formulas (last calculated result), while data_only=False reads the formula strings themselves. You need False to see the actual formulas.',
    },
    {
        id: 'fpdu-excel-adv-mc-3',
        question:
            'Which chart type is most appropriate for showing parts of a whole (like market share)?',
        options: [
            'LineChart',
            'BarChart',
            'PieChart',
            'ScatterChart',
        ],
        correctAnswer: 2,
        explanation:
            'PieChart is best for showing parts of a whole, where each slice represents a proportion of the total. Perfect for market share, budget allocation, etc.',
    },
    {
        id: 'fpdu-excel-adv-mc-4',
        question:
            'What does conditional formatting with ColorScaleRule do?',
        options: [
            'Changes text color based on value',
            'Applies gradient colors across a range based on cell values',
            'Creates a color picker dropdown',
            'Validates that cells contain specific colors',
        ],
        correctAnswer: 1,
        explanation:
            'ColorScaleRule applies a gradient of colors across a range, with colors corresponding to values (e.g., red for low, yellow for medium, green for high). Great for visualizing data trends.',
    },
    {
        id: 'fpdu-excel-adv-mc-5',
        question:
            'What is the purpose of Excel Tables (created with Table class in openpyxl)?',
        options: [
            'They are just for visual formatting',
            'They provide automatic filtering, sorting, and structured references',
            'They are required for formulas to work',
            'They allow multiple users to edit simultaneously',
        ],
        correctAnswer: 1,
        explanation:
            'Excel Tables are formatted ranges with built-in features: automatic filtering, sorting, formula auto-expansion, and structured references (e.g., TableName[ColumnName] instead of cell ranges).',
    },
];

