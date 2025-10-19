/**
 * Multiple choice questions for Advanced Visualization Techniques section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const advancedvisualizationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Which visualization library is best suited for creating interactive, browser-based dashboards?',
    options: [
      'Matplotlib',
      'Seaborn',
      'Plotly',
      'PIL (Python Imaging Library)',
    ],
    correctAnswer: 2,
    explanation:
      'Plotly is specifically designed for interactive, browser-based visualizations. It creates HTML/JavaScript output that allows zooming, panning, hovering, and clicking. Matplotlib and Seaborn create static images.',
  },
  {
    id: 'mc2',
    question:
      'What is the primary advantage of using hexbin plots over regular scatter plots for geographic data?',
    options: [
      'Hexbin plots are faster to create',
      'Hexbin plots aggregate dense regions, making patterns clearer in crowded areas',
      'Hexbin plots use less memory',
      'Hexbin plots work better for categorical data',
    ],
    correctAnswer: 1,
    explanation:
      'Hexbin plots aggregate points into hexagonal bins, which is crucial when you have thousands of overlapping points. This reveals density patterns that would be invisible in a standard scatter plot where points overlap.',
  },
  {
    id: 'mc3',
    question:
      'When creating visualizations for a non-technical business audience, you should:',
    options: [
      'Include all statistical details and technical jargon',
      'Use complex 3D visualizations to impress them',
      'Simplify, annotate key insights, and clearly state business implications',
      'Only show raw data tables',
    ],
    correctAnswer: 2,
    explanation:
      'Non-technical audiences need clear, annotated visualizations that tell a story and connect to business outcomes. Avoid jargon, focus on key insights, and explicitly state "what this means for the business". Complexity impresses nobody if they can\'t understand it.',
  },
  {
    id: 'mc4',
    question:
      'Which color scale is most appropriate for visualizing correlation coefficients ranging from -1 to +1?',
    options: [
      'Sequential scale (e.g., Blues, increasing darkness)',
      'Diverging scale (e.g., Red-White-Blue, centered at 0)',
      'Qualitative scale (e.g., discrete colors)',
      'Rainbow scale',
    ],
    correctAnswer: 1,
    explanation:
      "Correlation coefficients have a natural center at 0 (no correlation), with negative and positive extremes. A diverging color scale (e.g., blue for negative, white for zero, red for positive) best represents this structure. Sequential scales don't show the sign properly.",
  },
  {
    id: 'mc5',
    question: 'What is "chart junk" and why should it be avoided?',
    options: [
      'Old or outdated data in charts',
      'Unnecessary visual elements that distract from data (excessive gridlines, 3D effects, decorations)',
      'Charts that contain errors',
      'Charts with too much data',
    ],
    correctAnswer: 1,
    explanation:
      "Chart junk refers to unnecessary visual elements that don't convey information and distract from the data (heavy gridlines, 3D effects, excessive borders, decorative elements). The goal is to maximize the data-ink ratio: every bit of ink should convey information.",
  },
];
