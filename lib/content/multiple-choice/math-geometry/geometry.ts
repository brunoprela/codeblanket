/**
 * Multiple choice questions for Coordinate Geometry section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const geometryMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the distance formula between two points?',
    options: [
      'Sum coordinates',
      '√((x2-x1)² + (y2-y1)²) - Pythagorean theorem',
      'Subtract coordinates',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Euclidean distance: √((x2-x1)² + (y2-y1)²) from Pythagorean theorem. For Manhattan distance: |x2-x1| + |y2-y1|. Choose based on problem (taxi cab vs straight line).',
  },
  {
    id: 'mc2',
    question: 'How do you determine if three points are collinear?',
    options: [
      'Random',
      'Calculate area of triangle - if 0, collinear. Or check slope equality',
      'Distance only',
      'Cannot determine',
    ],
    correctAnswer: 1,
    explanation:
      'Collinear points: area = 0. Area = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|/2. If 0, collinear. Alternative: slope(P1,P2) = slope(P2,P3). Handle vertical lines carefully.',
  },
  {
    id: 'mc3',
    question: 'What is the area of a triangle given coordinates?',
    options: [
      'Base times height',
      '|x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|/2 - cross product formula',
      'Random',
      'Perimeter',
    ],
    correctAnswer: 1,
    explanation:
      'Triangle area: |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|/2 or 0.5 * |cross product of two sides|. Useful for checking collinearity (area = 0) and point-in-triangle tests.',
  },
  {
    id: 'mc4',
    question: 'How do you check if a point is inside a rectangle?',
    options: [
      'Complex calculation',
      'x in [x_min, x_max] AND y in [y_min, y_max] - boundary check',
      'Distance formula',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Point (px, py) in axis-aligned rectangle: px >= x_min AND px <= x_max AND py >= y_min AND py <= y_max. For rotated rectangles: use cross products or rotation transformation.',
  },
  {
    id: 'mc5',
    question: 'What is the dot product and what does it tell you?',
    options: [
      'Random operation',
      'a·b = a_x*b_x + a_y*b_y - gives angle info: >0 acute, =0 perpendicular, <0 obtuse',
      'Cross product',
      'Area',
    ],
    correctAnswer: 1,
    explanation:
      'Dot product: a·b = |a||b|cos(θ) = a_x*b_x + a_y*b_y. If >0: acute angle. If =0: perpendicular (90°). If <0: obtuse. Used for angle detection, projections.',
  },
];
