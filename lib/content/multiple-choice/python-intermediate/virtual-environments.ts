/**
 * Multiple choice questions for Virtual Environments & Package Management section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const virtualenvironmentsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What command creates a virtual environment with venv?',
    options: [
      'python -m venv myenv',
      'python create-venv myenv',
      'pip install venv myenv',
      'venv create myenv',
    ],
    correctAnswer: 0,
    explanation:
      'python -m venv myenv is the correct command to create a virtual environment named "myenv" using the built-in venv module.',
  },
  {
    id: 'mc2',
    question: 'How do you activate a virtual environment on Linux/Mac?',
    options: [
      'activate myenv',
      'source myenv/bin/activate',
      'python myenv',
      'myenv/activate',
    ],
    correctAnswer: 1,
    explanation:
      'source myenv/bin/activate is the correct command to activate a virtual environment on Linux and Mac systems.',
  },
  {
    id: 'mc3',
    question: 'What does pip freeze > requirements.txt do?',
    options: [
      "Freezes pip so it can't be updated",
      'Saves all installed packages and versions to a file',
      'Locks the Python version',
      'Creates a backup of pip',
    ],
    correctAnswer: 1,
    explanation:
      'pip freeze > requirements.txt outputs all installed packages with their exact versions and saves them to requirements.txt, enabling environment recreation.',
  },
  {
    id: 'mc4',
    question:
      'Should you commit your virtual environment folder (venv/) to git?',
    options: [
      'Yes, always',
      'No, add it to .gitignore',
      'Only for small projects',
      'Only the bin/ directory',
    ],
    correctAnswer: 1,
    explanation:
      'No, never commit virtual environments to git. They are large, platform-specific, and can be recreated from requirements.txt. Add venv/ to .gitignore.',
  },
  {
    id: 'mc5',
    question: 'What is the advantage of Poetry over pip?',
    options: [
      'Faster installation',
      'Uses less disk space',
      'Better dependency resolution and lockfiles',
      'No advantages',
    ],
    correctAnswer: 2,
    explanation:
      'Poetry provides better dependency resolution (solves conflicts automatically), lockfiles for reproducibility, and cleaner dependency management with pyproject.toml.',
  },
];
