/**
 * Multiple choice questions for Type System Understanding section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const typesystemunderstandingMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'cuam-typesystemunderstanding-mc-1',
        question:
            'What is type inference?',
        options: [
            'Manually declaring all types',
            'Automatically deducing types from code without explicit annotations',
            'Converting types at runtime',
            'Removing type checking',
        ],
        correctAnswer: 1,
        explanation:
            'Type inference automatically deduces variable types from context. Example: x = 5 infers x: int. This provides type safety without verbose annotations.',
    },
    {
        id: 'cuam-typesystemunderstanding-mc-2',
        question:
            'How do type checkers like mypy analyze this code?\n\ndef add(x: int, y: int) -> int:\n    return x + y\n\nresult = add("hello", "world")',
        options: [
            'Execute the code to check types',
            'Build type environment, infer types, check parameter types match annotations',
            'Ignore type hints',
            'Convert strings to integers',
        ],
        correctAnswer: 1,
        explanation:
            'Type checkers parse type annotations, build type environment, then verify call arguments match parameter types. Here, str arguments don\'t match int parameters → error.',
    },
    {
        id: 'cuam-typesystemunderstanding-mc-3',
        question:
            'What is a generic type?\n\nfrom typing import List\ndef process(items: List[int]) -> int',
        options: [
            'A type that can be any value',
            'A parameterized type (List[T] where T can vary)',
            'A deprecated type',
            'A dynamic type',
        ],
        correctAnswer: 1,
        explanation:
            'Generic types are parameterized (List[T], Dict[K,V]). The parameter (int here) specializes the generic. Enables type-safe containers without duplicating code for each element type.',
    },
    {
        id: 'cuam-typesystemunderstanding-mc-4',
        question:
            'What does Protocol mean in Python typing?\n\nfrom typing import Protocol\n\nclass Drawable(Protocol):\n    def draw(self) -> None: ...',
        options: [
            'A network protocol',
            'Structural subtyping - any class with draw() method matches',
            'An abstract base class requiring inheritance',
            'A deprecated feature',
        ],
        correctAnswer: 1,
        explanation:
            'Protocol enables structural (duck) typing - classes match based on having required methods/attributes, not inheritance. More flexible than nominal typing (ABC).',
    },
    {
        id: 'cuam-typesystemunderstanding-mc-5',
        question:
            'Why are type systems important for code understanding tools?',
        options: [
            'They make code run faster',
            'They enable smarter autocomplete, refactoring, and error detection',
            'They are required by Python',
            'They replace documentation',
        ],
        correctAnswer: 1,
        explanation:
            'Type information powers IDE features: autocomplete knows available methods, refactoring knows what breaks, go-to-definition works across modules. Essential for tooling quality.',
    },
];

