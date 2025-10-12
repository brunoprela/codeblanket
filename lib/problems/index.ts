import { Problem } from '@/lib/types';
import { binarySearchProblems } from './binary-search';
import { twoPointersProblems } from './two-pointers';

export interface ProblemCategory {
    id: string;
    title: string;
    description: string;
    icon: string;
    problemCount: number;
    problems: Problem[];
}

export const problemCategories: ProblemCategory[] = [
    {
        id: 'binary-search',
        title: 'Binary Search',
        description:
            'Master the art of dividing and conquering with logarithmic time complexity',
        icon: '🔍',
        problemCount: binarySearchProblems.length,
        problems: binarySearchProblems,
    },
    {
        id: 'two-pointers',
        title: 'Two Pointers',
        description:
            'Learn to efficiently solve array problems with two-pointer technique',
        icon: '👉👈',
        problemCount: twoPointersProblems.length,
        problems: twoPointersProblems,
    },
];

export const allProblems: Problem[] = [
    ...binarySearchProblems,
    ...twoPointersProblems,
];

export function getProblemById(id: string): Problem | undefined {
    return allProblems.find((p) => p.id === id);
}

export function getCategoryById(id: string): ProblemCategory | undefined {
    return problemCategories.find((c) => c.id === id);
}

export function getProblemsByDifficulty(
    difficulty: 'Easy' | 'Medium' | 'Hard',
): Problem[] {
    return allProblems.filter((p) => p.difficulty === difficulty);
}
