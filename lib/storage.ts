// LocalStorage utilities for tracking problem completion and user code

const COMPLETED_PROBLEMS_KEY = 'codeblanket_completed_problems';
const USER_CODE_KEY_PREFIX = 'codeblanket_code_';

// Problem completion tracking
export function getCompletedProblems(): Set<string> {
    if (typeof window === 'undefined') return new Set();

    try {
        const stored = localStorage.getItem(COMPLETED_PROBLEMS_KEY);
        return stored ? new Set(JSON.parse(stored)) : new Set();
    } catch {
        return new Set();
    }
}

export function markProblemCompleted(problemId: string): void {
    if (typeof window === 'undefined') return;

    try {
        const completed = getCompletedProblems();
        completed.add(problemId);
        localStorage.setItem(
            COMPLETED_PROBLEMS_KEY,
            JSON.stringify([...completed]),
        );
    } catch (error) {
        console.error('Failed to save completion status:', error);
    }
}

export function markProblemIncomplete(problemId: string): void {
    if (typeof window === 'undefined') return;

    try {
        const completed = getCompletedProblems();
        completed.delete(problemId);
        localStorage.setItem(
            COMPLETED_PROBLEMS_KEY,
            JSON.stringify([...completed]),
        );
    } catch (error) {
        console.error('Failed to remove completion status:', error);
    }
}

export function isProblemCompleted(problemId: string): boolean {
    return getCompletedProblems().has(problemId);
}

export function clearCompletedProblems(): void {
    if (typeof window === 'undefined') return;

    try {
        localStorage.removeItem(COMPLETED_PROBLEMS_KEY);
    } catch (error) {
        console.error('Failed to clear completion status:', error);
    }
}

// User code storage
export function saveUserCode(problemId: string, code: string): void {
    if (typeof window === 'undefined') return;

    try {
        localStorage.setItem(`${USER_CODE_KEY_PREFIX}${problemId}`, code);
    } catch (error) {
        console.error('Failed to save user code:', error);
    }
}

export function getUserCode(problemId: string): string | null {
    if (typeof window === 'undefined') return null;

    try {
        return localStorage.getItem(`${USER_CODE_KEY_PREFIX}${problemId}`);
    } catch (error) {
        console.error('Failed to load user code:', error);
        return null;
    }
}

export function clearUserCode(problemId: string): void {
    if (typeof window === 'undefined') return;

    try {
        localStorage.removeItem(`${USER_CODE_KEY_PREFIX}${problemId}`);
    } catch (error) {
        console.error('Failed to clear user code:', error);
    }
}
