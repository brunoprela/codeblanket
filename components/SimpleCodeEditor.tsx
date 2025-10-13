'use client';

import { useState, useEffect } from 'react';
import SimpleEditor from 'react-simple-code-editor';
import { highlight, languages } from 'prismjs';
import 'prismjs/components/prism-python';

import { TestCase, TestResult } from '@/lib/types';
import { getPyodide } from '@/lib/pyodide';
import { usePyodide } from '@/lib/hooks/usePyodide';
import { useCodeStorage } from '@/lib/hooks/useCodeStorage';
import {
  markProblemCompleted,
  markProblemIncomplete,
} from '@/lib/helpers/storage';

interface SimpleCodeEditorProps {
  starterCode: string;
  testCases: TestCase[];
  problemId?: string;
  onSuccess?: () => void;
}

export function SimpleCodeEditor({
  starterCode,
  testCases,
  problemId,
  onSuccess,
}: SimpleCodeEditorProps) {
  // Custom hooks for managing state
  const {
    isReady: pyodideReady,
    isLoading: pyodideLoading,
    error: loadError,
  } = usePyodide();
  const {
    code,
    setCode,
    resetCode: resetCodeStorage,
  } = useCodeStorage(problemId, starterCode);

  // Local state
  const [results, setResults] = useState<TestResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isResultsCollapsed, setIsResultsCollapsed] = useState(false);
  const [consoleOutput, setConsoleOutput] = useState<string[]>([]);

  const runCode = async () => {
    setIsRunning(true);
    setResults([]);
    setConsoleOutput([]);

    try {
      const pyodide = await getPyodide();
      const testResults: TestResult[] = [];
      const capturedOutput: string[] = [];

      // Setup console output capture
      await pyodide.runPythonAsync(`
import sys
from io import StringIO
sys.stdout = StringIO()
sys.stderr = StringIO()
`);

      // Extract function name from code
      const functionMatch = code.match(/def\s+(\w+)\s*\(/);
      const functionName = functionMatch?.[1];

      if (!functionName) {
        throw new Error(
          'Could not find a function definition in your code. Make sure you define a function.',
        );
      }

      // Execute user's code to define the function
      await pyodide.runPythonAsync(code);

      // Capture any print output from user code
      const userOutput = await pyodide.runPythonAsync(`
stdout_value = sys.stdout.getvalue()
stderr_value = sys.stderr.getvalue()
# Reset for test cases
sys.stdout = StringIO()
sys.stderr = StringIO()
stdout_value
`);
      if (userOutput) {
        capturedOutput.push(userOutput);
      }

      // Run predefined test cases
      for (const test of testCases) {
        const startTime = performance.now();

        try {
          // Convert JavaScript input to Python-compatible format
          const inputStr = JSON.stringify(test.input);

          // Run the function with test input
          const result = await pyodide.runPythonAsync(`
import json
args = json.loads('''${inputStr}''')
result = ${functionName}(*args)
json.dumps(result)
          `);

          const endTime = performance.now();
          const actual = JSON.parse(result);

          testResults.push({
            passed: JSON.stringify(actual) === JSON.stringify(test.expected),
            input: test.input,
            expected: test.expected,
            actual,
            executionTime: endTime - startTime,
          });
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
        } catch (error: any) {
          testResults.push({
            passed: false,
            input: test.input,
            expected: test.expected,
            actual: null,
            error: error.message,
          });
        }
      }

      // Capture final console output (including any manual test prints)
      const finalOutput = await pyodide.runPythonAsync(`
sys.stdout.getvalue() + sys.stderr.getvalue()
`);
      if (finalOutput) {
        capturedOutput.push(finalOutput);
      }

      setResults(testResults);
      setConsoleOutput(capturedOutput);

      // Check if all tests passed
      if (testResults.every((r) => r.passed)) {
        // Mark problem as completed in localStorage
        if (problemId) {
          markProblemCompleted(problemId);
        }
        // Call success callback if provided
        if (onSuccess) {
          onSuccess();
        }
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (error: any) {
      setResults([
        {
          passed: false,
          input: [],
          expected: null,
          actual: null,
          error: `Error: ${error.message}`,
        },
      ]);
    } finally {
      setIsRunning(false);
    }
  };

  const handleReset = () => {
    resetCodeStorage();
    setResults([]);
    setConsoleOutput([]);

    if (problemId) {
      markProblemIncomplete(problemId);
      // Dispatch event to notify parent component
      window.dispatchEvent(
        new CustomEvent('problemReset', { detail: { problemId } }),
      );
    }
  };

  if (loadError) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-6">
        <h3 className="mb-2 text-lg font-semibold text-red-800">
          Failed to Load Python
        </h3>
        <p className="text-red-700">{loadError}</p>
        <p className="mt-2 text-sm text-red-600">
          Make sure you have an internet connection and try refreshing the page.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Pyodide Loading State */}
      {pyodideLoading && (
        <div className="m-4 rounded-lg border-2 border-[#bd93f9] bg-[#bd93f9]/10 p-4">
          <div className="flex items-center space-x-3">
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-[#bd93f9] border-t-transparent" />
            <div>
              <div className="font-semibold text-[#bd93f9]">
                Loading Python Environment...
              </div>
              <div className="text-sm text-[#f8f8f2]">
                This may take a few seconds on first load (~10MB)
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Code Editor */}
      <div className="flex-1 overflow-hidden">
        <div className="h-full overflow-auto bg-[#282a36] p-4">
          <pre className="language-python" style={{ margin: 0, background: 'transparent' }}>
            <SimpleEditor
              value={code}
              onValueChange={setCode}
              highlight={(code) => highlight(code, languages.python, 'python')}
              padding={16}
              style={{
                fontFamily: '"Fira Code", "Fira Mono", "SF Mono", Monaco, Inconsolata, "Roboto Mono", Consolas, "Courier New", monospace',
                fontSize: 15,
                minHeight: '100%',
                backgroundColor: 'transparent',
                color: '#f8f8f2',
                outline: 'none',
                lineHeight: '1.6',
                caretColor: '#f8f8f2',
              }}
              textareaClassName="focus:outline-none"
            />
          </pre>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-shrink-0 gap-3 bg-[#44475a] p-4">
        <button
          onClick={runCode}
          disabled={!pyodideReady || isRunning}
          className="rounded-lg bg-[#50fa7b] px-6 py-2.5 font-bold text-[#282a36] transition-colors hover:bg-[#50fa7b]/80 disabled:cursor-not-allowed disabled:bg-[#6272a4]"
        >
          {isRunning ? (
            <span className="flex items-center gap-2">
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-[#282a36] border-t-transparent" />
              Running...
            </span>
          ) : (
            '▶ Run Tests'
          )}
        </button>

        <button
          onClick={handleReset}
          disabled={!pyodideReady}
          className="rounded-lg bg-[#6272a4] px-6 py-2.5 font-semibold text-[#f8f8f2] transition-colors hover:bg-[#6272a4]/80 disabled:cursor-not-allowed disabled:bg-[#44475a]"
        >
          Reset Code
        </button>
      </div>

      {/* Test Results */}
      {results.length > 0 && (
        <div className="max-h-[40vh] min-h-0 flex-shrink-0 space-y-3 overflow-y-auto bg-[#282a36] p-4">
          <div className="flex items-center justify-between rounded-lg bg-[#44475a] p-4">
            <div className="text-lg font-semibold text-[#f8f8f2]">
              Test Results: {results.filter((r) => r.passed).length} /{' '}
              {results.length} Passed
            </div>
            <div className="flex items-center gap-3">
              {results.every((r) => r.passed) && (
                <div className="flex items-center gap-2 font-semibold text-[#50fa7b]">
                  <span className="text-2xl">🎉</span>
                  All tests passed!
                </div>
              )}
              <button
                onClick={() => setIsResultsCollapsed(!isResultsCollapsed)}
                className="rounded-lg bg-[#6272a4] px-4 py-2 font-semibold text-[#f8f8f2] transition-colors hover:bg-[#6272a4]/80"
              >
                {isResultsCollapsed ? '▼ Show Details' : '▲ Hide Details'}
              </button>
            </div>
          </div>

          {!isResultsCollapsed && (
            <div className="space-y-3">
              {results.map((result, i) => (
                <div
                  key={i}
                  className={`rounded-lg border-2 p-4 ${result.passed
                    ? 'border-[#50fa7b] bg-[#50fa7b]/10'
                    : 'border-[#ff5555] bg-[#ff5555]/10'
                    }`}
                >
                  <div className="mb-3 flex items-center justify-between">
                    <div className="text-lg font-semibold text-[#f8f8f2]">
                      {result.passed ? '✅' : '❌'} Test Case {i + 1}
                    </div>
                    {result.executionTime && (
                      <div className="text-sm text-[#6272a4]">
                        {result.executionTime.toFixed(2)}ms
                      </div>
                    )}
                  </div>

                  <div className="space-y-2 font-mono text-sm">
                    <div className="flex gap-2">
                      <span className="min-w-[80px] font-semibold text-[#bd93f9]">
                        Input:
                      </span>
                      <span className="text-[#f8f8f2]">
                        {formatValue(result.input)}
                      </span>
                    </div>

                    <div className="flex gap-2">
                      <span className="min-w-[80px] font-semibold text-[#bd93f9]">
                        Expected:
                      </span>
                      <span className="font-semibold text-[#50fa7b]">
                        {formatValue(result.expected)}
                      </span>
                    </div>

                    <div className="flex gap-2">
                      <span className="min-w-[80px] font-semibold text-[#bd93f9]">
                        Got:
                      </span>
                      <span
                        className={`font-semibold ${result.passed ? 'text-[#50fa7b]' : 'text-[#ff5555]'
                          }`}
                      >
                        {formatValue(result.actual)}
                      </span>
                    </div>

                    {result.error && (
                      <div className="mt-3 rounded border-2 border-[#ff5555] bg-[#ff5555]/10 p-3">
                        <div className="mb-1 font-semibold text-[#ff5555]">
                          Error:
                        </div>
                        <pre className="whitespace-pre-wrap text-xs text-[#ff5555]">
                          {result.error}
                        </pre>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Console Output */}
      {consoleOutput.length > 0 && (
        <div className="max-h-[30vh] min-h-0 flex-shrink-0 overflow-y-auto border-t border-[#44475a] bg-[#282a36] p-4">
          <div className="mb-3 rounded-lg bg-[#44475a] px-4 py-2">
            <div className="font-semibold text-[#f8f8f2]">📟 Console Output</div>
          </div>
          <div className="rounded-lg bg-[#282a36] border border-[#44475a] p-4 font-mono text-sm">
            {consoleOutput.map((output, i) => (
              <pre key={i} className="whitespace-pre-wrap text-[#f8f8f2]">
                {output}
              </pre>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function formatValue(value: any): string {
  if (value === null) return 'null';
  if (value === undefined) return 'undefined';
  if (typeof value === 'string') return `"${value}"`;
  return JSON.stringify(value);
}

