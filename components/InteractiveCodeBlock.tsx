'use client';

import { useState, useEffect } from 'react';
import SimpleEditor from 'react-simple-code-editor';
import { highlight, languages } from 'prismjs';
import 'prismjs/components/prism-python';

import { getPyodide } from '@/lib/pyodide';
import { usePyodide } from '@/lib/hooks/usePyodide';
import { ClientOnlySyntaxHighlighter } from './ClientOnlySyntaxHighlighter';

interface InteractiveCodeBlockProps {
  code: string;
  language?: string;
}

export function InteractiveCodeBlock({
  code: initialCode,
  language = 'python',
}: InteractiveCodeBlockProps) {
  const [isInteractive, setIsInteractive] = useState(false);
  const [code, setCode] = useState(initialCode);
  const [output, setOutput] = useState<string>('');
  const [plotImages, setPlotImages] = useState<string[]>([]);
  const [isRunning, setIsRunning] = useState(false);

  const { isReady: pyodideReady, isLoading: pyodideLoading } = usePyodide();

  const runCode = async () => {
    setIsRunning(true);
    setOutput('');
    setPlotImages([]);

    try {
      const pyodide = await getPyodide();

      // Setup console output capture and matplotlib
      await pyodide.runPythonAsync(`
import sys
from io import StringIO, BytesIO
import base64
sys.stdout = StringIO()
sys.stderr = StringIO()

# Setup matplotlib if imported
_plot_images = []
try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Monkey-patch plt.show() to capture figures
    _original_show = plt.show
    def _custom_show(*args, **kwargs):
        global _plot_images
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            _plot_images.append(img_base64)
            buf.close()
        plt.close('all')
    plt.show = _custom_show
except ImportError:
    pass
`);

      // Execute user's code
      await pyodide.runPythonAsync(code);

      // Capture text output
      const textOutput = await pyodide.runPythonAsync(`
stdout_value = sys.stdout.getvalue()
stderr_value = sys.stderr.getvalue()
stdout_value + stderr_value
`);

      // Capture plot images
      const plotsJson = await pyodide.runPythonAsync(`
import json
json.dumps(_plot_images)
`);

      const plots = JSON.parse(plotsJson);

      if (textOutput) {
        setOutput(textOutput);
      } else if (plots.length === 0) {
        setOutput('✓ Code executed successfully (no output)');
      }

      if (plots.length > 0) {
        setPlotImages(plots);
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (error: any) {
      setOutput(`❌ Error: ${error.message}`);
    } finally {
      setIsRunning(false);
    }
  };

  const resetCode = () => {
    setCode(initialCode);
    setOutput('');
    setPlotImages([]);
  };

  // Keyboard shortcut support (only when interactive)
  useEffect(() => {
    if (!isInteractive) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd+Enter (Mac) or Ctrl+Enter (Windows/Linux) to run code
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        if (pyodideReady && !isRunning) {
          runCode();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isInteractive, pyodideReady, isRunning, code]);

  if (!isInteractive) {
    // Show read-only code with overlaid "Try it yourself" button
    return (
      <div className="group relative my-4">
        <div className="overflow-x-auto rounded-lg">
          <ClientOnlySyntaxHighlighter language={language} code={initialCode} />
        </div>
        {/* Overlay button on bottom right */}
        <button
          onClick={() => setIsInteractive(true)}
          className="absolute right-3 bottom-3 flex items-center gap-2 rounded-lg bg-[#bd93f9] px-3 py-1.5 text-xs font-semibold text-[#282a36] opacity-0 shadow-lg transition-all group-hover:opacity-100 hover:scale-105 hover:bg-[#bd93f9]/90"
        >
          <svg
            className="h-3.5 w-3.5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          Try it yourself
        </button>
      </div>
    );
  }

  // Show interactive editor
  return (
    <div className="my-4 overflow-hidden rounded-lg border-2 border-[#bd93f9] bg-[#282a36]">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#44475a] bg-[#21222c] px-4 py-2">
        <div className="flex items-center gap-2">
          <div className="flex gap-1.5">
            <div className="h-3 w-3 rounded-full bg-[#ff5555]"></div>
            <div className="h-3 w-3 rounded-full bg-[#f1fa8c]"></div>
            <div className="h-3 w-3 rounded-full bg-[#50fa7b]"></div>
          </div>
          <span className="ml-2 text-sm font-semibold text-[#bd93f9]">
            Interactive Editor
          </span>
        </div>
        <button
          onClick={() => setIsInteractive(false)}
          className="text-xs text-[#6272a4] transition-colors hover:text-[#f8f8f2]"
        >
          Close ✕
        </button>
      </div>

      {/* Python Loading State */}
      {pyodideLoading && (
        <div className="border-b border-[#44475a] bg-[#bd93f9]/10 p-3">
          <div className="flex items-center gap-2 text-sm">
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-[#bd93f9] border-t-transparent" />
            <span className="text-[#bd93f9]">
              Loading Python environment...
            </span>
          </div>
        </div>
      )}

      {/* Code Editor */}
      <div className="bg-[#282a36] p-4">
        <pre
          className="language-python"
          style={{ margin: 0, background: 'transparent' }}
        >
          <SimpleEditor
            value={code}
            onValueChange={setCode}
            highlight={(code) => highlight(code, languages.python, 'python')}
            padding={0}
            style={{
              fontFamily:
                '"Fira Code", "Fira Mono", "SF Mono", Monaco, Inconsolata, "Roboto Mono", Consolas, "Courier New", monospace',
              fontSize: 14,
              minHeight: '100px',
              backgroundColor: 'transparent',
              color: '#f8f8f2',
              outline: 'none',
              lineHeight: '1.5',
              caretColor: '#f8f8f2',
            }}
            textareaClassName="focus:outline-none"
          />
        </pre>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center justify-between border-t border-[#44475a] bg-[#21222c] px-4 py-2">
        <div className="flex gap-2">
          <button
            onClick={runCode}
            disabled={!pyodideReady || isRunning}
            className="flex items-center gap-2 rounded bg-[#50fa7b] px-4 py-1.5 text-sm font-semibold text-[#282a36] transition-colors hover:bg-[#50fa7b]/80 disabled:cursor-not-allowed disabled:bg-[#6272a4]"
          >
            {isRunning ? (
              <>
                <div className="h-3 w-3 animate-spin rounded-full border-2 border-[#282a36] border-t-transparent" />
                Running...
              </>
            ) : (
              <>
                <svg
                  className="h-4 w-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                Run Code
              </>
            )}
          </button>
          <button
            onClick={resetCode}
            disabled={!pyodideReady}
            className="rounded bg-[#6272a4] px-4 py-1.5 text-sm font-semibold text-[#f8f8f2] transition-colors hover:bg-[#6272a4]/80 disabled:cursor-not-allowed disabled:bg-[#44475a]"
          >
            Reset
          </button>
        </div>
        <div className="text-xs text-[#6272a4]">Cmd/Ctrl + Enter to run</div>
      </div>

      {/* Output */}
      {(output || plotImages.length > 0) && (
        <div className="border-t border-[#44475a] bg-[#1e1f29] p-4">
          {output && (
            <>
              <div className="mb-2 text-sm font-semibold text-[#bd93f9]">
                Output:
              </div>
              <pre className="mb-4 font-mono text-sm whitespace-pre-wrap text-[#f8f8f2]">
                {output}
              </pre>
            </>
          )}

          {/* Display matplotlib plots */}
          {plotImages.length > 0 && (
            <div className="space-y-4">
              {plotImages.map((imgData, idx) => (
                <div
                  key={idx}
                  className="rounded-lg border border-[#44475a] bg-[#282a36] p-4"
                >
                  <div className="mb-2 text-sm font-semibold text-[#bd93f9]">
                    Plot {plotImages.length > 1 ? `${idx + 1}` : ''}:
                  </div>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`data:image/png;base64,${imgData}`}
                    alt={`Plot ${idx + 1}`}
                    className="w-full rounded"
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
