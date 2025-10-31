'use client';

import { Suspense, useState } from 'react';

function VerifyContent() {
  const [progressData, setProgressData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const checkDatabase = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/progress');
      const data = await res.json();
      setProgressData(data);
    } catch (error) {
      setProgressData({ error: String(error) });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-8">
      <h1 className="mb-8 text-3xl font-bold text-white">
        Verify PostgreSQL Data
      </h1>

      <div className="mb-8 rounded-lg bg-[#44475a] p-6">
        <h2 className="mb-4 text-xl font-bold text-white">
          Check What's Actually in PostgreSQL
        </h2>
        <button
          onClick={checkDatabase}
          disabled={loading}
          className="mb-4 rounded-md bg-[#bd93f9] px-6 py-3 font-bold text-white hover:bg-[#a070e0] disabled:opacity-50"
        >
          {loading ? 'Loading...' : '🔍 Fetch My Data from PostgreSQL'}
        </button>

        {progressData && (
          <div className="mt-4 space-y-4">
            <div className="rounded-lg bg-[#282a36] p-4">
              <h3 className="mb-2 font-bold text-white">Raw Response:</h3>
              <pre className="overflow-auto text-xs text-green-400">
                {JSON.stringify(progressData, null, 2)}
              </pre>
            </div>

            {progressData.data && (
              <div className="rounded-lg bg-[#282a36] p-4">
                <h3 className="mb-2 font-bold text-white">
                  Data Keys Found: {Object.keys(progressData.data).length}
                </h3>
                {Object.keys(progressData.data).length > 0 ? (
                  <ul className="space-y-2 text-sm text-white">
                    {Object.keys(progressData.data).map((key) => (
                      <li key={key} className="font-mono">
                        ✓ {key}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-yellow-400">
                    ⚠️ No data in PostgreSQL yet. Try completing a problem
                    first!
                  </p>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="rounded-lg border-2 border-[#bd93f9] bg-[#282a36] p-6">
        <h2 className="mb-4 text-xl font-bold text-[#bd93f9]">How to Test</h2>
        <ol className="space-y-2 text-white">
          <li>1. Go back to homepage</li>
          <li>2. Complete a coding problem or save some code</li>
          <li>3. Come back here and click "Fetch My Data"</li>
          <li>4. You should see your data keys listed above</li>
          <li>
            5. Or check Neon Console → SQL Editor →{' '}
            <code className="rounded bg-[#44475a] px-2 py-1 text-xs">
              SELECT * FROM user_progress;
            </code>
          </li>
        </ol>
      </div>
    </div>
  );
}

export default function VerifyPage() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center">
          <div className="text-center">
            <div className="mx-auto h-12 w-12 animate-spin rounded-full border-t-2 border-b-2 border-[#bd93f9]"></div>
            <p className="mt-4 text-white">Loading...</p>
          </div>
        </div>
      }
    >
      <VerifyContent />
    </Suspense>
  );
}
