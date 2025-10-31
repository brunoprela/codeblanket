'use client';

/**
 * Test page to verify progress data is loading correctly
 */

import { useEffect, useState } from 'react';
import { getUserStats } from '@/lib/helpers/storage-stats';
import { getCompletedProblems } from '@/lib/helpers/storage';

export default function TestProgressPage() {
  const [authStatus, setAuthStatus] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);
  const [problems, setProblems] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const runTests = async () => {
      console.log('=== STARTING PROGRESS TESTS ===');

      // Test 1: Auth check
      console.log('Test 1: Checking authentication...');
      const authRes = await fetch('/api/auth/check');
      const authData = await authRes.json();
      setAuthStatus(authData);
      console.log('Auth status:', authData);

      // Test 2: Get stats
      console.log('Test 2: Fetching stats...');
      const statsData = await getUserStats();
      setStats(statsData);
      console.log('Stats:', statsData);

      // Test 3: Get completed problems
      console.log('Test 3: Fetching completed problems...');
      const problemsData = await getCompletedProblems();
      setProblems(Array.from(problemsData));
      console.log('Completed problems:', Array.from(problemsData));

      console.log('=== TESTS COMPLETE ===');
      setLoading(false);
    };

    runTests();
  }, []);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#282a36] text-white">
        <div>Running tests... (check console)</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#282a36] p-8 text-white">
      <div className="mx-auto max-w-4xl">
        <h1 className="mb-8 text-3xl font-bold text-[#bd93f9]">
          Progress Data Test
        </h1>

        <div className="space-y-6">
          {/* Auth Status */}
          <div className="rounded-lg border border-[#44475a] bg-[#44475a]/20 p-4">
            <h2 className="mb-2 text-xl font-bold text-[#8be9fd]">
              Auth Status
            </h2>
            <pre className="overflow-auto rounded bg-[#282a36] p-4 text-xs">
              {JSON.stringify(authStatus, null, 2)}
            </pre>
          </div>

          {/* Stats */}
          <div className="rounded-lg border border-[#44475a] bg-[#44475a]/20 p-4">
            <h2 className="mb-2 text-xl font-bold text-[#50fa7b]">
              Stats from getUserStats()
            </h2>
            <pre className="overflow-auto rounded bg-[#282a36] p-4 text-xs">
              {JSON.stringify(stats, null, 2)}
            </pre>
          </div>

          {/* Problems */}
          <div className="rounded-lg border border-[#44475a] bg-[#44475a]/20 p-4">
            <h2 className="mb-2 text-xl font-bold text-[#f1fa8c]">
              Completed Problems
            </h2>
            <pre className="overflow-auto rounded bg-[#282a36] p-4 text-xs">
              {JSON.stringify(problems, null, 2)}
            </pre>
          </div>
        </div>

        <div className="mt-8">
          <a href="/" className="text-[#8be9fd] hover:underline">
            ← Back to Homepage
          </a>
        </div>
      </div>
    </div>
  );
}
