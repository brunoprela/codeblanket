'use client';

import { useUser } from '@stackframe/stack';
import { useState, Suspense } from 'react';

function DebugContent() {
  const user = useUser();
  const [authCheckResult, setAuthCheckResult] = useState<any>(null);
  const [progressResult, setProgressResult] = useState<any>(null);
  const [videosResult, setVideosResult] = useState<any>(null);
  const [testSaveResult, setTestSaveResult] = useState<any>(null);

  const testAuthCheck = async () => {
    try {
      const res = await fetch('/api/auth/check');
      const data = await res.json();
      setAuthCheckResult({ status: res.status, data });
    } catch (error) {
      setAuthCheckResult({ error: error instanceof Error ? error.message : 'Unknown error' });
    }
  };

  const testProgressAPI = async () => {
    try {
      const res = await fetch('/api/progress');
      const data = await res.json();
      setProgressResult({ status: res.status, data });
    } catch (error) {
      setProgressResult({ error: error instanceof Error ? error.message : 'Unknown error' });
    }
  };

  const testVideosAPI = async () => {
    try {
      const res = await fetch('/api/videos');
      const data = await res.json();
      setVideosResult({ status: res.status, data });
    } catch (error) {
      setVideosResult({ error: error instanceof Error ? error.message : 'Unknown error' });
    }
  };

  const testSaveProgress = async () => {
    try {
      const res = await fetch('/api/progress', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          key: 'test-key-' + Date.now(),
          value: { test: 'data', timestamp: Date.now() },
        }),
      });
      const data = await res.json();
      setTestSaveResult({ status: res.status, data });
    } catch (error) {
      setTestSaveResult({ error: error instanceof Error ? error.message : 'Unknown error' });
    }
  };

  return (
    <div className="container mx-auto p-8">
      <h1 className="mb-8 text-3xl font-bold text-white">Debug Dashboard</h1>

      {/* User Info */}
      <div className="mb-8 rounded-lg bg-[#44475a] p-6">
        <h2 className="mb-4 text-xl font-bold text-white">Client-Side User Info (Stack Auth)</h2>
        {user ? (
          <div className="space-y-2 font-mono text-sm text-green-400">
            <p><strong>Status:</strong> ✅ Logged In</p>
            <p><strong>User ID:</strong> {user.id}</p>
            <p><strong>Email:</strong> {user.primaryEmail || 'N/A'}</p>
            <p><strong>Display Name:</strong> {user.displayName || 'N/A'}</p>
            <p><strong>OAuth Provider:</strong> {user.oauthProviders?.join(', ') || 'Email/Password'}</p>
          </div>
        ) : (
          <p className="text-red-400">❌ Not logged in</p>
        )}
      </div>

      {/* API Tests */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Auth Check API */}
        <div className="rounded-lg bg-[#44475a] p-6">
          <h3 className="mb-4 text-lg font-bold text-white">1. Auth Check API</h3>
          <button
            onClick={testAuthCheck}
            className="mb-4 rounded-md bg-[#bd93f9] px-4 py-2 text-white hover:bg-[#a070e0]"
          >
            Test /api/auth/check
          </button>
          {authCheckResult && (
            <pre className="overflow-auto rounded bg-[#282a36] p-3 text-xs text-green-400">
              {JSON.stringify(authCheckResult, null, 2)}
            </pre>
          )}
        </div>

        {/* Progress API */}
        <div className="rounded-lg bg-[#44475a] p-6">
          <h3 className="mb-4 text-lg font-bold text-white">2. Progress API (GET)</h3>
          <button
            onClick={testProgressAPI}
            className="mb-4 rounded-md bg-[#bd93f9] px-4 py-2 text-white hover:bg-[#a070e0]"
          >
            Test /api/progress
          </button>
          {progressResult && (
            <pre className="overflow-auto rounded bg-[#282a36] p-3 text-xs text-green-400">
              {JSON.stringify(progressResult, null, 2)}
            </pre>
          )}
        </div>

        {/* Videos API */}
        <div className="rounded-lg bg-[#44475a] p-6">
          <h3 className="mb-4 text-lg font-bold text-white">3. Videos API (GET)</h3>
          <button
            onClick={testVideosAPI}
            className="mb-4 rounded-md bg-[#bd93f9] px-4 py-2 text-white hover:bg-[#a070e0]"
          >
            Test /api/videos
          </button>
          {videosResult && (
            <pre className="overflow-auto rounded bg-[#282a36] p-3 text-xs text-green-400">
              {JSON.stringify(videosResult, null, 2)}
            </pre>
          )}
        </div>

        {/* Save Test */}
        <div className="rounded-lg bg-[#44475a] p-6">
          <h3 className="mb-4 text-lg font-bold text-white">4. Save Progress (POST)</h3>
          <button
            onClick={testSaveProgress}
            className="mb-4 rounded-md bg-[#50fa7b] px-4 py-2 text-[#282a36] hover:bg-[#5ffb8f]"
          >
            Test Save to PostgreSQL
          </button>
          {testSaveResult && (
            <pre className="overflow-auto rounded bg-[#282a36] p-3 text-xs text-green-400">
              {JSON.stringify(testSaveResult, null, 2)}
            </pre>
          )}
        </div>
      </div>

      {/* Environment Check */}
      <div className="mt-8 rounded-lg bg-[#44475a] p-6">
        <h2 className="mb-4 text-xl font-bold text-white">Environment Variables (Client-Side)</h2>
        <div className="space-y-2 font-mono text-sm">
          <p className="text-white">
            <strong>NEXT_PUBLIC_STACK_PROJECT_ID:</strong>{' '}
            <span className={process.env.NEXT_PUBLIC_STACK_PROJECT_ID ? 'text-green-400' : 'text-red-400'}>
              {process.env.NEXT_PUBLIC_STACK_PROJECT_ID ? '✅ Set' : '❌ Missing'}
            </span>
          </p>
          <p className="text-white">
            <strong>NEXT_PUBLIC_STACK_PUBLISHABLE_CLIENT_KEY:</strong>{' '}
            <span className={process.env.NEXT_PUBLIC_STACK_PUBLISHABLE_CLIENT_KEY ? 'text-green-400' : 'text-red-400'}>
              {process.env.NEXT_PUBLIC_STACK_PUBLISHABLE_CLIENT_KEY ? '✅ Set' : '❌ Missing'}
            </span>
          </p>
          <p className="text-gray-400 text-xs mt-4">
            Note: STACK_SECRET_SERVER_KEY, DATABASE_URL, and BLOB_READ_WRITE_TOKEN are server-only and won't show here
          </p>
        </div>
      </div>

      {/* Instructions */}
      <div className="mt-8 rounded-lg border-2 border-[#bd93f9] bg-[#282a36] p-6">
        <h2 className="mb-4 text-xl font-bold text-[#bd93f9]">Debugging Instructions</h2>
        <ol className="space-y-3 text-white">
          <li><strong>1. Check User Info above</strong> - Should show your Google account</li>
          <li><strong>2. Click "Test /api/auth/check"</strong> - Should return authenticated: true</li>
          <li><strong>3. Click "Test /api/progress"</strong> - Should return your data or empty object</li>
          <li><strong>4. Click "Test /api/videos"</strong> - Should return empty videos array</li>
          <li><strong>5. Click "Test Save to PostgreSQL"</strong> - Should save successfully</li>
          <li><strong>6. Check terminal logs</strong> - Will show detailed server-side info</li>
        </ol>
        <div className="mt-4 rounded bg-[#44475a] p-4">
          <p className="mb-2 text-sm font-bold text-yellow-400">Common Issues:</p>
          <ul className="space-y-1 text-sm text-gray-300">
            <li>• <strong>Status 401:</strong> Stack Auth session not working - try signing out/in</li>
            <li>• <strong>Status 500:</strong> Database connection failed - check DATABASE_URL</li>
            <li>• <strong>Network error:</strong> Dev server issue - check terminal</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default function DebugPage() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center">
          <div className="text-center">
            <div className="mx-auto h-12 w-12 animate-spin rounded-full border-b-2 border-t-2 border-[#bd93f9]"></div>
            <p className="mt-4 text-white">Loading debug dashboard...</p>
          </div>
        </div>
      }
    >
      <DebugContent />
    </Suspense>
  );
}

