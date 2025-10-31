'use client';

/**
 * Debug page to verify PostgreSQL data
 */

import { Suspense, useState } from 'react';

export const dynamic = 'force-dynamic';

interface ApiResult {
  status?: number;
  data?: unknown;
  error?: string;
}

function VerifyContent() {
  const [statsData, setStatsData] = useState<ApiResult | null>(null);
  const [progressData, setProgressData] = useState<ApiResult | null>(null);
  const [videosData, setVideosData] = useState<ApiResult | null>(null);
  const [loading, setLoading] = useState(false);

  const checkAll = async () => {
    setLoading(true);

    // Check stats
    try {
      const statsRes = await fetch('/api/stats');
      const stats = await statsRes.json();
      setStatsData({
        status: statsRes.status,
        data: stats,
      });
    } catch (error) {
      setStatsData({
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }

    // Check progress
    try {
      const progressRes = await fetch('/api/progress');
      const progress = await progressRes.json();
      setProgressData({
        status: progressRes.status,
        data: progress,
      });
    } catch (error) {
      setProgressData({
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }

    // Check videos
    try {
      const videosRes = await fetch('/api/videos');
      const videos = await videosRes.json();
      setVideosData({
        status: videosRes.status,
        data: videos,
      });
    } catch (error) {
      setVideosData({
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-[#282a36] p-8 text-white">
      <div className="mx-auto max-w-4xl">
        <h1 className="mb-8 text-3xl font-bold text-[#bd93f9]">
          PostgreSQL Data Verification
        </h1>

        <button
          onClick={checkAll}
          disabled={loading}
          className="mb-8 rounded-lg bg-[#bd93f9] px-6 py-3 font-semibold text-white hover:bg-[#a070e0] disabled:opacity-50"
        >
          {loading ? 'Checking...' : 'Check All Data'}
        </button>

        {/* Stats API */}
        {statsData && (
          <div className="mb-6 rounded-lg border border-[#44475a] bg-[#44475a]/20 p-4">
            <h2 className="mb-2 text-xl font-bold text-[#8be9fd]">
              Stats API (/api/stats)
            </h2>
            <div className="mb-2 text-sm text-[#6272a4]">
              Status: {statsData.status || 'Error'}
            </div>
            <pre className="overflow-auto rounded bg-[#282a36] p-4 text-xs">
              {JSON.stringify(statsData.data || statsData.error, null, 2)}
            </pre>
          </div>
        )}

        {/* Progress API */}
        {progressData && (
          <div className="mb-6 rounded-lg border border-[#44475a] bg-[#44475a]/20 p-4">
            <h2 className="mb-2 text-xl font-bold text-[#50fa7b]">
              Progress API (/api/progress)
            </h2>
            <div className="mb-2 text-sm text-[#6272a4]">
              Status: {progressData.status || 'Error'}
            </div>
            <pre className="overflow-auto rounded bg-[#282a36] p-4 text-xs">
              {JSON.stringify(progressData.data || progressData.error, null, 2)}
            </pre>
          </div>
        )}

        {/* Videos API */}
        {videosData && (
          <div className="mb-6 rounded-lg border border-[#44475a] bg-[#44475a]/20 p-4">
            <h2 className="mb-2 text-xl font-bold text-[#f1fa8c]">
              Videos API (/api/videos)
            </h2>
            <div className="mb-2 text-sm text-[#6272a4]">
              Status: {videosData.status || 'Error'}
            </div>
            <pre className="overflow-auto rounded bg-[#282a36] p-4 text-xs">
              {JSON.stringify(videosData.data || videosData.error, null, 2)}
            </pre>
          </div>
        )}

        {!statsData && !progressData && !videosData && (
          <div className="text-center text-[#6272a4]">
            Click &quot;Check All Data&quot; to verify what&apos;s in PostgreSQL
          </div>
        )}
      </div>
    </div>
  );
}

export default function VerifyPage() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center bg-[#282a36]">
          <div className="text-white">Loading...</div>
        </div>
      }
    >
      <VerifyContent />
    </Suspense>
  );
}
