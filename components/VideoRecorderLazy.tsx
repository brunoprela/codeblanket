'use client';

import { useState, useRef, useEffect } from 'react';
import { loadVideoFromUrl } from '@/lib/helpers/storage-adapter';

interface VideoMetadata {
  id: string;
  blobUrl?: string; // Only present for authenticated users with Vercel Blob
  timestamp: number;
  size?: number;
}

interface VideoRecorderLazyProps {
  questionId: string;
  onSave?: (videoBlob: Blob, videoId: string) => void;
  onDelete?: (videoId: string) => void;
  existingVideos?: VideoMetadata[];
}

export function VideoRecorderLazy({
  questionId,
  onSave,
  onDelete,
  existingVideos = [],
}: VideoRecorderLazyProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [currentVideoUrl, setCurrentVideoUrl] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [showCamera, setShowCamera] = useState(false);

  // Track which videos have been loaded (for on-demand loading)
  const [loadedVideos, setLoadedVideos] = useState<Record<string, string>>({});
  const [loadingVideos, setLoadingVideos] = useState<Set<string>>(new Set());

  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Start camera
  const startCamera = async () => {
    try {
      setError(null);
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 480, max: 640 },
          height: { ideal: 360, max: 480 },
          frameRate: { ideal: 15, max: 24 },
          facingMode: 'user',
        },
        audio: true,
      });
      setStream(mediaStream);
      setShowCamera(true);
    } catch (err) {
      console.error('Error accessing camera:', err);
      setError(
        'Could not access camera. Please ensure you have granted camera permissions.',
      );
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStream(null);
    setShowCamera(false);
  };

  // Start recording
  const startRecording = () => {
    if (!stream) return;

    chunksRef.current = [];
    const mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'video/webm;codecs=vp9',
      videoBitsPerSecond: 600_000,
      audioBitsPerSecond: 64_000,
    });

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'video/webm' });
      const videoUrl = URL.createObjectURL(blob);
      setCurrentVideoUrl(videoUrl);

      // Generate unique ID for this video
      const videoId = `${questionId}-${Date.now()}`;

      // Save to storage
      if (onSave) {
        onSave(blob, videoId);
      }

      // Stop camera and reset after brief success message
      setTimeout(() => {
        setCurrentVideoUrl(null);
        stopCamera();
      }, 2000); // Show success briefly
    };

    mediaRecorderRef.current = mediaRecorder;
    mediaRecorder.start();
    setIsRecording(true);
    setRecordingTime(0);

    // Start timer
    timerRef.current = setInterval(() => {
      setRecordingTime((prev) => prev + 1);
    }, 1000);
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);

      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    }
  };

  // Load video on-demand (only for Vercel Blob videos)
  const loadVideo = async (videoId: string, blobUrl: string) => {
    if (loadedVideos[videoId]) {
      // Already loaded
      return;
    }

    setLoadingVideos((prev) => new Set(prev).add(videoId));

    try {
      const blob = await loadVideoFromUrl(blobUrl);
      if (blob) {
        const url = URL.createObjectURL(blob);
        setLoadedVideos((prev) => ({ ...prev, [videoId]: url }));
      }
    } catch (error) {
      console.error('Failed to load video:', error);
      setError('Failed to load video. Please try again.');
    } finally {
      setLoadingVideos((prev) => {
        const newSet = new Set(prev);
        newSet.delete(videoId);
        return newSet;
      });
    }
  };

  // Delete recording
  const deleteRecording = (videoId: string) => {
    // Clean up loaded video URL if exists
    if (loadedVideos[videoId]) {
      URL.revokeObjectURL(loadedVideos[videoId]);
      setLoadedVideos((prev) => {
        const newLoaded = { ...prev };
        delete newLoaded[videoId];
        return newLoaded;
      });
    }

    // Notify parent to delete from storage
    if (onDelete) {
      onDelete(videoId);
    }
  };

  // Format time
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Format file size
  const formatSize = (bytes?: number) => {
    if (!bytes) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // Ensure video element always has the stream
  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (currentVideoUrl) {
        URL.revokeObjectURL(currentVideoUrl);
      }
      // Clean up loaded video URLs
      Object.values(loadedVideos).forEach((url) => URL.revokeObjectURL(url));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="mt-4 rounded-lg border-2 border-[#bd93f9] bg-[#282a36] p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <svg
            className="h-5 w-5 text-[#bd93f9]"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
            />
          </svg>
          <h4 className="font-semibold text-[#f8f8f2]">
            Video Answers{' '}
            {existingVideos.length > 0 && `(${existingVideos.length})`}
          </h4>
          {existingVideos.length > 0 && (
            <span className="rounded-full bg-[#50fa7b]/20 px-2 py-0.5 text-xs font-semibold text-[#50fa7b]">
              ✓ Completed
            </span>
          )}
        </div>
      </div>

      {error && (
        <div className="mb-4 rounded-lg border border-[#ff5555] bg-[#ff5555]/10 p-3 text-sm text-[#ff5555]">
          {error}
        </div>
      )}

      {/* Existing videos grid - optimized for bandwidth */}
      {existingVideos.length > 0 && (
        <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
          {existingVideos.map((video, index) => {
            const isLoaded = !!loadedVideos[video.id];
            const isLoading = loadingVideos.has(video.id);
            const hasBlob = !!video.blobUrl;

            return (
              <div
                key={video.id}
                className="overflow-hidden rounded-lg border border-[#44475a] bg-[#1e1f29]"
              >
                {isLoaded ? (
                  // Show actual video when loaded
                  <video
                    src={loadedVideos[video.id]}
                    controls
                    className="w-full"
                    style={{ maxHeight: '200px', transform: 'scaleX(-1)' }}
                  />
                ) : (
                  // Show placeholder with load button
                  <div className="flex h-[200px] flex-col items-center justify-center gap-3 bg-[#282a36] p-4">
                    <svg
                      className="h-16 w-16 text-[#6272a4]"
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
                    <div className="text-center">
                      <p className="text-sm font-semibold text-[#f8f8f2]">
                        Your Answer - Recording {index + 1}
                      </p>
                      <p className="mt-1 text-xs text-[#6272a4]">
                        {new Date(video.timestamp).toLocaleDateString()}
                      </p>
                      {video.size && (
                        <p className="text-xs text-[#6272a4]">
                          {formatSize(video.size)}
                        </p>
                      )}
                    </div>
                    {hasBlob && (
                      <button
                        onClick={() => loadVideo(video.id, video.blobUrl!)}
                        disabled={isLoading}
                        className="rounded-md bg-[#bd93f9] px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-[#a070e0] disabled:opacity-50"
                      >
                        {isLoading ? (
                          <span className="flex items-center gap-2">
                            <svg
                              className="h-4 w-4 animate-spin"
                              viewBox="0 0 24 24"
                            >
                              <circle
                                className="opacity-25"
                                cx="12"
                                cy="12"
                                r="10"
                                stroke="currentColor"
                                strokeWidth="4"
                                fill="none"
                              />
                              <path
                                className="opacity-75"
                                fill="currentColor"
                                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                              />
                            </svg>
                            Loading...
                          </span>
                        ) : (
                          <>
                            <svg
                              className="h-5 w-5"
                              fill="currentColor"
                              viewBox="0 0 20 20"
                            >
                              <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
                            </svg>
                            Watch My Answer
                          </>
                        )}
                      </button>
                    )}
                  </div>
                )}
                <div className="flex items-center justify-between border-t border-[#44475a] p-2">
                  <span className="text-xs text-[#6272a4]">
                    {new Date(video.timestamp).toLocaleDateString()}
                  </span>
                  <button
                    onClick={() => deleteRecording(video.id)}
                    className="rounded px-2 py-1 text-xs font-semibold text-[#ff5555] transition-colors hover:bg-[#ff5555]/10"
                  >
                    Delete
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Success message */}
      {currentVideoUrl && (
        <div className="mb-4 rounded-lg border border-[#50fa7b] bg-[#50fa7b]/10 p-3 text-center text-sm text-[#50fa7b]">
          ✓ Video saved successfully!
        </div>
      )}

      {/* Camera preview */}
      {(showCamera || stream) && (
        <div className="mb-4 overflow-hidden rounded-lg bg-[#1e1f29]">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full"
            style={{ maxHeight: '300px', transform: 'scaleX(-1)' }}
          />
        </div>
      )}

      {/* Recording timer */}
      {isRecording && (
        <div className="mb-4 flex items-center justify-center gap-2 rounded-lg bg-[#ff5555]/10 py-2 text-[#ff5555]">
          <div className="h-3 w-3 animate-pulse rounded-full bg-[#ff5555]" />
          <span className="font-mono text-lg font-semibold">
            {formatTime(recordingTime)}
          </span>
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-wrap gap-2">
        {!showCamera && !stream && (
          <button
            onClick={startCamera}
            className="flex items-center gap-2 rounded-md bg-[#bd93f9] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#a070e0]"
          >
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
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
            Start Camera
          </button>
        )}

        {(showCamera || stream) && !isRecording && (
          <>
            <button
              onClick={startRecording}
              className="flex items-center gap-2 rounded-md bg-[#ff5555] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#ff6b6b]"
            >
              <div className="h-3 w-3 rounded-full bg-white" />
              Record
            </button>
            <button
              onClick={stopCamera}
              className="rounded-md border border-[#6272a4] px-4 py-2 text-sm font-semibold text-[#f8f8f2] transition-colors hover:bg-[#44475a]"
            >
              Cancel
            </button>
          </>
        )}

        {isRecording && (
          <button
            onClick={stopRecording}
            className="flex items-center gap-2 rounded-md bg-[#50fa7b] px-4 py-2 text-sm font-semibold text-[#282a36] transition-colors hover:bg-[#5ffb8f]"
          >
            <div className="h-3 w-3 rounded-sm bg-[#282a36]" />
            Stop & Save
          </button>
        )}
      </div>
    </div>
  );
}
