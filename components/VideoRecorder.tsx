'use client';

import { useState, useRef, useEffect } from 'react';

interface VideoRecorderProps {
  questionId: string;
  onSave?: (videoBlob: Blob, videoId: string) => void;
  onDelete?: (videoId: string) => void;
  existingVideos?: Array<{ id: string; url: string }>;
}

export function VideoRecorder({
  questionId,
  onSave,
  onDelete,
  existingVideos = [],
}: VideoRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [currentVideoUrl, setCurrentVideoUrl] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [showCamera, setShowCamera] = useState(false);

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
          width: { ideal: 640 },
          height: { ideal: 480 },
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

  // Delete recording
  const deleteRecording = (videoId: string) => {
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

      {/* Existing videos grid */}
      {existingVideos.length > 0 && (
        <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
          {existingVideos.map((video, index) => (
            <div
              key={video.id}
              className="overflow-hidden rounded-lg border border-[#44475a] bg-[#1e1f29]"
            >
              <video
                src={video.url}
                controls
                className="w-full"
                style={{ maxHeight: '200px', transform: 'scaleX(-1)' }}
              />
              <div className="flex items-center justify-between border-t border-[#44475a] p-2">
                <span className="text-xs text-[#6272a4]">
                  Recording {index + 1}
                </span>
                <button
                  onClick={() => deleteRecording(video.id)}
                  className="rounded px-2 py-1 text-xs font-semibold text-[#ff5555] transition-colors hover:bg-[#ff5555]/10"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Success message */}
      {currentVideoUrl && (
        <div className="mb-4 rounded-lg border border-[#50fa7b] bg-[#50fa7b]/10 p-3 text-center text-sm text-[#50fa7b]">
          ✓ Video saved successfully!
        </div>
      )}

      {/* Camera preview - smaller size */}
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
        {!stream && !isRecording && (
          <button
            onClick={startCamera}
            className="flex items-center gap-2 rounded-lg bg-[#bd93f9] px-4 py-2 text-sm font-semibold text-[#282a36] transition-colors hover:bg-[#bd93f9]/80"
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
                d="M12 4v16m8-8H4"
              />
            </svg>
            {existingVideos.length > 0
              ? 'Add Another Video'
              : 'Add Video Answer'}
          </button>
        )}

        {stream && !isRecording && (
          <>
            <button
              onClick={startRecording}
              className="flex items-center gap-2 rounded-lg bg-[#ff5555] px-4 py-2 text-sm font-semibold text-[#282a36] transition-colors hover:bg-[#ff5555]/80"
            >
              <div className="h-3 w-3 rounded-full bg-[#282a36]" />
              Start Recording
            </button>
            <button
              onClick={stopCamera}
              className="rounded-lg bg-[#6272a4] px-4 py-2 text-sm font-semibold text-[#f8f8f2] transition-colors hover:bg-[#6272a4]/80"
            >
              Cancel
            </button>
          </>
        )}

        {isRecording && (
          <button
            onClick={stopRecording}
            className="flex items-center gap-2 rounded-lg bg-[#ff5555] px-6 py-2 text-sm font-semibold text-[#282a36] transition-colors hover:bg-[#ff5555]/80"
          >
            <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="6" width="12" height="12" rx="2" />
            </svg>
            Stop & Save
          </button>
        )}
      </div>
    </div>
  );
}
