-- CodeBlanket User Progress Database Schema
-- For use with Neon PostgreSQL

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- User progress data table
-- Stores all user progress data (completed problems, code, quiz answers, etc.)
CREATE TABLE IF NOT EXISTS user_progress (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id TEXT NOT NULL, -- Stack Auth user ID
  key TEXT NOT NULL, -- Storage key (e.g., 'codeblanket_completed_problems', 'codeblanket_code_two-sum')
  value JSONB NOT NULL, -- The actual data stored as JSON
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Ensure unique key per user
  UNIQUE(user_id, key)
);

-- Index for faster queries by user_id
CREATE INDEX IF NOT EXISTS idx_user_progress_user_id ON user_progress(user_id);

-- Index for faster queries by user_id and key pattern
CREATE INDEX IF NOT EXISTS idx_user_progress_user_key ON user_progress(user_id, key);

-- Video recordings table
-- Stores metadata for videos stored in Vercel Blob Storage
CREATE TABLE IF NOT EXISTS user_videos (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id TEXT NOT NULL, -- Stack Auth user ID
  video_id TEXT NOT NULL, -- Video identifier (e.g., problem ID)
  blob_url TEXT NOT NULL, -- Vercel Blob Storage URL
  blob_pathname TEXT NOT NULL, -- Path in blob storage for deletion
  mime_type TEXT DEFAULT 'video/webm',
  file_size BIGINT, -- Size in bytes
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Ensure unique video per user
  UNIQUE(user_id, video_id)
);

-- Index for faster queries by user_id
CREATE INDEX IF NOT EXISTS idx_user_videos_user_id ON user_videos(user_id);

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers to automatically update updated_at
DROP TRIGGER IF EXISTS update_user_progress_updated_at ON user_progress;
CREATE TRIGGER update_user_progress_updated_at
  BEFORE UPDATE ON user_progress
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_user_videos_updated_at ON user_videos;
CREATE TRIGGER update_user_videos_updated_at
  BEFORE UPDATE ON user_videos
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

