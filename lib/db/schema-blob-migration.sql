-- Migration: Update user_videos table for Vercel Blob Storage
-- Run this migration if you already have the old schema

-- Rename old table as backup
ALTER TABLE IF EXISTS user_videos RENAME TO user_videos_backup;

-- Create new table with blob URL instead of binary data
CREATE TABLE user_videos (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id TEXT NOT NULL,
  video_id TEXT NOT NULL,
  blob_url TEXT NOT NULL, -- Vercel Blob Storage URL
  blob_pathname TEXT NOT NULL, -- Path in blob storage
  mime_type TEXT DEFAULT 'video/webm',
  file_size BIGINT, -- Size in bytes
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Ensure unique video per user
  UNIQUE(user_id, video_id)
);

-- Index for faster queries by user_id
CREATE INDEX idx_user_videos_user_id ON user_videos(user_id);

-- Trigger to automatically update updated_at
DROP TRIGGER IF EXISTS update_user_videos_updated_at ON user_videos;
CREATE TRIGGER update_user_videos_updated_at
  BEFORE UPDATE ON user_videos
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Note: Old video data in user_videos_backup can be migrated manually if needed
-- After confirming everything works, you can drop the backup table:
-- DROP TABLE user_videos_backup;



