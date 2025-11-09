-- Performance Indexes for CodeBlanket Database
-- Run these to speed up queries significantly

-- Index on user_progress for faster lookups
CREATE INDEX IF NOT EXISTS idx_user_progress_user_id 
ON user_progress(user_id);

CREATE INDEX IF NOT EXISTS idx_user_progress_user_key 
ON user_progress(user_id, key);

-- Index on user_videos for faster lookups
CREATE INDEX IF NOT EXISTS idx_user_videos_user_id 
ON user_videos(user_id);

-- Analyze tables to update statistics
ANALYZE user_progress;
ANALYZE user_videos;





