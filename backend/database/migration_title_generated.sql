-- ==============================================================================
-- Migration: Add title_generated column to chats table
-- Purpose: Support one-time automatic title generation and protect manual renames
-- ==============================================================================

-- 1. Add title_generated column with default FALSE
ALTER TABLE chats 
ADD COLUMN IF NOT EXISTS title_generated BOOLEAN NOT NULL DEFAULT FALSE;

-- 2. Safe Backfill for existing chats:
-- Any chat that already has a custom/meaningful title is permanently protected 
-- from automatic renaming.
-- Chats with the default placeholder title ("New Chat" or "Untitled chat") 
-- remain title_generated = FALSE so their first message can name them if still new.
UPDATE chats 
SET title_generated = TRUE 
WHERE title NOT IN ('New Chat', 'Untitled chat');
