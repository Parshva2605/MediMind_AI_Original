-- SQL script to update the tests table with ai_summary field

-- Add ai_summary column to the tests table
ALTER TABLE tests ADD COLUMN IF NOT EXISTS ai_summary TEXT;

-- Add any necessary indexes
CREATE INDEX IF NOT EXISTS idx_tests_ai_summary ON tests (ai_summary) WHERE ai_summary IS NOT NULL;
