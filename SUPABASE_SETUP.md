# Supabase Setup Guide

## 🚀 Quick Start

### 1. Create Supabase Account & Project

1. Go to [https://supabase.com](https://supabase.com)
2. Sign up for a free account
3. Create a new project
4. Wait for the project to be ready (takes ~2 minutes)

### 2. Get Your Credentials

1. Go to your project dashboard
2. Click on "Settings" → "API"
3. Copy:
   - **Project URL**: `https://your-project.supabase.co`
   - **Anon/Public Key**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

### 3. Create Database Table

1. Go to "SQL Editor" in your Supabase dashboard
2. Copy and run this SQL:

```sql
-- Create news_history table
CREATE TABLE IF NOT EXISTS news_history (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    url_hash TEXT UNIQUE NOT NULL,
    domain TEXT,
    title TEXT,
    content TEXT,
    prediction TEXT,
    confidence REAL,
    fake_probability REAL,
    real_probability REAL,
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_url_hash ON news_history(url_hash);
CREATE INDEX IF NOT EXISTS idx_checked_at ON news_history(checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_domain ON news_history(domain);
CREATE INDEX IF NOT EXISTS idx_prediction ON news_history(prediction);

-- Enable Row Level Security (optional but recommended)
ALTER TABLE news_history ENABLE ROW LEVEL SECURITY;

-- Create a policy to allow all operations (adjust as needed)
CREATE POLICY "Allow all operations" ON news_history
    FOR ALL USING (true);
```

### 4. Configure Your Application

1. Create folder `.streamlit` in your project root
2. Create file `.streamlit/secrets.toml`
3. Add your credentials:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-public-key-here"
```

### 5. Test Connection

Run your application:
```bash
streamlit run main.py
```

## 📊 Optional: Enable Realtime

If you want real-time updates:

1. Go to "Database" → "Replication"
2. Enable replication for `news_history` table
3. Toggle on the tables you want to sync

## 🔒 Security Best Practices

1. **Never commit** `.streamlit/secrets.toml` to Git
2. Add to `.gitignore`:
   ```
   .streamlit/secrets.toml
   .streamlit/
   ```

3. For production, use environment variables:
   ```bash
   export SUPABASE_URL="your-url"
   export SUPABASE_KEY="your-key"
   ```

## 🛠️ Troubleshooting

### Connection Error
- Check if your project is active (free tier pauses after 1 week)
- Verify credentials are correct
- Check if table exists in database

### Permission Error
- Ensure Row Level Security policies are set correctly
- Check if anon key has proper permissions

### Performance Issues
- Add indexes as shown in SQL above
- Consider pagination for large datasets
- Use Supabase Edge Functions for complex queries

## 📚 Advanced Features

### 1. Add Full-Text Search
```sql
-- Add full text search
ALTER TABLE news_history ADD COLUMN search_vector tsvector;

CREATE OR REPLACE FUNCTION news_history_search_trigger() RETURNS trigger AS $$
BEGIN
    NEW.search_vector := to_tsvector('indonesian', 
        COALESCE(NEW.title, '') || ' ' || 
        COALESCE(NEW.content, '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tsvectorupdate BEFORE INSERT OR UPDATE
ON news_history FOR EACH ROW EXECUTE FUNCTION news_history_search_trigger();
```

### 2. Add Analytics Views
```sql
-- Create view for domain statistics
CREATE OR REPLACE VIEW domain_stats AS
SELECT 
    domain,
    COUNT(*) as total_checks,
    SUM(CASE WHEN prediction = 'FAKE' THEN 1 ELSE 0 END) as fake_count,
    SUM(CASE WHEN prediction = 'REAL' THEN 1 ELSE 0 END) as real_count,
    AVG(confidence) as avg_confidence,
    ROUND(SUM(CASE WHEN prediction = 'FAKE' THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100, 2) as fake_rate
FROM news_history
GROUP BY domain
ORDER BY total_checks DESC;
```

## 🔗 Useful Links

- [Supabase Documentation](https://supabase.com/docs)
- [Supabase Python Client](https://github.com/supabase-community/supabase-py)
- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management)