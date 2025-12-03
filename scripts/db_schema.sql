-- PostgreSQL Database Schema for Bank Reviews Analysis
-- Task 3: Database Design and Implementation

-- =============================================
-- DROP TABLES IF THEY EXIST (FOR CLEAN SETUP)
-- =============================================

DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS banks;
DROP TABLE IF EXISTS app_versions;
DROP TABLE IF EXISTS themes;
DROP TABLE IF EXISTS review_themes;

-- =============================================
-- CREATE BANKS TABLE
-- =============================================

CREATE TABLE banks (
    bank_id SERIAL PRIMARY KEY,
    bank_code VARCHAR(10) UNIQUE NOT NULL,
    bank_name VARCHAR(100) NOT NULL,
    app_name VARCHAR(200),
    app_id VARCHAR(100) UNIQUE,
    current_rating DECIMAL(3,2),
    total_reviews INTEGER DEFAULT 0,
    total_ratings INTEGER DEFAULT 0,
    installs VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- CREATE REVIEWS TABLE
-- =============================================

CREATE TABLE reviews (
    review_id VARCHAR(100) PRIMARY KEY,
    bank_id INTEGER NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    review_text TEXT NOT NULL,
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    sentiment_label VARCHAR(10) CHECK (sentiment_label IN ('positive', 'negative', 'neutral')),
    sentiment_score DECIMAL(4,3) CHECK (sentiment_score BETWEEN 0 AND 1),
    review_date DATE NOT NULL,
    review_year INTEGER,
    review_month INTEGER,
    user_name VARCHAR(200),
    thumbs_up INTEGER DEFAULT 0,
    reply_content TEXT,
    app_version VARCHAR(50),
    source VARCHAR(50) DEFAULT 'Google Play',
    text_length INTEGER,
    has_speed_issue BOOLEAN DEFAULT FALSE,
    has_feature_request BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- CREATE THEMES TABLE
-- =============================================

CREATE TABLE themes (
    theme_id SERIAL PRIMARY KEY,
    theme_name VARCHAR(100) NOT NULL,
    theme_description TEXT,
    theme_category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- CREATE REVIEW_THEMES JUNCTION TABLE
-- =============================================

CREATE TABLE review_themes (
    review_id VARCHAR(100) REFERENCES reviews(review_id) ON DELETE CASCADE,
    theme_id INTEGER REFERENCES themes(theme_id) ON DELETE CASCADE,
    confidence_score DECIMAL(4,3) DEFAULT 1.0,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (review_id, theme_id)
);

-- =============================================
-- CREATE INDEXES FOR PERFORMANCE
-- =============================================

-- Indexes for banks table
CREATE INDEX idx_banks_bank_code ON banks(bank_code);
CREATE INDEX idx_banks_app_id ON banks(app_id);

-- Indexes for reviews table
CREATE INDEX idx_reviews_bank_id ON reviews(bank_id);
CREATE INDEX idx_reviews_rating ON reviews(rating);
CREATE INDEX idx_reviews_sentiment ON reviews(sentiment_label);
CREATE INDEX idx_reviews_date ON reviews(review_date);
CREATE INDEX idx_reviews_sentiment_score ON reviews(sentiment_score);
CREATE INDEX idx_reviews_bank_sentiment ON reviews(bank_id, sentiment_label);
CREATE INDEX idx_reviews_bank_rating ON reviews(bank_id, rating);
CREATE INDEX idx_reviews_date_range ON reviews(review_year, review_month);

-- Indexes for themes table
CREATE INDEX idx_themes_name ON themes(theme_name);

-- Indexes for review_themes table
CREATE INDEX idx_review_themes_review_id ON review_themes(review_id);
CREATE INDEX idx_review_themes_theme_id ON review_themes(theme_id);

-- =============================================
-- CREATE VIEWS FOR ANALYSIS
-- =============================================

-- View 1: Bank Summary Statistics
CREATE OR REPLACE VIEW bank_summary AS
SELECT 
    b.bank_id,
    b.bank_code,
    b.bank_name,
    b.current_rating,
    b.total_reviews,
    b.total_ratings,
    b.installs,
    COUNT(r.review_id) AS stored_reviews,
    ROUND(AVG(r.rating)::numeric, 2) AS avg_rating,
    ROUND(AVG(r.sentiment_score)::numeric, 3) AS avg_sentiment,
    SUM(CASE WHEN r.sentiment_label = 'positive' THEN 1 ELSE 0 END) AS positive_reviews,
    SUM(CASE WHEN r.sentiment_label = 'negative' THEN 1 ELSE 0 END) AS negative_reviews,
    SUM(CASE WHEN r.sentiment_label = 'neutral' THEN 1 ELSE 0 END) AS neutral_reviews,
    SUM(CASE WHEN r.rating = 5 THEN 1 ELSE 0 END) AS five_star_reviews,
    SUM(CASE WHEN r.rating = 1 THEN 1 ELSE 0 END) AS one_star_reviews,
    SUM(CASE WHEN r.has_speed_issue = TRUE THEN 1 ELSE 0 END) AS speed_issues,
    SUM(CASE WHEN r.has_feature_request = TRUE THEN 1 ELSE 0 END) AS feature_requests
FROM banks b
LEFT JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_id, b.bank_code, b.bank_name, b.current_rating, b.total_reviews, b.total_ratings, b.installs;

-- View 2: Monthly Sentiment Trends
CREATE OR REPLACE VIEW monthly_sentiment_trends AS
SELECT 
    b.bank_name,
    r.review_year,
    r.review_month,
    COUNT(r.review_id) AS review_count,
    ROUND(AVG(r.rating)::numeric, 2) AS avg_rating,
    ROUND(AVG(r.sentiment_score)::numeric, 3) AS avg_sentiment,
    SUM(CASE WHEN r.sentiment_label = 'positive' THEN 1 ELSE 0 END) AS positive_count,
    SUM(CASE WHEN r.sentiment_label = 'negative' THEN 1 ELSE 0 END) AS negative_count,
    SUM(CASE WHEN r.sentiment_label = 'neutral' THEN 1 ELSE 0 END) AS neutral_count
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name, r.review_year, r.review_month
ORDER BY b.bank_name, r.review_year DESC, r.review_month DESC;

-- View 3: Theme Analysis by Bank
CREATE OR REPLACE VIEW theme_analysis AS
SELECT 
    b.bank_name,
    t.theme_name,
    t.theme_description,
    COUNT(rt.review_id) AS theme_frequency,
    ROUND(AVG(r.sentiment_score)::numeric, 3) AS avg_sentiment_for_theme,
    ROUND(AVG(r.rating)::numeric, 2) AS avg_rating_for_theme
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
JOIN review_themes rt ON r.review_id = rt.review_id
JOIN themes t ON rt.theme_id = t.theme_id
GROUP BY b.bank_name, t.theme_id, t.theme_name, t.theme_description
ORDER BY b.bank_name, theme_frequency DESC;

-- View 4: Rating Distribution
CREATE OR REPLACE VIEW rating_distribution AS
SELECT 
    b.bank_name,
    r.rating,
    COUNT(r.review_id) AS review_count,
    ROUND(COUNT(r.review_id) * 100.0 / SUM(COUNT(r.review_id)) OVER (PARTITION BY b.bank_name), 2) AS percentage
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name, r.rating
ORDER BY b.bank_name, r.rating DESC;

-- =============================================
-- CREATE FUNCTIONS
-- =============================================

-- Function to update bank statistics
CREATE OR REPLACE FUNCTION update_bank_statistics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update the updated_at timestamp
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to get review statistics
CREATE OR REPLACE FUNCTION get_review_stats(bank_id_param INTEGER)
RETURNS TABLE(
    total_reviews BIGINT,
    avg_rating DECIMAL(4,2),
    avg_sentiment DECIMAL(4,3),
    positive_count BIGINT,
    negative_count BIGINT,
    neutral_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT,
        ROUND(AVG(rating)::numeric, 2),
        ROUND(AVG(sentiment_score)::numeric, 3),
        SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END)::BIGINT,
        SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END)::BIGINT,
        SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END)::BIGINT
    FROM reviews
    WHERE bank_id = bank_id_param;
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- CREATE TRIGGERS
-- =============================================

-- Trigger for banks table
CREATE TRIGGER update_banks_timestamp
    BEFORE UPDATE ON banks
    FOR EACH ROW
    EXECUTE FUNCTION update_bank_statistics();

-- Trigger for reviews table
CREATE TRIGGER update_reviews_timestamp
    BEFORE UPDATE ON reviews
    FOR EACH ROW
    EXECUTE FUNCTION update_bank_statistics();

-- =============================================
-- INSERT SAMPLE DATA FOR THEMES
-- =============================================

INSERT INTO themes (theme_name, theme_description, theme_category) VALUES
('Account Access Issues', 'Problems related to accessing accounts, login failures, or security issues', 'Technical'),
('Transaction Performance', 'Issues related to transaction speed, processing time, or delays', 'Performance'),
('User Interface & Experience', 'Feedback about app design, usability, and user experience', 'UI/UX'),
('App Stability & Bugs', 'Reports of app crashes, bugs, errors, or stability issues', 'Technical'),
('Customer Support', 'Feedback about customer service quality and responsiveness', 'Service'),
('Feature Requests', 'User requests for new features or enhancements', 'Enhancement'),
('Financial Services', 'Feedback on banking services, fees, and financial operations', 'Service'),
('Registration & Onboarding', 'Issues related to initial app registration and account setup', 'Technical'),
('Notifications & Alerts', 'Feedback about app notifications and alerts', 'Feature'),
('Security Concerns', 'User concerns about app security and data protection', 'Security');

-- =============================================
-- COMMIT THE SCHEMA
-- =============================================

COMMIT;

-- =============================================
-- VERIFICATION QUERIES
-- =============================================

-- Query to verify table creation
SELECT 
    table_name, 
    COUNT(*) as column_count
FROM information_schema.columns 
WHERE table_schema = 'public'
GROUP BY table_name
ORDER BY table_name;

-- Query to check data counts
SELECT 
    'banks' as table_name,
    COUNT(*) as row_count
FROM banks
UNION ALL
SELECT 
    'reviews' as table_name,
    COUNT(*) as row_count
FROM reviews
UNION ALL
SELECT 
    'themes' as table_name,
    COUNT(*) as row_count
FROM themes
UNION ALL
SELECT 
    'review_themes' as table_name,
    COUNT(*) as row_count
FROM review_themes;