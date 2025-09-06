import pandas as pd
import streamlit as st
from datetime import datetime
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import requests
from Homepage import load_navbar

# TMDB API Configuration
API_KEY = "6272a18b79461f7987d2f1047b3d1198"
API_BASE_URL = "https://api.themoviedb.org/3"

# TMDB API Functions
@st.cache_data
def fetch_movie_details_by_tmdb_id(tmdb_id):
    """Fetch movie details from TMDB API"""
    try:
        url = f"{API_BASE_URL}/movie/{tmdb_id}"
        params = {
            "api_key": API_KEY,
            "language": "en-US",
            "append_to_response": "credits",
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error fetching movie details: {e}")
        return None

@st.cache_data
def search_tmdb_by_title(movie_title, year=None):
    """Search TMDB for a movie by title and optionally year"""
    try:
        url = f"{API_BASE_URL}/search/movie"
        params = {
            "api_key": API_KEY,
            "query": movie_title,
            "language": "en-US"
        }
        if year and pd.notna(year):
            params["year"] = int(year)
        
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        return results[0] if results else None
    except Exception as e:
        print(f"TMDB search error: {e}")
        return None

def clean_title_for_search(title):
    """Clean movie title for better TMDB search results"""
    # Remove common suffixes that might interfere with search
    title = re.sub(r'\s*\(.*?\)\s*$', '', title)  # Remove anything in parentheses at the end
    title = re.sub(r'\s*,\s*The\s*$', '', title, flags=re.IGNORECASE)  # Remove ", The" at the end
    title = re.sub(r'^\s*The\s+', '', title, flags=re.IGNORECASE)  # Remove "The " at the beginning
    title = title.strip()
    return title

@st.cache_data
def get_poster_for_movie(original_title, clean_title, year=None):
    """Get poster URL for a movie from TMDB with improved fallback strategies"""
    
    # Strategy 1: Try with clean title and year
    if clean_title and year and pd.notna(year):
        tmdb_result = search_tmdb_by_title(clean_title, year)
        if tmdb_result and tmdb_result.get('poster_path'):
            return f"https://image.tmdb.org/t/p/w300{tmdb_result['poster_path']}"
    
    # Strategy 2: Try with clean title only
    if clean_title:
        tmdb_result = search_tmdb_by_title(clean_title)
        if tmdb_result and tmdb_result.get('poster_path'):
            return f"https://image.tmdb.org/t/p/w300{tmdb_result['poster_path']}"
    
    # Strategy 3: Try with further cleaned title
    if clean_title:
        further_cleaned = clean_title_for_search(clean_title)
        if further_cleaned and further_cleaned != clean_title:
            tmdb_result = search_tmdb_by_title(further_cleaned, year if pd.notna(year) else None)
            if tmdb_result and tmdb_result.get('poster_path'):
                return f"https://image.tmdb.org/t/p/w300{tmdb_result['poster_path']}"
    
    # Strategy 4: Try with original title (includes year) - last resort
    if original_title and original_title != clean_title:
        # Extract just the title part from original title if it has year
        title_only = re.sub(r'\s*\(\d{4}\)\s*$', '', original_title).strip()
        if title_only:
            tmdb_result = search_tmdb_by_title(title_only, year if pd.notna(year) else None)
            if tmdb_result and tmdb_result.get('poster_path'):
                return f"https://image.tmdb.org/t/p/w300{tmdb_result['poster_path']}"
    
    return None

def is_valid_movie_title(title):
    """Check if a movie title is valid and not a placeholder"""
    if not title or not isinstance(title, str):
        return False
    
    title_lower = title.lower().strip()
    
    # List of invalid/placeholder titles to filter out
    invalid_titles = [
        'untitled',
        'no title',
        'unknown',
        'tba',
        'to be announced',
        'placeholder',
        'temp',
        'test',
        'sample'
    ]
    
    # Check if title starts with any invalid patterns
    for invalid in invalid_titles:
        if title_lower.startswith(invalid):
            return False
    
    # Check for very short titles (less than 2 characters)
    if len(title_lower) < 2:
        return False
    
    # Check for titles that are just numbers or special characters
    if re.match(r'^[\d\s\-_\.]+$', title_lower):
        return False
    
    return True

def has_sufficient_movie_data(movie_row):
    """Check if movie has sufficient data to display properly"""
    # Check for valid title
    title = movie_row.get('clean_title', movie_row.get('title', ''))
    if not is_valid_movie_title(title):
        return False
    
    # Check for valid genres (not just "(no genres listed)")
    genres = movie_row.get('genres', '')
    if not genres or genres == "(no genres listed)" or pd.isna(genres):
        return False
    
    # Check if it has a valid year (helps with TMDB lookup)
    year = movie_row.get('year')
    if pd.isna(year) or year == 0:
        # Allow movies without year only if they have other good data
        pass
    
    return True

@st.cache_data
def check_movie_quality(original_title, clean_title, year, genres):
    """Check if a movie meets quality standards for display"""
    # Check title validity
    if not is_valid_movie_title(clean_title) and not is_valid_movie_title(original_title):
        return False, None
    
    # Check genre validity
    if not genres or genres == "(no genres listed)":
        return False, None
    
    # Try to get poster
    poster_url = get_poster_for_movie(original_title, clean_title, year if pd.notna(year) else None)
    
    # If no poster AND problematic title, filter out
    if not poster_url and not is_valid_movie_title(clean_title):
        return False, None
    
    return True, poster_url

# Load movies data
@st.cache_data
def load_movie_data():
    try:
        # Load MovieLens dataset
        df = pd.read_csv("dataset/ml-latest-small/movies.csv")
        
        # Extract year from title (e.g., "Toy Story (1995)")
        df['year'] = df['title'].str.extract(r'\((\d{4})\)')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # Clean title (remove year from title for display)
        df['clean_title'] = df['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
        
        # Create genres as a string instead of a list to avoid hashing issues
        df['genres_str'] = df['genres'].str.replace('|', ' ')
        
        return df
    except FileNotFoundError:
        st.error("❌ movies.csv not found. Please check the dataset path.")
        return pd.DataFrame()

# Load tags data
@st.cache_data
def load_tags_data():
    """Load tags data from MovieLens dataset"""
    try:
        tags_df = pd.read_csv("dataset/ml-latest-small/tags.csv")
        
        # Clean and normalize tags
        tags_df['tag'] = tags_df['tag'].astype(str).str.lower().str.strip()
        
        # Remove very short tags (less than 2 characters) and common words
        tags_df = tags_df[tags_df['tag'].str.len() >= 2]
        
        # Aggregate all tags for each movie
        movie_tags = tags_df.groupby('movieId')['tag'].apply(
            lambda x: ' '.join(x.unique())  # Use unique to avoid duplicates
        ).reset_index()
        
        return tags_df, movie_tags
    except FileNotFoundError:
        st.warning("⚠️ tags.csv not found. Recommendations will use only titles and genres.")
        return pd.DataFrame(), pd.DataFrame()

# Load ratings and calculate average ratings
@st.cache_data
def load_ratings_with_stats():
    try:
        ratings = pd.read_csv("dataset/ml-latest-small/ratings.csv")
        
        # Calculate average rating and count for each movie
        rating_stats = ratings.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).round(2)
        rating_stats.columns = ['avg_rating', 'rating_count']
        rating_stats = rating_stats.reset_index()
        
        return ratings, rating_stats
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Load data
movies_df = load_movie_data()
# Create genre_list as a string representation to avoid hashing issues
movies_df['genre_list'] = movies_df['genres'].str.split('|').apply(lambda x: ', '.join(x))
tags_df, movie_tags_df = load_tags_data()
ratings_df, rating_stats_df = load_ratings_with_stats()

# Merge movies with rating stats
if not movies_df.empty and not rating_stats_df.empty:
    movies_df = movies_df.merge(rating_stats_df, on='movieId', how='left')
    movies_df['avg_rating'] = movies_df['avg_rating'].fillna(0)
    movies_df['rating_count'] = movies_df['rating_count'].fillna(0)

# Merge movies with tags
if not movies_df.empty and not movie_tags_df.empty:
    movies_df = movies_df.merge(movie_tags_df, on='movieId', how='left')
    movies_df['tag'] = movies_df['tag'].fillna('')
else:
    movies_df['tag'] = ''

# QUALITY FILTER: Remove movies with insufficient data
if not movies_df.empty:
    print(f"Before quality filter: {len(movies_df)} movies")
    
    # Apply quality filter
    movies_df = movies_df[movies_df.apply(has_sufficient_movie_data, axis=1)]
    
    print(f"After quality filter: {len(movies_df)} movies")

# Cache vectorizer and tf-idf matrix for SEARCH (title + genres only, NO tags)
@st.cache_resource
def get_search_vectorizer_and_matrix(dataframe):
    # Create search blob from title and genres ONLY (no tags for search)
    dataframe['title'] = dataframe['title'].fillna('')
    dataframe['genres'] = dataframe['genres'].fillna('')
    dataframe['clean_title'] = dataframe['clean_title'].fillna('')
    
    # Create searchable text from title and genres only
    dataframe['search_blob'] = (
        dataframe['title'] + ' ' + 
        dataframe['clean_title'] + ' ' + 
        dataframe['genres'].str.replace('|', ' ')
    )
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(dataframe['search_blob'])
    return vectorizer, tfidf_matrix

def calculate_exact_match_bonus(query, title, clean_title):
    """Calculate bonus score for exact or partial matches"""
    query_lower = query.lower().strip()
    title_lower = title.lower()
    clean_title_lower = clean_title.lower()
    
    # Exact match gets highest bonus
    if query_lower == clean_title_lower or query_lower == title_lower:
        return 1.0
    
    # Partial match at start of title gets medium bonus
    if clean_title_lower.startswith(query_lower) or title_lower.startswith(query_lower):
        return 0.5
    
    # Contains query gets small bonus
    if query_lower in clean_title_lower or query_lower in title_lower:
        return 0.2
    
    return 0.0

if not movies_df.empty:
    # For search functionality (title + genres only)
    search_vectorizer, search_tfidf_matrix = get_search_vectorizer_and_matrix(movies_df)

# Page config with modern dark theme
st.set_page_config(
    page_title="MovieVerse - AI Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for creative modern UI (simplified for MovieLens data)
st.markdown("""
    <style>
    :root {
        --primary: #b31b1b;
        --primary-hover: #d11f1f;
        --dark-bg: #0f0f15;
        --card-bg: #141414;
        --accent: #ff6b6b;
        --text: #ffffff;
        --text-secondary: #cccccc;
        --neon-glow: 0 0 10px rgba(179, 27, 27, 0.8);
    }
    
    html, body, .stApp {
        background: linear-gradient(135deg, #1e1e2f, #2e2e4e);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-bottom: 2.5rem;
        padding: 1rem 0;
    }
    
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    .search-title {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
        background: linear-gradient(90deg, #b31b1b, #ff6b6b, #b31b1b);
        background-size: 200% auto;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: gradientBG 3s ease infinite;
        text-shadow: 0 0 5px rgba(179, 27, 27, 0.3);
        letter-spacing: -1px;
    }
    
    .search-subtitle {
        color: var(--text-secondary);
        font-size: 1.2rem;
        max-width: 600px;
        margin: 0 auto;
        font-weight: 300;
        line-height: 1.6;
    }
    
    .search-box-container {
        background: rgba(20, 20, 20, 0.7);
        border-radius: 16px;
        padding: 3rem;
        margin: 0 auto 4rem;
        max-width: 700px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            var(--neon-glow);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .stTextInput>div>div>input {
        background-color: rgba(10, 10, 10, 0.8) !important;
        color: var(--text) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 12px !important;
        padding: 18px 24px !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: var(--primary) !important;
        box-shadow: 
            0 0 0 2px rgba(179, 27, 27, 0.3),
            0 0 20px rgba(179, 27, 27, 0.2) !important;
    }
    
    div[data-testid="stButton"] > button[kind="primary"] {
        background: var(--primary) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 2rem !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: var(--primary-hover) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4) !important;
    }
    
    .features-container {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin-bottom: 4rem;
        flex-wrap: wrap;
    }
    
    .feature-chip {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        background: rgba(20, 20, 20, 0.7);
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 1rem;
        border: 1px solid rgba(255,255,255,0.15);
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .feature-chip:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(179, 27, 27, 0.3);
        border-color: var(--primary);
    }
    
    .movie-card {
        display: flex;
        gap: 25px;
        padding: 25px;
        border-radius: 16px;
        background-color: rgba(20, 20, 20, 0.7);
        margin-bottom: 30px;
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }
    
    .movie-card:hover {
        transform: translateY(-8px) scale(1.01);
        box-shadow: 
            0 15px 35px rgba(0, 0, 0, 0.4),
            var(--neon-glow);
        border-color: var(--primary);
    }
    
    .movie-poster {
        border-radius: 12px;
        width: 150px;
        height: 225px;
        object-fit: cover;
        flex-shrink: 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .movie-placeholder {
        border-radius: 12px;
        width: 150px;
        height: 225px;
        background: linear-gradient(135deg, #333, #555);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #999;
        font-size: 3rem;
        flex-shrink: 0;
    }
    
    .movie-info {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    
    .movie-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, var(--text), var(--text-secondary));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        letter-spacing: -0.5px;
    }
    
    .movie-meta {
        display: flex;
        gap: 20px;
        color: var(--text-secondary);
        font-size: 1rem;
        align-items: center;
        flex-wrap: wrap;
    }
    
    .movie-rating {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 193, 7, 0.2);
        color: #ffc107;
        padding: 6px 14px;
        border-radius: 50px;
        font-weight: 600;
        border: 1px solid rgba(255, 193, 7, 0.3);
    }
    
    .genre-badge {
        display: inline-block;
        background: rgba(179, 27, 27, 0.15);
        color: var(--primary);
        padding: 4px 12px;
        border-radius: 50px;
        margin-right: 8px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        border: 1px solid rgba(179, 27, 27, 0.3);
    }
    
    .rating-count {
        color: var(--text-secondary);
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    .results-count {
        color: var(--text-secondary);
        margin-bottom: 2rem;
        font-size: 1.1rem;
        text-align: center;
        font-weight: 300;
    }
    
    .tags-info {
        background: rgba(30, 30, 50, 0.6);
        padding: 10px 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid rgba(100, 100, 150, 0.3);
    }
    
    .tags-title {
        color: #9CA3AF;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .tag-item {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        padding: 2px 8px;
        border-radius: 15px;
        margin: 2px 4px 2px 0;
        font-size: 0.75rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    @media (max-width: 768px) {
        .search-title {
            font-size: 2.2rem;
        }
        
        .movie-card {
            flex-direction: column;
            gap: 20px;
            padding: 20px;
        }
        
        .movie-poster, .movie-placeholder {
            width: 100%;
            height: auto;
            aspect-ratio: 2/3;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Check if we should redirect to details page
raw_param = st.query_params.get("movie_id")
detail_id = raw_param if isinstance(raw_param, str) else raw_param[0] if raw_param else None

# If movie_id is provided, redirect to 01_Movie_Details.py
if detail_id is not None:
    # Store movie_id in session state for the details page to access
    st.session_state.movie_id = detail_id
    # Redirect to the details page
    st.switch_page("pages/01_Movie_Details.py")

# Main search page
load_navbar()

# Main header with centered design
st.markdown("""
    <div class="header-container">
        <h1 class="search-title">Discover Films Just For You</h1>
    </div>
""", unsafe_allow_html=True)

# Enhanced feature chips to reflect search capabilities (no tags mentioned for search)
st.markdown("""
    <div class="features-container">
        <div class="feature-chip">🎬 Search by title</div>
        <div class="feature-chip">🎭 Browse by genre</div>
        <div class="feature-chip">📅 Search by year</div>
        <div class="feature-chip">🔍 NLP-powered search</div>
    </div>
""", unsafe_allow_html=True)

# Search box with proper label to avoid accessibility warning
search_query = st.text_input(
    "Search movies",
    placeholder="Search by movie title, genre, or keywords...",
    key="search_input",
    label_visibility="collapsed"
)

st.button("Find Movies", type="primary")

# FIXED SEARCH RESULTS SECTION
if search_query and not movies_df.empty:
    with st.spinner('🔍 Searching with NLP and vector similarity...'):
        # Use SEARCH vectorizer (title + genres only, NO tags)
        query_vec = search_vectorizer.transform([search_query])
        similarity_scores = cosine_similarity(query_vec, search_tfidf_matrix).flatten()
        
        # Get top 50 results first
        top_indices = similarity_scores.argsort()[::-1][:50]
        results = movies_df.iloc[top_indices].copy()
        
        # Add similarity scores
        results['similarity'] = similarity_scores[top_indices]
        
        # Calculate exact match bonus for better ranking
        results['exact_match_bonus'] = results.apply(
            lambda row: calculate_exact_match_bonus(search_query, row['title'], row['clean_title']), 
            axis=1
        )
        
        # Create combined score: similarity + exact match bonus
        results['combined_score'] = results['similarity'] + results['exact_match_bonus']
        
        # Sort by combined score (exact matches will appear first even with same similarity)
        results = results.sort_values('combined_score', ascending=False)
        
        # Apply lower threshold for common words like "the" to show more results
        # Use dynamic threshold: lower for short queries, higher for longer ones
        min_threshold = max(0.001, 0.01 - (len(search_query.split()) * 0.002))
        results = results[results['similarity'] > min_threshold].head(20)

        if not results.empty:
            st.markdown(f'<div class="results-count">Showing {len(results)} results</div>', unsafe_allow_html=True)
            
            for _, result in results.iterrows():
                movie_id = result.get('movieId')
                title = result.get('clean_title', result.get('title', 'No Title'))
                original_title = result.get('title', title)  # Keep original title with year
                year = result.get('year', '')
                avg_rating = result.get('avg_rating', 0)
                rating_count = result.get('rating_count', 0)
                
                # FIX: Properly handle genres from the original 'genres' column
                raw_genres = result.get('genres', '')
                if raw_genres and raw_genres != "(no genres listed)":
                    # Split by pipe and clean up
                    genre_list = [g.strip() for g in raw_genres.split('|') if g.strip()]
                else:
                    genre_list = []
                
                similarity = result.get('similarity', 0)

                # Get poster URL from TMDB using improved function (only when displaying)
                poster_url = get_poster_for_movie(original_title, title, year if pd.notna(year) else None)

                # Create poster HTML with better error handling and fallback
                if poster_url:
                    poster_html = f'<img src="{poster_url}" class="movie-poster" alt="{title} Poster" loading="lazy" onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'flex\';"><div class="movie-placeholder" style="display: none;">🎬</div>'
                else:
                    poster_html = '<div class="movie-placeholder">🎬</div>'

                # Escape any special characters in title to prevent HTML issues
                safe_title = title.replace('"', '&quot;').replace("'", "&#39;").replace('<', '&lt;').replace('>', '&gt;')
                
                # FIX: Build genre badges with proper genre names
                genre_badges = ''
                if genre_list:
                    # Take first 5 genres and create badges
                    safe_genres = [g.replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;') for g in genre_list[:5]]
                    genre_badges = ''.join([f'<span class="genre-badge">{g}</span>' for g in safe_genres])

                # Create clickable movie card with poster - maintain original horizontal design
                year_display = f'<span class="release-year">📅 {int(year)}</span>' if pd.notna(year) else ''
                rating_display = f'<span class="movie-rating">⭐ {avg_rating:.1f}</span>' if avg_rating > 0 else '<span class="movie-rating">⭐ Not rated</span>'
                count_display = f'<span class="rating-count">({int(rating_count)} ratings)</span>' if rating_count > 0 else ''
                match_display = f'<span style="color: #10B981; font-size: 0.85rem;">Match: {similarity:.1%}</span>'
                
                # Build the complete movie card HTML with proper structure
                movie_card_html = f"""
                <a href="?movie_id={movie_id}" style="text-decoration: none; color: inherit;">
                    <div class="movie-card">
                        {poster_html}
                        <div class="movie-info">
                            <h3 class="movie-title">{safe_title}</h3>
                            <div class="movie-meta">
                                {year_display}
                                {rating_display}
                                {count_display}
                                {match_display}
                            </div>
                            <div style="margin-top: 10px;">
                                {genre_badges}
                            </div>
                        </div>
                    </div>
                </a>
                """
                
                st.markdown(movie_card_html, unsafe_allow_html=True)
        else:
            st.warning("No movies found matching your search. Try different keywords!")
elif search_query:
    st.error("Unable to search - movie data not loaded properly.")