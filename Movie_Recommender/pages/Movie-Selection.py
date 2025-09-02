import pandas as pd
import streamlit as st
from datetime import datetime
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import requests

# TMDB API Configuration
API_KEY = "6272a18b79461f7987d2f1047b3d1198"
API_BASE_URL = "https://api.themoviedb.org/3"

# Navigation bar function
def load_navbar():
    st.markdown("""
    <style>
    /* App background */
    html, body, .stApp {
        background: linear-gradient(135deg, #1e1e2f, #2e2e4e);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Navigation bar container */
    .navbar {
        background: linear-gradient(90deg, #2e2f47, #3c3e5f);
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
        margin-bottom: 2rem;
        position: sticky;
        top: 0;
        z-index: 999;
    }

    /* Logo styling */
    .navbar-logo a {
        font-size: 1.7rem;
        font-weight: 700;
        color: white;
        text-decoration: none;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .navbar-logo a:hover {
        color: #d8b4fe;
        transform: scale(1.03);
        text-shadow: 0 0 8px rgba(216, 180, 254, 0.4);
    }

    /* Navigation links */
    .navbar-links {
        display: flex;
        gap: 1rem;
    }

    .navbar-links a {
        color: white;
        text-decoration: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
        transition: all 0.3s ease;
    }

    .navbar-links a:hover {
        background-color: rgba(255, 255, 255, 0.07);
        transform: translateY(-2px);
        color: #d8b4fe;
        text-shadow: 0 0 8px rgba(216, 180, 254, 0.5);
    }

    /* Active link */
    .navbar-links a.active {
        background-color: rgba(216, 180, 254, 0.15);
        color: #d8b4fe;
    }

    /* Responsive behavior */
    @media (max-width: 768px) {
        .navbar {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.8rem;
        }

        .navbar-links {
            flex-direction: column;
            align-items: flex-start;
        }
    }
    </style>

    <div class="navbar">
        <div class="navbar-logo">
            <a href="/" target="_self">🎬 CineSearch Pro</a>
        </div>
        <div class="navbar-links">
            <a href="/" target="_self">🏠 Home</a>
            <a href="/Movie-Selection" target="_self" class="active">🔍 Movie Search</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        if year:
            params["year"] = int(year)
        
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        return results[0] if results else None
    except Exception:
        return None

@st.cache_data
def get_poster_for_movie(movie_title, year=None):
    """Get poster URL for a movie from TMDB"""
    tmdb_result = search_tmdb_by_title(movie_title, year)
    if tmdb_result and tmdb_result.get('poster_path'):
        return f"https://image.tmdb.org/t/p/w300{tmdb_result['poster_path']}"
    return None

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
        
        # Split genres into list
        df['genre_list'] = df['genres'].str.split('|')
        
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

if not movies_df.empty:
    # For search functionality (title + genres only)
    search_vectorizer, search_tfidf_matrix = get_search_vectorizer_and_matrix(movies_df)

# Page config with modern dark theme
st.set_page_config(
    page_title="MovieVerse - AI Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
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
        <div class="feature-chip">📅 Filter by year</div>
        <div class="feature-chip">🔍 NLP-powered search</div>
    </div>
""", unsafe_allow_html=True)

# Search box
search_query = st.text_input(
    "",
    placeholder="Search by movie title, genre, or keywords...",
    key="search_input",
    label_visibility="collapsed"
)

st.button("Find Movies", type="primary")

if search_query and not movies_df.empty:
    with st.spinner('🔍 Searching with NLP and vector similarity...'):
        # Use SEARCH vectorizer (title + genres only, NO tags)
        query_vec = search_vectorizer.transform([search_query])
        similarity_scores = cosine_similarity(query_vec, search_tfidf_matrix).flatten()
        top_indices = similarity_scores.argsort()[::-1][:50]
        results = movies_df.iloc[top_indices]
        
        # Filter out results with very low similarity scores
        results_with_scores = results.copy()
        results_with_scores['similarity'] = similarity_scores[top_indices]
        results = results_with_scores[results_with_scores['similarity'] > 0.01].head(20)

        if not results.empty:
            st.markdown(f'<div class="results-count">Showing {len(results)} results</div>', unsafe_allow_html=True)
            
            for _, result in results.iterrows():
                movie_id = result.get('movieId')
                title = result.get('clean_title', result.get('title', 'No Title'))
                year = result.get('year', '')
                avg_rating = result.get('avg_rating', 0)
                rating_count = result.get('rating_count', 0)
                genres = result.get('genre_list', [])
                similarity = result.get('similarity', 0)

                # Get poster URL from TMDB
                poster_url = get_poster_for_movie(title, year if pd.notna(year) else None)

                # Create poster HTML
                if poster_url:
                    poster_html = f'<img src="{poster_url}" class="movie-poster" alt="{title} Poster">'
                else:
                    poster_html = '<div class="movie-placeholder">🎬</div>'

                # Create clickable movie card with poster
                st.markdown(f"""
                    <a href="?movie_id={movie_id}" style="text-decoration: none; color: inherit;">
                        <div class="movie-card">
                            {poster_html}
                            <div class="movie-info">
                                <h3 class="movie-title">{title}</h3>
                                <div class="movie-meta">
                                    {f'<span class="release-year">📅 {int(year)}</span>' if pd.notna(year) else ''}
                                    {f'<span class="movie-rating">⭐ {avg_rating:.1f}</span>' if avg_rating > 0 else '<span class="movie-rating">⭐ Not rated</span>'}
                                    {f'<span class="rating-count">({int(rating_count)} ratings)</span>' if rating_count > 0 else ''}
                                    <span style="color: #10B981; font-size: 0.85rem;">Match: {similarity:.1%}</span>
                                </div>
                                <div style="margin-top: 10px;">
                                    {''.join([f'<span class="genre-badge">{g}</span>' for g in genres[:5] if g != "(no genres listed)"]) if genres else ''}
                                </div>
                            </div>
                        </div>
                    </a>
                """, unsafe_allow_html=True)
        else:
            st.warning("No movies found matching your search. Try different keywords!")
elif search_query:
    st.error("Unable to search - movie data not loaded properly.")