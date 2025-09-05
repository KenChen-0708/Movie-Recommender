import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import requests
from datetime import datetime
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import re
from datetime import datetime, timedelta
import ast
import concurrent.futures

# Page configuration
st.set_page_config(
    page_title="CineSearch Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API configuration
API_KEY = "6272a18b79461f7987d2f1047b3d1198"
API_BASE_URL = "https://api.themoviedb.org/3"

# Helper function to generate star display for MovieLens ratings (1-5 scale)
def generate_star_rating(rating, max_rating=5):
    """Generate star display for ratings. Returns HTML for stars including half stars."""
    if pd.isna(rating) or rating == 0:
        return "☆☆☆☆☆"
    
    # Ensure rating is within valid range
    rating = max(0, min(rating, max_rating))
    
    full_stars = int(rating)
    half_star = 1 if (rating - full_stars) >= 0.5 else 0
    empty_stars = max_rating - full_stars - half_star
    
    stars_html = "★" * full_stars
    if half_star:
        stars_html += "⭐"  # Half star (can be replaced with a proper half-star icon)
    stars_html += "☆" * empty_stars
    
    return stars_html

def generate_star_rating_html(rating, max_rating=5):
    """Generate HTML star display with proper styling"""
    if pd.isna(rating) or rating == 0:
        return '<span class="star-rating">☆☆☆☆☆</span>'
    
    rating = max(0, min(rating, max_rating))
    full_stars = int(rating)
    has_half = (rating - full_stars) >= 0.5
    
    html = '<span class="star-rating">'
    
    # Full stars
    for i in range(full_stars):
        html += '<span class="star full">★</span>'
    
    # Half star
    if has_half:
        html += '<span class="star half">⭐</span>'
        empty_count = max_rating - full_stars - 1
    else:
        empty_count = max_rating - full_stars
    
    # Empty stars
    for i in range(empty_count):
        html += '<span class="star empty">☆</span>'
    
    html += f' <span class="rating-number">({rating:.1f}/5)</span></span>'
    
    return html

# Load MovieLens datasets
@st.cache_data
def load_movielens_data():
    """Load MovieLens datasets and merge them"""
    try:
        # Load ratings data
        ratings_df = pd.read_csv("dataset/ml-latest-small/ratings.csv")
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        
        # Load movies data
        movies_df = pd.read_csv("dataset/ml-latest-small/movies.csv")
        
        # Load links data for TMDB IDs
        links_df = pd.read_csv("dataset/ml-latest-small/links.csv")
        
        # Load tags data
        tags_df = pd.read_csv("dataset/ml-latest-small/tags.csv")
        tags_df['timestamp'] = pd.to_datetime(tags_df['timestamp'], unit='s')
        
        # Merge datasets
        merged_df = movies_df.merge(links_df, on='movieId', how='left')
        
        # Extract year from title for movies without release dates
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)')
        
        # Validate data
        if ratings_df.empty or movies_df.empty or merged_df.empty:
            st.warning("MovieLens dataset appears to be empty. Please check the data files.")
            return None, None, None, None, None, None
            
        return ratings_df, movies_df, links_df, merged_df, tags_df, movies_df['year']
    except FileNotFoundError as e:
        st.error(f"❌ MovieLens dataset files not found: {e}")
        st.info("Please ensure the ml-latest-small folder is in the dataset directory")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading MovieLens data: {e}")
        return None, None, None, None, None, None

# New MovieLens-based recommendation functions
@st.cache_data
def get_trending_movies_movielens(ratings_df, merged_df, days):
    """Get trending movies based on recent ratings activity"""
    try:
        if ratings_df is None or merged_df is None:
            return []
        
        # Calculate cutoff date for recent activity
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter recent ratings
        recent_ratings = ratings_df[ratings_df['timestamp'] >= cutoff_date]
        
        if recent_ratings.empty:
            # If no recent ratings, use all ratings from last 365 days
            cutoff_date = datetime.now() - timedelta(days=365)
            recent_ratings = ratings_df[ratings_df['timestamp'] >= cutoff_date]
        
        # Calculate average rating and count for recent ratings
        trending_stats = recent_ratings.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        trending_stats.columns = ['movieId', 'avg_rating', 'recent_ratings']
        
        # Merge with movie info and links
        trending_movies = trending_stats.merge(merged_df, on='movieId', how='left')
        
        # Sort by recent activity and get top 30
        trending_movies = trending_movies.sort_values('recent_ratings', ascending=False).head(30)
        
        # Convert to API-like format
        movies_list = []
        for _, movie in trending_movies.iterrows():
            tmdb_id = movie.get('tmdbId')
            if pd.notna(tmdb_id) and tmdb_id > 0:
                movies_list.append({
                    'id': str(int(tmdb_id)),
                    'title': movie.get('title', 'Unknown Title'),
                    'vote_average': round(movie.get('avg_rating', 0), 1),  # MovieLens rating (1-5)
                    'poster_url': None,  # Will be fetched from TMDB API
                    'source': 'movielens',
                    'movieId': movie.get('movieId'),
                    'recent_ratings': movie.get('recent_ratings', 0),
                    'movielens_rating': movie.get('avg_rating', 0)
                })
        
        return movies_list
    except Exception as e:
        st.error(f"Error getting trending movies: {e}")
        return []

@st.cache_data
def get_popular_movies_movielens(ratings_df, merged_df):
    """Get popular movies based on total rating count"""
    try:
        if ratings_df is None or merged_df is None:
            return []
        
        # Calculate average rating and total count per movie
        popularity_stats = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        popularity_stats.columns = ['movieId', 'avg_rating', 'total_ratings']
        
        # Merge with movie info and links
        popular_movies = popularity_stats.merge(merged_df, on='movieId', how='left')
        
        # Sort by total ratings and get top 30
        popular_movies = popular_movies.sort_values('total_ratings', ascending=False).head(30)
        
        # Convert to API-like format
        movies_list = []
        for _, movie in popular_movies.iterrows():
            tmdb_id = movie.get('tmdbId')
            if pd.notna(tmdb_id) and tmdb_id > 0:
                movies_list.append({
                    'id': str(int(tmdb_id)),
                    'title': movie.get('title', 'Unknown Title'),
                    'vote_average': round(movie.get('avg_rating', 0), 1),  # MovieLens rating (1-5)
                    'poster_url': None,  # Will be fetched from TMDB API
                    'source': 'movielens',
                    'movieId': movie.get('movieId'),
                    'total_ratings': movie.get('total_ratings', 0),
                    'movielens_rating': movie.get('avg_rating', 0)
                })
        
        return movies_list
    except Exception as e:
        st.error(f"Error getting popular movies: {e}")
        return []

@st.cache_data
def get_top_rated_movies_movielens(ratings_df, merged_df, min_ratings=20):
    """Get top rated movies based on average rating with minimum rating threshold"""
    try:
        if ratings_df is None or merged_df is None:
            return []
        
        # Calculate average rating and count per movie
        rating_stats = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        rating_stats.columns = ['movieId', 'avg_rating', 'rating_count']
        
        # Filter by minimum rating count to avoid bias
        qualified_movies = rating_stats[rating_stats['rating_count'] >= min_ratings]
        
        # Merge with movie info and links
        top_rated_movies = qualified_movies.merge(merged_df, on='movieId', how='left')
        
        # Filter out movies without TMDB ID
        top_rated_movies = top_rated_movies[
            (top_rated_movies['tmdbId'].notna()) & 
            (top_rated_movies['tmdbId'] > 0)
        ]
        
        # Sort by average rating and get top 30
        top_rated_movies = top_rated_movies.sort_values('avg_rating', ascending=False).head(30)
        
        # Convert to API-like format
        movies_list = []
        for _, movie in top_rated_movies.iterrows():
            tmdb_id = movie.get('tmdbId')
            movies_list.append({
                'id': str(int(tmdb_id)),
                'title': movie.get('title', 'Unknown Title'),
                'vote_average': round(movie.get('avg_rating', 0), 1),  # MovieLens rating (1-5)
                'poster_url': None,  # Will be fetched from TMDB API
                'source': 'movielens',
                'movieId': movie.get('movieId'),
                'avg_rating': movie.get('avg_rating', 0),
                'rating_count': movie.get('rating_count', 0),
                'movielens_rating': movie.get('avg_rating', 0)
            })
        
        return movies_list
    except Exception as e:
        st.error(f"Error getting top rated movies: {e}")
        return []

def fetch_movie_poster_and_details(tmdb_id):
    """Fetch movie poster and basic details from TMDB API"""
    try:
        url = f"{API_BASE_URL}/movie/{tmdb_id}"
        params = {
            'api_key': API_KEY,
            'language': 'en-US'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return {
            'poster_url': f"https://image.tmdb.org/t/p/w300{data.get('poster_path', '')}" if data.get('poster_path') else None,
            'tmdb_vote_average': data.get('vote_average', 0),  # TMDB rating (1-10)
            'release_date': data.get('release_date', ''),
            'overview': data.get('overview', '')
        }
    except Exception as e:
        return {
            'poster_url': None,
            'tmdb_vote_average': 0,
            'release_date': '',
            'overview': ''
        }

def enhance_movie_with_tmdb(movie):
    """
    Enhance a single movie with TMDB data (fetch posters, ratings, etc.)
    """
    tmdb_data = fetch_movie_poster_and_details(movie['id'])  # Assume this is your TMDB fetching function
    movie['poster_url'] = tmdb_data['poster_url']  # Add TMDB data to movie
    movie['tmdb_rating'] = tmdb_data['tmdb_vote_average']
    return movie

def enhance_movies_concurrently(movies_list):
    """
    Enhance a list of movies concurrently using ThreadPoolExecutor
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        enhanced_movies = list(executor.map(enhance_movie_with_tmdb, movies_list))
    return enhanced_movies
    
# Modified fetch_filtered_movies function to include AI recommendations
def fetch_filtered_movies_with_ai(genre_ids=None, release_date_gte=None, release_date_lte=None, language=None):
    """Enhanced filtered movies with AI recommendations"""
    
    # First try to get from API
    api_movies = fetch_filtered_movies(genre_ids, release_date_gte, release_date_lte, language)
    
    return api_movies

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

    /* Active link (optional logic can assign class='active') */
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
            <a href="/Movie-Selection" target="_self">🔍 Movie Search</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Helper function to safely parse JSON-like strings
def safe_parse_json(json_str):
    try:
        if pd.isna(json_str) or json_str == '':
            return []
        return ast.literal_eval(json_str)
    except:
        return []

# Helper function to extract names from genre/cast data
def extract_names(data_list, key='name'):
    if not data_list:
        return []
    return [item.get(key, '') for item in data_list if isinstance(item, dict)]

# Fetch genres from TMDB API
@st.cache_data
def fetch_genres():
    try:
        url = f"{API_BASE_URL}/genre/movie/list"
        params = {
            'api_key': API_KEY,
            'language': 'en-US'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('genres', [])
    except Exception as e:
        st.error(f"Error fetching genres: {e}")
        return []

# Fetch languages from TMDB API
@st.cache_data
def fetch_languages():
    try:
        url = f"{API_BASE_URL}/configuration/languages"
        params = {
            'api_key': API_KEY
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Define a function to check if a string is composed of only "?" characters
        def is_invalid(value):
            return value and all(c == '?' for c in value)
        
        # Filter out languages with invalid names or iso_639_1 codes
        valid_languages = [
            lang for lang in data
            if lang.get('name') and not is_invalid(lang.get('name')) and len(lang.get('iso_639_1', '')) == 2 and not is_invalid(lang.get('iso_639_1'))
        ]
        
        return sorted(valid_languages, key=lambda x: x.get('name', ''))
        
    except Exception as e:
        st.error(f"Error fetching languages: {e}")
        return []


# Fetch filtered movies from TMDB API
@st.cache_data
def fetch_filtered_movies(genre_ids=None, release_date_gte=None, release_date_lte=None, language=None):
    try:
        url = f"{API_BASE_URL}/discover/movie"
        params = {
            'api_key': API_KEY,
            'language': 'en-US',
            'sort_by': 'vote_average.desc',
            'vote_count.gte': 100,  # Minimum vote count for better recommendations
            'page': 1,
            'per_page': 20
        }
        
        # Add filters if provided
        if genre_ids:
            params['with_genres'] = ','.join(map(str, genre_ids))
        if release_date_gte:
            params['primary_release_date.gte'] = release_date_gte
        if release_date_lte:
            params['primary_release_date.lte'] = release_date_lte
        if language and language != 'all':
            params['with_original_language'] = language
        
        # Get first page
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        movies = []
        # Get movies from first page
        for movie in data['results']:
            if movie.get('poster_path') and not movie.get('adult', False):
                movies.append({
                    'id': str(movie.get('id')),
                    'title': movie.get('title', 'Unknown Title'),
                    'vote_average': movie.get('vote_average', 0),
                    'poster_url': f"https://image.tmdb.org/t/p/w300{movie['poster_path']}",
                    'release_date': movie.get('release_date', ''),
                    'source': 'api'
                })
        
        # Get second page if we need more movies
        if len(movies) < 40 and data.get('total_pages', 1) > 1:
            params['page'] = 2
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for movie in data['results']:
                if len(movies) >= 40:
                    break
                if movie.get('poster_path') and not movie.get('adult', False):
                    movies.append({
                        'id': str(movie.get('id')),
                        'title': movie.get('title', 'Unknown Title'),
                        'vote_average': movie.get('vote_average', 0),
                        'poster_url': f"https://image.tmdb.org/t/p/w300{movie['poster_path']}",
                        'release_date': movie.get('release_date', ''),
                        'source': 'api'
                    })
        
        return movies[:40]  # Return maximum 40 movies
    except Exception as e:
        st.error(f"Error fetching filtered movies: {e}")
        return []
    
# Fetch trending movies from API
@st.cache_data
def fetch_trending_movies():
    try:
        url = f"{API_BASE_URL}/trending/movie/day"
        params = {
            'api_key': API_KEY,
            'language': 'en-US'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        movies = []
        for movie in data['results'][:30]:  # Get top 30
            if movie.get('poster_path') and not movie.get('adult', False):
                movies.append({
                    'id': str(movie.get('id')),
                    'title': movie.get('title', 'Unknown Title'),
                    'vote_average': movie.get('vote_average', 0),
                    'poster_url': f"https://image.tmdb.org/t/p/w300{movie['poster_path']}",
                    'source': 'api'
                })
        return movies
    except Exception as e:
        st.error(f"Error fetching trending movies: {e}")
        return []

# Fetch now playing movies from API
@st.cache_data
def fetch_now_playing_movies():
    try:
        url = f"{API_BASE_URL}/movie/now_playing"
        params = {
            'api_key': API_KEY,
            'language': 'en-US',
            'page': 1
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        movies = []
        for movie in data['results'][:30]:  # Get top 30
            if movie.get('poster_path') and not movie.get('adult', False):
                movies.append({
                    'id': str(movie.get('id')),
                    'title': movie.get('title', 'Unknown Title'),
                    'vote_average': movie.get('vote_average', 0),
                    'poster_url': f"https://image.tmdb.org/t/p/w300{movie['poster_path']}",
                    'source': 'api'
                })
        return movies
    except Exception as e:
        st.error(f"Error fetching now playing movies: {e}")
        return []

# Custom CSS for modern UI (enhanced from original search page)
def load_custom_css():
    st.markdown("""
        <style>
        :root {
            --primary: #00d4ff;
            --dark-bg: #0f0f15;
            --card-bg: #1a1a25;
            --text: #f0f0f0;
            --text-secondary: #b0b0b0;
        }
        
        html, body, .stApp {
            background-color: var(--dark-bg);
            color: var(--text);
        }
        
        .detail-container {
            background-color: var(--card-bg);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            border: 1px solid #2a2a35;
        }
        
        .back-button {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 16px;
            background: var(--card-bg);
            border-radius: 8px;
            margin-bottom: 20px;
            transition: all 0.2s ease;
            border: 1px solid #333;
        }
        
        .back-button:hover {
            background: #252535;
            border-color: var(--primary);
        }
        
        .detail-poster {
            border-radius: 12px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
        }
        
        .tagline {
            font-style: italic;
            color: var(--primary);
            margin-bottom: 10px;
        }
        
        .detail-section {
            margin-bottom: 30px;
        }
        
        .genre-badge {
            display: inline-block;
            background: rgba(0, 212, 255, 0.15);
            color: var(--primary);
            padding: 4px 12px;
            border-radius: 50px;
            margin-right: 8px;
            margin-bottom: 8px;
            font-size: 0.8rem;
        }
        
        .rating-display {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: rgba(255, 193, 7, 0.15);
            color: #ffc107;
            padding: 8px 16px;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        /* Star rating styles */
        .star-rating {
            font-size: 1.2rem;
            margin-bottom: 10px;
        }
        
        .star-rating .star {
            margin-right: 2px;
        }
        
        .star-rating .star.full {
            color: #ffc107;
        }
        
        .star-rating .star.half {
            color: #ffc107;
        }
        
        .star-rating .star.empty {
            color: #6c757d;
        }
        
        .star-rating .rating-number {
            margin-left: 8px;
            font-size: 0.9rem;
            color: #b0b0b0;
        }
        
        .filter-container {
            background: var(--card-bg);
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 30px;
            border: 1px solid #2a2a35;
        }
        
        .filter-row {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .filter-group {
            flex: 1;
            min-width: 200px;
        }
        
        .filter-label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--text);
        }
        
        .filter-button {
            background: linear-gradient(135deg, var(--primary), #0099cc);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 10px;
        }
        
        .filter-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
        }
        
        .clear-button {
            background: #dc3545;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 10px;
            margin-left: 10px;
        }
        
        .clear-button:hover {
            background: #c82333;
            transform: translateY(-2px);
        }
        
        .filtered-results {
            margin-top: 30px;
        }
        
        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .results-count {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)

# Generate carousel HTML with clickable cards using anchor tags and star ratings
def generate_carousel_html(movies, carousel_id, title):
    poster_html = f"""
<style>
.carousel-container {{
    position: relative;
    width: 100%;
    margin: 0 auto;
    padding: 20px 0;
}}

.carousel-wrapper {{
    position: relative;
    overflow: hidden;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}}

.carousel-track {{
    display: flex;
    transition: transform 0.5s ease;
    gap: 1%;
    padding: 20px 0.5%;
}}

.movie-card {{
    flex: 0 0 19.2%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    text-decoration: none;
    color: inherit;
    display: block;
}}

.movie-card:hover {{
    transform: translateY(-10px) scale(1.05);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
    text-decoration: none;
    color: inherit;
}}

.movie-card:visited {{
    color: inherit;
    text-decoration: none;
}}

.movie-card:focus {{
    outline: 2px solid #d8b4fe;
    outline-offset: 2px;
}}

.movie-poster {{
    width: 100%;
    height: 320px;
    object-fit: cover;
    transition: transform 0.3s ease;
}}

.movie-card:hover .movie-poster {{
    transform: scale(1.1);
}}

.movie-info {{
    padding: 15px;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
}}

.movie-title {{
    font-size: 14px;
    font-weight: bold;
    color: #333;
    margin: 0 0 8px 0;
    line-height: 1.3;
    height: 40px;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}}

.movie-rating {{
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    color: #666;
}}

.star-rating-carousel {{
    display: flex;
    align-items: center;
    gap: 2px;
    font-size: 14px;
}}

.star-rating-carousel .star {{
    color: #ffc107;
}}

.star-rating-carousel .star.empty {{
    color: #ddd;
}}

.star-rating-carousel .rating-text {{
    margin-left: 5px;
    font-size: 11px;
    color: #666;
}}

.carousel-nav {{
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    background: rgba(255, 255, 255, 0.9);
    border: none;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 20px;
    font-weight: bold;
    color: #333;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
    z-index: 10;
}}

.carousel-nav:hover {{
    background: rgba(255, 255, 255, 1);
    transform: translateY(-50%) scale(1.1);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}}

.carousel-nav:active {{
    transform: translateY(-50%) scale(0.95);
}}

.carousel-nav.prev {{
    left: 10px;
}}

.carousel-nav.next {{
    right: 10px;
}}

.carousel-dots {{
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 20px;
}}

.dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.5);
    cursor: pointer;
    transition: background 0.3s ease;
}}

.dot.active {{
    background: #667eea;
}}

@media (max-width: 768px) {{
    .movie-card {{
        flex: 0 0 19.2%;
    }}
    
    .movie-poster {{
        height: 240px;
    }}
    
    .movie-title {{
        font-size: 12px;
        height: 32px;
    }}
    
    .movie-info {{
        padding: 10px;
    }}
    
    .carousel-nav {{
        width: 40px;
        height: 40px;
        font-size: 16px;
    }}
}}
</style>

<div class="carousel-container">
    <div class="carousel-wrapper">
        <button class="carousel-nav prev" onclick="previousSlide{carousel_id}()">‹</button>
        <div class="carousel-track" id="carouselTrack{carousel_id}">
"""

    # Add movie cards to HTML with anchor tags and proper star ratings
    for movie in movies:
        title = str(movie.get('title', 'Unknown Title'))[:50]
        rating = float(movie.get('vote_average', 0))
        poster_url = movie['poster_url']
        movie_id = movie.get('id', '')
        source = movie.get('source', 'dataset')
        
        # Generate star rating based on source
        if source == 'movielens':
            # MovieLens rating (1-5 scale)
            star_display = generate_star_display_html(rating, max_rating=5)
            rating_text = f"{rating:.1f}/5"
        else:
            # TMDB rating (1-10 scale, convert to 5-star display)
            converted_rating = rating / 2  # Convert 10-point to 5-point scale
            star_display = generate_star_display_html(converted_rating, max_rating=5)
            rating_text = f"{rating:.1f}/10"
        
        # Create the URL with proper parameters
        # For MovieLens-sourced items, pass MovieLens movieId; for API items, pass TMDB id
        if source == 'movielens' and movie.get('movieId') is not None:
            movie_id_for_url = movie.get('movieId')
            detail_source = 'movielens'
        else:
            movie_id_for_url = movie_id
            detail_source = 'api'

        detail_url = f"/Movie_Details?movie_id={movie_id_for_url}&source={detail_source}"
        
        poster_html += f"""
            <a href="{detail_url}" class="movie-card" target="_blank">
                <img src="{poster_url}" alt="{title}" class="movie-poster" onerror="this.src='https://via.placeholder.com/200x280?text=No+Image'">
                <div class="movie-info">
                    <h3 class="movie-title">{title}</h3>
                    <div class="movie-rating">
                        <div class="star-rating-carousel">
                            {star_display}
                            <span class="rating-text">{rating_text}</span>
                        </div>
                    </div>
                </div>
            </a>
        """

    poster_html += f"""
        </div>
        <button class="carousel-nav next" onclick="nextSlide{carousel_id}()">›</button>
    </div>
    <div class="carousel-dots" id="carouselDots{carousel_id}"></div>
</div>

<script>
let currentSlide{carousel_id} = 0;
const track{carousel_id} = document.getElementById('carouselTrack{carousel_id}');
const cards{carousel_id} = document.querySelectorAll('#carouselTrack{carousel_id} .movie-card');
const totalCards{carousel_id} = cards{carousel_id}.length;
const cardsPerView{carousel_id} = 5;
const totalSlides{carousel_id} = Math.ceil(totalCards{carousel_id} / cardsPerView{carousel_id});

function createDots{carousel_id}() {{
    const dotsContainer = document.getElementById('carouselDots{carousel_id}');
    dotsContainer.innerHTML = '';
    
    for (let i = 0; i < totalSlides{carousel_id}; i++) {{
        const dot = document.createElement('div');
        dot.className = 'dot';
        if (i === 0) dot.classList.add('active');
        dot.onclick = () => goToSlide{carousel_id}(i);
        dotsContainer.appendChild(dot);
    }}
}}

function updateDots{carousel_id}() {{
    const dots = document.querySelectorAll('#carouselDots{carousel_id} .dot');
    dots.forEach((dot, index) => {{
        dot.classList.toggle('active', index === currentSlide{carousel_id});
    }});
}}

function goToSlide{carousel_id}(slideIndex) {{
    currentSlide{carousel_id} = slideIndex;
    const translateX = -slideIndex * 100;
    track{carousel_id}.style.transform = `translateX(${{translateX}}%)`;
    updateDots{carousel_id}();
}}

function previousSlide{carousel_id}() {{
    if (currentSlide{carousel_id} > 0) {{
        currentSlide{carousel_id}--;
        goToSlide{carousel_id}(currentSlide{carousel_id});
    }} else {{
        currentSlide{carousel_id} = totalSlides{carousel_id} - 1;
        goToSlide{carousel_id}(currentSlide{carousel_id});
    }}
}}

function nextSlide{carousel_id}() {{
    if (currentSlide{carousel_id} < totalSlides{carousel_id} - 1) {{
        currentSlide{carousel_id}++;
        goToSlide{carousel_id}(currentSlide{carousel_id});
    }} else {{
        currentSlide{carousel_id} = 0;
        goToSlide{carousel_id}(currentSlide{carousel_id});
    }}
}}

createDots{carousel_id}();
goToSlide{carousel_id}(0);

let autoPlayInterval{carousel_id};

function startAutoPlay{carousel_id}() {{
    autoPlayInterval{carousel_id} = setInterval(() => {{
        nextSlide{carousel_id}();
    }}, 4000);
}}

function stopAutoPlay{carousel_id}() {{
    clearInterval(autoPlayInterval{carousel_id});
}}

const carousel{carousel_id} = document.querySelector('#carouselTrack{carousel_id}').closest('.carousel-container');
carousel{carousel_id}.addEventListener('mouseenter', stopAutoPlay{carousel_id});
carousel{carousel_id}.addEventListener('mouseleave', startAutoPlay{carousel_id});

let startX{carousel_id} = 0;
let currentX{carousel_id} = 0;
let isDragging{carousel_id} = false;

track{carousel_id}.addEventListener('touchstart', (e) => {{
    startX{carousel_id} = e.touches[0].clientX;
    isDragging{carousel_id} = true;
    stopAutoPlay{carousel_id}();
}});

track{carousel_id}.addEventListener('touchmove', (e) => {{
    if (!isDragging{carousel_id}) return;
    currentX{carousel_id} = e.touches[0].clientX;
    const deltaX = currentX{carousel_id} - startX{carousel_id};
    
    if (Math.abs(deltaX) > 50) {{
        if (deltaX > 0) {{
            previousSlide{carousel_id}();
        }} else {{
            nextSlide{carousel_id}();
        }}
        isDragging{carousel_id} = false;
    }}
}});

track{carousel_id}.addEventListener('touchend', () => {{
    isDragging{carousel_id} = false;
}});

// Prevent carousel navigation when clicking on movie cards
document.querySelectorAll('#carouselTrack{carousel_id} .movie-card').forEach(card => {{
    card.addEventListener('click', (e) => {{
        e.stopPropagation();
    }});
}});
</script>
"""
    
    return poster_html

def generate_star_display_html(rating, max_rating=5):
    """Generate HTML for star display in carousel"""
    if pd.isna(rating) or rating == 0:
        return ''.join(['<span class="star empty">☆</span>' for _ in range(max_rating)])
    
    rating = max(0, min(rating, max_rating))
    full_stars = int(rating)
    has_half = (rating - full_stars) >= 0.5
    
    html = ''
    
    # Full stars
    for i in range(full_stars):
        html += '<span class="star">⭐</span>'
    
    # Half star
    if has_half:
        html += '<span class="star">★</span>'
        empty_count = max_rating - full_stars - 1
    else:
        empty_count = max_rating - full_stars
    
    # Empty stars
    for i in range(empty_count):
        html += '<span class="star empty">☆</span>'
    
    return html

def display_movie_filters():
    st.subheader("🎯 Filter Movies")
    
    # Initialize session state for filters
    if 'filtered_movies' not in st.session_state:
        st.session_state.filtered_movies = []
    if 'filter_applied' not in st.session_state:
        st.session_state.filter_applied = False
    if 'filter_expanded' not in st.session_state:
        st.session_state.filter_expanded = False
    
    # Fetch data for dropdowns
    genres = fetch_genres()
    languages = fetch_languages()
    
    # Create collapsible filter section
    with st.expander("Show / Hide", expanded=st.session_state.filter_expanded):
        # Create filter form
        with st.form(key='movie_filters'):

            # Genre filter
            genre_options = [{'label': 'All Genres', 'value': None}] + [{'label': genre['name'], 'value': genre['id']} for genre in genres]
            selected_genres = st.multiselect(
                "🎭 Select Genres",
                options=[g['value'] for g in genre_options if g['value'] is not None],
                format_func=lambda x: next(g['label'] for g in genre_options if g['value'] == x),
                help="Choose one or more genres"
            )
            st.write("")

            # Language filter
            language_options = [{'label': 'All Languages', 'value': 'all'}] + [{'label': lang['name'], 'value': lang['iso_639_1']} for lang in languages[:100]]
            selected_language = st.selectbox(
                "🌐 Original Language",
                options=[l['value'] for l in language_options],
                format_func=lambda x: next(l['label'] for l in language_options if l['value'] == x),
                help="Choose original language"
            )
            st.write("")

            # Release date filter
            st.write("📅 Release Date Range:")
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input(
                    "From",
                    value=None,
                    min_value=datetime(1900, 1, 1).date(),
                    max_value=datetime.now().date(),
                    help="Select start date"
                )
            with date_col2:
                end_date = st.date_input(
                    "To",
                    value=None,
                    min_value=datetime(1900, 1, 1).date(),
                    max_value=datetime.now().date(),
                    help="Select end date"
                )
            
            st.write("")
            st.write("")
            
            # Filter and Clear buttons
            button_col1, button_col2 = st.columns([1, 1])
            with button_col1:
                filter_submitted = st.form_submit_button("🔍 Apply Filters", use_container_width=True)
            with button_col2:
                clear_submitted = st.form_submit_button("🗑️ Clear Filters", use_container_width=True)
        
        # Handle filter submission
        if filter_submitted:
            # Prepare filter parameters
            genre_ids = selected_genres if selected_genres else None
            release_date_gte = start_date.strftime('%Y-%m-%d') if start_date else None
            release_date_lte = end_date.strftime('%Y-%m-%d') if end_date else None
            language = selected_language if selected_language != 'all' else None
            
            # Validate date range
            if start_date and end_date and start_date > end_date:
                st.error("Start date must be before end date!")
                return
            
            with st.spinner("🔍 Searching for movies with AI recommendations..."):
                # Fetch filtered movies with AI enhancement
                filtered_movies = fetch_filtered_movies_with_ai(
                    genre_ids=genre_ids,
                    release_date_gte=release_date_gte,
                    release_date_lte=release_date_lte,
                    language=language
                )
                
                st.session_state.filtered_movies = filtered_movies
                st.session_state.filter_applied = True
        
        # Handle clear filters
        if clear_submitted:
            st.session_state.filtered_movies = []
            st.session_state.filter_applied = False
            st.rerun()
    
    
    # Display filtered results
    if st.session_state.filter_applied:
        st.markdown('<div class="filtered-results">', unsafe_allow_html=True)
        
        if st.session_state.filtered_movies:
            # Results header
            st.markdown('<div class="results-header">', unsafe_allow_html=True)
            st.subheader("🎬 Filtered Results")
            st.markdown(f'<span class="results-count">Found {len(st.session_state.filtered_movies)} movies</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display filtered movies in carousel
            components.html(
                generate_carousel_html(st.session_state.filtered_movies, "Filtered", "Filtered Movies"), 
                height=480
            )
        else:
            st.warning("No movies found matching your criteria. Try adjusting your filters.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Main app
def main():
    load_navbar()
    load_custom_css()
        
    trending_day = 3000

    with st.spinner("🎬 Loading MovieLens data..."):
        ratings_df, movies_df, links_df, merged_df, tags_df, movies_year = load_movielens_data()
        
        if ratings_df is None:
            st.error("Failed to load MovieLens data. Please check if the dataset files are available.")
            return

        # Get all recommendations using the loaded data
        trending_movies = get_trending_movies_movielens(ratings_df, merged_df, days=trending_day)
        popular_movies = get_popular_movies_movielens(ratings_df, merged_df)
        top_rated_movies = get_top_rated_movies_movielens(ratings_df, merged_df, min_ratings=50)
        
        # Assuming you've already fetched your movie lists (e.g., trending_movies, popular_movies, top_rated_movies)
        trending_movies = enhance_movies_concurrently(trending_movies)
        popular_movies = enhance_movies_concurrently(popular_movies)
        top_rated_movies = enhance_movies_concurrently(top_rated_movies)

    
    # Display sections
    st.title("🎬 Movie Discovery")
    
    # Add the filtering UI
    display_movie_filters()
    
        # Only show the original carousels if no filter is applied
    if not st.session_state.get('filter_applied', False):
        # Trending Movies (based on recent rating activity)
        st.header("🔥 Trending Movies")
        st.caption(f"Based on recent rating activity in the last {trending_day} days")
        if trending_movies:
            components.html(generate_carousel_html(trending_movies, "Trending", "Trending Movies"), height=480)
        else:
            st.info("No trending movies data available. This may be due to limited recent rating activity.")

        # Top Rated Movies (based on MovieLens ratings)
        st.header("🏆 Top Rated Movies")
        st.caption("Based on average ratings from MovieLens users (minimum 50 ratings)")
        if top_rated_movies:
            components.html(generate_carousel_html(top_rated_movies, "TopRated", "Top Rated Movies"), height=480)
        else:
            st.info("No top rated movies data available.")
        
        # Popular Movies (based on total rating count)
        st.header("🌟 Popular Movies")
        st.caption("Based on total number of ratings from MovieLens users")
        if popular_movies:
            components.html(generate_carousel_html(popular_movies, "Popular", "Popular Movies"), height=480)
        else:
            st.info("No popular movies data available.")
                
if __name__ == "__main__":
    main()