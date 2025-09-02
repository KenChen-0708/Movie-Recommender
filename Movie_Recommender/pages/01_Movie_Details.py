# pages/01_Movie_Details.py
import streamlit as st
from datetime import datetime
import pandas as pd
import requests
# Content Based Recommendations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TMDB API Configuration
API_KEY = "6272a18b79461f7987d2f1047b3d1198"
API_BASE_URL = "https://api.themoviedb.org/3"

st.set_page_config(page_title="Movie Details", page_icon="🎬", layout="wide")

# ----- TMDB API Functions -----
@st.cache_data
def fetch_movie_details_by_tmdb_id(tmdb_id: int | str):
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
    except Exception:
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
        return f"https://image.tmdb.org/t/p/w200{tmdb_result['poster_path']}"
    return None

# ----- Content-Based Recommendation Data Loading Functions -----
@st.cache_data
def load_movie_data():
    """Load MovieLens movies dataset"""
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

@st.cache_data
def load_ratings_with_stats():
    """Load ratings data and calculate statistics"""
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

# ----- Content-Based Recommendation Engine -----
@st.cache_resource
def get_recommendation_vectorizer_and_matrix(dataframe):
    """Create TF-IDF vectorizer and matrix for content-based recommendations"""
    # Create enhanced recommendation blob with title, genres, AND tags
    dataframe['title'] = dataframe['title'].fillna('')
    dataframe['genres'] = dataframe['genres'].fillna('')
    dataframe['clean_title'] = dataframe['clean_title'].fillna('')
    dataframe['tag'] = dataframe['tag'].fillna('')
    
    # Create comprehensive content features for recommendations
    dataframe['recommendation_blob'] = (
        dataframe['title'] + ' ' + 
        dataframe['clean_title'] + ' ' + 
        dataframe['genres'].str.replace('|', ' ') + ' ' +
        dataframe['tag']
    ).str.lower()
    
    # Clean the recommendation blob
    dataframe['recommendation_blob'] = dataframe['recommendation_blob'].str.replace(r'[^\w\s]', ' ', regex=True)
    dataframe['recommendation_blob'] = dataframe['recommendation_blob'].str.replace(r'\s+', ' ', regex=True)
    
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=10000,  # More features for better tag representation
        min_df=1,  # Include even rare words (good for specific tags)
        ngram_range=(1, 2)  # Include both single words and bigrams
    )
    tfidf_matrix = vectorizer.fit_transform(dataframe['recommendation_blob'])
    return vectorizer, tfidf_matrix

# ----- CSS Styling -----
def load_custom_css():
    """Load custom CSS for movie details page"""
    st.markdown(
        """
    <style>
    
    :root{
        --primary:#00d4ff;
        --card-bg:#1a1a25;
        --text:#f0f0f0;
        --text-secondary:#b0b0b0
        }
        
    .detail-container{
        background:var(--card-bg);
        padding:30px;
        border-radius:12px;
        margin-bottom:20px;
        border:1px solid #2a2a35
        }
        
    .tagline{
        font-style:italic;
        color:var(--primary);
        margin-bottom:10px
        }
        
    .genre-badge{
        display:inline-block;
        background:rgba(0,212,255,.15);
        color:var(--primary);
        padding:4px 12px;
        border-radius:50px;
        margin:0 8px 8px 0;
        font-size:.8rem
        }
        
    .rating-display{
        display:inline-flex;
        align-items:center;
        gap:4px;
        background:rgba(255,193,7,.15);
        color:#ffc107;
        padding:8px 16px;
        border-radius:50px;
        font-weight:600
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
        
    </style>
    """,
        unsafe_allow_html=True,
    )

def load_content_based_css():
    """Load CSS specifically for content-based recommendations"""
    st.markdown("""
    <style>
    /* Recommendation styles */
    .rec-container {
        display: flex;
        overflow-x: auto;
        overflow-y: hidden;
        gap: 20px;
        padding: 15px 5px;
        margin: 20px 0;
        scrollbar-width: thin;
        scrollbar-color: rgba(179, 27, 27, 0.5) rgba(0, 0, 0, 0.1);
        flex-wrap: nowrap;
        white-space: nowrap;
    }
    
    .rec-container::-webkit-scrollbar {
        height: 8px;
    }
    
    .rec-container::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
        border-radius: 4px;
    }
    
    .rec-container::-webkit-scrollbar-thumb {
        background: rgba(179, 27, 27, 0.5);
        border-radius: 4px;
    }
    
    .rec-container::-webkit-scrollbar-thumb:hover {
        background: rgba(179, 27, 27, 0.7);
    }
    
    .rec-card {
        flex: 0 0 160px;
        min-width: 160px;
        max-width: 160px;
        border-radius: 12px;
        background: rgba(30, 30, 30, 0.8);
        padding: 10px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
        white-space: normal;
    }
    .rec-card:hover {
        transform: translateY(-6px) scale(1.03);
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
        border-color: #b31b1b;
    }
    
    .rec-poster {
        border-radius: 8px;
        width: 100%;
        aspect-ratio: 2/3;
        object-fit: cover;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    .rec-placeholder {
        background: linear-gradient(135deg, #333, #555);
        border-radius: 8px;
        width: 100%;
        aspect-ratio: 2/3;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #999;
        font-size: 2rem;
    }
    
    .rec-title {
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 8px;
        color: white;
        line-height: 1.2;
        height: 2.4rem;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .rec-year {
        font-size: 0.8rem;
        color: #aaa;
        margin-top: 4px;
    }
    
    @media (max-width: 768px) {
        .rec-container {
            gap: 15px;
        }
        
        .rec-card {
            min-width: 140px;
            max-width: 140px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ----- Content-Based Recommendation Rendering -----
def render_content_based_recommendations(detail_id_int, movies_df, vectorizer, tfidf_matrix):
    """Render content-based recommendations for a given movie"""
    # Load the CSS for recommendations
    load_content_based_css()
    
    st.markdown("---")
    st.subheader("🍿 Because you liked this movie...")

    try:
        # Find index of the current movie
        current_index = movies_df[movies_df['movieId'] == detail_id_int].index[0]

        # Compute similarity scores for this movie
        similarity_scores = list(enumerate(cosine_similarity(tfidf_matrix[current_index], tfidf_matrix).flatten()))

        # Sort by similarity
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Top N recommendations (skip first one, because it's the same movie)
        top_indices = [i for i, score in similarity_scores[1:11]]  # Top 10
        recommended_movies = movies_df.iloc[top_indices]

        # Build the entire horizontal container using local data + TMDB posters
        rec_cards_html = '<div class="rec-container">'
        
        for _, rec in recommended_movies.iterrows():
            rec_id = rec.get('movieId')
            rec_title = rec.get('clean_title', rec.get('title', ''))
            rec_year = rec.get('year', '')
            rec_genres = rec.get('genre_list', [])
            
            # Limit title length for better display and escape HTML characters
            display_title = rec_title[:25] + "..." if len(rec_title) > 25 else rec_title
            display_title = display_title.replace('"', '&quot;').replace("'", "&#x27;")
            
            # Get poster URL from TMDB
            poster_url = get_poster_for_movie(rec_title, rec_year if pd.notna(rec_year) else None)
            
            # Use actual poster or placeholder
            if poster_url:
                poster_html = f'<img src="{poster_url}" class="rec-poster" alt="{rec_title} Poster">'
            else:
                poster_html = '<div class="rec-placeholder">🎬</div>'
            
            # Create genre display
            genre_display = ''
            if rec_genres and isinstance(rec_genres, list):
                valid_genres = [g for g in rec_genres[:2] if g != "(no genres listed)"]
                if valid_genres:
                    genre_display = f'<div style="font-size: 0.7rem; color: #888; margin-top: 2px;">{" • ".join(valid_genres)}</div>'

            rec_cards_html += f'<a href="?movie_id={rec_id}" style="text-decoration:none; color:inherit;" class="rec-link"><div class="rec-card">{poster_html}<div class="rec-title">{display_title}</div><div class="rec-year">{int(rec_year) if pd.notna(rec_year) else ""}</div>{genre_display}</div></a>'
        
        rec_cards_html += '</div>'
        
        # Render the complete horizontal container in one go
        st.markdown(rec_cards_html, unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Could not load recommendations: {e}")

# ----- Data Initialization -----
# Load MovieLens data for content-based recommendations
movies_df = load_movie_data()
tags_df, movie_tags_df = load_tags_data()
ratings_df, rating_stats_df = load_ratings_with_stats()

# Merge data for enhanced recommendations
if not movies_df.empty and not rating_stats_df.empty:
    movies_df = movies_df.merge(rating_stats_df, on='movieId', how='left')
    movies_df['avg_rating'] = movies_df['avg_rating'].fillna(0)
    movies_df['rating_count'] = movies_df['rating_count'].fillna(0)

if not movies_df.empty and not movie_tags_df.empty:
    movies_df = movies_df.merge(movie_tags_df, on='movieId', how='left')
    movies_df['tag'] = movies_df['tag'].fillna('')
else:
    movies_df['tag'] = ''

# Initialize recommendation system
if not movies_df.empty:
    rec_vectorizer, rec_tfidf_matrix = get_recommendation_vectorizer_and_matrix(movies_df)

# ----- Main Application Logic -----
load_custom_css()

# Read query parameters
qp = st.query_params
movie_id = qp.get("movie_id")

# If no movie_id in query params, try session state
if not movie_id and "movie_id" in st.session_state:
    movie_id = st.session_state.movie_id

source = qp.get("source", "movielens")
if isinstance(movie_id, list):
    movie_id = movie_id[0]
if isinstance(source, list):
    source = source[0]

if not movie_id:
    st.error("No movie_id provided.")
    st.stop()

# Handle different data sources
try:
    detail_id_int = int(str(movie_id).strip())
    
    # Look up in MovieLens data first
    movie_detail = movies_df[movies_df['movieId'] == detail_id_int] if not movies_df.empty else pd.DataFrame()
    
    if movie_detail.empty:
        st.error("Movie details not found.")
        st.stop()
        
    movie = movie_detail.iloc[0]
except ValueError:
    st.error("Invalid movie ID")
    st.stop()

# Extract MovieLens movie info
movie_title = movie.get("clean_title", movie.get("title", ""))
movie_year = movie.get("year")
avg_rating = movie.get("avg_rating", 0)
rating_count = movie.get("rating_count", 0)

# Try to get enhanced movie details from TMDB
tmdb_data = None
if movie_title:
    tmdb_data = search_tmdb_by_title(movie_title, movie_year)
    if tmdb_data:
        # Get full details from TMDB
        tmdb_id = tmdb_data.get('id')
        if tmdb_id:
            tmdb_data = fetch_movie_details_by_tmdb_id(tmdb_id)

# ----- Movie Details Display -----
col1, col2 = st.columns([1, 2])

with col1:
    # Display poster
    if tmdb_data and tmdb_data.get("poster_path"):
        poster_url = f"https://image.tmdb.org/t/p/w400{tmdb_data['poster_path']}"
        st.image(poster_url, use_container_width=True)
    else:
        # Fallback placeholder
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #333, #555);
                border-radius: 12px;
                width: 100%;
                aspect-ratio: 2/3;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #999;
                font-size: 4rem;
                margin-bottom: 1rem;
            ">
                🎬
            </div>
        """, unsafe_allow_html=True)

with col2:
    # Title
    title_display = movie_title or "No Title"
    st.title(title_display)
    
    # Tagline (if available from TMDB)
    if tmdb_data:
        tagline = (tmdb_data.get("tagline") or "").strip()
        if tagline:
            st.markdown(f'<div class="tagline">"{tagline}"</div>', unsafe_allow_html=True)

    # Release date and runtime
    year = movie.get("year", "")
    if pd.notna(year):
        try:
            if tmdb_data and tmdb_data.get("release_date"):
                release_fmt = datetime.strptime(tmdb_data["release_date"], "%Y-%m-%d").strftime("%B %d, %Y")
            else:
                release_fmt = f"{int(year)}"
            st.write(f"**📅 Release Date:** {release_fmt}")
        except:
            st.write(f"**📅 Release Date:** {int(year)}")

    # Runtime (from TMDB if available)
    if tmdb_data and tmdb_data.get("runtime"):
        runtime = tmdb_data.get("runtime", 0)
        if runtime:
            st.write(f"**⏱️ Runtime:** {runtime} min")
        else:
            st.write("**⏱️ Runtime:** N/A")

    # Rating - prioritize TMDB rating if available
    if tmdb_data and tmdb_data.get("vote_average"):
        tmdb_rating = tmdb_data.get("vote_average")
        st.markdown(f'<div class="rating-display">⭐ {tmdb_rating}/10</div>', unsafe_allow_html=True)
    elif avg_rating > 0:
        st.markdown(f'<div class="rating-display">⭐ {avg_rating:.1f}/5.0</div>', unsafe_allow_html=True)

    # Genres - prioritize TMDB if available
    if tmdb_data and tmdb_data.get("genres"):
        st.markdown("**🎞️ Genres:**")
        for genre in tmdb_data["genres"]:
            st.markdown(f'<span class="genre-badge">{genre["name"]}</span>', unsafe_allow_html=True)
    else:
        # Fallback to MovieLens genres
        genres = movie.get("genre_list", [])
        if genres and isinstance(genres, list):
            st.markdown("**🎞️ Genres:**")
            for g in genres:
                if g != "(no genres listed)":
                    st.markdown(f'<span class="genre-badge">{g}</span>', unsafe_allow_html=True)

    # User Tags from MovieLens
    movie_tags = movie.get("tag", "").strip()
    if movie_tags:
        st.markdown("**🏷️ User Tags:**")
        
        # Split tags by pipe separator and display them as badges
        if '|' in movie_tags:
            tags_list = [tag.strip() for tag in movie_tags.split('|') if len(tag.strip()) > 1]
        else:
            # Fallback to space splitting if no pipes found
            tags_list = [tag.strip() for tag in movie_tags.split() if len(tag.strip()) > 1]
        
        unique_tags = list(dict.fromkeys(tags_list))[:15]  # Remove duplicates, limit to 15
        
        if unique_tags:
            tags_html = ''.join([f'<span class="tag-item">{tag}</span>' for tag in unique_tags])
            st.markdown(f'<div>{tags_html}</div>', unsafe_allow_html=True)
        else:
            st.write("_No user tags available for this movie._")

    # Overview/Description
    if tmdb_data and tmdb_data.get("overview"):
        st.subheader("📝 Overview")
        st.write(tmdb_data["overview"])

    # Cast information (from TMDB)
    if tmdb_data and tmdb_data.get("credits", {}).get("cast"):
        cast = tmdb_data["credits"]["cast"][:8]  # Top 8 cast members
        if cast:
            st.subheader("👤 Top Cast")
            cast_names = [c.get("name", "") for c in cast if c.get("name")]
            st.write(", ".join(cast_names))

    # Additional Information
    st.subheader("📌 Additional Information")
    
    if tmdb_data:
        # Budget and Revenue
        budget = tmdb_data.get("budget", 0)
        revenue = tmdb_data.get("revenue", 0)
        if budget > 0:
            st.write(f"**Budget:** ${int(budget):,}")
        if revenue > 0:
            st.write(f"**Revenue:** ${int(revenue):,}")
        
        # Original Language
        lang = (tmdb_data.get("original_language") or "").upper()
        if lang:
            st.write(f"**Original Language:** {lang}")
        
        # Status
        status = tmdb_data.get("status") or ""
        if status:
            st.write(f"**Status:** {status}")
        
        # Homepage
        hp = (tmdb_data.get("homepage") or "").strip()
        if hp:
            st.markdown(f"🔗 [Official Website]({hp})")

# ----- Content-Based Recommendations Section -----
if not movies_df.empty and 'rec_vectorizer' in globals() and 'rec_tfidf_matrix' in globals():
    render_content_based_recommendations(detail_id_int, movies_df, rec_vectorizer, rec_tfidf_matrix)