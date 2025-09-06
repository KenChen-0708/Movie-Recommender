# pages/01_Movie_Details.py
import streamlit as st
from datetime import datetime
import pandas as pd
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import math
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import scipy.sparse as sp
from Homepage import load_navbar
import time, os
from sklearn.preprocessing import normalize

# TMDB API Configuration
API_KEY = "6272a18b79461f7987d2f1047b3d1198"
API_BASE_URL = "https://api.themoviedb.org/3"

st.set_page_config(page_title="Movie Details", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

# Persist graphs + rendered HTML across reruns
if "figs" not in st.session_state:
    st.session_state["figs"] = {}

# cache for rendered recommendation sections (avoid reloading movies on button clicks)
if "html_cache" not in st.session_state:
    st.session_state["html_cache"] = {
        "content": {},   # key: ml_movie_id -> HTML string
        "cf": {},        # key: tmdb_id -> HTML string
        "hybrid": {}     # key: tmdb_id -> HTML string
    }

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
    df = dataframe.copy()

    df["clean_title"] = df["clean_title"].fillna("").astype(str)
    df["genres"]      = df["genres"].fillna("").astype(str)
    df["tag"]         = df["tag"].fillna("").astype(str)

    title_vec = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2),
        sublinear_tf=True, min_df=2, max_features=8000
    )
    X_title = title_vec.fit_transform(df["clean_title"])

    tags_vec = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2),
        sublinear_tf=True, min_df=2, max_features=15000
    )
    X_tags = tags_vec.fit_transform(df["tag"])

    genre_lists = (
        df["genres"].str.replace("(no genres listed)", "", regex=False)
                     .str.split("|")
                     .apply(lambda lst: [g for g in lst if g])
    )
    mlb = MultiLabelBinarizer()
    X_genre = sp.csr_matrix(mlb.fit_transform(genre_lists))

    # weighted stack
    X = sp.hstack([3.0 * X_title, 2.0 * X_genre, 1.5 * X_tags], format="csr")
    # global L2 normalize so cosine is truly cosine
    X = normalize(X, norm="l2", axis=1, copy=False)

    model_dict = {"title_vec": title_vec, "tags_vec": tags_vec, "genre_mlb": mlb}
    return model_dict, X

def bayesian_average(avg, cnt, global_mean, m=50):
    # m = strength of prior (tune 25–200). 50 works well on ml-latest-small.
    return (cnt * avg + m * global_mean) / (cnt + m) if cnt and cnt > 0 else global_mean


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

    html, body, .stApp {
        background: linear-gradient(135deg, #1e1e2f, #2e2e4e);
        color: white;
        font-family: 'Segoe UI', sans-serif;
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
    
    .rating-text{
        font-size:.8rem;
        font-weight:600;
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

def load_eval_css():
    st.markdown("""
    <style>
      .metric-row{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0 2px;}
      .metric-card{
        flex:1 1 180px;background:#1f2230;border:1px solid rgba(255,255,255,.08);
        border-radius:12px;padding:12px 14px;
      }
      .metric-title{font-size:.85rem;color:#9aa4b2;margin-bottom:6px}
      .metric-value{font-size:1.35rem;font-weight:700;color:#fff}
      .kpi-pill{
        display:inline-flex;align-items:center;gap:6px;background:rgba(99,102,241,.15);
        border:1px solid rgba(99,102,241,.35);color:#A5B4FC;padding:6px 10px;border-radius:999px;
        font-size:.8rem;margin-right:8px
      }
      .bar{
        width:100%;height:8px;border-radius:8px;background:rgba(255,255,255,.06);overflow:hidden;margin-top:6px
      }
      .bar>span{display:block;height:100%;background:linear-gradient(90deg,#4ECDC4,#45B7D1)}
      .subtle{color:#9aa4b2;font-size:.85rem}
      .chip{display:inline-block;padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.07);
            border:1px solid rgba(255,255,255,.08);margin:4px 6px 0 0;font-size:.85rem}
      .help{font-size:.8rem;color:#9aa4b2}
      .ok{color:#4ECDC4}.meh{color:#FFC107}.bad{color:#FF6B6B}
    </style>
    """, unsafe_allow_html=True)

def kpi_card(title: str, value: float):
    pct = max(0.0, min(1.0, float(value)))
    html = f"""
<div class="metric-card">
  <div class="metric-title">{title}</div>
  <div class="metric-value">{value:.3f}</div>
  <div class="bar"><span style="width:{pct*100:.1f}%"></span></div>
</div>
""".strip()
    return html



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

# Content-Based Filtering Functions
def content_based_recommendations(target_movie_id, movies_df, tfidf_matrix, top_n=12):
    try:
        idx_list = movies_df.index[movies_df['movieId'] == target_movie_id].tolist()
        if not idx_list:
            return pd.DataFrame()
        target_idx = idx_list[0]

        sims = cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix).ravel()
        sims[target_idx] = -1  # exclude itself

        # Pull basic fields
        df = movies_df[["movieId","clean_title","year","genres","avg_rating","rating_count"]].copy()
        df["similarity"] = sims

        # Bayesian popularity smoothing
        global_mean = float(movies_df["avg_rating"].replace(0, np.nan).mean())
        df["bayes"] = [
            bayesian_average(r, c, global_mean, m=50)
            for r, c in zip(df["avg_rating"].fillna(0), df["rating_count"].fillna(0))
        ]

        # Final content score: tune alpha to taste (0.8 favors pure similarity)
        alpha = 0.8
        # rank-normalize pieces to [0,1] to be safe
        df["sim_norm"]  = (df["similarity"] - df["similarity"].min()) / (df["similarity"].max() - df["similarity"].min() + 1e-12)
        df["bayes_norm"]= (df["bayes"] - df["bayes"].min()) / (df["bayes"].max() - df["bayes"].min() + 1e-12)
        df["content_score"] = alpha * df["sim_norm"] + (1 - alpha) * df["bayes_norm"]

        # compute genre_list for rendering
        def _gl(genres):
            if isinstance(genres, str):
                return [g for g in genres.split("|") if g and g != "(no genres listed)"]
            return []
        df["genre_list"] = df["genres"].apply(_gl)

        out = (df.sort_values("content_score", ascending=False)
                 .head(top_n)
                 .rename(columns={"clean_title":"title"})
                 [["movieId","title","year","genres","genre_list","avg_rating","rating_count","similarity","content_score"]])
        return out
    except Exception as e:
        print(f"Error in content_based_recommendations: {e}")
        return pd.DataFrame()


# ----- Content-Based Recommendation Rendering -----
def render_content_based_recommendations(detail_id_int, movies_df, vectorizer, tfidf_matrix):
    """Render content-based recommendations for a given movie; reuse cached HTML on reruns."""
    load_content_based_css()
    st.markdown("---")
    st.subheader("🍿 More like this movie...")
    st.caption("Content-based filtering using genre similarities.")

    # Serve from cache if we already built it
    cached = st.session_state["html_cache"]["content"].get(detail_id_int)
    if cached:
        st.markdown(cached, unsafe_allow_html=True)
        return

    with st.spinner("Loading similar movies..."):
        recommendations = content_based_recommendations(detail_id_int, movies_df, tfidf_matrix, top_n=12)
        if recommendations.empty:
            st.info("No content-based recommendations available for this movie.")
            return

        rec_cards_html = ['<div class="rec-container">']
        for _, rec in recommendations.iterrows():
            rec_id = rec.get('movieId')
            rec_title = rec.get('title', '')
            rec_year = rec.get('year', '')
            rec_genres = rec.get('genre_list', [])
            rec_avg_rating = rec.get('avg_rating', 'N/A')
            similarity_score = rec.get('similarity', 0)

            display_title = rec_title[:25] + "..." if len(rec_title) > 25 else rec_title
            display_title = display_title.replace('"', '&quot;').replace("'", "&#x27;")

            poster_url = get_poster_for_movie(rec_title, rec_year if pd.notna(rec_year) else None)
            poster_html = (
                f'<img src="{poster_url}" class="rec-poster" alt="{rec_title} Poster">'
                if poster_url else '<div class="rec-placeholder">🎬</div>'
            )

            genre_display = ''
            if rec_genres and isinstance(rec_genres, list):
                valid_genres = [g for g in rec_genres[:2] if g != "(no genres listed)"]
                if valid_genres:
                    genre_display = f'<div style="font-size: 0.7rem; color: #888; margin-top: 2px;">{" • ".join(valid_genres)}</div>'

            similarity_html = f'<div style="font-size: 0.75rem; color: #aaa; margin-top: 5px;">📈Similarity: {similarity_score:.2f}</div>'
            rating_html = f'<div style="font-size: 0.75rem; color: #aaa; margin-top: 5px;">⭐Rating: {rec_avg_rating}</div>' if rec_avg_rating != 'N/A' else ''

            rec_cards_html.append(
                f'<a href="/Movie_Details?movie_id={rec_id}&source=movielens" '
                f'style="text-decoration:none; color:inherit;" class="rec-link">'
                f'<div class="rec-card">{poster_html}'
                f'<div class="rec-title">{display_title}</div>'
                f'<div class="rec-year">{int(rec_year) if pd.notna(rec_year) else ""}</div>'
                f'{genre_display}{similarity_html}{rating_html}</div></a>'
            )
        rec_cards_html.append('</div>')
        html = ''.join(rec_cards_html)

    # store + render
    st.session_state["html_cache"]["content"][detail_id_int] = html
    st.markdown(html, unsafe_allow_html=True)



# ---------- Item–Item CF (memory-based) ----------
@st.cache_resource
def build_user_item_matrix(ratings: pd.DataFrame):
    """Pivot to user–item matrix (rows=userId, cols=movieId, values=rating)."""
    mat = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    return mat
    
@st.cache_resource
def build_cached_correlation_matrix(ratings: pd.DataFrame, min_overlap: int = 15):
    user_item = build_user_item_matrix(ratings)

    rating_counts = user_item.notna().sum()
    valid_movies = rating_counts[rating_counts >= min_overlap].index

    if len(valid_movies) == 0:
        # Always return a (corr_mat, valid_movies) tuple
        return pd.DataFrame(), pd.Index([])   # <- FIX

    user_item_filtered = user_item[valid_movies]
    corr_mat = user_item_filtered.corr(method="pearson", min_periods=min_overlap)
    return corr_mat, valid_movies

   
@st.cache_data
def fetch_tmdb_brief(tmdb_id: int):
    """Poster + vote_average (used for the rec grid)."""
    try:
        url = f"{API_BASE_URL}/movie/{tmdb_id}"
        r = requests.get(url, params={"api_key": API_KEY, "language":"en-US"}, timeout=10)
        r.raise_for_status()
        d = r.json()
        poster = f"https://image.tmdb.org/t/p/w300{d.get('poster_path')}" if d.get("poster_path") else None
        return poster, d.get("vote_average", None), d.get("release_date","")
    except Exception:
        return None, None, ""

@st.cache_data
def item_item_recommendations(
    target_tmdb_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    links: pd.DataFrame,
    min_overlap: int = 15,
    top_n: int = 12,
):
    """
    Optimized item-item collaborative filtering recommendations.
    Uses precomputed correlation matrix for faster results.
    """
    try:
        # Get MovieLens ID from TMDB ID
        target_row = links.loc[links["tmdbId"] == int(target_tmdb_id)]
        if target_row.empty:
            return pd.DataFrame()
        
        target_ml_id = int(target_row.iloc[0]["movieId"])
        
        # Get precomputed correlation matrix
        corr_mat, valid_movies = build_cached_correlation_matrix(ratings, min_overlap)
        
        if corr_mat.empty or target_ml_id not in valid_movies:
            return pd.DataFrame()
        
        # Get similarities for target movie
        if target_ml_id not in corr_mat.columns:
            return pd.DataFrame()
            
        sim = corr_mat[target_ml_id].drop(labels=[target_ml_id], errors="ignore")
        sim = sim.dropna().sort_values(ascending=False).head(top_n * 2)  # Get more candidates
        
        if sim.empty:
            return pd.DataFrame()
        
        # Get user-item matrix for overlap calculation
        user_item = build_user_item_matrix(ratings)
        
        # Calculate overlap counts more efficiently
        mask_target = user_item[target_ml_id].notna()
        overlap_counts = {}
        
        for movie_id in sim.index:
            overlap = (mask_target & user_item[movie_id].notna()).sum()
            overlap_counts[movie_id] = overlap
        
        # Convert to Series
        overlap = pd.Series(overlap_counts)
        
        # Precompute rating stats
        stats = ratings.groupby("movieId").agg(
            avg_rating=("rating", "mean"),
            rating_cnt=("rating", "count")
        )
        
        # Build recommendations
        recs = (
            pd.DataFrame({"similarity": sim})
            .assign(overlap=overlap)
            .join(stats, how="left")
            .sort_values(["similarity", "overlap", "rating_cnt"], ascending=False)
        )
        
        # Filter by minimum overlap
        recs = recs[recs["overlap"] >= min_overlap].head(top_n)
        
        if recs.empty:
            return pd.DataFrame()
        
        # Add titles and TMDB IDs
        recs = (
            recs.reset_index()
            .rename(columns={"index": "movieId"})
            .merge(movies[["movieId", "title"]], on="movieId", how="left")
            .merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
        )
        
        recs = recs[recs["tmdbId"].notna()].copy()
        recs["tmdbId"] = recs["tmdbId"].astype(int)
        
        return recs.head(top_n)
    
    except Exception as e:
        print(f"Item-item recommendation error: {e}")
        return pd.DataFrame()

# Updated render_cf_section to fix the captions
def render_cf_section_horizontal(target_tmdb_id: int):
    """Render CF recommendations using the horizontal 'rec-card' design; reuse cached HTML on reruns."""
    load_content_based_css()
    st.markdown("---")
    st.subheader("🔁 Because You Liked This")
    st.caption("Item–Item Collaborative Filtering (Pearson). Requires at least 15 users who rated both movies.")

    cached = st.session_state["html_cache"]["cf"].get(target_tmdb_id)
    if cached:
        st.markdown(cached, unsafe_allow_html=True)
        return

    with st.spinner("Loading collaborative recommendations..."):
        ratings, movies, links = load_ml_small()
        recs = item_item_recommendations(
            target_tmdb_id=target_tmdb_id,
            ratings=ratings,
            movies=movies,
            links=links,
            min_overlap=15,
            top_n=12
        )

        if recs.empty:
            st.info("Not enough overlapping ratings to compute reliable neighbors for this title.")
            return

        cards_html = ['<div class="rec-container">']
        for _, row in recs.iterrows():
            tmdb_id = int(row["tmdbId"])
            title_full = (row.get("title") or "").strip()
            clean_title = title_full.rsplit(" (", 1)[0] if " (" in title_full else title_full
            display_title = (clean_title[:25] + "...") if len(clean_title) > 25 else clean_title

            poster_url, _, release_date = fetch_tmdb_brief(tmdb_id)
            year_display = ""
            if release_date:
                try:
                    year_display = str(datetime.strptime(release_date, "%Y-%m-%d").year)
                except Exception:
                    year_display = ""

            sim = row.get("similarity", np.nan)
            overlap = int(row.get("overlap", 0)) if pd.notna(row.get("overlap", np.nan)) else 0
            ml_avg = row.get("avg_rating", np.nan)
            ml_cnt = int(row.get("rating_cnt", 0)) if pd.notna(row.get("rating_cnt", np.nan)) else 0
            ml_text = f"{ml_avg:.2f}/5 • {ml_cnt} ratings" if pd.notna(ml_avg) and ml_cnt > 0 else "New"

            poster_html = (
                f'<img src="{poster_url}" class="rec-poster" alt="{display_title} Poster">'
                if poster_url else '<div class="rec-placeholder">🎬</div>'
            )

            metrics_html = (
                f'<div style="font-size:.75rem;color:#aaa;margin-top:4px;">'
                f'📈 Similarity: {sim:.2f} • 👥 Overlap: {overlap}<br>⭐ {ml_text}'
                f'</div>'
            ) if pd.notna(sim) else f'<div style="font-size:.75rem;color:#aaa;margin-top:4px;">⭐ {ml_text}</div>'

            cards_html.append(
                f'<a href="/Movie_Details?movie_id={int(row.get("movieId", 0))}&source=movielens" '
                f'style="text-decoration:none;color:inherit;" class="rec-link">'
                f'<div class="rec-card">{poster_html}'
                f'<div class="rec-title">{display_title}</div>'
                f'<div class="rec-year">{year_display}</div>'
                f'{metrics_html}</div></a>'
            )
        cards_html.append('</div>')
        html = ''.join(cards_html)

    st.session_state["html_cache"]["cf"][target_tmdb_id] = html
    st.markdown(html, unsafe_allow_html=True)

@st.cache_data
def hybrid_recommendations(
    target_tmdb_id: int,
    ratings: pd.DataFrame,
    links: pd.DataFrame,
    cf_weight: float = 0.6,
    content_weight: float = 0.4,
    top_n: int = 12
):
    """
    Hybrid = weighted blend of:
      • Content-based: uses the SAME TF-IDF + movies_df used by the content UI
      • Item–Item CF : uses the SAME function used by the CF UI

    Note: We intentionally DO NOT accept a 'movies' arg here to avoid mixing
    TF-IDF (built on movies_df) with a different dataframe.
    """
    try:
        # --- Map TMDB -> MovieLens
        row = links.loc[links["tmdbId"] == int(target_tmdb_id)]
        if row.empty:
            return pd.DataFrame()
        target_ml_id = int(row.iloc[0]["movieId"])

        # --- Content candidates: STRICTLY from (movies_df, rec_tfidf_matrix)
        content_recs = content_based_recommendations(
            target_movie_id=target_ml_id,
            movies_df=movies_df,                 # <<< IMPORTANT: same df used to build TF-IDF
            tfidf_matrix=rec_tfidf_matrix,       # <<< IMPORTANT: same matrix
            top_n=top_n * 2
        )

        # If content fails, fall back to CF only
        if content_recs.empty:
            cf_only = item_item_recommendations(
                target_tmdb_id=target_tmdb_id,
                ratings=ratings,
                movies=movies_df,                # titles can be joined later; not used for sims
                links=links,
                min_overlap=15,
                top_n=top_n
            )
            return cf_only.head(top_n)

        # --- CF candidates: SAME function as CF UI
        cf_recs = item_item_recommendations(
            target_tmdb_id=target_tmdb_id,
            ratings=ratings,
            movies=movies_df,                    # safe to pass; titles merged later
            links=links,
            min_overlap=15,
            top_n=top_n * 2
        )

        if cf_recs.empty:
            return content_recs.head(top_n)

        # --- Normalize scores in each space
        cf_sim = cf_recs["similarity"].values
        cont_sim = content_recs["similarity"].values

        cf_min, cf_max = float(np.min(cf_sim)), float(np.max(cf_sim))
        ct_min, ct_max = float(np.min(cont_sim)), float(np.max(cont_sim))

        cf_recs = cf_recs.copy()
        content_recs = content_recs.copy()

        cf_recs["cf_score_norm"] = cf_recs["similarity"].rank(pct=True)  # higher sim -> higher pct
        content_recs["content_score_norm"] = content_recs["similarity"].rank(pct=True)

        if cf_max > cf_min:
            cf_recs["cf_score_norm"] = (cf_sim - cf_min) / (cf_max - cf_min)
        else:
            cf_recs["cf_score_norm"] = 0.0

        if ct_max > ct_min:
            content_recs["content_score_norm"] = (cont_sim - ct_min) / (ct_max - ct_min)
        else:
            content_recs["content_score_norm"] = 0.0

               # --- Blend
        cont_dict = {int(r.movieId): r for _, r in content_recs.iterrows()}
        cf_dict   = {int(r.movieId): r for _, r in cf_recs.iterrows()}
        all_ids = set(cont_dict) | set(cf_dict)

        rows = []
        for mid in all_ids:
            c_row = cont_dict.get(mid)
            f_row = cf_dict.get(mid)

            c = float(c_row["content_score_norm"]) if c_row is not None else np.nan
            f = float(f_row["cf_score_norm"])      if f_row is not None else np.nan

            w_c = content_weight if not np.isnan(c) else 0.0
            w_f = cf_weight      if not np.isnan(f) else 0.0
            denom = w_c + w_f
            hybrid = (w_c * (0.0 if np.isnan(c) else c) + w_f * (0.0 if np.isnan(f) else f)) / denom if denom > 0 else 0.0

            # title + tmdbId
            title = None
            tmdb  = None
            if f_row is not None:
                title = f_row.get("title", None)
                tmdb  = int(f_row.get("tmdbId", 0)) if pd.notna(f_row.get("tmdbId", np.nan)) else None
            if (title is None or pd.isna(title)) and c_row is not None:
                title = c_row.get("title", None)
            if tmdb is None:
                link = links.loc[links["movieId"] == mid]
                tmdb = int(link.iloc[0]["tmdbId"]) if not link.empty else 0

            rows.append({
                "movieId": mid,
                "title": title,
                "tmdbId": tmdb,
                "cf_score": f,
                "content_score": c,
                "hybrid_score": hybrid,
                "avg_rating": (f_row.get("avg_rating") if f_row is not None else c_row.get("avg_rating") if c_row is not None else np.nan),
                "rating_cnt": (f_row.get("rating_cnt") if f_row is not None else c_row.get("rating_count") if c_row is not None else 0),
            })

        out = pd.DataFrame(rows).sort_values("hybrid_score", ascending=False).head(top_n)
        return out

    except Exception as e:
        print(f"Hybrid filtering error: {e}")
        return pd.DataFrame()


def render_hybrid_section_horizontal(target_tmdb_id: int):
    """Render Hybrid recommendations using the horizontal 'rec-card' design; reuse cached HTML on reruns."""
    load_content_based_css()
    st.markdown("---")
    st.subheader("🚀 Best of Both Worlds")
    st.caption("Hybrid filtering combining collaborative (60%) and content (40%).")

    cached = st.session_state["html_cache"]["hybrid"].get(target_tmdb_id)
    if cached:
        st.markdown(cached, unsafe_allow_html=True)
        return

    with st.spinner("Loading hybrid recommendations..."):
        ratings, movies, links = load_ml_small()
        recs = hybrid_recommendations(
            target_tmdb_id=target_tmdb_id,
            ratings=ratings,
            links=links,            # <- keep
            cf_weight=0.6,
            content_weight=0.4,
            top_n=12
        )


        if recs.empty:
            st.info("No hybrid recommendations available.")
            return

        cards_html = ['<div class="rec-container">']
        for _, row in recs.iterrows():
            tmdb_id = int(row["tmdbId"])
            title_full = (row.get("title") or "").strip()
            clean_title = title_full.rsplit(" (", 1)[0] if " (" in title_full else title_full
            display_title = (clean_title[:25] + "...") if len(clean_title) > 25 else clean_title

            poster_url, _, release_date = fetch_tmdb_brief(tmdb_id)
            year_display = ""
            if release_date:
                try:
                    year_display = str(datetime.strptime(release_date, "%Y-%m-%d").year)
                except Exception:
                    year_display = ""

            hybrid_score = row.get("hybrid_score", np.nan)
            cf_score = row.get("cf_score", 0.0)
            content_score = row.get("content_score", 0.0)
            ml_avg = row.get("avg_rating", np.nan)
            ml_cnt = int(row.get("rating_cnt", 0)) if pd.notna(row.get("rating_cnt", np.nan)) else 0
            ml_text = f"{ml_avg:.2f}/5 • {ml_cnt} ratings" if pd.notna(ml_avg) and ml_cnt > 0 else "New"

            poster_html = (
                f'<img src="{poster_url}" class="rec-poster" alt="{display_title} Poster">'
                if poster_url else '<div class="rec-placeholder">🎬</div>'
            )

            score_lines = []
            if pd.notna(hybrid_score):
                score_lines.append(f'⚡ Score: {hybrid_score:.2f}')
            score_lines.append(f'CF {cf_score:.2f} • Content {content_score:.2f}')
            score_lines.append(f'⭐ {ml_text}')
            metrics_html = f'<div style="font-size:.75rem;color:#aaa;margin-top:4px;">' + "<br>".join(score_lines) + "</div>"

            cards_html.append(
                f'<a href="/Movie_Details?movie_id={int(row.get("movieId", 0))}&source=movielens" '
                f'style="text-decoration:none;color:inherit;" class="rec-link">'
                f'<div class="rec-card">{poster_html}'
                f'<div class="rec-title">{display_title}</div>'
                f'<div class="rec-year">{year_display}</div>'
                f'{metrics_html}</div></a>'
            )

        cards_html.append('</div>')
        html = ''.join(cards_html)

    st.session_state["html_cache"]["hybrid"][target_tmdb_id] = html
    st.markdown(html, unsafe_allow_html=True)

# Positive threshold & top-K
POS_THRESHOLD = 4.0
TOP_K = 12

# Evaluation
@st.cache_data
def build_relevant_set_for_seed(seed_ml_movie_id: int, ratings: pd.DataFrame, pos_threshold: float = POS_THRESHOLD) -> set[int]:
    """
    Relevant set for the seed movie:
      All movies (excluding the seed) that were positively rated by users who positively rated the seed.
    """
    # Users who positively rated the seed
    pos_users = ratings.loc[
        (ratings["movieId"] == int(seed_ml_movie_id)) & (ratings["rating"] >= pos_threshold),
        "userId"
    ].unique()
    if pos_users.size == 0:
        return set()

    # Other positive movies from these users (exclude the seed)
    rel = ratings.loc[
        (ratings["userId"].isin(pos_users)) & (ratings["rating"] >= pos_threshold),
        "movieId"
    ]
    relevant = set(map(int, rel.unique()))
    relevant.discard(int(seed_ml_movie_id))
    return relevant


def evaluate_algorithm_for_seed(
    algorithm_type: str,
    seed_ml_movie_id: int,
    tmdb_id: int | None,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    links: pd.DataFrame,
    top_k: int = TOP_K
):
    """
    Evaluate a single algorithm using the CURRENT SELECTED MOVIE as the seed.
    Returns: (metrics_dict, score_list)
    """
    # 1) Build relevant set for this seed
    relevant = build_relevant_set_for_seed(seed_ml_movie_id, ratings, POS_THRESHOLD)

    # 2) Get recommendations and scores for this seed
    rec_movie_ids: list[int] = []
    score_list: list[float] = []

    if algorithm_type == "content":
        # content recs via TF-IDF that you already computed globally
        rec_movie_ids = _recommend_content_seed(seed_ml_movie_id, top_n=top_k)
        # collect content similarity scores
        try:
            if not movies_df.empty and rec_movie_ids:
                seed_idx_list = movies_df.index[movies_df['movieId'] == seed_ml_movie_id].tolist()
                if seed_idx_list:
                    seed_idx = seed_idx_list[0]
                    for mid in rec_movie_ids:
                        ridx = movies_df.index[movies_df['movieId'] == mid].tolist()
                        if ridx:
                            ridx = ridx[0]
                            sim = cosine_similarity(rec_tfidf_matrix[seed_idx], rec_tfidf_matrix[ridx])[0][0]
                            score_list.append(float(sim))
        except Exception:
            pass

    elif algorithm_type == "cf":
        # Use the same CF pipeline as the UI (Pearson corr + min_overlap)
        # 1) Ensure we have the seed's MovieLens ID
        seed_ml = int(seed_ml_movie_id)

        # 2) Get CF recommendations exactly like the UI
        cf_recs_df = item_item_recommendations(
            target_tmdb_id=(tmdb_id if tmdb_id is not None else int(
                links.loc[links["movieId"] == seed_ml, "tmdbId"].iloc[0]
            )),
            ratings=ratings,
            movies=movies,
            links=links,
            min_overlap=15,      # keep this the same as your UI section
            top_n=top_k
        )

        # If no recs, bail early
        if cf_recs_df.empty:
            rec_movie_ids = []
            score_list = []
        else:
            rec_movie_ids = list(map(int, cf_recs_df["movieId"].tolist()))

            # 3) Pull the same Pearson correlation matrix (with the same min_overlap)
            corr_mat, valid_movies = build_cached_correlation_matrix(ratings, min_overlap=15)

            score_list = []
            if (seed_ml in valid_movies) and (seed_ml in corr_mat.columns):
                col = corr_mat[seed_ml]  # similarities vs seed (Pearson)
                for mid in rec_movie_ids:
                    if (mid in valid_movies) and (mid in corr_mat.index):
                        sim_val = col.get(mid, float('nan'))
                        if pd.notna(sim_val):
                            score_list.append(float(sim_val))
                        else:
                            score_list.append(0.0)
                    else:
                        score_list.append(0.0)
            else:
                # Seed wasn't in the valid set; fallback to zeros to avoid crashing
                score_list = [0.0] * len(rec_movie_ids)


    elif algorithm_type == "hybrid":
        # hybrid needs tmdb_id for your existing pipeline
        if tmdb_id is None:
            # map seed ML id to tmdb_id if missing
            row = links.loc[links["movieId"] == int(seed_ml_movie_id)]
            if not row.empty:
                tmdb_id = int(row.iloc[0]["tmdbId"])
        if tmdb_id is not None:
            hybrid_recs = hybrid_recommendations(
                target_tmdb_id=int(tmdb_id),
                ratings=ratings,
                links=links,
                cf_weight=0.6,
                content_weight=0.4,
                top_n=top_k
            )

            if not hybrid_recs.empty:
                rec_movie_ids = list(map(int, hybrid_recs["movieId"].tolist()))
                if "hybrid_score" in hybrid_recs.columns:
                    score_list = list(map(float, hybrid_recs["hybrid_score"].tolist()))
                else:
                    score_list = [0.5] * len(rec_movie_ids)
        else:
            rec_movie_ids = []

    # 3) Compute P/R/F1 for this seed
    p, r, f1 = _precision_recall_f1(rec_movie_ids, relevant)
    metrics = {"Precision@K": p, "Recall@K": r, "F1@K": f1}
    return metrics, score_list

# helper
@st.cache_resource
def build_item_user_sparse(ratings: pd.DataFrame):
    """
    Build cached sparse matrices for fast item–item similarity:
      - R_csr: (n_items x n_users) user-mean centered ratings
      - B_csr: binary co-rating matrix (same shape) for fast overlap counts
      - mlid_to_row / row_to_mlid: id maps
      - norms: L2 norms per item row
    """
    # Map users and movies to contiguous indices
    user_ids = ratings["userId"].astype(int).unique()
    movie_ids = ratings["movieId"].astype(int).unique()
    user_ids.sort()
    movie_ids.sort()

    uid_to_col = {u: i for i, u in enumerate(user_ids)}
    mlid_to_row = {m: i for i, m in enumerate(movie_ids)}
    row_to_mlid = np.array(movie_ids)

    # User mean-centering for Pearson-like cosine
    user_means = ratings.groupby("userId")["rating"].mean()

    rows, cols, data = [], [], []
    b_rows, b_cols, b_data = [], [], []

    for _, r in ratings.iterrows():
        u = int(r.userId); m = int(r.movieId); rt = float(r.rating)
        if u in uid_to_col and m in mlid_to_row:
            rows.append(mlid_to_row[m])
            cols.append(uid_to_col[u])
            data.append(rt - float(user_means.loc[u]))

            b_rows.append(mlid_to_row[m])
            b_cols.append(uid_to_col[u])
            b_data.append(1.0)

    n_items = len(movie_ids)
    n_users = len(user_ids)
    R_csr = sp.csr_matrix((data, (rows, cols)), shape=(n_items, n_users))
    B_csr = sp.csr_matrix((b_data, (b_rows, b_cols)), shape=(n_items, n_users))

    # L2 norms per item row (avoid div by zero)
    norms = np.sqrt(R_csr.multiply(R_csr).sum(axis=1)).A1
    norms[norms == 0.0] = 1e-12

    return R_csr, B_csr, mlid_to_row, row_to_mlid, norms

def _recommend_cf_seed(seed_ml_movie_id: int, ratings: pd.DataFrame, min_overlap=20, top_n=TOP_K):
    """
    FAST item–item CF using sparse cosine similarity on user-mean–centered ratings.
    Respects min_overlap via a binary co-rating matrix.
    """
    R_csr, B_csr, mlid_to_row, row_to_mlid, norms = build_item_user_sparse(ratings)
    seed_row = mlid_to_row.get(int(seed_ml_movie_id))
    if seed_row is None:
        return []

    # Cosine similarity: (R[i] @ R.T) / (||R[i]|| * ||R||)
    # 1 x n_items vector
    numer = (R_csr[seed_row] @ R_csr.T).toarray().ravel()
    denom = norms[seed_row] * norms
    sims = np.divide(numer, denom, out=np.zeros_like(numer), where=(denom != 0))

    # Co-rater overlap via binary matrix
    overlaps = (B_csr[seed_row] @ B_csr.T).toarray().ravel()

    # Remove self
    sims[seed_row] = -np.inf

    # Filter by min_overlap
    valid = np.where(overlaps >= min_overlap)[0]
    if valid.size == 0:
        # fall back: no overlap filter, just take highest sims excluding self
        valid = np.argsort(-sims)[:top_n * 5]

    # Top-N among valid
    top_idx = valid[np.argsort(-sims[valid])][:top_n]
    top_ml_ids = [int(row_to_mlid[i]) for i in top_idx if i != seed_row]
    return top_ml_ids[:top_n]

def _recommend_content_seed(seed_ml_movie_id: int, top_n=TOP_K):
    """Use your content-based pipeline (TF-IDF from movies_df) with a seed ML movie."""
    if movies_df.empty:
        return []
    try:
        if 'movieId' not in movies_df.columns:
            return []
        # locate seed index
        idx_list = movies_df.index[movies_df['movieId'] == seed_ml_movie_id].tolist()
        if not idx_list:
            return []
        seed_idx = idx_list[0]

        # similarity vector
        sims = list(enumerate(cosine_similarity(rec_tfidf_matrix[seed_idx], rec_tfidf_matrix).flatten()))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)

        # top-N movieIds (skip itself)
        out = []
        for i, score in sims[1: top_n + 50]:  # take a bit more then dedup
            mid = int(movies_df.iloc[i]['movieId'])
            if mid != seed_ml_movie_id:
                out.append(mid)
            if len(out) >= top_n:
                break
        return out
    except Exception:
        return []

def _precision_recall_f1(recommended: list[int], relevant: set[int]):
    if not recommended:
        return 0.0, 0.0, 0.0
    rec_set = set(recommended)
    hit = len(rec_set & relevant)
    precision = hit / len(recommended)
    recall = hit / len(relevant) if relevant else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

# GRAPH
# Add this function to create evaluation plots
def create_evaluation_plots(metrics_dict, algorithm_name, similarity_scores=None):
    """Create evaluation plots for a specific algorithm"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Precision, Recall, F1
    metrics = ['Precision@K', 'Recall@K', 'F1@K']
    values = [metrics_dict[algorithm_name][metric] for metric in metrics]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax1.bar(metrics, values, color=colors)
    ax1.set_title(f'{algorithm_name} Performance Metrics')
    ax1.set_ylabel('Score')
    
    # Calculate appropriate y-axis limits
    max_value = max(values) if values else 1.0
    y_margin = max_value * 0.15  # 15% margin
    ax1.set_ylim(0, max_value + y_margin)
    
    # Add value labels on bars with adjusted position
    for i, v in enumerate(values):
        ax1.text(i, v + (y_margin * 0.1), f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Similarity distribution (if available)
    if similarity_scores and len(similarity_scores) > 0:
        ax2.hist(similarity_scores, bins=20, color='#6A0DAD', alpha=0.7, edgecolor='black')
        ax2.set_title('Similarity Score Distribution')
        ax2.set_xlabel('Similarity Score')
        ax2.set_ylabel('Frequency')
        
        # Calculate appropriate position for mean line
        mean_val = np.mean(similarity_scores)
        ax2.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, 
                   label=f'Mean: {mean_val:.3f}')
        ax2.legend()
    
    plt.tight_layout()
    return fig

def create_hybrid_evaluation_plots(metrics_dict, hybrid_scores=None):
    """Create evaluation plots specifically for hybrid algorithm"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Precision, Recall, F1 for hybrid
    metrics = ['Precision@K', 'Recall@K', 'F1@K']
    values = [metrics_dict["Hybrid"][metric] for metric in metrics]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax1.bar(metrics, values, color=colors)
    ax1.set_title('Hybrid Algorithm Performance Metrics')
    ax1.set_ylabel('Score')
    
    # Calculate appropriate y-axis limits
    max_value = max(values) if values else 1.0
    y_margin = max_value * 0.15  # 15% margin
    ax1.set_ylim(0, max_value + y_margin)
    
    # Add value labels on bars with adjusted position
    for i, v in enumerate(values):
        ax1.text(i, v + (y_margin * 0.1), f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Hybrid score distribution
    if hybrid_scores and len(hybrid_scores) > 0:
        ax2.hist(hybrid_scores, bins=20, color='#6A0DAD', alpha=0.7, edgecolor='black')
        ax2.set_title('Hybrid Score Distribution')
        ax2.set_xlabel('Hybrid Score')
        ax2.set_ylabel('Frequency')
        
        # Calculate appropriate position for mean line
        mean_val = np.mean(hybrid_scores)
        ax2.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, 
                   label=f'Mean: {mean_val:.3f}')
        ax2.legend()
    else:
        # Fallback if no hybrid scores available
        ax2.text(0.5, 0.5, 'No hybrid scores available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Hybrid Score Distribution')
    
    plt.tight_layout()
    return fig

def plot_score_distribution(title: str, scores: list[float], xlabel: str = "Score"):
    """Histogram-only plot with a dashed mean line."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    if scores and len(scores) > 0:
        ax.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")

        mean_val = float(np.mean(scores))
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1)
        # place mean label neatly inside the plot area
        y_top = ax.get_ylim()[1]
        ax.text(mean_val, y_top * 0.95, f"Mean: {mean_val:.3f}",
                color='red', ha='right', va='top')
    else:
        ax.text(0.5, 0.5, 'No scores available',
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig

def summarize_scores(scores):
    """Return a compact distribution summary for a list of scores."""
    if not scores or len(scores) == 0:
        return {
            "N": 0, "Mean": np.nan, "Std": np.nan,
            "Min": np.nan, "25%": np.nan, "50%": np.nan, "75%": np.nan, "Max": np.nan
        }
    arr = np.asarray(scores, dtype=float)
    return {
        "N": int(arr.size),
        "Mean": float(np.mean(arr)),
        "Std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "Min": float(np.min(arr)),
        "25%": float(np.percentile(arr, 25)),
        "50%": float(np.percentile(arr, 50)),  # median
        "75%": float(np.percentile(arr, 75)),
        "Max": float(np.max(arr)),
    }


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
    movies_df["genre_list"] = (
        movies_df["genres"].fillna("")
        .str.replace("(no genres listed)", "", regex=False)
        .str.split("|")
        .apply(lambda lst: [g for g in lst if g])  # drop empty strings
    )
else:
    movies_df['tag'] = ''

# Initialize recommendation system
if not movies_df.empty:
    rec_vectorizer, rec_tfidf_matrix = get_recommendation_vectorizer_and_matrix(movies_df)

# ----- Main Application Logic -----
# ---------- MovieLens loaders ----------
@st.cache_data
def load_ml_small(base_dir: str = "dataset/ml-latest-small"):
    """Load MovieLens ratings/movies/links (small) and return dataframes."""
    ratings = pd.read_csv(f"{base_dir}/ratings.csv")      # userId, movieId, rating, timestamp
    movies  = pd.read_csv(f"{base_dir}/movies.csv")       # movieId, title, genres
    links   = pd.read_csv(f"{base_dir}/links.csv")        # movieId, imdbId, tmdbId
    # keep only rows that have a valid TMDB id
    links = links[links["tmdbId"].notna() & (links["tmdbId"] > 0)]
    links["tmdbId"] = links["tmdbId"].astype(int)
    return ratings, movies, links


# ---------------------------------------------------------------------------
load_navbar() 
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

# Map between MovieLens movieId and TMDB tmdbId based on source
ratings_small, movies_small, links_small = load_ml_small()

movie = None
ml_movie_id = None
tmdb_id = None

try:
    incoming_id = int(str(movie_id).strip())
except ValueError:
    st.error("Invalid movie ID")
    st.stop()

if str(source).lower() == "api":
    # Treat incoming id as TMDB id
    tmdb_id = incoming_id
    # Try to find corresponding MovieLens movieId for ratings and content-based sections
    if not links_small.empty:
        row_map = links_small.loc[links_small["tmdbId"] == tmdb_id]
        if not row_map.empty:
            ml_movie_id = int(row_map.iloc[0]["movieId"])
            if not movies_df.empty:
                movie_rows = movies_df.loc[movies_df["movieId"] == ml_movie_id]
                if not movie_rows.empty:
                    movie = movie_rows.iloc[0]
else:
    # Treat incoming id as MovieLens movieId
    ml_movie_id = incoming_id
    if not movies_df.empty:
        movie_rows = movies_df.loc[movies_df["movieId"] == ml_movie_id]
        if not movie_rows.empty:
            movie = movie_rows.iloc[0]
    if not links_small.empty:
        row_map = links_small.loc[links_small["movieId"] == ml_movie_id]
        if not row_map.empty:
            tmdb_id = int(row_map.iloc[0]["tmdbId"]) if not pd.isna(row_map.iloc[0]["tmdbId"]) else None

# --- Clear cached recommendation HTML if the context movie changes ---
# Make sure our HTML caches exist
st.session_state.setdefault("html_cache", {"content": {}, "cf": {}, "hybrid": {}})
st.session_state.setdefault("last_ids", {"ml": None, "tmdb": None})
st.session_state.setdefault("evaluation_cache", {"content": {}, "cf": {}, "hybrid": {}, "compare": {}, "rmse": {}})

current_ids = {"ml": ml_movie_id, "tmdb": tmdb_id}
last_ids = st.session_state["last_ids"]

if current_ids != last_ids:
    # Clear only the entries for the current movie ids (if present)
    for k in ("content", "cf", "hybrid"):
        if ml_movie_id is not None:
            st.session_state["html_cache"][k].pop(ml_movie_id, None)
            st.session_state["evaluation_cache"][k].pop(ml_movie_id, None)
        if tmdb_id is not None:
            st.session_state["html_cache"][k].pop(tmdb_id, None)
            st.session_state["evaluation_cache"][k].pop(tmdb_id, None)
    
    # Clear comparison and RMSE results
    st.session_state["evaluation_cache"]["compare"] = {}
    st.session_state["evaluation_cache"]["rmse"] = {}
    
    # Clear all figures
    st.session_state["figs"] = {}
    
    st.session_state["last_ids"] = current_ids



# If we still have no movie detail, continue with TMDB-only rendering; else use MovieLens info too
movie_title = movie.get("clean_title", movie.get("title", "")) if movie is not None else None
movie_year = movie.get("year") if movie is not None else None
avg_rating = movie.get("avg_rating", 0) if movie is not None else 0
rating_count = movie.get("rating_count", 0) if movie is not None else 0

# Fetch TMDB details using tmdb_id if available; otherwise fall back to search by title
tmdb_data = None
if tmdb_id:
    tmdb_data = fetch_movie_details_by_tmdb_id(tmdb_id)
elif movie_title:
    tmdb_result = search_tmdb_by_title(movie_title, movie_year)
    if tmdb_result:
        tmdb_id = tmdb_result.get("id")
        if tmdb_id:
            tmdb_data = fetch_movie_details_by_tmdb_id(tmdb_id)

# Define detail_id_int for content-based section (uses MovieLens movieId)
detail_id_int = ml_movie_id if ml_movie_id is not None else None

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
    # Prefer TMDB title if available; fall back to MovieLens title
    title_display = (tmdb_data.get("title") if tmdb_data and tmdb_data.get("title") else movie_title) or "No Title"
    st.title(title_display)
    
    # Tagline (if available from TMDB)
    if tmdb_data:
        tagline = (tmdb_data.get("tagline") or "").strip()
        if tagline:
            st.markdown(f'<div class="tagline">"{tagline}"</div>', unsafe_allow_html=True)

    # Release date and runtime
    # Release date from TMDB if present; otherwise use MovieLens year (if available)
    release_fmt = None
    try:
        if tmdb_data and tmdb_data.get("release_date"):
            release_fmt = datetime.strptime(tmdb_data["release_date"], "%Y-%m-%d").strftime("%B %d, %Y")
        elif movie_year is not None and pd.notna(movie_year):
            release_fmt = f"{int(movie_year)}"
    except Exception:
        if movie_year is not None and pd.notna(movie_year):
            release_fmt = f"{int(movie_year)}"
    if release_fmt:
        st.write(f"**📅 Release Date:** {release_fmt}")

    # Runtime (from TMDB if available)
    if tmdb_data and tmdb_data.get("runtime"):
        runtime = tmdb_data.get("runtime", 0)
        if runtime:
            st.write(f"**⏱️ Runtime:** {runtime} min")
        else:
            st.write("**⏱️ Runtime:** N/A")

    # Rating
    if avg_rating > 0:
        star_html = generate_star_display_html(avg_rating, max_rating=5)
        st.markdown(f'<div class="rating-display">{star_html} <div class="rating-text">{avg_rating:.1f}/5</div></div>', unsafe_allow_html=True)
        st.markdown("")

    # Genres - prioritize TMDB if available
    if tmdb_data and tmdb_data.get("genres"):
        st.markdown("**🎞️ Genres:**")
        # Create a single string with all genres joined horizontally
        genres_html = ''.join([f'<span class="genre-badge">{genre["name"]}</span>' for genre in tmdb_data["genres"]])

        # Display the genres in a single <p> tag
        st.markdown(f'<p class="genre-container">{genres_html}</p>', unsafe_allow_html=True)
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
        
# --- All recommendation sections under the details ---
try:
    # Content-Based Filtering (MovieLens-based)
    if detail_id_int is not None and not movies_df.empty:
        render_content_based_recommendations(detail_id_int, movies_df, rec_vectorizer, rec_tfidf_matrix)

        # content_eval_slot = st.empty()

        # # Content-Based evaluation (SEEDED)
        # if st.button("Evaluate Content-Based Algorithm", key="eval_content"):
        #     ratings_small, movies_small, links_small = load_ml_small()
        #     if detail_id_int is None:
        #         st.warning("This evaluation needs a MovieLens movieId.")
        #     else:
        #         _, content_similarities = evaluate_algorithm_for_seed(
        #             "content",
        #             seed_ml_movie_id=int(detail_id_int),
        #             tmdb_id=tmdb_id,
        #             ratings=ratings_small,
        #             movies=movies_small,
        #             links=links_small,
        #             top_k=TOP_K
        #         )
        #         fig = plot_score_distribution("Content-Based Similarity Distribution", content_similarities, "Similarity")
        #         st.session_state["figs"]["content_scores_fig"] = fig

        # # show only the histogram
        # if st.session_state["figs"].get("content_scores_fig") is not None:
        #     st.pyplot(st.session_state["figs"]["content_scores_fig"])


                

    # In the collaborative filtering section, add the evaluation button and display
    if tmdb_id is not None:
        render_cf_section_horizontal(int(tmdb_id))
        
        # # Add evaluation button for collaborative filtering
        # cf_eval_slot = st.empty()

        # # Collaborative Filtering evaluation (SEEDED)
        # if st.button("Evaluate Collaborative Filtering Algorithm", key="eval_cf"):
        #     ratings_small, movies_small, links_small = load_ml_small()
        #     if detail_id_int is None:
        #         st.warning("This evaluation needs a MovieLens movieId.")
        #     else:
        #         _, cf_similarities = evaluate_algorithm_for_seed(
        #             "cf",
        #             seed_ml_movie_id=int(detail_id_int),
        #             tmdb_id=tmdb_id,
        #             ratings=ratings_small,
        #             movies=movies_small,
        #             links=links_small,
        #             top_k=TOP_K
        #         )
        #         fig = plot_score_distribution("Collaborative Filtering Similarity Distribution", cf_similarities, "Similarity")
        #         st.session_state["figs"]["cf_scores_fig"] = fig

        # # show only the histogram
        # if st.session_state["figs"].get("cf_scores_fig") is not None:
        #     st.pyplot(st.session_state["figs"]["cf_scores_fig"])


                

    # In the hybrid recommendations section, add the evaluation button and display
    if tmdb_id is not None:
        render_hybrid_section_horizontal(int(tmdb_id))
        
        # # Add evaluation button for hybrid
        # hybrid_eval_slot = st.empty()

        # # Hybrid evaluation (SEEDED)
        # if st.button("Evaluate Hybrid Algorithm", key="eval_hybrid"):
        #     ratings_small, movies_small, links_small = load_ml_small()
        #     if detail_id_int is None:
        #         st.warning("This evaluation needs a MovieLens movieId.")
        #     else:
        #         _, hybrid_scores = evaluate_algorithm_for_seed(
        #             "hybrid",
        #             seed_ml_movie_id=int(detail_id_int),
        #             tmdb_id=tmdb_id,
        #             ratings=ratings_small,
        #             movies=movies_small,
        #             links=links_small,
        #             top_k=TOP_K
        #         )
        #         fig = plot_score_distribution("Hybrid Score Distribution", hybrid_scores, "Hybrid Score")
        #         st.session_state["figs"]["hybrid_scores_fig"] = fig

        # # show only the histogram
        # if st.session_state["figs"].get("hybrid_scores_fig") is not None:
        #     st.pyplot(st.session_state["figs"]["hybrid_scores_fig"])


    
except Exception as e:
    st.warning(f"Recommendations unavailable: {e}")

# st.markdown("---")
# st.header("📊 Score Distribution Summary")
# st.caption("Per-algorithm distribution of similarity / hybrid scores for Top-K recommendations of the selected seed movie.")

# if st.button("🔎 Compare Score Distributions", key="compare_score_distributions"):
#     with st.spinner("Computing score distributions..."):
#         ratings_small, movies_small, links_small = load_ml_small()
#         if detail_id_int is None:
#             st.warning("This comparison needs a MovieLens movieId.")
#         else:
#             # get only the score lists (we ignore the precision/recall metrics)
#             _, content_scores = evaluate_algorithm_for_seed(
#                 "content",
#                 seed_ml_movie_id=int(detail_id_int),
#                 tmdb_id=tmdb_id,
#                 ratings=ratings_small,
#                 movies=movies_small,
#                 links=links_small,
#                 top_k=TOP_K
#             )
#             _, cf_scores = evaluate_algorithm_for_seed(
#                 "cf",
#                 seed_ml_movie_id=int(detail_id_int),
#                 tmdb_id=tmdb_id,
#                 ratings=ratings_small,
#                 movies=movies_small,
#                 links=links_small,
#                 top_k=TOP_K
#             )
#             _, hybrid_scores = evaluate_algorithm_for_seed(
#                 "hybrid",
#                 seed_ml_movie_id=int(detail_id_int),
#                 tmdb_id=tmdb_id,
#                 ratings=ratings_small,
#                 movies=movies_small,
#                 links=links_small,
#                 top_k=TOP_K
#             )

#             # summarize
#             summary = {
#                 "Content-Based": summarize_scores(content_scores),
#                 "Collaborative": summarize_scores(cf_scores),
#                 "Hybrid": summarize_scores(hybrid_scores),
#             }
#             df = pd.DataFrame(summary).T[
#                 ["N", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
#             ]
#             st.session_state["score_distribution_table"] = df

# # show the distribution table (no precision/recall/F1, no graphs)
# if "score_distribution_table" in st.session_state:
#     fmt = {k: "{:.3f}" for k in ["Mean", "Std", "Min", "25%", "50%", "75%", "Max"]}
#     st.dataframe(st.session_state["score_distribution_table"].style.format(fmt))


# st.markdown("---")
# st.subheader("📝 Quick Feedback")
# with st.form("feedback_form"):
#     algo_viewed = st.selectbox("Which algorithm are you evaluating?", ["Content-Based", "Collaborative", "Hybrid"])
#     q_rel = st.slider("How relevant were the recommendations?", 1, 5, 3)
#     q_sat = st.slider("Overall satisfaction?", 1, 5, 3)
#     q_div = st.slider("Perceived diversity?", 1, 5, 3)
#     comments = st.text_area("Any comments?")
#     submitted = st.form_submit_button("Submit feedback")
#     if submitted:
#         fb_row = {
#             "timestamp": int(time.time()),
#             "movieId": detail_id_int,
#             "tmdbId": tmdb_id,
#             "algorithm": algo_viewed,
#             "relevance": q_rel,
#             "satisfaction": q_sat,
#             "diversity": q_div,
#             "comments": comments
#         }
#         fb_file = "user_feedback.csv"
#         if os.path.exists(fb_file):
#             existing = pd.read_csv(fb_file)
#             pd.concat([existing, pd.DataFrame([fb_row])], ignore_index=True).to_csv(fb_file, index=False)
#         else:
#             pd.DataFrame([fb_row]).to_csv(fb_file, index=False)
#         st.success("Thanks for your feedback!")

# ------------------ EVALUATION HUB + FEEDBACK ------------------
st.markdown("---")
load_eval_css()
st.header("🧪 Evaluation Hub")

# Controls row
with st.container():
    c1, c2 = st.columns([1, 1])
    with c1:
        top_k = st.slider("Top-K", 5, 30, TOP_K, help="How many recs to evaluate per algorithm")
    # with c2:
    #     pos_threshold = st.slider("Positive ≥", 3.0, 5.0, POS_THRESHOLD, 0.5,
    #                               help="What counts as a 'positive' rating")
    with c2:
        run_all = st.checkbox("Run all", value=False, help="Evaluate Content, CF, and Hybrid together")
    # with c4:
    #     st.markdown(
    #         f"""<span class="kpi-pill">Seed: {detail_id_int or "—"}</span>
    #             <span class="kpi-pill">TMDB: {tmdb_id or "—"}</span>""",
    #         unsafe_allow_html=True
    #     )

# Make sliders affect the evaluator (don’t touch global constants)
_local_TOP_K = int(top_k)
# _local_POS_THRESHOLD = float(pos_threshold)

tabs = st.tabs(["📈 Evaluate", "⚖️ Compare", "📝 Quick Feedback"])

# --------- Tab 1: Evaluate ---------
with tabs[0]:
    if detail_id_int is None:
        st.info("Select a MovieLens title to evaluate recommendations.")
    else:
        cA, cB, cC = st.columns(3)

        def run_eval(alg_name:str):
            # temporarily override POS_THRESHOLD inside the helper via param
            # -> reuse evaluator by passing through
            metrics, scores = evaluate_algorithm_for_seed(
                "content" if alg_name=="Content" else "cf" if alg_name=="CF" else "hybrid",
                seed_ml_movie_id=int(detail_id_int),
                tmdb_id=tmdb_id,
                ratings=ratings_small,
                movies=movies_small,
                links=links_small,
                top_k=_local_TOP_K
            )
            return metrics, scores

        # Ensure base datasets loaded
        ratings_small, movies_small, links_small = load_ml_small()

        def render_block(title, key_prefix, run_now=False):
            placeholder = st.empty()
            with placeholder.container():
                st.subheader(title)
                go = st.button(f"Run {title}", key=f"btn_{key_prefix}") or run_now
                box = st.container()
                if go:
                    with st.spinner("Evaluating…"):
                        m, s = run_eval(
                            "Content" if key_prefix=="content" else
                            "CF" if key_prefix=="cf" else "hybrid"
                        )
                        # KPIs
                        # cards = "".join([
                        #     kpi_card("Precision@K", m.get("Precision@K", 0.0)),
                        #     kpi_card("Recall@K",    m.get("Recall@K",    0.0)),
                        #     kpi_card("F1@K",       m.get("F1@K",       0.0)),
                        # ])
                        # st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)

                        # Distribution
                        label = "Similarity" if key_prefix!="hybrid" else "Hybrid Score"
                        fig = plot_score_distribution(f"{title} — Score Distribution", s, label)
                        st.pyplot(fig)
                        # Cache last run
                        st.session_state["figs"][f"{key_prefix}_scores_fig"] = fig
                        st.session_state["evaluation_cache"].setdefault(key_prefix, {})[detail_id_int] = {
                            "metrics": m, "scores": s
                        }
                else:
                    # Show last run if available
                    prev = st.session_state["evaluation_cache"].get(key_prefix, {}).get(detail_id_int)
                    if prev:
                        # cards = "".join([
                        #     kpi_card("Precision@K", prev["metrics"].get("Precision@K", 0.0)),
                        #     kpi_card("Recall@K",    prev["metrics"].get("Recall@K",    0.0)),
                        #     kpi_card("F1@K",       prev["metrics"].get("F1@K",       0.0)),
                        # ])
                        # st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)
                        st.pyplot(st.session_state["figs"].get(f"{key_prefix}_scores_fig"))
                    else:
                        st.caption("No results yet. Click **Run** to evaluate.")

        with cA:
            render_block("Content-Based", "content", run_now=run_all)
        with cB:
            render_block("Collaborative Filtering", "cf", run_now=run_all)
        with cC:
            render_block("Hybrid", "hybrid", run_now=run_all)


# --------- Tab 2: Compare ---------
with tabs[1]:
    if detail_id_int is None:
        st.info("Select a MovieLens title to compare distributions.")
    else:
        # Pull fresh scores at current K/threshold
        _, content_scores = evaluate_algorithm_for_seed("content", int(detail_id_int), tmdb_id,
                                                        ratings_small, movies_small, links_small, _local_TOP_K)
        _, cf_scores      = evaluate_algorithm_for_seed("cf", int(detail_id_int), tmdb_id,
                                                        ratings_small, movies_small, links_small, _local_TOP_K)
        _, hybrid_scores  = evaluate_algorithm_for_seed("hybrid", int(detail_id_int), tmdb_id,
                                                        ratings_small, movies_small, links_small, _local_TOP_K)

        summary = {
            "Content-Based": summarize_scores(content_scores),
            "Collaborative": summarize_scores(cf_scores),
            "Hybrid": summarize_scores(hybrid_scores),
        }
        df = pd.DataFrame(summary).T[["N","Mean","Std","Min","25%","50%","75%","Max"]]
        fmt = {
            "N": "{:.0f}",  # no decimals
            "Mean": "{:.3f}",
            "Std": "{:.3f}",
            "Min": "{:.3f}",
            "25%": "{:.3f}",
            "50%": "{:.3f}",
            "75%": "{:.3f}",
            "Max": "{:.3f}",
        }        
        st.dataframe(df.style.format(fmt))
        st.download_button("⬇️ Download CSV", data=df.to_csv().encode("utf-8"),
                            file_name="score_distribution_summary.csv", mime="text/csv")

# --------- Tab 3: Quick Feedback ---------
with tabs[2]:
    st.subheader("Your take on these recommendations")
    st.caption("Help us improve: quick ratings below (anonymous).")

    with st.form("feedback_form_v2", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            algo_viewed = st.selectbox("Algorithm", ["Content-Based", "Collaborative", "Hybrid"])
        with c2:
            tags = st.multiselect(
                "What stood out?",
                ["Accurate", "Surprising", "Too Niche", "Too Popular",
                 "Diverse", "Repetitive", "New to me", "I’ve seen most"]
            )

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            q_rel = st.slider("Relevance", 1, 5, 4, help="How on-target were the picks?")
        with col_r2:
            q_sat = st.slider("Satisfaction", 1, 5, 4, help="Overall, how happy are you?")
        with col_r3:
            q_div = st.slider("Diversity", 1, 5, 3, help="Did the list feel varied?")

        comments = st.text_area(
            "Anything else? (optional)",
            placeholder="Tell us what you loved or what missed."
        )

        # Now this updates live because agree is outside the form
        submitted = st.form_submit_button("Submit feedback ✅")

        if submitted:
            fb_row = {
                "timestamp": int(time.time()),
                "movieId": detail_id_int,
                "tmdbId": tmdb_id,
                "algorithm": algo_viewed,
                "relevance": q_rel,
                "satisfaction": q_sat,
                "diversity": q_div,
                "tags": "|".join(tags),
                "comments": comments
            }
            fb_file = "user_feedback.csv"
            if os.path.exists(fb_file):
                existing = pd.read_csv(fb_file)
                pd.concat([existing, pd.DataFrame([fb_row])], ignore_index=True).to_csv(fb_file, index=False)
            else:
                pd.DataFrame([fb_row]).to_csv(fb_file, index=False)
            st.success("Thanks for your feedback!")
            st.balloons()
