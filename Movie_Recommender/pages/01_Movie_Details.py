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

# Add this function to evaluate a specific algorithm
@st.cache_data
def evaluate_single_algorithm(algorithm_type, ratings, movies, links, sample_users=150, top_k=10):
    """
    Evaluate a single algorithm (content, cf, or hybrid)
    Returns metrics and similarity scores if available
    """
    seeds = _build_movie_seed_map(ratings, min_pos=3)
    if len(seeds) > sample_users:
        random.shuffle(seeds)
        seeds = seeds[:sample_users]
    
    user_pos = _build_user_positives(ratings)
    results = {"p": [], "r": [], "f1": []}
    similarity_scores = []  # To store similarity scores if available
    
    for uid, seed_mid, held_mid in seeds:
        relevant = set(user_pos[uid]) - {seed_mid}
        if not relevant:
            continue
            
        if algorithm_type == "content":
            rec = _recommend_content_seed(seed_mid, top_n=top_k)
            # For content-based, we can get similarity scores
            try:
                if not movies_df.empty:
                    seed_idx = movies_df.index[movies_df['movieId'] == seed_mid].tolist()
                    if seed_idx:
                        seed_idx = seed_idx[0]
                        # Get similarity scores for recommendations
                        for rec_mid in rec:
                            rec_idx = movies_df.index[movies_df['movieId'] == rec_mid].tolist()
                            if rec_idx:
                                rec_idx = rec_idx[0]
                                sim = cosine_similarity(rec_tfidf_matrix[seed_idx], rec_tfidf_matrix[rec_idx])[0][0]
                                similarity_scores.append(sim)
            except:
                pass
                
        elif algorithm_type == "cf":
            rec = _recommend_cf_seed(seed_mid, ratings, min_overlap=15, top_n=top_k)
            # For CF, we can get similarity scores from the CF function
            try:
                R_csr, B_csr, mlid_to_row, row_to_mlid, norms = build_item_user_sparse(ratings)
                seed_row = mlid_to_row.get(int(seed_mid))
                if seed_row is not None:
                    # Get similarity scores for recommendations
                    for rec_mid in rec:
                        rec_row = mlid_to_row.get(int(rec_mid))
                        if rec_row is not None:
                            numer = (R_csr[seed_row] @ R_csr[rec_row].T).toarray()[0][0]
                            denom = norms[seed_row] * norms[rec_row]
                            if denom != 0:
                                sim = numer / denom
                                similarity_scores.append(sim)
            except:
                pass
                
        elif algorithm_type == "hybrid":
            # For hybrid, we need to get the actual hybrid scores
            try:
                # Get the TMDB ID for the seed movie
                seed_tmdb_row = links[links["movieId"] == seed_mid]
                if not seed_tmdb_row.empty:
                    seed_tmdb_id = int(seed_tmdb_row.iloc[0]["tmdbId"])
                    
                    # Get hybrid recommendations with scores
                    hybrid_recs = hybrid_recommendations(
                        target_tmdb_id=seed_tmdb_id,
                        ratings=ratings,
                        movies=movies,
                        links=links,
                        top_n=top_k
                    )
                    
                    if not hybrid_recs.empty:
                        # Extract the movie IDs for precision/recall calculation
                        rec = hybrid_recs['movieId'].tolist()
                        
                        # Extract the hybrid scores for the histogram (with safety check)
                        if 'hybrid_score' in hybrid_recs.columns:
                            hybrid_scores_list = hybrid_recs['hybrid_score'].tolist()
                            similarity_scores.extend(hybrid_scores_list)
                        else:
                            # If hybrid_score column doesn't exist, create default scores
                            hybrid_scores_list = [0.5] * len(rec)  # Default score
                            similarity_scores.extend(hybrid_scores_list)
                    else:
                        rec = []
                else:
                    rec = []
            except Exception as e:
                print(f"Error getting hybrid recommendations: {e}")
                rec = []
        
        p, r, f1 = _precision_recall_f1(rec, relevant)
        results["p"].append(p)
        results["r"].append(r)
        results["f1"].append(f1)
    
    # Calculate averages
    def avg(lst): 
        return float(np.mean(lst)) if lst else 0.0
    
    metrics = {
        "Precision@K": avg(results["p"]),
        "Recall@K": avg(results["r"]),
        "F1@K": avg(results["f1"])
    }
    
    return metrics, similarity_scores

@st.cache_data
def evaluate_rmse_baseline(ratings_df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Pure-Python baseline predictor: r_hat(u,i) = mu + b_u + b_i
    (global mean + user bias + item bias). No external deps.
    Returns RMSE/MAE quickly and reliably.
    """
    # Shuffle & split
    df = ratings_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    split = int(len(df) * (1 - test_size))
    train = df.iloc[:split]
    test  = df.iloc[split:]

    mu = float(train["rating"].mean())
    bu = train.groupby("userId")["rating"].mean() - mu
    bi = train.groupby("movieId")["rating"].mean() - mu

    # Predict baseline
    # (vectorized via map; fallback to 0 if unseen user/movie in test)
    pred = mu \
        + test["userId"].map(bu).fillna(0.0).astype(float) \
        + test["movieId"].map(bi).fillna(0.0).astype(float)

    # Clip to MovieLens scale
    pred = np.clip(pred.values, 0.5, 5.0)
    true = test["rating"].values.astype(float)

    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    mae  = float(np.mean(np.abs(true - pred)))
    return {"RMSE": rmse, "MAE": mae, "model": "Baseline (μ + bᵤ + bᵢ)"}


def _plot_bar(ax, labels, values, title, ylabel):
    """Simple labeled bar chart for metrics (no custom colors/styles)."""
    # Coerce to floats and handle empty/None
    vals = [float(v) if v is not None else 0.0 for v in values]
    ax.bar(labels, vals)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    
    # Calculate appropriate y-axis limits
    max_value = max(vals) if vals else 1.0
    y_margin = max_value * 0.15  # 15% margin
    ax.set_ylim(0, max_value + y_margin)
    
    # value labels above bars with adjusted position
    for i, v in enumerate(vals):
        ax.text(i, v + (y_margin * 0.1), f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)

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

# ---- Helper: Positive threshold & top-K
POS_THRESHOLD = 4.0
TOP_K = 10

def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

@st.cache_data
def _build_user_positives(ratings_df: pd.DataFrame, pos_threshold: float = POS_THRESHOLD):
    """Return dict: userId -> set(movieId) with ratings >= threshold."""
    pos = ratings_df.loc[ratings_df["rating"] >= pos_threshold, ["userId", "movieId"]]
    user_pos = defaultdict(set)
    for _, r in pos.iterrows():
        user_pos[int(r.userId)].add(int(r.movieId))
    return user_pos

@st.cache_data
def _build_movie_seed_map(ratings_df: pd.DataFrame, min_pos=3):
    """
    For each user, pick a 'seed' positive movie (to simulate user context) and
    a 'held-out' positive movie to test hit@K.
    Returns list of tuples: (userId, seed_ml_movieId, heldout_ml_movieId).
    """
    user_pos = _build_user_positives(ratings_df)
    seeds = []
    for uid, items in user_pos.items():
        if len(items) >= max(min_pos, 2):
            items_list = list(items)
            random.shuffle(items_list)
            seed = items_list[0]
            held = items_list[1]
            seeds.append((uid, seed, held))
    return seeds

@st.cache_data
def _ml_to_tmdb_map(links_df: pd.DataFrame):
    to_tmdb = {}
    to_ml = {}
    for _, r in links_df.iterrows():
        ml_id = _safe_int(r.movieId)
        tmdb_id = _safe_int(r.tmdbId)
        if ml_id and tmdb_id:
            to_tmdb[ml_id] = tmdb_id
            to_ml[tmdb_id] = ml_id
    return to_tmdb, to_ml

def _precision_recall_f1(recommended: list[int], relevant: set[int]):
    if not recommended:
        return 0.0, 0.0, 0.0
    rec_set = set(recommended)
    hit = len(rec_set & relevant)
    precision = hit / len(recommended)
    recall = hit / len(relevant) if relevant else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

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


def _recommend_hybrid_seed(seed_ml_movie_id: int, ratings: pd.DataFrame, movies: pd.DataFrame, links: pd.DataFrame,
                           cf_weight=0.6, content_weight=0.4, top_n=TOP_K):
    """
    Combine CF + Content with normalized ranks for a seed ML movie.
    """
    # content list (ML ids)
    content_list = _recommend_content_seed(seed_ml_movie_id, top_n=top_n*5)
    # cf list (ML ids) via Pearson
    cf_list = _recommend_cf_seed(seed_ml_movie_id, ratings, min_overlap=10, top_n=top_n*5)

    if not content_list and not cf_list:
        return []

    # Rank normalization: higher rank => better score (invert position)
    def rank_scores(lst):
        scores = {}
        n = len(lst)
        for pos, mid in enumerate(lst):
            scores[mid] = (n - pos) / n  # 1.0 (best) -> small
        return scores

    c_scores = rank_scores(content_list)
    f_scores = rank_scores(cf_list)

    # combine
    all_ids = set(content_list) | set(cf_list)
    merged = []
    for mid in all_ids:
        c = c_scores.get(mid, 0.0)
        f = f_scores.get(mid, 0.0)
        score = cf_weight * f + content_weight * c
        merged.append((mid, score))
    merged.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in merged[:top_n]]

@st.cache_data
def evaluate_precision_recall_f1_all(ratings, movies, links, sample_users=150, top_k=TOP_K, seed_min_pos=3, random_state=42):
    """
    For a subset of users with >= seed_min_pos positives:
      - pick one positive as seed, one as held-out relevant target (+ keep all positives as 'relevant set')
      - generate top-K via each method from the seed movie
      - compute Precision/Recall/F1 vs user's positive set (excluding the seed)
    Returns dict of averaged metrics.
    """
    random.seed(random_state)
    seeds = _build_movie_seed_map(ratings, min_pos=seed_min_pos)
    if len(seeds) > sample_users:
        random.shuffle(seeds)
        seeds = seeds[:sample_users]

    user_pos = _build_user_positives(ratings)
    results = {
        "content": {"p": [], "r": [], "f1": []},
        "cf": {"p": [], "r": [], "f1": []},
        "hybrid": {"p": [], "r": [], "f1": []},
    }

    for uid, seed_mid, held_mid in seeds:
        relevant = set(user_pos[uid]) - {seed_mid}  # all other positives are relevant
        if not relevant:
            continue

        # Content
        rec_c = _recommend_content_seed(seed_mid, top_n=top_k)
        p, r, f1 = _precision_recall_f1(rec_c, relevant)
        results["content"]["p"].append(p)
        results["content"]["r"].append(r)
        results["content"]["f1"].append(f1)

        # CF
        rec_f = _recommend_cf_seed(seed_mid, ratings, min_overlap=15, top_n=top_k)
        p, r, f1 = _precision_recall_f1(rec_f, relevant)
        results["cf"]["p"].append(p)
        results["cf"]["r"].append(r)
        results["cf"]["f1"].append(f1)

        # Hybrid
        rec_h = _recommend_hybrid_seed(seed_mid, ratings, movies, links, cf_weight=0.6, content_weight=0.4, top_n=top_k)
        p, r, f1 = _precision_recall_f1(rec_h, relevant)
        results["hybrid"]["p"].append(p)
        results["hybrid"]["r"].append(r)
        results["hybrid"]["f1"].append(f1)

    # average
    def avg(lst): 
        return float(np.mean(lst)) if lst else 0.0

    summary = {
        "Content-Based": {
            "Precision@K": avg(results["content"]["p"]),
            "Recall@K": avg(results["content"]["r"]),
            "F1@K": avg(results["content"]["f1"])
        },
        "Collaborative": {
            "Precision@K": avg(results["cf"]["p"]),
            "Recall@K": avg(results["cf"]["r"]),
            "F1@K": avg(results["cf"]["f1"])
        },
        "Hybrid": {
            "Precision@K": avg(results["hybrid"]["p"]),
            "Recall@K": avg(results["hybrid"]["r"]),
            "F1@K": avg(results["hybrid"]["f1"])
        }
    }
    return summary



# TMDB API Configuration
API_KEY = "6272a18b79461f7987d2f1047b3d1198"
API_BASE_URL = "https://api.themoviedb.org/3"

st.set_page_config(page_title="Movie Details", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

# Persist graphs + rendered HTML across reruns
if "figs" not in st.session_state:
    st.session_state["figs"] = {}

def _persist_and_show(slot, key, fig):
    st.session_state["figs"][key] = fig
    slot.pyplot(fig)

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

@st.cache_data
def build_content_features(movies: pd.DataFrame, links: pd.DataFrame):
    """
    Build content-based features from MovieLens movies dataset.
    Creates TF-IDF vectors from genres for similarity computation.
    """
    # Merge movies with links to get TMDB IDs
    content_df = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="inner")
    content_df = content_df[content_df["tmdbId"].notna()].copy()
    content_df["tmdbId"] = content_df["tmdbId"].astype(int)
    
    # Process genres - convert pipe-separated genres to list
    content_df["genres_list"] = content_df["genres"].str.split("|")
    
    # Create genre features using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    genre_features = mlb.fit_transform(content_df["genres_list"])
    genre_df = pd.DataFrame(genre_features, columns=mlb.classes_, index=content_df.index)
    
    # Combine with movie metadata
    features_df = pd.concat([
        content_df[["movieId", "tmdbId", "title", "genres"]],
        genre_df
    ], axis=1)
    
    return features_df, genre_df

def content_based_recommendations(target_movie_id, movies_df, tfidf_matrix, top_n=10):
    """
    Generate content-based recommendations for a given movie ID.
    
    Args:
        target_movie_id: MovieLens movie ID of the target movie
        movies_df: DataFrame containing movie information
        tfidf_matrix: Precomputed TF-IDF matrix for all movies
        top_n: Number of recommendations to return
    
    Returns:
        DataFrame with recommended movies and their similarity scores
    """
    try:
        # Find index of the target movie
        target_idx = movies_df[movies_df['movieId'] == target_movie_id].index
        if len(target_idx) == 0:
            return pd.DataFrame()
        
        target_idx = target_idx[0]
        
        # Compute similarity scores
        similarity_scores = list(enumerate(cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix).flatten()))
        
        # Sort by similarity (descending)
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations (excluding the target movie itself)
        recommendations = []
        for i, (idx, score) in enumerate(similarity_scores[1:top_n+1]):  # Skip the first one (itself)
            movie_data = movies_df.iloc[idx]
            # Handle missing genre_list
            genres = movie_data.get('genres', '')
            genre_list = genres.split('|') if isinstance(genres, str) else []

            recommendations.append({
                'movieId': movie_data['movieId'],
                'title': movie_data.get('clean_title', movie_data.get('title', '')),
                'year': movie_data.get('year', ''),
                'genres': movie_data.get('genres', ''),
                'genre_list': movie_data.get('genre_list', []),
                'avg_rating': movie_data.get('avg_rating', 0),
                'rating_count': movie_data.get('rating_count', 0),
                'similarity': score
            })
        
        return pd.DataFrame(recommendations)
    
    except Exception as e:
        print(f"Error in content_based_recommendations: {e}")
        return pd.DataFrame()

@st.cache_data
def hybrid_recommendations(
    target_tmdb_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    links: pd.DataFrame,
    cf_weight: float = 0.6,
    content_weight: float = 0.4,
    top_n: int = 12
):
    """
    Optimized hybrid recommendation system combining collaborative and content-based filtering.
    
    Args:
        target_tmdb_id: Target movie TMDB ID
        cf_weight: Weight for collaborative filtering (default 0.6)
        content_weight: Weight for content-based filtering (default 0.4)
        top_n: Number of recommendations to return
    """
    try:
        # Get MovieLens ID from TMDB ID - optimized lookup
        target_row = links.loc[links["tmdbId"] == int(target_tmdb_id)]
        if target_row.empty:
            return pd.DataFrame()
        
        target_ml_id = int(target_row.iloc[0]["movieId"])
        
        # Precompute content-based recommendations first (usually faster)
        content_recs = content_based_recommendations(
            target_movie_id=target_ml_id,
            movies_df=movies,
            tfidf_matrix=rec_tfidf_matrix,
            top_n=top_n * 2  # Get fewer candidates initially
        )
        
        # Only compute CF recommendations if content-based has results
        if content_recs.empty:
            # Fall back to CF if content-based fails
            cf_recs = item_item_recommendations(
                target_tmdb_id=target_tmdb_id,
                ratings=ratings,
                movies=movies,
                links=links,
                min_overlap=5,  # Even lower threshold for fallback
                top_n=top_n
            )
            return cf_recs.head(top_n)
        
        # Get CF recommendations in parallel with content processing
        cf_recs = item_item_recommendations(
            target_tmdb_id=target_tmdb_id,
            ratings=ratings,
            movies=movies,
            links=links,
            min_overlap=8,  # Slightly lower threshold for hybrid
            top_n=top_n * 2  # Get fewer candidates
        )
        
        # If CF fails, return content-based results
        if cf_recs.empty:
            return content_recs.head(top_n)
        
        # Precompute normalization parameters
        cf_similarities = cf_recs['similarity'].values
        content_similarities = content_recs['similarity'].values
        
        cf_max, cf_min = np.max(cf_similarities), np.min(cf_similarities)
        content_max, content_min = np.max(content_similarities), np.min(content_similarities)
        
        # Normalize CF scores
        if cf_max > cf_min:
            cf_recs['cf_score_norm'] = (cf_similarities - cf_min) / (cf_max - cf_min)
        else:
            cf_recs['cf_score_norm'] = 1.0 if cf_max > 0 else 0.0
        
        # Normalize content scores
        if content_max > content_min:
            content_recs['content_score_norm'] = (content_similarities - content_min) / (content_max - content_min)
        else:
            content_recs['content_score_norm'] = 1.0 if content_max > 0 else 0.0
        
        # Create lookup dictionaries for faster access
        content_dict = {row['movieId']: row for _, row in content_recs.iterrows()}
        cf_dict = {row['movieId']: row for _, row in cf_recs.iterrows()}
        
        # Get all unique movie IDs from both methods
        all_movie_ids = set(content_dict.keys()) | set(cf_dict.keys())
        
        # Calculate hybrid scores
        hybrid_scores = []
        for movie_id in all_movie_ids:
            cf_data = cf_dict.get(movie_id, {})
            content_data = content_dict.get(movie_id, {})
            
            cf_score = cf_data.get('cf_score_norm', 0.0)
            content_score = content_data.get('content_score_norm', 0.0)
            
            # Calculate hybrid score
            hybrid_score = (cf_weight * cf_score) + (content_weight * content_score)
            
            # Use title from whichever source has it
            title = cf_data.get('title') or content_data.get('title', 'Unknown')
            
            # Get TMDB ID if available
            tmdb_id = 0
            if movie_id in cf_dict and 'tmdbId' in cf_dict[movie_id]:
                tmdb_id = cf_dict[movie_id]['tmdbId']
            else:
                # Look up TMDB ID from links
                link_row = links.loc[links['movieId'] == movie_id]
                if not link_row.empty:
                    tmdb_id = int(link_row.iloc[0]['tmdbId'])
            
            hybrid_scores.append({
                'movieId': movie_id,
                'title': title,
                'tmdbId': tmdb_id,
                'cf_score': cf_score,
                'content_score': content_score,
                'hybrid_score': hybrid_score,
                'avg_rating': cf_data.get('avg_rating', content_data.get('avg_rating', np.nan)),
                'rating_cnt': cf_data.get('rating_cnt', content_data.get('rating_count', 0))
            })
        
        # Convert to DataFrame and sort by hybrid score
        hybrid_df = pd.DataFrame(hybrid_scores)
        hybrid_df = hybrid_df.sort_values('hybrid_score', ascending=False).head(top_n)
        
        if not hybrid_df.empty and 'hybrid_score' not in hybrid_df.columns:
            # Calculate hybrid score if missing
            hybrid_df['hybrid_score'] = (cf_weight * hybrid_df.get('cf_score', 0.0)) + \
                                    (content_weight * hybrid_df.get('content_score', 0.0))
        
        return hybrid_df
    
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
            movies=movies,
            links=links,
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


            

# Updated render_cf_section to fix the captions
def render_cf_section_horizontal(target_tmdb_id: int):
    """Render CF recommendations using the horizontal 'rec-card' design; reuse cached HTML on reruns."""
    load_content_based_css()
    st.markdown("---")
    st.subheader("🔁 Because You Liked This")
    st.caption("Item–Item Collaborative Filtering (Pearson). Requires at least 5 users who rated both movies.")

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
            min_overlap=5,
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
        recommendations = content_based_recommendations(detail_id_int, movies_df, tfidf_matrix, top_n=10)
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

@st.cache_resource
def build_user_item_matrix(ratings: pd.DataFrame):
    """Pivot to user–item matrix (rows=userId, cols=movieId, values=rating)."""
    mat = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    return mat

@st.cache_data
def get_movielens_rating_for_tmdb(tmdb_id: int, base_dir: str = "dataset/ml-latest-small"):
    ratings = pd.read_csv(f"{base_dir}/ratings.csv")
    links   = pd.read_csv(f"{base_dir}/links.csv")
    links = links[links["tmdbId"].notna() & (links["tmdbId"] > 0)]
    links["tmdbId"] = links["tmdbId"].astype(int)

    row = links.loc[links["tmdbId"] == int(tmdb_id)]
    if row.empty:
        return None, 0  # no ML mapping

    ml_id = int(row.iloc[0]["movieId"])
    sub = ratings.loc[ratings["movieId"] == ml_id, "rating"]
    if sub.empty:
        return None, 0

    return float(sub.mean()), int(sub.count())


# ---------- Item–Item CF (memory-based) ----------
@st.cache_resource
def build_cached_correlation_matrix(ratings: pd.DataFrame, min_overlap: int = 5):
    """
    Precompute and cache the correlation matrix for faster recommendations.
    Only computes correlations for movies with sufficient ratings.
    """
    user_item = build_user_item_matrix(ratings)
    
    # Filter movies with at least min_overlap ratings
    rating_counts = user_item.notna().sum()
    valid_movies = rating_counts[rating_counts >= min_overlap].index
    
    if len(valid_movies) == 0:
        return pd.DataFrame()
    
    # Only compute correlations for movies with sufficient ratings
    user_item_filtered = user_item[valid_movies]
    
    # Compute correlation matrix
    corr_mat = user_item_filtered.corr(method="pearson", min_periods=min_overlap)
    
    return corr_mat, valid_movies

@st.cache_data
def item_item_recommendations(
    target_tmdb_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    links: pd.DataFrame,
    min_overlap: int = 20,
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

    
def render_cf_section(target_tmdb_id: int):
    ratings, movies, links = load_ml_small()
    recs = item_item_recommendations(
        target_tmdb_id=target_tmdb_id,
        ratings=ratings,
        movies=movies,
        links=links,
        min_overlap=20,   # rubric: need enough shared users
        top_n=12
    )

    st.markdown("### 🔁 Because you liked this")
    st.caption("Content-based filtering.")


    st.caption("Item–Item Collaborative Filtering (Pearson). Similarity requires at least 20 users who rated both movies.")

    if recs.empty:
        st.info("Not enough overlapping ratings to compute reliable neighbors for this title.")
        return

    # Show a responsive grid (4 per row)
    cols = st.columns(4, gap="large")
    for i, (_, row) in enumerate(recs.iterrows()):
        poster, _, _ = fetch_tmdb_brief(int(row.tmdbId))  # only need poster
        with cols[i % 4]:
            link = f"/Movie_Details?movie_id={int(row.tmdbId)}&source=movielens"
            ml_avg = row.get("avg_rating", np.nan)
            ml_cnt = row.get("rating_cnt", 0)
            ml_text = f"{ml_avg:.2f}/5 • {int(ml_cnt)} ratings" if pd.notna(ml_avg) else "—"

            st.markdown(
                f"""
                <a href="{link}" target="_self" style="text-decoration:none;color:inherit">
                <div style="background:#1a1a25;border:1px solid #2a2a35;border-radius:12px;
                            padding:12px;margin:12px 12px;display:flex;flex-direction:column;
                            justify-content:space-between;height:100%">
                    <img src="{poster or 'https://via.placeholder.com/300x450?text=No+Poster'}"
                        style="width:100%;border-radius:8px;margin-bottom:10px" />
                    <div style="font-weight:700;line-height:1.2;margin-bottom:6px;
                                height:40px;overflow:hidden;display:-webkit-box;
                                -webkit-line-clamp:2;-webkit-box-orient:vertical;">
                        {row.title}
                    </div>
                    <div style="font-size:.9rem;color:#b0b0b0">
                        ⭐ {ml_text}
                    </div>
                </div>
                </a>
                """,
                unsafe_allow_html=True
            )

    st.caption("Hybrid filtering.")


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

current_ids = {"ml": ml_movie_id, "tmdb": tmdb_id}
last_ids = st.session_state["last_ids"]

if current_ids != last_ids:
    # Clear only the entries for the current movie ids (if present)
    for k in ("content", "cf", "hybrid"):
        if ml_movie_id is not None:
            st.session_state["html_cache"][k].pop(ml_movie_id, None)
        if tmdb_id is not None:
            st.session_state["html_cache"][k].pop(tmdb_id, None)
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
        content_eval_slot = st.empty()

        # Content-Based evaluation
        if st.button("Evaluate Content-Based Algorithm", key="eval_content"):
            ratings_small, movies_small, links_small = load_ml_small()
            content_metrics, content_similarities = evaluate_single_algorithm("content", ratings_small, movies_small, links_small)
            metrics_dict = {"Content-Based": content_metrics}
            fig = create_evaluation_plots(metrics_dict, "Content-Based", content_similarities)
            st.session_state["figs"]["content_eval_fig"] = fig

        # show graph always below the button
        if st.session_state["figs"].get("content_eval_fig") is not None:
            st.pyplot(st.session_state["figs"]["content_eval_fig"])

                

    # In the collaborative filtering section, add the evaluation button and display
    if tmdb_id is not None:
        render_cf_section_horizontal(int(tmdb_id))
        
        # Add evaluation button for collaborative filtering
        cf_eval_slot = st.empty()

        # ===== Collaborative Filtering evaluation =====
        if st.button("Evaluate Collaborative Filtering Algorithm", key="eval_cf"):
            ratings_small, movies_small, links_small = load_ml_small()
            cf_metrics, cf_similarities = evaluate_single_algorithm("cf", ratings_small, movies_small, links_small)
            metrics_dict = {"Collaborative": cf_metrics}
            fig = create_evaluation_plots(metrics_dict, "Collaborative", cf_similarities)
            st.session_state["figs"]["cf_eval_fig"] = fig

        # show graph below the button
        if st.session_state["figs"].get("cf_eval_fig") is not None:
            st.pyplot(st.session_state["figs"]["cf_eval_fig"])

                

    # In the hybrid recommendations section, add the evaluation button and display
    if tmdb_id is not None:
        render_hybrid_section_horizontal(int(tmdb_id))
        
        # Add evaluation button for hybrid
        hybrid_eval_slot = st.empty()

        # ===== Hybrid evaluation =====
        if st.button("Evaluate Hybrid Algorithm", key="eval_hybrid"):
            ratings_small, movies_small, links_small = load_ml_small()
            hybrid_metrics, hybrid_scores = evaluate_single_algorithm("hybrid", ratings_small, movies_small, links_small)
            metrics_dict = {"Hybrid": hybrid_metrics}
            fig = create_hybrid_evaluation_plots(metrics_dict, hybrid_scores)
            st.session_state["figs"]["hybrid_eval_fig"] = fig

        # show graph below the button
        if st.session_state["figs"].get("hybrid_eval_fig") is not None:
            st.pyplot(st.session_state["figs"]["hybrid_eval_fig"])

    
except Exception as e:
    st.warning(f"Recommendations unavailable: {e}")

st.markdown("---")
st.header("📊 Algorithm Comparison")
st.caption("Compare performance of all three recommendation algorithms")

compare_slot = st.empty()

# ===== Algorithm Comparison =====
if st.button("🚀 Run Algorithm Comparison", key="compare_algorithms"):
    with st.spinner("Comparing algorithms..."):
        ratings_small, movies_small, links_small = load_ml_small()
        content_metrics, _ = evaluate_single_algorithm("content", ratings_small, movies_small, links_small)
        cf_metrics, _ = evaluate_single_algorithm("cf", ratings_small, movies_small, links_small)
        hybrid_metrics, _ = evaluate_single_algorithm("hybrid", ratings_small, movies_small, links_small)
        
        # Store results in session state
        st.session_state["compare_metrics"] = {
            "Content-Based": content_metrics, 
            "Collaborative": cf_metrics, 
            "Hybrid": hybrid_metrics
        }

# Display comparison results if they exist in session state
if "compare_metrics" in st.session_state:
    metrics_dict = st.session_state["compare_metrics"]
    
    # Display table
    metrics_df = pd.DataFrame(metrics_dict).T
    st.dataframe(metrics_df.style.format("{:.3f}").highlight_max(axis=0))

    # Display figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    algorithms = list(metrics_dict.keys())

    precision_values = [metrics_dict[a]['Precision@K'] for a in algorithms]
    m = max(precision_values) if precision_values else 1.0; y = m * 0.15
    ax1.bar(algorithms, precision_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('Precision@K Comparison'); ax1.set_ylabel('Precision@K')
    ax1.set_ylim(0, m + y); ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(precision_values): ax1.text(i, v + (y * 0.1), f'{v:.3f}', ha='center', va='bottom', fontsize=10)

    recall_values = [metrics_dict[a]['Recall@K'] for a in algorithms]
    m = max(recall_values) if recall_values else 1.0; y = m * 0.15
    ax2.bar(algorithms, recall_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('Recall@K Comparison'); ax2.set_ylabel('Recall@K')
    ax2.set_ylim(0, m + y); ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(recall_values): ax2.text(i, v + (y * 0.1), f'{v:.3f}', ha='center', va='bottom', fontsize=10)

    f1_values = [metrics_dict[a]['F1@K'] for a in algorithms]
    m = max(f1_values) if f1_values else 1.0; y = m * 0.15
    ax3.bar(algorithms, f1_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax3.set_title('F1@K Comparison'); ax3.set_ylabel('F1@K')
    ax3.set_ylim(0, m + y); ax3.tick_params(axis='x', rotation=45)
    for i, v in enumerate(f1_values): ax3.text(i, v + (y * 0.1), f'{v:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)


# show graph below the button
if st.session_state["figs"].get("alg_compare_fig") is not None:
    st.pyplot(st.session_state["figs"]["alg_compare_fig"])


# Add RMSE/MAE Evaluation section
st.markdown("---")
st.header("📈 Rating Prediction Evaluation")
st.caption("Evaluate rating prediction accuracy using RMSE and MAE")

# ===== RMSE/MAE evaluation =====
if st.button("📊 Run RMSE/MAE Evaluation", key="eval_rmse"):
    with st.spinner("Evaluating rating prediction..."):
        ratings_small, movies_small, links_small = load_ml_small()
        rmse_res = evaluate_rmse_baseline(ratings_small)
        
        # Store results in session state
        st.session_state["rmse_results"] = rmse_res

# Display RMSE/MAE results if they exist in session state
if "rmse_results" in st.session_state:
    rmse_res = st.session_state["rmse_results"]
    
    # Display table
    rmse_df = pd.DataFrame({"RMSE": [rmse_res.get("RMSE", 0.0)], "MAE": [rmse_res.get("MAE", 0.0)]})
    st.dataframe(rmse_df.style.format("{:.4f}"))

    # Display figure (narrower width)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    metrics = ['RMSE', 'MAE']
    values = [rmse_res.get("RMSE", 0.0), rmse_res.get("MAE", 0.0)]
    m = max(values) if values else 1.0
    y = m * 0.15
    ax.bar(metrics, values, color=['#FF6B6B', '#4ECDC4'])
    ax.set_title('Rating Prediction Error')
    ax.set_ylabel('Error')
    ax.set_ylim(0, m + y)
    for i, v in enumerate(values):
        ax.text(i, v + (y * 0.1), f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)


# show graph below the button
if st.session_state["figs"].get("rmse_fig") is not None:
    st.pyplot(st.session_state["figs"]["rmse_fig"], use_container_width=False)