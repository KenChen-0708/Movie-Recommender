# pages/01_Movie_Details.py
import streamlit as st
from datetime import datetime
import pandas as pd
import requests
import numpy as np

API_KEY = "6272a18b79461f7987d2f1047b3d1198"
API_BASE_URL = "https://api.themoviedb.org/3"

st.set_page_config(page_title="Movie Details", page_icon="🎬", layout="wide")


# ----- helpers (copy/paste from your main file if you keep them there) -----
@st.cache_data
def fetch_movie_details_by_tmdb_id(tmdb_id: int | str):
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


def load_custom_css():
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
        
    </style>
    """,
        unsafe_allow_html=True,
    )

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
    Return top-N movies most similar to the target movie using Pearson corr
    on co-rating users (minimum co-rater count = min_overlap).
    """
    # TMDB -> MovieLens mapping
    row = links.loc[links["tmdbId"] == int(target_tmdb_id)]
    if row.empty:
        return pd.DataFrame()

    target_ml_id = int(row.iloc[0]["movieId"])

    # user–item matrix
    user_item = build_user_item_matrix(ratings)

    # must exist as a column
    if target_ml_id not in user_item.columns:
        return pd.DataFrame()

    # ---- Pearson item–item sim with min co-raters (works in older pandas) ----
    corr_mat = user_item.corr(method="pearson", min_periods=min_overlap)

    if target_ml_id not in corr_mat.columns:
        return pd.DataFrame()

    sim = corr_mat[target_ml_id].drop(labels=[target_ml_id], errors="ignore")
    sim = sim.dropna().sort_values(ascending=False).head(200)

    # ---- overlap counts (users who rated both movies) ----
    mask_target = user_item[target_ml_id].notna()
    overlap = (mask_target & user_item.notna()).sum()
    overlap = overlap.reindex(sim.index)

    # stability stats
    stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_cnt=("rating", "count")
    )

    recs = (
        pd.DataFrame({"similarity": sim})
        .join(overlap.rename("overlap"))
        .join(stats, how="left")
        .sort_values(["similarity", "overlap", "rating_cnt"], ascending=False)
    )

    # titles + tmdb
    recs = (
        recs.reset_index()
        .rename(columns={"index": "movieId"})
        .merge(movies[["movieId", "title"]], on="movieId", how="left")
        .merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    )

    recs = recs[recs["tmdbId"].notna()].head(top_n).copy()
    recs["tmdbId"] = recs["tmdbId"].astype(int)
    return recs

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



# ---------------------------------------------------------------------------

load_custom_css()

# Read query params
qp = st.query_params
movie_id = qp.get("movie_id")
source = qp.get("source", "movielens")
if isinstance(movie_id, list):
    movie_id = movie_id[0]
if isinstance(source, list):
    source = source[0]

if not movie_id:
    st.error("No movie_id provided.")
    st.stop()

# Always fetch from TMDB for details
data = fetch_movie_details_by_tmdb_id(movie_id)
if not data:
    st.error("Could not load movie details.")
    st.stop()

# Render
col1, col2 = st.columns([1, 2])
with col1:
    poster = data.get("poster_path")
    if poster:
        st.image(f"https://image.tmdb.org/t/p/w400{poster}", use_container_width=True)
    else:
        st.image(
            "https://via.placeholder.com/400x600?text=No+Poster",
            use_container_width=True,
        )

with col2:
    st.title(data.get("title", "No Title"))
    
    ml_avg, ml_cnt = get_movielens_rating_for_tmdb(int(movie_id))
    if ml_avg is not None and ml_cnt > 0:
        st.markdown(f'<div class="rating-display">⭐ MovieLens: {ml_avg:.2f}/5 · {ml_cnt} ratings</div>', unsafe_allow_html=True)

    # (Optional) Keep TMDB as a secondary, lighter caption:
    tmdb_rating = data.get("vote_average")
    if tmdb_rating:
        st.caption(f"TMDB: {tmdb_rating:.1f}/10")

    tg = (data.get("tagline") or "").strip()
    if tg:
        st.markdown(f'<div class="tagline">"{tg}"</div>', unsafe_allow_html=True)

    # meta
    release = data.get("release_date") or ""
    try:
        release_fmt = (
            datetime.strptime(release, "%Y-%m-%d").strftime("%B %d, %Y")
            if release
            else "N/A"
        )
    except:
        release_fmt = release or "N/A"
    runtime = data.get("runtime") or 0
    st.write(f"**📅 Release Date:** {release_fmt}")
    st.write(f"**⏱️ Runtime:** {runtime} min" if runtime else "**⏱️ Runtime:** N/A")

    genres = [g.get("name", "") for g in (data.get("genres") or [])]
    if genres:
        st.markdown("**🎞️ Genres:**")
        for g in genres:
            st.markdown(f'<span class="genre-badge">{g}</span>', unsafe_allow_html=True)

    st.subheader("📝 Overview")
    st.write(data.get("overview") or "No description available.")

    cast = data.get("credits", {}).get("cast", [])[:8]
    if cast:
        st.subheader("👤 Top Cast")
        st.write(", ".join([c.get("name", "") for c in cast]))

    st.subheader("📌 Additional Information")
    budget = data.get("budget") or 0
    revenue = data.get("revenue") or 0
    lang = (data.get("original_language") or "").upper()
    status = data.get("status") or ""
    if budget > 0:
        st.write(f"**Budget:** ${int(budget):,}")
    if revenue > 0:
        st.write(f"**Revenue:** ${int(revenue):,}")
    if lang:
        st.write(f"**Original Language:** {lang}")
    if status:
        st.write(f"**Status:** {status}")
    hp = (data.get("homepage") or "").strip()
    if hp:
        st.markdown(f"🔗 [Official Website]({hp})")
        
# --- CF recommendations under the details ---
try:
    render_cf_section(int(movie_id))
except Exception as e:
    st.warning(f"CF recommendations unavailable: {e}")  