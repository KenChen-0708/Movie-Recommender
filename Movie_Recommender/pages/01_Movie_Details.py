# pages/01_Movie_Details.py
import streamlit as st
from datetime import datetime
import pandas as pd
import requests

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

    rating = data.get("vote_average")
    if rating:
        st.markdown(
            f'<div class="rating-display">⭐ {rating}/10</div>', unsafe_allow_html=True
        )

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
