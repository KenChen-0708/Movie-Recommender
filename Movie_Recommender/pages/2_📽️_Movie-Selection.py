import pandas as pd
import streamlit as st
from datetime import datetime
import ast

# Load the CSV data
@st.cache_data
def load_movie_data():
    try:
        df = pd.read_csv("dataset/TMDB_movie_dataset_v11.csv")
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please ensure 'dataset/TMDB_movie_dataset_v11.csv' exists.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return pd.DataFrame()

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

# Page config with modern dark theme
st.set_page_config(
    page_title="CineSearch Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
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
    
    .stTextInput>div>div>input {
        background-color: var(--card-bg);
        color: var(--text);
        border: 1px solid #333 !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
    }
    
    .movie-card {
        display: flex;
        gap: 20px;
        padding: 20px;
        border-radius: 12px;
        background-color: var(--card-bg);
        margin-bottom: 24px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        border: 1px solid #2a2a35;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        border-color: var(--primary);
    }
    
    .movie-poster {
        border-radius: 10px;
        width: 180px;
        height: 270px;
        object-fit: cover;
        transition: transform 0.3s ease;
    }
    
    .movie-poster:hover {
        transform: scale(1.03);
    }
    
    .movie-info {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    
    .movie-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
        color: var(--primary);
        transition: color 0.2s ease;
    }
    
    .movie-title:hover {
        color: #00a8ff;
    }
    
    .movie-meta {
        display: flex;
        gap: 15px;
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    .movie-rating {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(255, 193, 7, 0.15);
        color: #ffc107;
        padding: 4px 10px;
        border-radius: 50px;
        font-weight: 600;
    }
    
    .movie-overview {
        color: var(--text-secondary);
        line-height: 1.6;
        margin-top: 10px;
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
    }
    
    .back-button:hover {
        background: #252535;
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
    </style>
""", unsafe_allow_html=True)

# Load data
df = load_movie_data()

if df.empty:
    st.stop()

# Get movie_id from query params
query_params = st.query_params
detail_id = query_params.get("movie_id", [None])[0]

# ---------------- Detail Page ---------------- #
if detail_id:
    try:
        detail_id = int(detail_id)
        movie_detail = df[df['id'] == detail_id]
        
        if not movie_detail.empty:
            movie = movie_detail.iloc[0]
            
            # Back button
            st.markdown(
                '<a href="?" class="back-button" target="_self">⬅ Back to Search</a>',
                unsafe_allow_html=True
            )
            
            # Main content
            col1, col2 = st.columns([1, 2])
            
            with col1:
                poster_path = movie["poster_path"] if "poster_path" in movie else None
                if pd.notna(poster_path) and poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                    st.image(poster_url, use_column_width=True, output_format="auto", caption=movie["title"] if "title" in movie else "No Title")
                else:
                    st.image("https://via.placeholder.com/500x750?text=No+Poster", use_column_width=True, caption=movie["title"] if "title" in movie else "No Title")
            
            with col2:
                st.title(movie["title"] if "title" in movie else "No Title")
                
                # Tagline
                tagline = movie["tagline"] if "tagline" in movie else None
                if pd.notna(tagline) and tagline and str(tagline).strip():
                    st.markdown(f'<p class="tagline">"{tagline}"</p>', unsafe_allow_html=True)
                
                # Basic info
                release_date = movie["release_date"] if "release_date" in movie else None
                if pd.notna(release_date) and release_date:
                    try:
                        date_obj = datetime.strptime(str(release_date), "%Y-%m-%d")
                        formatted_date = date_obj.strftime("%B %d, %Y")
                        runtime = movie["runtime"] if "runtime" in movie else "?"
                        runtime_str = f"{runtime} min" if pd.notna(runtime) and runtime != "?" else "? min"
                        st.caption(f"📅 {formatted_date} • ⏱️ {runtime_str}")
                    except:
                        st.caption(f"📅 {release_date}")
                
                # Rating
                vote_avg = movie["vote_average"] if "vote_average" in movie else None
                if pd.notna(vote_avg) and vote_avg:
                    st.markdown(f'<div class="movie-rating">⭐ {vote_avg}/10</div>', unsafe_allow_html=True)
                
                # Genres
                genres_data = movie["genres"] if "genres" in movie else None
                if pd.notna(genres_data) and genres_data:
                    genres = safe_parse_json(genres_data)
                    if genres:
                        st.markdown("<div style='margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)
                        for genre in genres:
                            genre_name = genre.get("name", "") if isinstance(genre, dict) else str(genre)
                            if genre_name:
                                st.markdown(f'<span class="genre-badge">{genre_name}</span>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Overview
                st.markdown('<div class="detail-section">', unsafe_allow_html=True)
                st.subheader("Overview")
                overview = movie["overview"] if "overview" in movie else "No description available."
                st.write(overview if pd.notna(overview) and str(overview).strip() else "No description available.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Cast
                cast_data = movie["cast"] if "cast" in movie else None
                if pd.notna(cast_data) and cast_data:
                    cast = safe_parse_json(cast_data)
                    if cast:
                        st.markdown('<div class="detail-section">', unsafe_allow_html=True)
                        st.subheader("Top Cast")
                        cast_names = extract_names(cast[:6])  # Show top 6 cast members
                        if cast_names:
                            st.write(", ".join(cast_names))
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Additional Info
                st.markdown('<div class="detail-section">', unsafe_allow_html=True)
                st.subheader("Additional Information")
                
                # Budget and Revenue
                budget = movie["budget"] if "budget" in movie else None
                revenue = movie["revenue"] if "revenue" in movie else None
                
                if pd.notna(budget) and budget > 0:
                    st.write(f"**Budget:** ${budget:,}")
                if pd.notna(revenue) and revenue > 0:
                    st.write(f"**Revenue:** ${revenue:,}")
                
                # Original Language
                original_language = movie["original_language"] if "original_language" in movie else None
                if pd.notna(original_language) and original_language:
                    st.write(f"**Original Language:** {original_language.upper()}")
                
                # Status
                status = movie["status"] if "status" in movie else None
                if pd.notna(status) and status:
                    st.write(f"**Status:** {status}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Movie not found.")
    except ValueError:
        st.error("Invalid movie ID.")
    except Exception as e:
        st.error(f"Error loading movie details: {str(e)}")

# ---------------- Search Page ---------------- #
else:
    # Header with search
    st.write('# Search your movie!')
    st.markdown("Enter here!")
    
    search_query = st.text_input(
        "",
        placeholder="Search for movies...",
        key="search_input",
        label_visibility="collapsed"
    )
    
    if search_query:
        with st.spinner('🔍 Searching movies...'):
            # Search in title, overview, and other relevant fields
            search_mask = (
                df['title'].str.contains(search_query, case=False, na=False) |
                df['overview'].str.contains(search_query, case=False, na=False) |
                df['original_title'].str.contains(search_query, case=False, na=False)
            )
            
            results = df[search_mask].head(20)  # Limit to top 20 results
            
            if not results.empty:
                st.markdown(f"### Found {len(results)} results")
                for _, result in results.iterrows():
                    movie_id = result.get('id')
                    title = result.get('title', 'No Title')
                    release_date = result.get('release_date', 'N/A')
                    overview = result.get('overview', 'No overview available.')
                    vote_avg = result.get('vote_average', 0)
                    poster_path = result.get('poster_path')
                    
                    # Handle overview truncation
                    if pd.notna(overview) and len(str(overview)) > 200:
                        overview = str(overview)[:200] + "..."
                    elif pd.isna(overview):
                        overview = "No overview available."
                    
                    # Handle poster URL
                    if pd.notna(poster_path) and poster_path:
                        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                    else:
                        poster_url = "https://via.placeholder.com/500x750?text=No+Poster"
                    
                    # Handle vote average
                    if pd.isna(vote_avg):
                        vote_avg = 0
                    
                    st.markdown(
                        f"""
                        <a href="?movie_id={movie_id}" style="text-decoration: none; color: inherit;" target="_self">
                            <div class="movie-card">
                                <img src="{poster_url}" alt="{title}" class="movie-poster" />
                                <div class="movie-info">
                                    <h3 class="movie-title">{title}</h3>
                                    <div class="movie-meta">
                                        <span>{release_date}</span>
                                        <span class="movie-rating">⭐ {vote_avg}</span>
                                    </div>
                                    <p class="movie-overview">{overview}</p>
                                </div>
                            </div>
                        </a>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No movies found. Try a different search term.")
    else:
        st.info("Enter a movie title to begin your search")