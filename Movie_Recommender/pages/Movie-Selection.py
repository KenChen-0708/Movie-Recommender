import pandas as pd
import streamlit as st
from datetime import datetime
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Load the CSV data
@st.cache_data
def load_movie_data():
    try:
        df = pd.read_csv("dataset/TMDB_movie_dataset_v11.csv")

        # Convert 'adult' column to Boolean
        df['adult'] = df['adult'].astype(str).str.lower() == 'true'

        # Filter out adult movies
        df = df[df['adult'] == False]

        return df

    except FileNotFoundError:
        st.error("CSV file not found. Please ensure 'dataset/TMDB_movie_dataset_v11.csv' exists.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
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
    page_title="MovieVerse - AI Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for creative modern UI
st.markdown("""
    <style>
    :root {
        --primary: #b31b1b;  /* Dark red color */
        --primary-hover: #d11f1f;  /* Slightly brighter on hover */
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
    
    /* Centered header container */
    .header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-bottom: 2.5rem;
        padding: 1rem 0;
    }
    
    /* Animated gradient text for header */
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
    
    /* Cinematic search box */
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
    
    .search-box-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            to bottom right,
            transparent 45%,
            rgba(179, 27, 27, 0.1) 50%,
            transparent 55%
        );
        animation: shine 4s infinite;
        transform: rotate(30deg);
    }
    
    @keyframes shine {
        0% {transform: translateX(-100%) rotate(30deg);}
        100% {transform: translateX(100%) rotate(30deg);}
    }
    
    /* Search input with film strip effect */
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
    
    /* New selector (add this) */
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
    
    /* Feature chips with film reel effect */
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
    
    .feature-chip::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--primary), transparent);
    }
    
    .feature-chip:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(179, 27, 27, 0.3);
        border-color: var(--primary);
    }
    
    /* Movie cards with cinematic effect */
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
    }
    
    .movie-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--accent));
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
        width: 200px;
        height: 300px;
        object-fit: cover;
        transition: transform 0.4s ease;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        position: relative;
    }
    
    .movie-poster::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
            to bottom,
            transparent 60%,
            rgba(0,0,0,0.7) 90%
        );
        border-radius: 12px;
    }
    
    .movie-poster:hover {
        transform: scale(1.05) rotate(1deg);
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
    
    .movie-overview {
        color: var(--text-secondary);
        line-height: 1.7;
        margin-top: 10px;
        font-size: 1rem;
    }
    
    
    .release-year {
        color: var(--text-secondary);
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Results count */
    .results-count {
        color: var(--text-secondary);
        margin-bottom: 2rem;
        font-size: 1.1rem;
        text-align: center;
        font-weight: 300;
    }
    
    /* Detail page styles */
    .detail-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .tagline {
        font-style: italic;
        color: var(--primary);
        margin-bottom: 10px;
        font-size: 1.1rem;
    }
    
    .genre-badge {
        display: inline-block;
        background: rgba(179, 27, 27, 0.15);
        color: var(--primary);
        padding: 6px 14px;
        border-radius: 50px;
        margin-right: 8px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        border: 1px solid rgba(179, 27, 27, 0.3);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .search-title {
            font-size: 2.2rem;
        }
        
        .search-subtitle {
            font-size: 1rem;
            padding: 0 1rem;
        }
        
        .search-box-container {
            padding: 2rem 1.5rem;
        }
        
        .movie-card {
            flex-direction: column;
            gap: 20px;
            padding: 20px;
        }
        
        .movie-poster {
            width: 100%;
            height: auto;
            aspect-ratio: 2/3;
        }
        
        .detail-container {
            padding: 1.5rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load data
df = load_movie_data()
if df.empty:
    st.stop()

# Cache vectorizer and tf-idf matrix
@st.cache_resource
def get_vectorizer_and_matrix(dataframe):
    dataframe['title'] = dataframe['title'].fillna('')
    dataframe['overview'] = dataframe['overview'].fillna('')
    dataframe['original_title'] = dataframe['original_title'].fillna('')
    dataframe['search_blob'] = dataframe['title'] + ' ' + dataframe['overview'] + ' ' + dataframe['original_title']
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(dataframe['search_blob'])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = get_vectorizer_and_matrix(df)

# Get movie_id from query params
raw_param = st.query_params.get("movie_id")
detail_id = raw_param if isinstance(raw_param, str) else raw_param[0] if raw_param else None

# ---------------- Detail Page ---------------- #
if detail_id is not None:
    # Add navigation bar to detail page too
    load_navbar()
    
    try:
        detail_id = int(detail_id)
        movie_detail = df[df['id'] == detail_id]
        
        if not movie_detail.empty:
            movie = movie_detail.iloc[0]

            if st.button("⬅ Back to Search", key="back_button"):
                st.query_params.clear()
                st.rerun()

            col1, col2 = st.columns([1, 2])
            with col1:
                poster_path = movie.get("poster_path", None)
                if pd.notna(poster_path) and poster_path:
                    st.image(f"https://image.tmdb.org/t/p/w400{poster_path}", caption=movie.get("title", "No Title"), use_column_width=True)
                else:
                    st.image("https://via.placeholder.com/500x750?text=No+Poster", caption=movie.get("title", "No Title"), use_column_width=True)

            with col2:
                st.title(movie.get("title", "No Title"))
                tagline = movie.get("tagline", "")
                if pd.notna(tagline) and str(tagline).strip():
                    st.markdown(f"<div class='tagline'>*{tagline}*</div>", unsafe_allow_html=True)
                
                release = movie.get("release_date", "")
                runtime = movie.get("runtime", "?")
                if pd.notna(release):
                    try:
                        release = datetime.strptime(str(release), "%Y-%m-%d").strftime("%B %d, %Y")
                    except:
                        pass
                st.caption(f"📅 {release} • ⏱️ {runtime} min")
                
                rating = movie.get("vote_average", None)
                if pd.notna(rating):
                    st.markdown(f"**⭐ Rating:** {rating}/10")
                
                genres = safe_parse_json(movie.get("genres", ""))
                if genres:
                    genre_names = extract_names(genres)
                    st.markdown("**🎞️ Genres:** " + " ".join([f"<span class='genre-badge'>{g}</span>" for g in genre_names]), unsafe_allow_html=True)

                st.subheader("📝 Overview")
                overview = str(movie.get("overview", "")).strip()
                if overview:
                    st.write(overview)
                else:
                    st.write("No description available.")

                cast_data = safe_parse_json(movie.get("cast", ""))
                if cast_data:
                    cast_names = extract_names(cast_data[:8])
                    if cast_names:
                        st.subheader("👤 Top Cast")
                        st.markdown(", ".join(cast_names))

                st.subheader("📌 Additional Information")
                budget = movie.get("budget", 0)
                revenue = movie.get("revenue", 0)
                lang = movie.get("original_language", "")
                status = movie.get("status", "")
                if pd.notna(budget) and budget > 0:
                    st.write(f"**Budget:** ${int(budget):,}")
                if pd.notna(revenue) and revenue > 0:
                    st.write(f"**Revenue:** ${int(revenue):,}")
                if pd.notna(lang):
                    st.write(f"**Original Language:** {lang.upper()}")
                if pd.notna(status):
                    st.write(f"**Status:** {status}")
                homepage = movie.get("homepage", "")
                if pd.notna(homepage) and homepage.strip():
                    st.markdown(f"🔗 [Official Site]({homepage})")
        else:
            st.error("Movie details not found.")
            if st.button("⬅ Back to Search", key="back_button_error"):
                st.query_params.clear()
                st.rerun()
    except ValueError:
        st.error("Invalid movie ID")
        if st.button("⬅ Back to Search", key="back_button_invalid"):
            st.query_params.clear()
            st.rerun()

# ---------------- Search Page ---------------- #
else:
    # Add navigation bar
    load_navbar()
    
    # Main header with centered design
    st.markdown("""
        <div class="header-container">
            <h1 class="search-title">Discover Films Just For You</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature chips
    st.markdown("""
        <div class="features-container">
            <div class="feature-chip">🎬 Search by title</div>
            <div class="feature-chip">🌟 Find by actor</div>
            <div class="feature-chip">🎭 Browse by director</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Search box
    search_query = st.text_input(
        "",
        placeholder="Try search your movie here...",
        key="search_input",
        label_visibility="collapsed"
    )
    
    st.button("Find Movies", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if search_query:
        with st.spinner('🔍 Finding the perfect movies...'):
            query_vec = vectorizer.transform([search_query])
            similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_indices = similarity_scores.argsort()[::-1][:50]
            results = df.iloc[top_indices]
            
            # Filter results to only include complete entries
            results = results[
                results['poster_path'].notna() &
                results['poster_path'].str.strip().ne('') &
                results['overview'].notna() &
                results['overview'].str.strip().ne('') &
                results['release_date'].notna() &
                results['release_date'].str.strip().ne('') &
                results['vote_average'].notna()
            ].head(20)

            if not results.empty:
                # Extract release year from date
                results['year'] = pd.to_datetime(results['release_date']).dt.year
                
                st.markdown(f'<div class="results-count">Showing {len(results)} results</div>', unsafe_allow_html=True)
                
                for _, result in results.iterrows():
                    movie_id = result.get('id')
                    title = result.get('title', 'No Title')
                    year = result.get('year', '')
                    overview = str(result.get('overview', '')).strip()
                    vote_avg = result.get('vote_average', 0)
                    poster_path = result.get('poster_path')

                    if overview and len(overview) > 200:
                        overview = overview[:200] + "..."
                    
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if pd.notna(poster_path) and poster_path else "https://via.placeholder.com/500x750?text=No+Poster"

                    st.markdown(
                        f"""
                        <a href="?movie_id={movie_id}" style="text-decoration: none; color: inherit;">
                            <div class="movie-card">
                                <img src="{poster_url}" alt="{title}" class="movie-poster" />
                                <div class="movie-info">
                                    <h3 class="movie-title">{title}</h3>
                                    <div class="movie-meta">
                                        <span class="release-year">{year}</span>
                                        <span class="movie-rating">⭐ {vote_avg:.1f}</span>
                                    </div>
                                    <p class="movie-overview">{overview or "No overview available"}</p>
                                </div>
                            </div>
                        </a>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No complete movie entries found. Try a different search term.")