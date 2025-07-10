import requests
import streamlit as st
from datetime import datetime

# Setup
api_key = "6272a18b79461f7987d2f1047b3d1198"
api_base_url = "https://api.themoviedb.org/3"
search_endpoint = f"{api_base_url}/search/movie"
detail_endpoint = f"{api_base_url}/movie"

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

# Get movie_id from query params
query_params = st.query_params
detail_id = query_params.get("movie_id", [None])[0]

# ---------------- Detail Page ---------------- #
if detail_id:
    detail_url = f"{detail_endpoint}/{detail_id}?api_key={api_key}&append_to_response=credits"
    detail_res = requests.get(detail_url).json()
    
    # Back button
    st.markdown(
        '<a href="?" class="back-button" target="_self">⬅ Back to Search</a>',
        unsafe_allow_html=True
    )
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        poster_path = detail_res.get("poster_path")
        if poster_path:
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
            st.image(poster_url, use_column_width=True, output_format="auto", caption=detail_res.get("title"))
    
    with col2:
        st.title(detail_res.get("title", "No Title"))
        
        # Tagline
        if detail_res.get("tagline"):
            st.markdown(f'<p class="tagline">"{detail_res.get("tagline")}"</p>', unsafe_allow_html=True)
        
        # Basic info
        release_date = detail_res.get("release_date")
        if release_date:
            try:
                date_obj = datetime.strptime(release_date, "%Y-%m-%d")
                formatted_date = date_obj.strftime("%B %d, %Y")
                st.caption(f"📅 {formatted_date} • ⏱️ {detail_res.get('runtime', '?')} min")
            except:
                st.caption(f"📅 {release_date}")
        
        # Rating
        vote_avg = detail_res.get("vote_average")
        if vote_avg:
            st.markdown(f'<div class="movie-rating">⭐ {vote_avg}/10</div>', unsafe_allow_html=True)
        
        # Genres
        genres = detail_res.get("genres", [])
        if genres:
            st.markdown("<div style='margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)
            for genre in genres:
                st.markdown(f'<span class="genre-badge">{genre["name"]}</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Overview
        st.markdown('<div class="detail-section">', unsafe_allow_html=True)
        st.subheader("Overview")
        st.write(detail_res.get("overview", "No description available."))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Cast
        credits = detail_res.get("credits", {})
        if credits.get("cast"):
            st.markdown('<div class="detail-section">', unsafe_allow_html=True)
            st.subheader("Top Cast")
            cast = credits["cast"][:6]  # Show top 6 cast members
            cols = st.columns(6)
            for i, person in enumerate(cast):
                with cols[i]:
                    profile_path = person.get("profile_path")
                    if profile_path:
                        st.image(
                            f"https://image.tmdb.org/t/p/w200{profile_path}",
                            width=80,
                            caption=person["name"]
                        )
                    else:
                        st.image(
                            "https://via.placeholder.com/200x300?text=No+Image",
                            width=80,
                            caption=person["name"]
                        )
            st.markdown('</div>', unsafe_allow_html=True)

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
            endpoint = f"{search_endpoint}?api_key={api_key}&query={search_query}"
            response = requests.get(endpoint)
            
            if response.status_code == 200:
                results = response.json().get('results', [])
                
                if results:
                    st.markdown(f"### Found {len(results)} results")
                    for result in results:
                        movie_id = result.get('id')
                        title = result.get('title')
                        release_date = result.get('release_date', 'N/A')
                        overview = result.get('overview', 'No overview available.')
                        vote_avg = result.get('vote_average', 0)
                        poster_path = result.get('poster_path')
                        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/500x750?text=No+Poster"
                        
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
                                        <p class="movie-overview">{overview[:200]}...</p>
                                    </div>
                                </div>
                            </a>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("No movies found. Try a different search term.")
            else:
                st.error("Failed to connect to movie database. Please try again later.")
    else:
        st.info("Enter a movie title to begin your search")