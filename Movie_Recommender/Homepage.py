import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import requests
import json
from datetime import datetime
import ast

# Page configuration
st.set_page_config(
    page_title="CineSearch Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_KEY = "6272a18b79461f7987d2f1047b3d1198"
API_BASE_URL = "https://api.themoviedb.org/3"

# Load dataset for top rated movies
@st.cache_data
def load_data():
    return pd.read_csv("dataset/TMDB_movie_dataset_v11.csv", dtype={'id': str})

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

# Fetch movie details from API by TMDB ID
@st.cache_data
def fetch_movie_details_by_tmdb_id(tmdb_id):
    try:
        url = f"{API_BASE_URL}/movie/{tmdb_id}"
        params = {
            'api_key': API_KEY,
            'language': 'en-US',
            'append_to_response': 'credits'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching movie details: {e}")
        return None

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
        </style>
    """, unsafe_allow_html=True)

# Generate carousel HTML with clickable cards using anchor tags
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
    outline: 2px solid #00d4ff;
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

.rating-star {{
    color: #ffc107;
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

    # Add movie cards to HTML with anchor tags
    for movie in movies:
        title = str(movie.get('title', 'Unknown Title'))[:50]
        rating = round(float(movie.get('vote_average', 0)), 1)
        poster_url = movie['poster_url']
        movie_id = movie.get('id', '')
        source = movie.get('source', 'dataset')
        
        # Create the URL with proper parameters
        detail_url = f"?movie_id={movie_id}&source={source}"
        
        poster_html += f"""
            <a href="{detail_url}" class="movie-card" target="_blank">
                <img src="{poster_url}" alt="{title}" class="movie-poster" onerror="this.src='https://via.placeholder.com/200x280?text=No+Image'">
                <div class="movie-info">
                    <h3 class="movie-title">{title}</h3>
                    <div class="movie-rating">
                        <span class="rating-star">★</span>
                        <span>{rating}/10</span>
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

# Movie detail page display
def display_movie_details(movie_id, source='dataset'):
    if source == 'api':
        # Fetch from API
        movie_data = fetch_movie_details_by_tmdb_id(movie_id)
        if not movie_data:
            st.error("Movie details not found.")
            return
        
        # Convert API data to match dataset format
        movie = {
            'id': str(movie_data.get('id')),
            'title': movie_data.get('title', 'No Title'),
            'original_title': movie_data.get('original_title', ''),
            'overview': movie_data.get('overview', ''),
            'tagline': movie_data.get('tagline', ''),
            'release_date': movie_data.get('release_date', ''),
            'runtime': movie_data.get('runtime', 0),
            'vote_average': movie_data.get('vote_average', 0),
            'budget': movie_data.get('budget', 0),
            'revenue': movie_data.get('revenue', 0),
            'original_language': movie_data.get('original_language', ''),
            'status': movie_data.get('status', ''),
            'homepage': movie_data.get('homepage', ''),
            'poster_path': movie_data.get('poster_path', ''),
            'genres': movie_data.get('genres', []),
            'cast': movie_data.get('credits', {}).get('cast', [])[:8] if movie_data.get('credits') else []
        }
    else:
        # Fetch from dataset
        df = load_data()
        movie_detail = df[df['id'] == movie_id]
        if movie_detail.empty:
            st.error("Movie details not found.")
            return
        movie = movie_detail.iloc[0]

    # Display movie details
    col1, col2 = st.columns([1, 2])

    with col1:
        poster_path = movie.get("poster_path", None)
        if pd.notna(poster_path) and poster_path:
            if source == 'api':
                poster_url = f"https://image.tmdb.org/t/p/w400{poster_path}"
            else:
                poster_url = f"https://image.tmdb.org/t/p/w400{poster_path}"
            st.image(poster_url, caption=movie.get("title", "No Title"), use_container_width=True)
        else:
            st.image("https://via.placeholder.com/400x600?text=No+Poster", 
                    caption=movie.get("title", "No Title"), use_container_width=True)

    with col2:
        st.title(movie.get("title", "No Title"))
        
        tagline = movie.get("tagline", "")
        if pd.notna(tagline) and str(tagline).strip():
            st.markdown(f'<div class="tagline">"{tagline}"</div>', unsafe_allow_html=True)

        # Movie metadata
        release = movie.get("release_date", "")
        runtime = movie.get("runtime", "")
        if pd.notna(release) and release:
            try:
                if isinstance(release, str):
                    release = datetime.strptime(release, "%Y-%m-%d").strftime("%B %d, %Y")
            except:
                pass
        
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            st.write(f"**📅 Release Date:** {release if release else 'N/A'}")
        with col_meta2:
            st.write(f"**⏱️ Runtime:** {runtime} min" if runtime else "**⏱️ Runtime:** N/A")

        # Rating
        rating = movie.get("vote_average", None)
        if pd.notna(rating) and rating:
            st.markdown(f'<div class="rating-display">⭐ {rating}/10</div>', unsafe_allow_html=True)

        # Genres
        if source == 'api':
            genres = movie.get("genres", [])
            genre_names = [genre.get('name', '') for genre in genres] if genres else []
        else:
            genres = safe_parse_json(movie.get("genres", ""))
            genre_names = extract_names(genres)
        
        if genre_names:
            st.markdown("**🎞️ Genres:**")
            for genre in genre_names:
                st.markdown(f'<span class="genre-badge">{genre}</span>', unsafe_allow_html=True)

        # Overview
        # st.markdown("---")
        st.subheader("📝 Overview")
        overview = movie.get("overview", "")
        if pd.notna(overview) and str(overview).strip():
            st.write(overview)
        else:
            st.write("No description available.")

        # Cast
        if source == 'api':
            cast_data = movie.get("cast", [])
            cast_names = [actor.get('name', '') for actor in cast_data] if cast_data else []
        else:
            cast_data = safe_parse_json(movie.get("cast", ""))
            cast_names = extract_names(cast_data)
        
        if cast_names:
            st.subheader("👤 Top Cast")
            st.write(", ".join(cast_names))

        # Additional Information
        # st.markdown("---")
        st.subheader("📌 Additional Information")
        
        budget = movie.get("budget", 0)
        revenue = movie.get("revenue", 0)
        lang = movie.get("original_language", "")
        status = movie.get("status", "")

        if pd.notna(budget) and budget and int(budget) > 0:
            st.write(f"**Budget:** ${int(budget):,}")
        if pd.notna(revenue) and revenue and int(revenue) > 0:
            st.write(f"**Revenue:** ${int(revenue):,}")
        if pd.notna(lang) and lang:
            st.write(f"**Original Language:** {lang.upper()}")
        if pd.notna(status) and status:
            st.write(f"**Status:** {status}")

        homepage = movie.get("homepage", "")
        if pd.notna(homepage) and homepage.strip():
            st.markdown(f"🔗 [Official Website]({homepage})")

# Main app
def main():
    load_custom_css()
    
    # Check for movie_id in query params for detail view
    movie_id = st.query_params.get("movie_id")
    source = st.query_params.get("source", "dataset")
    
    if movie_id:
        # Detail page
        # if st.button("⬅ Back to Discovery"):
        #     st.query_params.clear()
        #     st.rerun()
        
        # st.markdown('<div class="detail-container">', unsafe_allow_html=True)
        display_movie_details(movie_id, source)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Main discovery page
        # Load top rated movies from dataset
        df = load_data()
        df = df[df["adult"] != True]
        
        # Calculate top rated movies
        C = df["vote_average"].mean()
        m = df["vote_count"].quantile(0.95)
        qualified = df[df["vote_count"] >= m].copy()
        
        def weighted_rating(x, m=m, C=C):
            v = x["vote_count"]
            R = x["vote_average"]
            return (v / (v + m) * R) + (m / (v + m) * C)
        
        qualified["score"] = qualified.apply(weighted_rating, axis=1)
        top_rated = qualified.sort_values("score", ascending=False)
        top_rated = top_rated.dropna(subset=["poster_path"]).head(30)
        
        # Convert to same format as API data
        top_rated_movies = []
        for _, movie in top_rated.iterrows():
            top_rated_movies.append({
                'id': str(movie.get('id')),
                'title': movie.get('title', 'Unknown Title'),
                'vote_average': movie.get('vote_average', 0),
                'poster_url': f"https://image.tmdb.org/t/p/w300{movie['poster_path']}",
                'source': 'dataset'
            })
        

        # Popular movies
        popular_movies = (
            df.dropna(subset=["poster_path"])
            .sort_values("popularity", ascending=False)
            .head(30)
        )

        # Convert to API-like format
        popular_movies_list = []
        for _, movie in popular_movies.iterrows():
            popular_movies_list.append({
                'id': str(movie.get('id')),
                'title': movie.get('title', 'Unknown Title'),
                'vote_average': movie.get('vote_average', 0),
                'poster_url': f"https://image.tmdb.org/t/p/w300{movie['poster_path']}",
                'source': 'dataset'
            })

        # Fetch API data
        trending_movies = fetch_trending_movies()
        now_playing_movies = fetch_now_playing_movies()
        
        # Display sections
        st.title("🎬 Movie Discovery")
        
        # Trending Movies
        st.header("🔥 Trending Movies")
        if trending_movies:
            components.html(generate_carousel_html(trending_movies, "Trending", "Trending Movies"), height=480)
        else:
            st.error("Unable to load trending movies")

        # Top Rated Movies
        st.header("🏆 Top Rated Movies")
        if top_rated_movies:
            components.html(generate_carousel_html(top_rated_movies, "TopRated", "Top Rated Movies"), height=480)
        else:
            st.error("Unable to load top rated movies")
        
        # Popular Movies
        st.header("🌟 Popular Movies")
        if popular_movies_list:
            components.html(generate_carousel_html(popular_movies_list, "Popular", "Popular Movies"), height=480)
        else:
            st.error("Unable to load popular movies")

        # Now Playing Movies
        st.header("🎭 Now Playing")
        if now_playing_movies:
            components.html(generate_carousel_html(now_playing_movies, "NowPlaying", "Now Playing Movies"), height=480)
        else:
            st.error("Unable to load now playing movies")

if __name__ == "__main__":
    main()