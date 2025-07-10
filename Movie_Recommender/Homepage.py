import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Content Discovery Page",
    page_icon="🎬",
)

# API configuration
API_KEY = "6272a18b79461f7987d2f1047b3d1198"
API_BASE_URL = "https://api.themoviedb.org/3"

# Load dataset for top rated movies
@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/kench/Downloads/AI/Movie-Recommender/Movie_Recommender/dataset/TMDB_movie_dataset_v11.csv")

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
                    'title': movie.get('title', 'Unknown Title'),
                    'vote_average': movie.get('vote_average', 0),
                    'poster_url': f"https://image.tmdb.org/t/p/w300{movie['poster_path']}"
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
                    'title': movie.get('title', 'Unknown Title'),
                    'vote_average': movie.get('vote_average', 0),
                    'poster_url': f"https://image.tmdb.org/t/p/w300{movie['poster_path']}"
                })
        return movies
    except Exception as e:
        st.error(f"Error fetching now playing movies: {e}")
        return []

# Generate carousel HTML
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
    cursor: pointer;
}}

.movie-card:hover {{
    transform: translateY(-10px) scale(1.05);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
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

    # Add movie cards to HTML
    for movie in movies:
        title = str(movie.get('title', 'Unknown Title'))[:50]
        rating = round(float(movie.get('vote_average', 0)), 1)
        poster_url = movie['poster_url']
        
        poster_html += f"""
            <div class="movie-card">
                <img src="{poster_url}" alt="{title}" class="movie-poster" onerror="this.src='https://via.placeholder.com/200x280?text=No+Image'">
                <div class="movie-info">
                    <h3 class="movie-title">{title}</h3>
                    <div class="movie-rating">
                        <span class="rating-star">★</span>
                        <span>{rating}/10</span>
                    </div>
                </div>
            </div>
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
</script>
"""
    
    return poster_html

# Main app
def main():
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
            'title': movie.get('title', 'Unknown Title'),
            'vote_average': movie.get('vote_average', 0),
            'poster_url': f"https://image.tmdb.org/t/p/w300{movie['poster_path']}"
        })
    
    # Fetch API data
    trending_movies = fetch_trending_movies()
    now_playing_movies = fetch_now_playing_movies()
    
    # Display sections
    st.title("🎬 Movie Discovery")
    
    # Top Rated Movies
    st.header("🏆 Top Rated Movies")
    if top_rated_movies:
        components.html(generate_carousel_html(top_rated_movies, "TopRated", "Top Rated Movies"), height=480)
    else:
        st.error("Unable to load top rated movies")
    
    # Trending Movies
    st.header("🔥 Trending Movies")
    if trending_movies:
        components.html(generate_carousel_html(trending_movies, "Trending", "Trending Movies"), height=480)
    else:
        st.error("Unable to load trending movies")
    
    # Now Playing Movies
    st.header("🎭 Now Playing")
    if now_playing_movies:
        components.html(generate_carousel_html(now_playing_movies, "NowPlaying", "Now Playing Movies"), height=480)
    else:
        st.error("Unable to load now playing movies")

if __name__ == "__main__":
    main()