import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Content Discovery Page",
    page_icon="🎬",
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/kench/Downloads/AI/Movie-Recommender/Movie_Recommender/dataset/TMDB_movie_dataset_v11.csv")

df = load_data()

# Filter out adult content
df = df[df["adult"] != True]  # or df[~df["adult"]] if it's boolean

# Calculate C (mean vote average) and m (minimum vote count threshold)
C = df["vote_average"].mean()
m = df["vote_count"].quantile(0.95)  # only consider movies in top 20% by vote count

# Filter qualified movies
qualified = df[df["vote_count"] >= m].copy()

# Compute the weighted rating using IMDB formula
def weighted_rating(x, m=m, C=C):
    v = x["vote_count"]
    R = x["vote_average"]
    return (v / (v + m) * R) + (m / (v + m) * C)

qualified["score"] = qualified.apply(weighted_rating, axis=1)

# Sort movies based on the score
top_rated = qualified.sort_values("score", ascending=False)

# Drop missing posters and create poster URLs
top_rated = top_rated.dropna(subset=["poster_path"]).head(30)
top_rated["poster_url"] = "https://image.tmdb.org/t/p/w300" + top_rated["poster_path"]

# Title
st.title("🎥 Top Rated Movies")

# Generate poster HTML for carousel
poster_html = """
<style>
.carousel-container {
    position: relative;
    width: 100%;
    margin: 0 auto;
    padding: 20px 0;
}

.carousel-wrapper {
    position: relative;
    overflow: hidden;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.carousel-track {
    display: flex;
    transition: transform 0.5s ease;
    gap: 1%;
    padding: 20px 0.5%;
}

.movie-card {
    flex: 0 0 19.2%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    cursor: pointer;
}

.movie-card:hover {
    transform: translateY(-10px) scale(1.05);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
}

.movie-poster {
    width: 100%;
    height: 320px;
    object-fit: cover;
    transition: transform 0.3s ease;
}

.movie-card:hover .movie-poster {
    transform: scale(1.1);
}

.movie-info {
    padding: 15px;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
}

.movie-title {
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
}

.movie-rating {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    color: #666;
}

.rating-star {
    color: #ffc107;
}

.carousel-nav {
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
}

.carousel-nav:hover {
    background: rgba(255, 255, 255, 1);
    transform: translateY(-50%) scale(1.1);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

.carousel-nav:active {
    transform: translateY(-50%) scale(0.95);
}

.carousel-nav.prev {
    left: 10px;
}

.carousel-nav.next {
    right: 10px;
}

.carousel-dots {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 20px;
}

.dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.5);
    cursor: pointer;
    transition: background 0.3s ease;
}

.dot.active {
    background: #667eea;
}

@media (max-width: 768px) {
    .movie-card {
        flex: 0 0 19.2%;
    }
    
    .movie-poster {
        height: 240px;
    }
    
    .movie-title {
        font-size: 12px;
        height: 32px;
    }
    
    .movie-info {
        padding: 10px;
    }
    
    .carousel-nav {
        width: 40px;
        height: 40px;
        font-size: 16px;
    }
}
</style>

<div class="carousel-container">
    <div class="carousel-wrapper">
        <button class="carousel-nav prev" onclick="previousSlide()">‹</button>
        <div class="carousel-track" id="carouselTrack">
"""

# Add movie cards to HTML
for _, movie in top_rated.iterrows():
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

poster_html += """
        </div>
        <button class="carousel-nav next" onclick="nextSlide()">›</button>
    </div>
    <div class="carousel-dots" id="carouselDots"></div>
</div>

<script>
let currentSlide = 0;
const track = document.getElementById('carouselTrack');
const cards = document.querySelectorAll('.movie-card');
const totalCards = cards.length;
const cardsPerView = 5; // Always show exactly 5 posters
const totalSlides = Math.ceil(totalCards / cardsPerView);

// Create dots
function createDots() {
    const dotsContainer = document.getElementById('carouselDots');
    dotsContainer.innerHTML = '';
    
    for (let i = 0; i < totalSlides; i++) {
        const dot = document.createElement('div');
        dot.className = 'dot';
        if (i === 0) dot.classList.add('active');
        dot.onclick = () => goToSlide(i);
        dotsContainer.appendChild(dot);
    }
}

// Update dots
function updateDots() {
    const dots = document.querySelectorAll('.dot');
    dots.forEach((dot, index) => {
        dot.classList.toggle('active', index === currentSlide);
    });
}

// Go to specific slide
function goToSlide(slideIndex) {
    currentSlide = slideIndex;
    const translateX = -slideIndex * 100; // Move by 100% to show next 5 cards
    track.style.transform = `translateX(${translateX}%)`;
    updateDots();
}

// Previous slide
function previousSlide() {
    if (currentSlide > 0) {
        currentSlide--;
        goToSlide(currentSlide);
    } else {
        // Loop to last slide
        currentSlide = totalSlides - 1;
        goToSlide(currentSlide);
    }
}

// Next slide
function nextSlide() {
    if (currentSlide < totalSlides - 1) {
        currentSlide++;
        goToSlide(currentSlide);
    } else {
        // Loop to first slide
        currentSlide = 0;
        goToSlide(currentSlide);
    }
}

// Auto-play functionality (optional)
let autoPlayInterval;

function startAutoPlay() {
    autoPlayInterval = setInterval(() => {
        nextSlide();
    }, 4000); // Change slide every 4 seconds
}

function stopAutoPlay() {
    clearInterval(autoPlayInterval);
}

// Initialize carousel
createDots();
goToSlide(0);

// Optional: Start auto-play (uncomment if desired)
// startAutoPlay();

// Pause auto-play on hover
const carousel = document.querySelector('.carousel-container');
carousel.addEventListener('mouseenter', stopAutoPlay);
carousel.addEventListener('mouseleave', startAutoPlay);

// Touch/swipe support for mobile
let startX = 0;
let currentX = 0;
let isDragging = false;

track.addEventListener('touchstart', (e) => {
    startX = e.touches[0].clientX;
    isDragging = true;
    stopAutoPlay();
});

track.addEventListener('touchmove', (e) => {
    if (!isDragging) return;
    currentX = e.touches[0].clientX;
    const deltaX = currentX - startX;
    
    if (Math.abs(deltaX) > 50) {
        if (deltaX > 0) {
            previousSlide();
        } else {
            nextSlide();
        }
        isDragging = false;
    }
});

track.addEventListener('touchend', () => {
    isDragging = false;
    // startAutoPlay(); // Uncomment if using auto-play
});

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') {
        previousSlide();
    } else if (e.key === 'ArrowRight') {
        nextSlide();
    }
});

// Responsive handling
window.addEventListener('resize', () => {
    goToSlide(0);
    createDots();
});
</script>
"""

# Render centered poster grid
components.html(poster_html, height=480)