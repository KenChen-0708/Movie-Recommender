import pandas as pd
import requests
import time

API_KEY = "6272a18b79461f7987d2f1047b3d1198"

def fetch_poster(title):
    url = f"https://api.themoviedb.org/3/search/movie"
    params = {"api_key": API_KEY, "query": title}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get("results")
        if results:
            poster_path = results[0].get("poster_path")
            return poster_path
    return None

# Load your original dataset
df = pd.read_csv("title.csv")

# Fetch posters for each movie
df['poster_path'] = df['title'].apply(fetch_poster)

# Save to new CSV
df.to_csv("netflix_with_posters.csv", index=False)
print("✅ Posters added and saved to netflix_with_posters.csv")
