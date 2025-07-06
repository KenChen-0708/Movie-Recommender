# Movie Recommendation App

## Overview

This is a Python-based web application that provides various movie-related functionalities. It is built using Streamlit as the front-end framework and leverages several Python libraries to deliver a seamless movie selection and recommendation experience.

### Key Features

1. **Movie Selection**: You can search for any movie that exists using this feature. It utilizes an API call system to fetch movie details.

2. **Movie Recommender**: The application provides movie recommendations based on a dataset from Kaggle. It recommends existing Netflix movies based on user preferences.

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/movie-recommendation-app.git
   ```

2. Navigate to the project directory:

   ```bash
   cd movie-recommendation-app
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open your web browser and visit `http://localhost:8501` to access the Movie Recommendation App.

## Dependencies

The project uses the following Python libraries:

- [Streamlit](https://streamlit.io/): For creating the web application interface.
- [Pandas](https://pandas.pydata.org/): For data manipulation and analysis.
- [Requests](https://docs.python-requests.org/en/latest/): For making API requests.
- [Scikit-learn (Sklearn)](https://scikit-learn.org/stable/): For machine learning and recommendation algorithms.
- [Plotly](https://plotly.com/): For interactive data visualization.

## Data Sources

- Movie data for the "Movie Selection" feature is obtained through API calls (ThemovieDB).
- Movie recommendation data is sourced from Kaggle (Netflix Movies 2019).

## Screenshots 

### 1.Homepage
![Alt text](image.png)

### 2.Movie Recommendation
![Alt text](image-1.png)

### 3.Movie Selection
![Alt text](image-2.png)
