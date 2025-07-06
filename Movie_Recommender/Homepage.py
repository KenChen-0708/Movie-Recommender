import streamlit as st

st.set_page_config(
    page_title="Movie Gallary App",
    page_icon="👋",
)

st.title("Main Page")
st.sidebar.success("Select a page above.")
st.write("""### Welcome to Movie Gallary, we provide the following services :
##### 1. Netflix Movie Recommendation 
     Contains the latest netflix movies
##### 2. Movie Selection and choice! 
     Contains different movies which exist currently 
##### 👈 Pick your choice from the side menu present """)

