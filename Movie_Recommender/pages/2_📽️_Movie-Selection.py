import requests
import streamlit as st
import pandas as pd
api_key = "6272a18b79461f7987d2f1047b3d1198"
movie_id = 550
api_version = 3
api_base_url = f"https://api.themoviedb.org/{api_version}"
endpoint_path = f"/search/movie"


st.write('# Search your movie!')
search_query = st.text_input(" Enter here! ")
st.write('### The current movie title is', search_query)
endpoint = f"{api_base_url}{endpoint_path}?api_key={api_key}&query={search_query}"


#print(endpoint)
r=requests.get(endpoint) 
#pprint.pprint(r.json())
if r.status_code in range(200,299):
    
    data=r.json()
    
    results=data['results']
    if len(results)>0: 
        st.success("The movie is available!")
        #st.write("## Keys")
        #st.write(results[0].keys())
        movie_ids=list()
        movie_title=set()
        movie_RD=set()
        st.write('### Here are the movies related to', search_query)
        col1, col2 ,col3 ,col4 = st.columns(4) 
        with col1:
            st.write("#### Poster")  
        with col2:
            st.write("#### Name")  
        with col3:
            st.write("#### Movie ID")  
        with col4:
            st.write("#### Release Date")  
        

        for result in results:
            _id = result['id']
            _RD = result['release_date']
            _title = result['title']
            
            poster = (result['poster_path'])
            full_path = f"https://image.tmdb.org/t/p/w500/{poster}"
            col1, col2 ,col3 ,col4 = st.columns(4) 
            with col1:
                st.image(full_path)  
            with col2:
                st.write(result['title'])
            with col3:
                st.write(_id)
            with col4:
                st.write(_RD)

                     
            #st.image(full_path, 50)
            movie_ids.append(_id)
            movie_title.add(_title)
            movie_RD.add(_RD)
            
        






            
    else :
        if len(search_query)<=0:
            st.info("Enter the movie name")   
        elif len(results)<=0:
            st.error("The movie is not available!!")        