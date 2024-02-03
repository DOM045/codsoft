import streamlit as st
import pickle
import pandas as pd
import requests

def fetch_poster(movie_id):
    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=22680c1a582fdaae8c4a3effec046465&language=en-US".format(movie_id)'.format(movie_id))
    data=response.json()
    return 'https.//image.tmdb.org/t/p/w500/'+data['poster_path']
# recommending 5 similar movies
def recommend(movie):
    movies_index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[movies_index])), reverse=True, key=lambda x: x[1])
    recommended_movies=[]
    recommended_movies_poster=[]
    for i in distances[1:6]:
        movie_id=movies.iloc[i[0]].movie_id

        # poster

        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_poster.append(fetch_poster(movie_id))
    return recommended_movies, recommended_movies_poster




movies_dict=pickle.load(open('movies_dic.pkl','rb'))
movies=pd.DataFrame(movies_dict)


similarity=pickle.load(open('similarity.pkl','rb'))
st.title('Moive Recommender System')

selected_movie_name = st.selectbox(
    'how would you like to be contacted?',
    movies['title'].values
)
if st.button('Recommend'):
    names,posters= recommend(selected_movie_name)
#     display posters
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])

    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])

