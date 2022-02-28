import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

def transform(data,new_df):
    for index in data.index:
        title = data.loc[index,'title'].strip()
        director = data.loc[index,'director'].strip().replace(',',' ')
        cast = data.loc[index,'cast'].strip().replace(',',' ')
        country = data.loc[index,'country'].strip().replace(',',' ')
        rating = data.loc[index,'rating'].strip().replace(',',' ')
        description = data.loc[index,'description'].strip('.').replace(',',' ')
        new_df.loc[index,'Id'] = data.loc[index,'show_id']
        new_df.loc[index,'Title'] = data.loc[index,'title']
        info = description+' '+title+' '+director+' '+cast+' '+country+' '+rating
        new_df.loc[index,'Info'] = info
    return new_df

def recommend(movie):   
    rec_list = []                                      
    movies = new_df[new_df['Title'] == movie].index[0]
    distance = sorted(list(enumerate(similarity[movies])),reverse=True,key = lambda x: x[1])
    for i in distance[1:11]:
        rec_list.append(new_df.iloc[i[0]].Title)
    return rec_list
         
data = pd.read_csv('netflix_titles.csv')
data = data.replace(np.nan,'')
data = data[['show_id','type','title','director','cast','country','rating','description']]
new_df = pd.DataFrame(columns=['Id','Title','Info'])
new_df = transform(data, new_df)

tfidf = TfidfVectorizer(strip_accents='ascii',analyzer='word',stop_words='english',max_features=15000)
vectorizer = tfidf.fit_transform(new_df['Info'])
similarity = cosine_similarity(vectorizer)


st.title('Netflix Recommender System')
type_ = st.radio(
    'What you want to watch?',
    ('Movie','Series')
)

if type_ == 'Movie':
    options = st.selectbox(
        'Type movie name...',
        data[data['type']=='Movie']['title'].tolist()
    )
else:
    options = st.selectbox(
        'Type series name...',
        data[data['type']=='TV Show']['title'].tolist()
    )
recommendations = recommend(options)
st.write(f"If you're watching {options} then...")
st.subheader('You should also watch')
with st.container():
    for suggestions in recommendations:
        st.write(suggestions)










