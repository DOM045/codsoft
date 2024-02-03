#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# In[5]:


movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")


# In[6]:


movies.head(6)


# In[7]:


credits.head(5)


# In[8]:


movies=movies.merge(credits, on='title')


# In[9]:


movies.head(3)


# In[10]:


movies.shape


# In[11]:


movies.info


# In[12]:


movies.columns


# In[13]:


movies=movies[['movie_id','title','overview','release_date','genres','keywords','cast','crew','spoken_languages']]


# In[14]:


movies.head()


# In[15]:


movies.shape


# In[16]:


movies.isnull().sum()


# In[17]:


movies.dropna(inplace=True)


# In[18]:


movies.duplicated().sum()


# In[19]:


movies.iloc[0]['genres']


# In[20]:


import ast
def convert(obj):
    list=[]
    for i in ast.literal_eval(obj):
        list.append(i['name'])
        
    return list


# In[21]:


movies['genres']=movies['genres'].apply(convert)


# In[22]:


movies.head(1)


# In[23]:


movies['keywords']=movies['keywords'].apply(convert)


# In[24]:


movies['spoken_languages']=movies['spoken_languages'].apply(convert)


# In[25]:


# only show main 3 charaters or cast 
movies['cast'][0]


# In[26]:


import ast
def convert(obj):
    list=[]
    counter =0
    
    for i in ast.literal_eval(obj):
        if counter!=3:
#             to print only three main charaters of movies
# first counter =0 then after first loop counter vvalue will be equal to 1
# then 2 then3 when the counter ==3 the loop will stop and output will be printed 
            list.append(i['name'])
            counter+=1
#         counter=0 -} counter=1
        else:
            break
    return list


# In[27]:


movies['cast']=movies['cast'].apply(convert)


# In[28]:


movies.head(1)


# In[29]:


def fetch_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=="Director":
            l.append(i('name'))
            break
    return l


# In[30]:


movies['crew']=movies['crew'].apply(convert)


# In[31]:


movies.head(1)


# In[32]:


movies['overview'][0]


# In[33]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[34]:


movies.head(1)


# In[35]:


# "Sam Worthington"
# "SamWorthington"


# In[36]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[37]:


movies.head(1)


# In[38]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[39]:


movies.head(1)


# In[40]:


new_df=movies[['movie_id','title','tags']]


# In[41]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[42]:


new_df.head(1)


# In[43]:


get_ipython().system('pip install nltk')


# In[44]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[45]:


def stem(text):
    y=[]
    for i in text.splitO():
        y.append(ps.stem(i))
    return " ".join(y)


# In[100]:


new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df.head()


# In[73]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[74]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[75]:


vectors


# In[76]:


ps.stem('loving')


# In[77]:


from sklearn.metrics.pairwise import cosine_similarity


# In[78]:


# the simlarity variable helps use to understand the a single movies simlarity betn the other remaining 
similarity=cosine_similarity(vectors)


# In[92]:


similarity


# In[101]:


def recommend(movie):
    movies_index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[movies_index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)
        
 
    


# In[103]:


recommend('Avatar')


# In[104]:


# to find the index possions movies
new_df[new_df['title']=='Avatar'].index[0]


# In[105]:


new_df.iloc[1216].title


# In[106]:


import pickle


# In[107]:


pickle.dump(new_df.to_dict(),open('movies_dic.pkl','wb'))


# In[86]:


new_df['title'].values


# In[88]:


pickle.dump(similarity,open("similarity.pkl",'wb'))


# In[ ]:




