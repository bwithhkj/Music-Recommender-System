# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:13:07 2021

@author: Bearded Khan
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.externals import joblib
import Test0 as test0
import Evaluation as Evalutaion 


triplets_file =  'https://static.turi.com/datasets/millionsong/10000.txt'
songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'

song_df_1 = pd.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

#Read song metadata
song_df_2 =  pd.read_csv(songs_metadata_file)

#Merge the two dataframes above to create input dataframe for recommender systems
song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

#EXPLORE DATA (This show how many times a user )
song_df.head()

#Length of the dataset
len(song_df)

#Creating the subset of the dataset
song_df = song_df.head(10000)
#Merge sort song title and artist name columns to make a merged column
song_df['song'] = song_df['title'].map(str) + " _ " + song_df['artist_name']


#Most POPULAR Songs in the dataset
song_grouped = song_df.groupby(['song']).agg({'listen_count' : 'count'}).reset_index() 
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage'] = song_grouped['listen_count'].div(grouped_sum)*100 
song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])


#Counting the number of unique users in the dataset
users = song_df['user_id'].unique()
len(users)

#Counting the unique songs in the dataset 
songs = song_df['song'].unique()
len(songs)

#CREATE A SONG RECOMMENDER 
train_data, test_data = train_test_split(song_df , test_size = 0.20, random_state = 0)
print(train_data.head(5))

#Simple Popularity-based recommender class(Can be used as a black box) 
#Create an instance of popularity based recommender class
pm = test0.popularity_recommender_py()
pm.create(train_data, 'user_id', 'song')

#Use the popularity model to make some predictions
user_id = users[8]
pm.recommend(user_id)

#Personalization Recommender Approach 
#Instance of the class/module that is written previously
#test0.item_similarity_recommender_py()

#Create an instance of item similarity based recommender class
is_model = test0.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')

#Personalized model used to make some song recommendations(Trainning data) 
user_id = users[5]
user_items = is_model.get_user_items(user_id)
print("*******************************************************************")
print("-----------------------Trainning data songs for the user userid: %s:" % user_id)
print("*******************************************************************")

for user_item in user_items:
    print(user_item)
print("*******************************************************************")
print("Recommendation INPROCESS :")
print("*******************************************************************")

#Recommending songs for the user using personalized model
g = is_model.recommend(user_id)
###########################################################################################################################
#Personalized model to make Recommendations for the Following user Id 
user_id = users[7]
user_items = is_model.get_user_items(user_id)


print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")

x = is_model.recommend(user_id)
###########################################################################################################################

# we can use the presonalized recommender model to get similar songs for the any song
song = 'Yellow _ Coldplay'
is_model.get_similar_items([song]) # was getting an error on this stage as the meta data has underScore '_' and i was putting an hiphen - here Opps!


#CODE to Plot precision recall curve 
import pylab as pl

def plot_precision_recall(m1_precision_list, m1_recall_list, m1_label, m2_precision_list, m1_label,
                          m2_precision_list, m2_recall_list, m2_label):
    pl.clf()
    pl.plot(m1_recall_list, m1_precision_list, label=m1_label)
    pl.plot(m2_recall_list, m2_precision_list, label=m2_label)
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 0.20])
    pl.xlim([0.0, 0.20])
    pl.title('Precision-Recall curve')
    pl.legend(loc = 9, bbox_to_anchor=(0.5, -0.2))
    pl.show()
































