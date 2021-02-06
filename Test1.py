# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:13:07 2021

@author: Bearded Khan
"""
import pandas 
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn.externals import joblib
import Test0 as Test0
import Evaluation as Evalutaion 


triplets_file =  'https://static.turi.com/datasets/millionsong/10000.txt'
songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'

song_df_1 = pandas.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

#Read song metadata
song_df_2 =  pandas.read_csv(songs_metadata_file)

#Merge the two dataframes above to create input dataframe for recommender systems
song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

#EXPLORE DATA (This show how many times a user )
song_df.head()

#Length of the dataset
len(song_df)

#Creating the subset of the dataset
song_df = song_df.head(10000)
#Merge sort song title and artist name columns to make a merged column
song_df['Song'] = song_df['title'].map(str) + " _ " + song_df['artist_name']


#Most POPULAR Songs in the dataset
song_grouped = song_df.groupby(['Song']).agg({'listen_count' : 'count'}).reset_index() 
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage'] = song_grouped['listen_count'].div(grouped_sum)*100 
song_grouped.sort_values(['listen_count', 'Song'], ascending = [0,1])


#Counting the number of unique users in the dataset
users = song_df['user_id'].unique()
len(users)

#Counting the unique songs in the dataset 
songs = song_df['Song'].unique()
len(songs)

#CREATE A SONG RECOMMENDER 
train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state = 0)
print(train_data.head(5))

#Simple Popularity-based recommender class(Can be used as a black box) 
#Create an instance of popularity based recommender class
pm = Test0.popularity_recommender_py()
pm.create(train_data, 'user_id', 'Song')

#Use the popularity model to make some predictions
user_id = users[5]
pm.recommend(user_id)