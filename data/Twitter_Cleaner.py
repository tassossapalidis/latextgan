# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 15:32:06 2021

@author: Andre
"""

import pandas as pd
import numpy as np

df = pd.read_csv('2016_US_election_tweets_100k.csv')

df = df[['tweet_text']]
df.dropna(inplace=True)
pat = "(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
df['tweet_text'] = df.tweet_text.str.replace(pat,"<URL>")

important_twitter_names = ['@realDonaldTrump','@BarackObama','@HillaryClinton','@BernieSanders']
name_pat = f'(?!{"|".join(important_twitter_names)})@([A-Za-z0-9_]+)'
df['tweet_text'] = df.tweet_text.str.replace(name_pat, "<USER>")

df.to_csv(r'Tweets.txt', header=None, index=None, sep=' ', mode='w')

train, dev, test = \
              np.split(df.sample(frac=1, random_state=42), 
                       [int(.8*len(df)), int(.9*len(df))])
train.to_csv(r'Tweets_Train.txt', header=None, index=None, sep=' ', mode='w')
dev.to_csv(r'Tweets_Dev.txt', header=None, index=None, sep=' ', mode='w')
test.to_csv(r'Tweets_Test.txt', header=None, index=None, sep=' ', mode='w')