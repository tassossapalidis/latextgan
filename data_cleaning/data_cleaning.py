# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:15:36 2021

@author: Andre
"""

import pandas as pd
import numpy as np
import re

df = pd.read_csv('wiki_movie_plots_deduped.csv')
#df = df.loc[df.Title =="What the Daisy Said"]
df = df[['Plot']]
#Don't match Initial. or Mr/Jr/.../"..." or Mrs/ie/eg AND 
#   don't match future punctuation (i.e. the first two dots in an ellipsis)
filter_words_2 = ['Mr', 'Jr', 'Sr', 'Fr', 'Ms', 'Mz', 'Dr', 'St', 'Rd']
filter_words_3 = ['Mrs', 'Rev', 'i\.e', 'e\.g', 'Ave', 'Cir', 'Crt']
match_str = """(?<!\.)(?<![A-Z])(?<!""" + '|'.join(filter_words_2)+""")(?<!""" +\
    '|'.join(filter_words_3)+""")[\.](?!([a-z]?\.))"""
min_sent = 5
sents = df.Plot.str.replace(match_str,".<SPLIT>").tolist()
sents = [re.sub("!","!<SPLIT>", x) for x in sents]
sents = [re.sub("\?","?<SPLIT>", x) for x in sents]
sents = [x for y in [a.split('<SPLIT>') for a in sents] for x in y]
#Match [#] or (_anything_)
bracket_match = '(\[[0-9]+\])|(\([^\)]*\))|(\r)|(\n)'
sents = [re.sub(bracket_match,'',x) for x in sents]
sents = [re.sub('""',"",re.sub(' \,', ',', x)) for x in sents]
sents = [x.lstrip().rstrip() for x in sents if len(x) > min_sent]
df_new = pd.DataFrame(sents, columns = ['Sentences'])

df_new.to_csv(r'Sentences.txt', header=None, index=None, sep=' ', mode='w')
