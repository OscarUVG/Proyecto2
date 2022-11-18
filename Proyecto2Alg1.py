import numpy as np
import pandas as pd
import re

df1 = pd.read_csv('abstracts_train.csv', sep='\t')
df2 = pd.read_csv('entities_train.csv', sep='\t')
df1['full'] = df1['title'] + ' ' + df1['abstract']

# Train y test
df1_train_size = int(df1.shape[0]*0.7)
df2_train_size = df2[df2['abstract_id'] == df1['abstract_id'].iloc[df1_train_size-1]]['id'].iloc[-1]
df1_train = df1.iloc[:df1_train_size]
df2_train = df2.iloc[:df2_train_size]
df1_test = df1.iloc[df1_train_size:]
df2_test = df2.iloc[df2_train_size:]

# ====== ALGORITMO 1 ====== 

# Enfrenamiento del algoritmo

def find_mentions(mentions_dict, full):
    for i in range(len(mentions_dict)):
        mentions_dict[i] = (mentions_dict[i][0].replace('+','\+'), mentions_dict[i][1])

    mentions_found = []
    for mention in mentions_dict: 
        matches = re.finditer(r'[\s(]'+mention[0]+'[\s.,;)]', full)
        new_mentions = [(m.span()[0]+1, m.span()[1]-1, mention[0], mention[1]) for m in matches]
        mentions_found += new_mentions
    
    mentions_found = sorted(mentions_found, key = lambda m: m[0])
    return mentions_found

def translate_mentions(full, mentions_found):
    new_full = []
    i_full = 0
    for mention in mentions_found:
        new_full.append(full[i_full:mention[0]])
        new_full.append((mention[2], mention[3]))
        i_full = mention[1]
    return new_full
        
def freq_mentions(mentions_found):
    types = [m[3] for m in mentions_found]
    types_df = pd.DataFrame({'Type':types})
    freqs = types_df.value_counts()
    return freqs
        
mentions_dict = df2_train[['mention', 'type']].value_counts().index.tolist()
full = df1_test['full'].iloc[-1]

mentions_found = find_mentions(mentions_dict, full)
translated_mentions = translate_mentions(full, mentions_found)

freqs = freq_mentions(mentions_found)
#print(df2_test[df2_test['abstract_id'] == df1_test['abstract_id'].iloc[-1]][['offset_start', 'offset_finish', 'mention']])


    
