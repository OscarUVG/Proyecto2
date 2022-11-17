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

fulls = df1_test['full'].tolist()
mentions_dict = df2_train[['mention', 'type']].value_counts().index.tolist()
for i in range(len(mentions_dict)):
    mentions_dict[i] = (mentions_dict[i][0].replace('+','\+'), mentions_dict[i][1])

mentions_found = []
for i in range(len(mentions_dict)): 
    matches = re.finditer(r'\s'+mentions_dict[i][0]+'[\s.,;]', fulls[1])
    new_mentions = [(m.span()[0], m.span()[1], mentions_dict[i][0]) for m in matches]
    mentions_found += new_mentions
    
print(fulls[1])
print(10*'-')
print(df2_test[df2_test['abstract_id'] == df1_test['abstract_id'].iloc[1]][['offset_start', 'offset_finish', 'mention']])
print(10*'-')
for mention in mentions_found:
    print(mention,'\n')

            










# new_mentions = re.finditer(r'\s'+mention+'[\s,.;]', full)



