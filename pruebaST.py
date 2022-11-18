# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:51:49 2022

@author: rafae
"""

import streamlit as st
from annotated_text import annotated_text, annotation
import pandas as pd
import numpy as np
import plotly.express as px 
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
        
mentions_dict = df2_train[['mention', 'type']].value_counts().index.tolist()
full = df1_test['full'].iloc[-1]

mentions_found = find_mentions(mentions_dict, full)
translated_mentions = translate_mentions(full, mentions_found)





#----------------------------INICIO DE APP---------------------------------

st.title('APP: Caracterización de artículos científicos')

"""
Por Rafael Dubois y Oscar Godoy.
"""

st.header('¡Bienvenido a la app de caracterización de artículos científicos!')

st.subheader('Ingreso de título')
Titulo = st.text_input('Ingrese el título de su artículo científico:') 
st.write('El título del artículo científico es: ', Titulo)

st.subheader('Ingreso de resumen')
Resumen = st.text_area('Ingrese el resumen (abstract) de su artículo científico:')
st.write('El resumen del artículo científico es: ', Resumen)

concat = st.checkbox('Mostrar concatenación de título y resumen')

Completo = Titulo + ' ' + Resumen


if concat:
    st.subheader('Concatenación')
    st.write('A continuación vemos la concatenación del título y el resumen.\n',Completo)

#ENCONTRANDO MENCIONES DEL ARTÍCULO 'COMPLETO'

mentionsFound = find_mentions(mentions_dict,Completo)

mentionsTranslated = translate_mentions(Completo,mentionsFound)

st.header('Sección de datos detallados')

detalle = st.checkbox('Mostrar resultados detallados')

if detalle:
    
    modelo = st.radio(
        "¿Qué modelo desea utilizar?",
        ('Modelo 1', 'Modelo 2')
        )

    if modelo == 'Modelo 1':
            
        """
        # Ejemplo de texto anotado
        
        A continuación vemos un ejemplo de texto con anotaciones.
        """    
    
        with st.echo():
            annotated_text(
            *mentionsTranslated
            )
            
        chart_data = pd.DataFrame(
        np.random.randn(20, 6),
        columns=["GeneOrGeneProduct", "DiseaseOrPhenotypicFeature", "ChemicalEntity",
                 "OrganismTaxon","SequenceVariant","CellLine"])
        
        st.bar_chart(chart_data)
        
        df=px.data.tips()
        fig=px.bar(df,x='total_bill',y='day', orientation='h')
        st.write(fig)
            
    if modelo == 'Modelo 2':
        
        """
        # Aún no tenemos nada
        
        :(((
        """



#We report on a new allele at the arylsulfatase A (ARSA) locus causing 
#late-onset metachromatic leukodystrophy (MLD). In that allele arginine84,
#a residue that is highly conserved in the arylsulfatase gene family, is 
#replaced by glutamine. In contrast to alleles that cause early-onset MLD, 
#the arginine84 to glutamine substitution is associated with some residual 
#ARSA activity. A comparison of genotypes, ARSA activities, and clinical 
#data on 4 individuals carrying the allele of 81 patients with MLD examined, 
#further validates the concept that different degrees of residual ARSA 
#activity are the basis of phenotypical variation in MLD..