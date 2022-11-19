# -*- coding: utf-8 -*-
"""
@author: Rafael Dubois, Oscar Godoy
"""

import streamlit as st
from annotated_text import annotated_text, annotation
import pandas as pd
import plotly.express as px 
import re
import nltk

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

# Train y test 2

df1_train_size2 = int(df1.shape[0]*0.9)
df2_train_size2 = df2[df2['abstract_id'] == df1['abstract_id'].iloc[df1_train_size2-1]]['id'].iloc[-1]
df1_train2 = df1.iloc[:df1_train_size2]
df2_train2 = df2.iloc[:df2_train_size2]
df1_test2 = df1.iloc[df1_train_size2:]
df2_test2 = df2.iloc[df2_train_size2:]

# ====== ALGORITMO 1 ====== 

# Entrenamiento del algoritmo

#encontrando menciones
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

#traducción de menciones al formato de annotated_text
def translate_mentions(full, mentions_found):
    new_full = []
    i_full = 0
    for mention in mentions_found:
        new_full.append(full[i_full:mention[0]])
        new_full.append((mention[2], mention[3]))
        i_full = mention[1]
    return new_full
        
#frecuencia de menciones (para gráfica)
def freq_mentions(mentions_found):
    types = [m[3] for m in mentions_found]
    types_df = pd.DataFrame({'type':types})
    names = types_df['type'].value_counts().index.tolist()
    values = [v for v in types_df['type'].value_counts()]
    freqs_df = pd.DataFrame({'names':names, 'values':values})
    return freqs_df
   
#DICCIONARIOS (IMPORTANTE!!!)

mentions_dict100 = df2[['mention', 'type']].value_counts().index.tolist() #100%
mentions_dict80 = df2_train2[['mention', 'type']].value_counts().index.tolist() #80%
mentions_dict = df2_train[['mention', 'type']].value_counts().index.tolist() #70%

# ====== ALGORITMO 2 ====== 



sent = df1_test['full'].iloc[-1]
#chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent)), binary=False)



#=======ESTO ERA UN EJEMPLO=======     
#full = df1_test['full'].iloc[-1]

#mentions_found = find_mentions(mentions_dict, full)
#translated_mentions = translate_mentions(full, mentions_found)

#freqs = freq_mentions(mentions_found)
#print(df2_test[df2_test['abstract_id'] == df1_test['abstract_id'].iloc[-1]][['offset_start', 'offset_finish', 'mention']])

#----------------------------INICIO DE APP---------------------------------

st.title('APP: Caracterización de artículos científicos (biomedicina)')

"""
Por Rafael Dubois y Oscar Godoy.
"""

st.header('¡Bienvenid@ a la app de caracterización de artículos científicos!')

#Título del artículo
st.subheader('Ingreso de título')
Titulo = st.text_input('Ingrese el título de su artículo científico:') 
largoTitulo = len(Titulo)
palabrasTitulo = len(re.findall(r'\w+', Titulo))
#st.write('El título del artículo científico es: ', Titulo)

#Abstract del artículo
st.subheader('Ingreso de resumen')
Resumen = st.text_area('Ingrese el resumen (abstract) de su artículo científico:',height=500)
largoResumen = len(Resumen)
palabrasResumen = len(re.findall(r'\w+', Resumen))
#st.write('El resumen del artículo científico es: ', Resumen)

#Concatenación
Completo = Titulo + ' ' + Resumen
largoCompleto = len(Completo)
palabrasCompleto = len(re.findall(r'\w+', Completo))


concat = st.checkbox('¿Mostrar concatenación de título y resumen?')

if concat:
    st.subheader('Concatenación')
    """
    *A continuación vemos la concatenación del título y el resumen.*
    """
    st.write(Completo)

datosRapidos = st.checkbox('¿Mostrar los datos simples sobre el artículo?')

#-------ENCONTRANDO MENCIONES DEL ARTÍCULO 'COMPLETO' (concatenado)-------

#70%
mentionsFound = find_mentions(mentions_dict,Completo)

mentionsTranslated = translate_mentions(Completo,mentionsFound)

df = freq_mentions(mentionsFound)

palabrasModelo1 = len(mentionsFound)

#80%

mentionsFound80 = find_mentions(mentions_dict80,Completo)

mentionsTranslated80 = translate_mentions(Completo,mentionsFound80)

df80 = freq_mentions(mentionsFound80)

palabrasModelo2 = len(mentionsFound80)


#100%
mentionsFound100 = find_mentions(mentions_dict100,Completo)

mentionsTranslated100 = translate_mentions(Completo,mentionsFound100)

df100 = freq_mentions(mentionsFound100)

palabrasReal = len(mentionsFound100)

#----------------------SECCIÓN DE DATOS RÁPIDOS----------------------
if datosRapidos:
    st.subheader('Datos simples')
    st.write('La cantidad de *caracteres* en el _título_ es:',largoTitulo)
    st.write('La cantidad de *caracteres* en el _resumen_ es:',largoResumen)
    st.write('La cantidad de *caracteres* en la _concatenación_ es:',largoCompleto)
    st.write('La cantidad de *palabras* en el _título_ es:',palabrasTitulo)
    st.write('La cantidad de *palabras* en el _resumen_ es:',palabrasResumen)
    st.write('La cantidad de *palabras* en la _concatenación_ es:',palabrasCompleto)
    
    st.subheader('Conclusión preliminar')
    st.write('La categoría de entidades mencionadas de forma más recurrente en el artículo es:',
             df100.loc[df100['values']==max(df100["values"])]["names"][0])

#----------------------SECCIÓN DE DATOS DETALLADOS----------------------

st.header('Sección de datos detallados')

detalle = st.checkbox('¿Mostrar resultados detallados?')

if detalle:
    
    modelo = st.radio(
        "¿Qué modelo desea utilizar?",
        ('Modelo 1', 'Modelo 2', 'Control')
        )

    #-------------------------------MODELO 1----------------------------------  

    if modelo == 'Modelo 1':
            
        """
        # Texto con anotaciones (Modelo 1: Naive Pattern Search)
        
        A continuación vemos el texto completo con anotaciones.
        """    
    
        with st.echo():
            annotated_text(
            *mentionsTranslated
            )
        
        """
        # Gráfica de frecuencia
        
        A continuación vemos la gráfica de frecuencias de tipos.
        """    
        
        fig = px.bar(df,x='values',y='names',color='names',orientation='h')
        st.write(fig)
        
        """
        # Eficiencia del modelo
        
        """ 
        
        st.write('La eficiencia del modelo 1 sobre este artículo es del',100*palabrasModelo1/palabrasReal,'%.')
    
    #-------------------------------MODELO 2----------------------------------  
        
    if modelo == 'Modelo 2':
        
            
        """
        # Texto con anotaciones (Modelo 2: Named Entity Recognition)
        
        A continuación vemos el texto completo con anotaciones.
        """    
        
        with st.echo():
            annotated_text(
            *mentionsTranslated80
            )
        
        """
        # Gráfica de frecuencia
        
        A continuación vemos la gráfica de frecuencias de tipos.
        """    
        
        fig = px.bar(df80,x='values',y='names',color='names',orientation='h')
        st.write(fig)
        
        """
        # Eficiencia del modelo
        
        """ 
        
        st.write('La eficiencia del modelo 2 sobre este artículo es del',100*palabrasModelo2/palabrasReal,'%.')

    #-------------------------------CONTROL----------------------------------  

    if modelo == 'Control':
        """
        # Modelo de control
        
        A continuación se presenta un *modelo* con efectividad de 100%.
        """
        annotated_text(*mentionsTranslated100)
            
        """
        # Gráfica de frecuencia
        
        A continuación vemos la gráfica de frecuencias de tipos.
        """    
        
        fig = px.bar(df100,x='values',y='names',color='names',orientation='h')
        st.write(fig)
        
#        chart_data = pd.DataFrame(
#        np.random.randn(20, 6),
#        columns=["GeneOrGeneProduct", "DiseaseOrPhenotypicFeature", "ChemicalEntity",
#                 "OrganismTaxon","SequenceVariant","CellLine"])       
#        st.bar_chart(chart_data)
        
        """
        # Eficiencia del modelo
        
        """ 
        
        st.write('La eficiencia del modelo de control sobre este artículo es del',100*palabrasReal/palabrasReal,'%.')