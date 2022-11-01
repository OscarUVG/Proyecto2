import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin

df1 = pd.read_csv('abstracts_train.csv', sep='\t')
df2 = pd.read_csv('entities_train.csv', sep='\t')

# 400 entradas en 3 variables.
print(df1.keys())
print(df1.shape)

# 13636 entradas en 7 variables.
print(df2.keys())
print(df2.shape)

# Longitud titulo
# Media de 116 caracteres, std de 34 caracteres.
# Max 233, min 33. 
lenTitle = df1['title'].apply(len)
plt.clf()
plt.hist(lenTitle)
plt.xlabel('Longitud')
plt.ylabel('Frecuencia')
plt.title('Longitud de título')
#plt.show()

# Longitud abstract
# Media de 1562 caracteres, std de 424 caracteres.
# Max 3125, min 269.
lenAbstract = df1['abstract'].apply(len)
plt.clf()
plt.hist(lenAbstract)
plt.xlabel('Longitud')
plt.ylabel('Frecuencia')
plt.title('Longitud de abstract')
#plt.show()

# Longitud titulo + abstract
# Media de 1678 caracteres, std de 431 caracteres.
# Max 3293, min 354.
plt.clf()
plt.hist(lenTitle+1+lenAbstract)
plt.xlabel('Longitud')
plt.ylabel('Frecuencia')
plt.title('Longitud de título + abstract')
#plt.show()

# Longitud titulo vs longitud abstract
plt.clf()
plt.scatter(lenTitle,lenAbstract)
plt.xlabel('Longitud de título')
plt.ylabel('Frecuencia de abstract')
plt.title('Longitud de título vs longitud de abstract')
#plt.show()

# Conceptos por articulo
# Articulo con mas conceptos tiene 88 conceptos.
# El articulo con menos conceptos tiene 5 conceptos.
# Media 34, std 14. 
entityCount = df2['abstract_id'].value_counts()
plt.clf()
plt.hist(entityCount)
plt.xlabel('Cantidad de conceptos')
plt.ylabel('Frecuencia')
plt.title('Cantidad de conceptos por artículo')
#plt.show()

# Hay 4796 menciones diferentes entre las 13636 mencionaes totales.
# Las menciones mas populares son 'patiens', 'mice', 'patient', 'human', 'rats',
# con frecuencias de 511, 156, 133, 130, 116.
# len(df2['mention'].value_counts())


# Largo de las menciones
# Media de 10 caracteres, std de 8 caracteres.
# Min 1 caracter, max de 110.
# El 80% de las menciones tienen menos de 13 caracteres.
# Ejemplos de min son abreviaturas. Un abstract puede contener una mencion muy recurrente que se decide
# abreviar a una sola letra, como Testosterona (T).
# Deciles array([ 1.,  3.,  4.,  5.,  5.,  7.,  8., 10., 13., 20.])

# df1.loc[df1['abstract_id']==15064320]['abstract'].values
# np.percentile(df2['offset_finish'] - df2['offset_start'], np.arange(0, 100, 10))
plt.clf()
plt.hist(df2['offset_finish'] - df2['offset_start'], bins=13, range=(1,14))
plt.xlabel('Longitud')
plt.ylabel('Frecuencia')
plt.title('Longitud de mención')
#plt.show()


# Hay 6 tipos de concepto. Las frecuencias de estas son:
# Mayor concepto GeneOrGeneProduct con 4764.
# Menor concepto CellLine con 117. 
print(df2['type'].value_counts())
plt.clf()
df2['type'].value_counts().sort_values(ascending=True).plot(kind='barh')
plt.xlabel('Frecuencia')
plt.title('Menciones por tipo de concepto')
#plt.show()


# Posicion de las menciones dentro de los abstract, en medida porcentual.
# Distribucion casi uniforme, con una ligera concentracion de
# menciones al inicio de los articulos (posiblemente por la presencia de los titulos)
dfLen = pd.DataFrame({'abstract_id':df1['abstract_id'],
             'lenTotal': lenTitle + lenAbstract})

df2_Lens = pd.merge(dfLen, df2, how='right', on=['abstract_id'])
offset_start_percent = df2_Lens['offset_start']/df2_Lens['lenTotal']
plt.clf()
plt.hist(offset_start_percent)
plt.title('Posición de mención respecto longitud total')
plt.xlabel('offset_start / lenTotal')
plt.ylabel('Frecuencia')
#plt.show()

# ---------------------------MODELO NPS-------------------------------------------
def search(palabra, texto):
    M = len(palabra)
    N = len(texto)
 
    contador = 0
    for i in range(N - M + 1):
        j = 0
 
        while(j < M):
            if (texto[i + j] != palabra[j]):
                break
            j += 1
 
        if (j == M):
            print("Palabra",palabra,"encontrada en el índice", i)
            contador = contador + 1
    if contador != 0:    
        print("La palabra", palabra, "fue utilizada", contador, "veces en el texto.")
    
    return contador
 
texto = "tengo ADN en todo el cuerpo aaaaaa"
palabra = "ADN"
     
search(palabra, texto)

diccionario = df2['mention'].value_counts().index.tolist()

abstractEjemplo = df1['abstract'][125]

contadores = []
for palabra in diccionario:
    contadores.append(search(palabra, abstractEjemplo))

palabras = []
for i in range(len(contadores)):
    if contadores[i]!=0:
        palabras.append([diccionario[i],contadores[i]])
        
print(palabras)

# ==== NER ====

df = pd.merge(df1, df2, how='right', on=['abstract_id'])
dfT = df.iloc[:int(df.shape[0]*0.8)]
TRAIN_DATA = []
for i in range(dfT.shape[0]):
    entry = dfT.loc[i]
    abstract_id = entry['abstract_id']
    start = entry['offset_start']
    end = entry['offset_finish']
    typeEntity = entry['type']
    abstract = entry['abstract']
    dataEntry = (abstract, {'entities': [(start, end, typeEntity)]})
    TRAIN_DATA.append(dataEntry)


nlp=spacy.blank("en")
nlp.add_pipe('ner')
nlp.begin_training()

nlp = spacy.load('en_core_web_sm')
sentence = df1.loc[0]['abstract']
doc = nlp(sentence)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

