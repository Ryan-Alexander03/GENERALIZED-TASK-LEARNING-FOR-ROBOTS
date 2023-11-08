import json  
import numpy as np
import os
import matplotlib.pyplot as plt
import hdbscan
from sklearn.cluster import DBSCAN, dbscan
from sklearn.cluster import HDBSCAN
import pandas as pd
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.feature_extraction.text import TfidfVectorizer
from leven import levenshtein
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score

model = SentenceTransformer(r"sentence-transformers/paraphrase-MiniLM-L6-v2")
#how many documents should cluster

#reads in templates from JSON, concatentes each template in string adds all templates to list
templates=[]
file = "traj_data.json"
f = open(file)
data = json.load(f)
tasks = data['turk_annotations']['anns']
for task in tasks:
    sentences = task['high_descs']
    for sen in sentences:
        templates.append(sen)


LlamaEmbedTemplates = templates.copy()
senTransEmbedTemplates = templates.copy()
TfidfEmbedTemplates = templates.copy()

LLM = Llama(model_path="/mnt/Drive2/research/llama.cpp/models/llama-2-13b/ggml-model-q4_0.bin",embedding=True)
for i in range(len(LlamaEmbedTemplates)):
    LlamaEmbedTemplates[i] = LLM.embed(LlamaEmbedTemplates[i])
LlamaArrTemplates = np.asarray(LlamaEmbedTemplates, dtype=object)


for i in range(len(senTransEmbedTemplates)):
     senTransEmbedTemplates[i] = model.encode(senTransEmbedTemplates[i]) 
senTransArrTemplates = np.asarray(senTransEmbedTemplates, dtype=object)
vectorizer = TfidfVectorizer()
TfidfArrTemplates = vectorizer.fit_transform(TfidfEmbedTemplates)


#####Standard dbscan
print('STANDARD DBSCAN \n \n')
##### 0.5 epsilon
epsilon = 0.5

LlamadbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(LlamaArrTemplates)
enTransdbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(senTransArrTemplates)
TfidfdbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(TfidfArrTemplates)

# LlamaLables = LlamadbClustering.labels_
# Llamasilhouette = silhouette_score(LlamaArrTemplates,LlamaLables)
# LlamaCH = calinski_harabasz_score(LlamaArrTemplates,LlamaLables)
# LlamaDB = davies_bouldin_score(LlamaArrTemplates,LlamaLables)
# print('Llamma \n')
# print('silhouette_score: '+str(Llamasilhouette))
# print('\n')
# print('calinski_harabasz_score: '+str(LlamaCH))
# print('\n')
# print('davies_bouldin_score: '+str(LlamaDB))
# print('\n \n')

# senTransLables = enTransdbClustering.labels_
# senTranssilhouette = silhouette_score(senTransArrTemplates,senTransLables)
# senTransCH = calinski_harabasz_score(senTransArrTemplates,senTransLables)
# senTransDB = davies_bouldin_score(senTransArrTemplates,senTransLables)
# print('Sentence Transformer \n')
# print('silhouette_score: '+str(senTranssilhouette))
# print('\n')
# print('calinski_harabasz_score: '+str(senTransCH))
# print('\n')
# print('davies_bouldin_score: '+str(senTransDB))
# print('\n \n')

# TfidfLables = TfidfdbClustering.labels_
# Tfidfsilhouette = silhouette_score(TfidfArrTemplates,TfidfLables)
# TfidfCH = calinski_harabasz_score(TfidfArrTemplates.toarray(),TfidfLables)
# TfidfDB = davies_bouldin_score(TfidfArrTemplates.toarray(),TfidfLables)
# print('Tfidf\n')
# print('silhouette_score: '+str(Tfidfsilhouette))
# print('\n')
# print('calinski_harabasz_score: '+str(TfidfCH))
# print('\n')
# print('davies_bouldin_score: '+str(TfidfDB))
# print('\n \n')

with open("cluster_sen_results.txt", "w") as text_file:
    text_file.write('Llamma \n')
    LlamaLables = LlamadbClustering.labels_
    for i in range(len(templates)):
        text_file.write(templates[i]+str(LlamaLables[i])+"\n")
    text_file.write('\n \n')

    senTransLables = enTransdbClustering.labels_

    text_file.write('Sentence Transformer \n')
    for i in range(len(templates)):
        text_file.write(templates[i]+str(senTransLables[i])+"\n")
    text_file.write('\n \n')

    TfidfLables = TfidfdbClustering.labels_
    text_file.write('Tfidf\n')
    for i in range(len(templates)):
        text_file.write(templates[i]+str(TfidfLables[i])+"\n")
    text_file.write('\n \n')
# print('cos_sim \n')
# LlamadbClustering = DBSCAN(eps=epsilon, min_samples=2,metric = cos_sim).fit(LlamaArrTemplates)
# enTransdbClustering = DBSCAN(eps=epsilon, min_samples=2,metric = cos_sim).fit(senTransArrTemplates)
# LlamaLables = LlamadbClustering.labels_
# Llamasilhouette = silhouette_score(LlamaArrTemplates,LlamaLables)
# LlamaCH = calinski_harabasz_score(LlamaArrTemplates,LlamaLables)
# LlamaDB = davies_bouldin_score(LlamaArrTemplates,LlamaLables)
# print('Llamma \n')
# print('silhouette_score: '+str(Llamasilhouette))
# print('\n')
# print('calinski_harabasz_score: '+str(LlamaCH))
# print('\n')
# print('davies_bouldin_score: '+str(LlamaDB))
# print('\n \n')

# senTransLables = enTransdbClustering.labels_
# senTranssilhouette = silhouette_score(senTransArrTemplates,senTransLables)
# senTransCH = calinski_harabasz_score(senTransArrTemplates,senTransLables)
# senTransDB = davies_bouldin_score(senTransArrTemplates,senTransLables)
# print('Sentence Transformer \n')
# print('silhouette_score: '+str(senTranssilhouette))
# print('\n')
# print('calinski_harabasz_score: '+str(senTransCH))
# print('\n')
# print('davies_bouldin_score: '+str(senTransDB))
# print('\n \n')

# TfidfLables = TfidfdbClustering.labels_
# Tfidfsilhouette = silhouette_score(TfidfArrTemplates,TfidfLables)
# TfidfCH = calinski_harabasz_score(TfidfArrTemplates.toarray(),TfidfLables)
# TfidfDB = davies_bouldin_score(TfidfArrTemplates.toarray(),TfidfLables)
# print('Tfidf\n')
# print('silhouette_score: '+str(Tfidfsilhouette))
# print('\n')
# print('calinski_harabasz_score: '+str(TfidfCH))
# print('\n')
# print('davies_bouldin_score: '+str(TfidfDB))
# print('\n \n')
# TfidfdbClustering05 = DBSCAN(eps=epsilon, min_samples=2,metric = cos_sim).fit(TfidfArrTemplates[0])   
# print('Llamma \n')
# for i in range(len(templates)):
#     print(templates[i]+str(LlamaLables[i])+"\n")
# print('\n \n')

# senTransLables = enTransdbClustering.labels_

# print('Sentence Transformer \n')
# for i in range(len(templates)):
#     print(templates[i]+str(senTransLables[i])+"\n")
# print('\n \n')

# TfidfLables = TfidfdbClustering.labels_
# print('Tfidf\n')
# for i in range(len(templates)):
#     print(templates[i]+str(TfidfLables[i])+"\n")
# print('\n \n')


# def lev_metric(x, y):
#         i, j = int(x[0]), int(y[0])     # extract indices
#         return levenshtein(templates[i], templates[j])

# X = np.arange(len(templates)).reshape(-1, 1)
# dbClustering = dbscan(X, eps=epsilon, min_samples=2, metric=lev_metric)

# print('Leven \n \n')
# Lables = dbClustering[1]
# silhouette = silhouette_score(X,Lables)
# CH = calinski_harabasz_score(X,Lables)
# DB = davies_bouldin_score(X,Lables)
# print('silhouette_score: '+str(silhouette))
# print('\n')
# print('calinski_harabasz_score: '+str(CH))
# print('\n')
# print('davies_bouldin_score: '+str(DB))
# print('\n \n')
    


   





