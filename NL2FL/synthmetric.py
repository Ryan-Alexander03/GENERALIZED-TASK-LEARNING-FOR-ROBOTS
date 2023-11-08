import json  
import numpy as np
import os, os.path
from sklearn.cluster import DBSCAN, dbscan
import pandas as pd
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.feature_extraction.text import TfidfVectorizer
from leven import levenshtein

model = SentenceTransformer(r"sentence-transformers/paraphrase-MiniLM-L6-v2")
templates=[]

#use dictionary or something to keep track of template adn which folder it came from
#reads in templates from JSON, concatentes each template in string adds all templates to list  
f = open('synthetictemps.json')
data = json.load(f)
sentences = data['sentences']
for sentence in sentences:                           
    #check for compoound sentences
    sen = sentence['sentence']
    if str(type(sen)) =="<class 'list'>":
        for s in sen:
            res = ""
        # Convert Dictionary to Concatenated String
        # Using for loop and empty string
        for item in s:
            res += item + ": "+ str(s[item]) +" "
        templates.append(res)
    
    else:
        res = ""
        # Convert Dictionary to Concatenated String
        # Using for loop and empty string
        for item in sen:
            res += item + ": "+ str(sen[item]) +" "
        templates.append(res)


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


df = pd.read_csv('SimpleSyntheticData.csv',delimiter=';') 
sentences = df['sentences'].tolist()
clusters = df['cluster'].tolist()


##### 0.5 epsilon
epsilon = 0.9

LlamadbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(LlamaArrTemplates)
enTransdbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(senTransArrTemplates)
TfidfdbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(TfidfArrTemplates)

LlamaLables = LlamadbClustering.labels_
senTransLables = enTransdbClustering.labels_
TfidfLables = TfidfdbClustering.labels_

# results = pd.DataFrame(list(zip(sentences,clusters,LlamaLables,senTransLables,TfidfLables)),
#                 columns = ['sentences','clusters','Llama_clusters','senTrasn_clusters','tfidf_clusters'])


tfcount = 0
stcount = 0
lcount = 0
for i in range(len(TfidfLables)-25):
        if  TfidfLables[i] == TfidfLables[i+25] and TfidfLables[i] != -1:
            tfcount +=1 
        if  senTransLables[i] == senTransLables[i+25] and senTransLables[i] != -1:
            stcount +=1
        if  LlamaLables[i] == LlamaLables[i+25] and LlamaLables[i] != -1:
            lcount +=1

print(str(tfcount)+"/"+str(len(TfidfLables)))
print(str(stcount)+"/"+str(len(TfidfLables)))
print(str(lcount)+"/"+str(len(TfidfLables)))

epsilon = 1

LlamadbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(LlamaArrTemplates)
enTransdbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(senTransArrTemplates)
TfidfdbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(TfidfArrTemplates)

LlamaLables = LlamadbClustering.labels_
senTransLables = enTransdbClustering.labels_
TfidfLables = TfidfdbClustering.labels_

tfcount = 0
stcount = 0
lcount = 0
for i in range(len(TfidfLables)-25):
        if  TfidfLables[i] == TfidfLables[i+25] and TfidfLables[i] != -1:
            tfcount +=1 
        if  senTransLables[i] == senTransLables[i+25] and senTransLables[i] != -1:
            stcount +=1
        if  LlamaLables[i] == LlamaLables[i+25] and LlamaLables[i] != -1:
            lcount +=1

print(str(tfcount)+"/"+str(len(TfidfLables)))
print(str(stcount)+"/"+str(len(TfidfLables)))
print(str(lcount)+"/"+str(len(TfidfLables)))
# results = pd.DataFrame(list(zip(sentences,clusters,LlamaLables,senTransLables,TfidfLables)),
#                 columns = ['sentences','clusters','Llama_clusters','senTrasn_clusters','tfidf_clusters'])

# print(results.head())
# print(TfidfLables)



# with open("cluster_large_llamma_results.csv", "w") as text_file:
#     text_file.write("template, cluster, task \n")
#     for i in range(len(templates)):
#         text_file.write(templates[i]+";"+str(LlamaLables[i])+"\n")

# senTransLables = enTransdbClustering.labels_
# with open("cluster_large_sentrans_results.csv", "w") as text_file:   
#     text_file.write("template, cluster, task \n")
#     for i in range(len(templates)):
#         text_file.write(templates[i]+";"+str(senTransLables[i])+"\n")
    


# TfidfLables = TfidfdbClustering.labels_
# with open("cluster_large_tfidf_results.csv", "w") as text_file: 
#     text_file.write("template, cluster, task \n")
#     for i in range(len(templates)):
#         text_file.write(templates[i]+";"+str(TfidfLables[i])+"\n")

# epsilon = 0.5

# LlamadbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(LlamaArrTemplates)
# enTransdbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(senTransArrTemplates)
# TfidfdbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(TfidfArrTemplates)

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

    
    


   





