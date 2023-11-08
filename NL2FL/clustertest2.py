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
for i in range(0,3):
    file = "templates"+str(i)+".json"
    f = open(file)
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


epsilon = 0.5


LlamadbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(LlamaArrTemplates)
senTransdbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(senTransArrTemplates)
TfidfdbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(TfidfArrTemplates)


# with open("cluster_text_results.txt", "w") as text_file:


#     #####Standard dbscan
#     text_file.write('STANDARD DBSCAN \n \n')
#     ##### 0.5 epsilon
#     epsilon = 1

#     LlamadbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(LlamaArrTemplates)
#     enTransdbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(senTransArrTemplates)
#     TfidfdbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(TfidfArrTemplates)

#     LlamaLables = LlamadbClustering.labels_
   
#     text_file.write('Llamma \n')
#     for i in range(len(templates)):
#         text_file.write(templates[i]+str(LlamaLables[i])+"\n")
#     text_file.write('\n \n')

#     senTransLables = enTransdbClustering.labels_
    
#     text_file.write('Sentence Transformer \n')
#     for i in range(len(templates)):
#         text_file.write(templates[i]+str(senTransLables[i])+"\n")
#     text_file.write('\n \n')

#     TfidfLables = TfidfdbClustering.labels_
#     text_file.write('Tfidf\n')
#     for i in range(len(templates)):
#         text_file.write(templates[i]+str(TfidfLables[i])+"\n")
#     text_file.write('\n \n')

    
    


   





