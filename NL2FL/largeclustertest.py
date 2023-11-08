import json  
import numpy as np
import os, os.path
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
folder = 'data/json_2.1.0/templated'


my_file = open("tasks.txt", "r")
# reading the file
file = my_file.read()
tasks = file.split("\n")
dataset = {}
templates=[]
taskset = []
for task in tasks:
    templatedFolder = os.path.join(folder,task)
    numberFiles = len([entry for entry in os.listdir(templatedFolder) if os.path.isfile(os.path.join(templatedFolder, entry))])

    
    #use dictionary or something to keep track of template adn which folder it came from
    #reads in templates from JSON, concatentes each template in string adds all templates to list
    for i in range(0,numberFiles):
        temps  =[]
        file = "template_"+str(i)+".json"
        thing = os.path.join(templatedFolder,file)
        try:
            f = open(thing)
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
                    taskset.append(task)
                    temps.append(res)
                else:
                    res = ""
                    # Convert Dictionary to Concatenated String
                    # Using for loop and empty string
                    for item in sen:
                        res += item + ": "+ str(sen[item]) +" "
                    templates.append(res)
                    temps.append(res)
                    taskset.append(task)
            dataset[task] = temps
        except:
            error_file = open("errors.txt", "w")
            error_file.write(thing)

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



##### 0.5 epsilon
epsilon = 0.5

LlamadbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(LlamaArrTemplates)
enTransdbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(senTransArrTemplates)
TfidfdbClustering = DBSCAN(eps=epsilon, min_samples=2).fit(TfidfArrTemplates)

LlamaLables = LlamadbClustering.labels_
with open("cluster_large_llamma_results.csv", "w") as text_file:
    text_file.write("template, cluster, task \n")
    for i in range(len(templates)):
        text_file.write(templates[i]+";"+str(LlamaLables[i])+";"+taskset[i]+"\n")

senTransLables = enTransdbClustering.labels_
with open("cluster_large_sentrans_results.csv", "w") as text_file:   
    text_file.write("template, cluster, task \n")
    for i in range(len(templates)):
        text_file.write(templates[i]+";"+str(senTransLables[i])+";"+taskset[i]+"\n")
    


TfidfLables = TfidfdbClustering.labels_
with open("cluster_large_tfidf_results.csv", "w") as text_file: 
    text_file.write("template, cluster, task \n")
    for i in range(len(templates)):
        text_file.write(templates[i]+";"+str(TfidfLables[i])+";"+taskset[i]+"\n")

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

    
    


   





