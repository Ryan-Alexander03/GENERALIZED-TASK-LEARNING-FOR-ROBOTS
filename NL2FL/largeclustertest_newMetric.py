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
taskTotals = {}
templates = []
taskset = []
for task in tasks:
    templatedFolder = os.path.join(folder,task)
    numberFiles = len([entry for entry in os.listdir(templatedFolder) if os.path.isfile(os.path.join(templatedFolder, entry))])
    
    #use dictionary or something to keep track of template adn which folder it came from
    #reads in templates from JSON, concatentes each template in string adds all templates to list
    for i in range(0,numberFiles):
        temps  = []
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
                    temps.append(res)
                    taskset.append(task)
                else:
                    res = ""
                    # Convert Dictionary to Concatenated String
                    # Using for loop and empty string
                    for item in sen:
                        res += item + ": "+ str(sen[item]) +" "
                    templates.append(res)
                    taskset.append(task)
                    temps.append(res)
            taskTotals[task] = len(temps)
        except:
            error_file = open("errors.txt", "w")
            error_file.write(thing)

df = pd.read_csv('cluster_large_llamma_results.csv',delimiter=';') 
#df.sort_values(by=['cluster'])
print(df.query("task == 'look_at_obj_in_light-BasketBall-None-DeskLamp-301'"))

print("\n")

df = pd.read_csv('cluster_large_tfidf_results.csv',delimiter=';') 
#df.sort_values(by=['cluster'])
print(df.query("task == 'look_at_obj_in_light-BasketBall-None-DeskLamp-301'"))

print("\n")

df = pd.read_csv('cluster_large_sentrans_results.csv',delimiter=';') 
#df.sort_values(by=['cluster'])
print(df.query("task == 'look_at_obj_in_light-BasketBall-None-DeskLamp-301'"))

    
    


   





