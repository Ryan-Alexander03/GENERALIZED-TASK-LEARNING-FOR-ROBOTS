import json  
import numpy as np
import os
from sklearn.cluster import DBSCAN
import pandas as pd
from llama_cpp import Llama
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.neighbors import KernelDensity
import re



def removearticles(text):
  return re.sub('\s+(a|an|and|the|of|in|in the)(\s+)',  ' ', text)

def formalize(sentence): # concatenate objects
    formalSen=""
    if "Action" and "Direction" in sentence:
        if sentence["Direction"] != None:
            formalSen+= sentence["Action"]+"_"+sentence["Direction"] +" "
        else:
            formalSen+= sentence["Action"]+" "
        for item in sentence:
            if item != "Action" and item !="Direction" and sentence[item]!= None:
                formalSen += str(sentence[item]) +" " 
    # elif "Preposition" and "Landmark" in sentence:
    #     if sentence["Preposition"] != None:
    #         formalSen+= sentence["Preposition"]+"_"+sentence["Landmark"]
    else:
        for item in sentence:
            if item == "Action" or item == "Object" :
               action = str(sentence[item]).replace(" ", "_")
               formalSen += action +" "
            elif sentence[item]!= None:
                formalSen += str(sentence[item])+" "

    formalSen = removearticles(formalSen)
    return formalSen.strip()
    

def paramterise(base):
    """
    Action(x,y)
    Action(object,location)
    Action(location)
    Action(object)
    """
    format = base.split()
    #identify format
    action = format[0]
    
    # if len(format)>2:
    #     x = format[1]
    #     y=""
    #     for i in range(2,len(format)):
    #         y += format[i]+" "
    #     y = y.strip(" ").replace(" ", "_") 
    #     paramaterisedBase = f"{action}({x},{y})"
    # elif len(format)==1:
    #     #x = format[1]
    #     paramaterisedBase = f"{action}()"
    # else:
    #     x = format[1]
    #     paramaterisedBase = f"{action}({x})"

    
    format = base.split()
    if len(format)>2:
        x = format[1]
        y=""
        for i in range(2,len(format)):
            y += format[i]+" "
        y = y.strip(" ").replace(" ", "_") 
        paramaterisedSen  = {
        "m_name":action,
        "m_args":[x,y],
        }
    elif len(format)==1:
        paramaterisedSen ={
        "m_name":action,
        }
    else:
        x = format[1]
        paramaterisedSen  ={
        "m_name":action,
        "m_args":x,
        }

    return paramaterisedSen

# if __name__ == '__main__':
#     #reads in templates from JSON, concatentes each template in string adds all templates to list
#     templates=[]
#     for i in range(0,3):
#         file = "template_"+str(i)+".json"
#         f = open(file)
#         data = json.load(f)
#         sentences = data['sentences']
#         for sentence in sentences:                           
#             #check for compoound sentences
#             sen = sentence['sentence']
#             if str(type(sen)) =="<class 'list'>":
#                 res=""
#                 for s in sen:
#                     res = formalize(s)
                    
#                 # res=""
#                 # for i in range(len(sen)):
#                 #     if i == len(sen)-1:
#                 #         res+= formalize(sen[i])
#                 #     else:
#                 #         res+= formalize(sen[i]) +", "
#                 # for s in sen:
#                 #     res = ""
#                 #     # Convert Dictionary to Concatenated String
#                 #     # Using for loop and empty string
#                 #     for item in s:
#                 #         res += item + ": "+ str(s[item]) +" "

#                     templates.append(res.strip())
#             else:
#                 res = formalize(sen)
#                 # res = ""
#                 # # Convert Dictionary to Concatenated String
#                 # # Using for loop and empty string
#                 # for item in sen:
#                 #     res += item + ": "+ str(sen[item]) +" "
#                 templates.append(res.strip())

#     # print(templates)
#     TfidfEmbedTemplates = templates.copy()

#     vectorizer = TfidfVectorizer()
#     TfidfArrTemplates = vectorizer.fit_transform(TfidfEmbedTemplates)
#     TfidfdbClustering = DBSCAN(eps=1, min_samples=2).fit(TfidfArrTemplates)

#     TfidfArr = TfidfArrTemplates.toarray()


#     clusters = defaultdict(list)
#     temps = defaultdict(list)

#     for i,c in enumerate(TfidfdbClustering.labels_):
#         clusters[c].append(TfidfArr[i])
#         temps[c].append(templates[i])

#     bases=[]
#     for i in clusters.keys():
#         kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(clusters[i])
#         log_density = kde.score_samples(clusters[i])
#         mostLikely = np.argmax(log_density)
#         clusters[i][mostLikely]

#         for j in range (len(TfidfArr)):
#             if (clusters[i][mostLikely] == TfidfArr[j]).all():
#                 bases.append(templates[j])
#                 break # break becuase sometimes there are multiple identical templates
    
#     #print(bases)
#     # for b in bases:
#     #     print(paramterise(b))
#     for i in temps.keys():
#         print('\n')
#         print(bases[i])
#         for j in temps[i]:
#             print(paramterise(bases[i],j))


    