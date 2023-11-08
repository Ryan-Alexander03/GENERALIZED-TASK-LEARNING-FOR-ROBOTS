import json  
# load the large language model file
from llama_cpp import Llama
import os
import json
import revtok
import torch
import copy
import progressbar
import pandas as pd



def templating(sentences):
   
    
    #file = f
    #"templates"+str(i)+".json"

    with open('synthetictemps2.json', "w") as text_file:
        text_file.write('{"sentences":[')
        text_file.write("\n")
        
        for index, sen in enumerate(sentences):
            #si='{"sentence'+str(index)+'":['
            output = LLM(SysPrompt)
            input = SysPrompt

            s =  sen.split(",") # spliting by comma, to deal with compound sentences
            if len(s) >1:
                text_file.write('{"sentence":[')
                text_file.write("\n")
            else:
                text_file.write('{"sentence":')
                text_file.write("\n") 

            for i, w in enumerate(s):
                input += output["choices"][0]["text"]+ prefix + w + suffix
                output = LLM(input)
                
                #delete any words the LLM outputs before and/or after template
                template = output["choices"][0]["text"]
                result = template[template.find('{'):]
                result = result.split('}', 1)[0] + '}'
                print(template)
                
                if i == len(s)-1: # this checks so that the last part of a compound sentence doesn't have a comma, which breask the JSON
                    text_file.write(result)
                    text_file.write("\n")
                else:
                    text_file.write(result)
                    text_file.write(", \n")

            if len(s) > 1:
                text_file.write(']')
                
            if index == len(sentences)-1: # this checks so that the last sentences doesn't have a comma, which breask the JSON
                text_file.write("} \n")
            else:
                text_file.write("}, \n")

            if len(input) > 3950:
                output = LLM(SysPrompt)
                input = SysPrompt
                print('!!!!!!!!!!!!!!!!!!!!RESETING CONTEXT!!!!!!!!!!!!!!!!!!!!')
                
        
        text_file.write("]}")

if __name__ == '__main__':
    df = pd.read_csv('SimpleSyntheticData.csv',delimiter=';') 
    sentences = df['sentences'].tolist()
    LLM = Llama(model_path="/mnt/Drive2/research/llama.cpp/models/llama-2-13b-chat/ggml-model-q4_0.bin",n_ctx=4000) # context length of 4000, LLama 2's max
    #prevoius and suffix used for formating converstaion
    prefix = " </s><s>[INST] "
    suffix = " [/INST]"

    # create a system prompt
    SysPrompt = """"
    <s>[INST] <<SYS>>
    Task is to identify the; Action, Object and location in the sentence if there is one. For example the sentence, "Pick up the yellow-handled knife against the wall to the right of the sink."  will be in form
        Action: pick up
        Object: yellow-handled knife
        From: the wall
    for sentences that instruct navigation, "Turn left to face the dresser.". the template should be
        Action: turn
        Direction: left
        Preposition: face
        Landmark: dresser

    only output the answer in JSON format
    <</SYS>>
    
    """
    templating(sentences)