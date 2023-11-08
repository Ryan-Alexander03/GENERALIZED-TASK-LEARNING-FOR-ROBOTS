import json  
# load the large language model file
from llama_cpp import Llama
import os
import json
import revtok
import torch
import copy
import progressbar

def templating(json_path):
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

    Pick up the empty toilet paper roll from the floor. [/INST]

    """

    # returns JSON object as
    # a dictionary
    f = open(json_path)
    data = json.load(f)
    
    #tasks = data['turk_annotations']['anns']
    sentence = data['turk_annotations']['anns'][0]['task_desc']
    output = LLM(SysPrompt)
    input = SysPrompt
    #file = f
    #"templates"+str(i)+".json"

    
    input += output["choices"][0]["text"]+ prefix + sentence + suffix
    output = LLM(input)
    
    #delete any words the LLM outputs before and/or after template
    template = output["choices"][0]["text"]
    result = template[template.find('{'):]
    result = result.split('}', 1)[0] + '}'
    
    return json.loads(result)
                


# if __name__ == '__main__':
#     LLM = Llama(model_path="/mnt/Drive2/research/llama.cpp/models/llama-2-13b-chat/ggml-model-q4_0.bin",n_ctx=4000) # context length of 4000, LLama 2's max


#     #prevoius and suffix used for formating converstaion
#     prefix = " </s><s>[INST] "
#     suffix = " [/INST]"

#     # create a system prompt
#     SysPrompt = """"
#     <s>[INST] <<SYS>>
#     Task is to identify the; Action, Object and location in the sentence if there is one. For example the sentence, "Pick up the yellow-handled knife against the wall to the right of the sink."  will be in form
#         Action: pick up
#         Object: yellow-handled knife
#         From: the wall
#     for sentences that instruct navigation, "Turn left to face the dresser.". the template should be
#         Action: turn
#         Direction: left
#         Preposition: face
#         Landmark: dresser

#     only output the answer in JSON format
#     <</SYS>>

#     Pick up the empty toilet paper roll from the floor. [/INST]

#     """
#     splitsFile = 'data/splits/train_split.json'
#     data = 'data/json_2.1.0'
#     folder = 'templates'
#     # load train/valid/tests splits
#     with open(splitsFile) as f:
#         splits = json.load(f)
#     print("\nPreprocessing dataset and saving to %s folders ... This will take a while. Do this once as required." % folder)
#     preprocess_splits(splits,data,folder)
  

    


    