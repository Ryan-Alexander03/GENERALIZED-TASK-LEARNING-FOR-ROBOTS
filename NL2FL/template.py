import json  
# load the large language model file
from llama_cpp import Llama
import os
import json
import revtok
import torch
import copy
import progressbar


def preprocess_splits(splits,data,folder):
        '''
        saves preprocessed data as jsons in specified folder
        '''
        for k, d in splits.items():
            print('Preprocessing {}'.format(k))
           
            for task in progressbar.progressbar(d):
                # load json file
                json_path = os.path.join(data, k, task['task'], 'traj_data.json')
                with open(json_path) as f:
                    ex = json.load(f)

                # # copy trajectory
                r_idx = task['repeat_idx'] # repeat_idx is the index of the annotation for each trajectory
                traj = ex.copy()

                # root & split
                traj['root'] = os.path.join(data, task['task'])
                traj['split'] = k
                traj['repeat_idx'] = r_idx


                # check if preprocessing storage folder exists
                preprocessed_folder = os.path.join(data, task['task'])
                if not os.path.isdir(preprocessed_folder):
                    os.makedirs(preprocessed_folder)

                # save preprocessed json
                preprocessed_json_path = os.path.join(preprocessed_folder, "template_%d.json" % r_idx)
                templating(json_path,preprocessed_json_path)
                #with open(preprocessed_json_path, 'w') as f:
                
                    # json.dump(traj, f, sort_keys=True, indent=4)


def templating(json_path,file):
    # returns JSON object as
    # a dictionary
    f = open(json_path)
    data = json.load(f)
    
    tasks = data['turk_annotations']['anns']

    for i, task in enumerate(tasks):
        sentences = task['high_descs']
        output = LLM(SysPrompt)
        input = SysPrompt
        #file = f
        #"templates"+str(i)+".json"

        with open(file, "w") as text_file:
            text_file.write('{"sentences":[')
            text_file.write("\n")
            
            for index, sen in enumerate(sentences):
                #si='{"sentence'+str(index)+'":['
                

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
                    
            
            text_file.write("]}")



if __name__ == '__main__':
    LLM = Llama(model_path="/mnt/Drive2/GENERALIZED-TASK-LEARNING-FOR-ROBOTS/llama.cpp/models/llama-2-13b-chat/ggml-model-q4_0.bin",n_ctx=4000) # context length of 4000, LLama 2's max


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
    splitsFile = 'data/splits/train_split.json'
    data = 'data/json_2.1.0'
    folder = 'templates'
    # load train/valid/tests splits
    with open(splitsFile) as f:
        splits = json.load(f)
    print("\nPreprocessing dataset and saving to %s folders ... This will take a while. Do this once as required." % folder)
    preprocess_splits(splits,data,folder)
  

    


    