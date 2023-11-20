import json
import numpy as np
from llama_cpp import Llama

#pick_clean_then_place_in_recep-Bowl-None-Microwave-23.json
#pick_heat_then_place_in_recep-Mug-None-CoffeeMachine-18.json
#pick_heat_then_place_in_recep-Tomato-None-Fridge-23.json

file = 'pick_clean_then_place_in_recep-Bowl-None-Microwave-23.json'

#test with & without "M_name" and stuff

def makeSen(task):
    sentence=""
    for x,y in task.items():
        if x == 'm_args' :
            sentence+= x+ ": "
            for word in y:
                sentence += word+ " "
        elif x =='s_args':
            if str(type(y)) ==  "<class 'list'>":
                sentence+= x+ ": "
                for word in y:
                    sentence += word+ " "
            else:
                sentence += x+": "+y+" "

        elif x == "subtasks":
            sentence+= x+ ": "
            for sub in y:
                sentence+= makeSen(sub)
        else:
            sentence += x+": "+y+" "

    return sentence


with open(file) as f:
    data = json.load(f)
    sentence = makeSen(data)
    print(sentence.strip())

