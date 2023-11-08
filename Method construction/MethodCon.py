import json  
import re
import progressbar
import os
import template
import formalSentence

def simpleMethod(plan,name):
    highLvl = plan["high_pddl"]
    lowLvl = plan["low_actions"]
    methods = []
    
    for highPlan in highLvl: 
        methodName = highPlan['discrete_action']['action']
        if methodName == "NoOp":
            break
        args = highPlan['discrete_action']['args']
        index = highPlan['high_idx']
        
        subtasks = []
        for action in lowLvl:
            if action["high_idx"] == index:
                if "objectId" in action["api_action"]:
                    if "receptacleObjectId"in action["api_action"]:
                        arg1 = action["api_action"]["objectId"]
                        arg1 = re.sub(r'([^a-zA-Z ]+?)', '', arg1)

                        arg2 = action["api_action"]["receptacleObjectId"]
                        arg2 = re.sub(r'([^a-zA-Z ]+?)', '', arg2)
                        s_args = (arg1,arg2)
                        sub = {
                        's_name':action["api_action"]["action"],
                        's_args':s_args
                        }
                        
                        subtasks.append(sub)
                    else:
                        s_args = action["api_action"]["objectId"]
                        s_args = re.sub(r'([^a-zA-Z ]+?)', '', s_args)
                        sub = {
                        's_name':action["api_action"]["action"],
                        's_args':s_args
                        }
                        
                        subtasks.append(sub)
                else:
                    sub = {
                        's_name':action["api_action"]["action"],
                        }
                    subtasks.append(sub)
        
            #outfile.write(makeJSON(methodName,args,subtasks))
        methods.append(makeJSON(methodName,args,subtasks))

    # for m in methods:
    #     print(m)
    
    with open(name, "w") as outfile:
        outfile.write(json.dumps(methods,indent=4))
    

def makeJSON(name,args,subtasks):
    x  ={
        "m_name":name,
        "m_args":args,
        "subtasks":subtasks
        }
    
    #method = json.dumps(x) # ,indent=4
    #print(method)
    return x
    

if __name__ == '__main__':
    splitsFile = '/mnt/Drive2/research/NL2FL/data/splits/train_split.json'
    data = '/mnt/Drive2/research/NL2FL/data/json_2.1.0'
    folder = 'methods'
    # load train/valid/tests splits
    # with open(splitsFile) as f:
    #     splits = json.load(f)

    # for k, d in splits.items():
    #         print('methoding {}'.format(k))

    #         for task in progressbar.progressbar(d):
    #             # load json file
    #             json_path = os.path.join(data, k, task['task'], 'traj_data.json')
    #             with open(json_path) as f:
    #                 file = json.load(f) 
    #                 plan = file['plan']
    #                 taskName = task['task'].split('/', 1)[0]
    #                 name = f"{taskName}.json"
    #                 simpleMethod(plan,name)

    temp = template.templating('traj_data.json')
    print(temp)
    formal = formalSentence.formalize(temp)
    print(formal)
    param = formalSentence.paramterise(formal)
    print(param)

        # simpleMethod(plan) 
    # with open('traj_data.json') as f:
    #     #file = json.load(f)
    #     #f = open(json_path)
    #     data = json.load(f)
    #     tasks = data['turk_annotations']['anns'][0]['task_desc']
    #     #plan = file['plan']
    #     print(tasks)
    #     # simpleMethod(plan)
    


    