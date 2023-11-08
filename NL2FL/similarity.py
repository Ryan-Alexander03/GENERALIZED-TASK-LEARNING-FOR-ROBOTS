import json  
import numpy as np

f = open('templates2.json')
# returns JSON object as
# a dictionary
data = json.load(f)

    

def SIM(temp1,temp2, gamma):
    print(SAN(temp1,temp2))
    print("\n")
    print(SSL(temp1,temp2))
    print("\n")
    return np.add(gamma*SAN(temp1,temp2),(1-gamma)*SSL(temp1,temp2))


def SAN(temp1,temp2): # Action similarity
    action1 = temp1['Action']
    action2 = temp2['Action']
    return levenshteinDistanceDP(action1, action2)

def SSL(temp1,temp2): # The average similarity for each non-action word
    #compound sentences

    t1Keys = temp1.keys()
    t2Keys = temp2.keys()

    # Action to action  
    my_list = ['Object','Location']
    common_keys = [key for key in temp1 if key in my_list]

    if t1Keys == t2Keys and common_keys !=[]:
        
        object1 = temp1['Object']
        object2 = temp2['Object']
        obDiff = levenshteinDistanceDP(object1, object2)

        location1 = temp1['Location']
        location2 = temp2['Location']
        locDiff = levenshteinDistanceDP(location1, location2)
        
        return (np.add(obDiff,locDiff))
    
    #nav to nav
    my_list = ['Direction','Preposition','Landmark']
    common_keys = [key for key in temp1 if key in my_list]

    if t1Keys == t2Keys and common_keys !=[]:
        
        Direction1 = temp1['Direction']
        Direction2 = temp2['Direction']
        dirDiff = levenshteinDistanceDP(Direction1, Direction2)

        Preposition1 = temp1['Preposition']
        Preposition2 = temp2['Preposition']
        PrepDiff = levenshteinDistanceDP(Preposition1, Preposition2)

        Landmark1 = temp1['Landmark']
        Landmark2 = temp2['Landmark']
        LandmarkDiff = levenshteinDistanceDP(Landmark1, Landmark2)
        
        return (np.add(np.add(dirDiff,PrepDiff),LandmarkDiff))

    #action to nav
    my_list = ['Action','Object','Location']
    common_keys = [key for key in temp1 if key in my_list]

    if common_keys !=[]:
        object1 = temp1['Object']
        Direction2 = temp2['Direction']
        otherDiff = levenshteinDistanceDP(object1, Direction2)

        location1 = temp1['Location']
        Preposition2 = temp2['Preposition']
        Landmark2 = temp2['Landmark']
        location2 = Preposition2 + Landmark2
        LocDiff = levenshteinDistanceDP(location1, location2)

        return (np.add(otherDiff,LocDiff))
    else:
        object1 = temp2['Object']
        
        Direction2 = temp1['Direction']
        otherDiff = levenshteinDistanceDP(object1, Direction2)

        location1 = temp2['Location']
        Preposition2 = temp1['Preposition']
        Landmark2 = temp1['Landmark']
        location2 = Preposition2 + Landmark2
        LocDiff = levenshteinDistanceDP(location1, location2)

        return (np.add(otherDiff,LocDiff))

def levenshteinDistanceDP(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

   # printDistances(distances, len(token1), len(token2))
   
    return distances

def printDistances(distances, token1Length, token2Length):
    for t1 in range(token1Length + 1):
        for t2 in range(token2Length + 1):
            print(int(distances[t1][t2]), end=" ")
        print()

sentences = data['sentences']
temp1 = sentences[0]['sentence']
temp2 = sentences[2]['sentence']
print(temp2)
print(SIM(temp1,temp2,0.5))
