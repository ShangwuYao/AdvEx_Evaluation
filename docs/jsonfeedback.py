import json

"""
 
 Robustness Score calculated by taking an average of the accuracy degrade for each of the attack_methods. 
 Lesser the score, higher the robustness of the user's model.
 Scale for Robustness Score (RS) is:
    Very Good : RS < 5
    Good : 5 <= RS < 10
    Average: 10 <= RS < 20 
    Bad: 20 <= RS < 30
    Very Bad: 30 <= RS
"""

data = {}  
data['robustness'] = '9'
data['rating']= 'Good'
data['details'] = []

data['details'].append({  
    'attack_method': 'CLEAN',
    'accuracy': '80.05%',
    'confidence': '95%'
})
data['details'].append({  
    'attack_method': 'FGSM',
    'accuracy': '80.05%',
    'confidence': '95%'
})
data['details'].append({  
    'attack_method': 'MI-FGSM',
    'accuracy': '92.10%',
    'confidence': '91%'
})


data['details'].append({  
    'attack_method': 'I-FGSM',
    'accuracy': '94.10%',
    'confidence': '93.7%'
})

data['graph_link'] = 'https://www.smartdraw.com/bar-graph/'
data['suggestion'] = 'Your model can be made more robust by training it with some of the adversarial examples which you can download for free from your dashboard.'  

#data['detailed attack_results']
with open('jsonfeedback.txt', 'w') as outfile:  
    j=json.dumps(data, indent=4)
    #print(j)
    outfile.write(j)
