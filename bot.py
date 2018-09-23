import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random

import pickle
import json





#------------------------------Data Cleaning-----------------------------------
import pandas as pd
from fuzzywuzzy import fuzz
from nltk.stem import WordNetLemmatizer

dataframe = pd.read_excel('Med-TechChatbotDataset.xlsx') 

file = "stopwords.txt"
stoplist = []
with open(file) as f:
    lines = f.readlines()
    lines = [x.strip() for x in lines]
    for line in lines:
        stoplist.append(line)

'''words = []
def applyStopwords(s):
    words = s.split()
    for words not in stoplist:'''
        

dataset = []
for i in dataframe.index:
    j = 1
    symfinal = []
    disease = dataframe.loc[i,'Disease']
    while j<1023:
        if(dataframe.loc[i,'Symptoms.'+str(j)]==dataframe.loc[i,'Symptoms.'+str(j)]):
            symfinal.append(dataframe.loc[i,'Symptoms.'+str(j)])
        else:
            break
        j += 1
        
    dataset.append([disease,symfinal])

dataset_fetch = dataset

word_net = WordNetLemmatizer()
for i in range(len(dataset)):
    sym = []
    for j in dataset[i][1]:
        words = j.split()
        temp = []
        for word in words:
            if word in stoplist:
                words.remove(word)
            else:   
                temp.append(word)
        temp = " ".join(temp)
        sym.append(temp)
    dataset[i][1] = sym
#------------------------------------------------------------------------------







ERROR_THRESHOLD = 0.25

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag %s" % w)
    return np.array(bag)

def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    return random.choice(i['responses'])

            results.pop(0)

data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']



with open('intents.json') as json_data:
    
    print(str(json_data))
    intents = json.load(json_data)
    
    
tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)





fin_sym = []

dis_scr = []



def findDiseases(dataset,fin_sym):
    pred_dis = []
    dis_scr = []
    for i in range(len(dataset)):
        score = 0
        for j in range(len(dataset[i][1])):
            for k in fin_sym:
                if(fuzz.partial_ratio(dataset[i][1][j],k))>60:
                    score+=1/len(dataset[i][1])
                    j+=1
                    if(dataset[i][0] not in pred_dis):
                        pred_dis.append(dataset[i][0])
                    break
        dis_scr.append([dataset[i][0],str(score)])   
    
    return pred_dis,dis_scr
def intersection(list1,list2):
    #lst3 = [value for value in list1 if fuzz.ratio()]

    for value in list1:
        for var in list2:
            if(fuzz.ratio(value,var)>70):
                list1.remove(value)
                list2.remove(var)
    list1 = [value for value in list2]
    return list1

def swap(x,y):
    return y,x
def sort(dis_scr):
    for i in range(len(dis_scr)):
        for j in range(i,len(dis_scr)):
            if (float(dis_scr[i][1])<float(dis_scr[j][1])):
                dis_scr[i],dis_scr[j] = swap(dis_scr[i],dis_scr[j])
  
temp_count = 1
while(True):
    res = input('B: ')
    answer = response(res)
    print (answer)
    print(classify(res)[0][0])
    intent = classify(res)[0][0]
    if intent == 'symptoms' or intent == 'additional_symptoms':
        
        symp = res.split()
        temp = []
        useful_words = [words for words in symp if words.lower() not in stoplist]
        print(useful_words)
        useful_words = " ".join(useful_words)
        fin_sym.append(useful_words)
        countmatch = 0
        pred_dis,dis_scr = findDiseases(dataset,fin_sym)
        sort(dis_scr)
        #ask for more symptoms
        
        print('-----------------------------')
        print('Current prediction status of the model {} and {} '.format(dis_scr[0],dis_scr[1]))
        
        
        
        
        print(temp_count)
        if temp_count % 2 == 0:
            print('B: You could provide me with more symptoms and I would be more confident')
        if temp_count % 1 == 0:
            print(" ")
        if(temp_count % 3==0):
            top_list = []
            #top_list.append(dis_scr[0][0])
            #top_list.append(dis_scr[1][0])
            #top_list.append(dis_scr[2][0])
            for imp in range(len(dataset)):
                if(dataset[imp][0]==dis_scr[0][0]):
                    top_list.append(imp)
                    
                if(dataset[imp][0]==dis_scr[1][0]):
                    top_list.append(imp)
                    
                if(dataset[imp][0]==dis_scr[2][0]):
                    top_list.append(imp)
                
            insec_sym = []
            temp = intersection(dataset_fetch[top_list[0]][1],dataset_fetch[top_list[1]][1])
            insec_sym = [value for value in temp]
            #print(insec_sym)
            temp = intersection(insec_sym,dataset_fetch[top_list[2]][1])
            insec_sym = [value for value in temp]
            #print(insec_sym)
            
            if(len(insec_sym) > 4):
                save_rand_sym = []
                save_rand_sym.append(random.choice(insec_sym))
                save_rand_sym.append(random.choice(insec_sym))
                save_rand_sym.append(random.choice(insec_sym))
                print('Do you also suffer from any one of these symptoms?')
                print(save_rand_sym[0])
                print(save_rand_sym[1])
                print(save_rand_sym[2])
                inp = input('\n1,2,3, or n')
                if inp == '1':
                    for i in range(len(dataset)):
                        if( i in top_list and save_rand_sym[0] in dataset[i][1]):
                            print("You may have {}".format(dataset[i][0]))
                if inp == '2':
                    for i in range(len(dataset)):
                        if( i in top_list and save_rand_sym[1] in dataset[i][1]):
                            print("You may have {}".format(dataset[i][0]))
                if inp == '3':
                    for i in range(len(dataset)):
                        if( i in top_list and save_rand_sym[2] in dataset[i][1]):
                            print("You may have {}".format(dataset[i][0]))
                if inp == 'n':
                    print('You seem to be suffering from {} with score {} or {} with a confidence score {}, please consider going to your doctor'.format(dis_scr[0][0],dis_scr[0][1],dis_scr[1][0],dis_scr[1][1]))
                    
                    

            #else:
             #   print('You seem to be suffering from {} or {}, please consider going to your doctor'.format(dis_scr[0],dis_scr[1]))
                
        temp_count+=1
            
            
        
        
