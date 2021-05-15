#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
from time import time
import pandas as pd
import warnings

import seaborn as sns  # Plot utility
from sklearn.utils import shuffle # to shuffle the data    

import re
import contractions

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
#from nltk import word_tokenize

import inflect

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


#######################################
### DATA PRINTING/LOADING FUNCTIONS ###
#######################################

def create_res():
    result = pd.DataFrame(columns=['Multinomial Naive Bayes', 'Logistic Regression','ULMFiT'],
                         index=['Tiempo modelo(seg.)',
                                'Matriz confusion',
                                'Accuracy'])
    return result

def writecsv (text,filename):
    with open(filename, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for t in text:
            writer.writerow(text)

def read_data(fileName,rows_to_process):
    """Loads and prepares the data in fileName.
    - Columns:
        text: the review of the movie 
        label : the sentiment label of the movie review
    """
    # Load CSV file
    sampleData=pd.read_csv(fileName)
    sampleData = shuffle(sampleData, random_state=123)
    textData = sampleData.iloc[0:rows_to_process,0]
    targetData = sampleData.iloc[0:rows_to_process,1]

    # Return the standardized data and the truth
    return [textData,targetData]


# In[3]:


#####################
### Lookup Data   ###
#####################

def lookup_data(data):
    VSHORT = 3
    SHORT = 50
    NORMAL = 250
    LARGE = 1000
    VLARGE = 10000
    
    print('########## Resume   #################')
    print(data.head())
    print(' ')
    print('########## Describe #################')
    print(data.describe())
    print(' ')
    
    print('########## Nulls    #################')
    print(data.isnull().sum())
    print(' ')
    
    
    if data.name == 'text': 
        print('########## Max Length   #################')
        print(max(len(x) for x in data))
        print(' ')

        print('########## Plot #################')  
        #rawDataTrain.loc[len(rawDataTrain['text']) <= 3, '<=3'] = 'True' 
        test = data.apply(lambda x: 'V.Short' if len(x) <= VSHORT 
                          else ('Short' if len(x) > VSHORT and len(x)<=SHORT 
                                else ('Normal' if len(x) >SHORT and len(x)<=NORMAL 
                                      else ('Large' if len(x) > NORMAL and len(x) <= LARGE 
                                            else ('Very Large' if len(x) > LARGE and len(x) <= VLARGE 
                                                  else 'Extra large')))))
        print(test.value_counts())
        print(' ')
        sns.countplot(test)
    elif data.name == 'label': 
        print('###########Plot ################')  
        print(data.value_counts())
        print(' ')
        sns.countplot(data)
    


# In[4]:


###########################
### Preprocessing Data  ###
###########################

def preprocess_text(text):
    """Basic cleaning of texts."""
    # Remove http links
    text=re.sub(r'http\S+',' ', str(text))

    # remove html markup
    text=re.sub('(<.*?>)',' ',str(text))

    # remove between square brackets
    text=re.sub('\[[^]]*\]', ' ', text)

    #remove non-ascii
    text=re.sub('[^\x00-\x7F]',' ',str(text))
    
    #remove hyphen not between characters
    text=re.sub('(-[^a-zA-Z0-9])',' ',str(text))

    #remove whitespace
    text=text.strip()

    #lowercase
    for f in re.findall("([A-Z]+)", text):
        text = text.replace(f, f.lower())
    
    #Replace contractions
    text= contractions.fix(str(text)) 

    return text
    


# In[5]:


###########################
### Vocabulary          ###
###########################

def remove_short_words(words):
    words_ok = []
    for word in words:
        if len(word) > 1: #Only words with size>1
            words_ok.append(re.sub('[^A-Za-z]+','',str(word)))
    return words_ok

def replace_numbers(words):
    p = inflect.engine()
    words_ok = []
    for word in words:
        try:
            if word.isdigit():
                words_ok.append(p.number_to_words(word))
            else:
                words_ok.append(word)
        except:
            print('[ERROR: word numOutOfRangeError...]'+str(word))#special cases with large numbers
            pass
    return words_ok

def stop_words(words):
    stopWords = set(stopwords.words('english'))
    words_ok = []
    for w in words:
        if w not in stopWords:
            words_ok.append(w)
    return words_ok

def lemmatize(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemmas.append(lemmatizer.lemmatize(word))
    return lemmas

def stemming(words):
    """Stem words in list of tokenized words"""
    stemmer = PorterStemmer()
    stems = []
    for word in words:
        stems.append(stemmer.stem(word))
    return stems

def tokenize(words):
    return nltk.word_tokenize(words)

def preprocess_words(words):
    temp=remove_short_words(words)
    temp=replace_numbers(words)
    temp=stop_words(words)
    return temp


# In[6]:


###########################
### Vectorization       ###
###########################

def vectorization (Train,Test):
    #Transform from words to full sentences
    print('Vectorization.....')
    cleanedTrain = [] 
    cleanedTest = []
    for words in Train:
        cleanedTrain.append(' '.join(words))
    for words in Test:
        cleanedTest.append(' '.join(words))

    #Create tf-idf data
    tfidf_vectorizer = TfidfVectorizer()
    tfidfVect = tfidf_vectorizer.fit(cleanedTrain+cleanedTest)
    
    return [tfidfVect.transform(cleanedTrain), tfidfVect.transform(cleanedTest)]


# In[11]:


#######################
### Classification  ###
#######################

def classification(algor,Train,Test,Y_train,Y_test):
    time_ini = time()
    if algor == 'MNB':
        model = MultinomialNB(alpha=0.3)
    elif algor == 'LR':
        model = LogisticRegression(multi_class='multinomial')

    model = model.fit(Train,Y_train)
    time_fin = time()

    y_pred = model.predict(Test)
    y_accu = model.score(Test,Y_test)

    #Store results
    #result.iat[0,0] = time_fin - time_ini
    #result.iat[1,0] = confusion_matrix(Y_test, y_pred)
    #result.iat[2,0] = "{0:0%}".format(y_accu)
    print("#######################")
    if algor == 'MNB':
        print('MULTINOMIAL NAIVE BAYES')
    elif algor == 'LR':
        print('LOGISTIC REGRESSION')
    print("#######################")

    print("Total time....","{0:.{1}f}".format(time_fin - time_ini,2),"secs")
    print("Confusion matrix....")
    print(confusion_matrix(Y_test, y_pred))
    print("Accuracy....","{0:.{1}%}".format(y_accu,2))
    print(classification_report(Y_test, y_pred))
    

    


# In[12]:


#####################
### Run process   ###
#####################

def run():

    #Initial time mark
    time_ini = time()

    #Supress warning messages on output
    warnings.filterwarnings('ignore')
    ROWS_TO_PROCESS = 1000000  #FOR TESTING PURPOSES!!!!!!!!!!!!

    # Loading train data
    #[rawData,targetData]=read_data('/content/drive/My Drive/tweet_dataset.csv',ROWS_TO_PROCESS)
    [rawDataTrain,targetDataTrain]=read_data('Train.csv',ROWS_TO_PROCESS)
    
    # Loading test data
    #[rawData,targetData]=read_data('/content/drive/My Drive/tweet_dataset.csv',ROWS_TO_PROCESS)
    [rawDataTest,targetDataTest]=read_data('Test.csv',ROWS_TO_PROCESS)
    
    # Checking data
    #lookup_data(rawDataTrain) #FOR INITIAL CHECKINGS!!!!!!!!!!!!
    #lookup_data(targetDataTrain) #FOR INITIAL CHECKINGS!!!!!!!!!!!!
    #lookup_data(rawDataTest) #FOR INITIAL CHECKINGS!!!!!!!!!!!!
    #lookup_data(targetDataTest) #FOR INITIAL CHECKINGS!!!!!!!!!!!!
    
    #Preprocessing text
    print('Preprocessing text.....')
    cleanDataTrain=rawDataTrain.apply(preprocess_text)
    cleanDataTest=rawDataTest.apply(preprocess_text)
        
    #Obtaining vocabulary 
    #Tokenize
    print('Tokenizing.....')  
    tokenizedDataTrain=cleanDataTrain.apply(tokenize)
    tokenizedDataTest=cleanDataTest.apply(tokenize)
    #Cleaning words
    print('Preprocessing words.....')
    cleanwDataTrain=tokenizedDataTrain.apply(preprocess_words)
    cleanwDataTest=tokenizedDataTest.apply(preprocess_words)
    #Lemmatize/Stemming
    print('Lemmatizing.....')
    vocabularyDataTrain=cleanwDataTrain.apply(lemmatize)
    vocabularyDataTest=cleanwDataTest.apply(lemmatize)
    #print('Stemming.....')
    #vocabularyDataTrain=cleanwDataTrain.apply(stemming)
    #vocabularyDataTest=cleanwDataTest.apply(stemming)

    #Vectorization - Tf-idf
    [tfidfVectDataTrain,tfidfVectDataTest] = vectorization(vocabularyDataTrain,vocabularyDataTest)

    #Final time mark (before modelling)
    time_fin = time()
    print("Total time....","{0:.{1}f}".format(time_fin - time_ini,2),"secs")
    
    #Standard modeling
    classification('MNB',tfidfVectDataTrain,tfidfVectDataTest,targetDataTrain,targetDataTest)
    classification('LR',tfidfVectDataTrain,tfidfVectDataTest,targetDataTrain,targetDataTest)
    
    


# In[13]:


run()

