import pickle
import io
import sys

path='./Python'

loaded_model = pickle.load(open(path+'/modelMNB.pkl', 'rb'))
vectorizer = pickle.load(open(path+'/tfidf_vectorizer.pkl', 'rb'))

#Loading text
text=sys.argv[1]

#Saving result
sentiment=loaded_model.predict(vectorizer.transform([text]))[0] #0-negative/1-positive
[number1,number2]=loaded_model.predict_proba(vectorizer.transform([text]))[0,]

if sentiment==0:
    print("--> Sentiment: Negative")
    print("--> Probability: "+str(round(number1*100,2))+'%')
else:
    print("--> Sentiment: Positive")
    print("--> Probability: "+str(round(number2*100,2))+'%')
    