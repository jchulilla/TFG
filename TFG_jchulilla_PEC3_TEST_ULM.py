import io
import fastai
from fastai.text import *
import sys,os

#Turning off stdout,stderr
_stderr = sys.stderr
_stdout = sys.stdout
null = open(os.devnull,'wb')
sys.stdout = sys.stderr = null

#Loading model
path='./Python'
learn = load_learner(path,'learn.pkl')

#Turning on stdout,stderr
sys.stderr = _stderr
sys.stdout = _stdout

#Loading text
text=sys.argv[1]

#Saving result
result = str(learn.predict(text))

sentiment=result[17] #0-negative/1-positive
numbers=result[41:-1].replace(')','').replace(']','').replace(',','').replace(' ','\n')
buffer=io.StringIO(numbers)
number1=float(buffer.readline())
number2=float(buffer.readline())

if sentiment=="0":
    print("--> Sentiment: Negative")
    print("--> Probability: "+str(round(number1*100,2))+'%')
else:
    print("--> Sentiment: Positive")
    print("--> Probability: "+str(round(number2*100,2))+'%')