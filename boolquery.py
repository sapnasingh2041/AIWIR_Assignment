from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
import time

def intersect(p1,p2):
    res=[]
    i,j=0,0
    while(i<len(p1) and j<len(p2)):
        if(p1[i]==p2[j]):
            res.append(p1[i])
            i+=1
            j+=1
        elif p1[i]>p2[j]:
            j+=1
        else:
            i+=1
    return res

def postunion(p1,p2):
    res=[]
    res=sorted(list(set().union(p1,p2)))
    return res

def notquery(dfsize,p1):
    res=[]
    for i in range(dfsize):
        if i not in p1:
            res.append(i)
    return res
    
df=pd.read_csv('tweet_emotions.csv')
stop_words=set(stopwords.words('english'))
df['texttoken']=df['content'].apply(word_tokenize)
df['texttoken']=df['texttoken'].apply(lambda words:[word.lower() for word in words if word.isalpha()])
df['stop_remd']=df['texttoken'].apply(lambda x:[item for item in x if item not in stop_words])
st_time=time.time()
post_list={}
for postext,text in enumerate(df['stop_remd']):
    for pos,term in enumerate(text):
        if term not in post_list.keys():
            post_list[term]=[]
        post_list[term].append(postext)
for i in list(post_list)[:20]:
    print(i,post_list[i])
end_time=time.time()-st_time
f=open("benchmark.txt",'a')
f.write("=====Tweet Emotions Dataset=====\n")
f.write("Time taken to build inverted index: "+str(end_time)+"\n")

def querysearch(inp,post_list):
    if len(inp)==3:
        if inp[1].lower() == 'and':
            print(intersect(post_list[inp[0]],post_list[inp[2]]))
        elif inp[1].lower() == 'or':
            print(postunion(post_list[inp[0]],post_list[inp[2]]))
        else:
            print("Invalid query")
    elif len(inp)==4:
        if inp[0].lower() == 'not' and inp[2].lower() == 'and':
            print(intersect(notquery(df.shape[0],post_list[inp[1]]),post_list[inp[3]]))
        elif inp[0].lower() == 'not' and inp[2].lower() == 'or':
            print(postunion(notquery(df.shape[0],post_list[inp[1]]),post_list[inp[3]]))
        elif inp[2].lower() == 'not' and inp[1].lower() == 'and':
            print(intersect(notquery(df.shape[0],post_list[inp[3]]),post_list[inp[0]]))
        elif inp[2].lower() == 'not' and inp[1].lower() == 'or':
            print(postunion(notquery(df.shape[0],post_list[inp[3]]),post_list[inp[0]]))
        else:
            print("Invalid query")
    elif len(inp)==5:
        if inp[2].lower()=='and':
            p1=notquery(df.shape[0],post_list[inp[1]])
            p2=notquery(df.shape[0],post_list[inp[4]])
            print(intersect(p1,p2))
        elif inp[2].lower()=='or':
            print(postunion(notquery(df.shape[0],post_list[inp[1]]),notquery(df.shape[0],post_list[inp[4]])))
        else:
            print("Invalid query")
    else:
        print("Invalid query")

inp=input("Enter query:").split()
st_time=time.time()
querysearch(inp,post_list)
end_time=time.time()-st_time
f.write("Time taken to search: "+str(end_time)+"\n")
f.close()