from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import time


df=pd.read_csv('shopee_reviews.csv',usecols=["text"])
stop_words=set(stopwords.words('english'))
df=df.dropna()
df['texttoken']=df['text'].apply(word_tokenize)
df['texttoken']=df['texttoken'].apply(lambda words:[word.lower() for word in words if word.isalpha()])
df['stop_remd']=df['texttoken'].apply(lambda x:[item for item in x if item not in stop_words])

ps=PorterStemmer()
df['stemmed']=df['stop_remd'].apply(lambda x:[ps.stem(item) for item in x])


idx_dict={}
st_time=time.time()
for postext,text in enumerate(df['stemmed']):
    for pos,term in enumerate(text):
        if term not in idx_dict.keys():
            idx_dict[term]=[0,{}]
        idx_dict[term][0]+=1
        if postext not in idx_dict[term][1].keys():
            idx_dict[term][1][postext]=[]
        idx_dict[term][1][postext].append(pos)
end_time=time.time()-st_time
f=open("benchmark.txt",'a')
f.write("=====Shopee Review Dataset=====\n")
f.write("Time taken to build inverted index: "+str(end_time)+"\n")
print("Term [Frequency,Entry number:[Positions in that entry]]")
for i in list(idx_dict)[:10]:
    print(i,idx_dict[i])

def search(t1,t2,prox,idx_dict):
    res=[]
    if t1 not in idx_dict.keys() or t2 not in idx_dict.keys():
        print("Query does not exist")
    else:
        idx1=idx_dict[t1][1]
        idx2=idx_dict[t2][1]
        interset=set(idx1.keys()).intersection(idx2.keys())
        for i in interset:
            l1=idx1[i]
            l2=idx2[i]
            for j in l1:
                if (any(x in l2 for x in range(j-prox,j+prox+1))):
                    if i not in res:
                        res.append(i)
    if len(res)==0:
        print('Query does not exist')
    return res

inp=input("Enter query with query proximity separated by spaces:").split()
prox=1
if(len(inp)==3):
  prox=int(inp[1][1:])
  inp.pop(1)
inp=[ps.stem(i) for i in inp]
st_time=time.time()
res=search(inp[0],inp[1],prox,idx_dict)
end_time=time.time()-st_time
f.write("Time taken to search for phrase: "+str(end_time)+"\n")
f.close()
print(res)
if len(res)!=0:
    for i in res:
        print(i,"\t",df['text'].iloc[i])

