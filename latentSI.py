import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

pd.set_option("display.max_colwidth",200)

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers','footers','quotes'))

documents = dataset.data
len(documents)
#print(dataset.target_names)

news_df = pd.DataFrame({'document':documents})

news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z#]", " ")
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())

stop_words = stopwords.words('english')

tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())

tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

detokenized_doc = []

for i in range(len(news_df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

news_df['clean_doc'] = detokenized_doc

vectorizer = TfidfVectorizer(stop_words = 'english',
        #max_features = 10000,
        max_df = 0.5,
        smooth_idf = True)

X = vectorizer.fit_transform(news_df['clean_doc'])

print("Corpus size: ",X.shape)
n_components=int(input("How many topics you want to find?: "))

svd_model = TruncatedSVD(n_components=n_components,algorithm='randomized',n_iter=100,random_state=22)

svd_model.fit(X)

print(len(svd_model.components_))

terms = vectorizer.get_feature_names()

for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key = lambda x:x[1], reverse=True)[:7]
    print("Topic "+str(i)+": ",end="")
    for t in sorted_terms:
        print(t[0],end="")
        print(" ",end="")
    print("")
