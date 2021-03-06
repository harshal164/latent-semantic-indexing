import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import os
from fpdf import FPDF


from flask import Flask, request, redirect, render_template, url_for
# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='/home/anupam/exdir/LSI/LSI_FRONT_END')



pd.set_option("display.max_colwidth",200)

print("Training "+(20*'*'))
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers','footers','quotes'))

documents = dataset.data
len(documents)
#print(dataset.target_names)
if False:#multi-line comment
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
            #max_features = 10,
            max_df = 0.5,
            smooth_idf = True)

    X = vectorizer.fit_transform(news_df['clean_doc'])

    print(X.shape)

    svd_model = TruncatedSVD(n_components=6,algorithm='randomized',n_iter=100,random_state=22)

    svd_model.fit(X)
    X_transformed = svd_model.transform(X)
    print(X_transformed)

    with open('vectorizer.pkl','wb') as f:
        pickle.dump(vectorizer,f,pickle.HIGHEST_PROTOCOL)
    with open('svd_model.pkl','wb') as f:
        pickle.dump(svd_model,f,pickle.HIGHEST_PROTOCOL)
    with open('X_transformed.pkl','wb') as f:
        pickle.dump(X_transformed,f,pickle.HIGHEST_PROTOCOL)
    #return "<html><body>'hello'</body></html>"

if True:#multi-line comment
    with open('vectorizer.pkl','rb') as f:
        vectorizer = pickle.load(f)
    with open('svd_model.pkl','rb') as f:
        svd_model = pickle.load(f)
    with open('X_transformed.pkl','rb') as f:
        X_transformed = pickle.load(f)

output = []
def query_matching(q):
    global output
    q_vector = vectorizer.transform([q])
    q_transformed = svd_model.transform(q_vector)

    coses = []
    for j in range(len(X_transformed)):
        ap = np.dot(q_transformed,X_transformed[j])/(np.linalg.norm(q_transformed)*np.linalg.norm(X_transformed[j]))
        if str(ap[0]) == 'nan':
            coses.append(-1)
        else:
            coses.append(ap[0])
        #print("cos(q,doc["+str(j)+"] = "+str(coses[-1]))

    coses = np.asarray(coses)
    p = coses.argsort()
    largest = p[::-1][:20]

    max_num_docs = 8
    print("Top "+str(max_num_docs)+" similar documents are: ")

    output=[]
    #os.mkdir('./output')
    os.chdir('./static/output')
    output.append("<html><body>")
    for i in range(max_num_docs):
        output.append(' <p class="w3-wide">DOCUMENT '+str(i)+" with similarity "+str(coses[largest[i]])+" </p> ")
        #output.append(' <pre> '+documents[largest[i]]+" </pre> <hr>")
        output.append('<embed src="'+url_for('static',filename='output/doc '+str(i)+"["+ str(coses[largest[i]])+'].pdf')+'" type="application/pdf" width="50%" height="80%">')
        output.append("</body></html>")
        name="doc "+str(i)+"["+ str(coses[largest[i]])+"]"

        with open(name,'w') as f:
            f.write("%s\n" %  documents[largest[i]])
            
        pdf = FPDF()    
        
        # Add a page 
        pdf.add_page() 
        
        # set style and size of font  
        # that you want in the pdf 
        pdf.set_font("Arial", size = 15) 
        
        # open the text file in read mode 
        f = open(name, "r") 
        
        # insert the texts in pdf 
        for x in f: 
            pdf.cell(200, 10, txt = x, ln = 1, align = 'C') 
        
        # save the pdf with name .pdf 
        name+=".pdf"
        pdf.output(name)
    os.chdir('../../')

@app.route('/',methods=['GET'])
def root():
    q = request.args.get('q')
    if q == None:
        return render_template('index.html')
    else:
        query_matching(q.lower())
        return redirect("/query")

@app.route('/query')
def query():
    global output
    return ' '.join(output)



if __name__=="__main__":
    app.run(debug=True)




