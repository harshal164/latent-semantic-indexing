import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

docs = ["the house had a tiny little mouse",
        "the cat saw the mouse",
        "the mouse ran away from the house",
        "the cat finally ate the mouse",
        "the end of mouse story"
        ]

def tfidftransformer():
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(docs)

    print(word_count_vector.shape)

    tfidf_transformer = TfidfTransformer(smooth_idf = True, use_idf = True)
    tfidf_transformer.fit(word_count_vector)

    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])

    df_idf.sort_values(by=['idf_weights'])

    print(df_idf)

    count_vector = cv.transform(docs)
    tf_idf_vector = tfidf_transformer.transform(count_vector)

    feature_names = cv.get_feature_names()
    first_doc_vector = tf_idf_vector[0]

    df = pd.DataFrame(first_doc_vector.T.todense(), index=feature_names, columns=['tfidf'])
    df.sort_values(by=['tfidf'],ascending=False)

    print(df)

def tfidfvectorizer():
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(docs)

    first_vector = tfidf_vectorizer_vectors[0]
    df = pd.DataFrame(first_vector.T.todense(),index=tfidf_vectorizer.get_feature_names(), columns = ['tfidf'])
    df.sort_values(by=['tfidf'],ascending = False)

    print(df)

tfidftransformer()
tfidfvectorizer()
