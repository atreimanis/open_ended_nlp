import gensim
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



data_path = 'data/Wiki-proc-p.csv'
data = pd.read_csv(data_path)
corpus_a_large = data['Answer'].values #documents
corpus_a = corpus_a_large[0:100]

corpus_s_large = data['Sentence'].values #query
corpus_s = corpus_s_large[0:100]

#prev
# settings that you use for count vectorizer will go here 
#tfidf_vectorizer=TfidfVectorizer(use_idf=True) 
#tfidf_vector_ans=tfidf_vectorizer.fit_transform(corpus_a)
#tfidf_vector_sen = tfidf_vectorizer.fit_transform(corpus_s)



#train_set = [corpus_s[0],corpus_a[5],corpus_a[4],corpus_a[3],corpus_a[2],corpus_a[1],corpus_a[0]]
train_set = [corpus_s[0],corpus_a[5]]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)  #finds the tfidf score with normalization
 #here the first element of tfidf_matrix_train is matched with other  elements
print("cosine scores ==> ",cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train)) 


#get question group
def fetch_group(data,group_id):
    answer_group = data.loc[data['Group_NR'] == group_id]
    answers = answer_group['Answer'].values
    return answers

#print(fetch_group(data,2))

def compare(query,answer):
    train_set = [query, answer]
    tfidf_vectors= tfidf_vectorizer.fit_transform(train_set)
    tfidf_similarity = cosine_similarity(tfidf_vectors[0], tfidf_vectors)
    return tfidf_similarity

print("cosine scores => ", compare(corpus_s[0],corpus_a[5]))