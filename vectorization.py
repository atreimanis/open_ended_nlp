import gensim
import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

data_path = 'data/Wiki-proc-p.csv'
data = pd.read_csv(data_path)

#get answer group
def fetch_group(data,group_id):
    answer_group = data.loc[data['Group_NR'] == group_id]
    answers = answer_group['Answer'].values
    #to avoid duplicates
#    answers = answers.remove(deletion)
    return answers

def fetch_query(data,group_id):
    group = data.loc[data['Group_NR'] == group_id]
    queries = group['Sentence'].values
    return queries[0]


def compare(query,*args):
    train_set = [query]
    for array in args:
        for x in array:
            train_set.append(x)

    #find the normalized TD IDF score
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectors= tfidf_vectorizer.fit_transform(train_set)
    #the first element of tfidf_matrix_train is matched with other  elements
    tfidf_similarity = cosine_similarity(tfidf_vectors[0], tfidf_vectors)
    return tfidf_similarity


#clean cos similarity values in the array
def clean_cos_vector(array,columns):
    #similarity is passes as matrix - split into array
    similarity = np.hsplit(array,columns)
    clean_values = []
    for x in similarity:
        value = str(x)
        value = value.replace('[','')
        value = value.replace(']','')
        clean_values.append(value)
    clean_values.pop(0)
    return clean_values


#writing the file
group_id = 1
row_id = 0

#total group num = 940
while group_id <= 940:
    query = fetch_query(data,group_id)
    answer_group = fetch_group(data, group_id)
    comparison = compare(query, answer_group)

    sim_vector = clean_cos_vector(comparison,len(answer_group)+1)

    for similarity in sim_vector:
        data['Similarity_score'][row_id] = float(similarity)
        row_id+=1
    
    group_id+=1

print(data.head(10))

data = data.drop(data.columns[[0]], axis=1) #drop the randomly added id column
data.to_csv('TD-IDF.csv',sep=',', encoding='utf-8')

