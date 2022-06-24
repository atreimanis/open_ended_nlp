import gensim
import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity



data_path = 'data/Wiki-proc-p.csv'
data = pd.read_csv(data_path)
corpus_a_large = data['Answer'].values #documents
corpus_a = corpus_a_large[0:100]

corpus_s_large = data['Sentence'].values #query
corpus_s = corpus_s_large[0:100]


train_set = [corpus_s[0],corpus_a[4],corpus_a[5],corpus_a[3],corpus_a[2]]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)  #finds the tfidf score with normalization
 #here the first element of tfidf_matrix_train is matched with other  elements
#print("cosine scores ==> ",cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train)) 


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

#print(fetch_group(data,2))
#print(fetch_query(data,2))


def compare(query,*args):
    train_set = [query]
    for array in args:
        for x in array:
            train_set.append(x)

    tfidf_vectors= tfidf_vectorizer.fit_transform(train_set)
    tfidf_similarity = cosine_similarity(tfidf_vectors[0], tfidf_vectors)
    return tfidf_similarity


#clean similarity values in the array
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

cos_data = data
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

print(cos_data.head(10))

data = data.drop(data.columns[[0]], axis=1)
data.to_csv('TD-IDF.csv',sep=',', encoding='utf-8')



def write_file_TFIDF(output_path):
    inputFile = open('data/Wiki-proc-p.csv',encoding="utf8")
    csvreader = csv.reader(inputFile)
    header = []
    header = next(csvreader)
    outputFile = open(output_path, 'w', encoding="utf8", newline='')
    # create the csv writer
    writer = csv.writer(outputFile)
    writer.writerow(header)
    group_id = 1

    for inputRow in csvreader:
        if group_id > 940 : break
        query = inputRow[2]
        group_num = inputRow[1]
        answer = inputRow[3]
        answer_group = fetch_group(data, group_num,answer)
        compare(query, answer, answer_group)

        writer.writerow(outputRow)

    # close the file
    outputFile.close()