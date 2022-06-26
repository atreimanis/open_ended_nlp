import pandas as pd
from sentence_transformers import SentenceTransformer, util
from td_idf_vectorization import fetch_group, fetch_query

data_path = 'data/Wiki-proc-p.csv'
data = pd.read_csv(data_path)

def bert_encoding():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    #writing the file
    group_id = 1
    row_id = 0

    #total group num = 940
    while group_id <= 940:
        query = fetch_query(data,group_id)
        answer_group = fetch_group(data, group_id)

        #Encode all sentences
        embeddings = model.encode(answer_group)
        query_emb = model.encode(query)
        #Compute cosine similarity between all pairs
        cos_sim = util.cos_sim(query_emb, embeddings)
        #convert to array from tensor class
        sim_values = cos_sim.numpy()


        for similarity in sim_values[0]:
            #restrict float precision for performance 
            float_sim = float(similarity)
            float_sim = "{:.8f}".format(float_sim) #same as TD-IDF precision

            data['Similarity_score'][row_id] = float_sim
            row_id+=1


        if group_id%100 == 0: print(group_id, " groups compared")    
        group_id+=1

    print(data.head(10))

    data = data.drop(data.columns[[0]], axis=1) #drop the randomly added id column
    data.to_csv('sentenceBERT.csv',sep=',', encoding='utf-8')



#analysis
td_data = pd.read_csv('data/TD-IDF.csv')
bert_data = pd.read_csv('data/sentenceBERT.csv')

td_data_values = td_data[['Label','Similarity_score']]
bert_data_values = bert_data[['Label','Similarity_score']]

def get_stats(data):
    grouped = data.groupby('Label')['Similarity_score']
    print('max values: ',grouped.max())
    print('min values: ',grouped.min())
    print('average values: ',grouped.mean())
    print('median values: ',grouped.median())

    negative = data.loc[data['Label'] == 0]
    positive = data.loc[data['Label'] == 1]
    print('false positives:', negative[negative.Similarity_score >= 1].count())
    print('false negatives:', positive[positive.Similarity_score <= 0].count())

get_stats(td_data_values)