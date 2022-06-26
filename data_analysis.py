import pandas as pd

#all csv paths 
td_path = 'data/TD-IDF.csv'
bert_path = 'data/sentenceBert.csv'
vec_path = 'data/sentence2vec.csv'
infersent_path = 'data/infersent.csv'

#data to input

def get_values(path):
    main_data = pd.read_csv(path)
    values = main_data[['Label','Similarity_score']] 
    negative = values.loc[values['Label'] == 0] #all answers labeled as wrong
    positive = values.loc[values['Label'] == 1] #all answers labeled as correct
    print('data fetched')
    return positive,negative



def get_stats(path):
    positive, negative = get_values(path)
    print('Dataset used: ', path)

    print('min values pos: ',positive['Similarity_score'].min())
    print('min values neg: ',negative['Similarity_score'].min())

    print('max values pos: ',positive['Similarity_score'].max())
    print('max values neg: ',negative['Similarity_score'].max())

    print('avg values pos: ',positive['Similarity_score'].mean())
    print('avg values neg: ',negative['Similarity_score'].mean())

    print('median values pos: ',positive['Similarity_score'].median())
    print('median values neg: ',negative['Similarity_score'].median())


    #print('false positives:', negative[negative.Similarity_score >= 1].count())
    #print('false negatives:', positive[positive.Similarity_score <= 0].count())

def get_precision(path):
    positive, negative = get_values(path)
    print('Dataset used: ', path)

    print('positives with high score:', positive[positive.Similarity_score >=0.9].count())
    print('negatives with high score:', negative[negative.Similarity_score >=0.9].count())
    print('positives with low score:', positive[positive.Similarity_score <0.5].count())

get_precision(infersent_path)
