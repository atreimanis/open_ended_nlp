import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import csv

tf.compat.v1.disable_eager_execution()
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(module_url)

def get_features(texts):
    if type(texts) is str:
        texts = [texts]
    with tf.compat.v1.Session() as sess:
        sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
        return sess.run(embed(texts))

def cosine_similarity(v1, v2):
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if (not mag1) or (not mag2):
        return 0
    return np.dot(v1, v2) / (mag1 * mag2)

def get_similarity(text1, text2):
    vec1 = get_features(text1)[0]
    vec2 = get_features(text2)[0]
    print(vec1.shape)
    return cosine_similarity(vec1, vec2)

inputFile = open('data/Wiki-processed.csv', encoding="utf8")
csvreader = csv.reader(inputFile)
header = []
header = next(csvreader)
outputFile = open('data/universalencoder.csv', 'w', encoding="utf8", newline='')

writer = csv.writer(outputFile)
writer.writerow(header)


for inputRow in csvreader:
    sentence = inputRow[3]
    answer = inputRow[4]
    print(sentence)
    print(answer)
    inputRow[6] = get_similarity(sentence, answer)
    print(inputRow)
    writer.writerow(inputRow)
