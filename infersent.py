import nltk
nltk.download('punkt')

from models import InferSent
import numpy as np
import torch
import csv

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

model_version = 2
MODEL_PATH = "encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else '../fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

model.build_vocab_k_words(K=100000)


inputFile = open('data/Wiki-processed.csv', encoding="utf8")
csvreader = csv.reader(inputFile)
header = []
header = next(csvreader)
outputFile = open('data/infersent.csv', 'w', encoding="utf8", newline='')

writer = csv.writer(outputFile)
writer.writerow(header)

for inputRow in csvreader:
    sentence = inputRow[3]
    answer = inputRow[4]
    print(sentence)
    print(answer)
    inputRow[6] = cosine(model.encode([sentence])[0], model.encode([answer])[0])
    print(inputRow)
    writer.writerow(inputRow)

