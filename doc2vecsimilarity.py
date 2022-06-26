import csv
import spacy
from sent2vec.vectorizer import Vectorizer

inputFile = open('data/Wiki-processed.csv', encoding="utf8")
csvreader = csv.reader(inputFile)
header = []
header = next(csvreader)
outputFile = open('data/sentence2vec.csv', 'w', encoding="utf8", newline='')

writer = csv.writer(outputFile)
writer.writerow(header)

counter = 0

nlp = spacy.load("en_core_web_lg")

for inputRow in csvreader:
    sentence = inputRow[3]
    answer = inputRow[4]
    print(sentence)
    print(answer)
    sentenceDoc = nlp(sentence)
    answerDoc = nlp(answer)
    inputRow[6] = sentenceDoc.similarity(answerDoc)
    print(inputRow)
    writer.writerow(inputRow)