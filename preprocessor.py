import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
import contractions
import csv
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def pre_process(corpus):
    # convert input corpus to lower case.
    corpus = corpus.lower()

    #expand contractions
    expanded_input = []

    for word in corpus.split():
        expanded_input.append(contractions.fix(word))

    expanded_text = ' '.join(expanded_input)
    # collecting a list of stop words from nltk and punctuation form
    # string class and create single array.
    stopset = stopwords.words('english') + list(string.punctuation)
    # remove stop words and punctuations from string.
    # word_tokenize is used to tokenize the input corpus in word tokens.
    corpus = " ".join([i for i in word_tokenize(expanded_text) if i not in stopset])
    # remove non-ascii characters
    corpus = unidecode(corpus)

    lemmatizer = WordNetLemmatizer()

    corpus = word_tokenize(corpus)

    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in corpus])

    return lemmatized_output

#clean answer column
def remove_punctuation(path):
    inputFile = open(path,encoding="utf8")
    csvreader = csv.reader(inputFile)
    header = []
    header = next(csvreader)
    outputFile = open('data/Wiki-proc-p.csv', 'w', encoding="utf8", newline='')
    # create the csv writer
    writer = csv.writer(outputFile)
    writer.writerow(header)

    for inputRow in csvreader:
        outputRow = inputRow

        outputRow[4] = inputRow[4].replace('`', '')
        outputRow[4] = inputRow[4].replace('-', '')
        outputRow[4] = inputRow[4].replace('"', '')
        outputRow[4] = inputRow[4].replace('\'', '')
        outputRow[4] = inputRow[4].replace('...', '')

        writer.writerow(outputRow)

    # close the file
    outputFile.close()


def writefile_1():
    inputFile = open('data/WikiQA.csv', encoding="utf8")
    csvreader = csv.reader(inputFile)
    header = []
    header = next(csvreader)
    outputFile = open('data/WikiQA-processed.csv', 'w', encoding="utf8", newline='')
    # create the csv writer
    writer = csv.writer(outputFile)
    writer.writerow(header)
    counter = 0
    for inputRow in csvreader:
        outputRow = inputRow
        outputRow[5] = pre_process(inputRow[5])
        print(outputRow)
        writer.writerow(outputRow)
        counter = counter+1
        if counter == 1000:
            break

    # close the file
    outputFile.close()

def writefile_2(path):
    inputFile = open(path, encoding="utf8")
    csvreader = csv.reader(inputFile)
    header = []
    header = next(csvreader)
    outputFile = open('data/Wiki-proc.csv', 'w', encoding="utf8", newline='')
    # create the csv writer
    writer = csv.writer(outputFile)
    writer.writerow(header)

    for inputRow in csvreader:
        #only process rows with filled answer phrases
        if inputRow[1] != '':
            outputRow = inputRow
            #process answer phrase
            outputRow[1] = pre_process(inputRow[1])
            #process answer
            outputRow[2] = pre_process(inputRow[2])
            writer.writerow(outputRow)

    # close the file
    outputFile.close()

path = 'data/Wiki-processed.csv'
#writefile_2(path)
remove_punctuation(path)

