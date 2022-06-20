import pandas as pd
import csv

#add answer phrases for each answer 
def phrase_add(dictionary):
    x = 0
    for row in answers.itertuples(index=True, name='Pandas'):
        question = row.Question
        if question in dictionary and row.Sentence == '':
            answers_final.at[row.Index,'Sentence'] = dictionary[question]

        #to see the progress
        x+=1
        if x%10000 == 0:
            print(x, 'rows iterated')

#answer phrases
answer_phrases = pd.read_csv('data\WikiQASent.pos.ans.tsv', sep='\t')
answer_phrases2 = answer_phrases[['Question', 'AnswerPhrase2']]
answer_phrases3 = answer_phrases[['Question', 'AnswerPhrase3']]
#issues with dataframe loc values; use dict instead
phrase_dict2 = dict(answer_phrases2.values)
phrase_dict3 = dict(answer_phrases3.values)

#answers
answers = pd.read_csv('data\WikiQA.csv')
answers = answers[['Question','Sentence', 'Label']]
#rename column to now have duplicate names
answers.rename(columns = {'Sentence':'Answer'}, inplace = True)
#insert empty columns
answers["Sentence"] = ""
answers["Similarity_score"] = ""
#rearrange column order
answers = answers[['Question', 'Sentence', 'Answer','Label','Similarity_score']]

answers_final = answers

#run twice since there a missing answer phrases in columns
phrase_add(phrase_dict2)
phrase_add(phrase_dict3)
print(answers_final.head(50))

answers_final.to_csv('data/Wiki.csv', sep=',', encoding='utf-8')