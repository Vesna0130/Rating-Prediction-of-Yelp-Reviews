#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import nltk
#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import re
#from collections import Counter
#import numpy as np

TRAIN_CSV = 'train_data.csv'
TRAIN_OUT = 'new_train.csv'
TRAIN_TEXT = 'text_train.csv'
TRAIN_TRANS = 'train_trans.csv'

TEST_CSV = 'testval_data.csv'
TEST_OUT = 'new_test.csv'
TEST_TEXT = 'text_test.csv'
TEST_TRANS = 'test_trans.csv'

snowball = SnowballStemmer('english')

# Set path
#os.chdir('/Users/vesna/VesnaLi/Courses/STAT628_Data_Science_Practicum/Module2/data')


### Spell checking
# which actually will not be used in the following process
"""
def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('dictionary.txt').read()))


def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N


def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)


def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])


def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)


def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
"""

### Text cleaning
def proc_text(text):
    "Cleaning text."
    # Noise removing
    new_text = re.sub("n't", ' not', text)

    # Removing punctuation and numbers
    new_text = re.sub('[^0-9a-zA-Z]', ' ', new_text)

    # Converting text to lower case
    new_text = new_text.lower()

    # Tokenizing text into bags of words
    new_text = nltk.word_tokenize(new_text)

    # Stemming and Lemmatizing
    # There are several ways
    # wordnet_lematizer = WordNetLemmatizer()
    # new_text = [wordnet_lematizer.lemmatize(word, pos='v') for word in new_Text]
    new_text = [snowball.stem(word) for word in new_text]

    # Spell checking
    # Waste a lot of time(200 times!)
    #new_text = [correction(word) for word in new_text]

    # Cleaning text of stopwords
    #stops = set(stopwords.words("english")) - set(['not'])
    #new_text = [word for word in new_text if word not in stops]

    #return {word: True for word in filtered_words}
    return ' '.join(new_text).replace("\n","")


def make_text(input_csv, output_csv):
    "Save cleaned text into a csv file"
    # About 75 mins.
    table = pd.read_csv(input_csv)
    new_table = table[['text']].copy()

    # Add a cleaned text column
    new_table['clean'] = pd.Series([proc_text(t) for t in new_table['text']])

    # Write to a new file
    new_table.to_csv(output_csv, sep=',', encoding='utf-8', header=True,
                     doublequote=True, index=False)


def combine(input_csv, text_csv):
    "Add stars into the text csv"
    text = pd.read_csv(text_csv)
    table = pd.read_csv(input_csv)
    text['stars'] = table[['stars']]

    # Write to a new file
    text.to_csv(text_csv, sep=',', encoding='utf-8', header=True,
                 doublequote=True, index=False)


### Main function
def main():
    #make_text(TRAIN_CSV, TRAIN_TEXT)
    make_text(TEST_CSV, TEST_TEXT)
    #make_file(TRAIN_CSV, TRAIN_TRANS, TRAIN_OUT)
    #make_file(TEST_CSV, TEST_TRANS, TEST_OUT)
    #combine(TRAIN_CSV, TRAIN_TEXT)


if __name__ == '__main__':
    main()
