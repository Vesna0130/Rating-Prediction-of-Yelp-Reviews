#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

TRAIN_CSV = 'train_data.csv'
TRAIN_OUT = 'new_train.csv'
TRAIN_TEXT = 'text_train.csv'
TRAIN_TRANS = 'train_trans.csv'

TEST_CSV = 'testval_data.csv'
TEST_OUT = 'new_test.csv'
TEST_TEXT = 'text_test.csv'
TEST_TRANS = 'test_trans.csv'

# Set path
#os.chdir('/Users/vesna/VesnaLi/Courses/STAT628_Data_Science_Practicum/Module2/data')


def n_word(text):
    "Count the number of words in a review."
    return len(text.split(' '))


def n_capital(text):
    "Count the number of capital letters in a review."
    words = text.split(' ')
    delims = "?.,!:;"
    words = text.split()
    count = sum([w.strip(delims).isupper() for w in words if w != 'I'])
    p_len = count / len(text)
    p_nword = count / len(words)
    return np.array([count, p_len, p_nword])


def n_punc(text):
    "Count the number of puncutuations in a review, and return a list."
    stop = '.'
    question = '?'
    exclamation = '!'
    return np.array([text.count(stop), text.count(question), text.count(exclamation)])


def make_file(input_csv, text_csv, output_csv):
    # About 15 mins.
    "Convert given csv file to a new csv file without cleaned text."
    table = pd.read_csv(input_csv)
    text = pd.read_csv(text_csv)
    new_table = table[['name', 'date', 'longitude', 'latitude', 'categories']].copy()

    new_table['city'] = pd.Series([t.lower() for t in table['city']])

    if input_csv == TRAIN_CSV:
        new_table['stars'] = table[['stars']]

    # Add a cleaned text column
    # It will take more than one hour
    # so we firstly built a new function 'make_text' to do it seperately
    # and try it later
    new_table['clean'] = pd.Series([t for t in text['trans_text']])

    new_table['nword'] = pd.Series([n_word(t) for t in table['text']])
    new_table['length'] = pd.Series([len(t) for t in table['text']])

    new_table['ncapital'] = pd.Series([n_capital(t)[0] for t in table['text']])
    new_table['plen'] = pd.Series([n_capital(t)[1] for t in table['text']])
    new_table['pword'] = pd.Series([n_capital(t)[2] for t in table['text']])

    new_table['comma'] = pd.Series([n_punc(t)[0] for t in table['text']])
    new_table['ques'] = pd.Series([n_punc(t)[1] for t in table['text']])
    new_table['excla'] = pd.Series([n_punc(t)[2] for t in table['text']])

    # Write to a new file
    new_table.to_csv(output_csv, sep=',', encoding='utf-8', header=True,
                     doublequote=True, index=False)


### Main function
def main():
    #make_file(TRAIN_CSV, TRAIN_TRANS, TRAIN_OUT)
    make_file(TEST_CSV, TEST_TRANS, TEST_OUT)


if __name__ == '__main__':
    main()
