#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
#from nltk import word_tokenize
from nltk.corpus import stopwords
import collections


TRAIN_TRANS = 'train_trans.csv'
stops = stopwords.words("english")

#os.chdir('/Users/vesna/VesnaLi/Courses/STAT628_Data_Science_Practicum/Module2/data')

full_table = pd.read_csv(TRAIN_TRANS)
table = full_table#.iloc[0:1000, ]

def find_ngrams(input_list, n):
    "Find all of the n-grams words in a list"
    return (list(zip(*[input_list[i:] for i in range(n)])))


def count_stops(words):
    "count the number of stop-words in a list."
    return (sum(t in stops for t in words))


def frequency(text, freq):
    "Count the numbers of each word."
    text_tokens = text.split(' ')
    #freq = collections.defaultdict(int)
    for word in text_tokens:
        if word not in stops and len(word) > 2:
            freq[word] += 1
    return freq


def frequency_2gram(text, freq):
    "Count the numbers of each 2-gram words."
    text_tokens = text.split(' ')
    text_tokens_2gram = find_ngrams(text_tokens, 2)
    for words in text_tokens_2gram:
        if count_stops(words) < 2:
            w = ' '.join(words)
            freq[w] += 1
    return freq


def freq_stars(star, dim):
    "Calculate the frequency of word in the reviews, group by stars"
    freq = collections.defaultdict(int)
    if dim == 1:
        for text in table[table['stars'] == star]['trans_text']:
            freq = frequency(text, freq)
    elif dim == 2:
        for text in table[table['stars'] == star]['trans_text']:
            freq = frequency_2gram(text, freq)
    return freq


def save_dict(file, d, star):
    "Save a dictionary into a file."
    f = open(file, 'a')
    for key, value in d.items():
        f.write(key + ',' + str(value) + ',' + str(star))
        f.write('\n')
    f.close()


def combine_dict(file, dim):
    f = open(file, 'w')
    f.truncate()
    f.write("words,freq,star\n")
    f.close()
    for star in [1, 2, 3, 4, 5]:
        d = freq_stars(star, dim)
        save_dict(file, d, star)
    f.close()


def main():
    # About 20 mins.
    combine_dict("words_by_stars.csv", 1)
    combine_dict("words_by_stars_2gram.csv", 2)


if __name__ == '__main__':
    main()