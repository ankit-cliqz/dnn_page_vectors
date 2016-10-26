#!/usr/bin/python
# -*- coding: UTF-8 -*-

from utils.data_utils import DataUtils

du  = DataUtils()

train_data = "statue of liberty"
train_data_list = ["statue of liberty", "new york"]

print du.get_text_word_splits(train_data, mode='word')
print du.get_text_word_splits(train_data_list, mode='word')
print du.get_text_word_splits(train_data, mode='char')
print du.get_text_word_splits(train_data_list, mode='char')
print "ngram ..."
print du.get_text_word_splits(train_data, mode='ngram')
print du.get_text_word_splits(train_data_list, mode='ngram')

