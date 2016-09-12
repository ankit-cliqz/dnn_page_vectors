#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re
import itertools
from collections import Counter
import numpy as np

class DataUtils(object):
    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def get_text_word_splits(self, training_data):
        """ Get each line in the data as individual items in a list and the split each sentence on spaces. """
        text = [self.clean_str(i) for i in training_data]
        text = [s.split(" ") for s in text]
        return text

    def pad_sentences(self, sentences, padding_word="<PAD/>"):
        """
        Pads all sentences to the same length. The length is defined by the longest sentence.
        Returns padded sentences.
        """
        sequence_length = max(len(x) for x in sentences)

        print "Maximum Length of the Sequence: {}".format(sequence_length)

        padded_sentences = []
        for i in range(len(sentences)):
            sentence = sentences[i]
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        return padded_sentences

    def build_vocab(self, sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word, HAS list of words
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        # Mapping from word to index, HAS A Dict where key is word and value is the index position or a unique identifier which corresponds to the word.
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return vocabulary_inv

    def build_input_data(self, sentences, vocabulary, return_array=True):
        """
        Maps sentences and labels to vectors based on a vocabulary.
        """
        y = [[vocabulary.get(word) for word in sentence] for sentence in sentences]
        if return_array:
            x = np.array(y)
            return x
        else:
            return y
