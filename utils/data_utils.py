#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re
import itertools
from collections import Counter
import numpy as np

class DataUtils(object):

    def __init__(self):
        self.rgx = re.compile(r"[^\wäöüß€#\n.$]", re.UNICODE)

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(self.rgx, ' ', string)
        return string.strip().lower()

    def get_text_feature_splits(self, training_data, cutoff=None, mode="word", ngram=3):
        """
        Get each line in the data as individual items in a list and the split each sentence for feature at word,
        ngram (default: trigram) or character level
        """
        if mode=="word":
            if type(training_data) is list:
                text = [self.clean_str(i) for i in training_data]
                text = [s.split(" ")[:cutoff] for s in text]
                return text
            else:
                text = self.clean_str(training_data)
                text = text.split(" ")[:cutoff]
                return [text]

        elif mode=="ngram":
            if type(training_data) is list:
                text = [self.clean_str(i) for i in training_data]
                text = [[x[i:i+ngram] for i in range(len(x)-ngram+1)][:cutoff] for x in text]
                return text
            else:
                text = self.clean_str(training_data)
                text = [text[i:i+ngram] for i in range(len(text)-ngram+1)][:cutoff]
                return [text]

        elif mode=="char":
            if type(training_data) is list:
                text = [self.clean_str(i) for i in training_data]
                text = [list(x)[:cutoff] for x in text]
                return text
            else:
                text = self.clean_str(training_data)
                text = [list(x) for x in text]
                text = [x for x in itertools.chain.from_iterable(text)][:cutoff]
                return [text]

    def pad_sentences(self, sentences, sequence_length, padding_word="<PAD/>"):
        """
        Pads all sentences to the same length. The length is defined by the longest sentence.
        Returns padded sentences.
        """
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
