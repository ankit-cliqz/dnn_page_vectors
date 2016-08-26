#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''

TASK: Train a recurrent convolutional network on  query - url page data.

Dataset: (Normalized Text lines)
similar_dataset  = <query_bow> <url_page_data>
non_similar_dataset = <query_bow> <url_page_data>


{ 'q': query, 'doc_corr': correct_document, 'doc_incorr': [incorrect_doc1, incorrect_doc2, incorrect_doc3] }

Training to maximize correct classification, between query and page being similar.

Given a query and a corresponding url_page_content, construct a vector which defines its similarity and emit the
score out.


'''
from __future__ import print_function

import numpy as np

np.random.seed(1337)  # for reproducibility
from dssm_lstm import data_helpers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Merge
from keras.layers.core import Reshape
import joblib

# Embedding

# maxlen = 100
embedding_size = 100

# Convolution
filter_length = 3
nb_filter = 64
pool_length = 2

# LSTM
lstm_output_size = 64

# Dense layer units
dense_units = 32
final_dense_units = 4
# Training
batch_size = 64
nb_epoch = 2


load_picked_data=True
'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''
if not load_picked_data:
    print('Loading data from scratch ...')
    # Load data
    ([x_query, vocabulary_query, vocabulary_inv_query],
     [x_similar, vocabulary_similar, vocabulary_inv_similar],
     [x_nonsimilar1, vocabulary_nonsimilar1, vocabulary_inv_nonsimilar1],
     [x_nonsimilar2, vocabulary_nonsimilar2, vocabulary_inv_nonsimilar2],
     [x_nonsimilar3, vocabulary_nonsimilar3, vocabulary_inv_nonsimilar3]), data_output_labels \
        = data_helpers.load_data()
else:
    print('Loading data from picked file ...')
    # Load data
    ([x_query, vocabulary_query, vocabulary_inv_query],
     [x_similar, vocabulary_similar, vocabulary_inv_similar],
     [x_nonsimilar1, vocabulary_nonsimilar1, vocabulary_inv_nonsimilar1],
     [x_nonsimilar2, vocabulary_nonsimilar2, vocabulary_inv_nonsimilar2],
     [x_nonsimilar3, vocabulary_nonsimilar3, vocabulary_inv_nonsimilar3]), data_output_labels \
        = joblib.load("/raid/ankit/data/lstm_data/lstm_input_data.pkl")

print('x (Input) shape:', x_query.shape)
print('y (Output) shape:', data_output_labels.shape)


vocab_dim = 100 # dimensionality of your word vectors
index_dict = {}
n_symbols = 8516772 # adding 1 to account for 0th index (for masking) [Number of word:vector pairs is 7115783]
embedding_weights = np.zeros((n_symbols+1,vocab_dim))

# Path to word vectors file
word_vectors_file = "/raid/ankit/wordvectors/vectors_wholecorpus100.txt"
i = 0

words_set = set()
with open(word_vectors_file) as fopen:
    for line in fopen:
        i+=1
        try:
            components = line.strip().split()
            if not len(components) < vocab_dim:

                if i % 1000000 == 0:
                    print("Words added to embedding matrix ... {}".format(i))

                word = components[0]
                words_set.add(word)
                vec = np.asarray([float(x) for x in components[1:vocab_dim + 1]])
                index_dict[word] = i
                embedding_weights[i, :] = vec
        except Exception as e:
            print("Exception Encountered: ".format(e))
print("Word Embeddings added. Current value of i : {}".format(i))

vocab_list = [vocabulary_query, vocabulary_similar, vocabulary_nonsimilar1, vocabulary_nonsimilar2, vocabulary_nonsimilar3 ]

for vc in vocab_list:
    for k, _ in vc.iteritems():
        if not k in words_set:
            i+=1
            index_dict[k] = i
            embedding_weights[i, :] = np.random.uniform(-0.25, 0.25, vocab_dim)
print("Added Random Vectors for the unseen words in the corpus. Current value of i: {}".format(i))





print('Build model...')

# For training your own embeddings as one hot representations
# model.add(Embedding(max_features, embedding_size, input_length=embedding_size))

# query : Query Input
query = Sequential()
# Co-train the existing word embeddings as weights of the embeddings layer
query.add(Embedding(n_symbols + 1, embedding_size, mask_zero=True, weights=[embedding_weights])) # note you have to put embedding weights in a list by convention
query.add(Dropout(0.25))
query.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
query.add(MaxPooling1D(pool_length=pool_length))
query.add(LSTM(lstm_output_size))
query.add(Dense(dense_units))


# Correct_Doc: Document Input correct
correct_doc = Sequential()
# Co-train the existing word embeddings as weights of the embeddings layer
correct_doc.add(Embedding(n_symbols + 1, embedding_size, mask_zero=True, weights=[embedding_weights])) # note you have to put embedding weights in a list by convention
correct_doc.add(Dropout(0.25))
correct_doc.add(Convolution1D(nb_filter=nb_filter,
                              filter_length=filter_length,
                              border_mode='valid',
                              activation='relu',
                              subsample_length=1))
correct_doc.add(MaxPooling1D(pool_length=pool_length))
correct_doc.add(LSTM(lstm_output_size))
correct_doc.add(Dense(dense_units))

# incorrect_doc: Document Input in-correct: 1
incorrect_doc1 = Sequential()
# Co-train the existing word embeddings as weights of the embeddings layer
incorrect_doc1.add(Embedding(n_symbols + 1, embedding_size, mask_zero=True, weights=[embedding_weights])) # note you have to put embedding weights in a list by convention
incorrect_doc1.add(Dropout(0.25))
incorrect_doc1.add(Convolution1D(nb_filter=nb_filter,
                              filter_length=filter_length,
                              border_mode='valid',
                              activation='relu',
                              subsample_length=1))
incorrect_doc1.add(MaxPooling1D(pool_length=pool_length))
incorrect_doc1.add(LSTM(lstm_output_size))
incorrect_doc1.add(Dense(dense_units))



# incorrect_doc: Document Input in-correct: 2
incorrect_doc2 = Sequential()
# Co-train the existing word embeddings as weights of the embeddings layer
incorrect_doc2.add(Embedding(n_symbols + 1, embedding_size, mask_zero=True, weights=[embedding_weights])) # note you have to put embedding weights in a list by convention
incorrect_doc2.add(Dropout(0.25))
incorrect_doc2.add(Convolution1D(nb_filter=nb_filter,
                              filter_length=filter_length,
                              border_mode='valid',
                              activation='relu',
                              subsample_length=1))
incorrect_doc2.add(MaxPooling1D(pool_length=pool_length))
incorrect_doc2.add(LSTM(lstm_output_size))
incorrect_doc2.add(Dense(dense_units))


# incorrect_doc: Document Input in-correct: 3
incorrect_doc3 = Sequential()
# Co-train the existing word embeddings as weights of the embeddings layer
incorrect_doc3.add(Embedding(n_symbols + 1, embedding_size, mask_zero=True, weights=[embedding_weights])) # note you have to put embedding weights in a list by convention
incorrect_doc3.add(Dropout(0.25))
incorrect_doc3.add(Convolution1D(nb_filter=nb_filter,
                              filter_length=filter_length,
                              border_mode='valid',
                              activation='relu',
                              subsample_length=1))
incorrect_doc3.add(MaxPooling1D(pool_length=pool_length))
incorrect_doc3.add(LSTM(lstm_output_size))
incorrect_doc3.add(Dense(dense_units))



## Cosine Merged Layers, later each layer is reshaped and then a lamda layer is applied which gives the cosine similarity.
# Query - Correct Document
merged_q_cd = Sequential()
merged_q_cd.add(Merge([query, correct_doc], mode='cos', name='q_cd' , dot_axes=1))
merged_q_cd.add(Reshape((1,)))
merged_q_cd.add(Lambda(lambda x: 1-x))

# Query - Incorrect Document 1
merged_q_incd1 = Sequential()
merged_q_incd1.add(Merge([query, incorrect_doc1], mode='cos', name='q_incd1' , dot_axes=1))
merged_q_incd1.add(Reshape((1,)))
merged_q_incd1.add(Lambda(lambda x: 1-x))

# Query - Incorrect Document 2
merged_q_incd2 = Sequential()
merged_q_incd2.add(Merge([query, incorrect_doc2], mode='cos', name='q_incd2' , dot_axes=1))
merged_q_incd2.add(Reshape((1,)))
merged_q_incd2.add(Lambda(lambda x: 1-x))

# Query - Incorrect Document 3
merged_q_incd3 = Sequential()
merged_q_incd3.add(Merge([query, incorrect_doc3], mode='cos', name='q_incd3' , dot_axes=1))
merged_q_incd3.add(Reshape((1,)))
merged_q_incd3.add(Lambda(lambda x: 1-x))



final_model = Sequential()
final_model.add(Merge([merged_q_cd, merged_q_incd1, merged_q_incd2, merged_q_incd3], mode='concat', name='final_layer'))
final_model.add(Dense(final_dense_units, init='uniform', activation='softmax'))

print("Compiling the model.... ")
final_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("Neural Network Model Compiled Successfully!")


print("Generate a visualization of the model.")
from keras.utils.visualize_util import plot
plot(final_model, to_file='model.png')
print("Plot Generated!")


print('Train...')
final_model.fit([x_query, x_similar,x_nonsimilar1, x_nonsimilar2, x_nonsimilar3], data_output_labels, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.2)
print('Model Fitting Completed!')