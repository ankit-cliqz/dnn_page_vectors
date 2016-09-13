#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Imports
from __future__ import print_function
import data_helpers
import joblib
import theano
import json
from keras import backend as TK
from keras.layers import Input, merge
from keras.layers import Dense, Lambda, Reshape, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import Model, Sequential, Graph
from keras.engine.topology import Layer
from itertools import chain
from keras.layers.embeddings import Embedding
from keras import callbacks
import theano.tensor as tt
import numpy as np
from config import Configuration
np.random.seed(1337)  # for reproducibility
from data_helpers import DataHelpers
# Word Embedding Size
embedding_dim = 100

# Training Parameters
batch_size = 1024
nb_epoch = 5

# CNN Model Parameters
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150
J = 3
GAMMA = 10

load_vocab_pickled = False

conf = Configuration()
dh = DataHelpers()





'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''
load_pickled_data=False
embedding_weights_masking=False # Mask the Embedding Weights If false, data get written from zero-index of the array.

if embedding_weights_masking:
    masking_value = ""  # For masked embedding weights leave it blank "", else for masked use "_non_masked"
else:
    masking_value = "_non_masked"  # For masked embedding weights leave it blank "", else for masked use "_non_masked"
embedding_weights_file_path = '/raid/ankit/lstm/we_embedding_weights{}.pkl'.format(masking_value)

if not load_pickled_data:
    print('Loading data from scratch ...')
    # Load data
    [([x_query, _],
        [x_similar, _],
        [x_nonsimilar1, _],
        [x_nonsimilar2,  _],
        [x_nonsimilar3,  _]), data_output_labels, _], embedding_weights \
        = data_helpers.load_data(embedding_dim, embedding_weights_masking, load_embeddings_pickled=False)
else:
    print('Loading data from pickled file ...')
    # Load data
    ([x_query, _],
        [x_similar, _],
        [x_nonsimilar1, _],
        [x_nonsimilar2,  _],
        [x_nonsimilar3,  _]), data_output_labels, _ \
        = joblib.load("/raid/ankit/lstm/lstm_input_data_compact.pkl")
    embedding_weights = joblib.load(embedding_weights_file_path)
print('Input Data and Word Embeddings... loaded in Memory!')

# Size of Embeddings Weights layer, also the size of the vocabulary
vocab_size = embedding_weights.shape[0]

print('Input Data Dimensionality:')
print('x (Input) shape:', x_query.shape)
print('x-similar (Input) shape:', x_similar.shape)
print('x-nonsimilar1 (Input) shape:', x_nonsimilar1.shape)
print('x-nonsimilar2 (Input) shape:', x_nonsimilar2.shape)
print('x-nonsimilar3 (Input) shape:', x_nonsimilar3.shape)
print('y (Output) shape:', data_output_labels.shape)
print('Vocabulary Shape:', vocab_size)


def R(vects):
    def _squared_magnitude(x):
        return tt.sqr(x).sum(axis=-1)

    def _magnitude(x):
        return tt.sqrt(
            tt.maximum(
                _squared_magnitude(x),
                np.finfo(
                    x.dtype).tiny))

    def _cosine(x, y):
        return tt.clip((x * y).sum(axis=-1) /
                       (_magnitude(x) * _magnitude(y)), 0, 1)

    return _cosine(*vects).reshape([-1, 1])


def model(sequence_length=None):
    graph = Graph()
    graph.add_input(name='input', input_shape=(sequence_length, embedding_dim))
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                             filter_length=fsz,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1,
                             input_dim=embedding_dim,
                             input_length=sequence_length)
        pool = MaxPooling1D(pool_length=sequence_length - fsz + 1)
        graph.add_node(conv, name='conv-%s' % fsz, input='input')
        graph.add_node(pool, name='maxpool-%s' % fsz, input='conv-%s' % fsz)
        graph.add_node(
            Flatten(),
            name='flatten-%s' %
            fsz,
            input='maxpool-%s' %
            fsz)

    if len(filter_sizes) > 1:
        graph.add_output(name='output',
                         inputs=['flatten-%s' % fsz for fsz in filter_sizes],
                         merge_mode='concat')
    else:
        graph.add_output(name='output', input='flatten-%s' % filter_sizes[0])

    # main sequential model
    model = Sequential()
    model.add(
        Embedding(
            vocab_size,
            embedding_dim,
            input_length=sequence_length,
            weights=[embedding_weights]))
    model.add(
        Dropout(
            dropout_prob[0],
            input_shape=(
                sequence_length,
                embedding_dim)))
    model.add(graph)
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    return model

# Input Layer with all the query, similar and non similar documents.
query = Input(shape=(x_query.shape[1],), dtype='int32')
pos_doc = Input(shape=(x_similar.shape[1],), dtype='int32')
neg_docs = [Input(shape=(x_nonsimilar1.shape[1],), dtype='int32'),
            Input(shape=(x_nonsimilar2.shape[1],), dtype='int32'),
            Input(shape=(x_nonsimilar3.shape[1],), dtype='int32')]

query_model = model(sequence_length=x_query.shape[1])
sim_doc_model = model(sequence_length=x_similar.shape[1])
nonsim_doc_model1 = model(sequence_length=x_nonsimilar1.shape[1])
nonsim_doc_model2 = model(sequence_length=x_nonsimilar2.shape[1])
nonsim_doc_model3 = model(sequence_length=x_nonsimilar3.shape[1])

pos_doc_sem = sim_doc_model(pos_doc)
neg_doc_sems = [nonsim_doc_model1(neg_docs[0]),nonsim_doc_model2(neg_docs[1]),nonsim_doc_model3(neg_docs[2])]
query_sem = query_model(query)


R_layer = Lambda(R, output_shape=(1,))  # See equation (4).
R_Q_D_p = R_layer([query_sem, pos_doc_sem])  # See equation (4).

# See equation (4).
R_Q_D_ns = [R_layer([query_sem, neg_doc_sem]) for neg_doc_sem in neg_doc_sems]
concat_Rs = merge([R_Q_D_p] + R_Q_D_ns, mode="concat", concat_axis=1)
concat_Rs = Reshape((J + 1,))(concat_Rs)

# See equation (5).
with_gamma = Lambda(lambda x: x * GAMMA, output_shape=(J + 1,))(concat_Rs)

# See equation (5).
exponentiated = Lambda(lambda x: TK.exp(x), output_shape=(J + 1,))(with_gamma)
exponentiated = Reshape((J + 1,))(exponentiated)

# See equation (5).
prob = Lambda(lambda x: TK.expand_dims(
    x[:, 0] / TK.sum(x, axis=1), 1), output_shape=(1,))(exponentiated)

inputs = [query, pos_doc] + neg_docs

# Model Compile
model = Model(input=inputs, output=prob)
model.compile(optimizer="adam", loss="binary_crossentropy")

# Model Summary
#print(model.summary())

# verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
print('Train on Data...')
hist = model.fit([x_query, x_similar, x_nonsimilar1, x_nonsimilar2, x_nonsimilar3], data_output_labels, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.2)

# History Call back to record: training / validation loss / accuracy at each epoch.
print(hist.history)

print('Model Fitting Completed! Now saving trained Model on Disk ... ')


# Save the model configuration and model weights.
model.save('/raid/ankit/lstm/cnn_model_dssm.h5')  # creates a HDF5 file

# save model configuration and model weights seperately.
fw = open("/raid/ankit/lstm/cnn_dssm_model_only.json", "w")
json_string = model.to_json()
fw.write(json_string)
fw.close()
model.save_weights('/raid/ankit/lstm/cnn_dssm_model_weights.h5') # creates a HDF5 file