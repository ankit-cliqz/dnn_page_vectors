#CNN-DSSM-V2 Network for Query Document Similarity


##TODO LIST

1. Batch Generator using HDF5 or a normal json based text file for model.fit_generator method in Keras.
2. Test on the g2.8xlarge Machine
3. Test the normalizations and shuffling techniques while training.
4. For future releases, have a configuration file to select the type of model to be used and also the features to use:
i.e. charachter level, charachter n-grams or word level.
5. Have a mechanism to change the word vector file (Word2Vec, FAST and Glove) to test them easily.
6. Auto downloading of dataset from s3.