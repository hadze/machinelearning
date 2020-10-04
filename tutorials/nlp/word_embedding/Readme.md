# Visualize Word Embeddings with Tensorboard

Example for the embedded words extracted from the wiki page about Italy and shown in the tensorboard window:
<img src="https://github.com/hadze/machinelearning/blob/master/tutorials/nlp/word_embedding/doc/italy_tensorboard.gif" width="800" height="500"/>

## Description

This code will execute the following steps:

* Create a checkpoint folder in order to store the meta and vectorfiles
* Get any text from a folder with textfiles
* Extract the tokens and vocabs
* Create a model
* Create the word embeddings and meta data from the model
* Save checkpoints
* Finally visualize the embeddings in tensorboard


## Sample textfile
italy.txt taken from https://en.wikipedia.org/wiki/Italy
