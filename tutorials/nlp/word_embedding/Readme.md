# Visualize Word Embeddings with Tensorboard

Example for the embedded words extracted from the wiki page about Italy and shown in the tensorboard window:
<img src="https://github.com/hadze/machinelearning/blob/master/tutorials/nlp/word_embedding/doc/italy_tensorboard.gif" width="800" height="500"/>


Example for the embedded words extracted from the protocols of the german parliament and shown in the tensorboard window:
<img src="https://github.com/hadze/machinelearning/blob/master/tutorials/nlp/word_embedding/doc/pointcloud_class_party.gif" width="800" height="500"/>



## Description

Generally speaking, converting words to vectors, or word vectorization, is a natural language processing (NLP) process. The process uses language models / techniques to map words into vector space. In fact it represents each word by a vector of real numbers. Meanwhile, it allows words with similar meanings have similar representations.

This code will execute the following steps:

* Create a checkpoint folder in order to store the meta and vectorfiles
* Get all text content from a folder with textfiles
* Extract the tokens and vocabs from the text
* Create a model
* Create the word embeddings and meta data from the model
* Save checkpoints
* Finally visualize the embeddings in tensorboard

**Attention:**<br/>
The vectors and also the distances between them are **not trained** and therefore **not meaningful** in this version. I will show an trained example in the next episode.

## Pre-requisites
Anaconda was already installed on my system. It makes the installation of packages and libraries much easier. So I suggest you to do the same. The version is not important at first. Just get the [current version](https://www.anaconda.com/products/individual) and you are good to go.

## Environment

In the config section you can find the [myenvironment.yml](https://github.com/hadze/machinelearning/blob/master/tutorials/nlp/word_embedding/config/myenvironment.yml) file containing the needed packages and libraries for this example. Basically I have used 
  - python=3.7.9
  - gensim=3.8.0
  - tensorboard==1.12.2
  - tensorflow==1.13.0rc1

You just have to execute the following steps to get the needed environment running on your system:

* Open terminal or console application on your system
* Create conda environment and install all needed packages: **conda env create -f myenvironment.yml**

## Sample textfile
italy.txt taken from https://en.wikipedia.org/wiki/Italy

protocols from the german parliament taken from (history from "Plenarprotokoll 148. Sitzung am 13.01.2016 - Plenarprotokoll 13.Sitzung am 21.02.2018")
https://www.bundestag.de/dokumente/protokolle/plenarprotokolle/plenarprotokolle
