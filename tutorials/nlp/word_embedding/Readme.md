# Visualize Word Embeddings with Tensorboard

Example for the embedded words extracted from the [protocols](https://www.bundestag.de/dokumente/protokolle/plenarprotokolle/plenarprotokolle) of the german parliament and show them in the tensorboard window:
<img src="https://github.com/hadze/machinelearning/blob/master/tutorials/nlp/word_embedding/doc/pointcloud_class_party.gif" width="800" height="500"/><br/>
Here we can see that the system has learned from almost 40.000 words that e.g. the term CDU belongs to one party. In other words: The system is able to list the other parties as well. In fact, it shows their vectorial proximity to the search term CDU. Other keywords that one could use are:
* Umwelt (environment)
* Auto (car)
* Krieg (war)

<img src="https://github.com/hadze/machinelearning/blob/master/tutorials/nlp/word_embedding/doc/pointcloud_bild_zeiteinheiten.png" width="800" height="500"/><br/>
Here we can see that the system can assign time units. So the search for "days" results in hits like "years" and "weeks".
<br/>The training took about 15 min on a MacBook Pro  


Example for the embedded words extracted from the wiki page about [Italy](https://en.wikipedia.org/wiki/Italy) and show them in the tensorboard window:
<img src="https://github.com/hadze/machinelearning/blob/master/tutorials/nlp/word_embedding/doc/italy_tensorboard.gif" width="800" height="500"/>

## Description

You can find more details for this project on my [homepage](https://www.hadzalic.de/visualize-word-embeddings-with-tensorboard#page-content).
Generally speaking, converting words to vectors, or word vectorization, is a natural language processing (NLP) process. The process uses language models / techniques to map words into vector space. In fact it represents each word by a vector of real numbers. This finally leads to the fact that words with similar vectors also have similar meanings or at least a strong relationship. 

This code will execute the following steps:

* Create a checkpoint folder in order to store the meta and vectorfiles
* Get all text content from a folder with textfiles
* Extract the tokens and vocabs from the text
* Create a model
* Create the word embeddings and meta data from the model
* Save checkpoints
* Finally visualize the embeddings in tensorboard

<!--
**Attention:**<br/>
The vectors and also the distances between them are **not trained** and therefore **not meaningful** in this version. I will show an trained example in the next episode.
-->

## Parameters

**used codesnippet**
~~~python
model = Word2Vec(
        documents,
        size=150,
        window=10,
        min_count=4,
        workers=10,
        iter=10,
        callbacks=[epoch_logger])

#model.train(documents, total_examples=len(documents), epochs=10, callbacks=[epoch_logger])
~~~
**size**

The size of the dense vector to represent each token or word (i.e. the neighboring words). In this context one sometimes also speaks of the dimensions of a token. If you have only few data, it is better to work with a smaller size since you would only have so many unique neighbors for a given word. For larger amounts of data you are welcome to experiment with various sizes. A value of 100–150 has worked well for me for similarity lookups.

**window**

The maximum distance between the target word and its neighboring word. In theory, a smaller window should give you terms that are more related. 
Important note:
if your data is not sparse, then the window size should not matter too much. If you are not too sure about this, just use the default value.

**min_count**

Minimum frequency of words. The model ignores words that do not meet the min_count. Extremely rare words are usually unimportant, so it is best to get rid of them. The parameter does not really influence the model in terms of your final results. The settings here rather affect the memory usage and memory requirements of the model files.

**workers**

Specifies how many threads can be started in the background. Faster training possible depending on how many CPUs can be used.

**epochs**

Number of iterations (epochs) over the text corpus. 5 is a good starting point. I always use a minimum of 10 iterations.


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

protocols of the german parliament taken from (history from "Plenarprotokoll 148. Sitzung am 13.01.2016 - Plenarprotokoll 13.Sitzung am 21.02.2018")
https://www.bundestag.de/dokumente/protokolle/plenarprotokolle/plenarprotokolle
