##################################################################################################
# Imports

import tensorflow as tf
#assert tf.__version__ == '1.0.0-rc0' # if code breaks, check tensorflow version
from tensorflow.contrib.tensorboard.plugins import projector

import os
import vocabs as voc

from gensim.models import KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import gensim.downloader as api

import numpy as np

##################################################################################################
# Constants
CHECKPOINT = './checkpoints/'
CHECKPOINT_OUT_FILE = "word_embeddings"
METADATA = 'metadata.tsv'
METADATA_PATH = CHECKPOINT + METADATA
VECTORDATA = ''
TENSORNAME = 'word_embedding_sample'

##################################################################################################
# Methods
def create_checkpoint():
    if not os.path.exists(CHECKPOINT): os.mkdir(CHECKPOINT)

def get_some_text(inputdir):
    #textfile = '/Users/projects/nlp/data/19009-data.txt'
   
    text = ''
    for filename in os.listdir(inputdir):
        path = inputdir + filename

        with open(path, mode='r', encoding='UTF-8', errors='replace') as f:
            text = text + f.read()
            f.close()
        print(f'Getting and processed file: {filename}')
 
    #text = "Armin ist Fußballer. Neymar ist ein Fußballer. Fußball ist ein Sport. Handball ist ein Sport."
    return text

def create_sentences(text):
    print('Creating the sentences...')
    tokenizedVocabs = voc.Vocabs(text)
    print('Creating the sentences...done')
    return tokenizedVocabs.tokenized

def create_model(sentences, getbigmodell = False):
    
    # load pre-trained word-vectors from gensim-data
    if getbigmodell:
        print('Creating big model...')
        model = api.load("glove-wiki-gigaword-100")  
        print('Creating big model...done.')
        return model
    
    print('Creating the model...')
    model = Word2Vec([sentences], min_count=1)
    print('Creating the model...done.')
    return model

def create_word_embedding(model):
    
    EMBED_SIZE = model.vector_size
    VOCAB_LEN = len(model.wv.index2word)

    modelspace = model.wv.index2word[:100000]
    placeholder = np.zeros((VOCAB_LEN, EMBED_SIZE))

    tsv_row_template = "{}\t{}\n"

    print('Creating meta file...')
    with open(METADATA_PATH, 'w+') as file_metadata:
        header_row = tsv_row_template.format("Name", "Index")
        file_metadata.write(header_row)
        for value,word in enumerate(modelspace):
            placeholder[value] = model[word]
            data_row = tsv_row_template.format(word, str(value))
            file_metadata.write(data_row)
    print(f'{METADATA_PATH} has been written.')
    print(f'Vocab length is: {VOCAB_LEN}.')
    print(f'Embedding size is: {EMBED_SIZE}.')

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable=False, name=TENSORNAME)
    tf.global_variables_initializer().run()

    return modelspace, embedding

def save_checkpoint():

    # Saver
    saver = tf.train.Saver(tf.global_variables())
    # start session
    session = tf.Session()
    summary_writer = tf.summary.FileWriter(CHECKPOINT, graph=session.graph)
    session.run([tf.global_variables_initializer()]) # init variables

    #... do stuff with session

    # save checkpoints periodically
    filename = os.path.join(CHECKPOINT, CHECKPOINT_OUT_FILE)
    saver.save(session, filename, global_step=1)
    print(f'Word_embeddings checkpoint saved under: {filename}')
    return summary_writer

def visualize_embeddings(summary_writer, word_embeddings_name, metadata_path = METADATA):
    """
        Link metadata tsv file to embedding
    """
    config = projector.ProjectorConfig()

    embedding = config.embeddings.add() # could add more metadata files here
    embedding.tensor_name = word_embeddings_name
    embedding.metadata_path = metadata_path

    projector.visualize_embeddings(summary_writer, config)

    print("Metadata linked to checkpoint")
    print("run: tensorboard --logdir checkpoints/")


##################################################################################################
# Main

create_checkpoint()
text = get_some_text(r'/Users/arminhadzalic/Projects/nlp/data_raw/textfiles/')

sentences = create_sentences(text)
model = create_model(sentences, getbigmodell=False)
vocabs, embeddings = create_word_embedding(model)
summary_writer = save_checkpoint()
visualize_embeddings(summary_writer, embeddings.name, METADATA)

# Start cmd with:
# python /Users/arminhadzalic/opt/ananda3/pkgs/tensorboard-1.13.1-py37haf313ee_0/lib/python3.7/site-packages/tensorboard/main.py --logdir /Users/arminhadzalic/Projects/nlp/checkpoints
