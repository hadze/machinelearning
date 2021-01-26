from bs4 import BeautifulSoup
import re
import requests
import heapq
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords


# Inputs
url = str(input('Paste the url: '))
num = int(input('Enter the Number of Sentence you want in the summary: '))
num = int(num)
summary = ''
res = requests.get(url)

# Methods
def clean(text):
    text = re.sub(r"\[[0-9]*\]",' ',text)
    text = text.lower()
    text = re.sub(r'\s+',' ',text)
    text = re.sub(r",",' ',text)
    return text

def get_key(val, sentences_score): 
    for key, value in sentences_score.items(): 
        if val == value: 
            return key 

def get_stopwords():
    return set(stopwords.words('english'))

def get_word_frequencies(word_tokens, stopwords):
    word_frequency = {}
    for word in word_tokens:
        if word not in stopwords:
            if word not in word_frequency.keys():
                word_frequency[word]=1
            else:
                word_frequency[word] +=1
    maximum_frequency = max(word_frequency.values())

    for word in word_frequency.keys():
        word_frequency[word] = (word_frequency[word]/maximum_frequency)

    return maximum_frequency, word_frequency

def get_sentence_score(sentence_tokens):
    sentences_score = {}
    for sentence in sentence_tokens:
        for word in word_tokenize(sentence):
            if word in word_frequency.keys():
                if (len(sentence.split(' '))) <30:
                    if sentence not in sentences_score.keys():
                        sentences_score[sentence] = word_frequency[word]
                    else:
                        sentences_score[sentence] += word_frequency[word]
    return sentences_score


############################################## Main
soup = BeautifulSoup(res.text,'html.parser') 
content = soup.findAll('p')
for text in content:
    summary +=text.text 

summary = clean(summary)

print('Getting the data......\n')

# Tokenizing
sentence_tokens = sent_tokenize(summary)
summary = re.sub(r"[^a-zA-z]",' ',summary)
word_tokens = word_tokenize(summary)

# Removing Stop words
stopwords =  get_stopwords()

# Get word frequency
maximum_frequency, word_frequency = get_word_frequencies(word_tokens, stopwords)
print(f'Maximum frequency is: {maximum_frequency} \n')
print(f'Word frequencies in detail:')
print(word_frequency)
print(f'\n')

sentences_score = get_sentence_score(sentence_tokens)
print(f'Sentence max. score value is:')
max_score = max(sentences_score.values())
print(max_score)
print('\n')

print(f'Sentence with max. score is:')
key = get_key(max_score, sentences_score)
print(key + '\n')

print(f'Other sentence score values are:')
print(sentences_score)
print(f'\n')

summary = heapq.nlargest(num,sentences_score,key=sentences_score.get)
print(f'Summary:')
print(' '.join(summary))
summary = ' '.join(summary)

# Example pages
#https://en.wikipedia.org/wiki/Apple_Inc.
#https://de.wikipedia.org/wiki/Quelltext