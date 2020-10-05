import pprint
import re
import numpy as np

class Vocabs():
    def __init__(self, text):
        """get the tokenized version of the given text
        """
        self.text = text
        self.tokenized = re.sub('[,?.]','', self.text).lower()
        self.tokenized = re.sub('\d+','x', self.tokenized)
        self.tokenized = re.sub(r'\b[a-zA-Z]\b', '', self.tokenized)
        self.tokenized = self.tokenized.rstrip('\r\n')
        self.tokenized = self.tokenized.replace('\n', '')
        self.tokenized = self.tokenized.replace('\t', ' ') 
        self.tokenized = self.tokenized.replace('\\',' ')
        self.tokenized = self.tokenized.replace('\\',' ')
        #self.tokenized = self.tokenized.replace('…','')
        #self.tokenized = self.tokenized.replace('�','')
        self.tokenized = self.tokenized.strip()
        self.tokenized = self.removeSpecialChar(self.tokenized) 
        # self.tokenized = self.tokenized.replace('  ', ' ')  
        # self.tokenized = self.tokenized.replace('  ', ' ') 
        # self.tokenized = self.tokenized.replace('  ', ' ') 
        # self.tokenized = self.tokenized.replace('  ', ' ')  
        # self.tokenized = self.tokenized.replace('  ', ' ') 
        # self.tokenized = self.tokenized.replace('  ', ' ')  
        self.tokenized = self.tokenized.replace('_', '')


        self.tokenized = self.tokenized.split(' ')

        while '' in self.tokenized: self.tokenized.remove('')
        
        
    def numberize(self):
        """numbering the tokenized text by just taking each word
        """
        self.vocab = {k:v for v,k in enumerate(np.unique(self.tokenized))}
        pprint.pprint(self.vocab)

        return self.vocab

    def len(self):
        return len(self.vocab.keys())

    def removeSpecialChar(self, content):
        newText = re.subn(r"[-()\"#/@;:<>{}`+=~|.!?,“”\[\*\]'^°²³µ%§$&…�—]", " ", content)
        return newText[0]
    
    def cleanContent(content):
        return -1