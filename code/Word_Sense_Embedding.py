#!/usr/bin/env python
# coding: utf-8

# ## Preprocessing Eurosense Coverage Corpus

# In[1]:


import re
from lxml import etree
import numpy as np
import glob
import pickle
import numpy as np
import os
import scipy
import itertools
import pandas as pd
from keras.preprocessing import text
from keras.utils import to_categorical 
from keras.preprocessing import sequence
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import get_tmpfile


# In[2]:


def check_punc(word):
    """ Checks if word does not contain only punctuations
    """
    if not re.match(r'^[_\W]+$', word):
        return True
    else:
        return False


# In[3]:


file_path = "EuroSense/coverage.xml"
texts_en = [] # Sentences
synset_id_vector = [] # BabelnetId
lemma_vector = [] # Lemma
sense_vector = [] # Anchors
punc_lemma = 0 

def preprocess_eurosense(file_path):
    """ Parse xml file  provided in file_path and returns a sentence,
    babelnetId,lemma and anchors respectively for each sentences
    """
    context = etree.iterparse(file, events=('end',), tag='sentence')
    for event, elem in context:
        synset_id = []
        lemma = []
        sense = []
        try:
            for e in elem.iter():
                if e.tag == "text" and e.attrib.get('lang') == "en" and e.text == None:
                    raise Exception() 
                else:
                  if e.tag == "text" and e.attrib.get('lang') == "en" and e.text != None:
                        texts_en.append(e.text)
                  if e.tag == "annotations":
                    for f in e.iter():
                        if f.tag == "annotation" and f.attrib.get('lang') == "en":
                            l = f.attrib['lemma'].strip()
                            a = f.attrib['anchor'].strip()
                            sid = f.text.strip()
                            if len(l) > 0 and (check_punc(l)== True) and len(a) > 0 and len(sid) == 12:
                              if (l not in lemma) and (a not in sense) and (sid not in synset_id):
                                lemma.append(l)
                                sense.append(a)
                                synset_id.append(sid)
                            else:
                                punc_lemma += 1
            if(len(synset_id) == len(lemma) == len(sense)):     
                synset_id_vector.append(synset_id)
                lemma_vector.append(lemma)
                sense_vector.append(sense)
            else:
             print(synset_id)
             print(lemma)
             print(sense)
        except Exception:
            continue 
        # It's safe to call clear() here because no descendants will be accessed
        elem.clear()
        # Also eliminate now-empty references from the root node to <Title> 
        while elem.getprevious() is not None:
            del elem.getparent()[0]


# In[4]:


lemma_synid = [] # Vector of Lemma_synset_id for each sentences
def map_lemma_synsetID(lemma_vector,synset_id_vector):
    """ Concatenates lemma and synsetId for each sentences with '_'
    """
    for lemma,synid in zip(lemma_vector,synset_id_vector):
        syn = []
        for i,j in zip(lemma,synid):
              rep = i.replace(' ','_')+"_"+j
              syn.append(re.sub(r'\b'+re.escape(i)+r'\b', rep, i))
        lemma_synid.append(syn)


# In[5]:


coverage_corpus = []
texts_en_copy = texts_en # A copy of all sentences
def replace_anchors(sense_vector,lemma_synid,texts_en_copy):
    """ Replace anchors with lemma_synsetid in sentences
    """
    for j in range(len(sense_vector)):
      for a,b in zip(sense_vector[j],lemma_synid[j]):
        if a in texts_en_copy[j]:
            texts_en_copy[j] = re.sub(r'\b'+re.escape(a)+r'\b', b, texts_en_copy[j])
      coverage_corpus.append(texts_en_copy[j])


# ## Preprocessing SEW Corpus

# In[6]:


texts_en = []
synset_id_vector = []
sense_vector = []

def preprocess_sew(file_path):
  infile = file_path
  punc_lemma = 0
  context = etree.iterparse(infile, events=('end',), tag='wikiArticle',recover=True)
  
  for event, elem in context:
      synset_id = []
      sense = []

      try:
        if elem.attrib.get('language') == "EN":
          for e in elem.iter():
            if e.tag == "text" and e.text == None:
              raise Exception() 
            else:
              if e.tag == "text" and e.text != None:
                texts_en.append(e.text)
              if e.tag == "annotations":
                for f in e.iter():
                  if f.tag == "annotation":

                    for a in f.iter():
                      if a.tag == "babelNetID" and a.text != None:
                        ID = a.text
                      if a.tag == "mention" and a.text != None:
                        mention = a.text

                    if len(mention) > 0 and (check_punc(mention)== True) and len(ID) == 12:
                      if (ID not in synset_id) and (mention not in sense):
                        sense.append(mention)
                        synset_id.append(ID)
                    else:
                        punc_lemma += 1
          if(len(synset_id) == len(sense)):     
            synset_id_vector.append(synset_id)
            sense_vector.append(sense)
          else:
            print("Wrong Length")
      except Exception:
          continue  
    
      elem.clear()
      
      while elem.getprevious() is not None:
          del elem.getparent()[0]


# In[7]:


for file in glob.iglob('sew_conservative'+'/**/*xml', recursive=True):
    preprocess_sew(file)


# In[8]:


mention_synid = [] # Vector of Lemma_synset_id for each sentences
def map_lemma_synsetID(sense_vector,synset_id_vector):
    for lemma,synid in zip(sense_vector,synset_id_vector):
        syn = []
        for i,j in zip(lemma,synid):
              rep = i.replace(' ','_')+"_"+j
              syn.append(rep)
        mention_synid.append(syn)


# In[9]:


sew_corpus = []
texts_en_copy = texts_en # A copy of all sentences
def replace_anchors(sense_vector,mention_synid,texts_en_copy):
    """ Replace anchors with mention_synid in sentences
    """
    for j in range(len(sense_vector)):
      for a,c in zip(sense_vector[j],mention_synid[j]):
        if a in texts_en_copy[j]:
            try:
                texts_en_copy[j] = re.sub(r'\b'+re.escape(a)+r'\b', c, texts_en_copy[j])
            except:
               print(str(a) + " not found")
      sew_corpus.append(texts_en_copy[j])


# ## Tokenization of SEW and Eurosense Coverage Corpus 

# In[10]:


coverage_corpus = [] 
sew_corpus = []
all_dataset = coverage_corpus + sew_corpus
corpus = []
def tokenize_corpus(all_dataset):
    """ Takes a list of all sentences that needs to be tokenized,
    which would be fed into the word2vec model.
    """
    for sentence in all_dataset:
            try:
                corp = text.text_to_word_sequence(sentence,filters='!"#$%&()*+,./;<=>?@[\\]^`{|}~\t\n',split=' ')
                corpus.append(corp)
            except:
                pass


# ## Word2Vec Model Building and Training

# In[11]:


def build_vocabulary(corpus,dimension):
    """ Build vocabulary from corpus given its dimension
    """
    model = Word2Vec(size=dimension,workers = multiprocessing.cpu_count())
    model.build_vocab(corpus)
    model.save("vocab.model")

def train_model(corpus,epoch):
    """ Train word2vec cbow model
    """
    model.train(corpus,total_examples = model.corpus_count,epochs=epoch)
    model.wv.save_word2vec_format('embeddings.vec', binary=False)


# ## Filtering All Context Embeddings to lemma_synset Embeddings

# In[12]:


def load_embedding(file_path):
    """ 
    Takes the file path of the trained word2vec embeddings
    and load into memory.
    """
    with open(file_path) as f:
      embeddings = f.readlines()
      lemma_synset_embedding = []
      for context in embeddings:
        if "_" in context: 
          lemma_synset_embedding.append(context)


# In[13]:


dimension = 400 # Vector dimension that was used to train the word2vec model
def save_embedding(file_path):
    """ Takes the file path of where you want to save the lemma_synset embeddings
    and saves it to the disk.
    """
    with open('embeddings_lemma_id.vec','w') as f:
      f.write(str(len(lemma_synset_embedding)) + " "+str(dimension)+"\n")
      for s in lemma_synset_embedding:
        f.write(s.strip())
        f.write("\n")
    print("Done writing lemma_synset embeddings to "+file_path)


# In[14]:


# Usage of the above functions
load_embedding("best_embeddings_400_30.vec")
save_embedding("embeddings.vec")


# ## Visualizations

# In[15]:


from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('embeddings.vec', binary=False) # Load trained sense embeddings


# In[ ]:


# ## Word Similarity Task
# * In order to perform this task with sense embeddings you have to:
# * For each pair w1 , w2
#     * S1 = all sense embeddings associated with word w1
#     * S2 = all sense embeddings associated with word w2
#     * score = - 1.0
#     * For each pair s1 in S1 and s2 in S2 do
#         * score = max(score, cos( s1, s2 ) ) 
#         
# (cos == cosine similarity of two vectors)

# In[2]:


from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('embeddings.vec', binary=False) # Load trained sense embeddings


# In[16]:


# Load vectors as dictionaries
vocab = model.wv.vocab
dictionary = list(vocab.keys())
print("dictionary size is: {}".format(len(dictionary)))


# In[17]:


def load_sense_embeddings(word):
    '''returns the sense embeddings for a given word'''
    
    word = word.split()
    word = "_".join(word) if len(word)>1 else  word[0]
    wordSenses = []
    
    for w in dictionary:
        newWord = w.split(":")
        newWord = newWord[0].split("_")
        newWord = "_".join(newWord[:-1])
        if word.lower() == newWord.lower():
            wordSenses.append(w)
    
    return wordSenses


# In[18]:


def word_similarity(w1, w2):
    '''take two words and outputs a score of their similarity'''
    w1_senses = load_sense_embeddings(w1)
    w2_senses = load_sense_embeddings(w2)
    score = - 1.0
    
    if len(w1_senses)!=0 and len(w2_senses)!=0:
        combinations = itertools.product(w1_senses,w2_senses)
        for s1, s2 in combinations:
            score = max(score, model.wv.similarity(s1, s2))
            
    return score


# In[19]:


def get_spearman_correlation(file_path):
    gold = pd.read_csv(file_path, delimiter = '\t')
    gold['cosine'] = gold.apply(lambda row: word_similarity(row['Word 1'],row['Word 2']),axis=1)
    correlation, _ = scipy.stats.spearmanr(gold['Human (mean)'], gold['cosine'])
    return correlation


# In[20]:


print("Spearman correlation: " +str(get_spearman_correlation('combined.tab')))

# In[ ]: