import os
import sys
#import sqlite3
from gensim.models.keyedvectors import KeyedVectors
import spacy
import numpy as np

#def loadDB():
#  connection = sqlite3.connect("German_English_dict_or.db")
#  cursor = connection.cursor()

def load_spacy_model():
  #for similarity
#  nlp = spacy.load('de_core_news_md')
  nlp = spacy.load('de_core_news_sm')
  return nlp

def load_word_vectors():
  word_vectors_model_filename = "german.model"
  print("Loading word vectors from",  word_vectors_model_filename + "...")
  word_vectors_model_size = os.path.getsize(word_vectors_model_filename) / (1024.0 * 1024.0)
  print("File size is {0:.2f}MB".format(word_vectors_model_size))
  print("Be patient! This might take a while...")
  word_vectors_model = None
  try:
    print("Attempting to load as binary-format...")
    word_vectors_model = KeyedVectors.load_word2vec_format(word_vectors_model_filename, binary=True)
  except:
    print("Failed!")
    exit(0)
   
  print("Success!")
  return word_vectors_model

def get_lemma(word, spacy_model):
  tok = spacy_model.tokenizer(word)
  lemma = tok[0].lemma_
  return lemma
  
  
def sort_by_lemma(similar_list, spacy_model):
  words = ' '.join([s[0] for s in similar_list])
  tok = spacy_model.tokenizer(words)
  lemmas = [t.lemma_ for t in tok]
  
  sorted_by_lemmas = {}
  for ii in range(len(similar_list)):
    if lemmas[ii] not in sorted_by_lemmas:
      sorted_by_lemmas[lemmas[ii]] = []
    sorted_by_lemmas[lemmas[ii]].append(similar_list[ii])
  return sorted_by_lemmas
 
def most_similar_spacy(word, spacy_model, topn=15):
  tvec = spacy_model(word).vector
  result = spacy_model.vocab.vectors.most_similar(tvec.reshape(1,tvec.shape[0]), n=topn)
  print(result)
  #dec = spacy_model.vocab.vectors.find(rows=result[0][0].tolist())
  dec_ = [spacy_model.vocab.strings[r] for r in result[0][0].tolist()]
  #print(dec)
  print("dec "+str(dec_))
  return result

def main(inword):
  model = load_word_vectors()
  spacy_model = load_spacy_model()
  most_similar_topten = model.most_similar(positive=[inword], topn=15)
  #most_similar_topten = most_similar_spacy(inword, spacy_model, topn=15)
  sorted_by_lemma = sort_by_lemma(most_similar_topten, spacy_model)
  print("\nQuery: "+inword+' , Lemma: '+get_lemma(inword, spacy_model)+'\n')
  for l in sorted_by_lemma:
    print(l+'\n=========')
    for w in sorted_by_lemma[l]:
      print(w)
    
  
if __name__=="__main__":
  main(sys.argv[1])
