import os
import sys
from gensim.models.keyedvectors import KeyedVectors

def load_word_vectors():
  word_vectors_model_filename = "german.model"
  print("Loading word vectors from",  word_vectors_model_filename + "...")
  word_vectors_model_size = os.path.getsize(word_vectors_model_filename) / (1024.0 * 1024.0)
  print("File size is {0:.2f}MB".format(word_vectors_model_size))
  print("Be patient! This might take a while...")
  word_vectors_model = None
  try:
    print("Attempting to load as vector-format...")
    word_vectors_model = KeyedVectors.load_word2vec_format(word_vectors_model_filename, binary=False)
    print("Success!")
  except:
    print("Failed!")
    try:
      print("Attempting to load as binary-format...")
      word_vectors_model = KeyedVectors.load_word2vec_format(word_vectors_model_filename, binary=True)
    except:
      print("Failed!")
      exit(0)
   
  print("Success!")
  return word_vectors_model

def main(inword):
  model = load_word_vectors()
  most_similar_topten = model.most_similar(positive=[inword], topn=10)
  print("Most similar top-10:")
  for i, most_similar in enumerate(most_similar_topten):
    print(str(i + 1) + ":", most_similar)
  
  
if __name__=="__main__":
  main(sys.argv[1])