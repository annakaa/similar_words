import spacy
import sys


def load_spacy_model():
  nlp = spacy.load('de_core_news_sm')
  return nlp



def main(sentence):
  spacy_model = load_spacy_model()
  tok = spacy_model(sentence)
  lemma = [t.lemma_ for t in tok]
  pos = [t.pos_ for t in tok]   
  words = [t.text for t in tok]
  all_info = zip(words, lemma, pos)
  for ii in range(len(words)):
    print(words[ii]+' '+lemma[ii]+' '+pos[ii])

  
if __name__=="__main__":
  main(' '.join(sys.argv[1:]))