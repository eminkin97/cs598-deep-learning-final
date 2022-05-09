import numpy as np
import json
import pandas as pd
import stanza              #stanford nlp library
import phrasemachine
import requests

#Read the data
with open('data.jl') as f:
    lines = f.readlines()
  
#parse into json
for i in range(len(lines)):
    lines[i] = json.loads(lines[i])

#convert to pandas dataframe
lines_df = pd.DataFrame(lines)

#filter out rows without either postContent or postHeading
lines_df = lines_df[~lines_df["postHeading"].isna() | ~lines_df["postContent"].isna()]

#initialize stanza
stanza.download('en')       # This downloads the English models for the neural pipeline

#create pipeline
nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos') # This sets up a default neural pipeline in English

headings = lines_df[~lines_df["postHeading"].isna()]["postHeading"].tolist()
contents = lines_df[~lines_df["postContent"].isna()]["postContent"].tolist()

#extract phrases
phrases = []
for h in headings + contents:
  doc = nlp(h)
  tokens = [word.text for sent in doc.sentences for word in sent.words]
  pos = [word.upos for sent in doc.sentences for word in sent.words]

  for phrase in list(phrasemachine.get_phrases(tokens=tokens, postags=pos)['counts']):
    phrases.append(phrase)

#Use metamap rest API to get the associated medical terms
def get_medical_term(phrase):
  url = 'https://ii.nlm.nih.gov/metamaplite/rest/annotate'
  acceptfmt = 'text/plain'
  params = [('inputtext', phrase), ('docformat', 'freetext'), ('resultformat', 'json'), ('sourceString', 'all'), ('semanticTypeString', 'all')]

  headers = {'Accept' : acceptfmt}
  return requests.post(url, params, headers=headers).json()

#associate medical term with each phrase
dataset = []
for phrase in phrases:
  resp = get_medical_term(phrase)
  concepts = []
  #print(resp)
  if resp:
    for r in resp[0]['evlist']:
      concepts.append(r['conceptinfo']['conceptstring'])

  concepts = list(set(concepts))
  for c in concepts:
    dataset.append([phrase,c])
  
#output results to csv
pd.DataFrame(dataset, columns=['phrases', 'concepts'])['phrases'].to_csv('phrases.csv', index=True, sep="|")