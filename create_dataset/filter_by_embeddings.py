import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

#process metamap results
metamap_df = pd.read_csv('metamap_result.txt', sep="|", header=None)
phrases = pd.read_csv('phrases.txt', sep="|", header=None)

#join phrases and metamap results
phrases[0] = phrases[0].astype(str)
phrases_metamap_df = phrases.merge(df, how='left', on=0)
phrases_metamap_df = phrases_metamap_df[['1_x', 2, 3, 4]]
phrases_metamap_df.columns = ['phrases', 'MMI_score', 'concepts', 'UMLS Identifier']
phrases_metamap_df['concepts'] = phrases_metamap_df['concepts'].astype(str)

#phrases and concepts
phrases = phrases_metamap_df['phrases'].tolist()
concepts = phrases_metamap_df['concepts'].tolist()

#determine neighbors for each concept
concept_neighbors = {}
for i in range(len(concepts)):
  if concepts[i] in concept_neighbors:
    concept_neighbors[concepts[i]].append(phrases[i])
  else:
    concept_neighbors[concepts[i]] = []
	

#initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

#compute embeddings
cosine_sims = {}
for concept in concept_neighbors:
  if concept_neighbors[concept]:
    concepts_embed = model.encode([concept], convert_to_tensor=True)
    phrases_embed = model.encode(concept_neighbors[concept], convert_to_tensor=True)

    #compute cosine similarity
    cosine_sims[concept] = util.cos_sim(concepts_embed, phrases_embed)

#use k nearest phrases with K=9 to select only phrases closely aligned with concepts
K=9

result_set = []
for concept in concept_neighbors:
  neighbors = []
  for i in range(len(concept_neighbors[concept])):
    cos_sims = cosine_sims[concept][0].tolist()
    neighbors.append((concept_neighbors[concept][i], cos_sims[i]))
  
  #sort neighbors
  neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)

  #put top K neighbors based on cosine similarity into the result set
  for t in neighbors[0:K]:
    result_set.append([concept, t[0], t[1]])

res = pd.DataFrame(result_set, columns=["concepts", "phrases", "cosine similarity"])
res.merge(phrases_metamap_df, on=['phrases', 'concepts']).to_csv('supervised_dataset.txt', sep="|", index=False)