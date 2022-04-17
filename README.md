# cs598-deep-learning-final

## Prerequisites

This code was tested with Python 3.6.7
The dependencies are listed in requirements.txt and can be installed with `pip`

## Scraper

This process scrapes the data from [patient.info/forums](http://patient.info/forums)
and outputs json lines containing the crawled content to data.jl which is the input to generate dataset

Command: `scrapy runspider scraper.py -o data.jl`

## Generate Dataset

This process consists of two separate scripts.

### Extract Phrases

This script takes the data.jl as input and outputs a csv file `result_dataset.csv` as output. This csv is the input to the next process.

The process extracts the noun phrases using [stanza](https://stanfordnlp.github.io/stanza/), the linguistics analysis library developed by Stanford NLP.
It then determines the formal medical terms for each extracted colloquial medical phrase using [MetaMap](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap.html)

### Filter by Embeddings

This script takes `result_dataset.csv` as input and outputs a csv file `final_dataset.csv`.

The process creates embeddings from the previous dataset using a pretrained MiniLCM model from the [Sentence Transformers](https://www.sbert.net/index.html) library; 
these embeddings are used to calculate cosine similarities between the informal and formal medical phrases. 
A k-nearest phrases approach is implemented to filter the dataset so that only the phrases with the highest cosine similarity are retained for each medical concept.