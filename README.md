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

This script takes the data.jl as input and outputs a csv file `phrases.txt` as output. This csv is the input to the next process.

The process extracts the noun phrases using [stanza](https://stanfordnlp.github.io/stanza/), the linguistics analysis library developed by Stanford NLP.
It then determines the formal medical terms for each extracted colloquial medical phrase using [MetaMap](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap.html)

### Medical Entity Linking

[MetaMap Batch Service](https://ii.nlm.nih.gov/Batch/UTS_Required/MetaMap.html) is a publicly available service provided by the National Library
of Medicine for linking free text to formal UMLS concepts. This service is used to obtain formal concepts for the phrases obtained in the previous section.
We set the following parameters for MetaMap:
- MMI Output
- Term Processing
- Single Line Delimited Input
- Ignore word order

The resulting csv file obtained from the batch service is included in the github directory. See `metamap_result.txt`

### Filter by Embeddings

This script takes `phrases.txt` and `metamap_result.txt` as inputs and outputs a csv file `supervised_dataset.txt`.

The process creates embeddings from the dataset obtained after entity linking using a pretrained MiniLCM model from the [Sentence Transformers](https://www.sbert.net/index.html) library; 
these embeddings are used to calculate cosine similarities between the informal and formal medical phrases. 
A k-nearest phrases approach is implemented to filter the dataset so that only the phrases with the highest cosine similarity are retained for each medical concept.

## Evaluate Dataset

This process evaluates the distantly supervised dataset constructed in the previous sections on a two layer feed forward neural network.

### Read Cadec

This script, `read_cadec.py`, reads the [CADEC](https://pubmed.ncbi.nlm.nih.gov/25817970/) dataset that will be used as the test set for evaluating the performance of the model.
This dataset is in the AMT-SCT subfolder and has informal medical phrases obtained from [askapatient.com](https://www.askapatient.com) mapped to SNOMED Clinical Terms.
The output of this script is `cadec_dataset.txt`

### Train

This script trains a two-layer feedforward neural network using the supervised_dataset obtained in the previous sections. 
It then uses a 1 nearest neighbor approach to classify each informal medical phrase to a concept and outputs the observed cosine similarity.
The cadec dataset is used to evaluate the performance of the model after each iteration of training.

The inputs to this script are `supervised_dataset.txt` and `cadec_dataset.txt`
