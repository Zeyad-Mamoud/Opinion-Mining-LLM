---
dataset_info:
  features:
  - name: sentence_id
    dtype: large_string
  - name: split
    dtype: large_string
  - name: domain
    dtype: large_string
  - name: input
    dtype: large_string
  - name: output
    dtype: large_string
  - name: instruction
    dtype: large_string
  splits:
  - name: train
    num_bytes: 1535822
    num_examples: 5959
  - name: test
    num_bytes: 419424
    num_examples: 1572
  - name: dev
    num_bytes: 45614
    num_examples: 200
  download_size: 563723
  dataset_size: 2000860
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
  - split: dev
    path: data/dev-*
task_categories:
- text-generation
language:
- en
tags:
- reviews
- absa
- sentiment
pretty_name: SemEval 2014 Task 4 Aspect-Based Sentiment Analysis
size_categories:
- 1K<n<10K
---

Examples with one or more aspects that were labeled with the polarity 'conflict' were excluded. Examples are formatted in the Alpaca format. The purpose is to train an LLM top predict the aspects (output) based on the text (input).

@inproceedings{pontiki_semeval-2014_2014,
	title = {{SemEval}-2014 {Task} 4: {Aspect} {Based} {Sentiment} {Analysis}},
	doi = {10.3115/v1/S14-2004},
	booktitle = {Proceedings of the 8th {International} {Workshop} on {Semantic} {Evaluation} ({SemEval} 2014)},
	publisher = {Association for Computational Linguistics},
	author = {Pontiki, Maria and Galanis, Dimitris and Pavlopoulos, John and Papageorgiou, Harris and Androutsopoulos, Ion and Manandhar, Suresh},
	year = {2014},
	pages = {27--35},
}
