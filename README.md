# MultiCQA: Zero-Shot Transfer of Self-Supervised Text Matching Models on a Massive Scale

This repository contains the data and code to reproduce the results of our paper: 
https://arxiv.org/abs/2010.00980

Please use the following citation:

```
@inproceedings{rueckle-etal-2020-multicqa,
    title = "{MultiCQA}: Zero-Shot Transfer of Self-Supervised Text Matching Models on a Massive Scale",
    author = {R{\"u}ckl{\'e}, Andreas  and
      Pfeiffer, Jonas and
      Gurevych, Iryna},
    booktitle = "Proceedings of The 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP-2020)",
    year = "2020",
    address = "Virtual Conference",
    url = "https://arxiv.org/abs/2010.00980",
}
```

> **Abstract:** We study the zero-shot transfer capabilities of text matching models on a massive scale, by self-supervised training on 140 source domains from community question answering forums in English. We investigate the model performances on nine benchmarks of answer selection and question similarity tasks, and show that all 140 models transfer surprisingly well, where the large majority of models substantially outperforms common IR baselines. We also demonstrate that considering a broad selection of source domains is crucial for obtaining the best zero-shot transfer performances, which contrasts the standard procedure that merely relies on the largest and most similar domains. In addition, we extensively study how to best combine multiple source domains. We propose to incorporate self-supervised with supervised multi-task learning on all available source domains. Our best zero-shot transfer model considerably outperforms in-domain BERT and the previous state of the art on six benchmarks. Fine-tuning of our model with in-domain data results in additional large gains and achieves the new state of the art on all nine benchmarks.


Contact person: Andreas Rücklé

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


> This repository contains experimental software and is published for the sole purpose of giving additional background 
  details on the respective publication. 


## Usage

Our source code consists of two components:

1. [**./bert-ranker**](./bert-ranker): Code to train and evaluate BERT/RoBERTa models for our tasks.

2. [**./data-creation**](./data-creation): Code to create training data for _all_ StackExchange forums.


We also provide several pre-trained models:
 * [A BERT-base model containing all adapters](https://public.ukp.informatik.tu-darmstadt.de/rueckle/multicqa/checkpoints/bert-base-all-adapters.zip) (e.g., useful for AdapterFusion)
 * (More soon)
 
Our adapters are also available at [AdapterHub.ml](https://adapterhub.ml/explore/sts/stackexchange/).