---
# Stock sentiment analysis
This repository contains a benchmark system for our paper <b>Classification and Analysis for Chinese Stock Investor Sentiment: Using SeqGAN as Oversampling Method</b>.

## Dependencies
- Python 3.6.x
- Tensorflow(newest version)
- Anaconda(newest version)


## Word Embeddings
The word embeddings we adopted comes from [here](https://github.com/Embedding/Chinese-Word-Vectors), we adopt the financial news word embeddings.

## SeqGAN model
We adopt SeqGAN model in the platform of [Texygen](https://github.com/geek-ai/Texygen)


## Usage
1.Change corresponding file path in run_cnn.py(including corpus path,word embeddings path and model path)
2.Training:Run the run_cnn.py by command in commanding window:python run_cnn.py train     
3.Predicting:Run the run_cnn.py by command in commanding window:python run_cnn.py test

