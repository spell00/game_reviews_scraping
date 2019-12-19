

First, run 

python scrape.py

then

python vectorize --train 

and finally

python vectorize --valid

Linear Discriminant Analysis and the most discriminant features are found by running the notebooks


The model ran on Mac OSX Mojave but not on Ubuntu 18.04, though the Ubuntu machine was 
much better on every specs. It must have to do with the memory usage. On Ubuntu (and Windnows for sklearn's 
Linear Discriminant Analysis)  we experiences memory errors when running large models, like when using more than 2000 features
for the LDA for the VectorCounts, which can be limiting, or to train an embedding model using pretrained word2vec to
fine-tune it on a new text corpus.

At the moments, these models only run on Mac OS X  