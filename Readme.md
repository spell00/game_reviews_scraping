

First, run 

bash get_pretrained_word2vec.sh

To get the pretrained GoogleNews-vectors-negative300.bin.gz for word2vec, then extract it in the same folder 
(you can delete the .gz file after)

Then, scrape the game reviews with

python scrape.py

Then, once the reviews have been extracted and normalized, run

python vectorize --train 

and finally

python vectorize --valid

Linear Discriminant Analysis and the discriminant features are found by running the notebooks 
(LDA_1000 contains the code for cross-validation)

Lists of words found with word2vec related to the game reviews are found by running the notebook word2vec

The model ran on Mac OSX Mojave but not on Ubuntu 18.04, though the Ubuntu machine was 
much better on every specs. It must have to do with the memory usage. On Ubuntu (and Windnows for sklearn's 
Linear Discriminant Analysis)  we experiences memory errors when running large models, like when using more than 2000 features
for the LDA for the VectorCounts, which can be limiting, or to train an embedding model using pretrained word2vec to
fine-tune it on a new text corpus.

This project was made for the class IFT-6285 (Natural Language Processing) at Universite de Montreal, so the report (rapport.pdf) is in french.
