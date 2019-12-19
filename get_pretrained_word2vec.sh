#!/usr/bin/env bash
cd data
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
unzip GoogleNews-vectors-negative300.bin.gz
del GoogleNews-vectors-negative300.bin.gz