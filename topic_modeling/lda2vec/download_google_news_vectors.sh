#!/bin/bash
#
# Download Google News word vectors
#


wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz" -O tmp.bin.gz
gunzip -c tmp.bin.gz > $GOOGLE_NEWS_VECTORS
rm tmp.bin.gz
