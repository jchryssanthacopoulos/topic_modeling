# Topic Modeling
This is a testbed for evaluating different topic modeling algorithms.

## lda2vec

Copy `.env.sample` to `.env`, fill in the lda2vec portion of the config parameters, then source them into your
environment. Make sure `DOCUMENTS_FILE` points to the correct location (use the `data` directory for this purpose).

Download the Google News word vectors with:

```bash
topic_modeling/lda2vec/download_google_news_vectors.sh
```

Then preprocess the data and save the files with:

```bash
python topic_modeling/lda2vec/preprocess.py
```

Finally, to train the model, run:

```bash
python topic_modeling/lda2vec/lda2vec_run.py
```

To install the Jupyter kernel to visualize the results, run:

```bash
./install_new_kernel.sh
```

Then load `notebooks/lda2vec.ipynb` with the `topic_modeling` kernel.
