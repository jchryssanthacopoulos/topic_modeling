"""Train BERT-based cluster model."""

import hdbscan
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap


# load messages
messages = pd.read_csv('messages_100k.csv')
messages = messages['message'].tolist()

# get BERT embeddings
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(messages, show_progress_bar=True)

# use UMAP to reduce dimensions
umapper = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine')
umap_embeddings = umapper.fit_transform(embeddings)

# cluster using HDBSCAN
hdbscanner = hdbscan.HDBSCAN(min_cluster_size=15,
    metric='euclidean', cluster_selection_method='eom')
cluster = hdbscanner.fit(umap_embeddings)
