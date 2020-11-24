"""Run lda2vec."""

import os
import os.path
import pickle
import time
import shelve

import chainer
from chainer import cuda
from chainer import serializers
import chainer.optimizers as O
import numpy as np

from lda2vec import utils
from lda2vec import prepare_topics, print_top_words_per_topic, topic_coherence
from lda2vec_model import LDA2Vec


gpu_id = int(os.getenv("CUDA_GPU", 0))
cuda.get_device(gpu_id).use()
print("Using GPU", gpu_id)

fn_vocab = os.getenv("VOCAB_FILE", "vocab.pkl")
fn_corpus = os.getenv("CORPUS_FILE", "corpus.pkl")
fn_flatnd = os.getenv("FLATTENED_FILE", "flattened")
fn_docids = os.getenv("DOC_IDS_FILE", "doc_ids")
vocab = pickle.load(open(fn_vocab, 'rb'))
corpus = pickle.load(open(fn_corpus, 'rb'))
flattened = np.load(fn_flatnd)
doc_ids = np.load(fn_docids)

# Model Parameters
# Number of documents
n_docs = doc_ids.max() + 1
# Number of unique words in the vocabulary
n_vocab = flattened.max() + 1

# 'Strength' of the dircihlet prior
clambda = float(os.getenv("DIRICHLET_STRENGTH", 200))

# Number of topics to fit
n_topics = int(os.getenv("N_TOPICS", 20))

# Batch size
batchsize = int(os.getenv("BATCH_SIZE", 4096))

# Number of epochs
n_epochs = int(os.getenv("N_EPOCHS", 200))

# Power for neg sampling
power = float(os.getenv("POWER", 0.75))

# Intialize with pretrained word vectors
pretrained = False if os.getenv("PRETRAINED", "False") == "False" else True

# Sampling temperature
temperature = float(os.getenv("TEMPERATURE", 1.0))

# Number of dimensions in a single word vector
n_units = int(os.getenv("N_UNITS", 300))

# Get the string representation for every compact key
words = corpus.word_list(vocab)[:n_vocab]

# How many tokens are in each document
doc_idx, lengths = np.unique(doc_ids, return_counts=True)
doc_lengths = np.zeros(doc_ids.max() + 1, dtype='int32')
doc_lengths[doc_idx] = lengths

# Count all token frequencies
tok_idx, freq = np.unique(flattened, return_counts=True)
term_frequency = np.zeros(n_vocab, dtype='int32')
term_frequency[tok_idx] = freq

for key in sorted(locals().keys()):
    val = locals()[key]
    if len(str(val)) < 100 and '<' not in str(val):
        print(key, val)

model = LDA2Vec(n_documents=n_docs, n_document_topics=n_topics,
                n_units=n_units, n_vocab=n_vocab, counts=term_frequency,
                n_samples=15, power=power, temperature=temperature)

model_file = os.getenv("LDA_HDF5_FILE", "lda2vec.hdf5")
if os.path.exists(model_file):
    print("Reloading from saved")
    serializers.load_hdf5(model_file, model)
if pretrained:
    fn_vectors = os.getenv("VECTORS_FILE", "vectors")
    vectors = np.load(fn_vectors)
    model.sampler.W.data[:, :] = vectors[:n_vocab, :]

model.to_gpu()
optimizer = O.Adam()
optimizer.setup(model)
clip = chainer.optimizer.GradientClipping(5.0)
optimizer.add_hook(clip)

j = 0
epoch = 0
fraction = batchsize * 1.0 / flattened.shape[0]
with shelve.open(os.getenv("PROGRESS_FILE", "progress.shelve")) as progress:
    for epoch in range(n_epochs):
        data = prepare_topics(cuda.to_cpu(model.mixture.weights.W.data).copy(),
                            cuda.to_cpu(model.mixture.factors.W.data).copy(),
                            cuda.to_cpu(model.sampler.W.data).copy(),
                            words)
        top_words = print_top_words_per_topic(data)
        if j % 100 == 0 and j > 100:
            coherence = topic_coherence(top_words)
            for j in range(n_topics):
                print(j, coherence[(j, 'cv')])
            kw = dict(top_words=top_words, coherence=coherence, epoch=epoch)
            progress[str(epoch)] = pickle.dumps(kw)
        data["doc_lengths"] = doc_lengths
        data["term_frequency"] = term_frequency
        np.savez(os.getenv("LDA_VIZ_FILE", "topics.pyldavis"), **data)
        for d, f in utils.chunks(batchsize, doc_ids, flattened):
            t0 = time.time()
            model.cleargrads()
            # optimizer.zero_grads()
            l = model.fit_partial(d.copy(), f.copy())
            prior = model.prior()
            loss = prior * fraction
            loss.backward()
            optimizer.update()
            msg = ("J:{j:05d} E:{epoch:05d} L:{loss:1.3e} "
                "P:{prior:1.3e} R:{rate:1.3e}")
            prior.to_cpu()
            loss.to_cpu()
            t1 = time.time()
            dt = t1 - t0
            rate = batchsize / dt
            logs = dict(loss=float(l), epoch=epoch, j=j,
                        prior=float(prior.data), rate=rate)
            print(msg.format(**logs))
            j += 1
        serializers.save_hdf5(model_file, model)
