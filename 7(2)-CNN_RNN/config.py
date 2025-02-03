from typing import Literal


device = "cpu"
d_model = 256

# Word2Vec
window_size = 7
method: Literal["cbow", "skipgram"] = "skipgram"
lr_word2vec = 1e-04 * 5
num_epochs_word2vec = 10

# GRU
hidden_size = 256
num_classes = 4
lr = 1e-03 * 5
num_epochs = 50
batch_size = 32