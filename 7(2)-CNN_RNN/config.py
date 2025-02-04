from typing import Literal
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
d_model = 1024
# print(device)

# Word2Vec
window_size = 7
method: Literal["cbow", "skipgram"] = "cbow"
lr_word2vec = 1e-03
num_epochs_word2vec = 60

# GRU
hidden_size = 1024
num_classes = 4
lr = 1e-03
num_epochs = 100
batch_size = 128