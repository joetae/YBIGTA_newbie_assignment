import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal
from config import *

# 구현하세요!


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        # 구현하세요!
        self.to(device)

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach().to(device)

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        # 구현하세요!
        token_corpus = [tokenizer.encode(sentence, add_special_tokens=False) for sentence in corpus]
        
        for epoch in range(num_epochs):
            total_loss: float = 0
            for sentence in token_corpus:
                if self.method == "skipgram":
                    loss = self._train_skipgram(sentence, criterion, optimizer)
                elif self.method == "cbow":
                    loss = self._train_cbow(sentence, criterion, optimizer)
                total_loss = total_loss + float(loss)
            print(f"epoch {epoch+1}, loss: {total_loss}")

    def _train_cbow(
        self,
        # 구현하세요!
        sentence: list[int],
        criterion: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> float:
        # 구현하세요!
        total_loss: float = 0
        for i in range(self.window_size, len(sentence) - self.window_size):
            context = sentence[i - self.window_size:i] + sentence[i + 1:i + 1 + self.window_size]
            target = sentence[i]

            context_embeds = self.embeddings(LongTensor(context).to(device)).mean(dim=0)
            output = self.weight(context_embeds)
            loss = criterion(output.unsqueeze(0), LongTensor([target]).to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = total_loss + float(loss.item())
        return total_loss

    def _train_skipgram(
        self,
        # 구현하세요!
        sentence: list[int],
        criterion: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> float:
        # 구현하세요!
        total_loss:float = 0
        for i in range(self.window_size, len(sentence) - self.window_size):
            target = sentence[i]
            context = sentence[i - self.window_size:i] + sentence[i + 1:i + 1 + self.window_size]

            target_embed = self.embeddings(LongTensor([target]).to(device)) 
            optimizer.zero_grad()

            for context_word in context:
                output = self.weight(target_embed).squeeze(0)
                loss = criterion(output.unsqueeze(0), LongTensor([context_word]).to(device))

                loss.backward(retain_graph=True) 
                optimizer.step()
                optimizer.zero_grad()

                total_loss = total_loss + float(loss.item())

        return total_loss