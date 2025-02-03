# 구현하세요!
from datasets import load_dataset

def load_corpus() -> list[str]:
    corpus: list[str] = []
    # 구현하세요!
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    for row in dataset["train"]:
        corpus.append(row["verse_text"])
        
    return corpus