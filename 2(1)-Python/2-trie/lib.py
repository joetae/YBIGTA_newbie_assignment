from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable


"""
TODO:
- Trie.push 구현하기
- (필요할 경우) Trie에 추가 method 구현하기
"""


T = TypeVar("T")


@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: list[int] = field(default_factory=lambda: [])
    is_end: bool = False


class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable[T]) -> None:
        """
        seq: T의 열 (list[int]일 수도 있고 str일 수도 있고 등등...)

        action: trie에 seq을 저장하기
        """
        # 구현하세요!
        cur = 0  
        for char in seq:
            found = False
            for child_idx in self[cur].children:
                if self[child_idx].body == char:
                    cur = child_idx
                    found = True
                    break

            if not found:
                new_idx = len(self)
                self.append(TrieNode(body=char))
                self[cur].children.append(new_idx)
                cur = new_idx

        self[cur].is_end = True

    def count_answer(self, mod: int = 1000000007) -> int:
        
        factorial_cache = {0: 1}

        def factorial(n: int) -> int:
            if n not in factorial_cache:
                factorial_cache[n] = (factorial(n - 1) * n) % mod
            return factorial_cache[n]
        
        def dfs(node_index: int) -> int:
            node = self[node_index]
            num_children = len(node.children)
            result = 1

            for child_index in node.children:
                result = result * dfs(child_index) % mod

            result = result * factorial(num_children) % mod

            if node.is_end:
                result = result * (num_children + 1) % mod

            return result
        
        return dfs(0)