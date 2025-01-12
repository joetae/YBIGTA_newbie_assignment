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


import sys


"""
TODO:
- 일단 Trie부터 구현하기
- count 구현하기
- main 구현하기
"""


def count(trie: Trie, query_seq: str) -> int:
    """
    trie - 이름 그대로 trie
    query_seq - 단어 ("hello", "goodbye", "structures" 등)

    returns: query_seq의 단어를 입력하기 위해 버튼을 눌러야 하는 횟수
    """
    pointer = 0
    cnt = 1
    first_char = query_seq[0]

    for child_idx in trie[pointer].children:
        if trie[child_idx].body == first_char:
            pointer = child_idx
            break

    for char in query_seq[1:]:
        if len(trie[pointer].children) > 1 or trie[pointer].is_end:
            cnt = cnt + 1

        for child_idx in trie[pointer].children:
            if trie[child_idx].body == char:
                pointer = child_idx
                break
    
    return cnt

def main() -> None:
    input_data = sys.stdin.read().strip().split()
    idx = 0
    
    while idx < len(input_data):
        n = int(input_data[idx])
        idx = idx + 1
        if n <= 0:
            break

        trie: Trie = Trie()
        words = input_data[idx:idx+n]
        idx += n
        for w in words:
            trie.push(w)

        total_pressed = 0
        for w in words:
            total_pressed += count(trie, w)

        average = total_pressed / n
        print(f"{average:.2f}")

if __name__ == "__main__":
    main()