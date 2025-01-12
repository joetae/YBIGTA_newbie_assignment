from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable


"""
TODO:
- Trie.push �����ϱ�
- (�ʿ��� ���) Trie�� �߰� method �����ϱ�
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
        seq: T�� �� (list[int]�� ���� �ְ� str�� ���� �ְ� ���...)

        action: trie�� seq�� �����ϱ�
        """
        # �����ϼ���!
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
- �ϴ� Trie���� �����ϱ�
- main �����ϱ�

��Ʈ: �� ����¥�� �ڷῡ�� �׳� str�� ���⿡�� �޸𸮰� �Ʊ���...
"""


def main() -> None:
    # �����ϼ���!
    data_list = sys.stdin.read().splitlines()
    N = int(data_list[0])
    name_list = data_list[1:]

    trie: Trie = Trie()
    for name in name_list:
        trie.push(name)

    MOD = 1000000007

    fact = [1]*(N+1)
    for i in range(1, N+1):
        fact[i] = (fact[i-1]*i) % MOD

    def solve(words):
        if not words:
            return 1

        end_count = sum(1 for w in words if len(w) == 0)

        groups = {}
        for w in words:
            if w:  
                c = w[0]
                groups.setdefault(c, []).append(w[1:])

        k = len(groups)
        
        result = 1
        for g in groups.values():
            result = (result * solve(g)) % MOD

        result = (result * fact[k]) % MOD

        if end_count > 0:
            result = (result * pow(k+1, end_count, MOD)) % MOD

        return result

    answer = solve(name_list)
    print(answer % MOD)


if __name__ == "__main__":
    main()