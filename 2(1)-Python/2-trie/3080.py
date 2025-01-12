from lib import Trie
import sys


"""
TODO:
- 일단 Trie부터 구현하기
- main 구현하기

힌트: 한 글자짜리 자료에도 그냥 str을 쓰기에는 메모리가 아깝다...
"""


def main() -> None:
    # 구현하세요!
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