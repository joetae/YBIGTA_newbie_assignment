from lib import Trie
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
        idx = idx + n
        for w in words:
            trie.push(w)

        total_pressed = 0
        for w in words:
            total_pressed = total_pressed + count(trie, w)

        average = total_pressed / n
        print(f"{average:.2f}")

if __name__ == "__main__":
    main()