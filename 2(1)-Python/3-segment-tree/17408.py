from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


class Pair(tuple[int, int]):
    """
    힌트: 2243, 3653에서 int에 대한 세그먼트 트리를 만들었다면 여기서는 Pair에 대한 세그먼트 트리를 만들 수 있을지도...?
    """
    def __new__(cls, a: int, b: int) -> 'Pair':
        return super().__new__(cls, (a, b))

    @staticmethod
    def default() -> 'Pair':
        """
        기본값
        이게 왜 필요할까...?
        """
        return Pair(0, 0)

    @staticmethod
    def f_conv(w: int) -> 'Pair':
        """
        원본 수열의 값을 대응되는 Pair 값으로 변환하는 연산
        이게 왜 필요할까...?
        """
        return Pair(w, 0)

    @staticmethod
    def f_merge(a: Pair, b: Pair) -> 'Pair':
        """
        두 Pair를 하나의 Pair로 합치는 연산
        이게 왜 필요할까...?
        """
        return Pair(*sorted([*a, *b], reverse=True)[:2])

    def sum(self) -> int:
        return self[0] + self[1]


def main() -> None:
    # 구현하세요!
    data_list = sys.stdin.read().splitlines()

    n = int(data_list[0])
    array_list = list(map(int, data_list[1].split()))
    m = int(data_list[2])
    query = data_list[3:]

    # 처음에 초기화해줄 때에는 어차피 단일값만 존재해서
    # f_conv를 통해 쌍 형태만 만들어주고 단일값만 넣어줌.
    tree: SegmentTree = SegmentTree(n)
    """
    for i, value in enumerate(array_list, start=1):
        tree.update(1, 1, n, i, Pair.f_conv(value))

    ans_save = []
    for task in query:
        task_split = task.split()
        if task_split[0] == "1":
            i, v = int(task_split[1]), int(task_split[2])
            tree.update(1, 1, n, i, Pair.f_conv(v))
        elif task_split[0] == "2":
            l, r = int(task_split[0]), int(task_split[1])
            ans = tree.query(1, 1, n, l, r).sum()
            ans_save.append(ans)

    for ans in ans_save:
        print(ans)
"""

if __name__ == "__main__":
    main()