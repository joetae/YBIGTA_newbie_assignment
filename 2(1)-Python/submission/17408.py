from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable


"""
TODO:
- SegmentTree �����ϱ�
"""


T = TypeVar("T")
U = TypeVar("U")


class SegmentTree(Generic[T, U]):
    # �����ϼ���!
    def __init__(self, size: int) -> None:
        self.size = size
        self.tree = [0] * (4*size)
    
    def query(self, node: int, start: int, end: int, rank: int) -> int:
        if start == end:
            return start
        
        mid = (start + end) // 2
        left = self.tree[node*2]
        if left >= rank:
            return self.query(node*2, start, mid, rank)
        else:
            return self.query(node*2+1, mid+1, end, rank-left)
        

    def query_range(self, node: int, start: int, end: int, left: int, right: int) -> int:
        if right < start or end < left:
            return 0
        if left <= start and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_sum = self.query_range(node*2, start, mid, left, right)
        right_sum = self.query_range(node*2 + 1, mid + 1, end, left, right)
        return left_sum + right_sum

    def update(self, node: int, start: int, end: int, index: int, diff: int) -> None:
        if index < start or end < index:
            return

        self.tree[node] = self.tree[node] + diff
        
        if start != end:
            mid = (start + end) // 2
            self.update(node*2, start, mid, index, diff)
            self.update(node*2 + 1, mid + 1, end, index, diff)
    


import sys


"""
TODO:
- �ϴ� SegmentTree���� �����ϱ�
- main �����ϱ�
"""


class Pair(tuple[int, int]):
    """
    ��Ʈ: 2243, 3653���� int�� ���� ���׸�Ʈ Ʈ���� ������ٸ� ���⼭�� Pair�� ���� ���׸�Ʈ Ʈ���� ���� �� ��������...?
    """
    def __new__(cls, a: int, b: int) -> 'Pair':
        return super().__new__(cls, (a, b))

    @staticmethod
    def default() -> 'Pair':
        """
        �⺻��
        �̰� �� �ʿ��ұ�...?
        """
        return Pair(0, 0)

    @staticmethod
    def f_conv(w: int) -> 'Pair':
        """
        ���� ������ ���� �����Ǵ� Pair ������ ��ȯ�ϴ� ����
        �̰� �� �ʿ��ұ�...?
        """
        return Pair(w, 0)

    @staticmethod
    def f_merge(a: Pair, b: Pair) -> 'Pair':
        """
        �� Pair�� �ϳ��� Pair�� ��ġ�� ����
        �̰� �� �ʿ��ұ�...?
        """
        return Pair(*sorted([*a, *b], reverse=True)[:2])

    def sum(self) -> int:
        return self[0] + self[1]


def main() -> None:
    # �����ϼ���!
    data_list = sys.stdin.read().splitlines()

    n = int(data_list[0])
    array_list = list(map(int, data_list[1].split()))
    m = int(data_list[2])
    query = data_list[3:]

    # ó���� �ʱ�ȭ���� ������ ������ ���ϰ��� �����ؼ�
    # f_conv�� ���� �� ���¸� ������ְ� ���ϰ��� �־���.
    tree = SegmentTree(n)
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

if __name__ == "__main__":
    main()