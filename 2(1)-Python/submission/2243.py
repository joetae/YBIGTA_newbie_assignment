from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable


"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")
U = TypeVar("U")


class SegmentTree(Generic[T, U]):
    # 구현하세요!
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
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    # 구현하세요!
    data_list = sys.stdin.read().splitlines()

    task_num = int(data_list[0])
    task_list = data_list[1:]

    upper_bound = 1000000
    tree: SegmentTree = SegmentTree(upper_bound)
    
    ans_save = []

    for task in task_list:
        split_task = list(map(int, task.split()))
        if split_task[0] == 1:
            rank = split_task[1]
            flavor_num = tree.query(1, 1, upper_bound, rank)
            ans_save.append(flavor_num)
            tree.update(1, 1, upper_bound, flavor_num, -1)
        elif split_task[0] == 2:
            flavor_num, how_many = split_task[1], split_task[2]
            tree.update(1, 1, upper_bound, flavor_num, how_many)

    for ans in ans_save:
        print(ans)


if __name__ == "__main__":
    main()